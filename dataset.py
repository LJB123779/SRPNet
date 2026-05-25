"""
数据处理模块 (Dataset Module)

伪装物体分割 (Camouflaged Object Segmentation) 数据集
支持: CAMO, NC4K 等伪装物体分割数据集

数据格式:
- Images: 输入图像目录 (支持 jpg, png, bmp, tif)
- GT (Ground Truth): 分割掩码目录 (单通道二值图像)

修复说明 (2024):
- 修复了 Mask 维度处理的潜在歧义 (ToTensorV2 + 手动 unsqueeze)
- 统一了 CopyPaste 内部使用 0-255 范围, 输出使用 0-1 范围
- 增加了对 .tif/.tiff 格式的支持
- 增强了 Copy-Paste 的空候选集检查
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CopyPasteAugmentation:
    """
    CopyPaste 数据增强
    
    从其他图像中复制伪装物体并粘贴到当前图像，增强模型对伪装的识别能力。
    这是对付伪装检测的核心增强策略。
    
    注意: 本类内部统一使用 0-255 范围的mask
    """
    
    def __init__(
        self,
        paste_prob: float = 0.5,
        max_paste_objects: int = 3,
        min_scale: float = 0.5,   # 提高最小比例，避免过小
        max_scale: float = 1.2,
        blend_alpha_range: Tuple[float, float] = (0.8, 1.0)  # 提高透明度下限
    ):
        """
        Args:
            paste_prob: 执行粘贴的概率
            max_paste_objects: 最大粘贴物体数量
            min_scale: 最小缩放比例
            max_scale: 最大缩放比例
            blend_alpha_range: 混合透明度范围
        """
        self.paste_prob = paste_prob
        self.max_paste_objects = max_paste_objects
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.blend_alpha_range = blend_alpha_range
        
        # 存储候选粘贴对象
        self.paste_candidates: List[Tuple[np.ndarray, np.ndarray]] = []
        
    def add_candidate(self, image: np.ndarray, mask: np.ndarray):
        """添加候选粘贴对象"""
        # 提取前景区域
        if mask.sum() < 100:  # 太小的区域忽略
            return
            
        # 获取边界框
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 裁剪前景
        fg_image = image[y_min:y_max+1, x_min:x_max+1].copy()
        fg_mask = mask[y_min:y_max+1, x_min:x_max+1].copy()
        
        if fg_image.shape[0] > 16 and fg_image.shape[1] > 16:
            self.paste_candidates.append((fg_image, fg_mask))
            
        # 限制候选数量
        if len(self.paste_candidates) > 500:
            self.paste_candidates = self.paste_candidates[-300:]
            
    def _apply_geometric_transform(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用几何变换"""
        # 随机缩放
        scale = random.uniform(self.min_scale, self.max_scale)
        new_h = int(image.shape[0] * scale)
        new_w = int(image.shape[1] * scale)
        
        if new_h < 16 or new_w < 16:
            return image, mask
            
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 随机水平翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            
        # 随机旋转
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            center = (new_w // 2, new_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpAffine(mask, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT)
            
        return image, mask
        
    def _blend_paste(
        self,
        bg_image: np.ndarray,
        bg_mask: np.ndarray,
        fg_image: np.ndarray,
        fg_mask: np.ndarray,
        x: int,
        y: int,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将前景混合粘贴到背景"""
        h, w = fg_image.shape[:2]
        bg_h, bg_w = bg_image.shape[:2]
        
        # 边界检查
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
        
        fg_x1, fg_y1 = x1 - x, y1 - y
        fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)
        
        if x2 <= x1 or y2 <= y1:
            return bg_image, bg_mask
            
        # 获取前景和背景区域
        fg_region = fg_image[fg_y1:fg_y2, fg_x1:fg_x2]
        fg_mask_region = fg_mask[fg_y1:fg_y2, fg_x1:fg_x2]
        
        if fg_region.size == 0:
            return bg_image, bg_mask
            
        # 创建混合掩码
        blend_mask = (fg_mask_region > 0).astype(np.float32)
        
        # 边缘羽化
        kernel_size = max(3, int(min(fg_region.shape[:2]) * 0.1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        blend_mask = cv2.GaussianBlur(blend_mask, (kernel_size, kernel_size), 0)
        
        blend_mask = blend_mask * alpha
        blend_mask = blend_mask[:, :, np.newaxis]
        
        # 混合
        bg_region = bg_image[y1:y2, x1:x2].astype(np.float32)
        fg_region = fg_region.astype(np.float32)
        
        blended = bg_region * (1 - blend_mask) + fg_region * blend_mask
        bg_image[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        # 更新掩码
        bg_mask[y1:y2, x1:x2] = np.maximum(
            bg_mask[y1:y2, x1:x2],
            fg_mask_region
        )
        
        return bg_image, bg_mask
        
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行CopyPaste增强
        
        Args:
            image: 背景图像 (H, W, 3)
            mask: 背景掩码 (H, W)
            
        Returns:
            增强后的图像和掩码
        """
        if random.random() > self.paste_prob or len(self.paste_candidates) == 0:
            return image, mask
            
        image = image.copy()
        mask = mask.copy()
        
        # 随机选择粘贴数量
        num_paste = random.randint(1, min(self.max_paste_objects, len(self.paste_candidates)))
        
        for _ in range(num_paste):
            # 随机选择候选对象
            fg_image, fg_mask = random.choice(self.paste_candidates)
            fg_image = fg_image.copy()
            fg_mask = fg_mask.copy()
            
            # 几何变换
            fg_image, fg_mask = self._apply_geometric_transform(fg_image, fg_mask)
            
            # 随机位置
            max_x = image.shape[1] - fg_image.shape[1]
            max_y = image.shape[0] - fg_image.shape[0]
            
            if max_x <= 0 or max_y <= 0:
                continue
                
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # 随机透明度
            alpha = random.uniform(*self.blend_alpha_range)
            
            # 粘贴
            image, mask = self._blend_paste(image, mask, fg_image, fg_mask, x, y, alpha)
            
        return image, mask


class MaskAwareCrop:
    """
    目标感知裁剪 (Mask-Aware Crop)
    
    针对 COD 任务的数据增强策略：
    - 50% 概率使用 object-centric crop（基于 mask bbox 裁剪，保证目标在视野内）
    - 50% 概率使用普通 random crop（保持场景多样性）
    
    这能显著提升模型对小目标和极度伪装样本的检测能力。
    """
    
    def __init__(
        self,
        image_size: int = 512,
        object_centric_prob: float = 0.5,
        context_ratio: float = 0.3,
        min_object_ratio: float = 0.1,
        random_scale_range: Tuple[float, float] = (0.5, 1.0)
    ):
        """
        Args:
            image_size: 输出图像尺寸
            object_centric_prob: 使用 object-centric crop 的概率
            context_ratio: bbox 扩展比例（添加上下文）
            min_object_ratio: 目标最小占比（低于此值强制使用 object-centric）
            random_scale_range: 随机裁剪的缩放范围
        """
        self.image_size = image_size
        self.object_centric_prob = object_centric_prob
        self.context_ratio = context_ratio
        self.min_object_ratio = min_object_ratio
        self.random_scale_range = random_scale_range
        
    def _get_mask_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        获取 mask 的边界框
        
        Returns:
            (x_min, y_min, x_max, y_max) 或 None（如果 mask 为空）
        """
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return (x_min, y_min, x_max, y_max)
    
    def _expand_bbox(
        self, 
        bbox: Tuple[int, int, int, int], 
        img_shape: Tuple[int, int],
        context_ratio: float
    ) -> Tuple[int, int, int, int]:
        """
        扩展边界框以包含上下文
        
        Args:
            bbox: (x_min, y_min, x_max, y_max)
            img_shape: (H, W)
            context_ratio: 扩展比例
            
        Returns:
            扩展后的 bbox
        """
        x_min, y_min, x_max, y_max = bbox
        h, w = img_shape
        
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        # 随机扩展（添加一些随机性）
        expand_w = int(bbox_w * context_ratio * random.uniform(0.5, 1.5))
        expand_h = int(bbox_h * context_ratio * random.uniform(0.5, 1.5))
        
        # 扩展边界框
        x_min = max(0, x_min - expand_w)
        y_min = max(0, y_min - expand_h)
        x_max = min(w, x_max + expand_w)
        y_max = min(h, y_max + expand_h)
        
        return (x_min, y_min, x_max, y_max)
    
    def _random_crop(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """普通随机裁剪"""
        h, w = image.shape[:2]
        
        # 随机缩放比例
        scale = random.uniform(*self.random_scale_range)
        crop_size = int(min(h, w) * scale)
        crop_size = max(crop_size, 64)  # 最小裁剪尺寸
        
        # 随机裁剪位置
        if h > crop_size:
            y = random.randint(0, h - crop_size)
        else:
            y = 0
            crop_size = min(crop_size, h)
            
        if w > crop_size:
            x = random.randint(0, w - crop_size)
        else:
            x = 0
            crop_size = min(crop_size, w)
        
        # 裁剪
        image_crop = image[y:y+crop_size, x:x+crop_size]
        mask_crop = mask[y:y+crop_size, x:x+crop_size]
        
        return image_crop, mask_crop
    
    def _object_centric_crop(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """目标中心裁剪"""
        h, w = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        
        # 扩展 bbox 以包含上下文
        bbox_expanded = self._expand_bbox(bbox, (h, w), self.context_ratio)
        ex_min, ey_min, ex_max, ey_max = bbox_expanded
        
        # 计算裁剪区域（确保是正方形或接近正方形）
        crop_w = ex_max - ex_min
        crop_h = ey_max - ey_min
        crop_size = max(crop_w, crop_h)
        
        # 添加随机偏移（在扩展区域内随机移动）
        max_offset_x = max(0, crop_size - crop_w) // 2
        max_offset_y = max(0, crop_size - crop_h) // 2
        
        offset_x = random.randint(-max_offset_x, max_offset_x) if max_offset_x > 0 else 0
        offset_y = random.randint(-max_offset_y, max_offset_y) if max_offset_y > 0 else 0
        
        # 计算最终裁剪区域
        center_x = (ex_min + ex_max) // 2 + offset_x
        center_y = (ey_min + ey_max) // 2 + offset_y
        
        half_size = crop_size // 2
        crop_x1 = max(0, center_x - half_size)
        crop_y1 = max(0, center_y - half_size)
        crop_x2 = min(w, crop_x1 + crop_size)
        crop_y2 = min(h, crop_y1 + crop_size)
        
        # 调整以确保裁剪区域有效
        if crop_x2 - crop_x1 < crop_size:
            crop_x1 = max(0, crop_x2 - crop_size)
        if crop_y2 - crop_y1 < crop_size:
            crop_y1 = max(0, crop_y2 - crop_size)
        
        # 裁剪
        image_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        
        return image_crop, mask_crop
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用 mask-aware crop
        
        Args:
            image: RGB 图像 (H, W, 3)
            mask: 二值掩码 (H, W)，值域 0-255 或 0-1
            
        Returns:
            裁剪后的 (image, mask)，已 resize 到 image_size
        """
        h, w = image.shape[:2]
        
        # 获取 mask bbox
        bbox = self._get_mask_bbox(mask)
        
        # 计算目标占比
        if bbox is not None:
            object_pixels = (mask > 0).sum()
            total_pixels = h * w
            object_ratio = object_pixels / total_pixels
        else:
            object_ratio = 0
        
        # 决定使用哪种裁剪方式
        use_object_centric = False
        
        if bbox is not None:
            # 如果目标太小，强制使用 object-centric crop
            if object_ratio < self.min_object_ratio:
                use_object_centric = True
            # 否则按概率选择
            elif random.random() < self.object_centric_prob:
                use_object_centric = True
        
        # 执行裁剪
        if use_object_centric and bbox is not None:
            image_crop, mask_crop = self._object_centric_crop(image, mask, bbox)
        else:
            image_crop, mask_crop = self._random_crop(image, mask)
        
        # Resize 到目标尺寸
        image_resized = cv2.resize(
            image_crop, 
            (self.image_size, self.image_size), 
            interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            mask_crop, 
            (self.image_size, self.image_size), 
            interpolation=cv2.INTER_NEAREST
        )
        
        return image_resized, mask_resized


class CODTransforms:
    """伪装物体检测专用变换"""
    
    @staticmethod
    def get_train_transforms(image_size: int = 512, use_mask_aware_crop: bool = False) -> A.Compose:
        """
        训练时的数据增强
        
        Args:
            image_size: 输出图像尺寸
            use_mask_aware_crop: 是否使用 MaskAwareCrop（如果是，则不包含裁剪变换）
        """
        transforms_list = []
        
        # 如果不使用 MaskAwareCrop，则使用 Resize
        # 注意: RandomResizedCrop 在 albumentations 1.4.x + OpenCV 4.9 下有 bug
        if not use_mask_aware_crop:
            transforms_list.append(
                A.Resize(image_size, image_size)
            )
        
        # 几何变换
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15),  # 稍微减小，因为 MaskAwareCrop 已经做了缩放
                rotate=(-30, 30),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            # 颜色变换 - 对伪装检测很重要
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.5),
            
            # 模糊和噪声
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1), p=1.0),  # 新版参数
                A.ISONoise(p=1.0),
            ], p=0.2),
            
            # 对比度和亮度 - 增强伪装场景的鲁棒性
            A.OneOf([
                A.CLAHE(clip_limit=4.0, p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.4),
            
            # 天气和光照模拟
            A.OneOf([
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(1, 3),  # 新版参数
                    shadow_dimension=5,
                    p=1.0
                ),
                A.RandomFog(fog_coef_range=(0.1, 0.3), p=1.0),  # 新版参数
            ], p=0.2),
            
            # 归一化
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms_list)
        
    @staticmethod
    def get_val_transforms(image_size: int = 512) -> A.Compose:
        """验证/测试时的变换"""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
    @staticmethod
    def get_frequency_augments() -> A.Compose:
        """频域增强相关的变换"""
        return A.Compose([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.5, alpha=(0.2, 0.5), p=0.3),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2),
        ])


class CAMODataset(Dataset):
    """
    CAMO 数据集
    
    伪装物体检测训练数据集，包含自然伪装和人工伪装物体
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 512,
        transforms: Optional[A.Compose] = None,
        copy_paste: Optional[CopyPasteAugmentation] = None,
        mask_aware_crop: Optional[MaskAwareCrop] = None,
        use_mask_aware_crop: bool = True,
        return_path: bool = False,
        verbose: bool = True,
        test_split: str = "test1"  # 🔥 新增: 选择 test1 或 test2 作为测试集
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: 数据集划分 ("train", "val", "test")
            image_size: 图像尺寸
            transforms: 数据增强
            copy_paste: CopyPaste增强实例
            mask_aware_crop: MaskAwareCrop增强实例（训练时使用）
            use_mask_aware_crop: 是否启用 MaskAwareCrop（默认True）
            return_path: 是否返回文件路径
            verbose: 是否打印信息
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.return_path = return_path
        self.verbose = verbose
        self.test_split = test_split  # 🔥 test1 或 test2
        
        # MaskAwareCrop (仅训练时使用)
        if split == "train" and use_mask_aware_crop:
            if mask_aware_crop is not None:
                self.mask_aware_crop = mask_aware_crop
            else:
                # 使用默认配置
                self.mask_aware_crop = MaskAwareCrop(
                    image_size=image_size,
                    object_centric_prob=0.5,
                    context_ratio=0.3,
                    min_object_ratio=0.1
                )
            if verbose:
                print(f"✅ MaskAwareCrop 已启用 (object_centric_prob={self.mask_aware_crop.object_centric_prob})")
        else:
            self.mask_aware_crop = None
        
        # 设置变换
        if transforms is None:
            if split == "train":
                # 如果使用 MaskAwareCrop，transforms 不需要包含裁剪
                self.transforms = CODTransforms.get_train_transforms(
                    image_size, 
                    use_mask_aware_crop=(self.mask_aware_crop is not None)
                )
            else:
                self.transforms = CODTransforms.get_val_transforms(image_size)
        else:
            self.transforms = transforms
            
        self.copy_paste = copy_paste
        
        # 加载数据列表
        self.samples = self._load_samples()
        
        # 初始化CopyPaste候选
        if copy_paste is not None and split == "train":
            self._init_copy_paste_candidates()
            
    def _load_samples(self) -> List[Dict[str, str]]:
        """加载样本列表"""
        samples = []
        
        # 支持的数据集结构 (优先级从高到低):
        # 
        # 🔥 结构0 (COD10K+CAMO): images/train + images/test1 + images/test2
        #   root_dir/
        #   ├── images/train/  images/test1/  images/test2/
        #   └── masks/train/   masks/test1/   masks/test2/
        #
        # 结构1 (推荐): Images分train/test, GT不分
        #   root_dir/
        #   ├── Images/train/  Images/test/
        #   └── GT/  (直接放所有掩码)
        
        image_dir = None
        mask_dir = None
        
        # 🔥 优先检测 COD10K+CAMO 结构
        # 结构A: Train/Image + Train/GT, Test1/Image + Test1/GT
        if self.split == "train":
            for train_name in ["Train", "train"]:
                candidate = self.root_dir / train_name / "Image"
                if candidate.exists():
                    image_dir = candidate
                    mask_dir = self.root_dir / train_name / "GT"
                    break
                # 也尝试 images/masks 子目录
                candidate = self.root_dir / train_name / "images"
                if candidate.exists():
                    image_dir = candidate
                    mask_dir = self.root_dir / train_name / "masks"
                    break
        elif self.split == "test":
            # 根据 test_split 参数选择 Test1 或 Test2
            for test_name in [self.test_split, self.test_split.capitalize(), self.test_split.upper()]:
                candidate = self.root_dir / test_name / "Image"
                if candidate.exists():
                    image_dir = candidate
                    mask_dir = self.root_dir / test_name / "GT"
                    if self.verbose:
                        print(f"  🔥 使用测试集: {test_name}")
                    break
                # 也尝试 images/masks 子目录
                candidate = self.root_dir / test_name / "images"
                if candidate.exists():
                    image_dir = candidate
                    mask_dir = self.root_dir / test_name / "masks"
                    if self.verbose:
                        print(f"  🔥 使用测试集: {test_name}")
                    break
        
        # 结构B (旧): images/train, images/test1, images/test2
        if image_dir is None or not image_dir.exists():
            if self.split == "train":
                candidate = self.root_dir / "images" / "train"
                if candidate.exists():
                    image_dir = candidate
                    mask_dir = self.root_dir / "masks" / "train"
            elif self.split == "test":
                candidate = self.root_dir / "images" / self.test_split
                if candidate.exists():
                    image_dir = candidate
                    mask_dir = self.root_dir / "masks" / self.test_split
        
        # 尝试多种目录结构
        # 支持 train/test 和 Train/Test 两种大小写
        split_variants = [self.split, self.split.capitalize()]  # ['train', 'Train'] 或 ['test', 'Test']
        
        # 结构1: Images/Train + GT (GT不分子目录) - 用户常用格式
        if image_dir is None or not image_dir.exists():
            for split_name in split_variants:
                candidate = self.root_dir / "Images" / split_name
                if candidate.exists():
                    image_dir = candidate
                    mask_dir = self.root_dir / "GT"
                    break
        
        # 结构2: Images/Train + GT/Train (都分子目录)
        if image_dir is None:
            for split_name in split_variants:
                candidate = self.root_dir / "Images" / split_name
                if candidate.exists():
                    image_dir = candidate
                    mask_dir = self.root_dir / "GT" / split_name
                    break
        
        if image_dir is None or not image_dir.exists():
            # 结构3: train/images + train/masks
            image_dir = self.root_dir / self.split / "images"
            mask_dir = self.root_dir / self.split / "masks"
            
        if image_dir is None or not image_dir.exists():
            # 结构4: images + masks (无子目录)
            image_dir = self.root_dir / "images"
            mask_dir = self.root_dir / "masks"
            
        if image_dir is None or not image_dir.exists():
            # 结构5: images + GT (无子目录)
            image_dir = self.root_dir / "images"
            mask_dir = self.root_dir / "GT"
            
        if image_dir is None or not image_dir.exists():
            # 结构6: Images + GT (无子目录)
            image_dir = self.root_dir / "Images"
            mask_dir = self.root_dir / "GT"
            
        # 支持的图像格式 (包括tif)
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # 打印实际路径便于调试
        if self.verbose:
            print(f"  [{self.split}] 图像目录: {image_dir} (存在: {image_dir.exists()})")
            print(f"  [{self.split}] 掩码目录: {mask_dir} (存在: {mask_dir.exists()})")
        
        if image_dir.exists():
            for img_path in sorted(image_dir.glob("*")):
                if img_path.suffix.lower() in valid_exts:
                    # 查找对应掩码 (尝试多种格式)
                    mask_path = None
                    for ext in ['.png', '.jpg', '.bmp', '.tif']:
                        p = mask_dir / f"{img_path.stem}{ext}"
                        if p.exists():
                            mask_path = p
                            break
                    # 尝试同名文件
                    if mask_path is None:
                        p = mask_dir / img_path.name
                        if p.exists():
                            mask_path = p
                        
                    if mask_path is not None:
                        samples.append({
                            'image': str(img_path),
                            'mask': str(mask_path)
                        })
                        
        return samples
        
    def _init_copy_paste_candidates(self, max_samples: int = 200):
        """初始化CopyPaste候选对象 (使用0-255范围的mask)"""
        if self.verbose:
            print("正在初始化 Copy-Paste 候选对象...")
        
        if len(self.samples) == 0:
            if self.verbose:
                print("警告: 数据集为空，跳过 Copy-Paste 初始化")
            return
            
        indices = random.sample(
            range(len(self.samples)), 
            min(max_samples, len(self.samples))
        )
        
        for idx in indices:
            sample = self.samples[idx]
            image = cv2.imread(sample['image'])
            mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
            
            if image is not None and mask is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 二值化到 0-255 范围 (CopyPaste内部统一使用此范围)
                mask = (mask > 127).astype(np.uint8) * 255
                self.copy_paste.add_candidate(image, mask)
        
        if self.verbose:
            print(f"✅ 初始化了 {len(self.copy_paste.paste_candidates)} 个 Copy-Paste 候选对象")
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 读取图像和掩码
        image = cv2.imread(sample['image'])
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码: {sample['mask']}")
        
        # 二值化掩码到 0-255 范围 (CopyPaste使用此范围)
        mask = (mask > 127).astype(np.uint8) * 255
        
        # CopyPaste增强 (使用 0-255 范围的mask)
        if self.copy_paste is not None and self.split == "train":
            if len(self.copy_paste.paste_candidates) > 0:
                image, mask = self.copy_paste(image, mask)
        
        # MaskAwareCrop (使用 0-255 范围的mask，输出已resize到image_size)
        if self.mask_aware_crop is not None:
            image, mask = self.mask_aware_crop(image, mask)
        
        # 归一化掩码到 0-1 供 albumentations 和后续训练使用
        mask = mask.astype(np.float32) / 255.0
        
        # 确保 image/mask 是 C-contiguous numpy array (albumentations 1.4+ 严格检查)
        # 使用 np.array(..., copy=True) 强制创建独立副本，避免内存布局问题
        image = np.array(image, dtype=np.uint8, copy=True)
        mask = np.array(mask, dtype=np.float32, copy=True)
        
        # 确保 C-contiguous
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        if not mask.flags['C_CONTIGUOUS']:
            mask = np.ascontiguousarray(mask)
            
        # 应用变换 (mask作为albumentations的mask参数)
        transformed = self.transforms(image=image, mask=mask)
        image = transformed['image']  # (3, H, W) Tensor via ToTensorV2
        mask = transformed['mask']    # (H, W) Tensor or ndarray
        
        # 确保掩码是 Float Tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.float()
        
        # 二值化 (transforms中的resize可能导致插值产生中间值)
        mask = (mask > 0.5).float()
        
        # 增加 Channel 维度: (H, W) -> (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        result = {
            'image': image,
            'mask': mask  # (1, H, W)
        }
        
        if self.return_path:
            result['image_path'] = sample['image']
            result['mask_path'] = sample['mask']
            
        return result


class NC4KDataset(Dataset):
    """
    NC4K 测试数据集
    
    大规模伪装物体检测测试集，包含4121张图像
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 512,
        transforms: Optional[A.Compose] = None,
        return_path: bool = True
    ):
        """
        Args:
            root_dir: 数据集根目录
            image_size: 图像尺寸
            transforms: 数据变换
            return_path: 是否返回文件路径（用于保存预测结果）
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.return_path = return_path
        
        if transforms is None:
            self.transforms = CODTransforms.get_val_transforms(image_size)
        else:
            self.transforms = transforms
            
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict[str, str]]:
        """加载样本列表"""
        samples = []
        
        # NC4K数据集结构
        image_dir = self.root_dir / "images"
        mask_dir = self.root_dir / "masks"
        
        if not image_dir.exists():
            image_dir = self.root_dir / "Images"
            mask_dir = self.root_dir / "GT"
            
        # 支持的图像格式 (包括tif)
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        if image_dir.exists():
            for img_path in sorted(image_dir.glob("*")):
                if img_path.suffix.lower() in valid_exts:
                    # 查找对应掩码
                    mask_path = None
                    for ext in ['.png', '.jpg', '.bmp', '.tif']:
                        p = mask_dir / f"{img_path.stem}{ext}"
                        if p.exists():
                            mask_path = p
                            break
                        
                    samples.append({
                        'image': str(img_path),
                        'mask': str(mask_path) if mask_path else None,
                        'name': img_path.stem
                    })
                    
        return samples
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 读取图像
        image = cv2.imread(sample['image'])
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # 应用变换
        transformed = self.transforms(image=image)
        image = transformed['image']
        
        result = {
            'image': image,
            'original_size': torch.tensor(original_size),
            'name': sample['name']
        }
        
        # 如果有掩码（用于评估）
        if sample['mask'] is not None and os.path.exists(sample['mask']):
            mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # 归一化到 0-1
                mask = (mask > 127).astype(np.float32)
                
                # 使用 A.Resize 调整掩码尺寸
                mask_resize = A.Resize(self.image_size, self.image_size)
                mask_transformed = mask_resize(image=np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8), mask=mask)
                mask = torch.from_numpy(mask_transformed['mask']).float()
                
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                result['mask'] = mask
            
        if self.return_path:
            result['image_path'] = sample['image']
            
        return result


class ExemplarBank:
    """
    示例库
    
    存储和管理伪装物体的视觉示例，用于few-shot推理
    """
    
    def __init__(
        self,
        exemplar_dir: str,
        num_exemplars: int = 5,
        image_size: int = 256
    ):
        """
        Args:
            exemplar_dir: 示例图像目录
            num_exemplars: 每次采样的示例数量
            image_size: 示例图像尺寸
        """
        self.exemplar_dir = Path(exemplar_dir)
        self.num_exemplars = num_exemplars
        self.image_size = image_size
        
        self.transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        self.exemplars = self._load_exemplars()
        
    def _load_exemplars(self) -> List[Dict[str, np.ndarray]]:
        """加载所有示例"""
        exemplars = []
        
        image_dir = self.exemplar_dir / "images"
        mask_dir = self.exemplar_dir / "masks"
        
        if not image_dir.exists():
            image_dir = self.exemplar_dir
            mask_dir = self.exemplar_dir
            
        for img_path in sorted(image_dir.glob("*")):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                mask_path = mask_dir / f"{img_path.stem}.png"
                
                image = cv2.imread(str(img_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    mask = None
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        mask = (mask > 127).astype(np.float32)
                        
                    exemplars.append({
                        'image': image,
                        'mask': mask,
                        'path': str(img_path)
                    })
                    
        return exemplars
        
    def sample(self, batch_size: int = 1) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        采样示例
        
        Args:
            batch_size: 批次大小
            
        Returns:
            exemplars: (B, N, 3, H, W)
            masks: (B, N, H, W) 或 None
        """
        if len(self.exemplars) == 0:
            return None, None
            
        all_images = []
        all_masks = []
        
        for _ in range(batch_size):
            indices = random.sample(
                range(len(self.exemplars)),
                min(self.num_exemplars, len(self.exemplars))
            )
            
            images = []
            masks = []
            
            for idx in indices:
                exemplar = self.exemplars[idx]
                transformed = self.transforms(
                    image=exemplar['image'],
                    mask=exemplar['mask'] if exemplar['mask'] is not None 
                         else np.zeros(exemplar['image'].shape[:2], dtype=np.float32)
                )
                images.append(transformed['image'])  # ToTensorV2 已转为 Tensor
                masks.append(transformed['mask'].float())  # 直接 float 即可
                
            all_images.append(torch.stack(images))
            all_masks.append(torch.stack(masks))
            
        return torch.stack(all_images), torch.stack(all_masks)


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 8,
    image_size: int = 512,
    num_workers: int = 4,
    use_copy_paste: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器 (分割任务 - 已弃用)
    
    Args:
        train_dir: 训练数据目录 (CAMO)
        test_dir: 测试数据目录 (NC4K)
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: 数据加载线程数
        use_copy_paste: 是否使用CopyPaste增强
        
    Returns:
        train_loader, test_loader
    """
    # CopyPaste增强
    copy_paste = CopyPasteAugmentation() if use_copy_paste else None
    
    # 训练数据集
    train_dataset = CAMODataset(
        root_dir=train_dir,
        split="train",
        image_size=image_size,
        copy_paste=copy_paste
    )
    
    # 测试数据集
    test_dataset = NC4KDataset(
        root_dir=test_dir,
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 测试时单张处理以保持原始尺寸
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


