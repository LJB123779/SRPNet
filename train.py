"""
V5 训练脚本 - 消融实验基准

设计理念：
    E0 Baseline: 纯 V0 单阶段模型，作为所有消融实验的起点。
    模型: SAM3 Encoder (冻结) → 1x1 Proj → CoarseDecoder (H/32) → Upsample
    损失: BCE + Dice (最基础组合)
    
    
    修改 V2Config 中的开关即可切换实验配置。
"""

import os
import sys
import re
import random
import warnings
from pathlib import Path
from typing import Dict, Optional

# 抑制警告
warnings.filterwarnings('ignore', message='Grad strides do not match')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加父目录到路径以导入共享模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_sam3_cod_model_v2, SAM3CamouflageDetectorV2
from loss import build_segmentation_criterion_v2
from dataset import CAMODataset
# CODEvaluator 延迟导入，避免 py-sod-metrics 未装时训练脚本无法启动


def get_simple_train_transforms(image_size: int = 512) -> A.Compose:
    """
    V0 基础训练增强: Resize + 4种几何变换 + 标准化
    
    设计理念: 最简增强作为 baseline，方便后续对比增强策略的效果
    注意: RandomResizedCrop 在 albumentations 1.4.x + OpenCV 4.9 下有 bug，改用 Resize
    """
    return A.Compose([
        # 直接 Resize 到目标尺寸 (避免 RandomResizedCrop 的兼容性问题)
        A.Resize(image_size, image_size),
        # 4种几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=0,  # cv2.BORDER_CONSTANT
            p=0.5
        ),
        # 标准化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


# =============================================================================
# Baseline 配置
# =============================================================================

class V2Config:
    """V5 训练配置 - 消融实验 (通过开关控制实验组)"""
    
    # -------------------------------------------------------------------------
    # 数据路径
    # -------------------------------------------------------------------------
    DATA_ROOT: str = "COD10K+CAMO"
    SAVE_DIR: str = "checkpoints_v5"
    TEST_SPLIT: str = "test2"
    
    # -------------------------------------------------------------------------
    # 模型配置
    # -------------------------------------------------------------------------
    SAM3_CHECKPOINT: str = "weights/sam3.pt"
    BACKBONE_TYPE: str = "vit_b"
    FREEZE_ENCODER: bool = True
    SMART_FINETUNE: bool = True
    
    # -------------------------------------------------------------------------
    # 消融实验开关
    # E0 (Baseline): USE_REFINEMENT=False
    # E1 (Refine):   USE_REFINEMENT=True
    # E3 (Boundary): USE_REFINEMENT=True + USE_BOUNDARY_HEAD=True
    # -------------------------------------------------------------------------
    #模块一：两阶段精修
    USE_REFINEMENT: bool = True    # True → 启用两阶段精修 (FeatureUpsampler+PromptProj+RefineDecoder)
    DETACH_PROMPT: bool = True     # True → detach prompt 梯度，避免梯度耦合
    USE_RESIDUAL: bool = True      # True → RefineDecoder 以残差修补模式运行

    #模块二：边界分支
    USE_BOUNDARY_HEAD: bool = False # True → 启用边界分支 + 残差注入 (需配合 LOSS_EDGE>0)

    # -------------------------------------------------------------------------
    # 训练超参数
    # -------------------------------------------------------------------------
    EPOCHS: int = 100
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 2e-4
    BACKBONE_LR_RATIO: float = 0.1
    WEIGHT_DECAY: float = 1e-4
    ENCODER_INPUT_SIZE: int = 640
    ENCODER_USE_AMP: bool = True
    ENCODER_AMP_DTYPE: str = "auto"
    ENCODER_TRAINABLE_USE_AMP: bool = False
    IMAGE_SIZE: int = 640
    
    # -------------------------------------------------------------------------
    # Loss 配置
    # -------------------------------------------------------------------------
    LOSS_TYPE: str = 'structure'  # 主损失类型: 'structure' (加权BCE+加权IoU) / 'bce_dice' (消融对比)
    LOSS_MAIN_WEIGHT: float = 1.0 # 主损失权重 (监督最终上采样输出)
    LOSS_COARSE: float = 0.4       # Coarse 阶段监督权重 (监督 m0, H/S)
    LOSS_NATIVE: float = 0.3       # Native refine 监督权重 (监督 m1 原生分辨率, H/T)
    
    # 配合模块二边界分支使用
    LOSS_EDGE: float = 0.5        # Edge Loss 权重 (先全分辨率提边再 maxpool 下采样)
    
    # -------------------------------------------------------------------------
    # 数据增强
    # -------------------------------------------------------------------------
    USE_SIMPLE_AUGMENT: bool = True  # True: 基础增强(裁剪+几何), False: 完整增强
    USE_MASK_AWARE_CROP: bool = False
    
    # -------------------------------------------------------------------------
    # 断点续训
    # -------------------------------------------------------------------------
    RESUME_CHECKPOINT: str = ""  # 填入 last_model.pth 或 best_model.pth 的完整路径即可续训，留空则从头训练

    # -------------------------------------------------------------------------
    # 其他配置
    # -------------------------------------------------------------------------
    SEED: int = 42
    NUM_WORKERS: int = 6
    GRADIENT_ACCUMULATION: int = 2
    USE_AMP: bool = True
    LOG_INTERVAL: int = 10
    WARMUP_EPOCHS: int = 5
    VIS_EVERY_N_BEST: int = 3  # 每 N 次刷新 best 才做可视化
    
    # -------------------------------------------------------------------------
    # 多卡训练
    # -------------------------------------------------------------------------
    USE_DDP: bool = True


CONFIG = V2Config()


# =============================================================================
# 日志工具: 终端输出同时写入文件
# =============================================================================

class _TeeStream:
    """将写入同时转发到终端和日志文件"""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file
    
    def write(self, message):
        self.terminal.write(message)
        if self.log_file and not self.log_file.closed:
            # 过滤 tqdm 进度条: 含 \r 但不含 \n 的是进度条刷新行，跳过写入日志
            if '\r' in message and '\n' not in message:
                return
            # 清理 ANSI 转义序列 (颜色/光标控制等)
            clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', message)
            if clean.strip():  # 跳过纯空白行
                self.log_file.write(clean)
                self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
    
    # tqdm 等库可能检查 encoding / isatty 等属性
    def __getattr__(self, name):
        return getattr(self.terminal, name)


# =============================================================================
# 辅助函数
# =============================================================================

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 性能优化: 开启 cuDNN benchmark + TF32
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def setup_ddp():
    """初始化 DDP 环境"""
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print("未检测到 DDP 环境变量，使用单卡模式")
        return None, 0, 1
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    return local_rank, rank, world_size


def cleanup_ddp():
    """清理 DDP 环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否是主进程"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path,
                    history=None, scaler=None, best_metric=None, best_count=None):
    """保存检查点 (含 scheduler/scaler/best/rng 状态，支持精确断点续训)"""
    model_to_save = model.module if hasattr(model, 'module') else model

    state = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'history': history or {},
        'best_metric': best_metric,
        'best_count': best_count,
        'scaler_state_dict': (scaler.state_dict() if scaler is not None else None),
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, save_path)
    print(f"检查点已保存: {save_path} (epoch={epoch})")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, scaler=None):
    """
    加载断点，恢复训练状态 (含 scaler/rng/best 精确恢复)

    Returns:
        start_epoch: 下一个要训练的 epoch 序号 (已训练 epoch + 1)
        history:     已有的训练历史字典
        best_metric: 已记录的最佳指标值
        best_count:  已记录的最佳模型刷新次数
    """
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"续训检查点不存在: {checkpoint_path}")

    print(f"🔄 加载续训检查点: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 恢复模型权重
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(ckpt['model_state_dict'])

    # 恢复优化器状态
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # 恢复 scheduler 状态 (旧检查点可能没有此字段，兼容处理)
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    else:
        print("  ⚠️  检查点不含 scheduler 状态，scheduler 将从当前 epoch 重建")

    # 恢复 AMP GradScaler 状态
    if scaler is not None and ckpt.get('scaler_state_dict') is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    elif scaler is not None:
        print("  ⚠️  检查点不含 scaler 状态，scaler 将从默认状态开始")

    # 恢复随机数状态 (精确复现数据增强/shuffle/dropout)
    rng = ckpt.get('rng_state')
    if rng is not None:
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if torch.cuda.is_available() and rng.get('cuda') is not None:
            torch.cuda.set_rng_state_all(rng['cuda'])
    else:
        print("  ⚠️  检查点不含 rng 状态，随机数序列将重新开始")

    start_epoch = ckpt['epoch'] + 1
    history = ckpt.get('history', {})

    # 直接读取保存的 best_metric/best_count，旧检查点则从 history 推断
    best_metric = ckpt.get('best_metric', None)
    if best_metric is None:
        best_metric = 0.0
        for key in ('Sm', 'IoU'):
            if history.get(key):
                best_metric = max(history[key])
                break

    best_count = ckpt.get('best_count', None)
    if best_count is None:
        sm_list = history.get('Sm', [])
        best_count = len(sm_list) if sm_list else len(history.get('IoU', []))

    print(f"   已训练到 epoch {ckpt['epoch']}，从 epoch {start_epoch} 继续")
    print(f"   当前最佳指标: {best_metric:.4f}, best_count: {best_count}")
    return start_epoch, history, best_metric, best_count


def visualize_segmentation(model, dataset, device, save_path, num_images=6, threshold=0.5):
    """可视化分割预测结果"""
    model.eval()
    
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    with torch.no_grad():
        for idx, img_idx in enumerate(indices):
            sample = dataset[img_idx]
            image = sample['image'].unsqueeze(0).to(device)
            gt_mask = sample['mask'].cpu().numpy().squeeze()
            
            outputs = model(image)
            pred_logits = outputs['refined_mask']
            pred_mask = torch.sigmoid(pred_logits).cpu().numpy().squeeze()
            pred_binary = (pred_mask > threshold).astype(np.float32)
            
            img_display = sample['image'].cpu()
            img_display = img_display * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # 统一使用 torch float32 插值后再转 numpy，避免 dtype 混用问题
            target_size = img_display.shape[:2]
            if pred_mask.shape != target_size:
                pred_mask_t = torch.tensor(pred_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                pred_mask_t = F.interpolate(pred_mask_t, size=target_size, mode='bilinear', align_corners=False)
                pred_mask = pred_mask_t.squeeze().cpu().numpy().astype(np.float32)
                pred_binary = (pred_mask > threshold).astype(np.float32)
            
            if gt_mask.shape != target_size:
                gt_mask_t = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                gt_mask_t = F.interpolate(gt_mask_t, size=target_size, mode='nearest')
                gt_mask = gt_mask_t.squeeze().cpu().numpy().astype(np.float32)
            
            overlay = img_display.copy()
            overlay[pred_binary > 0.5] = overlay[pred_binary > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5
            
            axes[idx, 0].imshow(img_display)
            axes[idx, 0].set_title('Input Image', fontsize=10)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            axes[idx, 1].set_title('Prediction', fontsize=10)
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            axes[idx, 2].set_title('Ground Truth', fontsize=10)
            axes[idx, 2].axis('off')
            
            axes[idx, 3].imshow(overlay)
            axes[idx, 3].set_title('Overlay (Red=Pred)', fontsize=10)
            axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 可视化已保存: {save_path}")


# =============================================================================
# V5 训练器
# =============================================================================

class V2Trainer:
    """V5 训练器"""
    
    def __init__(
        self,
        model: SAM3CamouflageDetectorV2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_dataset: CAMODataset,
        config: V2Config,
        device: torch.device,
        local_rank: int = None,
        train_sampler: Optional[DistributedSampler] = None
    ):
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.train_sampler = train_sampler
        
        # 模型
        self.model = model.to(device)
        
        # DDP 多卡训练
        self.use_ddp = False
        if local_rank is not None and dist.is_initialized():
            # 风险组合检测: DETACH_PROMPT=True + USE_RESIDUAL=False 时
            # Stage1 参数无梯度路径，必须开启 find_unused_parameters
            has_unused = (config.USE_REFINEMENT
                          and config.DETACH_PROMPT
                          and not config.USE_RESIDUAL)
            self.model = DDP(
                self.model, device_ids=[local_rank],
                find_unused_parameters=has_unused
            )
            self.use_ddp = True
            if is_main_process():
                if has_unused:
                    print(f"⚠️  DDP 多卡训练: {dist.get_world_size()} GPUs (find_unused_parameters=True, DETACH+无RESIDUAL)")
                else:
                    print(f"✅ 使用 DDP 多卡训练: {dist.get_world_size()} GPUs")
        
        # 获取可训练参数并统计
        base_model = self.model.module if self.use_ddp else self.model
        trainable_params = base_model.get_trainable_params()
        
        if is_main_process():
            total_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
            total_all = sum(p.numel() for p in base_model.parameters())
            print(f"  📊 参数统计: 可训练 {total_trainable/1e6:.2f}M / 总计 {total_all/1e6:.2f}M ({100*total_trainable/total_all:.1f}%)")
        
        # 损失函数: 两阶段监督 (refine + coarse + native) + Edge Loss
        self.criterion = build_segmentation_criterion_v2(
            loss_type=config.LOSS_TYPE,
            main_weight=config.LOSS_MAIN_WEIGHT,
            coarse_weight=config.LOSS_COARSE,
            native_weight=config.LOSS_NATIVE,
            edge_weight=config.LOSS_EDGE if config.USE_BOUNDARY_HEAD else 0.0,
        )
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        
        # 差分学习率: Backbone 小火慢炖，Decoder 全力冲刺
        backbone_params = []
        decoder_params = []
        
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if 'image_encoder' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
        
        param_groups = [
            {
                'params': backbone_params,
                'lr': config.LEARNING_RATE * config.BACKBONE_LR_RATIO,
                'weight_decay': config.WEIGHT_DECAY
            },
            {
                'params': decoder_params,
                'lr': config.LEARNING_RATE,
                'weight_decay': config.WEIGHT_DECAY
            }
        ]
        
        self.optimizer = AdamW(param_groups)
        
        if is_main_process():
            print(f"  🔧 优化策略: 差分学习率")
            print(f"     - Backbone LR: {config.LEARNING_RATE * config.BACKBONE_LR_RATIO:.1e} ({len(backbone_params)} params)")
            print(f"     - Decoder  LR: {config.LEARNING_RATE:.1e} ({len(decoder_params)} params)")
        
        # 学习率调度器
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.WARMUP_EPOCHS
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS - config.WARMUP_EPOCHS,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.WARMUP_EPOCHS]
        )
        
        self.use_amp = config.USE_AMP and self.device.type == "cuda"
        if self.use_amp:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except AttributeError:
                self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # 保存目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(config.SAVE_DIR) / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp

        # 训练状态 (默认从头开始)
        self.start_epoch = 1
        self.best_metric = 0.0
        self.best_count = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'MAE': [],
            'Sm': [],
            'wFm': [],
            'maxFm': [],
            'maxEm': [],
        }

        # 断点续训: 在日志开启之前加载，确保所有进程状态一致
        if config.RESUME_CHECKPOINT:
            self.start_epoch, loaded_history, self.best_metric, self.best_count = load_checkpoint(
                self.model, self.optimizer, self.scheduler,
                config.RESUME_CHECKPOINT, device,
                scaler=self.scaler
            )
            # 合并已有历史 (续训继续追加)
            for key in self.history:
                if key in loaded_history:
                    self.history[key] = loaded_history[key]

        # 终端日志同时写入文件 (仅主进程，续训时追加)
        if is_main_process():
            log_mode = 'a' if config.RESUME_CHECKPOINT else 'w'
            log_path = self.save_dir / 'train.log'
            self._log_file = open(log_path, log_mode, encoding='utf-8')
            self._tee_stdout = _TeeStream(sys.stdout, self._log_file)
            self._tee_stderr = _TeeStream(sys.stderr, self._log_file)
            sys.stdout = self._tee_stdout
            sys.stderr = self._tee_stderr
            print(f"📝 终端日志保存到: {log_path}")
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        # 性能优化: 在 GPU 上累加 loss tensor，避免每 step 调用 .item() 触发 GPU 同步
        total_loss_gpu     = torch.tensor(0.0, device=self.device)
        loss_refine_gpu    = torch.tensor(0.0, device=self.device)
        loss_coarse_gpu    = torch.tensor(0.0, device=self.device)
        loss_native_gpu    = torch.tensor(0.0, device=self.device)
        loss_edge_gpu      = torch.tensor(0.0, device=self.device)
        num_samples = 0
        
        accum = self.config.GRADIENT_ACCUMULATION
        num_batches = len(self.train_loader)
        remainder = num_batches % accum  # 尾巴 batch 数量

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not is_main_process())
        
        for batch_idx, batch in enumerate(pbar):
            # 在 CPU 上做 mask 归一化检查，避免 GPU 同步开销
            masks = batch['mask']
            if masks.max() > 1.0:
                masks = masks / 255.0
            
            images = batch['image'].to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            is_last = (batch_idx == num_batches - 1)
            # 判断是否是梯度累积窗口的最后一步 (需要 optimizer.step)
            is_accum_step = ((batch_idx + 1) % accum == 0) or is_last
            # 尾巴 batch 使用实际累积步数作为除数，避免梯度偏小
            if remainder != 0 and (num_batches - batch_idx) <= remainder:
                div = remainder
            else:
                div = accum

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                
                # 调试信息 (仅首个 batch)
                if batch_idx == 0 and is_main_process():
                    pred = outputs['refined_mask']
                    print(f"\n[DEBUG Epoch {epoch} Batch 0]")
                    print(f"  images : min={images.min():.3f}, max={images.max():.3f}")
                    print(f"  masks  : min={masks.min():.3f}, max={masks.max():.3f}")
                    print(f"  pred   : min={pred.min():.3f}, max={pred.max():.3f}, "
                          f"mean={torch.sigmoid(pred).mean():.4f}, nan={pred.isnan().any()}")
                    if 'edge_logits' in outputs:
                        edge = outputs['edge_logits']
                        edge_prob = torch.sigmoid(edge)
                        print(f"  edge   : min={edge.min():.3f}, max={edge.max():.3f}, "
                              f"mean={edge_prob.mean():.4f}, "
                              f"pos_ratio={( edge_prob > 0.5).float().mean():.4f}  ← 边界分支激活 ✅")
                    else:
                        print(f"  edge   : 未启用 (USE_BOUNDARY_HEAD=False)")

                losses = self.criterion(outputs, masks)
                loss = losses['total']

                if batch_idx == 0 and is_main_process():
                    parts = [f"refine={losses['loss_refine'].item():.4f}"]
                    if losses['loss_coarse'].item() > 0:
                        parts.append(f"coarse={losses['loss_coarse'].item():.4f}")
                    if losses['loss_native'].item() > 0:
                        parts.append(f"native={losses['loss_native'].item():.4f}")
                    if losses['loss_edge'].item() > 0:
                        parts.append(f"edge={losses['loss_edge'].item():.4f}")
                    print(f"[DEBUG] loss: {loss.item():.4f}, {', '.join(parts)}")
                
                loss = loss / div
            
            # 反向传播 (DDP: 非累积步跳过梯度同步以提速)
            if self.use_ddp and not is_accum_step:
                with self.model.no_sync():
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
            else:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # 梯度裁剪 + 梯度更新 (累积步结束时执行)
            if is_accum_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            # GPU 上累加，不调用 .item()，避免同步开销
            batch_size = images.size(0)
            total_loss_gpu     += losses['total'].detach() * batch_size
            loss_refine_gpu    += losses['loss_refine'].detach() * batch_size
            loss_coarse_gpu    += losses['loss_coarse'].detach() * batch_size
            loss_native_gpu    += losses['loss_native'].detach() * batch_size
            loss_edge_gpu      += losses['loss_edge'].detach() * batch_size
            num_samples += batch_size

            # 更新进度条 (只在 LOG_INTERVAL 时才 .item()，减少同步次数)
            if batch_idx % self.config.LOG_INTERVAL == 0:
                postfix = {
                    'loss':   f"{losses['total'].item():.4f}",
                    'ref': f"{losses['loss_refine'].item():.4f}",
                }
                if losses['loss_coarse'].item() > 0:
                    postfix['coa'] = f"{losses['loss_coarse'].item():.4f}"
                if losses['loss_native'].item() > 0:
                    postfix['nat'] = f"{losses['loss_native'].item():.4f}"
                if losses['loss_edge'].item() > 0:
                    postfix['edg'] = f"{losses['loss_edge'].item():.4f}"
                pbar.set_postfix(postfix)
        
        # 更新学习率
        self.scheduler.step()
        
        # epoch 结束时才统一 .item()，只同步一次
        n = num_samples if num_samples > 0 else 1
        return {
            'total':          total_loss_gpu.item() / n,
            'loss_refine':    loss_refine_gpu.item() / n,
            'loss_coarse':    loss_coarse_gpu.item() / n,
            'loss_native':    loss_native_gpu.item() / n,
            'loss_edge':      loss_edge_gpu.item() / n,
        }
    
    @torch.no_grad()
    def validate(self, compute_metrics: bool = True) -> Dict[str, float]:
        """
        验证并计算分割指标
        
        DDP 策略: 只在 rank0 上用全量验证集跑 full metrics
        """
        # DDP 模式下，只在主进程上做全量验证
        if self.use_ddp and not is_main_process():
            dist.barrier()
            return {}
        
        eval_model = self.model.module if self.use_ddp else self.model
        eval_model.eval()
        
        total_loss = 0.0
        loss_refine_sum = 0.0
        num_samples = 0
        
        # 延迟导入 CODEvaluator
        evaluator = None
        if compute_metrics:
            try:
                from evaluate import CODEvaluator
                evaluator = CODEvaluator()
            except ImportError as e:
                print(f"⚠️ 跳过指标计算: {e}")
                compute_metrics = False
        
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for batch_idx, batch in enumerate(pbar):
            masks = batch['mask']
            if masks.max() > 1.0:
                masks = masks / 255.0
            
            images = batch['image'].to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                outputs = eval_model(images)
                losses = self.criterion(outputs, masks)
            
            batch_size = images.size(0)
            total_loss += losses['total'].item() * batch_size
            loss_refine_sum += losses['loss_refine'].item() * batch_size
            num_samples += batch_size
            
            # 计算分割指标
            if compute_metrics and evaluator is not None:
                pred_logits = outputs['refined_mask']
                pred = torch.sigmoid(pred_logits)
                if pred.shape[-2:] != masks.shape[-2:]:
                    pred = F.interpolate(
                        pred, size=masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                evaluator.update(pred, masks)
        
        results = {
            'total': total_loss / num_samples if num_samples > 0 else 0.0,
            'loss_refine': loss_refine_sum / num_samples if num_samples > 0 else 0.0,
        }
        
        if compute_metrics and evaluator is not None:
            metrics = evaluator.compute_metrics()
            results['MAE'] = metrics.get('MAE', 0.0)
            if 'Sm' in metrics:
                results['Sm'] = metrics['Sm']
                results['wFm'] = metrics['wFm']
                results['maxFm'] = metrics['maxFm']
                results['maxEm'] = metrics['maxEm']
            else:
                results['IoU'] = metrics.get('IoU', 0.0)
                results['Dice'] = metrics.get('Dice', 0.0)
        
        # DDP: 验证完成后同步所有进程
        if self.use_ddp:
            dist.barrier()
        
        return results
    
    def train(self):
        """完整训练流程"""
        if is_main_process():
            print(f"\n{'='*60}")
            if self.config.RESUME_CHECKPOINT:
                print(f"🔄 V5 续训 (epoch {self.start_epoch} → {self.config.EPOCHS})")
                print(f"   续训来源: {self.config.RESUME_CHECKPOINT}")
            else:
                print("🚀 V5 训练开始")
            print(f"   时间戳: {self.timestamp}")
            print(f"   保存目录: {self.save_dir}")
            print(f"   设备: {self.device}")
            print(f"   Epochs: {self.start_epoch} → {self.config.EPOCHS}")
            print(f"   Batch Size: {self.config.BATCH_SIZE}")
            print(f"   Learning Rate: {self.config.LEARNING_RATE}")
            loss_type_name = 'Structure' if self.config.LOSS_TYPE == 'structure' else 'BCE+Dice'
            loss_parts = [f"{loss_type_name}({self.config.LOSS_MAIN_WEIGHT})"]
            if self.config.LOSS_COARSE > 0:
                loss_parts.append(f"Coarse({self.config.LOSS_COARSE})")
            if self.config.LOSS_NATIVE > 0:
                loss_parts.append(f"Native({self.config.LOSS_NATIVE})")
            if self.config.USE_BOUNDARY_HEAD and self.config.LOSS_EDGE > 0:
                loss_parts.append(f"Edge({self.config.LOSS_EDGE})")
            print(f"   损失: {' + '.join(loss_parts)}")
            aug_mode = "基础(裁剪+几何)" if self.config.USE_SIMPLE_AUGMENT else "完整(+颜色+模糊+噪声+天气)"
            print(f"   增强: {aug_mode}")
            # 模块激活状态摘要
            print(f"\n📋 模块激活状态:")
            print(f"   {'✅' if self.config.USE_REFINEMENT     else '❌'} RefineDecoder  (两阶段精修)")
            print(f"   {'✅' if self.config.USE_BOUNDARY_HEAD  else '❌'} BoundaryHead   (边界分支 + 残差注入)")
            print(f"   {'✅' if self.config.USE_RESIDUAL       else '❌'} ResidualMode   (RefineDecoder 残差修补)")
            print(f"   {'✅' if self.config.DETACH_PROMPT      else '❌'} DetachPrompt   (prompt 梯度 detach)")
            edge_status = f"Edge Loss = {self.config.LOSS_EDGE}" if (self.config.USE_BOUNDARY_HEAD and self.config.LOSS_EDGE > 0) else "未启用"
            print(f"   📌 边界 Loss: {edge_status}")
            print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.config.EPOCHS + 1):
            if hasattr(self, 'train_sampler') and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            train_losses = self.train_epoch(epoch)
            
            if is_main_process():
                print(f"\n[Epoch {epoch}/{self.config.EPOCHS}]")
                loss_parts = [f"Refine: {train_losses['loss_refine']:.4f}"]
                if train_losses.get('loss_coarse', 0.0) > 0:
                    loss_parts.append(f"Coarse: {train_losses['loss_coarse']:.4f}")
                if train_losses.get('loss_native', 0.0) > 0:
                    loss_parts.append(f"Native: {train_losses['loss_native']:.4f}")
                if train_losses.get('loss_edge', 0.0) > 0:
                    loss_parts.append(f"Edge: {train_losses['loss_edge']:.4f}")
                print(f"  Train - Total: {train_losses['total']:.4f}, {', '.join(loss_parts)}")
                self.history['train_loss'].append(train_losses['total'])
            
            # 验证
            val_results = self.validate(compute_metrics=True)
            
            if is_main_process():
                self.history['val_loss'].append(val_results['total'])
                self.history['MAE'].append(val_results.get('MAE', 0.0))
                
                if 'Sm' in val_results:
                    self.history['Sm'].append(val_results['Sm'])
                    self.history['wFm'].append(val_results['wFm'])
                    self.history['maxFm'].append(val_results['maxFm'])
                    self.history['maxEm'].append(val_results['maxEm'])
                
                print(f"  Val   - Loss: {val_results['total']:.4f}, "
                      f"Refine: {val_results['loss_refine']:.4f}")
                
                # 打印指标
                if 'Sm' in val_results:
                    print(f"  Pred  - MAE: {val_results['MAE']:.4f}, "
                          f"Sm: {val_results['Sm']:.4f}, "
                          f"wFm: {val_results['wFm']:.4f}, "
                          f"maxFm: {val_results['maxFm']:.4f}, "
                          f"maxEm: {val_results['maxEm']:.4f}")
                else:
                    print(f"  Pred  - MAE: {val_results['MAE']:.4f}, "
                          f"IoU: {val_results.get('IoU', 0):.4f}, "
                          f"Dice: {val_results.get('Dice', 0):.4f}")
                
                # 保存最佳模型
                metric_name = 'Sm' if 'Sm' in val_results else 'IoU'
                current_metric = val_results.get(metric_name, 0.0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.best_count += 1
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        val_results['total'],
                        str(self.save_dir / 'best_model.pth'),
                        history=self.history,
                        scaler=self.scaler,
                        best_metric=self.best_metric,
                        best_count=self.best_count
                    )
                    print(f"  ✅ 新的最佳模型! {metric_name}: {self.best_metric:.4f}")
                    
                    if self.best_count % self.config.VIS_EVERY_N_BEST == 0:
                        vis_path = self.save_dir / f'vis_epoch_{epoch}_{metric_name}_{self.best_metric:.4f}.png'
                        visualize_segmentation(
                            self.model,
                            self.val_dataset,
                            self.device,
                            str(vis_path),
                            num_images=6,
                            threshold=0.5
                        )
                
                # 每轮结束保存 last_model (确保中途中断也能续训)
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    train_losses['total'],
                    str(self.save_dir / 'last_model.pth'),
                    history=self.history,
                    scaler=self.scaler,
                    best_metric=self.best_metric,
                    best_count=self.best_count
                )

                self._save_history()
                self._plot_training_curves()

        if is_main_process():
            print(f"\n{'='*60}")
            print("✅ V5 训练完成!")
            print(f"   时间戳: {self.timestamp}")
            print(f"   最佳指标: {self.best_metric:.4f}")
            print(f"   保存目录: {self.save_dir}")
            print(f"{'='*60}\n")
            
            self._close_log()
    
    def _close_log(self):
        """关闭日志文件，恢复 stdout/stderr"""
        if hasattr(self, '_tee_stdout'):
            sys.stdout = self._tee_stdout.terminal
        if hasattr(self, '_tee_stderr'):
            sys.stderr = self._tee_stderr.terminal
        if hasattr(self, '_log_file') and self._log_file and not self._log_file.closed:
            self._log_file.close()
    
    def _save_history(self):
        """保存训练历史"""
        import json
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        if self.history['train_loss']:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        if self.history['MAE']:
            axes[0, 1].plot(self.history['MAE'], color='orange')
            axes[0, 1].set_title('MAE (↓)')
            axes[0, 1].grid(True)
        
        if self.history['Sm']:
            axes[0, 2].plot(self.history['Sm'], color='green')
            axes[0, 2].set_title('Sm (↑)')
            axes[0, 2].grid(True)
        
        if self.history['wFm']:
            axes[1, 0].plot(self.history['wFm'], color='purple')
            axes[1, 0].set_title('wFm (↑)')
            axes[1, 0].grid(True)
        
        if self.history['maxFm']:
            axes[1, 1].plot(self.history['maxFm'], color='red')
            axes[1, 1].set_title('maxFm (↑)')
            axes[1, 1].grid(True)
        
        if self.history['maxEm']:
            axes[1, 2].plot(self.history['maxEm'], color='cyan')
            axes[1, 2].set_title('maxEm (↑)')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    # 初始化 DDP
    local_rank = None
    if CONFIG.USE_DDP:
        result = setup_ddp()
        if result[0] is not None:
            local_rank, rank, world_size = result
            device = torch.device(f'cuda:{local_rank}')
            if is_main_process():
                print(f"DDP 已初始化: rank={rank}, world_size={world_size}")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(CONFIG.SEED)
    
    if is_main_process():
        print(f"\n{'='*60}")
        # 动态拼接模式描述
        modules = []
        if CONFIG.USE_REFINEMENT:
            modules.append("Refine")
        if CONFIG.USE_BOUNDARY_HEAD:
            modules.append("Boundary")
        mode_str = " + ".join(modules) if modules else "Baseline"
        print(f"📦 V5 - {mode_str}")
        print(f"   设备: {device}")
        print(f"   数据集: {CONFIG.DATA_ROOT}")
        print(f"   输入尺寸: dataset={CONFIG.IMAGE_SIZE}, encoder={CONFIG.ENCODER_INPUT_SIZE}")
        print(
            f"   Encoder AMP: {CONFIG.ENCODER_USE_AMP} "
            f"(dtype={CONFIG.ENCODER_AMP_DTYPE}, trainable={CONFIG.ENCODER_TRAINABLE_USE_AMP})"
        )
        # 动态拼接 loss 描述
        lt_name = 'Structure' if CONFIG.LOSS_TYPE == 'structure' else 'BCE+Dice'
        loss_parts = [f"{lt_name}({CONFIG.LOSS_MAIN_WEIGHT})"]
        if CONFIG.LOSS_COARSE > 0:
            loss_parts.append(f"Coarse({CONFIG.LOSS_COARSE})")
        if CONFIG.LOSS_NATIVE > 0:
            loss_parts.append(f"Native({CONFIG.LOSS_NATIVE})")
        if CONFIG.USE_BOUNDARY_HEAD and CONFIG.LOSS_EDGE > 0:
            loss_parts.append(f"Edge({CONFIG.LOSS_EDGE})")
        print(f"   Loss: {' + '.join(loss_parts)}")
        print(f"{'='*60}\n")
    
    # 构建模型
    model = build_sam3_cod_model_v2(
        model_type=CONFIG.BACKBONE_TYPE,
        sam3_checkpoint=CONFIG.SAM3_CHECKPOINT,
        encoder_input_size=CONFIG.ENCODER_INPUT_SIZE,
        encoder_use_amp=CONFIG.ENCODER_USE_AMP,
        encoder_amp_dtype=CONFIG.ENCODER_AMP_DTYPE,
        encoder_trainable_use_amp=CONFIG.ENCODER_TRAINABLE_USE_AMP,
        freeze_encoder=CONFIG.FREEZE_ENCODER,
        smart_finetune=CONFIG.SMART_FINETUNE,
        use_refinement=CONFIG.USE_REFINEMENT,
        detach_prompt=CONFIG.DETACH_PROMPT,
        use_residual=CONFIG.USE_RESIDUAL,
        use_boundary_head=CONFIG.USE_BOUNDARY_HEAD,
        verbose=is_main_process()
    )
    
    # 数据集
    if CONFIG.USE_SIMPLE_AUGMENT:
        train_transforms = get_simple_train_transforms(CONFIG.IMAGE_SIZE)
        if is_main_process():
            print("📦 使用基础增强: 裁剪 + 几何变换(4种)")
    else:
        train_transforms = None
        if is_main_process():
            print("📦 使用完整增强: 裁剪 + 几何 + 颜色 + 模糊 + 噪声 + 天气")
    
    train_dataset = CAMODataset(
        root_dir=CONFIG.DATA_ROOT,
        split="train",
        image_size=CONFIG.IMAGE_SIZE,
        transforms=train_transforms,
        copy_paste=None,
        use_mask_aware_crop=CONFIG.USE_MASK_AWARE_CROP,
        verbose=is_main_process()
    )
    
    val_dataset = CAMODataset(
        root_dir=CONFIG.DATA_ROOT,
        split="test",
        image_size=CONFIG.IMAGE_SIZE,
        copy_paste=None,
        use_mask_aware_crop=False,
        verbose=is_main_process(),
        test_split=CONFIG.TEST_SPLIT
    )
    
    if is_main_process():
        print(f"训练集: {len(train_dataset)} 张")
        print(f"验证集: {len(val_dataset)} 张")
    
    # 数据加载器 (num_workers=0 时不传 persistent_workers/prefetch_factor，避免兼容性问题)
    dl_kwargs = dict(num_workers=CONFIG.NUM_WORKERS, pin_memory=True)
    if CONFIG.NUM_WORKERS > 0:
        dl_kwargs.update(persistent_workers=True, prefetch_factor=8)

    train_sampler = None
    if local_rank is not None and dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            sampler=train_sampler,
            drop_last=True,
            **dl_kwargs
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            **dl_kwargs
        )

    # 验证集: 不使用 DistributedSampler，只在 rank0 上用全量数据验证
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        **dl_kwargs
    )
    
    # 训练器
    trainer = V2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        config=CONFIG,
        device=device,
        local_rank=local_rank,
        train_sampler=train_sampler
    )
    
    # 开始训练
    try:
        trainer.train()
    finally:
        if hasattr(trainer, '_close_log'):
            trainer._close_log()
        cleanup_ddp()


if __name__ == "__main__":
    # 如果启用 DDP 但没有环境变量，自动用 torchrun 重启
    if CONFIG.USE_DDP and 'RANK' not in os.environ:
        import subprocess
        import sys
        
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"检测到 {num_gpus} 个 GPU，自动启用 DDP 多卡训练...")
            cmd = [
                sys.executable, '-m', 'torch.distributed.run',
                f'--nproc_per_node={num_gpus}',
                sys.argv[0]
            ] + sys.argv[1:]
            subprocess.run(cmd)
        else:
            if num_gpus == 1:
                print("检测到 1 个 GPU，使用单卡模式")
            else:
                print("未检测到 GPU，使用 CPU 模式")
            main()
    else:
        main()
