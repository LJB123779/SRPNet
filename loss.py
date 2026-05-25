"""
V5 损失函数 - 主损失 (Structure / BCE+Dice) + Edge Loss

设计理念：
    主损失 (通过 loss_type 切换，用于消融对比):
        - 'structure':  Structure Loss (加权 BCE + 加权 IoU)，对边界/难区域赋更高权重
        - 'bce_dice':   BCE + Dice，经典分割损失组合
    辅助损失:
        - Edge Loss:    边界级二分类损失 (先全分辨率提边，再 maxpool 下采样到 edge_pred 尺寸)

关键设计:
    Edge Loss: 先在全分辨率 GT 上提取边界，再用 maxpool 下采样到 edge_pred 尺寸。
    相比“先下采样 mask 再提边”，maxpool 保边能保留更完整的边界信息。

开关控制：
    loss_type:    'structure' / 'bce_dice' 切换主损失
    edge_weight:  > 0 且 outputs 含 'edge_logits' 时，自动计算 Edge Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def structure_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Structure Loss: 加权 BCE + 加权 IoU

    用 avgpool 得到边界权重，对边界/难区域赋更高权重。
    COD/SOD 场景中比纯 BCE+Dice 更容易压 MAE。

    Args:
        logits: 预测 logits (B, 1, H, W)，未经 sigmoid
        mask:   GT mask (B, 1, H, W)，{0, 1}

    Returns:
        Structure Loss 标量
    """
    pred = logits
    with torch.amp.autocast(device_type=pred.device.type, enabled=False):
        pred = pred.float()
        mask = mask.float()

        # 经典做法: 用 avgpool 得到边界权重
        weit = 1.0 + 5.0 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )

        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred_prob = torch.sigmoid(pred)
        inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
        union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
        wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

        return (wbce + wiou).mean()


class DiceLoss(nn.Module):
    """
    Dice 损失

    Dice = 2|A∩B| / (|A| + |B|)
    DiceLoss = 1 - Dice
    有效解决类别不平衡问题
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   预测 logits (B, 1, H, W)，未经 sigmoid
            target: GT mask (B, 1, H, W)，{0, 1}

        Returns:
            Dice 损失标量
        """
        pred = pred.float()
        pred = torch.sigmoid(pred)
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1).float()

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class EdgeLoss(nn.Module):
    """
    边界损失 (BCE-based)

    监督策略: 先在全分辨率 GT 上提边，再 maxpool 下采样到 edge_pred 尺寸。
    相比“先下采样 mask 再提边”，能保留更完整的边界信息，显著增强边界分支的有效监督。
    maxpool 下采样确保边界像素不会被 nearest/bilinear 抑制。
    """

    def __init__(self):
        super().__init__()
        # Laplacian 内核提取边界
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

    def _get_edge_gt(self, mask: torch.Tensor) -> torch.Tensor:
        """从 GT mask 提取边界标签 (abs Laplacian 后二值化)"""
        # mask: (B, 1, H, W) float, {0, 1}
        kernel = self.laplacian.to(device=mask.device, dtype=mask.dtype)
        edge = F.conv2d(mask, kernel, padding=1)
        edge = edge.abs().clamp(0, 1)
        return (edge > 0.05).float()

    def forward(
        self,
        pred: torch.Tensor,
        target_fullres: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred:            边界预测 logits (B, 1, H_r, W_r)，低分辨率
            target_fullres:  GT 分割 mask (B, 1, H, W)，全分辨率 {0, 1}

        Returns:
            边界 BCE 损失

        Note:
            显式禁用 autocast，因为内部 F.conv2d(laplacian) 需要 float32 精度。
        """
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target_fullres = target_fullres.float()
            if target_fullres.max() > 1.0:
                target_fullres = target_fullres / 255.0
            target_fullres = (target_fullres > 0.5).float()

            # 1. 在全分辨率提边
            edge_gt_full = self._get_edge_gt(target_fullres)  # (B, 1, H, W)

            # 2. maxpool 下采样到 edge_pred 尺寸，保留边界像素
            if edge_gt_full.shape[-2:] != pred.shape[-2:]:
                # 自适应 kernel_size: 保证能整除，否则用 adaptive_max_pool2d
                edge_gt = F.adaptive_max_pool2d(edge_gt_full, output_size=pred.shape[-2:])
            else:
                edge_gt = edge_gt_full

            # 边界像素极少 (不平衡)，加权补偿；clamp 上限防止极端样本 loss 爆炸
            pos_ratio = edge_gt.mean().clamp(min=1e-4)
            pos_weight = ((1.0 - pos_ratio) / pos_ratio).clamp(max=100.0).detach()

            return F.binary_cross_entropy_with_logits(
                pred, edge_gt,
                pos_weight=pos_weight,
                reduction='mean'
            )


class CODSegmentationLossV2(nn.Module):
    """
    分割损失 - 两阶段监督 + Edge Loss

    组合策略:
        L = L_refine + coarse_weight * L_coarse + native_weight * L_native + edge_weight * L_edge
        
        - L_refine: 主损失，监督最终上采样输出 (H/1)
        - L_coarse: 监督 coarse 阶段 m0 (H/S)，确保第一阶段学会大体定位
        - L_native: 监督 refine 原生分辨率 m1 (H/T)，让精修阶段在中等分辨率先学好
        - L_edge:   边界损督 (可选)
    """

    VALID_LOSS_TYPES = ('structure', 'bce_dice')

    def __init__(
        self,
        loss_type: str = 'structure',
        main_weight: float = 1.0,
        coarse_weight: float = 0.4,
        native_weight: float = 0.3,
        edge_weight: float = 0.0,
        **kwargs  # 兼容上层传入的其他参数
    ):
        """
        Args:
            loss_type:      主损失类型 ('structure' / 'bce_dice')
            main_weight:    主损失权重 (监督最终输出)
            coarse_weight:  coarse 阶段监督权重 (监督 m0)
            native_weight:  native refine 监督权重 (监督 m1 原生分辨率)
            edge_weight:    Edge Loss 权重 (0.0 表示不启用)
            **kwargs: 兼容上层传入的其他参数
        """
        super().__init__()
        assert loss_type in self.VALID_LOSS_TYPES, (
            f"loss_type 必须是 {self.VALID_LOSS_TYPES} 之一，得到: '{loss_type}'"
        )

        self.loss_type = loss_type
        self.main_weight = main_weight
        self.coarse_weight = coarse_weight
        self.native_weight = native_weight
        self.edge_weight = edge_weight

        # BCE+Dice 模式下需要 DiceLoss 实例
        if loss_type == 'bce_dice':
            self.dice = DiceLoss()

        self.edge_loss = EdgeLoss() if edge_weight > 0.0 else None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 模型输出字典
                - 'refined_mask':      上采样到原始分辨率的掩码 (主输出, B,1,H,W)
                - 'masks':             coarse logits (B,1,H/S,W/S)，可选
                - 'refine_logits_low': refine 原生分辨率 logits (B,1,H/T,W/T)，可选
                - 'edge_logits':       边界预测 logits (B,1,H_r,W_r)，可选
            target: 真値掩码 (B, 1, H, W)

        Returns:
            损失字典: total / loss_refine / loss_coarse / loss_native / loss_edge
        """
        losses = {}

        # 安全归一化 target: 确保是 {0, 1}
        target = target.float()
        if target.max() > 1.0:
            target = target / 255.0
        target = (target > 0.5).float()

        total_loss = target.new_tensor(0.0)

        # ========== 1. 主损失: 监督最终上采样输出 (refined_mask) ==========
        pred_refine = outputs['refined_mask']
        if pred_refine.shape[-2:] != target.shape[-2:]:
            pred_refine = F.interpolate(
                pred_refine, size=target.shape[-2:],
                mode='bilinear', align_corners=False
            )

        if self.loss_type == 'structure':
            loss_refine = structure_loss(pred_refine, target)
        else:  # bce_dice
            loss_bce = F.binary_cross_entropy_with_logits(
                pred_refine.float(), target.float(), reduction='mean'
            )
            loss_dice = self.dice(pred_refine, target)
            loss_refine = loss_bce + loss_dice

        losses['loss_refine'] = loss_refine
        total_loss = total_loss + self.main_weight * loss_refine

        # ========== 2. Coarse 监督: 直接监督 m0，确保第一阶段学会大体定位 ==========
        if 'masks' in outputs and self.coarse_weight > 0:
            pred_coarse = outputs['masks']  # (B, 1, H/S, W/S)
            # 下采样 GT 到 coarse 尺寸
            target_coarse = F.interpolate(
                target, size=pred_coarse.shape[-2:],
                mode='bilinear', align_corners=False
            )
            target_coarse = (target_coarse > 0.5).float()

            if self.loss_type == 'structure':
                loss_coarse = structure_loss(pred_coarse, target_coarse)
            else:  # bce_dice
                loss_bce_c = F.binary_cross_entropy_with_logits(
                    pred_coarse.float(), target_coarse.float(), reduction='mean'
                )
                loss_dice_c = self.dice(pred_coarse, target_coarse)
                loss_coarse = loss_bce_c + loss_dice_c

            losses['loss_coarse'] = loss_coarse
            total_loss = total_loss + self.coarse_weight * loss_coarse
        else:
            losses['loss_coarse'] = target.new_tensor(0.0)

        # ========== 3. Native Refine 监督: 监督 m1 原生分辨率 (H/T) ==========
        if 'refine_logits_low' in outputs and self.native_weight > 0:
            pred_native = outputs['refine_logits_low']  # (B, 1, H/T, W/T)
            # 下采样 GT 到 native 尺寸
            target_native = F.interpolate(
                target, size=pred_native.shape[-2:],
                mode='bilinear', align_corners=False
            )
            target_native = (target_native > 0.5).float()

            if self.loss_type == 'structure':
                loss_native = structure_loss(pred_native, target_native)
            else:  # bce_dice
                loss_bce_n = F.binary_cross_entropy_with_logits(
                    pred_native.float(), target_native.float(), reduction='mean'
                )
                loss_dice_n = self.dice(pred_native, target_native)
                loss_native = loss_bce_n + loss_dice_n

            losses['loss_native'] = loss_native
            total_loss = total_loss + self.native_weight * loss_native
        else:
            losses['loss_native'] = target.new_tensor(0.0)

        # ========== 4. Edge Loss: 边界监督 (可选) ==========
        if self.edge_loss is not None and 'edge_logits' in outputs:
            edge_pred = outputs['edge_logits']  # (B, 1, H_r, W_r)
            loss_edge = self.edge_loss(edge_pred, target)  # 传全分辨率 target
            losses['loss_edge'] = loss_edge
            total_loss = total_loss + self.edge_weight * loss_edge
        else:
            losses['loss_edge'] = target.new_tensor(0.0)

        losses['total'] = total_loss
        return losses


def build_segmentation_criterion_v2(
    loss_type: str = 'structure',
    main_weight: float = 1.0,
    coarse_weight: float = 0.4,
    native_weight: float = 0.3,
    edge_weight: float = 0.0,
    **kwargs  # 兼容上层传入的其他参数
) -> CODSegmentationLossV2:
    """
    构建分割损失函数

    Args:
        loss_type:      主损失类型 ('structure' / 'bce_dice')
        main_weight:    主损失权重 (监督最终输出)
        coarse_weight:  coarse 阶段监督权重 (监督 m0)
        native_weight:  native refine 监督权重 (监督 m1 原生分辨率)
        edge_weight:    Edge Loss 权重 (0.0 表示不启用)
        **kwargs: 兼容参数

    Returns:
        CODSegmentationLossV2 实例
    """
    return CODSegmentationLossV2(
        loss_type=loss_type,
        main_weight=main_weight,
        coarse_weight=coarse_weight,
        native_weight=native_weight,
        edge_weight=edge_weight,
    )
