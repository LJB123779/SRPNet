"""
V5 模型 - 支持 Baseline (V0) 和两阶段精修 (V1)

模型路径:
    V0 (Baseline): SAM3 Encoder → 1x1 Proj → CoarseDecoder → Upsample
    V1 (Refine):   V0 + FeatureUpsampler → [BoundaryHead]
                       → PromptProjection → RefineDecoder → Upsample

关键设计:
    - FeatureUpsampler 根据 SAM3 实际 stride 自适应上采样层数，精修固定在 H/4 (target_stride=4)
    - Edge Loss 先在全分辨率 GT 上提边，再 maxpool 下采样到 edge_pred 尺寸

控制开关:
    - use_refinement:     False → V0 单阶段, True → V1 两阶段精修
    - use_boundary_head:  边界分支 + edge_feat 残差注入到 feat (zero-init 安全起步)
    - detach_prompt:      是否 detach prompt 梯度 (避免梯度耦合)
    - use_residual:       RefineDecoder 是否以残差修补模式运行
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# SAM3 模型配置
# =============================================================================

@dataclass
class SAM3Config:
    """SAM3 模型配置"""
    model_type: str = "vit_b"
    image_size: int = 512
    encoder_use_amp: bool = True
    encoder_amp_dtype: str = "auto"
    encoder_trainable_use_amp: bool = False
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_multimask_outputs: int = 3
    mask_decoder_dim: int = 256
    text_embed_dim: int = 512

    @classmethod
    def from_model_type(cls, model_type: str) -> 'SAM3Config':
        configs = {
            'vit_b': {'embed_dim': 768,  'depth': 12, 'num_heads': 12},
            'vit_l': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
            'vit_h': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16},
        }
        if model_type not in configs:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(configs.keys())}")
        return cls(model_type=model_type, **configs[model_type])


# =============================================================================
# SAM3 图像编码器
# =============================================================================

class SAM3ImageEncoder(nn.Module):
    """
    SAM3 图像编码器

    只返回深层特征 (H/32)，支持完全冻结 / 智能微调 / 全解冻三种策略。
    """

    def __init__(
        self,
        config: SAM3Config,
        checkpoint_path: str = "weights/sam3.pt",
        freeze: bool = True,
        smart_finetune: bool = True,
        verbose: bool = True
    ):
        super().__init__()
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.freeze = freeze
        self.smart_finetune = smart_finetune
        self.verbose = verbose
        self.sam3_model = None
        self._load_sam3()

    def _resolve_encoder_amp(self, device: torch.device) -> Tuple[bool, Optional[torch.dtype], str]:
        """Resolve whether encoder should use mixed precision on the current device."""
        if device.type != 'cuda' or not self.config.encoder_use_amp:
            return False, None, 'float32'

        if not self.freeze and not self.config.encoder_trainable_use_amp:
            return False, None, 'float32'

        amp_dtype = str(self.config.encoder_amp_dtype).lower()
        bf16_supported = hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()

        if amp_dtype == 'auto':
            if bf16_supported:
                return True, torch.bfloat16, 'bfloat16'
            return True, torch.float16, 'float16'

        if amp_dtype in ('bf16', 'bfloat16'):
            if bf16_supported:
                return True, torch.bfloat16, 'bfloat16'
            if self.verbose and not hasattr(self, '_encoder_amp_warned'):
                self._encoder_amp_warned = True
                print("⚠️  当前 GPU 不支持 bf16，SAM3 encoder AMP 将回退到 float16")
            return True, torch.float16, 'float16'

        if amp_dtype in ('fp16', 'float16', 'half'):
            return True, torch.float16, 'float16'

        raise ValueError(
            f"Unsupported encoder_amp_dtype: {self.config.encoder_amp_dtype}. "
            "Choose from ['auto', 'bf16', 'fp16']."
        )

    def _load_sam3(self):
        """加载 SAM3 官方模型"""
        try:
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:
            raise ImportError(
                "\n" + "="*70 + "\n"
                f"❌ SAM3 导入失败!\n\n错误原因: {e}\n\n"
                "请安装 SAM3: pip install git+https://github.com/facebookresearch/sam3.git\n"
                + "="*70
            )

        import os
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                "\n" + "="*70 + "\n"
                f"❌ SAM3 权重文件不存在: {self.checkpoint_path}\n" + "="*70
            )

        if self.verbose:
            print(f"正在加载 SAM3 预训练模型...")
            print(f"  权重路径: {self.checkpoint_path}")

        self.sam3_model = build_sam3_image_model(
            checkpoint_path=self.checkpoint_path,
            load_from_HF=False,
            eval_mode=True,
            image_size=self.config.image_size,
        )

        if self.freeze:
            for param in self.sam3_model.parameters():
                param.requires_grad = False
            self.sam3_model.eval()
            if self.verbose:
                print("❄️ Encoder 状态: [完全冻结]")
        else:
            if self.smart_finetune:
                for param in self.sam3_model.parameters():
                    param.requires_grad = False
                count_norm = 0
                for name, param in self.sam3_model.named_parameters():
                    if 'norm' in name.lower() or 'ln' in name.lower():
                        param.requires_grad = True
                        count_norm += 1
                count_block = 0
                blocks = self._find_blocks(self.sam3_model.backbone)
                if blocks is not None and len(blocks) >= 2:
                    for block in blocks[-2:]:
                        for param in block.parameters():
                            param.requires_grad = True
                        count_block += 1
                self.sam3_model.train()
                if self.verbose:
                    print(f"🔥 Encoder 状态: [智能微调] Norm×{count_norm} Block×{count_block}")
            else:
                for param in self.sam3_model.parameters():
                    param.requires_grad = True
                self.sam3_model.train()
                if self.verbose:
                    print("🔥🔥 Encoder 状态: [完全解冻]")

        # 探测输出维度 + 空间步幅
        _PROBE_SIZE = self.config.image_size
        with torch.no_grad(), torch.amp.autocast('cpu', enabled=False):
            backbone_cpu = self.sam3_model.backbone.cpu().float()
            dummy = torch.zeros(1, 3, _PROBE_SIZE, _PROBE_SIZE, device='cpu', dtype=torch.float32)
            out = backbone_cpu.forward_image(dummy)
            feat = out['vision_features']
            self.sam3_embed_dim = feat.shape[1]
            feat_h = feat.shape[-1]
            # 推断 stride: 取最近的 2 的幂次 (ViT stride 通常是 14/16/32)
            raw_stride = _PROBE_SIZE / feat_h
            self.sam3_raw_stride = raw_stride
            self.sam3_stride = 2 ** round(math.log2(raw_stride))
            # 保存探测时的实际特征图尺寸，供 forward 形状自检使用
            self.sam3_feat_size = feat_h
            del dummy, out, feat

        if self.verbose:
            print(
                f"   输出维度: {self.sam3_embed_dim}, raw_stride≈{self.sam3_raw_stride:.2f}, "
                f"used_stride=1/{self.sam3_stride} "
                f"(输入 {_PROBE_SIZE} → 特征图 {self.sam3_feat_size}×{self.sam3_feat_size})"
            )

    def _find_blocks(self, backbone) -> Optional[nn.ModuleList]:
        """探测 backbone 的 blocks 结构"""
        if hasattr(backbone, 'trunk') and hasattr(backbone.trunk, 'blocks'):
            return backbone.trunk.blocks
        if hasattr(backbone, 'image_encoder'):
            enc = backbone.image_encoder
            if hasattr(enc, 'blocks'):
                return enc.blocks
            if hasattr(enc, 'trunk') and hasattr(enc.trunk, 'blocks'):
                return enc.trunk.blocks
        if hasattr(backbone, 'blocks'):
            return backbone.blocks
        if hasattr(backbone, 'layers'):
            return backbone.layers
        for _, module in backbone.named_children():
            if isinstance(module, nn.ModuleList) and len(module) >= 4:
                return module
            for _, sub in module.named_children():
                if isinstance(sub, nn.ModuleList) and len(sub) >= 4:
                    return sub
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            deep_features: (B, C, H/32, W/32)
        """
        SAM3_INPUT_SIZE = self.config.image_size
        if self.freeze:
            self.sam3_model.eval()
            # 冻结 encoder 时也强制 float32 + 禁用 autocast，避免 fp16 注意力数值误差
            with torch.no_grad(), torch.amp.autocast(device_type=x.device.type, enabled=False):
                x_input = x.float()
                if x_input.shape[-1] != SAM3_INPUT_SIZE or x_input.shape[-2] != SAM3_INPUT_SIZE:
                    x_input = F.interpolate(
                        x_input, size=(SAM3_INPUT_SIZE, SAM3_INPUT_SIZE),
                        mode='bilinear', align_corners=False
                    )
                out = self.sam3_model.backbone.forward_image(x_input)
                return out['vision_features']

    def _forward_with_configured_precision(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with configurable encoder precision."""
        sam3_input_size = self.config.image_size
        use_amp, amp_dtype, amp_label = self._resolve_encoder_amp(x.device)
        autocast_ctx = (
            torch.amp.autocast(device_type=x.device.type, dtype=amp_dtype)
            if use_amp else
            nullcontext()
        )

        x_input = x.float()
        if x_input.shape[-1] != sam3_input_size or x_input.shape[-2] != sam3_input_size:
            x_input = F.interpolate(
                x_input,
                size=(sam3_input_size, sam3_input_size),
                mode='bilinear',
                align_corners=False
            )

        if self.freeze:
            self.sam3_model.eval()
            with torch.no_grad():
                with autocast_ctx:
                    out = self.sam3_model.backbone.forward_image(x_input)
        else:
            self.sam3_model.train(self.training)
            with autocast_ctx:
                out = self.sam3_model.backbone.forward_image(x_input)

        feat = out['vision_features']
        if use_amp:
            feat = feat.float()

        if self.verbose and not hasattr(self, '_encoder_amp_logged'):
            self._encoder_amp_logged = True
            print(f"   Encoder compute dtype: {amp_label} (freeze={self.freeze})")

        return feat

    forward = _forward_with_configured_precision
    """
        else:
            # 解冻训练: 显式禁用 autocast + float32，避免 ViT 注意力在 fp16 下数值不稳
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                x = x.float()
                if x.shape[-1] != SAM3_INPUT_SIZE or x.shape[-2] != SAM3_INPUT_SIZE:
                    x = F.interpolate(
                        x, size=(SAM3_INPUT_SIZE, SAM3_INPUT_SIZE),
                        mode='bilinear', align_corners=False
                    )
                self.sam3_model.train(self.training)
                out = self.sam3_model.backbone.forward_image(x)
                return out['vision_features']
# =============================================================================

    """

# =============================================================================
# SAM3 掩码解码器 (CoarseDecoder, H/32)
# =============================================================================

class SAM3MaskDecoder(nn.Module):
    """
    SAM3 掩码解码器 (V0 简化版)

    Object Query → TransformerDecoder → broadcast → mask_pred → H/32 logits
    """

    def __init__(self, config: SAM3Config):
        super().__init__()
        self.object_queries = nn.Embedding(1, config.mask_decoder_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.mask_decoder_dim,
                nhead=8,
                dim_feedforward=config.mask_decoder_dim * 4,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        self.mask_pred = nn.Sequential(
            nn.Conv2d(config.mask_decoder_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (B, 256, H/32, W/32)

        Returns:
            logits: (B, 1, H/32, W/32)
        """
        B, C, H, W = image_features.shape
        img_flat = image_features.flatten(2).transpose(1, 2)          # (B, HW, C)
        queries = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, C)
        decoded = self.transformer_decoder(tgt=queries, memory=img_flat)     # (B, 1, C)
        mask_feat = image_features + decoded.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return self.mask_pred(mask_feat)


# =============================================================================
# Refinement 组件: LKABlock / FeatureUpsampler / PromptProjection / RefineDecoder
# =============================================================================

class LKABlock(nn.Module):
    """
    Large Kernel Attention 块 (等效感受野 ~23x23)

    DW-5x5 → DW-dilated-7x7(d=3) → PW-1x1 → 残差连接
    COD 场景伪装目标形状不规则，需要大感受野捕捉远程结构。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw1  = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw2  = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels, bias=False)
        self.pw   = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.pw(x)
        return u + self.act(self.norm(x))


class FeatureUpsampler(nn.Module):
    """
    特征上采样模块: H/S → H/T (自适应层数)

    根据 encoder_stride 和 target_stride 动态计算需要几次 ×2 上采样。
    例如 encoder_stride=32, target_stride=4 → 3 次 ×2 (×8)。
    使用反卷积 + GroupNorm + GELU，逐步降维同时增加分辨率。
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 64,
        encoder_stride: int = 32,
        target_stride: int = 4,
    ):
        super().__init__()
        self.encoder_stride = encoder_stride
        self.target_stride = target_stride

        # 计算需要几次 ×2 上采样
        ratio = encoder_stride // target_stride
        assert ratio >= 1 and (ratio & (ratio - 1)) == 0, \
            f"encoder_stride/target_stride 必须是 2 的幂次, 得到 {encoder_stride}/{target_stride}={ratio}"
        num_stages = int(math.log2(ratio))
        assert num_stages >= 1, f"至少需要 1 次上采样, 但 ratio={ratio}"

        # 动态构建上采样层: 通道从 in_channels 线性过渡到 out_channels
        # 中间通道: 在 [in_channels, out_channels] 之间均匀插值
        channels = []
        for i in range(num_stages + 1):
            ch = int(in_channels + (out_channels - in_channels) * i / num_stages)
            # 确保通道数是 8 的倍数 (GroupNorm 需要)
            ch = max(ch // 8 * 8, out_channels)
            channels.append(ch)
        channels[0] = in_channels
        channels[-1] = out_channels

        # 保存每个上采样 stage 的输出 stride / channels，供多尺度特征融合使用
        # stage 输出顺序: stride 从大到小 (更粗 → 更细), 最后一个 stride=target_stride
        self.stage_strides = [encoder_stride // (2 ** (i + 1)) for i in range(num_stages)]
        self.stage_channels = [channels[i + 1] for i in range(num_stages)]

        stages = []
        for i in range(num_stages):
            c_in, c_out = channels[i], channels[i + 1]
            # GroupNorm 的 num_groups: 确保能整除
            ng = min(16, c_out)
            while c_out % ng != 0:
                ng -= 1
            stages.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(ng, c_out),
                nn.GELU(),
            ))
        self.up_stages = nn.ModuleList(stages)

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.GELU()
        )

    def forward(self, x: torch.Tensor, return_pyramid: bool = False):
        """
        Args:
            x: (B, in_channels, H/S, W/S)  S=encoder_stride
            return_pyramid:
                False → 返回最终特征 (B, out_channels, H/T, W/T)
                True  → 返回每个 stage 的输出特征列表 (从粗到细)，用于多尺度融合

        Returns:
            - return_pyramid=False: (B, out_channels, H/T, W/T)  T=target_stride
            - return_pyramid=True:  List[Tensor], len=num_stages
        """
        feats = []
        for stage in self.up_stages:
            x = stage(x)
            feats.append(x)
        if return_pyramid:
            return feats
        return self.refine(x)


class PromptProjection(nn.Module):
    """
    Prompt 投影模块

    将 coarse mask (sigmoid 后的软掩码, 1ch) 投影到特征空间 (out_channels ch)。
    告诉 RefineDecoder "去哪里精修"。
    """

    def __init__(self, out_channels: int = 64):
        super().__init__()
        mid = max(out_channels // 2, 8)
        self.proj = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid), nn.GELU(),
            nn.Conv2d(mid, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.GELU()
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: soft mask (B, 1, H, W)，值域 [0, 1]

        Returns:
            (B, out_channels, H, W)
        """
        return self.proj(mask)


class RefineDecoder(nn.Module):
    """
    精修解码器 (H/T 分辨率, T=target_stride)

    输入: concat(feat_upsampled, prompt_embed)
    流程: Conv → LKA → 2x 残差块 → 输出头
    残差修补: 输出 = m0_high + Delta (Decoder 只预测修正量)
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 96,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels), nn.GELU()
        )
        self.lka = LKABlock(hidden_channels)
        self.res1 = self._res_block(hidden_channels)
        self.res2 = self._res_block(hidden_channels)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels // 2), nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, 1)
        )

    def _res_block(self, ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
        )

    def forward(
        self,
        x: torch.Tensor,
        m0_high: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:       H/T 融合特征 (B, in_channels, H/T, W/T)
            m0_high: coarse mask 上采样到 H/T 的 logits (用于残差修补)

        Returns:
            mask logits (B, 1, H/T, W/T)
        """
        h = self.conv1(x)
        h = self.lka(h)
        h = h + self.res1(h)
        h = h + self.res2(h)
        delta = self.head(h)
        if self.use_residual and m0_high is not None:
            return m0_high + delta
        return delta


# =============================================================================
# E3 新增: BoundaryHead — 显式边界分支 + 反哺精修特征
# =============================================================================

class BoundaryHead(nn.Module):
    """
    边界预测头 + 边界特征提取模块

    流程:
        feat_T (B, 64, H/T) → boundary_head → edge_logits (B, 1, H/T)
                                                     │ edge_to_feat
                                                     ▼
                              edge_feat (B, 64, H/T)  ← 残差注入到 feat_T

    edge_feat 是 64ch 的富特征，通过残差相加注入到主干特征中 (edge_to_feat zero-init，训练初期不干扰)，
    保持 RefineDecoder 输入通道数不变。
    """

    def __init__(self, feat_channels: int = 64):
        super().__init__()
        mid = feat_channels // 2

        # 边界预测分支: 输出 H/T 边界 logits
        self.boundary_head = nn.Sequential(
            nn.Conv2d(feat_channels, mid, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1)
        )

        # 边界特征反哺: edge_logits → 与 feat_4 同维度
        self.edge_to_feat = nn.Sequential(
            nn.Conv2d(1, feat_channels, 1, bias=False),
            nn.GELU()
        )

        # 零初始化: 训练初期不干扰 feat_4，安全起步
        nn.init.zeros_(self.edge_to_feat[0].weight)

    def forward(self, feat_4: torch.Tensor):
        """
        Args:
            feat_4: 高分辨率特征 (B, feat_channels, H/T, W/T)

        Returns:
            edge_logits: 边界预测 logits (B, 1, H/T, W/T)，供 Edge Loss 使用
            edge_feat:   边界富特征 (B, feat_channels, H/T, W/T)，残差注入到主干特征
        """
        edge_logits = self.boundary_head(feat_4)           # (B, 1, H/T, W/T)
        edge_feat = self.edge_to_feat(edge_logits)         # (B, 64, H/T, W/T)
        return edge_logits, edge_feat


class SAM3CamouflageDetectorV2(nn.Module):
    """
    SAM3 伪装物体分割器

    V0 (use_refinement=False):
        Encoder → Proj → CoarseDecoder (H/32) → Upsample

    V1 (use_refinement=True):
        Stage1: Encoder → Proj → CoarseDecoder (H/S) → m0
        Stage2: FeatureUpsampler(feat_proj) → F_T (H/T, T=S/8)
                [BoundaryHead: F_T → edge_logits; F_T += edge_feat (残差注入)]
                PromptProjection(sigmoid(m0)) → E (H/T)
                RefineDecoder(concat(F_T, E), m0_high) → m1
                Upsample(m1) → output
    """

    REFINE_FEAT_DIM = 64  # FeatureUpsampler 和 PromptProjection 的输出维度

    def __init__(
        self,
        config: Optional[SAM3Config] = None,
        checkpoint_path: str = "weights/sam3.pt",
        freeze_encoder: bool = True,
        smart_finetune: bool = True,
        use_refinement: bool = False,
        detach_prompt: bool = True,
        use_residual: bool = True,
        use_boundary_head: bool = False,
        verbose: bool = True
    ):
        super().__init__()

        self.config = config or SAM3Config()
        self.freeze_encoder = freeze_encoder
        self.use_refinement = use_refinement
        self.detach_prompt = detach_prompt
        self.use_residual = use_residual
        self.use_boundary_head = use_boundary_head
        self.verbose = verbose

        # 1. SAM3 图像编码器
        self.image_encoder = SAM3ImageEncoder(
            self.config,
            checkpoint_path=checkpoint_path,
            freeze=freeze_encoder,
            smart_finetune=smart_finetune,
            verbose=verbose
        )

        # 2. 特征投影 (1×1 Conv: sam3_embed_dim → 256)
        self.feat_proj = nn.Conv2d(
            self.image_encoder.sam3_embed_dim,
            self.config.mask_decoder_dim,
            kernel_size=1
        )

        # 3. Coarse 解码器 (H/32)
        self.coarse_decoder = SAM3MaskDecoder(self.config)

        # 4. Refinement 组件 (仅 use_refinement=True 时初始化)
        self._shape_checked = False  # 首次 forward 时做形状断言
        if self.use_refinement:
            d = self.REFINE_FEAT_DIM
            encoder_stride = self.image_encoder.sam3_stride
            # 精修分辨率固定在 H/4，兼顾精度与显存
            # stride=16 → 上采样 ×4 (2级)，stride=32 → 上采样 ×8 (3级)
            self.target_stride = 4

            # 4.1 深层特征上采样: H/S → H/T (S=encoder_stride, T=target_stride, 自适应层数)
            self.feat_upsampler = FeatureUpsampler(
                in_channels=self.config.mask_decoder_dim,  # 256
                out_channels=d,                             # 64
                encoder_stride=encoder_stride,
                target_stride=self.target_stride,
            )

            if verbose:
                num_up = int(math.log2(encoder_stride // self.target_stride))
                print(f"   FeatureUpsampler: 1/{encoder_stride} → 1/{self.target_stride} ({num_up}×2 上采样)")

            # 4.2 Prompt 投影: coarse mask → 特征空间 (64ch)
            self.prompt_proj = PromptProjection(out_channels=d)

            # 4.3 边界分支 (仅 use_boundary_head=True 时初始化)
            if self.use_boundary_head:
                self.boundary_head = BoundaryHead(feat_channels=d)

            # 4.4 精修解码器
            # concat(F_T, prompt) = 64+64 = 128
            # BoundaryHead: edge_feat 残差注入到 feat_4，不增加通道
            refine_in_ch = d * 2  # F_T + prompt
            self.refine_decoder = RefineDecoder(
                in_channels=refine_in_ch,
                hidden_channels=96,
                use_residual=use_residual
            )

        if verbose:
            print(f"\n{'='*50}")
            print(f"🎯 V5 模型初始化完成")
            if self.use_refinement:
                print(f"   模式: V1 两阶段精修")
                print(f"   Stage1: Encoder → Proj → CoarseDecoder")
                boundary_str = " + BoundaryHead" if use_boundary_head else ""
                print(f"   Stage2: FeatureUpsampler → PromptProj{boundary_str} → RefineDecoder")
                print(f"   RefineDecoder in_channels: {refine_in_ch}")
                print(f"   Detach Prompt: {detach_prompt}, 残差修补: {use_residual}")
                print(f"   边界分支: {use_boundary_head}")
            else:
                print(f"   模式: V0 Baseline 单阶段")
                print(f"   路径: Encoder → Proj → CoarseDecoder → Upsample")
            print(f"   Encoder 冻结: {freeze_encoder}")
            print(f"{'='*50}\n")

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            image: 输入图像 (B, 3, H, W)，已归一化

        Returns:
            输出字典:
                - 'masks':             coarse logits (B, 1, H/S, W/S)
                - 'refined_mask':      最终上采样掩码 (B, 1, H, W)，用于 Loss 计算
                - 'refine_logits_low': refine 原生分辨率 logits (B, 1, H/T, W/T)，用于 native 监督
        """
        # Stage1: 提取特征 → Coarse mask
        H_in, W_in = image.shape[-2:]
        deep_features  = self.image_encoder(image)          # (B, C, H/S, W/S)
        feat_projected = self.feat_proj(deep_features)      # (B, 256, H/S, W/S)
        m0             = self.coarse_decoder(feat_projected) # (B, 1, H/S, W/S)

        # 首次 forward: 形状自检，确保特征图尺寸与初始化时探测一致
        # 直接比较特征图空间尺寸 (避免输入尺寸不能被 stride 整除导致的整数除法误差)
        if not self._shape_checked:
            self._shape_checked = True
            feat_h, feat_w = deep_features.shape[-2:]
            expected_size = self.image_encoder.sam3_feat_size
            assert feat_h == expected_size, (
                f"[形状自检失败] Encoder 特征图高度 {feat_h} != 探测值 {expected_size}, "
                f"输入 {H_in}×{W_in} → 特征 {feat_h}×{feat_w}"
            )
            stride = self.image_encoder.sam3_stride
            print(f"[形状自检] Encoder: 特征图 {feat_h}×{feat_w} "
                  f"(stride≈1/{stride}) ✓")

        if not self.use_refinement:
            # V0: 直接上采样输出
            return {
                'masks': m0,
                'refined_mask': F.interpolate(
                    m0, size=image.shape[-2:],
                    mode='bilinear', align_corners=False
                ),
            }

        # Stage2: Refinement
        # 4.1 深层特征上采样到 H/T (T=target_stride)

        # ✅ 关键修正: SAM3 的实际 stride 往往不是 2 的幂次 (例如 14)。
        # ConvTranspose×2 堆叠得到的尺寸可能与期望的 H/T 不一致，
        # 会导致 edge/prompt 与像素网格对不齐，影响 Sm/MAE。
        target_size = (math.ceil(H_in / self.target_stride), math.ceil(W_in / self.target_stride))

        feat_4 = self.feat_upsampler(feat_projected)        # (B, 64, H_up, W_up)
        raw_up_size = feat_4.shape[-2:]  # 插值前的原始尺寸 (诊断用)
        if feat_4.shape[-2:] != target_size:
            feat_4 = F.interpolate(feat_4, size=target_size, mode='bilinear', align_corners=False)

        # 首次 forward: 形状自检 (在插值前记录原始尺寸，确认对齐是否生效)
        if not hasattr(self, '_upsampler_checked'):
            self._upsampler_checked = True
            aligned = tuple(raw_up_size) != tuple(target_size)
            print(f"[形状自检] FeatureUpsampler 原始输出: {raw_up_size[0]}×{raw_up_size[1]}, "
                  f"目标: {target_size[0]}×{target_size[1]} (H_in/{self.target_stride})"
                  f"{' → 已插值对齐' if aligned else ' → 尺寸一致'} ✓")

        # 4.2 边界分支: 预测边界 logits，残差注入 edge_feat 到 feat_4
        # edge_to_feat 已 zero-init，训练初期不干扰主干特征
        edge_logits = None
        if self.use_boundary_head:
            edge_logits, edge_feat = self.boundary_head(feat_4)
            feat_4 = feat_4 + edge_feat  # 残差注入，保持 64ch 不变

        # 4.3 coarse mask 上采样到 H/T (用于 prompt 和残差修补)
        m0_high = F.interpolate(
            m0, size=feat_4.shape[-2:],
            mode='bilinear', align_corners=False
        )

        # 4.5 生成 soft prompt (sigmoid)，可选 detach 避免梯度耦合
        p0 = torch.sigmoid(m0_high.detach() if self.detach_prompt else m0_high)

        # 4.6 Prompt 投影到特征空间
        prompt_embed = self.prompt_proj(p0)

        # 4.7 拼接: feat_4(已融合边界) + prompt
        feat_combined = torch.cat([feat_4, prompt_embed], dim=1)
        m1 = self.refine_decoder(feat_combined, m0_high=m0_high)

        # 4.8 上采样到原始分辨率
        output = {
            'masks': m0,
            'refine_logits_low': m1,  # 保留 H/T 原生分辨率，用于 native 监督
            'refined_mask': F.interpolate(
                m1, size=image.shape[-2:],
                mode='bilinear', align_corners=False
            ),
        }

        # 携带 edge_logits 供 Loss 计算 (保持原生分辨率，Loss 侧下采样 GT)
        if edge_logits is not None:
            output['edge_logits'] = edge_logits  # (B, 1, H_refine, W_refine)

        return output

    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        trainable = []

        if not self.freeze_encoder:
            trainable.extend(self.image_encoder.parameters())

        trainable.extend(self.feat_proj.parameters())
        trainable.extend(self.coarse_decoder.parameters())

        if self.use_refinement:
            trainable.extend(self.feat_upsampler.parameters())
            trainable.extend(self.prompt_proj.parameters())
            trainable.extend(self.refine_decoder.parameters())
            if self.use_boundary_head:
                trainable.extend(self.boundary_head.parameters())

        return trainable


# =============================================================================
# 工厂函数
# =============================================================================

def build_sam3_cod_model_v2(
    model_type: str = "vit_b",
    sam3_checkpoint: str = "weights/sam3.pt",
    encoder_input_size: int = 640,
    encoder_use_amp: bool = True,
    encoder_amp_dtype: str = "auto",
    encoder_trainable_use_amp: bool = False,
    freeze_encoder: bool = True,
    smart_finetune: bool = True,
    use_refinement: bool = False,
    detach_prompt: bool = True,
    use_residual: bool = True,
    use_boundary_head: bool = False,
    verbose: bool = True,
    **kwargs  # 兼容上层传入的其他参数
) -> SAM3CamouflageDetectorV2:
    """
    构建 V5 模型

    Args:
        model_type:        骨干网络类型 ("vit_b", "vit_l", "vit_h")
        sam3_checkpoint:   SAM3 权重路径
        encoder_input_size: SAM3 encoder 实际输入尺寸
        freeze_encoder:    是否冻结 SAM3 Encoder
        smart_finetune:    智能微调 (True=Norm+TopBlocks, False=全解冻)
        use_refinement:    False → V0 Baseline, True → V1 两阶段精修
        detach_prompt:     是否 detach prompt 梯度
        use_residual:      RefineDecoder 是否使用残差修补模式
        use_boundary_head: True → 启用边界分支 + 残差注入 (需配合 Edge Loss)
        verbose:           是否打印信息
        **kwargs:          兼容参数 (忽略)

    Returns:
        SAM3CamouflageDetectorV2 模型实例
    """
    config = SAM3Config.from_model_type(model_type)
    config.image_size = encoder_input_size
    config.encoder_use_amp = encoder_use_amp
    config.encoder_amp_dtype = encoder_amp_dtype
    config.encoder_trainable_use_amp = encoder_trainable_use_amp

    if verbose:
        mode = "V1 两阶段精修" if use_refinement else "V0 Baseline 单阶段"
        print(f"\n{'='*50}")
        print(f"🔧 构建 V5 模型: {model_type.upper()}")
        print(f"   Embed 维度: {config.embed_dim}")
        print(f"   Encoder 输入: {config.image_size}")
        print(f"   Encoder 冻结: {freeze_encoder}")
        print(f"   模式: {mode}")
        print(f"{'='*50}")

    model = SAM3CamouflageDetectorV2(
        config=config,
        checkpoint_path=sam3_checkpoint,
        freeze_encoder=freeze_encoder,
        smart_finetune=smart_finetune,
        use_refinement=use_refinement,
        detach_prompt=detach_prompt,
        use_residual=use_residual,
        use_boundary_head=use_boundary_head,
        verbose=verbose
    )

    if verbose:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.get_trainable_params())
        print(f"   总参数量: {total / 1e6:.1f}M")
        print(f"   可训练参数: {trainable / 1e6:.1f}M")
        print(f"{'='*50}\n")

    return model
