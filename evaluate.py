"""
V4 Baseline 分割评估指标模块

伪装物体分割 (Camouflaged Object Segmentation) 评估指标
使用 py-sod-metrics 社区标准实现

包含指标:
- MAE: Mean Absolute Error
- S-measure: Structure-measure (ICCV 2017)
- E-measure: Enhanced-alignment Measure (IJCAI 2018)
- Weighted F-measure: Frequency-tuned F-measure (CVPR 2009)
- maxF / meanF / adpF: 多阈值 F-measure
- maxE / meanE / adpE: 多阈值 E-measure

安装依赖:
    pip install py-sod-metrics
"""

import numpy as np
import torch
from typing import Dict, List

# 可选依赖: py-sod-metrics
# 未安装时使用简化版指标 (MAE/IoU/Dice)
try:
    from py_sod_metrics import MAE, Smeasure, Emeasure, WeightedFmeasure, Fmeasure
    HAS_SOD_METRICS = True
except ImportError:
    HAS_SOD_METRICS = False
    import warnings
    warnings.warn(
        "\n" + "="*60 + "\n"
        "⚠️ py-sod-metrics 未安装，将使用简化版指标 (MAE/IoU/Dice)\n"
        "完整指标请安装: pip install py-sod-metrics\n"
        + "="*60,
        UserWarning
    )


class SimpleCODMetrics:
    """
    简化版 COD 指标 (不依赖 py-sod-metrics)
    
    仅计算基础指标: MAE, IoU, Dice
    用于 py-sod-metrics 未安装时的 fallback
    """
    
    def __init__(self):
        self.mae_sum = 0.0
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.count = 0
        
    def reset(self):
        self.mae_sum = 0.0
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.count = 0
        
    def update(self, pred: np.ndarray, target: np.ndarray):
        """更新指标 (单张图像)"""
        pred = pred.astype(np.float64)
        target = (target > 0.5).astype(np.float64)
        
        # MAE
        self.mae_sum += np.abs(pred - target).mean()
        
        # 二值化预测
        pred_binary = (pred > 0.5).astype(np.float64)
        
        # IoU
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou = intersection / (union + 1e-8)
        self.iou_sum += iou
        
        # Dice
        dice = (2 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)
        self.dice_sum += dice
        
        self.count += 1
        
    def compute_metrics(self) -> Dict[str, float]:
        if self.count == 0:
            return {'MAE': 0.0, 'IoU': 0.0, 'Dice': 0.0}
        return {
            'MAE': self.mae_sum / self.count,
            'IoU': self.iou_sum / self.count,
            'Dice': self.dice_sum / self.count,
        }
    
    def sync_across_processes(self):
        """DDP: 聚合各 rank 的统计值"""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
                stats = torch.tensor([self.mae_sum, self.iou_sum, self.dice_sum, self.count],
                                    dtype=torch.float64, device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                self.mae_sum, self.iou_sum, self.dice_sum, self.count = stats.tolist()
                self.count = int(self.count)
        except Exception:
            pass  # 非 DDP 模式或未初始化，跳过


class CODMetrics:
    """
    COD/SOD 标准评估指标
    
    py-sod-metrics 已安装时使用完整指标，否则使用简化版
    
    指标说明:
    - MAE: 越低越好
    - Sm (S-measure): 结构相似性，越高越好
    - wFm (Weighted F-measure): 加权 F-measure，越高越好
    - maxFm / meanFm / adpFm: 多阈值 F-measure
    - maxEm / meanEm / adpEm: 多阈值 E-measure
    
    重要: 输入 pred 必须是 [0,1] 概率图，如果模型输出是 logits，需要先 sigmoid:
        probs = torch.sigmoid(model_output['refined_mask'])
        evaluator.update(probs, masks)
    """
    
    def __init__(self):
        """初始化评估器"""
        self.use_full_metrics = HAS_SOD_METRICS
        if self.use_full_metrics:
            self.mae = MAE()
            self.sm = Smeasure()
            self.em = Emeasure()
            self.wfm = WeightedFmeasure()
            self.fm = Fmeasure()
        else:
            self.simple_metrics = SimpleCODMetrics()
        
    def reset(self):
        """重置所有指标"""
        if self.use_full_metrics:
            self.mae = MAE()
            self.sm = Smeasure()
            self.em = Emeasure()
            self.wfm = WeightedFmeasure()
            self.fm = Fmeasure()
        else:
            self.simple_metrics.reset()
        
    @torch.no_grad()
    def update(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ):
        """
        更新指标
        
        Args:
            pred: 预测掩码 (B, 1, H, W) 或 (B, H, W)，值域 [0, 1]
            target: 真值掩码 (B, 1, H, W) 或 (B, H, W)，值域 {0, 1}
        """
        # 转换为 numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
            
        # 调整维度
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)
        
        # 处理单张图像
        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]
            
        # 确保值域正确
        # py-sod-metrics 需要 uint8 [0,255] 格式
        pred = np.clip(pred, 0, 1)
        target = (target > 0.5).astype(np.float64)
        
        # 批量处理
        batch_size = pred.shape[0]
        for i in range(batch_size):
            p = pred[i]
            t = target[i]
            
            if self.use_full_metrics:
                # py-sod-metrics 需要 uint8 [0,255] 格式
                # P2-7: 使用四舍五入而非截断，更接近标准量化
                p_uint8 = (p * 255 + 0.5).astype(np.uint8)
                t_uint8 = (t * 255 + 0.5).astype(np.uint8)
                self.mae.step(pred=p_uint8, gt=t_uint8)
                self.sm.step(pred=p_uint8, gt=t_uint8)
                self.em.step(pred=p_uint8, gt=t_uint8)
                self.wfm.step(pred=p_uint8, gt=t_uint8)
                self.fm.step(pred=p_uint8, gt=t_uint8)
            else:
                self.simple_metrics.update(p, t)
            
    def sync_across_processes(self):
        """DDP: 聚合各 rank 的统计值"""
        if not self.use_full_metrics:
            self.simple_metrics.sync_across_processes()
        # 注意: py-sod-metrics 完整指标的 DDP 聚合较复杂，暂不支持
        # 完整指标模式下建议在主进程单独验证或使用 SimpleCODMetrics
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            指标字典
        """
        if not self.use_full_metrics:
            return self.simple_metrics.compute_metrics()
            
        mae_result = self.mae.get_results()
        sm_result = self.sm.get_results()
        em_result = self.em.get_results()
        wfm_result = self.wfm.get_results()
        fm_result = self.fm.get_results()
        
        # 空数据保护: curve 可能是数组或标量
        fm_curve = np.atleast_1d(fm_result['fm']['curve'])  
        em_curve = np.atleast_1d(em_result['em']['curve'])
        
        # 过滤 NaN
        fm_valid = fm_curve[~np.isnan(fm_curve)]
        em_valid = em_curve[~np.isnan(em_curve)]
        
        return {
            'MAE': mae_result.get('mae', 0.0),
            'Sm': sm_result.get('sm', 0.0),
            'wFm': wfm_result.get('wfm', 0.0),
            'maxFm': float(fm_valid.max()) if len(fm_valid) > 0 else 0.0,
            'meanFm': float(fm_valid.mean()) if len(fm_valid) > 0 else 0.0,
            'adpFm': fm_result['fm'].get('adp', 0.0),
            'maxEm': float(em_valid.max()) if len(em_valid) > 0 else 0.0,
            'meanEm': float(em_valid.mean()) if len(em_valid) > 0 else 0.0,
            'adpEm': em_result['em'].get('adp', 0.0),
        }


class CODEvaluator:
    """
    伪装物体检测评估器 (直接使用 py-sod-metrics)
    
    重要: 输入 pred 必须是 [0,1] 概率图，如果模型输出是 logits，需要先 sigmoid:
        with torch.no_grad():
            logits = model(images)['refined_mask']
            probs = torch.sigmoid(logits)
            evaluator.update(probs, masks)
    """
    
    def __init__(self):
        self.metrics = CODMetrics()
            
    def reset(self):
        self.metrics.reset()
        
    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metrics.update(pred, target)
    
    def sync_across_processes(self):
        """DDP: 聚合各 rank 的统计值"""
        self.metrics.sync_across_processes()
        
    def compute_metrics(self) -> Dict[str, float]:
        return self.metrics.compute_metrics()


def evaluate_predictions(
    predictions: List[np.ndarray],
    targets: List[np.ndarray]
) -> Dict[str, float]:
    """
    评估预测结果
    
    Args:
        predictions: 预测掩码列表，值域 [0, 1]
        targets: 真值掩码列表，值域 {0, 1}
        
    Returns:
        评估指标字典
    """
    evaluator = CODEvaluator()
    
    for pred, target in zip(predictions, targets):
        # P1-5: 统一维度为 (1, 1, H, W)，评估接口更规范
        # pred/target: (H, W) -> (1, 1, H, W)
        pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
        target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
        evaluator.update(pred_tensor, target_tensor)
        
    return evaluator.compute_metrics()
