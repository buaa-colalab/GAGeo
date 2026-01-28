"""
Cross-View Localization Loss Functions

支持四种监督信号：
- bbox: L1 + GIoU loss
- mask: BCE + Dice loss  
- yaw: 周期性角度loss
- camera_position: MSE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class MultiTaskLoss(nn.Module):
    """
    多任务Loss，支持开关控制
    
    Args:
        weight_bbox: BBox loss权重 (L1)
        weight_giou: GIoU loss权重
        weight_mask: Mask loss权重 (BCE + Dice)
        weight_yaw: Yaw角度loss权重
        weight_position: 位置loss权重
    """
    
    def __init__(
        self,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        weight_mask: float = 1.0,
        weight_yaw: float = 1.0,
        weight_position: float = 1.0,
    ):
        super().__init__()
        self.w = {
            'bbox': weight_bbox,
            'giou': weight_giou,
            'mask': weight_mask,
            'yaw': weight_yaw,
            'position': weight_position,
        }
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: {
                'pred_boxes': [B, 4] or [B, N, 4],
                'yaw_radians': [B],
                'position': [B, 2],
                'masks': [B, 1, H, W],
            }
            targets: {
                'sat_bbox': [B, 4],
                'yaw_radians': [B],
                'camera_position': [B, 2],
                'masks': [B, 1, H, W],
            }
        """
        # Cast targets to match output dtype for mixed precision compatibility
        # Get dtype from any output tensor
        target_dtype = None
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                target_dtype = v.dtype
                break
        
        if target_dtype is not None:
            targets = {k: v.to(dtype=target_dtype) if isinstance(v, torch.Tensor) else v 
                      for k, v in targets.items()}
        
        losses = {}
        total = 0.0
        
        # BBox Loss
        if 'pred_boxes' in outputs and 'sat_bbox' in targets:
            l_bbox, l_giou = self._bbox_loss(outputs['pred_boxes'], targets['sat_bbox'])
            losses['loss_bbox'] = l_bbox
            losses['loss_giou'] = l_giou
            total += self.w['bbox'] * l_bbox + self.w['giou'] * l_giou
        
        # Mask Loss
        if 'masks' in outputs and 'masks' in targets:
            losses['loss_mask'] = self._mask_loss(outputs['masks'], targets['masks'])
            total += self.w['mask'] * losses['loss_mask']
        
        # Yaw Loss
        if 'yaw_radians' in outputs and 'yaw_radians' in targets:
            losses['loss_yaw'] = self._yaw_loss(outputs['yaw_radians'], targets['yaw_radians'])
            total += self.w['yaw'] * losses['loss_yaw']
        
        # Position Loss
        if 'position' in outputs and 'camera_position' in targets:
            losses['loss_position'] = F.mse_loss(outputs['position'], targets['camera_position'])
            total += self.w['position'] * losses['loss_position']
        
        losses['loss'] = total
        return losses
    
    def _bbox_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """BBox L1 + GIoU loss"""
        if pred.dim() == 3:
            pred = pred[:, 0, :]  # [B, N, 4] -> [B, 4]
        
        l1 = F.l1_loss(pred, target)
        
        # Clamp box coordinates to valid range [0, 1] for numerical stability
        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        target_clamped = torch.clamp(target, 0.0, 1.0)
        
        giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_clamped), box_cxcywh_to_xyxy(target_clamped))
        l_giou = (1 - torch.diag(giou)).mean()
        
        # Clamp GIoU loss to prevent extreme values (GIoU is in [-1, 1], so loss is in [0, 2])
        l_giou = torch.clamp(l_giou, 0.0, 2.0)
        
        return l1, l_giou
    
    def _mask_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """Mask BCE + Dice loss"""
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        p = torch.sigmoid(pred)
        inter = (p * target).sum(dim=(2, 3))
        union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - (2 * inter + 1) / (union + 1)
        
        return bce + dice.mean()
    
    def _yaw_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """周期性角度loss，处理[-pi, pi]边界"""
        # atan2(sin, cos) is numerically stable for any input, no need to clamp
        diff = torch.atan2(torch.sin(pred - target), torch.cos(pred - target))
        loss = (diff ** 2).mean()
        
        # Clamp loss to reasonable range (max angular error is pi, so max loss is pi^2 ≈ 9.87)
        loss = torch.clamp(loss, 0.0, 10.0)
        
        return loss
