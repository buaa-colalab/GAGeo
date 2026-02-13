"""Cross-View Localization Loss Functions V2

Changes from V1:
1. Added Mask Loss (BCE + Dice, SAM-style)
2. Added Deep Supervision support (intermediate layer predictions)
3. Heatmap Loss simplified to position MSE only (no distribution)
4. Fixed classification focal loss normalization
5. Added intermediate loss weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    DETR-style Hungarian Matcher.
    Computes assignment between predicted and ground truth boxes.
    """
    
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_boxes' [B, N, 4] and 'class_logits' [B, N, num_classes]
            targets: dict with 'sat_bbox' [B, 4]
        Returns:
            List of (pred_indices, gt_indices) tuples
        """
        from scipy.optimize import linear_sum_assignment
        
        B, N = outputs['pred_boxes'].shape[:2]
        tgt_boxes = targets['sat_bbox']  # [B, 4]
        
        indices = []
        for b in range(B):
            pred_b = outputs['pred_boxes'][b]  # [N, 4]
            tgt_b = tgt_boxes[b:b+1]  # [1, 4]
            logits_b = outputs['class_logits'][b].sigmoid()  # [N, num_classes]
            
            cost_class = -logits_b[:, 0:1]
            cost_bbox = torch.cdist(pred_b.float(), tgt_b.float(), p=1)
            
            pred_xyxy = box_cxcywh_to_xyxy(pred_b)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_b)
            cost_giou = -generalized_box_iou(pred_xyxy, tgt_xyxy)
            
            C = (self.cost_bbox * cost_bbox +
                 self.cost_giou * cost_giou +
                 self.cost_class * cost_class)
            
            C = C.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64),
            ))
        
        return indices


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice loss for mask prediction.
    
    Args:
        inputs: [B, H, W] sigmoid probabilities
        targets: [B, H, W] binary ground truth
        smooth: smoothing factor
    
    Returns:
        scalar dice loss
    """
    inputs = inputs.flatten(1)   # [B, H*W]
    targets = targets.flatten(1)  # [B, H*W]
    
    intersection = (inputs * targets).sum(dim=1)
    union = inputs.sum(dim=1) + targets.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def mask_bce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for mask prediction.
    
    Args:
        inputs: [B, H, W] raw logits (before sigmoid)
        targets: [B, H, W] binary ground truth
    
    Returns:
        scalar BCE loss
    """
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')


class DETRCriterionV2(nn.Module):
    """
    Loss function for DETR-style cross-view localization V2.
    
    Losses:
    1. BBox: L1 + GIoU loss for matched predictions
    2. Classification: Focal loss for object vs no-object
    3. Mask: BCE + Dice loss (SAM-style)
    4. Heatmap: Position MSE only (simplified)
    5. Rotation: Geodesic/smooth distance on SO(3)
    6. Contrastive: MoCo cross-view loss
    7. Deep Supervision: weighted sum of intermediate losses
    """
    
    def __init__(
        self,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        weight_mask_bce: float = 2.0,
        weight_mask_dice: float = 5.0,
        weight_heatmap: float = 1.0,
        weight_rotation: float = 1.0,
        weight_contrastive: float = 0.1,
        weight_class: float = 2.0,
        img_size: int = 518,
        matcher_cost_class: float = 1.0,
        matcher_cost_bbox: float = 5.0,
        matcher_cost_giou: float = 2.0,
        smooth_rotation: bool = True,
        supervision_layers: List[int] = None,
        supervision_weights: List[float] = None,
    ):
        super().__init__()
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_mask_bce = weight_mask_bce
        self.weight_mask_dice = weight_mask_dice
        self.weight_heatmap = weight_heatmap
        self.weight_rotation = weight_rotation
        self.weight_contrastive = weight_contrastive
        self.weight_class = weight_class
        self.smooth_rotation = smooth_rotation
        self.img_size = img_size
        
        self.supervision_layers = supervision_layers or [3, 10, 16]
        self.supervision_weights = supervision_weights or [0.1, 0.3, 0.6]
        
        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou,
        )
    
    def _compute_bbox_loss(self, outputs, targets):
        """Compute BBox + Classification loss."""
        losses = {}
        
        if 'pred_boxes' not in outputs or 'sat_bbox' not in targets:
            return losses
        
        indices = self.matcher(outputs, targets)
        pred_boxes = outputs['pred_boxes']
        target_boxes = targets['sat_bbox']
        B = pred_boxes.shape[0]
        
        src_idx = torch.cat([src for (src, _) in indices])
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        
        src_boxes = pred_boxes[batch_idx, src_idx]
        tgt_boxes = target_boxes[batch_idx]
        num_boxes = max(src_boxes.shape[0], 1)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none').sum() / num_boxes
        
        # GIoU loss
        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou_matrix = generalized_box_iou(src_xyxy, tgt_xyxy)
        loss_giou = (1 - torch.diag(giou_matrix)).sum() / num_boxes
        
        losses['loss_bbox'] = loss_bbox
        losses['loss_giou'] = loss_giou
        
        # IoU metric
        with torch.no_grad():
            losses['bbox_iou'] = torch.diag(giou_matrix).clamp(min=0).mean().detach()
        
        # Classification loss
        if 'class_logits' in outputs and self.weight_class > 0:
            pred_logits = outputs['class_logits']
            target_classes = torch.zeros_like(pred_logits)
            for b_idx, (src, _) in enumerate(indices):
                target_classes[b_idx, src, :] = 1.0
            
            loss_class = sigmoid_focal_loss(
                pred_logits.flatten(0, 1),
                target_classes.flatten(0, 1),
                alpha=0.25, gamma=2.0,
            ) / max(num_boxes, 1)
            losses['loss_class'] = loss_class
        
        return losses
    
    def _compute_mask_loss(self, outputs, targets):
        """Compute Mask BCE + Dice loss."""
        losses = {}
        
        if 'mask_logits' not in outputs or 'sat_mask' not in targets:
            return losses
        
        mask_logits = outputs['mask_logits']  # [B, num_masks, H, W]
        # Use the first mask prediction (single target)
        mask_logits = mask_logits[:, 0]  # [B, H, W]
        
        target_mask = targets['sat_mask']  # [B, 1, H_orig, W_orig]
        # Resize target mask to match prediction
        if target_mask.shape[-2:] != mask_logits.shape[-2:]:
            target_mask = F.interpolate(
                target_mask.float(), size=mask_logits.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        target_mask = target_mask.squeeze(1)  # [B, H, W]
        target_mask = (target_mask > 0.5).float()
        
        # BCE loss
        loss_mask_bce = mask_bce_loss(mask_logits, target_mask)
        
        # Dice loss
        mask_prob = mask_logits.sigmoid()
        loss_mask_dice = dice_loss(mask_prob, target_mask)
        
        losses['loss_mask_bce'] = loss_mask_bce
        losses['loss_mask_dice'] = loss_mask_dice
        
        # Mask IoU metric
        with torch.no_grad():
            pred_binary = (mask_prob > 0.5).float()
            intersection = (pred_binary * target_mask).sum(dim=(-2, -1))
            union = pred_binary.sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1)) - intersection
            mask_iou = (intersection / (union + 1e-8)).mean()
            losses['mask_iou'] = mask_iou.detach()
        
        return losses
    
    def _compute_heatmap_loss(self, outputs, targets):
        """Compute Heatmap loss (position MSE only)."""
        losses = {}
        
        if 'position' not in outputs or 'camera_position' not in targets:
            return losses
        
        pred_pos = outputs['position']          # [B, 2]
        target_pos = targets['camera_position']  # [B, 2]
        
        # Simple MSE loss on position
        loss_heatmap = F.mse_loss(pred_pos, target_pos)
        losses['loss_heatmap'] = loss_heatmap
        
        # Position error metric (normalized distance)
        with torch.no_grad():
            pos_error = (pred_pos - target_pos).norm(dim=-1).mean()
            losses['pos_error'] = pos_error.detach()
        
        return losses
    
    def _compute_rotation_loss(self, outputs, targets):
        """Compute Rotation Geodesic loss on SO(3)."""
        losses = {}
        
        if 'rotation_matrix' not in outputs or 'rotation_matrix' not in targets:
            return losses
        
        pred_R = outputs['rotation_matrix']
        target_R = targets['rotation_matrix']
        
        loss_rotation, rotation_error_deg = self._geodesic_loss(
            pred_R, target_R, smooth=self.smooth_rotation
        )
        losses['loss_rotation'] = loss_rotation
        losses['rotation_error_deg'] = rotation_error_deg
        
        return losses
    
    def forward(self, outputs, targets):
        """
        Compute all losses including deep supervision.
        
        Args:
            outputs: Model outputs dict (includes 'intermediate_preds')
            targets: Target dict
        
        Returns:
            losses: Dict with all loss components and total loss
        """
        losses = {}
        
        # ============ Final prediction losses ============
        losses.update(self._compute_bbox_loss(outputs, targets))
        losses.update(self._compute_mask_loss(outputs, targets))
        losses.update(self._compute_heatmap_loss(outputs, targets))
        losses.update(self._compute_rotation_loss(outputs, targets))
        
        if 'contrastive_loss' in outputs:
            losses['loss_contrastive'] = outputs['contrastive_loss']
        
        # ============ Deep Supervision losses ============
        if 'intermediate_preds' in outputs:
            for layer_idx, weight in zip(self.supervision_layers, self.supervision_weights):
                if layer_idx in outputs['intermediate_preds']:
                    inter_outputs = outputs['intermediate_preds'][layer_idx]
                    
                    # Intermediate BBox loss
                    inter_bbox_losses = self._compute_bbox_loss(inter_outputs, targets)
                    for k, v in inter_bbox_losses.items():
                        if k.startswith('loss_'):
                            losses[f'inter_{layer_idx}_{k}'] = weight * v
                    
                    # Intermediate Mask loss
                    inter_mask_losses = self._compute_mask_loss(inter_outputs, targets)
                    for k, v in inter_mask_losses.items():
                        if k.startswith('loss_'):
                            losses[f'inter_{layer_idx}_{k}'] = weight * v
        
        # ============ Total Loss ============
        total_loss = 0.0
        
        # Final prediction losses
        if 'loss_bbox' in losses:
            total_loss = total_loss + self.weight_bbox * losses['loss_bbox']
        if 'loss_giou' in losses:
            total_loss = total_loss + self.weight_giou * losses['loss_giou']
        if 'loss_mask_bce' in losses:
            total_loss = total_loss + self.weight_mask_bce * losses['loss_mask_bce']
        if 'loss_mask_dice' in losses:
            total_loss = total_loss + self.weight_mask_dice * losses['loss_mask_dice']
        if 'loss_heatmap' in losses:
            total_loss = total_loss + self.weight_heatmap * losses['loss_heatmap']
        if 'loss_rotation' in losses:
            total_loss = total_loss + self.weight_rotation * losses['loss_rotation']
        if 'loss_contrastive' in losses:
            total_loss = total_loss + self.weight_contrastive * losses['loss_contrastive']
        if 'loss_class' in losses:
            total_loss = total_loss + self.weight_class * losses['loss_class']
        
        # Deep supervision losses (already weighted by layer weight)
        for k, v in losses.items():
            if k.startswith('inter_') and k.split('_', 2)[-1].startswith('loss_'):
                loss_type = k.split('_', 2)[-1]  # e.g., 'loss_bbox', 'loss_mask_bce'
                if loss_type == 'loss_bbox':
                    total_loss = total_loss + self.weight_bbox * v
                elif loss_type == 'loss_giou':
                    total_loss = total_loss + self.weight_giou * v
                elif loss_type == 'loss_mask_bce':
                    total_loss = total_loss + self.weight_mask_bce * v
                elif loss_type == 'loss_mask_dice':
                    total_loss = total_loss + self.weight_mask_dice * v
                elif loss_type == 'loss_class':
                    total_loss = total_loss + self.weight_class * v
        
        losses['loss'] = total_loss
        
        return losses
    
    @staticmethod
    def _geodesic_loss(
        pred_R: torch.Tensor, target_R: torch.Tensor, smooth: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Geodesic distance between two rotation matrices on SO(3).
        
        d(R1, R2) = arccos( (tr(R1^T @ R2) - 1) / 2 )
        
        When smooth=True, uses 1-cos(angle) instead of raw angle.
        """
        R_diff = pred_R.transpose(-2, -1) @ target_R
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        
        if smooth:
            loss = (1.0 - cos_angle).mean()
        else:
            angle = torch.acos(cos_angle)
            loss = angle.mean()
        
        angle = torch.acos(cos_angle)
        error_deg = torch.rad2deg(angle).mean().detach()
        
        return loss, error_deg
