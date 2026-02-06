"""Cross-View Localization Loss Functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class DETRCriterion(nn.Module):
    """
    Loss function for DETR-style cross-view localization.
    
    Losses:
    1. BBox: L1 + GIoU loss for matched predictions
    2. Heatmap: Cross-entropy loss for camera position
    3. Rotation: Geodesic distance on SO(3) for relative pose
    """
    
    def __init__(
        self,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        weight_heatmap: float = 1.0,
        weight_rotation: float = 1.0,
        img_size: int = 518,
    ):
        super().__init__()
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_heatmap = weight_heatmap
        self.weight_rotation = weight_rotation
        self.img_size = img_size
    
    def forward(self, outputs, targets):
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs dict
            targets: Target dict with 'sat_bbox', 'camera_position', 'rotation_matrix'
        """
        losses = {}
        
        # ============ BBox Loss ============
        if 'pred_boxes' in outputs and 'sat_bbox' in targets:
            pred_boxes = outputs['pred_boxes']  # [B, N, 4]
            target_boxes = targets['sat_bbox']  # [B, 4]
            bbox_scores = outputs['bbox_scores']  # [B, N]
            
            B = pred_boxes.shape[0]
            loss_bbox = 0.0
            loss_giou = 0.0
            
            for b in range(B):
                # Find best prediction by score
                best_idx = bbox_scores[b].argmax()
                pred_box = pred_boxes[b, best_idx]  # [4]
                target_box = target_boxes[b]  # [4]
                
                # L1 loss
                loss_bbox = loss_bbox + F.l1_loss(pred_box, target_box)
                
                # GIoU loss
                pred_xyxy = box_cxcywh_to_xyxy(pred_box.unsqueeze(0))
                tgt_xyxy = box_cxcywh_to_xyxy(target_box.unsqueeze(0))
                giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
                loss_giou = loss_giou + (1 - giou[0, 0])
            
            losses['loss_bbox'] = loss_bbox / B
            losses['loss_giou'] = loss_giou / B
        
        # ============ Heatmap Loss ============
        if 'heatmap' in outputs and 'camera_position' in targets:
            heatmap = outputs['heatmap']  # [B, H, W] probability
            target_pos = targets['camera_position']  # [B, 2] normalized [0, 1]
            
            B, H, W = heatmap.shape
            
            # Create target heatmap (Gaussian around target position)
            y_coords = torch.linspace(0, 1, H, device=heatmap.device)
            x_coords = torch.linspace(0, 1, W, device=heatmap.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            sigma = 0.02  # Gaussian sigma
            target_heatmaps = []
            for b in range(B):
                tx, ty = target_pos[b, 0], target_pos[b, 1]
                dist_sq = (xx - tx) ** 2 + (yy - ty) ** 2
                target_hm = torch.exp(-dist_sq / (2 * sigma ** 2))
                target_hm = target_hm / (target_hm.sum() + 1e-8)  # Normalize
                target_heatmaps.append(target_hm)
            
            target_heatmap = torch.stack(target_heatmaps, dim=0)  # [B, H, W]
            
            # KL divergence loss
            heatmap_log = torch.log(heatmap + 1e-8)
            loss_heatmap = F.kl_div(heatmap_log, target_heatmap, reduction='batchmean')
            losses['loss_heatmap'] = loss_heatmap
            
            # Position error (for logging)
            pred_pos = outputs['position']  # [B, 2]
            pos_error = (pred_pos - target_pos).norm(dim=-1).mean()
            losses['pos_error'] = pos_error
        
        # ============ Rotation Loss (Geodesic Distance on SO(3)) ============
        if 'rotation_matrix' in outputs and 'rotation_matrix' in targets:
            pred_R = outputs['rotation_matrix']    # [B, 3, 3]
            target_R = targets['rotation_matrix']  # [B, 3, 3]
            
            loss_rotation, rotation_error_deg = self._geodesic_loss(pred_R, target_R)
            losses['loss_rotation'] = loss_rotation
            losses['rotation_error_deg'] = rotation_error_deg  # for logging
        
        # ============ Total Loss ============
        total_loss = 0.0
        if 'loss_bbox' in losses:
            total_loss = total_loss + self.weight_bbox * losses['loss_bbox']
        if 'loss_giou' in losses:
            total_loss = total_loss + self.weight_giou * losses['loss_giou']
        if 'loss_heatmap' in losses:
            total_loss = total_loss + self.weight_heatmap * losses['loss_heatmap']
        if 'loss_rotation' in losses:
            total_loss = total_loss + self.weight_rotation * losses['loss_rotation']
        
        losses['loss'] = total_loss
        
        return losses
    
    @staticmethod
    def _geodesic_loss(
        pred_R: torch.Tensor, target_R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Geodesic distance between two rotation matrices on SO(3).
        
        d(R1, R2) = arccos( (tr(R1^T @ R2) - 1) / 2 )
        
        Args:
            pred_R: [B, 3, 3] predicted rotation
            target_R: [B, 3, 3] target rotation
        
        Returns:
            loss: scalar, mean geodesic angle (radians)
            error_deg: scalar, mean angle error in degrees (for logging)
        """
        # R_diff = R_pred^T @ R_target
        R_diff = pred_R.transpose(-2, -1) @ target_R  # [B, 3, 3]
        
        # trace of R_diff
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]  # [B]
        
        # clamp for numerical stability: (trace - 1) / 2 in [-1, 1]
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        
        angle = torch.acos(cos_angle)  # [B], geodesic angle in radians
        
        loss = angle.mean()
        error_deg = torch.rad2deg(angle).mean().detach()
        
        return loss, error_deg
