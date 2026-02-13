"""Cross-View Localization Loss Functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    DETR-style Hungarian Matcher.
    
    Computes an assignment between predicted boxes and ground truth boxes
    using a cost matrix based on L1, GIoU, and classification costs.
    
    Args:
        cost_class: Weight for classification cost
        cost_bbox: Weight for L1 bbox cost
        cost_giou: Weight for GIoU bbox cost
    """
    
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform Hungarian matching.
        
        Args:
            outputs: dict with 'pred_boxes' [B, N, 4] and 'class_logits' [B, N, num_classes]
            targets: dict with 'sat_bbox' [B, 4] (single GT box per image)
        
        Returns:
            List of (pred_indices, gt_indices) tuples for each batch element
        """
        B, N = outputs['pred_boxes'].shape[:2]
        
        pred_boxes = outputs['pred_boxes'].flatten(0, 1)  # [B*N, 4]
        pred_logits = outputs['class_logits'].flatten(0, 1).sigmoid()  # [B*N, num_classes]
        
        # Each image has 1 GT box
        tgt_boxes = targets['sat_bbox']  # [B, 4]
        
        indices = []
        for b in range(B):
            pred_b = outputs['pred_boxes'][b]  # [N, 4]
            tgt_b = tgt_boxes[b:b+1]  # [1, 4]
            logits_b = outputs['class_logits'][b].sigmoid()  # [N, num_classes]
            
            # Classification cost: -prob of target class (class 0 for single-class)
            cost_class = -logits_b[:, 0:1]  # [N, 1]
            
            # L1 cost
            cost_bbox = torch.cdist(pred_b.float(), tgt_b.float(), p=1)  # [N, 1]
            
            # GIoU cost
            pred_xyxy = box_cxcywh_to_xyxy(pred_b)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_b)
            cost_giou = -generalized_box_iou(pred_xyxy, tgt_xyxy)  # [N, 1]
            
            # Total cost matrix
            C = (self.cost_bbox * cost_bbox + 
                 self.cost_giou * cost_giou + 
                 self.cost_class * cost_class)  # [N, 1]
            
            C = C.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64),
            ))
        
        return indices


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss for object detection classification.
    
    Helps distinguish matched (object) vs unmatched (no-object) queries.
    Focal loss down-weights easy negatives, focusing on hard examples.
    
    Args:
        inputs: [N, C] raw logits
        targets: [N, C] binary targets (0 or 1)
        alpha: balancing factor (0.25 for positive)
        gamma: focusing parameter
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


class DETRCriterion(nn.Module):
    """
    Loss function for DETR-style cross-view localization.
    
    Losses:
    1. BBox: L1 + GIoU loss for matched predictions
    2. Classification: Focal loss for object vs no-object
    3. Heatmap: KL-div + MSE for camera position
    4. Rotation: Geodesic/smooth distance on SO(3)
    5. Contrastive: MoCo cross-view loss
    """
    
    def __init__(
        self,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        weight_heatmap: float = 1.0,
        weight_rotation: float = 1.0,
        weight_contrastive: float = 0.1,
        img_size: int = 518,
        matcher_cost_class: float = 1.0,
        matcher_cost_bbox: float = 5.0,
        matcher_cost_giou: float = 2.0,
        heatmap_sigma: float = 0.05,
        heatmap_label_smooth: float = 0.01,
        weight_class: float = 2.0,
        smooth_rotation: bool = True,
    ):
        super().__init__()
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_heatmap = weight_heatmap
        self.weight_rotation = weight_rotation
        self.weight_contrastive = weight_contrastive
        self.weight_class = weight_class
        self.smooth_rotation = smooth_rotation
        self.img_size = img_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_label_smooth = heatmap_label_smooth
        
        # DETR Hungarian Matcher
        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou,
        )
    
    def forward(self, outputs, targets):
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs dict
            targets: Target dict with 'sat_bbox', 'camera_position', 'rotation_matrix'
        """
        losses = {}
        
        # ============ BBox Loss (DETR Hungarian Matching) ============
        if 'pred_boxes' in outputs and 'sat_bbox' in targets:
            # Hungarian matching to find optimal assignment
            indices = self.matcher(outputs, targets)
            
            pred_boxes = outputs['pred_boxes']  # [B, N, 4]
            target_boxes = targets['sat_bbox']  # [B, 4]
            
            B = pred_boxes.shape[0]
            
            # Gather matched predictions
            src_idx = torch.cat([src for (src, _) in indices])  # matched pred indices
            tgt_idx = torch.cat([tgt for (_, tgt) in indices])  # matched GT indices
            
            # Build matched boxes
            batch_idx = torch.cat([
                torch.full_like(src, i) for i, (src, _) in enumerate(indices)
            ])
            
            src_boxes = pred_boxes[batch_idx, src_idx]  # [num_matched, 4]
            tgt_boxes = target_boxes[batch_idx]  # [num_matched, 4]
            
            num_boxes = max(src_boxes.shape[0], 1)
            
            # L1 loss on matched pairs
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none').sum() / num_boxes
            
            # GIoU loss on matched pairs
            src_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            giou_matrix = generalized_box_iou(src_xyxy, tgt_xyxy)
            loss_giou = (1 - torch.diag(giou_matrix)).sum() / num_boxes
            
            losses['loss_bbox'] = loss_bbox
            losses['loss_giou'] = loss_giou
            
            # BBox IoU metric (for monitoring, not in loss)
            with torch.no_grad():
                losses['bbox_iou'] = torch.diag(giou_matrix).clamp(min=0).mean().detach()
            
            # Classification loss (focal loss: matched=object, unmatched=no-object)
            # Essential for DETR: tells unmatched queries to predict "background"
            if 'class_logits' in outputs and self.weight_class > 0:
                pred_logits = outputs['class_logits']  # [B, N, num_classes]
                target_classes = torch.zeros_like(pred_logits)
                for b_idx, (src, _) in enumerate(indices):
                    target_classes[b_idx, src, :] = 1.0
                
                loss_class = sigmoid_focal_loss(
                    pred_logits.flatten(0, 1),  # [B*N, C]
                    target_classes.flatten(0, 1),  # [B*N, C]
                    alpha=0.25,
                    gamma=2.0,
                ) / max(num_boxes, 1)
                losses['loss_class'] = loss_class
        
        # ============ Heatmap Loss ============
        if 'heatmap' in outputs and 'camera_position' in targets:
            heatmap = outputs['heatmap']  # [B, H, W] probability
            target_pos = targets['camera_position']  # [B, 2] normalized [0, 1]
            
            B, H, W = heatmap.shape
            
            # Create target heatmap (Gaussian around target position)
            y_coords = torch.linspace(0, 1, H, device=heatmap.device)
            x_coords = torch.linspace(0, 1, W, device=heatmap.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # FIX: Use configurable sigma (default 0.05 instead of 0.02)
            # sigma=0.05 covers ~77 pixels (3σ) on 518×518, much more stable for KL
            sigma = self.heatmap_sigma
            eps_smooth = self.heatmap_label_smooth
            
            # Vectorized target heatmap construction (no for-loop)
            tx = target_pos[:, 0].view(B, 1, 1)  # [B, 1, 1]
            ty = target_pos[:, 1].view(B, 1, 1)  # [B, 1, 1]
            dist_sq = (xx.unsqueeze(0) - tx) ** 2 + (yy.unsqueeze(0) - ty) ** 2  # [B, H, W]
            target_heatmap = torch.exp(-dist_sq / (2 * sigma ** 2))
            target_heatmap = target_heatmap / (target_heatmap.sum(dim=(-2, -1), keepdim=True) + 1e-8)
            
            # Label smoothing: mix target with uniform distribution
            # This prevents KL from exploding when pred is slightly misaligned
            if eps_smooth > 0:
                uniform = 1.0 / (H * W)
                target_heatmap = (1 - eps_smooth) * target_heatmap + eps_smooth * uniform
            
            # KL divergence loss (primary)
            heatmap_log = torch.log(heatmap + 1e-8)
            loss_kl = F.kl_div(heatmap_log, target_heatmap, reduction='batchmean')
            
            # MSE auxiliary loss on position (more stable gradient signal)
            pred_pos = outputs['position']  # [B, 2]
            loss_pos_mse = F.mse_loss(pred_pos, target_pos)
            
            # Combined heatmap loss: KL for distribution shape + MSE for position accuracy
            loss_heatmap = loss_kl + 10.0 * loss_pos_mse
            losses['loss_heatmap'] = loss_heatmap
            
            # Position error (for logging)
            pos_error = (pred_pos - target_pos).norm(dim=-1).mean()
            losses['pos_error'] = pos_error
        
        # ============ Rotation Loss (Geodesic Distance on SO(3)) ============
        if 'rotation_matrix' in outputs and 'rotation_matrix' in targets:
            pred_R = outputs['rotation_matrix']    # [B, 3, 3]
            target_R = targets['rotation_matrix']  # [B, 3, 3]
            
            loss_rotation, rotation_error_deg = self._geodesic_loss(pred_R, target_R, smooth=self.smooth_rotation)
            losses['loss_rotation'] = loss_rotation
            losses['rotation_error_deg'] = rotation_error_deg  # for logging
        
        # ============ Contrastive Loss (computed in model, passed through outputs) ============
        if 'contrastive_loss' in outputs:
            losses['loss_contrastive'] = outputs['contrastive_loss']
        
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
        if 'loss_contrastive' in losses:
            total_loss = total_loss + self.weight_contrastive * losses['loss_contrastive']
        if 'loss_class' in losses:
            total_loss = total_loss + self.weight_class * losses['loss_class']
        
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
        This avoids gradient explosion of arccos near 0° and 180°,
        where d/dx arccos(x) = -1/sqrt(1-x²) → ±∞.
        
        Args:
            pred_R: [B, 3, 3] predicted rotation
            target_R: [B, 3, 3] target rotation
            smooth: If True, use 1-cos(angle) loss (bounded gradient)
        
        Returns:
            loss: scalar loss value
            error_deg: scalar, mean angle error in degrees (for logging)
        """
        # R_diff = R_pred^T @ R_target
        R_diff = pred_R.transpose(-2, -1) @ target_R  # [B, 3, 3]
        
        # trace of R_diff
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]  # [B]
        
        # clamp for numerical stability: (trace - 1) / 2 in [-1, 1]
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        
        if smooth:
            # Smooth loss: 1 - cos(angle) ∈ [0, 2]
            # Gradient is sin(angle), bounded and smooth everywhere
            # Avoids arccos singularity at 0° and 180°
            loss = (1.0 - cos_angle).mean()
        else:
            angle = torch.acos(cos_angle)
            loss = angle.mean()
        
        # Always compute actual angle in degrees for logging
        angle = torch.acos(cos_angle)
        error_deg = torch.rad2deg(angle).mean().detach()
        
        return loss, error_deg
