"""Cross-View Localization Loss Utilities

Heatmap generation utilities for camera position supervision.
"""

import torch
from typing import Tuple

class DETRCriterion(nn.Module):
    """
    Loss function for DETR-style cross-view localization.
    
    Losses:
    1. BBox: L1 + GIoU loss for matched predictions
    2. Heatmap: Cross-entropy loss for camera position
    3. Yaw: Angular loss for camera orientation
    """
    
    def __init__(
        self,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        weight_heatmap: float = 1.0,
        weight_yaw: float = 1.0,
        img_size: int = 518,
    ):
        super().__init__()
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_heatmap = weight_heatmap
        self.weight_yaw = weight_yaw
        self.img_size = img_size
    
    def forward(self, outputs, targets):
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs dict
            targets: Target dict with 'sat_bbox', 'camera_position', 'yaw_radians'
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
        
        # ============ Yaw Loss ============
        if 'yaw_radians' in outputs and 'yaw_radians' in targets:
            pred_yaw = outputs['yaw_radians']  # [B]
            target_yaw = targets['yaw_radians']  # [B]
            
            # Angular difference (handle wraparound)
            yaw_diff = pred_yaw - target_yaw
            yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
            loss_yaw = yaw_diff.abs().mean()
            losses['loss_yaw'] = loss_yaw
        
        # ============ Total Loss ============
        total_loss = 0.0
        if 'loss_bbox' in losses:
            total_loss = total_loss + self.weight_bbox * losses['loss_bbox']
        if 'loss_giou' in losses:
            total_loss = total_loss + self.weight_giou * losses['loss_giou']
        if 'loss_heatmap' in losses:
            total_loss = total_loss + self.weight_heatmap * losses['loss_heatmap']
        if 'loss_yaw' in losses:
            total_loss = total_loss + self.weight_yaw * losses['loss_yaw']
        
        losses['loss'] = total_loss
        
        return losses

def generate_gaussian_heatmap(
    center: torch.Tensor,
    size: Tuple[int, int],
    device: torch.device,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    Generate a 2D Gaussian heatmap centered at the given position.
    
    Args:
        center: [B, 2] Normalized (x, y) coordinates in [0, 1]
        size: (H, W) Heatmap size
        device: Device to create tensor on
        sigma: Gaussian standard deviation in pixels
    
    Returns:
        heatmap: [B, H, W] Gaussian heatmap with peak at center
    """
    B = center.shape[0]
    H, W = size
    
    # Convert normalized coordinates to pixel coordinates
    center_x = center[:, 0] * (W - 1)  # [B]
    center_y = center[:, 1] * (H - 1)  # [B]
    
    # Create coordinate grids
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Expand for batch
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
    # Compute Gaussian
    center_x = center_x.view(B, 1, 1)
    center_y = center_y.view(B, 1, 1)
    
    gaussian = torch.exp(
        -((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * sigma ** 2)
    )
    
    return gaussian
