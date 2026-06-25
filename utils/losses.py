"""Loss functions for GAGeo cross-view localization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


def _sanitize_pred_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Keep predicted cxcywh boxes finite and inside the normalized image range."""
    return torch.nan_to_num(boxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _sanitize_target_boxes(boxes: torch.Tensor, name: str = "sat_bbox") -> torch.Tensor:
    """Validate target cxcywh boxes instead of silently hiding dataset bugs."""
    boxes = torch.nan_to_num(boxes, nan=0.5, posinf=1.0, neginf=0.0)
    min_val = float(boxes.min().detach().cpu())
    max_val = float(boxes.max().detach().cpu())
    min_wh = float(boxes[..., 2:].min().detach().cpu())
    if min_val < -1e-4 or max_val > 1.0 + 1e-4 or min_wh <= 0.0:
        raise ValueError(
            f"Invalid {name}: expected normalized cxcywh boxes in [0, 1] with positive w/h; "
            f"got min={min_val:.6f}, max={max_val:.6f}, min_wh={min_wh:.6f}. "
            "Check dataset crop/resize bbox clipping."
        )
    return boxes.clamp(0.0, 1.0)


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
            pred_b = _sanitize_pred_boxes(pred_b)
            tgt_b = _sanitize_target_boxes(tgt_b)
            logits_b = torch.nan_to_num(outputs['class_logits'][b], nan=0.0, posinf=30.0, neginf=-30.0).sigmoid()

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


class DETRCriterion(nn.Module):
    """
    Loss function for DETR-style cross-view localization.

    Losses:
    1. BBox: L1 + GIoU loss for matched predictions
    2. Classification: Focal loss for object vs no-object
    3. Mask: BCE + Dice loss (SAM-style)
    4. Heatmap: CornerNet-style pixel-wise focal loss
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
        use_deep_supervision: bool = True,
        use_contrastive_loss: bool = True,
        use_rot_pos_supervision: bool = True,
        use_heatmap_loss: bool = True,
        heatmap_sigma: float = 0.05,
        heatmap_focal_alpha: float = 2.0,
        heatmap_focal_beta: float = 4.0,
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
        self.use_deep_supervision = use_deep_supervision
        self.use_contrastive_loss = use_contrastive_loss
        self.use_rot_pos_supervision = use_rot_pos_supervision
        self.use_heatmap_loss = use_heatmap_loss
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_focal_alpha = heatmap_focal_alpha
        self.heatmap_focal_beta = heatmap_focal_beta

        self.supervision_layers = [4, 11, 17] if supervision_layers is None else list(supervision_layers)
        self.supervision_weights = [0.1, 0.3, 0.6] if supervision_weights is None else list(supervision_weights)
        self.extra_supervision_layers = set(sorted(self.supervision_layers)[:-1])

        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou,
        )

    def _compute_bbox_loss(self, outputs, targets, indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """Compute BBox + Classification loss."""
        losses = {}

        if 'pred_boxes' not in outputs or 'sat_bbox' not in targets:
            return losses

        if indices is None:
            indices = self.matcher(outputs, targets)
        pred_boxes = _sanitize_pred_boxes(outputs['pred_boxes'])
        target_boxes = _sanitize_target_boxes(targets['sat_bbox'])
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
            iou_matrix, _ = box_iou(src_xyxy, tgt_xyxy)
            losses['bbox_iou'] = torch.diag(iou_matrix).mean().detach()
            losses['bbox_giou'] = torch.diag(giou_matrix).mean().detach()
            losses['bbox_center_l1'] = F.l1_loss(src_boxes[:, :2], tgt_boxes[:, :2], reduction='none').sum(-1).mean().detach()
            losses['bbox_size_l1'] = F.l1_loss(src_boxes[:, 2:], tgt_boxes[:, 2:], reduction='none').sum(-1).mean().detach()
            losses['bbox_pred_w_mean'] = src_boxes[:, 2].mean().detach()
            losses['bbox_pred_h_mean'] = src_boxes[:, 3].mean().detach()
            losses['bbox_tgt_w_mean'] = tgt_boxes[:, 2].mean().detach()
            losses['bbox_tgt_h_mean'] = tgt_boxes[:, 3].mean().detach()

        # Classification loss
        if 'class_logits' in outputs and self.weight_class > 0:
            pred_logits = torch.nan_to_num(outputs['class_logits'], nan=0.0, posinf=30.0, neginf=-30.0)
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

    def _compute_mask_loss(
        self,
        outputs,
        targets,
        indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """Compute Mask BCE + Dice loss."""
        losses = {}

        if 'mask_logits' not in outputs or 'sat_mask' not in targets:
            return losses

        mask_logits = outputs['mask_logits']  # [B, Q, H, W]
        if mask_logits.dim() != 4:
            raise ValueError(f"Expected mask_logits [B, Q, H, W], got {tuple(mask_logits.shape)}")

        # Single-target setting: select one mask per image.
        # If Hungarian assignment is available, use matched bbox query index.
        if indices is not None and mask_logits.shape[1] > 1:
            selected_masks = []
            for b, (src_idx, _) in enumerate(indices):
                match_idx = int(src_idx[0].item()) if len(src_idx) > 0 else 0
                selected_masks.append(mask_logits[b, match_idx])
            mask_logits = torch.stack(selected_masks, dim=0)  # [B, H, W]
        else:
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
        """Compute Heatmap loss using CornerNet-style pixel-wise focal loss."""
        losses = {}

        if 'heatmap_logits' not in outputs or 'camera_position' not in targets:
            return losses

        # Compute in FP32 for numerical stability under bf16 mixed precision.
        heatmap_logits = outputs['heatmap_logits'].float()  # [B, H, W]
        if heatmap_logits.dim() != 3:
            raise ValueError(f"Expected heatmap_logits [B, H, W], got {tuple(heatmap_logits.shape)}")

        target_pos = targets['camera_position'].to(heatmap_logits.dtype)  # [B, 2], normalized [0, 1]
        B, H, W = heatmap_logits.shape

        y_coords = torch.linspace(0, 1, H, device=heatmap_logits.device, dtype=heatmap_logits.dtype)
        x_coords = torch.linspace(0, 1, W, device=heatmap_logits.device, dtype=heatmap_logits.dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_x = grid_x.unsqueeze(0)  # [1, H, W]
        grid_y = grid_y.unsqueeze(0)  # [1, H, W]

        target_x = target_pos[:, 0].view(B, 1, 1)
        target_y = target_pos[:, 1].view(B, 1, 1)
        sigma = max(float(self.heatmap_sigma), 1e-6)
        dist_sq = (grid_x - target_x) ** 2 + (grid_y - target_y) ** 2
        target_heatmap = torch.exp(-dist_sq / (2.0 * sigma * sigma)).clamp_(0.0, 1.0)

        # Ensure at least one positive pixel per sample (CornerNet-style center).
        center_x = torch.round(target_pos[:, 0] * (W - 1)).long().clamp_(0, W - 1)
        center_y = torch.round(target_pos[:, 1] * (H - 1)).long().clamp_(0, H - 1)
        target_heatmap[torch.arange(B, device=heatmap_logits.device), center_y, center_x] = 1.0

        pred = torch.sigmoid(heatmap_logits)
        pos_inds = (target_heatmap >= 1.0).float()
        neg_inds = (target_heatmap < 1.0).float()
        neg_weights = torch.pow(1.0 - target_heatmap, self.heatmap_focal_beta)

        # Logits-stable focal terms:
        # log(sigmoid(x)) = -softplus(-x), log(1-sigmoid(x)) = -softplus(x)
        log_p = -F.softplus(-heatmap_logits)
        log_not_p = -F.softplus(heatmap_logits)
        pos_loss = log_p * torch.pow(1.0 - pred, self.heatmap_focal_alpha) * pos_inds
        neg_loss = log_not_p * torch.pow(pred, self.heatmap_focal_alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum().clamp_min(1.0)
        if num_pos > 0:
            focal_loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        else:
            focal_loss = -neg_loss.sum()
        # Log-space dampening: compress large initial loss while preserving
        # gradients (gradient = 1/(1+loss), always non-zero).
        # This avoids destroying pretrained weights at the start, and keeps
        # meaningful gradients when the loss is small in later training.
        loss_heatmap = torch.log1p(focal_loss)
        losses['loss_heatmap'] = loss_heatmap

        with torch.no_grad():
            center_prob = pred[torch.arange(B, device=pred.device), center_y, center_x].mean()
            losses['heatmap_center_prob'] = center_prob.detach()

        # Position error metric (normalized distance)
        with torch.no_grad():
            if 'position' in outputs:
                pred_pos = outputs['position']
            else:
                flat_idx = heatmap_logits.flatten(1).argmax(dim=1)
                pred_y = (flat_idx // W).to(heatmap_logits.dtype) / max(H - 1, 1)
                pred_x = (flat_idx % W).to(heatmap_logits.dtype) / max(W - 1, 1)
                pred_pos = torch.stack([pred_x, pred_y], dim=1)
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
        final_indices = None
        if (
            ('pred_boxes' in outputs and 'sat_bbox' in targets)
            and ('class_logits' in outputs)
        ):
            final_indices = self.matcher(outputs, targets)

        # ============ Final prediction losses ============
        losses.update(self._compute_bbox_loss(outputs, targets, indices=final_indices))
        losses.update(self._compute_mask_loss(outputs, targets, indices=final_indices))
        if self.use_rot_pos_supervision:
            if self.use_heatmap_loss:
                losses.update(self._compute_heatmap_loss(outputs, targets))
            losses.update(self._compute_rotation_loss(outputs, targets))

        if self.use_contrastive_loss and 'contrastive_loss' in outputs:
            losses['loss_contrastive'] = outputs['contrastive_loss']

        # ============ Deep Supervision losses ============
        if self.use_deep_supervision and 'intermediate_preds' in outputs:
            for layer_idx, weight in zip(self.supervision_layers, self.supervision_weights):
                if layer_idx in outputs['intermediate_preds']:
                    inter_outputs = outputs['intermediate_preds'][layer_idx]
                    inter_indices = None
                    if (
                        ('pred_boxes' in inter_outputs and 'sat_bbox' in targets)
                        and ('class_logits' in inter_outputs)
                    ):
                        inter_indices = self.matcher(inter_outputs, targets)

                    # Intermediate BBox loss
                    inter_bbox_losses = self._compute_bbox_loss(inter_outputs, targets, indices=inter_indices)
                    for k, v in inter_bbox_losses.items():
                        if k.startswith('loss_'):
                            losses[f'inter_{layer_idx}_{k}'] = weight * v

                    # Intermediate Mask loss
                    inter_mask_losses = self._compute_mask_loss(inter_outputs, targets, indices=inter_indices)
                    for k, v in inter_mask_losses.items():
                        if k.startswith('loss_'):
                            losses[f'inter_{layer_idx}_{k}'] = weight * v

                    # Intermediate Heatmap + Rotation loss (only early/mid stages)
                    if self.use_rot_pos_supervision and layer_idx in self.extra_supervision_layers:
                        if self.use_heatmap_loss:
                            inter_heat_losses = self._compute_heatmap_loss(inter_outputs, targets)
                            for k, v in inter_heat_losses.items():
                                if k.startswith('loss_'):
                                    losses[f'inter_{layer_idx}_{k}'] = weight * v
                        inter_rot_losses = self._compute_rotation_loss(inter_outputs, targets)
                        for k, v in inter_rot_losses.items():
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
                elif loss_type == 'loss_heatmap':
                    total_loss = total_loss + self.weight_heatmap * v
                elif loss_type == 'loss_rotation':
                    total_loss = total_loss + self.weight_rotation * v

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
