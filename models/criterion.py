# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR-style SetCriterion for cross-view localization.
Computes losses using Hungarian matching between predictions and targets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class SetCriterion(nn.Module):
    """
    DETR-style loss computation with Hungarian matching.
    
    The process:
    1. Compute Hungarian matching between predictions and ground truth
    2. Supervise matched pairs (classification + bbox regression)
    
    Args:
        num_classes: Number of object categories (excluding no-object)
        matcher: Module to compute matching between predictions and targets
        weight_dict: Dict of loss weights
        eos_coef: Weight for no-object class (typically < 1 to handle class imbalance)
        losses: List of losses to compute ('labels', 'boxes', 'cardinality')
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        losses: List[str] = ['labels', 'boxes'],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # Class weights: lower weight for no-object class
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(
        self,
        outputs: Dict,
        targets: List[Dict],
        indices: List,
        num_boxes: int,
        log: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Classification loss (Cross-Entropy).
        
        Args:
            outputs: Model outputs with 'pred_logits'
            targets: List of target dicts with 'labels'
            indices: Matching indices from Hungarian matcher
            num_boxes: Number of target boxes (for normalization)
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [B, N, num_classes+1]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {'loss_ce': loss_ce}

        if log:
            # Classification accuracy for matched predictions
            losses['class_error'] = 100 - self._accuracy(src_logits[idx], target_classes_o)[0]
        
        return losses

    def loss_boxes(
        self,
        outputs: Dict,
        targets: List[Dict],
        indices: List,
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Bounding box losses (L1 + GIoU).
        
        Args:
            outputs: Model outputs with 'pred_boxes'
            targets: List of target dicts with 'boxes'
            indices: Matching indices
            num_boxes: Number of target boxes
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # [num_matched, 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        # GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    @torch.no_grad()
    def loss_cardinality(
        self,
        outputs: Dict,
        targets: List[Dict],
        indices: List,
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Cardinality error (for logging, not training).
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        
        # Count non-background predictions
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        
        return {'cardinality_error': card_err}

    def _get_src_permutation_idx(self, indices):
        """Get flattened source indices for gathering matched predictions."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Get flattened target indices."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @torch.no_grad()
    def _accuracy(self, output, target, topk=(1,)):
        """Compute top-k accuracy."""
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def get_loss(self, loss_name, outputs, targets, indices, num_boxes, **kwargs):
        """Get a specific loss by name."""
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
        }
        assert loss_name in loss_map, f'Unknown loss: {loss_name}'
        return loss_map[loss_name](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        Compute all losses.
        
        Args:
            outputs: Dict with 'pred_logits' and 'pred_boxes'
            targets: List of dicts with 'labels' and 'boxes'
        
        Returns:
            Dict of losses
        """
        # Match predictions to targets
        indices = self.matcher(outputs, targets)

        # Total number of target boxes (for normalization)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Apply weights
        weighted_losses = {}
        for k, v in losses.items():
            if k in self.weight_dict:
                weighted_losses[k] = v * self.weight_dict[k]
            else:
                weighted_losses[k] = v

        # Total loss
        weighted_losses['loss'] = sum(weighted_losses[k] for k in weighted_losses if k.startswith('loss_'))

        return weighted_losses


class SimpleCriterion(nn.Module):
    """
    Simplified criterion for single-object cross-view localization.
    
    Assumes one target per image, matches best prediction to target.
    """
    
    def __init__(
        self,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
    ):
        super().__init__()
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
    
    def forward(self, outputs, targets):
        """
        Compute bbox losses for best-matched predictions.
        
        Args:
            outputs: Dict with 'pred_boxes' [B, N, 4]
            targets: Dict with 'sat_bbox' [B, 4]
        """
        pred_boxes = outputs['pred_boxes']  # [B, N, 4]
        target_boxes = targets['sat_bbox']  # [B, 4]
        
        B, N, _ = pred_boxes.shape
        
        losses = {'loss_bbox': 0.0, 'loss_giou': 0.0}
        
        for b in range(B):
            # Find best prediction by GIoU
            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])  # [N, 4]
            tgt_xyxy = box_cxcywh_to_xyxy(target_boxes[b:b+1])  # [1, 4]
            
            giou = generalized_box_iou(pred_xyxy, tgt_xyxy).squeeze(-1)  # [N]
            best_idx = giou.argmax()
            
            # L1 loss
            loss_bbox = F.l1_loss(pred_boxes[b, best_idx], target_boxes[b])
            losses['loss_bbox'] = losses['loss_bbox'] + loss_bbox
            
            # GIoU loss
            loss_giou = 1 - giou[best_idx]
            losses['loss_giou'] = losses['loss_giou'] + loss_giou
        
        # Average over batch
        losses['loss_bbox'] = losses['loss_bbox'] / B
        losses['loss_giou'] = losses['loss_giou'] / B
        
        # Weighted total
        losses['loss'] = (
            self.weight_bbox * losses['loss_bbox'] +
            self.weight_giou * losses['loss_giou']
        )
        
        return losses


def build_criterion(
    num_classes: int = 1,
    matcher: nn.Module = None,
    weight_dict: Dict[str, float] = None,
    eos_coef: float = 0.1,
    losses: List[str] = ['labels', 'boxes'],
):
    """Build SetCriterion with specified parameters."""
    if weight_dict is None:
        weight_dict = {
            'loss_ce': 1,
            'loss_bbox': 5,
            'loss_giou': 2,
        }
    
    return SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses,
    )
