# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Hungarian Matcher for DETR-style object detection.
Computes optimal bipartite matching between predictions and ground truth.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    Computes an assignment between predictions and targets using Hungarian algorithm.
    
    For efficiency, targets don't include no_object. Since there are typically more
    predictions than targets, we do 1-to-1 matching of best predictions while others
    are treated as non-objects.
    
    Args:
        cost_class: Weight of classification cost
        cost_bbox: Weight of L1 bbox cost
        cost_giou: Weight of GIoU cost
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching.
        
        Args:
            outputs: Dict containing:
                - "pred_logits": [B, num_queries, num_classes] classification logits
                - "pred_boxes": [B, num_queries, 4] predicted boxes (cx, cy, w, h)
            
            targets: List of dicts (len=B), each containing:
                - "labels": [num_target_boxes] class labels
                - "boxes": [num_target_boxes, 4] target boxes (cx, cy, w, h)
        
        Returns:
            List of (pred_indices, target_indices) tuples for each batch element
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten to compute cost matrices in batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*num_queries, 4]

        # Concat target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Classification cost: -prob[target_class]
        cost_class = -out_prob[:, tgt_ids]

        # L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]


class SimpleMatcher(nn.Module):
    """
    Simplified matcher for single-object detection (cross-view localization).
    
    Since we typically have one target per image, this matcher simply matches
    the best prediction to the single target based on IoU.
    """
    
    def __init__(self, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Match predictions to targets.
        
        Args:
            outputs: Dict with "pred_boxes" [B, num_queries, 4]
            targets: Dict with "sat_bbox" [B, 4] (single target per image)
        
        Returns:
            List of (pred_idx, target_idx) tuples
        """
        pred_boxes = outputs["pred_boxes"]  # [B, N, 4]
        target_boxes = targets["sat_bbox"]  # [B, 4]
        
        B, N, _ = pred_boxes.shape
        
        indices = []
        for b in range(B):
            # Compute GIoU between all predictions and the single target
            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])  # [N, 4]
            tgt_xyxy = box_cxcywh_to_xyxy(target_boxes[b:b+1])  # [1, 4]
            
            giou = generalized_box_iou(pred_xyxy, tgt_xyxy).squeeze(-1)  # [N]
            
            # Select best prediction
            best_idx = giou.argmax().item()
            indices.append((torch.tensor([best_idx]), torch.tensor([0])))
        
        return indices


def build_matcher(cost_class=1, cost_bbox=5, cost_giou=2):
    """Build Hungarian matcher with specified costs."""
    return HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
