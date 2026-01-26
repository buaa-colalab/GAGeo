import torch
import numpy as np
from typing import List, Tuple

from .box_ops import box_iou


def compute_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute IoU between predicted and target boxes.
    
    Args:
        pred_boxes: [N, 4] predicted boxes in (x1, y1, x2, y2) format
        target_boxes: [M, 4] target boxes in (x1, y1, x2, y2) format
    
    Returns:
        iou: [N, M] IoU matrix
    """
    return box_iou(pred_boxes, target_boxes)


def compute_ap(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    target_boxes: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute Average Precision (AP) at given IoU threshold.
    
    Args:
        pred_boxes: List of [N_i, 4] predicted boxes for each image
        pred_scores: List of [N_i] confidence scores for each image
        target_boxes: List of [M_i, 4] target boxes for each image
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        ap: Average Precision
    """
    all_pred_boxes = []
    all_pred_scores = []
    all_target_boxes = []
    
    for i in range(len(pred_boxes)):
        all_pred_boxes.append(pred_boxes[i])
        all_pred_scores.append(pred_scores[i])
        all_target_boxes.extend([target_boxes[i]] * len(pred_boxes[i]))
    
    if len(all_pred_boxes) == 0:
        return 0.0
    
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    
    # Sort by confidence
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    
    # Compute precision-recall curve
    tp = torch.zeros(len(all_pred_boxes))
    fp = torch.zeros(len(all_pred_boxes))
    
    num_targets = sum(len(t) for t in target_boxes)
    
    matched_targets = set()
    
    for i, (pred_box, pred_score) in enumerate(zip(all_pred_boxes, all_pred_scores)):
        # Find best matching target
        best_iou = 0.0
        best_target_idx = -1
        
        for j, target_box in enumerate(all_target_boxes):
            if j in matched_targets:
                continue
            
            iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0))[0, 0]
            
            if iou > best_iou:
                best_iou = iou
                best_target_idx = j
        
        if best_iou >= iou_threshold:
            tp[i] = 1
            matched_targets.add(best_target_idx)
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    recalls = tp_cumsum / (num_targets + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max().item()
    ap /= 11
    
    return ap


def compute_localization_accuracy(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute localization accuracy metrics.
    
    Args:
        pred_boxes: [N, 4] predicted boxes
        target_boxes: [M, 4] target boxes
        threshold: IoU threshold for correct localization
    
    Returns:
        precision: Precision
        recall: Recall
        f1: F1 score
    """
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0.0, 0.0, 0.0
    
    iou_matrix = compute_iou(pred_boxes, target_boxes)
    
    # For each prediction, check if it matches any target
    max_iou_per_pred, _ = iou_matrix.max(dim=1)
    tp = (max_iou_per_pred >= threshold).sum().item()
    
    # For each target, check if it's matched by any prediction
    max_iou_per_target, _ = iou_matrix.max(dim=0)
    matched_targets = (max_iou_per_target >= threshold).sum().item()
    
    precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    recall = matched_targets / len(target_boxes) if len(target_boxes) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision, recall, f1


def compute_distance_error(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> float:
    """
    Compute average center distance error between predictions and targets.
    
    Args:
        pred_boxes: [N, 4] predicted boxes in (cx, cy, w, h) format
        target_boxes: [N, 4] target boxes in (cx, cy, w, h) format
    
    Returns:
        distance_error: Average Euclidean distance between centers
    """
    pred_centers = pred_boxes[:, :2]
    target_centers = target_boxes[:, :2]
    
    distances = torch.norm(pred_centers - target_centers, dim=1)
    return distances.mean().item()
