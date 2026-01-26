import torch
from typing import Tuple


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.
    
    Args:
        boxes: [..., 4] tensor in (cx, cy, w, h) format
    
    Returns:
        boxes: [..., 4] tensor in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.
    
    Args:
        boxes: [..., 4] tensor in (x1, y1, x2, y2) format
    
    Returns:
        boxes: [..., 4] tensor in (cx, cy, w, h) format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute area of boxes.
    
    Args:
        boxes: [N, 4] tensor in (x1, y1, x2, y2) format
    
    Returns:
        area: [N] tensor
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: [N, 4] tensor in (x1, y1, x2, y2) format
        boxes2: [M, 4] tensor in (x1, y1, x2, y2) format
    
    Returns:
        iou: [N, M] tensor
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / (union + 1e-6)
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of boxes.
    
    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C is the smallest enclosing box.
    
    Args:
        boxes1: [N, 4] tensor in (x1, y1, x2, y2) format
        boxes2: [M, 4] tensor in (x1, y1, x2, y2) format
    
    Returns:
        giou: [N, M] tensor
    """
    iou = box_iou(boxes1, boxes2)
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    # Compute enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area_c = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - iou * (area1[:, None] + area2)
    
    giou = iou - (area_c - union) / (area_c + 1e-6)
    return giou


def clip_boxes_to_image(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Clip boxes to image boundaries.
    
    Args:
        boxes: [N, 4] tensor in (x1, y1, x2, y2) format
        size: (height, width) of image
    
    Returns:
        clipped_boxes: [N, 4] tensor
    """
    h, w = size
    boxes = boxes.clone()
    boxes[:, 0].clamp_(min=0, max=w)
    boxes[:, 1].clamp_(min=0, max=h)
    boxes[:, 2].clamp_(min=0, max=w)
    boxes[:, 3].clamp_(min=0, max=h)
    return boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    """
    Remove boxes with width or height smaller than min_size.
    
    Args:
        boxes: [N, 4] tensor in (x1, y1, x2, y2) format
        min_size: Minimum size threshold
    
    Returns:
        keep: [K] tensor of indices to keep
    """
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return torch.where(keep)[0]
