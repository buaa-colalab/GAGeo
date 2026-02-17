"""
Prompt utilities for training with random prompt combination
"""

import random
import torch
from typing import Dict, List, Optional, Tuple


def _get_point_prompt(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取 point prompt"""
    B = batch['front_view'].shape[0]
    prompt_point = batch['mono_point'].to(device)
    point_coords = prompt_point.unsqueeze(1)  # [B, 1, 2]
    point_labels = torch.ones(B, 1, device=device)
    return point_coords, point_labels


def _get_bbox_prompt(batch: Dict, device: torch.device) -> torch.Tensor:
    """获取 bbox prompt，转换为像素空间 [x, y, w, h] 格式。"""
    B = batch['front_view'].shape[0]
    prompt_bbox = batch['mono_bbox'].to(device)  # [B, 4] normalized [cx, cy, w, h]
    H = batch['front_view'].shape[-2]
    W = batch['front_view'].shape[-1]

    cx = prompt_bbox[:, 0] * W
    cy = prompt_bbox[:, 1] * H
    bw = prompt_bbox[:, 2] * W
    bh = prompt_bbox[:, 3] * H

    x = cx - bw / 2.0
    y = cy - bh / 2.0

    boxes = torch.zeros(B, 1, 4, device=device, dtype=prompt_bbox.dtype)
    boxes[:, 0, 0] = x.clamp(min=0.0, max=float(W - 1))
    boxes[:, 0, 1] = y.clamp(min=0.0, max=float(H - 1))
    boxes[:, 0, 2] = bw.clamp(min=1.0, max=float(W))
    boxes[:, 0, 3] = bh.clamp(min=1.0, max=float(H))
    return boxes


def _assert_bbox_prompt_xywh_pixel(boxes: torch.Tensor, batch: Dict) -> None:
    """断言 bbox prompt 为像素空间 [x, y, w, h]。"""
    if boxes is None:
        return
    if boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"bbox prompt shape must be [B, N, 4], got {tuple(boxes.shape)}")

    H = batch['front_view'].shape[-2]
    W = batch['front_view'].shape[-1]
    x = boxes[..., 0]
    y = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    if torch.any(w <= 0) or torch.any(h <= 0):
        raise ValueError("bbox prompt width/height must be > 0 in pixel xywh format")
    if torch.any(x < -1e-4) or torch.any(y < -1e-4):
        raise ValueError("bbox prompt x/y must be non-negative pixel coordinates")
    if torch.any(x > (W - 1 + 1e-4)) or torch.any(y > (H - 1 + 1e-4)):
        raise ValueError("bbox prompt x/y exceed image boundary; expected pixel xywh")
    if torch.any(w > (W + 1e-4)) or torch.any(h > (H + 1e-4)):
        raise ValueError("bbox prompt w/h exceed image size; expected pixel xywh")


def _get_mask_prompt(batch: Dict, device: torch.device) -> torch.Tensor:
    """获取 mask prompt"""
    return batch['mono_mask'].to(device)  # [B, 1, H, W]


def prepare_random_prompt(
    batch: Dict,
    device: torch.device,
    prompt_types: List[str] = ['point', 'bbox', 'mask'],
    min_prompts: int = 1,
    max_prompts: int = 1,
) -> Tuple[Optional[Tuple], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    随机选择一种 prompt 类型进行训练（互斥）
    
    Args:
        batch: 数据 batch
        device: 设备
        prompt_types: 可选的 prompt 类型列表
        min_prompts: 最少使用的 prompt 数量（V3固定为1）
        max_prompts: 最多使用的 prompt 数量（V3固定为1）
    
    Returns:
        points: (coords, labels) 或 None
        boxes: bbox tensor 或 None
        masks: mask tensor 或 None
    """
    # V3: prompt 类型最多 1 个（point / bbox / mask 三选一）
    max_prompts = 1
    min_prompts = 1
    num_prompts = random.randint(min_prompts, min(max_prompts, len(prompt_types)))
    selected_types = random.sample(prompt_types, num_prompts)
    
    points = None
    boxes = None
    masks = None
    
    if 'point' in selected_types:
        point_coords, point_labels = _get_point_prompt(batch, device)
        points = (point_coords, point_labels)
    
    if 'bbox' in selected_types:
        boxes = _get_bbox_prompt(batch, device)
        _assert_bbox_prompt_xywh_pixel(boxes, batch)
    
    if 'mask' in selected_types:
        masks = _get_mask_prompt(batch, device)
    
    return points, boxes, masks


def prepare_single_prompt(
    batch: Dict,
    device: torch.device,
    prompt_type: str = 'point',
) -> Tuple[Optional[Tuple], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    准备单一类型的 prompt（用于稳定验证）
    
    Args:
        batch: 数据 batch
        device: 设备
        prompt_type: 'point', 'bbox', 或 'mask'
    
    Returns:
        points, boxes, masks
    """
    points = None
    boxes = None
    masks = None
    
    if prompt_type == 'point':
        point_coords, point_labels = _get_point_prompt(batch, device)
        points = (point_coords, point_labels)
    elif prompt_type == 'bbox':
        boxes = _get_bbox_prompt(batch, device)
        _assert_bbox_prompt_xywh_pixel(boxes, batch)
    elif prompt_type == 'mask':
        masks = _get_mask_prompt(batch, device)
    
    return points, boxes, masks


def prepare_all_prompts(
    batch: Dict,
    device: torch.device,
) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
    """
    准备所有 prompt（用于完整测试）
    
    Args:
        batch: 数据 batch
        device: 设备
    
    Returns:
        points: (coords, labels)
        boxes: bbox tensor
        masks: mask tensor
    """
    point_coords, point_labels = _get_point_prompt(batch, device)
    boxes = _get_bbox_prompt(batch, device)
    masks = _get_mask_prompt(batch, device)
    
    return (point_coords, point_labels), boxes, masks
