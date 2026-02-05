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
    """获取 bbox prompt，转换为 [x1, y1, x2, y2] 格式"""
    B = batch['front_view'].shape[0]
    prompt_bbox = batch['mono_bbox'].to(device)  # [B, 4] in [cx, cy, w, h]
    boxes = torch.zeros(B, 1, 4, device=device)
    boxes[:, 0, 0] = prompt_bbox[:, 0] - prompt_bbox[:, 2] / 2  # x1
    boxes[:, 0, 1] = prompt_bbox[:, 1] - prompt_bbox[:, 3] / 2  # y1
    boxes[:, 0, 2] = prompt_bbox[:, 0] + prompt_bbox[:, 2] / 2  # x2
    boxes[:, 0, 3] = prompt_bbox[:, 1] + prompt_bbox[:, 3] / 2  # y2
    return boxes


def _get_mask_prompt(batch: Dict, device: torch.device) -> torch.Tensor:
    """获取 mask prompt"""
    return batch['mono_mask'].to(device)  # [B, 1, H, W]


def prepare_random_prompt(
    batch: Dict,
    device: torch.device,
    prompt_types: List[str] = ['point', 'bbox', 'mask'],
    min_prompts: int = 1,
    max_prompts: int = 3,
) -> Tuple[Optional[Tuple], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    随机组合多种 prompt 类型进行训练
    
    Args:
        batch: 数据 batch
        device: 设备
        prompt_types: 可选的 prompt 类型列表
        min_prompts: 最少使用的 prompt 数量
        max_prompts: 最多使用的 prompt 数量
    
    Returns:
        points: (coords, labels) 或 None
        boxes: bbox tensor 或 None
        masks: mask tensor 或 None
    """
    # 随机选择使用几种 prompt（1 到 min(max_prompts, len(prompt_types))）
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
