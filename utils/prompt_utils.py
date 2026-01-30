"""
Prompt utilities for training with random prompt selection
"""

import random
import torch
from typing import Dict, Optional, Tuple


def prepare_random_prompt(
    batch: Dict,
    device: torch.device,
    prompt_types: list = ['point', 'bbox', 'mask'],
) -> Tuple[Optional[Tuple], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    随机选择一种prompt类型进行训练（模拟真实场景）
    
    Args:
        batch: 数据batch
        device: 设备
        prompt_types: 可选的prompt类型列表
    
    Returns:
        points: (coords, labels) 或 None
        boxes: bbox tensor 或 None
        masks: mask tensor 或 None
    """
    # 随机选择一种prompt类型
    prompt_type = random.choice(prompt_types)
    
    B = batch['front_view'].shape[0]
    points = None
    boxes = None
    masks = None
    
    if prompt_type == 'point':
        # Point prompt: 使用mono_point
        mono_point = batch['mono_point'].to(device)
        point_coords = mono_point.unsqueeze(1)  # [B, 1, 2]
        point_labels = torch.ones(B, 1, device=device)  # 正点
        points = (point_coords, point_labels)
        
    elif prompt_type == 'bbox':
        # Box prompt: 使用mono_bbox
        mono_bbox = batch['mono_bbox'].to(device)  # [B, 4] in [x, y, w, h]
        # 转换为 [x1, y1, x2, y2] 格式
        boxes = torch.zeros(B, 1, 4, device=device)
        boxes[:, 0, 0] = mono_bbox[:, 0]  # x1
        boxes[:, 0, 1] = mono_bbox[:, 1]  # y1
        boxes[:, 0, 2] = mono_bbox[:, 0] + mono_bbox[:, 2]  # x2 = x + w
        boxes[:, 0, 3] = mono_bbox[:, 1] + mono_bbox[:, 3]  # y2 = y + h
        
    elif prompt_type == 'mask':
        # Mask prompt: 使用mono_mask
        masks = batch['mono_mask'].to(device)  # [B, 1, H, W]
    
    return points, boxes, masks


def prepare_all_prompts(
    batch: Dict,
    device: torch.device,
) -> Tuple[Optional[Tuple], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    准备所有prompt（用于调试或特殊场景）
    
    Args:
        batch: 数据batch
        device: 设备
    
    Returns:
        points: (coords, labels)
        boxes: bbox tensor
        masks: mask tensor
    """
    B = batch['front_view'].shape[0]
    
    # Point prompt
    mono_point = batch['mono_point'].to(device)
    point_coords = mono_point.unsqueeze(1)  # [B, 1, 2]
    point_labels = torch.ones(B, 1, device=device)
    points = (point_coords, point_labels)
    
    # Box prompt
    mono_bbox = batch['mono_bbox'].to(device)  # [B, 4] in [x, y, w, h]
    boxes = torch.zeros(B, 1, 4, device=device)
    boxes[:, 0, 0] = mono_bbox[:, 0]  # x1
    boxes[:, 0, 1] = mono_bbox[:, 1]  # y1
    boxes[:, 0, 2] = mono_bbox[:, 0] + mono_bbox[:, 2]  # x2
    boxes[:, 0, 3] = mono_bbox[:, 1] + mono_bbox[:, 3]  # y2
    
    # Mask prompt
    masks = batch['mono_mask'].to(device)
    
    return points, boxes, masks


def prepare_point_prompt(
    batch: Dict,
    device: torch.device,
) -> Tuple[Optional[Tuple], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    只准备point prompt（向后兼容）
    
    Args:
        batch: 数据batch
        device: 设备
    
    Returns:
        points: (coords, labels)
        boxes: None
        masks: None
    """
    B = batch['front_view'].shape[0]
    mono_point = batch['mono_point'].to(device)
    point_coords = mono_point.unsqueeze(1)  # [B, 1, 2]
    point_labels = torch.ones(B, 1, device=device)
    points = (point_coords, point_labels)
    
    return points, None, None
