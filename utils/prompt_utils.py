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
    
    支持双向定位: 使用 prompt_point, prompt_bbox, prompt_mask 字段
    这些字段根据 direction 自动选择来自 mono 或 sat 视图的 prompt
    
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
    
    B = batch['mono_view'].shape[0]
    
    points = None
    boxes = None
    masks = None
    
    if prompt_type == 'point':
        # Point prompt: 使用 prompt_point (根据方向自动选择)
        prompt_point = batch.get('prompt_point', batch.get('mono_point')).to(device)
        point_coords = prompt_point.unsqueeze(1)  # [B, 1, 2]
        point_labels = torch.ones(B, 1, device=device)  # 正点
        points = (point_coords, point_labels)
        
    elif prompt_type == 'bbox':
        # Box prompt: 使用 prompt_bbox (根据方向自动选择)
        prompt_bbox = batch.get('prompt_bbox', batch.get('mono_bbox')).to(device)  # [B, 4] in [cx, cy, w, h]
        # 注意: prompt_bbox 是归一化的 [cx, cy, w, h] 格式
        # 转换为 [x1, y1, x2, y2] 格式 (归一化坐标)
        boxes = torch.zeros(B, 1, 4, device=device)
        boxes[:, 0, 0] = prompt_bbox[:, 0] - prompt_bbox[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 0, 1] = prompt_bbox[:, 1] - prompt_bbox[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 0, 2] = prompt_bbox[:, 0] + prompt_bbox[:, 2] / 2  # x2 = cx + w/2
        boxes[:, 0, 3] = prompt_bbox[:, 1] + prompt_bbox[:, 3] / 2  # y2 = cy + h/2
        
    elif prompt_type == 'mask':
        # Mask prompt: 使用 prompt_mask (根据方向自动选择)
        masks = batch.get('prompt_mask', batch.get('mono_mask')).to(device)  # [B, 1, H, W]
    
    return points, boxes, masks


def prepare_all_prompts(
    batch: Dict,
    device: torch.device,
) -> Tuple[Optional[Tuple], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    准备所有prompt（用于调试或特殊场景）
    
    支持双向定位: 使用 prompt_* 字段
    
    Args:
        batch: 数据batch
        device: 设备
    
    Returns:
        points: (coords, labels)
        boxes: bbox tensor
        masks: mask tensor
    """
    B = batch['mono_view'].shape[0]
    
    # Point prompt
    prompt_point = batch.get('prompt_point', batch.get('mono_point')).to(device)
    point_coords = prompt_point.unsqueeze(1)  # [B, 1, 2]
    point_labels = torch.ones(B, 1, device=device)
    points = (point_coords, point_labels)
    
    # Box prompt
    prompt_bbox = batch.get('prompt_bbox', batch.get('mono_bbox')).to(device)  # [B, 4] in [cx, cy, w, h]
    boxes = torch.zeros(B, 1, 4, device=device)
    boxes[:, 0, 0] = prompt_bbox[:, 0] - prompt_bbox[:, 2] / 2  # x1
    boxes[:, 0, 1] = prompt_bbox[:, 1] - prompt_bbox[:, 3] / 2  # y1
    boxes[:, 0, 2] = prompt_bbox[:, 0] + prompt_bbox[:, 2] / 2  # x2
    boxes[:, 0, 3] = prompt_bbox[:, 1] + prompt_bbox[:, 3] / 2  # y2
    
    # Mask prompt
    masks = batch.get('prompt_mask', batch.get('mono_mask')).to(device)
    
    return points, boxes, masks
