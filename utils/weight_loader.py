# Weight loading utilities for cross-view localization model
# Supports loading from DINOv2, VGGT, and custom checkpoints

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_dinov2_weights(
    model: nn.Module,
    dinov2_path: Optional[str] = None,
    dinov2_model_name: str = "dinov2_vitl14_reg",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load DINOv2 pretrained weights into the model's patch_embed (DinoVisionTransformer).
    """
    if dinov2_path is not None:
        logger.info(f"Loading DINOv2 weights from {dinov2_path}")
        state_dict = torch.load(dinov2_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    else:
        logger.info(f"Downloading DINOv2 weights: {dinov2_model_name}")
        dinov2_model = torch.hub.load('facebookresearch/dinov2', dinov2_model_name)
        state_dict = dinov2_model.state_dict()
    
    patch_embed = model.vggt.patch_embed
    result = patch_embed.load_state_dict(state_dict, strict=strict)
    
    logger.info(f"Loaded DINOv2 weights. Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    
    return {
        'missing_keys': result.missing_keys,
        'unexpected_keys': result.unexpected_keys,
    }


def load_vggt_weights(
    model: nn.Module,
    vggt_path: str,
    load_patch_embed: bool = True,
    load_aggregator: bool = True,
    load_heads: bool = False,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load VGGT pretrained weights into the model.
    """
    logger.info(f"Loading VGGT weights from {vggt_path}")
    state_dict = torch.load(vggt_path, map_location='cpu')
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    filtered_state = {}
    skipped_keys = []
    
    for k, v in state_dict.items():
        if k.startswith('aggregator.'):
            new_k = k[len('aggregator.'):]
        else:
            new_k = k
        
        include = False
        
        if new_k.startswith('patch_embed.') and load_patch_embed:
            include = True
        elif (new_k.startswith('frame_blocks.') or 
              new_k.startswith('global_blocks.') or
              new_k.startswith('camera_token') or
              new_k.startswith('register_token')) and load_aggregator:
            include = True
        elif any(new_k.startswith(h) for h in ['camera_head.', 'depth_head.', 'point_head.', 'track_head.']):
            if load_heads:
                include = True
            else:
                skipped_keys.append(k)
        
        if include:
            filtered_state[new_k] = v
    
    result = model.vggt.load_state_dict(filtered_state, strict=strict)
    
    logger.info(f"Loaded VGGT weights. Missing: {len(result.missing_keys)}, "
                f"Unexpected: {len(result.unexpected_keys)}, Skipped heads: {len(skipped_keys)}")
    
    return {
        'missing_keys': result.missing_keys,
        'unexpected_keys': result.unexpected_keys,
        'skipped_keys': skipped_keys,
    }


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Load a full training checkpoint.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    
    if load_optimizer and optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', None),
        'global_step': checkpoint.get('global_step', 0),
    }
    
    logger.info(f"Loaded checkpoint. Epoch: {info['epoch']}, Best metric: {info['best_metric']}")
    
    return info


def freeze_backbone(model: nn.Module, freeze_patch_embed: bool = True, freeze_aggregator: bool = False):
    """
    Freeze parts of the backbone for fine-tuning.
    """
    if freeze_patch_embed:
        for param in model.vggt.patch_embed.parameters():
            param.requires_grad = False
        logger.info("Frozen patch_embed (DINOv2)")
    
    if freeze_aggregator:
        for name, param in model.vggt.named_parameters():
            if not name.startswith('patch_embed.'):
                param.requires_grad = False
        logger.info("Frozen aggregator (frame/global blocks)")


def get_param_groups(
    model: nn.Module,
    lr_backbone: float = 1e-5,
    lr_heads: float = 1e-4,
    weight_decay: float = 0.01,
) -> list:
    """
    Get parameter groups with different learning rates.
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if name.startswith('vggt.'):
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr_heads, 'weight_decay': weight_decay},
    ]
    
    logger.info(f"Param groups: backbone ({len(backbone_params)} params, lr={lr_backbone}), "
                f"heads ({len(head_params)} params, lr={lr_heads})")
    
    return param_groups
