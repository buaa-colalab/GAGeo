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
    lr_new_tokens: float = None,
) -> list:
    """
    Get parameter groups with different learning rates.
    
    V2 (3-group strategy):
    - backbone: Pi3 encoder + decoder (pretrained, low LR)
    - new_tokens: learnable queries, intermediate projections, prompt encoder projections (mid LR)
    - heads: task heads, intermediate supervision heads (high LR)
    
    Falls back to 2-group (backbone + heads) if lr_new_tokens is None.
    """
    # Patterns for "new token" parameters in V2 backbone
    NEW_TOKEN_PATTERNS = (
        'backbone.learnable_queries',
        'backbone.register_token',
        'backbone.frame_pos_embed',
        'backbone.intermediate_projs.',
        'backbone.prompt_proj.',
        'prompt_encoder.sparse_proj',
        'prompt_encoder.dense_proj',
    )

    backbone_type = str(getattr(model, "backbone_type", "")).strip().lower()
    is_2d_cva = backbone_type in {"2d_cva", "cva2d", "vit_b16_cva", "dinov2_g14_cva"}
    is_dinov2_joint_vit = backbone_type in {
        "dinov2_joint_vit",
        "joint_vit",
        "dinov2_vit",
        "gageo_dinov2_vit",
    }
    
    backbone_params = []
    new_token_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check new-token patterns first (they live inside backbone.*)
        if lr_new_tokens is not None and any(name.startswith(pat) or ('.' + pat) in name for pat in NEW_TOKEN_PATTERNS):
            new_token_params.append(param)
        # For 2D-CVA ablations, only the visual encoder is pretrained. The
        # cross-view adapter stack, projection layers, register token and final
        # projection are randomly initialized and should not be trained with the
        # tiny backbone LR intended for pretrained weights.
        elif is_2d_cva and name.startswith('backbone.encoder.'):
            backbone_params.append(param)
        elif is_2d_cva and name.startswith('backbone.'):
            head_params.append(param)
        elif is_dinov2_joint_vit and (
            name.startswith('backbone.encoder.')
            or name.startswith('backbone.decoder.')
            or name.startswith('backbone.vit_norm.')
            or name.startswith('backbone.vit_pos_embedding')
        ):
            backbone_params.append(param)
        elif is_dinov2_joint_vit and name.startswith('backbone.'):
            head_params.append(param)
        elif name.startswith('backbone.') or name.startswith('vggt.'):
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Build param groups (skip empty groups)
    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay})
    if len(new_token_params) > 0 and lr_new_tokens is not None:
        param_groups.append({'params': new_token_params, 'lr': lr_new_tokens, 'weight_decay': weight_decay})
    if len(head_params) > 0:
        param_groups.append({'params': head_params, 'lr': lr_heads, 'weight_decay': weight_decay})

    num_backbone = sum(p.numel() for p in backbone_params)
    num_new_tokens = sum(p.numel() for p in new_token_params)
    num_heads = sum(p.numel() for p in head_params)
    
    log_parts = [f"backbone ({num_backbone/1e6:.2f}M params, lr={lr_backbone})"]
    if lr_new_tokens is not None:
        log_parts.append(f"new_tokens ({num_new_tokens/1e6:.2f}M params, lr={lr_new_tokens})")
    log_parts.append(f"heads ({num_heads/1e6:.2f}M params, lr={lr_heads})")
    logger.info(f"Param groups: {', '.join(log_parts)}, groups={len(param_groups)}")
    
    return param_groups
