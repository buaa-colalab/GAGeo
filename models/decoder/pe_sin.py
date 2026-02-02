# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from typing import Tuple, Optional, Union

from utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_or_size: Union[NestedTensor, Tuple[int, int]], 
                device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate positional encoding.
        
        Args:
            tensor_or_size: Either a NestedTensor or a (H, W) tuple
            device: torch device (only used when tensor_or_size is a tuple)
            
        Returns:
            pos: [B, C, H, W] if NestedTensor input, [C, H, W] if tuple input
        """
        if isinstance(tensor_or_size, NestedTensor):
            # Original DETR interface with NestedTensor
            x = tensor_or_size.tensors
            mask = tensor_or_size.mask
            assert mask is not None
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            return pos
        else:
            # Simplified interface with (H, W) tuple - no mask needed
            h, w = tensor_or_size
            if device is None:
                device = torch.device('cpu')
                
            # Create coordinate grids (no mask, all positions valid)
            y_embed = torch.arange(1, h + 1, dtype=torch.float32, device=device).unsqueeze(1).expand(h, w)
            x_embed = torch.arange(1, w + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(h, w)
            
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (h + eps) * self.scale
                x_embed = x_embed / (w + eps) * self.scale

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

            pos_x = x_embed[:, :, None] / dim_t  # [H, W, C/2]
            pos_y = y_embed[:, :, None] / dim_t  # [H, W, C/2]
            pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
            pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
            pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # [C, H, W]
            return pos