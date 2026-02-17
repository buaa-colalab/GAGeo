# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from SAM2's prompt encoder for cross-view localization.
#
# Supports optional dual-dimension mode:
#   - Internal SAM layers operate at sam_embed_dim (e.g. 256, native SAM dimension)
#   - Output projection layers upscale to embed_dim (e.g. 2048, matching backbone)
#   - SAM pretrained weights load directly into the 256-dim layers
#   - Only projection layers need training from scratch

from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pe_random import PositionEmbeddingRandom
from .layer_norm import LayerNorm2d


class GeometryPromptEncoder(nn.Module):
    """
    Encodes user geometry prompts (points, boxes, masks) for cross-view localization.
    
    Adapted from SAM's PromptEncoder. Supports loading SAM pretrained weights
    via the sam_embed_dim parameter.
    
    Args:
        embed_dim: Output embedding dimension (should match backbone, e.g. 2048)
        image_embedding_size: Spatial size of image features (H_patch, W_patch)
        input_image_size: Original input image size (H, W)
        mask_in_chans: Hidden channels for mask encoding
        sam_embed_dim: If set, use SAM-native dimension internally and add 
                       learnable projection layers to embed_dim. This allows
                       direct loading of SAM pretrained weights (256-dim).
                       If None, all layers use embed_dim directly.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        image_embedding_size: Tuple[int, int] = (37, 37),  # 518/14
        input_image_size: Tuple[int, int] = (518, 518),
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
        sam_embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        
        # Internal dimension: SAM-native (256) or full embed_dim (2048)
        self._internal_dim = sam_embed_dim if sam_embed_dim else embed_dim
        
        self.pe_layer = PositionEmbeddingRandom(self._internal_dim // 2)

        # Point embeddings: pos/neg point + 2 box corners (at internal dim)
        self.num_point_embeddings: int = 4
        point_embeddings = [
            nn.Embedding(1, self._internal_dim) for _ in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, self._internal_dim)

        # Mask encoding (at internal dim)
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, self._internal_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, self._internal_dim)
        
        # Output projection: internal_dim -> embed_dim (only when dimensions differ)
        # These are the ONLY layers that need training from scratch when loading SAM weights
        self.sparse_proj = None
        self.dense_proj = None
        if sam_embed_dim is not None and sam_embed_dim != embed_dim:
            self.sparse_proj = nn.Sequential(
                nn.Linear(sam_embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            self.dense_proj = nn.Sequential(
                nn.Conv2d(sam_embed_dim, embed_dim, kernel_size=1),
                LayerNorm2d(embed_dim),
            )

    def get_dense_pe(self) -> torch.Tensor:
        """Returns positional encoding for dense features."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """
        Embeds point prompts.
        
        Args:
            points: [B, N, 2] point coordinates in pixel space
            labels: [B, N] point labels (0=neg, 1=pos, 2/3=box corners)
            pad: Whether to add padding point
        
        Returns:
            [B, N(+1), C] point embeddings
        """
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # Add type embeddings based on labels
        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            self.not_a_point_embed.weight.expand_as(point_embedding),
            point_embedding,
        )
        for i in range(4):
            point_embedding = torch.where(
                (labels == i).unsqueeze(-1),
                point_embedding + self.point_embeddings[i].weight,
                point_embedding,
            )
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Embeds box prompts.
        
        Args:
            boxes: [B, N, 4] boxes in pixel-space (x, y, w, h) format
        
        Returns:
            [B, N*2, C] corner embeddings
        """
        B, N = boxes.shape[:2]
        
        # Convert (x, y, w, h) to corner coordinates
        # boxes[:, :, 0:2] is top-left (x, y)
        # boxes[:, :, 2:4] is (w, h)
        x1 = boxes[:, :, 0]
        y1 = boxes[:, :, 1]
        w = boxes[:, :, 2]
        h = boxes[:, :, 3]
        x2 = x1 + w
        y2 = y1 + h
        
        # Stack corners: [B, N, 2, 2] -> top-left and bottom-right
        corners = torch.stack([
            torch.stack([x1, y1], dim=-1),  # top-left
            torch.stack([x2, y2], dim=-1),  # bottom-right
        ], dim=2)  # [B, N, 2, 2]
        
        corners = corners + 0.5  # Shift to center of pixel
        corners = corners.view(B * N, 2, 2)  # [B*N, 2, 2]
        
        corner_embedding = self.pe_layer.forward_with_coords(corners, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # top-left corner
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight  # bottom-right corner
        
        return corner_embedding.view(B, N * 2, -1)  # [B, N*2, C]

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Embeds mask inputs.
        
        Args:
            masks: [B, 1, H, W] binary masks (any size)
        
        Returns:
            [B, C, 37, 37] mask embeddings (matches image_embedding_size)
        """
        # Resize to mask_input_size (148x148) if needed
        if masks.shape[2:] != self.mask_input_size:
            masks = F.interpolate(masks, size=self.mask_input_size, mode='bilinear', align_corners=False)
        # Downscale: 148 -> 74 -> 37
        return self.mask_downscaling(masks)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts.
        
        Args:
            points: Tuple of (coords [B, N, 2], labels [B, N])
            boxes: [B, M, 4] boxes in pixel-space (x, y, w, h) format
            masks: [B, 1, H, W] binary masks
        
        Returns:
            sparse_embeddings: [B, N_sparse, C] for points and boxes
            dense_embeddings: [B, C, H', W'] for masks
        """
        # Determine batch size
        if points is not None:
            bs = points[0].shape[0]
        elif boxes is not None:
            bs = boxes.shape[0]
        elif masks is not None:
            bs = masks.shape[0]
        else:
            bs = 1

        device = self.point_embeddings[0].weight.device
        sparse_embeddings = torch.empty((bs, 0, self._internal_dim), device=device)

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)  # Already [B, N*2, C]
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        # Project from internal SAM dim (256) to output dim (2048) if needed
        if self.sparse_proj is not None:
            proj_dtype = self.sparse_proj[0].weight.dtype
            if sparse_embeddings.dtype != proj_dtype:
                sparse_embeddings = sparse_embeddings.to(proj_dtype)
            sparse_embeddings = self.sparse_proj(sparse_embeddings)
        if self.dense_proj is not None:
            proj_dtype = self.dense_proj[0].weight.dtype
            if dense_embeddings.dtype != proj_dtype:
                dense_embeddings = dense_embeddings.to(proj_dtype)
            dense_embeddings = self.dense_proj(dense_embeddings)

        return sparse_embeddings, dense_embeddings


def load_sam_prompt_encoder_weights(
    prompt_encoder: GeometryPromptEncoder,
    sam_checkpoint_path: str,
) -> None:
    """
    Load SAM2 pretrained prompt encoder weights.
    
    Requires prompt_encoder to be created with sam_embed_dim=256 so internal
    layers match SAM's native dimension. The projection layers (sparse_proj,
    dense_proj) will remain randomly initialized and trainable.
    
    SAM2 checkpoint key format:
        sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix
        sam_prompt_encoder.point_embeddings.{0,1,2,3}.weight
        sam_prompt_encoder.not_a_point_embed.weight
        sam_prompt_encoder.mask_downscaling.{0,1,3,4,6}.{weight,bias}
        sam_prompt_encoder.no_mask_embed.weight
    
    Args:
        prompt_encoder: GeometryPromptEncoder with sam_embed_dim=256
        sam_checkpoint_path: Path to SAM2 checkpoint (.pt file)
    """
    import torch as _torch
    
    ckpt = _torch.load(sam_checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    # Extract sam_prompt_encoder.* keys and strip prefix
    pe_state = {}
    for k, v in state_dict.items():
        if k.startswith('sam_prompt_encoder.'):
            new_key = k[len('sam_prompt_encoder.'):]  # strip prefix
            pe_state[new_key] = v
    
    if not pe_state:
        print(f"  WARNING: No sam_prompt_encoder keys found in {sam_checkpoint_path}")
        return
    
    missing, unexpected = prompt_encoder.load_state_dict(pe_state, strict=False)
    
    # Report results
    loaded_count = len(pe_state) - len(unexpected)
    print(f"Loaded SAM prompt encoder weights from {sam_checkpoint_path}")
    print(f"  Loaded: {loaded_count} keys (SAM native layers)")
    
    # Separate missing keys into projection (expected) and others (unexpected)
    proj_missing = [k for k in missing if 'proj' in k]
    other_missing = [k for k in missing if 'proj' not in k]
    
    if proj_missing:
        print(f"  Projection layers (randomly initialized, trainable): {len(proj_missing)}")
    if other_missing:
        print(f"  WARNING - Missing non-projection keys: {other_missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
