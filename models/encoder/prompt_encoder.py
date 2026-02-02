# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from SAM2's prompt encoder for cross-view localization.

from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pe_random import PositionEmbeddingRandom
from .layer_norm import LayerNorm2d


class GeometryPromptEncoder(nn.Module):
    """
    Encodes user geometry prompts (points, boxes, masks) for cross-view localization.
    
    Adapted from SAM's PromptEncoder to work with VGGT features.
    
    Args:
        embed_dim: Embedding dimension (should match VGGT's 2*C output)
        image_embedding_size: Spatial size of image features (H_patch, W_patch)
        input_image_size: Original input image size (H, W)
        mask_in_chans: Hidden channels for mask encoding
    """

    def __init__(
        self,
        embed_dim: int = 2048,  # 2*C from VGGT
        image_embedding_size: Tuple[int, int] = (37, 37),  # 518/14
        input_image_size: Tuple[int, int] = (518, 518),
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Point embeddings: pos/neg point + 2 box corners
        self.num_point_embeddings: int = 4
        point_embeddings = [
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Mask encoding
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
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

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
            boxes: [B, N, 4] boxes in (x, y, w, h) or (x1, y1, x2, y2) format
                   We assume (x, y, w, h) format from the data
        
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
            boxes: [B, M, 4] boxes in (x1, y1, x2, y2) format
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
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=device)

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

        return sparse_embeddings, dense_embeddings
