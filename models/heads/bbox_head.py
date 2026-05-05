# BBox Prediction Head for DETR-style detection

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from ..layers.mlp import MLP


class BBoxHead(nn.Module):
    """
    DETR-style BBox prediction head.
    
    Takes object query outputs from decoder and predicts:
    - Bounding boxes (cx, cy, w, h) normalized to [0, 1]
    - Confidence scores
    
    Args:
        hidden_dim: Input feature dimension
        num_classes: Number of object classes (1 for single-object detection)
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_classes: int = 1,
        use_spatial_conditioning: bool = False,
        spatial_hidden_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_spatial_conditioning = bool(use_spatial_conditioning)
        
        # 3-layer MLP for bbox regression (DETR-style)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if self.use_spatial_conditioning:
            self.query_proj = nn.Linear(hidden_dim, spatial_hidden_dim)
            self.spatial_proj = nn.Linear(hidden_dim, spatial_hidden_dim)
            self.spatial_size_embed = MLP(hidden_dim * 2, hidden_dim, 2, 3)
            self.spatial_offset_embed = MLP(hidden_dim * 2, hidden_dim, 2, 3)
        
        # Classification head (confidence score)
        self.class_embed = nn.Linear(hidden_dim, num_classes)

    def _spatial_boxes(
        self,
        query_features: torch.Tensor,
        spatial_features: torch.Tensor,
        spatial_size,
    ) -> torch.Tensor:
        """Predict boxes from query-to-satellite spatial attention."""
        if spatial_features is None or spatial_size is None:
            raise ValueError("spatial_features and spatial_size are required for spatial bbox conditioning")

        B, Q, C = query_features.shape
        H, W = spatial_size
        if spatial_features.shape[1] != H * W:
            raise ValueError(
                f"spatial_features has {spatial_features.shape[1]} tokens, "
                f"but spatial_size={spatial_size} implies {H * W}"
            )

        q = F.normalize(self.query_proj(query_features), dim=-1)
        s = F.normalize(self.spatial_proj(spatial_features), dim=-1)
        attn_logits = torch.einsum("bqd,bpd->bqp", q, s)
        attn = attn_logits.softmax(dim=-1)

        y = (torch.arange(H, device=query_features.device, dtype=query_features.dtype) + 0.5) / H
        x = (torch.arange(W, device=query_features.device, dtype=query_features.dtype) + 0.5) / W
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, H * W, 2)

        center = torch.matmul(attn.to(grid.dtype), grid).to(query_features.dtype)
        spatial_context = torch.matmul(attn.to(spatial_features.dtype), spatial_features)
        fused = torch.cat([query_features, spatial_context], dim=-1)

        cell = torch.tensor([1.0 / W, 1.0 / H], device=query_features.device, dtype=query_features.dtype)
        offset = (self.spatial_offset_embed(fused).sigmoid() - 0.5) * cell
        center = (center + offset).clamp(0.0, 1.0)
        size = self.spatial_size_embed(fused).sigmoid()
        return torch.cat([center, size], dim=-1)
    
    def forward(
        self,
        query_features: torch.Tensor,
        spatial_features: torch.Tensor = None,
        spatial_size = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_features: [B, N_queries, C] decoder output for object queries
            
        Returns:
            Dict with:
                - pred_boxes: [B, N, 4] normalized (cx, cy, w, h)
                - bbox_scores: [B, N] confidence scores
                - class_logits: [B, N, num_classes] raw logits
        """
        # Predict boxes. The spatial path keeps the DETR query interface while
        # grounding bbox centers in satellite patch features.
        if self.use_spatial_conditioning:
            pred_boxes = self._spatial_boxes(query_features, spatial_features, spatial_size)
        else:
            pred_boxes = self.bbox_embed(query_features).sigmoid()  # [B, N, 4]
        
        # Predict class/confidence
        class_logits = self.class_embed(query_features)  # [B, N, num_classes]
        
        # For single-class detection, squeeze to get scores
        if self.num_classes == 1:
            bbox_scores = class_logits.squeeze(-1).sigmoid()  # [B, N]
        else:
            bbox_scores = class_logits.softmax(dim=-1).max(dim=-1)[0]  # [B, N]
        
        return {
            'pred_boxes': pred_boxes,
            'bbox_scores': bbox_scores,
            'class_logits': class_logits,
        }
