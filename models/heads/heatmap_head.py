# Heatmap Prediction Head using Mask2Former-style Dynamic Convolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class HeatmapHead(nn.Module):
    """
    Mask2Former-style heatmap prediction head.
    
    Core idea: dot product between query embeddings and spatial features.
    Simplified implementation without unnecessary dimension reduction.
    
    Args:
        hidden_dim: Input feature dimension from decoder
        output_size: Output heatmap size
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        output_size: int = 518,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        
        # Project queries to combination weights (for weighted sum of multiple queries)
        self.query_to_weight = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        query_features: torch.Tensor,
        spatial_features: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_features: [B, N_loc, C] decoder output for location queries
            spatial_features: [B, P, C] satellite patch features (P = H*W)
            spatial_size: (H, W) spatial dimensions of features
            
        Returns:
            Dict with:
                - heatmap: [B, output_size, output_size] probability distribution
                - heatmap_logits: [B, H, W] raw logits before upsampling
                - position: [B, 2] extracted (x, y) position
        """
        B = query_features.shape[0]
        H_feat, W_feat = spatial_size
        
        # Weighted combination of location queries
        query_weights = self.query_to_weight(query_features).softmax(dim=1)  # [B, N_loc, 1]
        combined_query = (query_features * query_weights).sum(dim=1)  # [B, C]
        
        # Reshape spatial features: [B, P, C] -> [B, C, H, W]
        spatial_2d = spatial_features.permute(0, 2, 1).view(B, -1, H_feat, W_feat)
        
        # Dot product: [B, C] x [B, C, H, W] -> [B, H, W]
        heatmap_logits = torch.einsum('bc,bchw->bhw', combined_query, spatial_2d)
        
        # Upsample to output size
        heatmap_upsampled = F.interpolate(
            heatmap_logits.unsqueeze(1),
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=True,
        ).squeeze(1)  # [B, output_size, output_size]
        
        # Apply softmax to get probability distribution
        heatmap_flat = heatmap_upsampled.view(B, -1)
        heatmap_prob = F.softmax(heatmap_flat, dim=-1).view(B, self.output_size, self.output_size)
        
        # Extract position using soft-argmax
        position = self._soft_argmax(heatmap_prob)  # [B, 2]
        
        return {
            'heatmap': heatmap_prob,
            'heatmap_logits': heatmap_logits,
            'position': position,
        }
    
    def _soft_argmax(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Differentiable soft-argmax to extract 2D coordinates from heatmap."""
        B, H, W = heatmap.shape
        device = heatmap.device
        
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        
        y_expected = (heatmap.sum(dim=2) * y_coords).sum(dim=1)
        x_expected = (heatmap.sum(dim=1) * x_coords).sum(dim=1)
        
        return torch.stack([x_expected, y_expected], dim=1)
