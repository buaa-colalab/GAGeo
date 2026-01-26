# Position head for cross-view localization
# Predicts camera position in satellite view coordinate system
# This is a cross-view matching task: find where the front view camera is located in the satellite image

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PositionHead(nn.Module):
    """
    Predicts camera position in satellite view.
    
    Given front view and satellite view features (already cross-view fused by VGGT),
    predicts the (x, y) position of the camera in the satellite image.
    
    Two prediction modes:
    1. Regression: Directly predict normalized (x, y) coordinates
    2. Heatmap: Predict a probability heatmap over the satellite image
    
    Args:
        d_model: Input feature dimension (2*C from VGGT = 2048)
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        output_mode: 'regression' or 'heatmap'
        heatmap_size: Output heatmap size if mode is 'heatmap'
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_heads: int = 8,
        num_layers: int = 2,
        output_mode: str = 'regression',  # 'regression' or 'heatmap'
        heatmap_size: Tuple[int, int] = (37, 37),  # Same as patch grid
    ):
        super().__init__()
        self.d_model = d_model
        self.output_mode = output_mode
        self.heatmap_size = heatmap_size

        # Normalization
        self.front_norm = nn.LayerNorm(d_model)
        self.sat_norm = nn.LayerNorm(d_model)

        # Cross-attention: front features attend to satellite features
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Self-attention on fused features
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        self.self_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # FFN
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(0.1),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Learnable position query token
        self.pos_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        if output_mode == 'regression':
            # Direct coordinate regression
            self.position_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 2),  # (x, y) normalized to [0, 1]
                nn.Sigmoid(),
            )
        else:
            # Heatmap prediction
            self.heatmap_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, heatmap_size[0] * heatmap_size[1]),
            )

        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        front_features: torch.Tensor,
        sat_features: torch.Tensor,
        return_heatmap: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            front_features: [B, P, C] Front view patch features from VGGT
            sat_features: [B, P, C] Satellite view patch features from VGGT
            return_heatmap: If True, also return attention heatmap for visualization
        
        Returns:
            Dict containing:
                - position: [B, 2] Predicted (x, y) position normalized to [0, 1]
                - confidence: [B] Confidence score
                - heatmap: [B, H, W] Probability heatmap (if output_mode='heatmap' or return_heatmap=True)
        """
        B = front_features.shape[0]

        # Normalize features
        front_feat = self.front_norm(front_features)  # [B, P, C]
        sat_feat = self.sat_norm(sat_features)        # [B, P, C]

        # Prepend position query to front features
        pos_query = self.pos_query.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat([pos_query, front_feat], dim=1)  # [B, 1+P, C]

        # Store attention weights for heatmap
        attn_weights_list = []

        # Cross-view attention layers
        for i in range(len(self.cross_attn_layers)):
            # Cross-attention: front (with pos query) attends to satellite
            attn_out, attn_weights = self.cross_attn_layers[i](
                x, sat_feat, sat_feat,
                need_weights=True,
                average_attn_weights=True,
            )
            x = self.cross_norms[i](x + attn_out)
            attn_weights_list.append(attn_weights[:, 0, :])  # Query token's attention

            # Self-attention on fused features
            attn_out, _ = self.self_attn_layers[i](x, x, x)
            x = self.self_norms[i](x + attn_out)

            # FFN
            x = self.ffn_norms[i](x + self.ffn_layers[i](x))

        # Extract position query output (first token)
        pos_token = x[:, 0]  # [B, C]

        # Predict position
        if self.output_mode == 'regression':
            position = self.position_head(pos_token)  # [B, 2]
        else:
            # Heatmap mode
            heatmap_logits = self.heatmap_head(pos_token)  # [B, H*W]
            heatmap = heatmap_logits.view(B, *self.heatmap_size)  # [B, H, W]
            heatmap_prob = F.softmax(heatmap_logits, dim=-1).view(B, *self.heatmap_size)
            
            # Extract position from heatmap (soft-argmax)
            position = self._soft_argmax(heatmap_prob)  # [B, 2]

        # Predict confidence
        confidence = self.confidence_head(pos_token).squeeze(-1)  # [B]

        result = {
            'position': position,
            'confidence': confidence,
        }

        # Generate attention heatmap for visualization
        if return_heatmap or self.output_mode == 'heatmap':
            if self.output_mode == 'heatmap':
                result['heatmap'] = heatmap_prob
            else:
                # Use attention weights as heatmap
                attn_heatmap = torch.stack(attn_weights_list, dim=0).mean(dim=0)  # [B, P]
                P = attn_heatmap.shape[1]
                H = W = int(P ** 0.5)
                result['heatmap'] = attn_heatmap.view(B, H, W)

        return result

    def _soft_argmax(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Soft-argmax to extract coordinates from heatmap.
        
        Args:
            heatmap: [B, H, W] Probability heatmap (should sum to 1)
        
        Returns:
            coords: [B, 2] Normalized (x, y) coordinates in [0, 1]
        """
        B, H, W = heatmap.shape
        device = heatmap.device

        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        
        # Compute expected coordinates
        y_expected = (heatmap.sum(dim=2) * y_coords).sum(dim=1)  # [B]
        x_expected = (heatmap.sum(dim=1) * x_coords).sum(dim=1)  # [B]

        return torch.stack([x_expected, y_expected], dim=1)  # [B, 2]


class PositionHeadWithCameraToken(nn.Module):
    """
    Alternative PositionHead that uses camera tokens (like CameraHead).
    
    Uses the camera token (index 0) from VGGT output for position prediction,
    following the same pattern as CameraHead.
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model

        # Cross-view fusion
        self.front_norm = nn.LayerNorm(d_model)
        self.sat_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        # Position prediction
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),
        )

        # Confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        front_camera_token: torch.Tensor,
        sat_camera_token: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using camera tokens.
        
        Args:
            front_camera_token: [B, C] Camera token from front view
            sat_camera_token: [B, C] Camera token from satellite view
        
        Returns:
            Dict containing:
                - position: [B, 2] Predicted (x, y) position normalized to [0, 1]
                - confidence: [B] Confidence score
        """
        # Ensure tokens have sequence dimension
        if front_camera_token.dim() == 2:
            front_camera_token = front_camera_token.unsqueeze(1)
        if sat_camera_token.dim() == 2:
            sat_camera_token = sat_camera_token.unsqueeze(1)

        # Cross-view fusion
        front_norm = self.front_norm(front_camera_token)
        sat_norm = self.sat_norm(sat_camera_token)

        attn_out, _ = self.cross_attn(front_norm, sat_norm, sat_norm)
        fused = self.cross_norm(front_camera_token + attn_out)
        fused = self.fusion_norm(fused + self.fusion_ffn(fused))

        # Squeeze sequence dimension
        fused = fused.squeeze(1)  # [B, C]

        # Predict
        position = self.position_head(fused)
        confidence = self.confidence_head(fused).squeeze(-1)

        return {
            'position': position,
            'confidence': confidence,
        }
