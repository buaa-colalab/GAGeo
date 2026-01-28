# BBox head for cross-view localization
# Predicts bounding boxes in satellite view based on geometry prompts

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for query-to-feature attention."""

    def __init__(
        self,
        d_model: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, C]
            key_value: [B, N_kv, C]
            key_padding_mask: [B, N_kv] True for padded positions
        
        Returns:
            [B, N_q, C]
        """
        # Cast to model dtype for mixed precision compatibility
        target_dtype = self.norm1.weight.dtype
        query = query.to(dtype=target_dtype)
        key_value = key_value.to(dtype=target_dtype)
        
        attn_out, _ = self.cross_attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask
        )
        query = self.norm1(query + self.dropout(attn_out))
        query = self.norm2(query + self.ffn(query))
        return query


class BBoxHead(nn.Module):
    """
    Detection head for satellite view localization.
    
    Takes geometry prompt embeddings as queries and cross-attends to
    satellite features to predict bounding boxes.
    
    Args:
        d_model: Model dimension (should be 2*C from VGGT = 2048)
        num_decoder_layers: Number of cross-attention layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output heads
        self.bbox_head = MLP(d_model, d_model, 4, 3)  # (cx, cy, w, h)
        self.score_head = nn.Linear(d_model, 1)

        # Learnable position embedding for satellite features
        self.sat_pos_embed = nn.Parameter(torch.randn(1, 1369 + 5, d_model) * 0.02)

    def forward(
        self,
        prompt_embeddings: torch.Tensor,
        sat_features: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of detection head.
        
        Args:
            prompt_embeddings: [B, N_prompts, C] Sparse prompt embeddings (points/boxes)
            sat_features: [B, P, C] Satellite view features from VGGT
            dense_prompt_embeddings: [B, C, H, W] Dense mask embeddings (optional)
        
        Returns:
            bbox_pred: [B, N_prompts, 4] Predicted boxes (cx, cy, w, h) normalized
            scores: [B, N_prompts] Confidence scores
        """
        B, N_prompts, C = prompt_embeddings.shape
        P = sat_features.shape[1]

        # Add positional embeddings to satellite features
        if P <= self.sat_pos_embed.shape[1]:
            sat_features = sat_features + self.sat_pos_embed[:, :P, :]

        # If dense embeddings provided, flatten and concatenate
        if dense_prompt_embeddings is not None:
            dense_flat = dense_prompt_embeddings.flatten(2).transpose(1, 2)
            sat_features = torch.cat([sat_features, dense_flat], dim=1)

        # Use prompt embeddings as queries
        queries = prompt_embeddings

        # Cross-attention layers
        for layer in self.layers:
            queries = layer(queries, sat_features)

        # Predict boxes and scores
        bbox_pred = self.bbox_head(queries).sigmoid()  # [B, N, 4]
        scores = self.score_head(queries).squeeze(-1).sigmoid()  # [B, N]

        return bbox_pred, scores


class MultiQueryBBoxHead(nn.Module):
    """
    Detection head with learnable object queries (DETR-style).
    
    Args:
        d_model: Model dimension
        num_queries: Number of learnable detection queries
        num_decoder_layers: Number of decoder layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Self-attention for queries
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_decoder_layers)
        ])

        # Cross-attention to satellite features
        self.cross_attn_sat_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_decoder_layers)
        ])

        # Cross-attention to geometry prompts
        self.cross_attn_geo_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_decoder_layers)
        ])

        # Layer norms
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_decoder_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_decoder_layers)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_decoder_layers)])
        self.norm4 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_decoder_layers)])

        # FFN
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_decoder_layers)
        ])

        # Output heads
        self.bbox_head = MLP(d_model, d_model, 4, 3)
        self.score_head = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        geo_embeddings: torch.Tensor,
        sat_features: torch.Tensor,
        geo_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            geo_embeddings: [B, N_geo, C] Geometry prompt embeddings
            sat_features: [B, P, C] Satellite features
            geo_mask: [B, N_geo] Mask for geometry prompts
        
        Returns:
            bbox_pred: [B, num_queries, 4]
            scores: [B, num_queries]
        """
        B = sat_features.shape[0]

        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        for i in range(len(self.self_attn_layers)):
            # Self-attention
            q = queries
            attn_out, _ = self.self_attn_layers[i](q, q, q)
            queries = self.norm1[i](queries + self.dropout(attn_out))

            # Cross-attention to geometry prompts
            attn_out, _ = self.cross_attn_geo_layers[i](
                queries, geo_embeddings, geo_embeddings,
                key_padding_mask=geo_mask
            )
            queries = self.norm2[i](queries + self.dropout(attn_out))

            # Cross-attention to satellite features
            attn_out, _ = self.cross_attn_sat_layers[i](
                queries, sat_features, sat_features
            )
            queries = self.norm3[i](queries + self.dropout(attn_out))

            # FFN
            queries = self.norm4[i](queries + self.ffn[i](queries))

        # Predict
        bbox_pred = self.bbox_head(queries).sigmoid()
        scores = self.score_head(queries).squeeze(-1).sigmoid()

        return bbox_pred, scores
