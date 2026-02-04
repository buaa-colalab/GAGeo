# Unified Query Decoder for DETR-style cross-view localization

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .detr import TransformerDecoder
from .pe_sin import PositionEmbeddingSine


class UnifiedQueryDecoder(nn.Module):
    """
    Unified DETR-style decoder that handles both object and location queries.
    
    Encapsulates:
    - Query embeddings (object + location)
    - Positional encodings for queries and memory
    - Target guidance injection
    - Decoder forward pass
    - Output splitting
    
    Args:
        hidden_dim: Feature dimension
        num_heads: Number of attention heads
        num_decoder_layers: Number of decoder layers
        num_object_queries: Number of object queries for bbox detection
        num_location_queries: Number of location queries for heatmap
        spatial_size: Spatial size of memory features (H, W)
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        num_object_queries: int = 10,
        num_location_queries: int = 16,
        spatial_size: Tuple[int, int] = (37, 37),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_object_queries = num_object_queries
        self.num_location_queries = num_location_queries
        self.spatial_size = spatial_size
        
        # Object queries: learnable embeddings for bbox detection
        self.object_queries = nn.Embedding(num_object_queries, hidden_dim)
        self.object_query_pos = nn.Embedding(num_object_queries, hidden_dim)
        
        # Location queries: learnable embeddings for heatmap
        self.location_queries = nn.Embedding(num_location_queries, hidden_dim)
        self.location_query_pos = nn.Embedding(num_location_queries, hidden_dim)
        
        # Target guidance projection
        self.target_guidance_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Memory positional encoding (sinusoidal)
        self.memory_pos_embed = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2,
            normalize=True,
        )
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            d_model=hidden_dim,
            nhead=num_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            normalize_before=False,
            return_intermediate=False,
        )
    
    def forward(
        self,
        memory: torch.Tensor,
        target_guidance: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the unified decoder.
        
        Args:
            memory: [B, P, C] satellite patch features (memory for cross-attention)
            target_guidance: [B, C] target guidance vector from prompt fusion
            
        Returns:
            Dict with:
                - obj_features: [B, N_obj, C] object query outputs
                - loc_features: [B, N_loc, C] location query outputs
                - decoder_out: [B, N_total, C] full decoder output
        """
        B = memory.shape[0]
        device = memory.device
        target_dtype = memory.dtype
        # Prepare object queries
        obj_queries = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1).to(target_dtype)  # [B, N_obj, C]
        obj_query_pos = self.object_query_pos.weight.unsqueeze(0).expand(B, -1, -1).to(target_dtype)
        
        # Prepare location queries
        loc_queries = self.location_queries.weight.unsqueeze(0).expand(B, -1, -1).to(target_dtype)  # [B, N_loc, C]
        loc_query_pos = self.location_query_pos.weight.unsqueeze(0).expand(B, -1, -1).to(target_dtype)
        
        # Add target guidance to query content (not position)
        target_proj = self.target_guidance_proj(target_guidance.to(target_dtype))  # [B, C]
        obj_queries = obj_queries + target_proj.unsqueeze(1)
        loc_queries = loc_queries + target_proj.unsqueeze(1)
        
        # Concatenate queries and positional encodings
        unified_queries = torch.cat([obj_queries, loc_queries], dim=1)  # [B, N_total, C]
        unified_query_pos = torch.cat([obj_query_pos, loc_query_pos], dim=1)
        
        # Memory positional encoding
        memory_pos = self.memory_pos_embed(self.spatial_size, device=device)  # [C, H, W]
        memory_pos = memory_pos.flatten(1).permute(1, 0).to(target_dtype)  # [P, C]
        memory_pos = memory_pos.unsqueeze(0).expand(B, -1, -1)  # [B, P, C]
        
        # Decoder forward
        decoder_out = self.decoder(
            tgt=unified_queries,
            memory=memory,
            pos=memory_pos,
            query_pos=unified_query_pos,
        )
        
        # Handle output shape (may have intermediate layers dimension)
        if decoder_out.dim() == 4:
            decoder_out = decoder_out[-1]  # Take last layer: [B, N_total, C]
        
        # Split outputs
        obj_features = decoder_out[:, :self.num_object_queries, :]
        loc_features = decoder_out[:, self.num_object_queries:, :]
        
        return {
            'obj_features': obj_features,
            'loc_features': loc_features,
            'decoder_out': decoder_out,
        }
