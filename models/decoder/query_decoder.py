# Stage 2: Unified Query Decoder for cross-view localization
# Uses TransformerDecoder for cross-attention to satellite features

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .detr import TransformerDecoder
from .pe_sin import PositionEmbeddingSine


class UnifiedQueryDecoder(nn.Module):
    """
    Stage 2: View Conditioning
    
    Object/Location Queries + Intent Features -> TransformerDecoder -> cross-attend to satellite features
    
    Flow:
    1. Concat Intent Features with Object/Location Queries as tgt
    2. TransformerDecoder: queries cross-attend to satellite features (memory)
    3. Output: obj_features, loc_features
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
        
        # Object queries
        self.object_queries = nn.Embedding(num_object_queries, hidden_dim)
        self.object_query_pos = nn.Embedding(num_object_queries, hidden_dim)
        
        # Location queries
        self.location_queries = nn.Embedding(num_location_queries, hidden_dim)
        self.location_query_pos = nn.Embedding(num_location_queries, hidden_dim)
        
        # Memory positional encoding (sinusoidal)
        self.memory_pos_embed = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2,
            normalize=True,
        )
        
        # TransformerDecoder: cross-attention with satellite features
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
        intent_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            memory: [B, P, C] satellite patch features
            intent_features: [B, N_intent, C] intent features from Stage 1
            
        Returns:
            Dict with obj_features, loc_features, decoder_out
        """
        B = memory.shape[0]
        device = memory.device
        dtype = memory.dtype
        
        # Prepare queries
        obj_q = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        obj_q_pos = self.object_query_pos.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        loc_q = self.location_queries.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        loc_q_pos = self.location_query_pos.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        
        # Concat: Intent Features + Object Queries + Location Queries
        queries = torch.cat([intent_features, obj_q, loc_q], dim=1)
        query_pos = torch.cat([
            torch.zeros_like(intent_features),  # No pos for intent
            obj_q_pos,
            loc_q_pos,
        ], dim=1)
        
        # Memory positional encoding
        memory_pos = self.memory_pos_embed(self.spatial_size, device=device)
        memory_pos = memory_pos.flatten(1).permute(1, 0).to(dtype)
        memory_pos = memory_pos.unsqueeze(0).expand(B, -1, -1)
        
        # TransformerDecoder forward
        out = self.decoder(
            tgt=queries,
            memory=memory,
            pos=memory_pos,
            query_pos=query_pos,
        )
        
        if out.dim() == 4:
            out = out[-1]
        
        # Split outputs (skip intent features)
        n_intent = intent_features.shape[1]
        obj_features = out[:, n_intent:n_intent + self.num_object_queries, :]
        loc_features = out[:, n_intent + self.num_object_queries:, :]
        
        return {
            'obj_features': obj_features,
            'loc_features': loc_features,
            'decoder_out': out,
        }
