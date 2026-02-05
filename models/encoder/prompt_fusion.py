# Stage 1: Intent Formation Module
# Uses learnable Intent Queries + TransformerDecoder to extract target info from front-view
# Reuses detr.py's TransformerDecoder for cross-attention

import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch import Tensor

from ..decoder.detr import TransformerDecoder
from ..decoder.pe_sin import PositionEmbeddingSine


class IntentFormation(nn.Module):
    """
    Stage 1: Intent Formation
    
    Uses learnable Intent Queries to extract target information from front-view features.
    Reuses TransformerDecoder from detr.py for cross-attention.
    
    Flow:
    1. Concat prompt tokens (sparse + dense) with Intent Queries
    2. TransformerDecoder: queries attend to front-view features (memory)
    3. Output: Z_intent [B, num_intent_queries, C]
    """
    
    def __init__(
        self,
        embed_dim: int = 2048,
        num_intent_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        spatial_size: Tuple[int, int] = (37, 37),
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_intent_queries = num_intent_queries
        self.spatial_size = spatial_size
        
        # Learnable Intent Queries
        self.intent_queries = nn.Embedding(num_intent_queries, embed_dim)
        self.intent_query_pos = nn.Embedding(num_intent_queries, embed_dim)
        
        # Memory positional encoding (sinusoidal, for front-view features)
        self.memory_pos_embed = PositionEmbeddingSine(
            num_pos_feats=embed_dim // 2,
            normalize=True,
        )
        
        # TransformerDecoder: Intent Queries cross-attend to front-view features
        self.decoder = TransformerDecoder(
            d_model=embed_dim,
            nhead=num_heads,
            num_decoder_layers=num_layers,
            dim_feedforward=embed_dim,
            dropout=dropout,
            normalize_before=False,
            return_intermediate=False,
        )
    
    def forward(
        self,
        front_features: Tensor,
        sparse_embeddings: Optional[Tensor] = None,
        dense_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            front_features: [B, P, C] Front-view patch features
            sparse_embeddings: [B, N_sparse, C] Sparse prompt embeddings (points/boxes)
            dense_embeddings: [B, C, H, W] Dense prompt embeddings (masks)
        
        Returns:
            intent_features: [B, num_intent_queries, C]
        """
        B = front_features.shape[0]
        device = front_features.device
        dtype = front_features.dtype
        
        # Prepare Intent Queries
        intent_q = self.intent_queries.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        intent_q_pos = self.intent_query_pos.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        
        # Build query tokens: Intent Queries + Prompt Tokens
        query_list = [intent_q]
        query_pos_list = [intent_q_pos]
        
        if sparse_embeddings is not None and sparse_embeddings.shape[1] > 0:
            query_list.append(sparse_embeddings)
            # Zero positional encoding for prompt tokens
            query_pos_list.append(torch.zeros_like(sparse_embeddings))
        
        if dense_embeddings is not None:
            dense_flat = dense_embeddings.flatten(2).transpose(1, 2)  # [B, H*W, C]
            query_list.append(dense_flat)
            query_pos_list.append(torch.zeros_like(dense_flat))
        
        queries = torch.cat(query_list, dim=1)
        query_pos = torch.cat(query_pos_list, dim=1)
        
        # Memory positional encoding
        memory_pos = self.memory_pos_embed(self.spatial_size, device=device)
        memory_pos = memory_pos.flatten(1).permute(1, 0).to(dtype)
        memory_pos = memory_pos.unsqueeze(0).expand(B, -1, -1)
        
        # TransformerDecoder forward
        out = self.decoder(
            tgt=queries,
            memory=front_features,
            pos=memory_pos,
            query_pos=query_pos,
        )
        
        if out.dim() == 4:
            out = out[-1]
        
        # Extract only Intent Queries
        intent_features = out[:, :self.num_intent_queries, :]
        
        return intent_features


class PromptFusionWithDense(nn.Module):
    """
    Stage 1 wrapper for backward compatibility.
    """
    
    def __init__(
        self,
        embedding_dim: int = 2048,
        num_intent_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 3,
        image_embedding_size: Tuple[int, int] = (37, 37),
        dropout: float = 0.1,
        **kwargs,  # Ignore legacy params
    ):
        super().__init__()
        
        self.intent_formation = IntentFormation(
            embed_dim=embedding_dim,
            num_intent_queries=num_intent_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            spatial_size=image_embedding_size,
        )
    
    def forward(
        self,
        image_features: Tensor,
        sparse_embeddings: Tensor,
        dense_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        """
        Returns:
            intent_features: [B, num_intent_queries, C]
        """
        intent_features = self.intent_formation(
            front_features=image_features,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
        )
        
        
        return intent_features
