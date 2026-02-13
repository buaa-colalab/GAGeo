# SAM-style Prompt Fusion Module
# Replaces learnable Intent Queries with SAM's TwoWayTransformer approach:
#   - Prompt tokens (sparse + dense) cross-attend to front-view features
#   - Front-view features cross-attend back to prompt tokens
#   - Output: updated front-view features (prompt-conditioned)

import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch import Tensor

from .transformer import TwoWayTransformer
from .pe_random import PositionEmbeddingRandom


class SAMStylePromptFusion(nn.Module):
    """
    SAM-style Prompt Fusion: uses TwoWayTransformer to fuse prompts with front-view features.
    
    Instead of learnable Intent Queries:
    1. Prompt tokens (sparse embeddings) act as queries
    2. Front-view patch features act as keys/values
    3. Through bidirectional cross-attention, front-view features become prompt-aware
    4. The updated front-view features replace intent_features for the decoder
    
    Flow:
        sparse_embeddings (Q) <-> front_patch_features (KV)  [bidirectional via TwoWayTransformer]
        -> updated front_patch_features are used as intent_features input to UnifiedQueryDecoder
    
    Args:
        embedding_dim: Feature dimension (2048 for Pi3 large)
        num_heads: Number of attention heads
        depth: Number of TwoWayAttentionBlock layers
        mlp_dim: Hidden dimension in MLP blocks
        image_embedding_size: Spatial size of image features (H_patches, W_patches)
        attention_downsample_rate: Downsample rate for attention efficiency
    """
    
    def __init__(
        self,
        embedding_dim: int = 2048,
        num_heads: int = 8,
        depth: int = 2,
        mlp_dim: int = 2048,
        image_embedding_size: Tuple[int, int] = (37, 37),
        attention_downsample_rate: int = 2,
        dropout: float = 0.1,
        **kwargs,  # Ignore legacy params like num_intent_queries, num_layers
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.image_embedding_size = image_embedding_size
        
        # Positional encoding for front-view features (same as SAM prompt encoder)
        self.pe_layer = PositionEmbeddingRandom(embedding_dim // 2)
        
        # SAM's TwoWayTransformer for bidirectional prompt <-> image fusion
        self.transformer = TwoWayTransformer(
            depth=depth,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            attention_downsample_rate=attention_downsample_rate,
        )
        
        # Optional: dense embedding projection (for mask prompts)
        self.dense_proj = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embedding_dim // 4, embedding_dim, kernel_size=1),
        )
    
    def forward(
        self,
        image_features: Tensor,
        sparse_embeddings: Tensor,
        dense_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """
        SAM-style prompt fusion.
        
        Args:
            image_features: [B, P, C] Front-view patch features (P = H*W patches)
            sparse_embeddings: [B, N_sparse, C] Sparse prompt embeddings (points/boxes)
            dense_embeddings: [B, C, H, W] Dense prompt embeddings (masks), optional
            
        Returns:
            fused_features: [B, P, C] Prompt-conditioned front-view features
                (used as intent_features input to UnifiedQueryDecoder)
        """
        B = image_features.shape[0]
        device = image_features.device
        dtype = image_features.dtype
        
        # Get positional encoding for image features
        # pe_layer returns [C, H, W], we need [B, P, C]
        image_pe = self.pe_layer(self.image_embedding_size)  # [C, H, W]
        image_pe = image_pe.flatten(1).permute(1, 0).to(dtype)  # [P, C]
        image_pe = image_pe.unsqueeze(0).expand(B, -1, -1)  # [B, P, C]
        
        # If dense embeddings exist, add them to image features
        keys = image_features
        if dense_embeddings is not None:
            # dense_embeddings: [B, C, H, W] -> project and add to image features
            dense_proj = self.dense_proj(dense_embeddings)  # [B, C, H, W]
            dense_flat = dense_proj.flatten(2).transpose(1, 2)  # [B, P, C]
            keys = keys + dense_flat
        
        # TwoWayTransformer: prompt tokens <-> image features
        # queries = sparse_embeddings (prompt tokens)
        # keys = front-view features
        # Returns: (updated_queries, updated_keys)
        updated_prompts, updated_image = self.transformer(
            image_embedding=keys,
            image_pe=image_pe,
            point_embedding=sparse_embeddings,
        )
        # updated_image: [B, P, C] - front-view features conditioned on prompts
        
        return updated_image
