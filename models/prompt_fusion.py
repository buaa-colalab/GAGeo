# SAM-style Prompt Fusion Module
# Fuses prompt embeddings with front-view features using SAM's two-way transformer approach
# Based on SAM2's TwoWayTransformer implementation

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Type, List
from torch import Tensor
from .encoder import TwoWayTransformer



class PromptFusionWithDense(nn.Module):
    """
    Prompt fusion module following SAM's approach.
    
    SAM's logic:
    1. Sparse prompts: TwoWayTransformer for bidirectional attention with image
    2. Dense prompts: Direct addition to image features (src = src + dense)
    
    Args:
        embedding_dim: Channel dimension (2048 for VGGT output)
        num_heads: Number of attention heads
        depth: Number of transformer layers
        mlp_dim: MLP hidden dimension
        image_embedding_size: Spatial size of image features (H, W)
    """
    
    def __init__(
        self,
        embedding_dim: int = 2048,
        num_heads: int = 8,
        depth: int = 2,
        mlp_dim: int = 2048,
        image_embedding_size: Tuple[int, int] = (37, 37),
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.image_embedding_size = image_embedding_size
        
        # Two-way transformer for sparse prompt fusion
        self.transformer = TwoWayTransformer(
            depth=depth,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation=activation,
            attention_downsample_rate=attention_downsample_rate,
        )
        
        # Learnable positional encoding for image features
        H, W = image_embedding_size
        self.image_pe = nn.Parameter(torch.randn(1, H * W, embedding_dim) * 0.02)
        
        # Output projection for target guidance
        self.target_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(
        self,
        image_features: Tensor,
        sparse_embeddings: Tensor,
        dense_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Fuse prompts with image features (SAM-style).
        
        Dimensions are guaranteed by upstream:
        - image_features: [B, 1369, 2048] from VGGT (37x37 patches)
        - sparse_embeddings: [B, N_sparse, 2048] from PromptEncoder
        - dense_embeddings: [B, 2048, 37, 37] from PromptEncoder (already 37x37)
        
        Returns:
            fused_sparse: [B, N_sparse, C] Target-aware sparse embeddings
            fused_image: [B, 1369, C] Prompt-guided image features
            target_guidance: [B, C] Pooled target guidance vector
        """
        B, P, C = image_features.shape
        
        # SAM Step 1: Add dense embeddings to image features (direct addition)
        if dense_embeddings is not None:
            # dense_embeddings: [B, C, 37, 37] -> [B, 1369, C]
            dense_flat = dense_embeddings.flatten(2).transpose(1, 2)
            image_features = image_features + dense_flat
        
        # SAM Step 2: TwoWayTransformer for sparse prompt fusion
        image_pe = self.image_pe.expand(B, -1, -1)
        
        if sparse_embeddings.shape[1] > 0:
            fused_sparse, fused_image = self.transformer(
                image_embedding=image_features,
                image_pe=image_pe,
                point_embedding=sparse_embeddings,
            )
        else:
            fused_sparse = sparse_embeddings
            fused_image = image_features
        
        # Compute target guidance (attention-weighted pooling)
        if fused_sparse.shape[1] > 0:
            attn = torch.bmm(fused_sparse, fused_image.transpose(1, 2))  # [B, N, P]
            attn = F.softmax(attn / math.sqrt(C), dim=-1)
            attended = torch.bmm(attn, fused_image)  # [B, N, C]
            target_guidance = attended.mean(dim=1)
        else:
            target_guidance = fused_image.mean(dim=1)
        
        target_guidance = self.target_proj(target_guidance)
        
        return fused_sparse, fused_image, target_guidance
