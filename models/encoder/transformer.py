import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Type, List
from torch import Tensor


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    
    Adapted from SAM2's Attention module.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C


class MLP(nn.Module):
    """
    SAM-style MLP with configurable layers.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: Type[nn.Module] = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = activation()
        self.sigmoid_output = sigmoid_output

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class TwoWayAttentionBlock(nn.Module):
    """
    A transformer block with four layers from SAM:
    1. Self-attention of sparse inputs (prompts)
    2. Cross attention of sparse inputs to dense inputs (prompts -> image)
    3. MLP block on sparse inputs
    4. Cross attention of dense inputs to sparse inputs (image -> prompts)
    
    Args:
        embedding_dim: Channel dimension of the embeddings
        num_heads: Number of heads in the attention layers
        mlp_dim: Hidden dimension of the mlp block
        activation: Activation of the mlp block
        attention_downsample_rate: Downsample rate for attention
        skip_first_layer_pe: Skip the PE on the first layer
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # SAM-style MLP with 2 layers
        self.mlp = MLP(
            input_dim=embedding_dim,
            hidden_dim=mlp_dim,
            output_dim=embedding_dim,
            num_layers=2,
            activation=activation,
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        
        self.skip_first_layer_pe = skip_first_layer_pe
    
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe: Tensor,
        key_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            queries: [B, N_q, C] Sparse prompt embeddings
            keys: [B, N_k, C] Image features (front-view)
            query_pe: [B, N_q, C] Positional encoding for prompts
            key_pe: [B, N_k, C] Positional encoding for image features
        
        Returns:
            queries: [B, N_q, C] Updated prompt embeddings
            keys: [B, N_k, C] Updated image features
        """
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        
        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        
        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        
        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        
        return queries, keys


class TwoWayTransformer(nn.Module):
    """
    A transformer decoder that attends to an input image using
    queries whose positional embedding is supplied.
    
    This is SAM's TwoWayTransformer adapted for cross-view localization.
    Fuses prompt embeddings with front-view features.
    
    Args:
        depth: Number of layers in the transformer
        embedding_dim: Channel dimension for the input embeddings (2048 for VGGT)
        num_heads: Number of heads for multihead attention
        mlp_dim: Channel dimension internal to the MLP block
        activation: Activation to use in the MLP block
        attention_downsample_rate: Downsample rate for attention
    """
    
    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 2048,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
        
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            image_embedding: [B, P, C] Front-view features to attend to
            image_pe: [B, P, C] Positional encoding to add to the image
            point_embedding: [B, N_points, C] Embedding to add to the query points (prompts)
        
        Returns:
            point_embedding: [B, N_points, C] Processed point_embedding
            image_embedding: [B, P, C] Processed image_embedding
        """
        # Prepare queries and keys
        queries = point_embedding
        keys = image_embedding
        
        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        
        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        
        return queries, keys