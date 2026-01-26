# Mask head for cross-view localization
# Predicts object masks in satellite view using DPT-style multi-scale feature fusion
# Adapted from VGGT's dpt_head.py for cross-view dense prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features: int, activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block for combining multi-scale features."""

    def __init__(
        self,
        features: int,
        activation: nn.Module = nn.ReLU(inplace=True),
        upsample: bool = True,
    ):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.resConfUnit2 = ResidualConvUnit(features, activation)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.upsample = upsample

    def forward(
        self, 
        x: torch.Tensor, 
        residual: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if residual is not None:
            x = x + self.resConfUnit1(residual)
        
        x = self.resConfUnit2(x)
        
        if self.upsample:
            if target_size is not None:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            else:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.out_conv(x)
        return x


class MaskHead(nn.Module):
    """
    Mask prediction head using DPT-style multi-scale feature fusion.
    
    Takes multi-layer features from VGGT and fuses them to produce
    high-resolution dense mask predictions for cross-view localization.
    
    Args:
        d_model: Input feature dimension (2*C from VGGT = 2048)
        patch_size: Patch size used in VGGT (default 14)
        num_classes: Number of output classes (1 for binary mask)
        features: Intermediate feature channels for fusion
        intermediate_layer_idx: Which VGGT layers to use for multi-scale fusion
    """

    def __init__(
        self,
        d_model: int = 2048,
        patch_size: int = 14,
        num_classes: int = 1,
        features: int = 256,
        intermediate_layer_idx: List[int] = [5, 11, 17, 23],
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.features = features
        self.intermediate_layer_idx = intermediate_layer_idx
        self.num_layers = len(intermediate_layer_idx)

        # Layer normalization for input features
        self.norm = nn.LayerNorm(d_model)

        # Project each layer's features to intermediate dimension
        self.projects = nn.ModuleList([
            nn.Conv2d(d_model, features, kernel_size=1, stride=1, padding=0)
            for _ in range(self.num_layers)
        ])

        # Resize layers for different scales
        # Layer 0 (shallow): 4x upsample
        # Layer 1: 2x upsample
        # Layer 2: identity
        # Layer 3 (deep): 2x downsample then will be upsampled in fusion
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(features, features, kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(features, features, kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1),
        ])

        # Feature fusion blocks (from deep to shallow)
        self.refinenet4 = FeatureFusionBlock(features, upsample=True)
        self.refinenet3 = FeatureFusionBlock(features, upsample=True)
        self.refinenet2 = FeatureFusionBlock(features, upsample=True)
        self.refinenet1 = FeatureFusionBlock(features, upsample=False)

        # Prompt conditioning: cross-attention to incorporate geometry prompts
        self.prompt_proj = nn.Linear(d_model, features)
        self.prompt_attn = nn.MultiheadAttention(
            features, num_heads=8, dropout=0.1, batch_first=True
        )
        self.prompt_norm = nn.LayerNorm(features)

        # Output layers
        self.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
        prompt_embeddings: Optional[torch.Tensor] = None,
        target_size: Tuple[int, int] = (518, 518),
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            aggregated_tokens_list: List of [B, S, P_total, C] features from VGGT layers
                                   S=2 for front+satellite views
            patch_start_idx: Index where patch tokens start (skip camera/register tokens)
            prompt_embeddings: [B, N, C] Geometry prompt embeddings for conditioning
            target_size: Output mask size (H, W)
        
        Returns:
            masks: [B, num_classes, H, W] Predicted mask logits
        """
        # Get satellite view features from each layer
        # aggregated_tokens_list[i] shape: [B, S, P_total, C]
        # We want satellite view (index 1) patch tokens only
        
        B = aggregated_tokens_list[0].shape[0]
        
        # Extract and process features from each intermediate layer
        layer_features = []
        for i, layer_idx in enumerate(self.intermediate_layer_idx):
            # Get satellite view features: [B, P, C]
            x = aggregated_tokens_list[layer_idx][:, 1, patch_start_idx:]
            
            # Normalize
            x = self.norm(x)
            
            # Reshape to spatial: [B, C, H, W]
            P = x.shape[1]
            H_patch = W_patch = int(P ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, self.d_model, H_patch, W_patch)
            
            # Project to intermediate features
            x = self.projects[i](x)
            
            # Resize to appropriate scale
            x = self.resize_layers[i](x)
            
            layer_features.append(x)

        # Multi-scale feature fusion (from deep to shallow)
        layer_1, layer_2, layer_3, layer_4 = layer_features

        # Fusion: deep -> shallow
        out = self.refinenet4(layer_4, target_size=layer_3.shape[2:])
        out = self.refinenet3(out, layer_3, target_size=layer_2.shape[2:])
        out = self.refinenet2(out, layer_2, target_size=layer_1.shape[2:])
        out = self.refinenet1(out, layer_1)

        # Condition on geometry prompts if provided
        if prompt_embeddings is not None:
            # Project prompts
            prompt_proj = self.prompt_proj(prompt_embeddings)  # [B, N, features]
            
            # Flatten spatial features for attention
            B, C, H, W = out.shape
            out_flat = out.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # Cross-attention: spatial features attend to prompts
            attn_out, _ = self.prompt_attn(out_flat, prompt_proj, prompt_proj)
            out_flat = self.prompt_norm(out_flat + attn_out)
            
            # Reshape back to spatial
            out = out_flat.transpose(1, 2).reshape(B, C, H, W)

        # Output convolutions
        out = self.output_conv1(out)
        out = F.relu(out, inplace=True)
        out = self.output_conv2(out)

        # Interpolate to target size
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

        return out

    def forward_single_layer(
        self,
        sat_features: torch.Tensor,
        prompt_embeddings: Optional[torch.Tensor] = None,
        target_size: Tuple[int, int] = (518, 518),
    ) -> torch.Tensor:
        """
        Simplified forward using only last layer features.
        For backward compatibility with existing code.
        
        Args:
            sat_features: [B, P, C] Satellite features from last VGGT layer
            prompt_embeddings: [B, N, C] Geometry prompt embeddings
            target_size: Output mask size
        
        Returns:
            masks: [B, num_classes, H, W]
        """
        B, P, C = sat_features.shape
        H_patch = W_patch = int(P ** 0.5)

        # Normalize and reshape
        x = self.norm(sat_features)
        x = x.permute(0, 2, 1).reshape(B, C, H_patch, W_patch)

        # Project
        x = self.projects[-1](x)  # Use last layer's projection

        # Simple upsampling without multi-scale fusion
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        # Condition on prompts
        if prompt_embeddings is not None:
            prompt_proj = self.prompt_proj(prompt_embeddings)
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)
            attn_out, _ = self.prompt_attn(x_flat, prompt_proj, prompt_proj)
            x_flat = self.prompt_norm(x_flat + attn_out)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)

        # Output
        x = self.output_conv1(x)
        x = F.relu(x, inplace=True)
        x = self.output_conv2(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x
