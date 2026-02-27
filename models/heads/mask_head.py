# SAM-style Mask Prediction Head
# References: SAM2 MaskDecoder (sam2/modeling/sam/mask_decoder.py)
#
# Key design:
# 1. Upscale satellite spatial features: 37x37 -> 74x74 -> 148x148
# 2. Hypernetwork MLP: learnable query token -> dynamic convolution kernel
# 3. Mask prediction: dot product of kernel and upscaled features
# 4. Loss: BCE + Dice (standard for segmentation)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from ..encoder.layer_norm import LayerNorm2d


class MLP(nn.Module):
    """Simple MLP with optional sigmoid output (from SAM2)."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class SAMMaskHead(nn.Module):
    """
    SAM-style mask prediction head.
    
    Takes one or more learnable query token outputs and satellite spatial
    features, predicts segmentation masks on the satellite view.
    
    Architecture (following SAM2 MaskDecoder):
    1. output_upscaling: ConvTranspose2d 37x37 -> 74x74 -> 148x148
    2. output_hypernetwork_mlp: query_token -> dynamic kernel [C//8]
    3. mask = kernel @ upscaled_features -> [1, 148, 148]
    4. Interpolate to output_size (518x518)
    
    Args:
        hidden_dim: Input feature dimension (2048 for Pi3 large)
        output_size: Final mask output size (default 518)
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        output_size: int = 518,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        
        # Upscale spatial features: 37x37 -> 74x74 -> 148x148
        # Following SAM2: ConvTranspose2d with stride 2
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        
        # Shared hypernetwork: each query token -> one dynamic convolution kernel
        self.output_hypernetwork_mlp = MLP(hidden_dim, hidden_dim, hidden_dim // 8, 3)

        # Per-query IoU prediction head
        self.iou_prediction_head = MLP(hidden_dim, hidden_dim // 4, 1, 3, sigmoid_output=True)
    
    def forward(
        self,
        query_tokens: torch.Tensor,
        spatial_features: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Predict segmentation mask.
        
        Args:
            query_tokens: [B, Q, C] or [B, C], learnable query outputs for bbox/mask
            spatial_features: [B, P, C] satellite patch features (P = H*W)
            spatial_size: (H, W) spatial dimensions (37, 37)
        
        Returns:
            Dict with:
                - mask_logits: [B, Q, output_size, output_size] raw logits
                - mask_pred: [B, Q, output_size, output_size] sigmoid probabilities
                - iou_pred: [B, Q] predicted IoU scores
        """
        if query_tokens.dim() == 2:
            query_tokens = query_tokens.unsqueeze(1)
        if query_tokens.dim() != 3:
            raise ValueError(f"query_tokens must be [B, Q, C] or [B, C], got {tuple(query_tokens.shape)}")

        B = query_tokens.shape[0]
        H, W = spatial_size
        C = self.hidden_dim
        
        # Reshape spatial features to 2D: [B, P, C] -> [B, C, H, W]
        src = spatial_features.permute(0, 2, 1).view(B, C, H, W)
        
        # Upscale: [B, C, 37, 37] -> [B, C//8, 148, 148]
        upscaled = self.output_upscaling(src)
        
        # Hypernetwork: query token(s) -> dynamic kernel(s)
        hyper_in = self.output_hypernetwork_mlp(query_tokens)  # [B, Q, C//8]
        
        # Dot product: [B, num_masks, C//8] @ [B, C//8, H*W] -> [B, num_masks, H*W]
        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)  # [B, Q, 148, 148]
        
        # Interpolate to output size
        mask_logits = F.interpolate(
            masks, size=(self.output_size, self.output_size),
            mode='bilinear', align_corners=False,
        )  # [B, Q, 518, 518]
        
        # IoU prediction
        iou_pred = self.iou_prediction_head(query_tokens).squeeze(-1)  # [B, Q]
        
        return {
            'mask_logits': mask_logits,
            'mask_pred': mask_logits.sigmoid(),
            'iou_pred': iou_pred,
        }
