# Pi3 Backbone for Cross-View Localization
# Wraps Pi3's encoder and decoder to extract features for our task

import torch
import torch.nn as nn
from functools import partial
from typing import List, Tuple, Optional

from ..dinov2.layers import Mlp
from ..layers.pos_embed import RoPE2D, PositionGetter
from ..layers.block import BlockRope
from ..layers.attention import FlashAttentionRope
from ..dinov2.hub.backbones import dinov2_vitl14_reg

class Pi3Backbone(nn.Module):
    """
    Pi3 Backbone for feature extraction.
    
    Uses DINOv2 encoder + Pi3's decoder blocks to extract features.
    Unlike VGGT, Pi3 doesn't require a fixed reference frame - all views are treated equally.
    
    Output: [B, N_views, num_patches, embed_dim] features
    
    Args:
        pos_type: Positional encoding type (default 'rope100')
        decoder_size: Decoder size ('small', 'base', 'large')
        img_size: Input image size (default 518)
        patch_size: Patch size (default 14)
    """
    
    def __init__(
        self,
        pos_type: str = 'rope100',
        decoder_size: str = 'large',
        img_size: int = 518,
        patch_size: int = 14,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size  # 37
        self.num_patches = self.num_patches_per_side ** 2  # 1369
        
        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        del self.encoder.mask_token
        
        # ----------------------
        #  Positional Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError(f"Position type {pos_type} not supported")
        
        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024
        
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError(f"Decoder size {decoder_size} not supported")
        
        self.dec_embed_dim = dec_embed_dim
        self.output_dim = 2 * dec_embed_dim  # Concatenate last two layers
        
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)
        ])
        
        # ----------------------
        #     Register tokens
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)
        
        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
    
    def decode(self, hidden: torch.Tensor, N: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply decoder blocks with alternating local/global attention.
        
        Args:
            hidden: [B*N, hw, C] encoded features
            N: number of views
            H, W: image height and width
            
        Returns:
            features: [B*N, hw, 2*C] concatenated features from last two layers
            pos: [B*N, hw, 2] position encoding
        """
        BN, hw, _ = hidden.shape
        B = BN // N
        
        final_output = []
        hidden = hidden.reshape(B * N, hw, -1)
        
        # Add register tokens
        register_token = self.register_token.to(hidden.device).repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]
        
        # Position encoding
        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)
        
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        # Apply decoder blocks with alternating local/global attention
        for i in range(len(self.decoder)):
            blk = self.decoder[i]
            
            if i % 2 == 0:
                # Local attention (within each view)
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                # Global attention (across all views)
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)
            
            hidden = blk(hidden, xpos=pos)
            
            # Collect last two layers
            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))
        
        # Concatenate last two layers
        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Extract features from multiple views.
        
        Args:
            images: [B, N, 3, H, W] input images (N views)
            
        Returns:
            features: [B, N, num_patches + register_tokens, 2*C] features
            patch_start_idx: index where patch tokens start (after register tokens)
        """
        # Normalize
        images = (images - self.image_mean) / self.image_std
        
        B, N, _, H, W = images.shape
        
        # Encode with DINOv2
        images_flat = images.reshape(B * N, 3, H, W)
        
        # 确保输入类型与模型权重一致（解决 bf16 混合精度问题）
        target_dtype = self.image_mean.dtype
        if images_flat.dtype != target_dtype:
            images_flat = images_flat.to(target_dtype)
        
        hidden = self.encoder(images_flat, is_training=True)
        
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        
        # Decode with Pi3 decoder
        features, pos = self.decode(hidden, N, H, W)
        
        # Reshape to [B, N, tokens, C]
        features = features.reshape(B, N, -1, self.output_dim)
        
        return features, self.patch_start_idx
    
    def get_front_sat_features(self, front_view: torch.Tensor, satellite_view: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method to extract features from front and satellite views.
        
        Args:
            front_view: [B, 3, H, W] front view image
            satellite_view: [B, 3, H, W] satellite view image
            
        Returns:
            front_patch_features: [B, num_patches, C]
            sat_patch_features: [B, num_patches, C]
            front_camera_token: [B, C] (first register token)
            sat_camera_token: [B, C] (first register token)
        """
        B = front_view.shape[0]
        
        # Stack views: [B, 2, 3, H, W]
        images = torch.stack([satellite_view, front_view], dim=1)
        
        # Forward
        features, patch_start_idx = self.forward(images)
        # features: [B, 2, tokens, C]
        
        sat_features = features[:, 0]    # [B, tokens, C]
        front_features = features[:, 1]  # [B, tokens, C]
        
        # Extract patch tokens (remove register tokens)
        front_patch_features = front_features[:, patch_start_idx:]  # [B, num_patches, C]
        sat_patch_features = sat_features[:, patch_start_idx:]      # [B, num_patches, C]
        
        # Extract camera tokens (first register token)
        front_camera_token = front_features[:, 0]  # [B, C]
        sat_camera_token = sat_features[:, 0]      # [B, C]
        
        return front_patch_features, sat_patch_features, front_camera_token, sat_camera_token


def load_pi3_weights(model: Pi3Backbone, checkpoint_path: str, strict: bool = False):
    """
    Load Pi3 pretrained weights into backbone.
    
    Args:
        model: Pi3Backbone instance
        checkpoint_path: Path to Pi3 checkpoint (.pt, .pth, or .safetensors)
        strict: Whether to strictly enforce matching keys
    """
    # Support safetensors format
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    # Filter to only encoder and decoder weights
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.') or k.startswith('decoder.') or k.startswith('register_token'):
            filtered_state_dict[k] = v
        if k.startswith('rope.') or k.startswith('position_getter.'):
            filtered_state_dict[k] = v
    
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=strict)
    
    print(f"Loaded Pi3 weights from {checkpoint_path}")
    print(f"  Loaded keys: {len(filtered_state_dict)}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    
    return missing, unexpected
