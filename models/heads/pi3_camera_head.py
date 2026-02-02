# Pi3-style Camera Head for Cross-View Localization
# Uses patch tokens (not camera tokens) with TransformerDecoder + ResConvBlock
# Outputs 4x4 pose matrix (absolute pose: cam2world)
#
# Key differences from VGGT-style camera head:
# - Uses all patch tokens instead of single camera token
# - TransformerDecoder processes features before pose prediction
# - Outputs full SE(3) pose matrix, not just quaternion

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy
from typing import Dict, Optional, Tuple

from ..layers.transformer_head import TransformerDecoder
from ..layers.camera_head import ResConvBlock, CameraHead
from ..layers.pos_embed import RoPE2D, PositionGetter

class Pi3CameraHead(nn.Module):
    """
    Pi3-style Camera Head for cross-view localization.
    
    Uses patch tokens from both views, processes them through a TransformerDecoder,
    then predicts relative pose between front and satellite views.
    
    Key features:
    - Uses all patch tokens (not just camera token)
    - TransformerDecoder with RoPE positional encoding
    - Outputs 4x4 SE(3) pose matrix
    - Extracts yaw from rotation matrix for supervision
    
    Args:
        in_dim: Input feature dimension (2048 for Pi3 large)
        dec_embed_dim: Decoder embedding dimension
        dec_num_heads: Number of attention heads
        out_dim: Output dimension before camera head
        depth: Number of transformer decoder layers
        patch_size: Patch size for position encoding
        rope_freq: RoPE frequency
    """
    
    def __init__(
        self,
        in_dim: int = 2048,
        dec_embed_dim: int = 1024,
        dec_num_heads: int = 16,
        out_dim: int = 512,
        depth: int = 5,
        patch_size: int = 14,
        rope_freq: float = 100.0,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        
        # RoPE positional encoding
        self.rope = RoPE2D(freq=rope_freq)
        self.position_getter = PositionGetter()
        
        # Transformer decoder for camera features
        self.camera_decoder = TransformerDecoder(
            in_dim=in_dim,
            dec_embed_dim=dec_embed_dim,
            dec_num_heads=dec_num_heads,
            out_dim=out_dim,
            rope=self.rope,
            use_checkpoint=False,
            depth=depth,
        )
        
        # Camera head (outputs 4x4 pose)
        self.camera_head = CameraHead(dim=out_dim)
    
    def forward(
        self,
        front_patch_features: torch.Tensor,
        sat_patch_features: torch.Tensor,
        img_size: int = 518,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            front_patch_features: [B, num_patches, C] Front view patch features
            sat_patch_features: [B, num_patches, C] Satellite view patch features
            img_size: Image size for computing patch grid
        
        Returns:
            Dict containing:
                - front_pose: [B, 4, 4] Front view absolute pose
                - sat_pose: [B, 4, 4] Satellite view absolute pose
                - relative_pose: [B, 4, 4] Relative pose (front to sat)
                - yaw_radians: [B] Yaw angle in radians
                - yaw_degrees: [B] Yaw angle in degrees
        """
        B = front_patch_features.shape[0]
        patch_h = patch_w = img_size // self.patch_size
        
        # Get position encoding
        pos = self.position_getter(B, patch_h, patch_w, front_patch_features.device)
        
        # Process front view
        front_hidden = self.camera_decoder(front_patch_features, xpos=pos)
        
        # Process satellite view
        sat_hidden = self.camera_decoder(sat_patch_features, xpos=pos)
        
        # Predict poses
        front_pose = self.camera_head(front_hidden, patch_h, patch_w)  # [B, 4, 4]
        sat_pose = self.camera_head(sat_hidden, patch_h, patch_w)      # [B, 4, 4]
        
        # Compute relative pose: T_front_to_sat = T_sat^{-1} @ T_front
        relative_pose = self._compute_relative_pose(front_pose, sat_pose)
        
        # Extract yaw from relative rotation
        yaw_radians = self._extract_yaw_from_rotation(relative_pose[:, :3, :3])
        yaw_degrees = torch.rad2deg(yaw_radians)
        
        return {
            'front_pose': front_pose,
            'sat_pose': sat_pose,
            'relative_pose': relative_pose,
            'pose_enc': relative_pose.reshape(B, -1)[:, :9],  # For compatibility
            'quaternion': self._rotation_to_quaternion(relative_pose[:, :3, :3]),
            'yaw_radians': yaw_radians,
            'yaw_degrees': yaw_degrees,
        }
    
    def _compute_relative_pose(self, T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
        """Compute relative pose T1_to_T2 = T2^{-1} @ T1"""
        # T2_inv
        R2 = T2[:, :3, :3]
        t2 = T2[:, :3, 3:4]
        R2_inv = R2.transpose(-2, -1)
        t2_inv = -R2_inv @ t2
        
        T2_inv = torch.zeros_like(T2)
        T2_inv[:, :3, :3] = R2_inv
        T2_inv[:, :3, 3:4] = t2_inv
        T2_inv[:, 3, 3] = 1.0
        
        # Relative pose
        return T2_inv @ T1
    
    def _extract_yaw_from_rotation(self, R: torch.Tensor) -> torch.Tensor:
        """
        Extract yaw angle from rotation matrix.
        Assumes rotation is primarily around Z-axis (bird's eye view).
        
        yaw = atan2(R[1,0], R[0,0])
        """
        return torch.atan2(R[:, 1, 0], R[:, 0, 0])
    
    def _rotation_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        B = R.shape[0]
        
        # Ensure proper rotation matrix
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        
        q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)
        
        # Case: trace > 0
        mask = trace > 0
        if mask.any():
            s = torch.sqrt(trace[mask] + 1.0) * 2
            q[mask, 0] = 0.25 * s
            q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
            q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
            q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
        
        # Other cases (simplified)
        mask = ~mask
        if mask.any():
            # Fallback: use simple approximation
            q[mask, 0] = 1.0
        
        # Normalize
        q = F.normalize(q, dim=-1)
        return q
