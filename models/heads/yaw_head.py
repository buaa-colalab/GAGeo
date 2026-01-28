# Camera head for cross-view localization
# Predicts camera pose (with yaw supervision only) using VGGT-style iterative refinement
# Adapted from VGGT's camera_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from ..layers.mlp import Mlp
from ..layers.block import Block


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate the input tensor using scaling and shifting parameters."""
    return x * (1 + scale) + shift


class CameraHead(nn.Module):
    """
    Camera pose prediction head for cross-view localization.
    
    Adapted from VGGT's CameraHead with modifications for cross-view matching:
    - Uses camera tokens from both front and satellite views
    - Fuses cross-view information before iterative refinement
    - Outputs full pose (T, quat, FoV) but only yaw is supervised
    
    The yaw angle can be extracted from the quaternion output.
    
    Args:
        dim_in: Input feature dimension (2*C from VGGT = 2048)
        trunk_depth: Number of transformer blocks for refinement
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        num_iterations: Number of iterative refinement steps
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        num_iterations: int = 4,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.trunk_depth = trunk_depth
        self.num_iterations = num_iterations

        # Output: Translation(3) + Quaternion(4) + FoV(2) = 9
        self.target_dim = 9

        # Cross-view fusion: fuse front and sat camera tokens
        self.front_norm = nn.LayerNorm(dim_in)
        self.sat_norm = nn.LayerNorm(dim_in)
        self.cross_attn = nn.MultiheadAttention(
            dim_in, num_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(dim_in)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(dim_in, dim_in * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_in * 2, dim_in),
        )
        self.fusion_norm = nn.LayerNorm(dim_in)

        # Trunk transformer blocks (from VGGT)
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty pose token (from VGGT)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # AdaLN modulation (from VGGT)
        self.poseLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_in, 3 * dim_in, bias=True)
        )
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)

        # Pose prediction branch
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0
        )

    def forward(
        self,
        front_camera_token: torch.Tensor,
        sat_camera_token: torch.Tensor,
        num_iterations: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            front_camera_token: [B, C] Camera token from front view (index 0 of VGGT output)
            sat_camera_token: [B, C] Camera token from satellite view
            num_iterations: Override default number of refinement iterations
        
        Returns:
            Dict containing:
                - pose_enc: [B, 9] Full pose encoding (T, quat, FoV)
                - quaternion: [B, 4] Quaternion (w, x, y, z)
                - yaw_radians: [B] Yaw angle in radians (extracted from quaternion)
                - yaw_degrees: [B] Yaw angle in degrees
                - pose_enc_list: List of pose encodings from each iteration
        """
        if num_iterations is None:
            num_iterations = self.num_iterations

        B = front_camera_token.shape[0]

        # Ensure tokens have sequence dimension
        if front_camera_token.dim() == 2:
            front_camera_token = front_camera_token.unsqueeze(1)  # [B, 1, C]
        if sat_camera_token.dim() == 2:
            sat_camera_token = sat_camera_token.unsqueeze(1)  # [B, 1, C]

        # Cast to model dtype for mixed precision compatibility
        target_dtype = self.front_norm.weight.dtype
        front_camera_token = front_camera_token.to(dtype=target_dtype)
        sat_camera_token = sat_camera_token.to(dtype=target_dtype)

        # ============ Cross-view fusion ============
        # Front camera token attends to satellite camera token
        front_norm = self.front_norm(front_camera_token)
        sat_norm = self.sat_norm(sat_camera_token)

        # Cross-attention
        attn_out, _ = self.cross_attn(front_norm, sat_norm, sat_norm)
        fused_token = self.cross_norm(front_camera_token + attn_out)

        # FFN
        fused_token = self.fusion_norm(fused_token + self.fusion_ffn(fused_token))

        # Normalize for trunk
        pose_tokens = self.token_norm(fused_token)  # [B, 1, C]

        # ============ Iterative refinement (from VGGT) ============
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use learned empty pose for first iteration
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, 1, -1))
            else:
                # Detach previous prediction
                pred_pose_enc_detached = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc_detached)

            # AdaLN modulation
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * modulate(
                self.adaln_norm(pose_tokens), shift_msa, scale_msa
            )
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            # Trunk processing
            pose_tokens_modulated = self.trunk(pose_tokens_modulated)

            # Predict delta
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            # Accumulate
            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            pred_pose_enc_list.append(pred_pose_enc.squeeze(1))  # [B, 9]

        # Final pose encoding
        pose_enc = pred_pose_enc.squeeze(1)  # [B, 9]

        # Extract components
        translation = pose_enc[:, :3]  # [B, 3]
        quaternion = pose_enc[:, 3:7]  # [B, 4]
        fov = pose_enc[:, 7:]          # [B, 2]

        # Normalize quaternion
        quaternion = F.normalize(quaternion, dim=-1)

        # Extract yaw from quaternion
        # Quaternion: (w, x, y, z)
        # Yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        yaw_radians = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        yaw_degrees = torch.rad2deg(yaw_radians)

        return {
            'pose_enc': pose_enc,
            'translation': translation,
            'quaternion': quaternion,
            'fov': fov,
            'yaw_radians': yaw_radians,
            'yaw_degrees': yaw_degrees,
            'pose_enc_list': pred_pose_enc_list,
        }


# Backward compatibility alias
YawHead = CameraHead
