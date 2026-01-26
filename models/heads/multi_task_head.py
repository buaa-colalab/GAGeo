# Multi-task head for cross-view localization
# Combines bbox detection, mask prediction, camera pose, and position prediction

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .bbox_head import BBoxHead
from .mask_head import MaskHead
from .yaw_head import CameraHead
from .position_head import PositionHead


class MultiTaskHead(nn.Module):
    """
    Combined multi-task head for bbox, mask, camera pose, and position prediction.
    
    Uses DPT-style multi-scale feature fusion for mask prediction,
    which is better suited for cross-view localization than SAM-style.
    
    Args:
        d_model: Feature dimension (2*C from VGGT = 2048)
        num_decoder_layers: Layers for detection head
        num_heads: Attention heads
        num_mask_classes: Number of mask output classes
        enable_bbox: Enable bbox prediction
        enable_mask: Enable mask prediction
        enable_camera: Enable camera pose prediction (yaw)
        enable_position: Enable position prediction
        intermediate_layer_idx: VGGT layers for multi-scale mask prediction
        position_output_mode: 'regression' or 'heatmap' for position prediction
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_mask_classes: int = 1,
        enable_bbox: bool = True,
        enable_mask: bool = True,
        enable_camera: bool = True,
        enable_position: bool = True,
        intermediate_layer_idx: List[int] = [5, 11, 17, 23],
        position_output_mode: str = 'regression',
    ):
        super().__init__()
        self.enable_bbox = enable_bbox
        self.enable_mask = enable_mask
        self.enable_camera = enable_camera
        self.enable_position = enable_position

        if enable_bbox:
            self.bbox_head = BBoxHead(
                d_model=d_model,
                num_decoder_layers=num_decoder_layers,
                num_heads=num_heads,
            )

        if enable_mask:
            self.mask_head = MaskHead(
                d_model=d_model,
                num_classes=num_mask_classes,
                intermediate_layer_idx=intermediate_layer_idx,
            )

        if enable_camera:
            self.camera_head = CameraHead(
                dim_in=d_model,
            )

        if enable_position:
            self.position_head = PositionHead(
                d_model=d_model,
                num_heads=num_heads,
                output_mode=position_output_mode,
            )

    def forward(
        self,
        front_features: torch.Tensor,
        sat_features: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
        target_size: Tuple[int, int] = (518, 518),
        aggregated_tokens_list: Optional[List[torch.Tensor]] = None,
        patch_start_idx: int = 5,
        front_camera_token: Optional[torch.Tensor] = None,
        sat_camera_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for all tasks.
        
        Args:
            front_features: [B, P, C] Front view patch features (last layer)
            sat_features: [B, P, C] Satellite view patch features (last layer)
            prompt_embeddings: [B, N, C] Sparse prompt embeddings
            dense_prompt_embeddings: [B, C, H, W] Dense mask embeddings (optional)
            target_size: Output size for mask prediction
            aggregated_tokens_list: List of [B, S, P_total, C] multi-layer features
                                   Required for multi-scale mask prediction
            patch_start_idx: Index where patch tokens start
            front_camera_token: [B, C] Camera token from front view (index 0)
            sat_camera_token: [B, C] Camera token from satellite view (index 0)
        
        Returns:
            Dict with predictions for enabled tasks:
                - pred_boxes: [B, N, 4]
                - scores: [B, N]
                - masks: [B, num_classes, H, W]
                - pose_enc: [B, 9] Full pose encoding
                - quaternion: [B, 4]
                - yaw_radians: [B]
                - yaw_degrees: [B]
        """
        outputs = {}

        # BBox detection (uses last layer patch features)
        if self.enable_bbox:
            bbox_pred, scores = self.bbox_head(
                prompt_embeddings=prompt_embeddings,
                sat_features=sat_features,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )
            outputs['pred_boxes'] = bbox_pred
            outputs['scores'] = scores

        # Mask prediction (uses multi-layer features if available)
        if self.enable_mask:
            if aggregated_tokens_list is not None:
                # Use multi-scale DPT fusion
                masks = self.mask_head(
                    aggregated_tokens_list=aggregated_tokens_list,
                    patch_start_idx=patch_start_idx,
                    prompt_embeddings=prompt_embeddings,
                    target_size=target_size,
                )
            else:
                # Fallback to single layer
                masks = self.mask_head.forward_single_layer(
                    sat_features=sat_features,
                    prompt_embeddings=prompt_embeddings,
                    target_size=target_size,
                )
            outputs['masks'] = masks

        # Camera pose prediction (uses camera tokens from both views)
        if self.enable_camera:
            if front_camera_token is not None and sat_camera_token is not None:
                camera_outputs = self.camera_head(
                    front_camera_token=front_camera_token,
                    sat_camera_token=sat_camera_token,
                )
                outputs.update(camera_outputs)

        # Position prediction (uses patch features from both views)
        if self.enable_position:
            position_outputs = self.position_head(
                front_features=front_features,
                sat_features=sat_features,
            )
            outputs['position'] = position_outputs['position']
            outputs['position_confidence'] = position_outputs['confidence']
            if 'heatmap' in position_outputs:
                outputs['position_heatmap'] = position_outputs['heatmap']

        return outputs
