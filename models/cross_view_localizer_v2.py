# Cross-View Object Localizer
# Front View -> Satellite View multi-task prediction
#
# Architecture:
# 1. VGGT Alternating Attention: processes front + satellite views jointly
# 2. SAM-style Geometry Prompt Encoder: encodes user clicks/boxes/masks
# 3. Multi-Task Head: predicts bboxes, segmentation masks, and camera yaw

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .vggt_aggregator import Aggregator
from .prompt_encoder import GeometryPromptEncoder
from .heads import BBoxHead, MultiQueryBBoxHead, MultiTaskHead


class CrossViewLocalizer(nn.Module):
    """
    Cross-view object localization: Front View -> Satellite View bbox.
    
    Data Flow:
    ---------
    Step 1: Input Preparation
        front_view: [B, 3, 518, 518]
        satellite_view: [B, 3, 518, 518]
        images = stack([front_view, satellite_view], dim=1)  # [B, 2, 3, 518, 518]
    
    Step 2: VGGT Alternating Attention (24 layers)
        - Frame attention: each view independently [B*2, P, C]
        - Global attention: cross-view interaction [B, 2*P, C]
        - Output: List of [B, 2, P, 2*C] (2*C = frame + global concat)
    
    Step 3: Geometry Prompt Encoding
        - Extract front view features: [B, P, 2*C]
        - Encode user prompts (points/boxes/masks) using SAM-style encoder
        - Output: sparse [B, N_prompts, 2*C], dense [B, 2*C, H, W]
    
    Step 4: Satellite Detection
        - Extract satellite features (already has cross-view info): [B, P, 2*C]
        - Detection head predicts boxes conditioned on geometry prompts
        - Output: bbox [B, N, 4], scores [B, N]
    
    Args:
        img_size: Input image size (default 518 for DINOv2)
        patch_size: Patch size (default 14)
        embed_dim: VGGT embedding dimension (default 1024, output is 2*embed_dim)
        vggt_depth: Number of VGGT blocks (default 24)
        num_heads: Number of attention heads
        num_decoder_layers: Detection head decoder layers
        use_multi_query: Use learnable object queries (DETR-style)
        num_queries: Number of object queries if use_multi_query=True
        freeze_vggt: Freeze VGGT backbone
        enable_bbox: Enable bounding box prediction
        enable_seg: Enable segmentation/mask prediction
        enable_camera: Enable camera pose prediction (yaw)
        enable_position: Enable camera position prediction
        num_seg_classes: Number of segmentation classes
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        vggt_depth: int = 24,
        num_heads: int = 16,
        num_decoder_layers: int = 6,
        use_multi_query: bool = False,
        num_queries: int = 100,
        freeze_vggt: bool = False,
        patch_embed: str = "dinov2_vitl14_reg",
        enable_bbox: bool = True,
        enable_seg: bool = False,
        enable_camera: bool = False,
        enable_position: bool = False,
        num_seg_classes: int = 1,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = 2 * embed_dim  # VGGT outputs 2*C (frame + global concat)

        # Calculate patch grid size
        self.num_patches_per_side = img_size // patch_size  # 518/14 = 37
        self.num_patches = self.num_patches_per_side ** 2  # 1369

        # ============ 1. VGGT Backbone ============
        self.vggt = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vggt_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            num_register_tokens=4,
            patch_embed=patch_embed,
            aa_order=["frame", "global"],
            aa_block_size=1,
            qk_norm=True,
            rope_freq=100,
            init_values=0.01,
        )
        self.patch_start_idx = self.vggt.patch_start_idx  # 5 (1 camera + 4 register)

        if freeze_vggt:
            self._freeze_vggt()

        # ============ 2. Geometry Prompt Encoder (SAM-style) ============
        self.prompt_encoder = GeometryPromptEncoder(
            embed_dim=self.output_dim,
            image_embedding_size=(self.num_patches_per_side, self.num_patches_per_side),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )

        # ============ 3. Task Heads ============
        self.use_multi_query = use_multi_query
        self.enable_bbox = enable_bbox
        self.enable_seg = enable_seg
        self.enable_camera = enable_camera
        self.enable_position = enable_position

        # Use MultiTaskHead if multiple tasks enabled, otherwise individual heads
        if enable_seg or enable_camera or enable_position:
            self.task_head = MultiTaskHead(
                d_model=self.output_dim,
                num_decoder_layers=num_decoder_layers,
                num_heads=num_heads // 2,
                num_mask_classes=num_seg_classes,
                enable_bbox=enable_bbox,
                enable_mask=enable_seg,
                enable_camera=enable_camera,
                enable_position=enable_position,
            )
        else:
            # Bbox only
            if use_multi_query:
                self.bbox_head = MultiQueryBBoxHead(
                    d_model=self.output_dim,
                    num_queries=num_queries,
                    num_decoder_layers=num_decoder_layers,
                    num_heads=num_heads,
                    dropout=0.1,
                )
            else:
                self.bbox_head = BBoxHead(
                    d_model=self.output_dim,
                    num_decoder_layers=num_decoder_layers,
                    num_heads=num_heads // 2,
                    dropout=0.1,
                )

    def _freeze_vggt(self):
        """Freeze VGGT backbone parameters."""
        for param in self.vggt.parameters():
            param.requires_grad = False

    def unfreeze_vggt(self):
        """Unfreeze VGGT backbone parameters."""
        for param in self.vggt.parameters():
            param.requires_grad = True

    def forward(
        self,
        front_view: torch.Tensor,
        satellite_view: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            front_view: [B, 3, H, W] Front view images (normalized to [0, 1])
            satellite_view: [B, 3, H, W] Satellite view images (normalized to [0, 1])
            points: Tuple of (coords [B, N, 2], labels [B, N]) - user clicked points
            boxes: [B, M, 4] User drawn boxes in (x1, y1, x2, y2) format
            masks: [B, 1, H, W] User drawn masks
        
        Returns:
            Dict containing:
                - pred_boxes: [B, N, 4] Predicted boxes (cx, cy, w, h) normalized
                - scores: [B, N] Confidence scores
                - front_features: [B, P, 2C] Front view features
                - sat_features: [B, P, 2C] Satellite view features
                - seg_masks: [B, C, H, W] Segmentation masks (if enable_seg)
                - yaw_sincos: [B, 2] Camera yaw (sin, cos) (if enable_yaw)
                - yaw_radians: [B] Camera yaw in radians (if enable_yaw)
                - yaw_degrees: [B] Camera yaw in degrees (if enable_yaw)
        """
        B = front_view.shape[0]

        # ============ Step 1: Stack views ============
        images = torch.stack([front_view, satellite_view], dim=1)  # [B, 2, 3, H, W]

        # ============ Step 2: VGGT Alternating Attention ============
        vggt_outputs, patch_start_idx = self.vggt(images)
        # vggt_outputs: List of [B, 2, P_total, 2*C]
        # P_total = patch_start_idx + num_patches = 5 + 1369 = 1374

        # Get last layer features
        features = vggt_outputs[-1]  # [B, 2, P_total, 2*C]

        # Split front and satellite features
        front_features = features[:, 0]  # [B, P_total, 2*C]
        sat_features = features[:, 1]    # [B, P_total, 2*C]

        # Extract only patch tokens (remove camera and register tokens)
        front_patch_features = front_features[:, patch_start_idx:]  # [B, P, 2*C]
        sat_patch_features = sat_features[:, patch_start_idx:]      # [B, P, 2*C]

        # Extract camera tokens (index 0) for CameraHead
        front_camera_token = front_features[:, 0]  # [B, 2*C]
        sat_camera_token = sat_features[:, 0]      # [B, 2*C]

        # ============ Step 3: Encode Geometry Prompts ============
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        # sparse_embeddings: [B, N_sparse, 2*C]
        # dense_embeddings: [B, 2*C, H', W']

        # ============ Step 4: Task Prediction ============
        if self.enable_seg or self.enable_camera or self.enable_position:
            # Multi-task head with multi-layer features for mask prediction
            outputs = self.task_head(
                front_features=front_patch_features,
                sat_features=sat_patch_features,
                prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings if masks is not None else None,
                target_size=(self.img_size, self.img_size),
                aggregated_tokens_list=vggt_outputs,  # Pass all layers for DPT-style mask
                patch_start_idx=patch_start_idx,
                front_camera_token=front_camera_token,  # For CameraHead
                sat_camera_token=sat_camera_token,
            )
        else:
            # Bbox only
            if self.use_multi_query:
                pred_boxes, scores = self.bbox_head(
                    geo_embeddings=sparse_embeddings,
                    sat_features=sat_patch_features,
                )
            else:
                pred_boxes, scores = self.bbox_head(
                    prompt_embeddings=sparse_embeddings,
                    sat_features=sat_patch_features,
                    dense_prompt_embeddings=dense_embeddings if masks is not None else None,
                )
            outputs = {
                'pred_boxes': pred_boxes,
                'scores': scores,
            }

        # Add features to output
        outputs['front_features'] = front_patch_features
        outputs['sat_features'] = sat_patch_features
        outputs['sparse_embeddings'] = sparse_embeddings

        return outputs

    def get_intermediate_features(
        self,
        front_view: torch.Tensor,
        satellite_view: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Get intermediate features from VGGT for visualization or multi-scale detection.
        
        Args:
            front_view: [B, 3, H, W]
            satellite_view: [B, 3, H, W]
            layer_indices: Which layers to return (default: all)
        
        Returns:
            Dict with 'front_features' and 'sat_features' lists
        """
        images = torch.stack([front_view, satellite_view], dim=1)
        vggt_outputs, patch_start_idx = self.vggt(images)

        if layer_indices is None:
            layer_indices = list(range(len(vggt_outputs)))

        front_features = []
        sat_features = []

        for idx in layer_indices:
            features = vggt_outputs[idx]
            front_features.append(features[:, 0, patch_start_idx:])
            sat_features.append(features[:, 1, patch_start_idx:])

        return {
            'front_features': front_features,
            'sat_features': sat_features,
        }


def build_cross_view_localizer(
    pretrained_vggt: Optional[str] = None,
    freeze_vggt: bool = True,
    **kwargs
) -> CrossViewLocalizer:
    """
    Build CrossViewLocalizer with optional pretrained VGGT weights.
    
    Args:
        pretrained_vggt: Path to pretrained VGGT checkpoint
        freeze_vggt: Whether to freeze VGGT backbone
        **kwargs: Additional arguments for CrossViewLocalizer
    
    Returns:
        CrossViewLocalizer model
    """
    model = CrossViewLocalizer(freeze_vggt=freeze_vggt, **kwargs)

    if pretrained_vggt is not None:
        state_dict = torch.load(pretrained_vggt, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter for aggregator weights
        vggt_state = {}
        for k, v in state_dict.items():
            if k.startswith('aggregator.'):
                new_k = k.replace('aggregator.', '')
                vggt_state[new_k] = v
            elif not any(k.startswith(p) for p in ['camera_head', 'point_head', 'depth_head', 'track_head']):
                vggt_state[k] = v

        missing, unexpected = model.vggt.load_state_dict(vggt_state, strict=False)
        print(f"Loaded VGGT weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    return model
