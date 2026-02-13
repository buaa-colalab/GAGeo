# Cross-View Drone Localization System with Pi3 Backbone
# Uses Pi3 (upgraded from VGGT) for feature extraction
#
# Key difference from VGGT version:
# - Pi3 doesn't require a fixed reference frame - all views are treated equally
# - Uses DINOv2 encoder + Pi3 decoder blocks with RoPE positional encoding

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .backbone import Pi3Backbone, load_pi3_weights
from .encoder import GeometryPromptEncoder, PromptFusionWithDense, SAMStylePromptFusion
from .decoder import UnifiedQueryDecoder
from .heads import BBoxHead, HeatmapHead, Pi3CameraHead, CrossViewContrastiveHead


class CrossViewLocalizerPi3(nn.Module):
    """
    Cross-View Drone Localization System with Pi3 backbone.
    
    Pipeline:
    1. Input: Front-view image, Satellite image, Prompts (any combination of point/bbox/mask)
    2. Pi3 Backbone: Extract features F_f and F_s with cross-view attention
    3. Prompt Encoder: Encode prompts -> E_p (sparse) and E_d (dense)
    4. Prompt Fusion: Fuse E_p with F_f -> F_target (target-aware features)
    5. Unified Query Decoder: Object + Location queries attend to satellite features
    6. Task Heads: BBox Head, Heatmap Head, Camera Head
    
    Args:
        img_size: Input image size (default 518)
        patch_size: Patch size (default 14)
        decoder_size: Pi3 decoder size ('small', 'base', 'large')
        num_intent_queries: Number of learnable Intent Queries in Stage 1 (default 32)
        num_object_queries: Number of object queries for bbox detection (default 10)
        num_location_queries: Number of location queries for heatmap (default 16)
        num_heads: Number of attention heads for both stages (default 8)
        prompt_fusion_layers: Number of decoder layers in Stage 1 Intent Formation (default 3)
        num_decoder_layers: Number of decoder layers in Stage 2 Query Decoder (default 6)
        dropout: Dropout rate (default 0.1)
        freeze_backbone: Freeze Pi3 backbone
    """
    
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        decoder_size: str = 'large',
        num_intent_queries: int = 32,
        num_object_queries: int = 10,
        num_location_queries: int = 16,
        num_heads: int = 8,
        prompt_fusion_layers: int = 3,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        contrastive: bool = True,
        contrastive_proj_dim: int = 256,
        contrastive_queue_size: int = 16384,
        contrastive_momentum: float = 0.999,
        contrastive_temperature: float = 0.07,
        sam_embed_dim: int = None,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size  # 37
        
        # ============ 1. Pi3 Backbone ============
        self.backbone = Pi3Backbone(
            pos_type='rope100',
            decoder_size=decoder_size,
            img_size=img_size,
            patch_size=patch_size,
        )
        self.output_dim = self.backbone.output_dim  # 2048 for large
        self.patch_start_idx = self.backbone.patch_start_idx
        
        if freeze_backbone:
            self._freeze_backbone()
        
        # ============ 2. SAM Prompt Encoder (supports point/bbox/mask any combination) ============
        self.prompt_encoder = GeometryPromptEncoder(
            embed_dim=self.output_dim,
            image_embedding_size=(self.num_patches_per_side, self.num_patches_per_side),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
            sam_embed_dim=sam_embed_dim,
        )
        
        # ============ 3. SAM-style Prompt Fusion (bidirectional cross-attention) ============
        self.prompt_fusion = SAMStylePromptFusion(
            embedding_dim=self.output_dim,
            num_heads=num_heads,
            depth=prompt_fusion_layers,
            mlp_dim=self.output_dim,
            image_embedding_size=(self.num_patches_per_side, self.num_patches_per_side),
            attention_downsample_rate=2,
            dropout=dropout,
        )
        
        # ============ 4. Unified Query Decoder ============
        self.query_decoder = UnifiedQueryDecoder(
            hidden_dim=self.output_dim,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            num_object_queries=num_object_queries,
            num_location_queries=num_location_queries,
            spatial_size=(self.num_patches_per_side, self.num_patches_per_side),
            dropout=dropout,
        )
        
        # ============ 5. Task-Specific Heads ============
        self.bbox_head = BBoxHead(
            hidden_dim=self.output_dim,
            num_classes=1,
        )
        
        self.heatmap_head = HeatmapHead(
            hidden_dim=self.output_dim,
            output_size=img_size,
        )
        
        # ============ 6. Camera Head (Pi3-style, uses patch tokens) ============
        self.camera_head = Pi3CameraHead(
            in_dim=self.output_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            depth=5,
            patch_size=patch_size,
            rope_freq=100.0,
        )
        
        # ============ 7. Cross-View Contrastive Head (MoCo-style) ============
        self.contrastive_head = None
        if contrastive:
            self.contrastive_head = CrossViewContrastiveHead(
                in_dim=self.output_dim,
                proj_dim=contrastive_proj_dim,
                queue_size=contrastive_queue_size,
                momentum=contrastive_momentum,
                temperature=contrastive_temperature,
            )
    
    def _freeze_backbone(self):
        """Freeze Pi3 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _freeze_prompt_encoder(self):
        """Freeze SAM prompt encoder (keep projection layers trainable if they exist)."""
        for name, param in self.prompt_encoder.named_parameters():
            if 'sparse_proj' in name or 'dense_proj' in name:
                continue  # Keep projection layers trainable
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze Pi3 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        front_view: torch.Tensor,
        satellite_view: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        mono_mask: Optional[torch.Tensor] = None,
        sat_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            front_view: [B, 3, H, W] Front-view image
            satellite_view: [B, 3, H, W] Satellite image
            points: Tuple of (coords [B, N, 2], labels [B, N]) - optional
            boxes: [B, M, 4] in (x, y, w, h) format - optional
            masks: [B, 1, H, W] Binary masks - optional (prompt mask for front-view)
            mono_mask: [B, 1, H, W] Front-view segmentation mask for contrastive learning
            sat_mask: [B, 1, H, W] Satellite segmentation mask for contrastive learning
            
            Note: Any combination of points/boxes/masks is supported.
        
        Returns:
            Dict containing predictions and intermediate features.
        """
        B = front_view.shape[0]
        
        # 确保输入类型与模型权重一致（解决 bf16 混合精度问题）
        target_dtype = self.backbone.image_mean.dtype
        if front_view.dtype != target_dtype:
            front_view = front_view.to(target_dtype)
            satellite_view = satellite_view.to(target_dtype)
            if points is not None:
                points = (points[0].to(target_dtype), points[1])
            if boxes is not None:
                boxes = boxes.to(target_dtype)
            if masks is not None:
                masks = masks.to(target_dtype)

        # ============ Step 1: Pi3 Feature Extraction ============
        front_patch_features, sat_patch_features, front_camera_token, sat_camera_token = \
            self.backbone.get_front_sat_features(front_view, satellite_view)
        
        # ============ Step 2: Prompt Encoding (supports any combination) ============

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        sparse_embeddings = sparse_embeddings.to(target_dtype)
        dense_embeddings = dense_embeddings.to(target_dtype)
        # ============ Step 3: SAM-style Prompt Fusion ============
        # Bidirectional cross-attention: prompt tokens <-> front-view features
        # Output: prompt-conditioned front-view features (replaces intent_features)
        dense_for_fusion = dense_embeddings if masks is not None else None
        intent_features = self.prompt_fusion(
            image_features=front_patch_features,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_for_fusion,
        )
        
        # ============ Step 4: Unified Query Decoder ============
        # Queries + prompt-conditioned front features cross-attend to satellite
        decoder_outputs = self.query_decoder(
            memory=sat_patch_features,
            intent_features=intent_features,
        )
        obj_features = decoder_outputs['obj_features']
        loc_features = decoder_outputs['loc_features']
        
        # ============ Step 5: Task Heads ============
        bbox_outputs = self.bbox_head(obj_features)
        heatmap_outputs = self.heatmap_head(
            query_features=loc_features,
            spatial_features=sat_patch_features,
            spatial_size=(self.num_patches_per_side, self.num_patches_per_side),
        )
        
        # ============ Step 6: Camera Yaw Prediction (Pi3-style, uses patch tokens) ============
        camera_output = self.camera_head(
            front_patch_features=front_patch_features,
            sat_patch_features=sat_patch_features,
            img_size=self.img_size,
        )
        
        # ============ Step 7: Cross-View Contrastive Loss ============
        contrastive_loss = None
        if self.contrastive_head is not None and mono_mask is not None and sat_mask is not None:
            contrastive_loss = self.contrastive_head(
                mono_features=front_patch_features,
                sat_features=sat_patch_features,
                mono_mask=mono_mask,
                sat_mask=sat_mask,
            )
        
        # ============ Combine Outputs ============
        result = {
            # BBox detection (include class_logits for Hungarian matching)
            'pred_boxes': bbox_outputs['pred_boxes'],
            'bbox_scores': bbox_outputs['bbox_scores'],
            'class_logits': bbox_outputs['class_logits'],
            
            # Camera position
            'heatmap': heatmap_outputs['heatmap'],
            'position': heatmap_outputs['position'],
            'heatmap_logits': heatmap_outputs['heatmap_logits'],
            
            # Camera rotation (relative pose)
            'rotation_matrix': camera_output['rotation_matrix'],
            'yaw': camera_output['yaw'],
            'pitch': camera_output['pitch'],
            'roll': camera_output['roll'],
            
            # Features for visualization/debugging
            'front_features': front_patch_features,
            'sat_features': sat_patch_features,
            'sparse_embeddings': sparse_embeddings,
            'intent_features': intent_features,
        }
        
        if contrastive_loss is not None:
            result['contrastive_loss'] = contrastive_loss
        
        return result


def build_cross_view_localizer_pi3(
    pretrained_pi3: Optional[str] = None,
    freeze_backbone: bool = True,
    freeze_prompt_encoder: bool = True,
    load_camera_head_weights: bool = True,
    sam_weights: Optional[str] = None,
    **kwargs
) -> CrossViewLocalizerPi3:
    """
    Build CrossViewLocalizerPi3 with optional pretrained Pi3 weights.
    
    Args:
        pretrained_pi3: Path to pretrained Pi3 checkpoint
        freeze_backbone: Whether to freeze Pi3 backbone
        freeze_prompt_encoder: Whether to freeze SAM prompt encoder
        load_camera_head_weights: Whether to load camera head weights from Pi3 checkpoint
        sam_weights: Path to SAM2 checkpoint for prompt encoder weights
        **kwargs: Additional arguments for CrossViewLocalizerPi3
    
    Returns:
        CrossViewLocalizerPi3 model
    """
    model = CrossViewLocalizerPi3(freeze_backbone=freeze_backbone, **kwargs)
    
    if pretrained_pi3 is not None:
        load_pi3_weights(model.backbone, pretrained_pi3, strict=False)
        
        # Load camera head weights from Pi3 checkpoint
        if load_camera_head_weights:
            _load_camera_head_from_pi3(model.camera_head, pretrained_pi3)
    
    # Load SAM2 pretrained weights into prompt encoder
    if sam_weights is not None:
        from .encoder.prompt_encoder import load_sam_prompt_encoder_weights
        load_sam_prompt_encoder_weights(model.prompt_encoder, sam_weights)
    
    if freeze_prompt_encoder:
        model._freeze_prompt_encoder()
    
    return model


def _load_camera_head_from_pi3(camera_head, checkpoint_path: str):
    """
    Load camera_decoder and camera_head weights from Pi3 checkpoint.
    
    Pi3 checkpoint keys:
        camera_decoder.* -> maps to Pi3CameraHead.camera_decoder.*
        camera_head.*    -> maps to Pi3CameraHead.camera_head.*
    
    Args:
        camera_head: Pi3CameraHead module
        checkpoint_path: Path to Pi3 checkpoint (.safetensors or .pt)
    """
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        import torch as _torch
        state_dict = _torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    # Filter keys: camera_decoder.* and camera_head.*
    camera_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('camera_decoder.') or k.startswith('camera_head.'):
            camera_state_dict[k] = v
    
    if not camera_state_dict:
        print(f"  WARNING: No camera_decoder/camera_head keys found in {checkpoint_path}")
        return
    
    missing, unexpected = camera_head.load_state_dict(camera_state_dict, strict=False)
    
    print(f"Loaded Pi3 camera head weights from {checkpoint_path}")
    print(f"  Loaded keys: {len(camera_state_dict)}")
    if missing:
        print(f"  Missing keys: {len(missing)} - {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

