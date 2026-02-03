# Cross-View Drone Localization System with Pi3 Backbone
# Uses Pi3 (upgraded from VGGT) for feature extraction
#
# Key features:
# - Pi3 doesn't require a fixed reference frame - all views are treated equally
# - Uses DINOv2 encoder + Pi3 decoder blocks with RoPE positional encoding
# - Supports bidirectional localization:
#   - mono_to_sat: prompt on mono, locate bbox on sat
#   - sat_to_mono: prompt on sat, locate bbox on mono
# - camera_position is always predicted on sat (satellite has wider coverage)

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from .backbone import Pi3Backbone, load_pi3_weights
from .encoder import GeometryPromptEncoder, PromptFusionWithDense
from .decoder import UnifiedQueryDecoder
from .heads import BBoxHead, HeatmapHead, Pi3CameraHead


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
        num_heads: Number of attention heads
        num_decoder_layers: DETR decoder layers
        num_object_queries: Number of object queries for bbox detection
        num_location_queries: Number of location queries for heatmap
        freeze_backbone: Freeze Pi3 backbone
    """
    
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        decoder_size: str = 'large',
        num_heads: int = 16,
        num_decoder_layers: int = 6,
        num_object_queries: int = 10,
        num_location_queries: int = 16,
        freeze_backbone: bool = False,
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
        )
        
        # ============ 3. Prompt Fusion Module (SAM-style) ============
        self.prompt_fusion = PromptFusionWithDense(
            embedding_dim=self.output_dim,
            num_heads=num_heads // 2,
            depth=2,
            mlp_dim=self.output_dim,
            image_embedding_size=(self.num_patches_per_side, self.num_patches_per_side),
            activation=nn.ReLU,
            attention_downsample_rate=2,
        )
        
        # ============ 4. Unified Query Decoder ============
        self.query_decoder = UnifiedQueryDecoder(
            hidden_dim=self.output_dim,
            num_heads=num_heads // 2,
            num_decoder_layers=num_decoder_layers,
            num_object_queries=num_object_queries,
            num_location_queries=num_location_queries,
            spatial_size=(self.num_patches_per_side, self.num_patches_per_side),
            dropout=0.1,
        )
        
        # Mono guidance projection (for location queries)
        self.mono_guidance_proj = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
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
    
    def _freeze_backbone(self):
        """Freeze Pi3 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze Pi3 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        mono_view: torch.Tensor,
        sat_view: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        prompt_views: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bidirectional localization support.
        
        Args:
            mono_view: [B, 3, H, W] Ground-view (mono) image
            sat_view: [B, 3, H, W] Satellite image
            points: Tuple of (coords [B, N, 2], labels [B, N]) - optional
            boxes: [B, M, 4] in (x, y, w, h) format - optional
            masks: [B, 1, H, W] Binary masks - optional
            prompt_views: List of 'mono' or 'sat' indicating prompt source for each sample
                         If None, defaults to 'mono' (mono_to_sat direction)
            
            Note: Any combination of points/boxes/masks is supported.
        
        Returns:
            Dict containing predictions and intermediate features.
        """
        B = mono_view.shape[0]
        
        # ============ Step 1: Pi3 Feature Extraction ============
        # 始终提取两个视图的特征
        mono_patch_features, sat_patch_features, mono_camera_token, sat_camera_token = \
            self.backbone.get_front_sat_features(mono_view, sat_view)
        
        # ============ Step 2: Prompt Encoding (supports any combination) ============
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        
        # ============ Step 3: Direction-aware Prompt Fusion ============
        # prompt_features: prompt 所在视图的特征
        prompt_features = self._select_features_by_view(
            mono_patch_features, sat_patch_features, prompt_views, default='mono'
        )
        
        dense_for_fusion = dense_embeddings if masks is not None else None
        fused_sparse, fused_prompt, target_guidance = self.prompt_fusion(
            image_features=prompt_features,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_for_fusion,
        )
        
        # ============ Step 4: Unified Query Decoder ============
        # obj_memory: 目标视图（与 prompt 相反）
        # loc_memory: 始终是 sat（camera_position 在 sat 上）
        # obj_guidance: 来自 prompt 视图（告诉模型“要找什么”）
        # loc_guidance: 始终来自 mono（告诉模型“从哪个视角拍的”）
        obj_memory = self._select_features_by_view(
            mono_patch_features, sat_patch_features, prompt_views, default='sat', invert=True
        )
        loc_memory = sat_patch_features
        
        # mono_guidance: 用投影层处理，保持与 target_guidance 一致性
        mono_guidance = self.mono_guidance_proj(mono_patch_features.mean(dim=1))
        
        decoder_outputs = self.query_decoder(
            obj_memory=obj_memory,
            loc_memory=loc_memory,
            obj_guidance=target_guidance,
            loc_guidance=mono_guidance,
        )
        obj_features = decoder_outputs['obj_features']
        loc_features = decoder_outputs['loc_features']
        
        # ============ Step 5: Task Heads ============
        # BBox: 在目标视图上预测
        bbox_outputs = self.bbox_head(obj_features)
        
        # Heatmap (camera position): 始终在 sat 图上预测
        # 因为 camera_position 是无人机在卫星图上的位置
        heatmap_outputs = self.heatmap_head(
            query_features=loc_features,
            spatial_features=sat_patch_features,
            spatial_size=(self.num_patches_per_side, self.num_patches_per_side),
        )
        
        # ============ Step 6: Camera Yaw Prediction (Pi3-style, uses patch tokens) ============
        camera_output = self.camera_head(
            front_patch_features=mono_patch_features,
            sat_patch_features=sat_patch_features,
            img_size=self.img_size,
        )
        
        # ============ Combine Outputs ============
        return {
            # BBox detection (在 sat 图上)
            'pred_boxes': bbox_outputs['pred_boxes'],
            'bbox_scores': bbox_outputs['bbox_scores'],
            
            # Camera position (在 sat 图上)
            'heatmap': heatmap_outputs['heatmap'],
            'position': heatmap_outputs['position'],
            'heatmap_logits': heatmap_outputs['heatmap_logits'],
            
            # Camera angle
            'yaw_radians': camera_output['yaw_radians'],
            'yaw_degrees': camera_output['yaw_degrees'],
            'quaternion': camera_output['quaternion'],
            'pose_enc': camera_output['pose_enc'],
            
            # Features for visualization/debugging
            'mono_features': mono_patch_features,
            'sat_features': sat_patch_features,
            'prompt_features': prompt_features,
            'sparse_embeddings': sparse_embeddings,
            'fused_prompt_features': fused_prompt,
            'target_guidance': target_guidance,
        }
    
    def _select_features_by_view(
        self,
        mono_features: torch.Tensor,
        sat_features: torch.Tensor,
        prompt_views: Optional[List[str]],
        default: str = 'mono',
        invert: bool = False,
    ) -> torch.Tensor:
        """
        统一的特征选择方法。
        
        Args:
            mono_features: [B, N, C] mono 视图特征
            sat_features: [B, N, C] sat 视图特征
            prompt_views: List of 'mono' or 'sat'，None 时使用 default
            default: prompt_views 为 None 时的默认值
            invert: True 时选择与 prompt_view 相反的视图（用于 obj_memory）
        
        Returns:
            selected_features: [B, N, C]
        """
        B = mono_features.shape[0]
        
        # 处理 None 情况
        if prompt_views is None:
            prompt_views = [default] * B
        
        # 如果 invert，则选择相反视图
        if invert:
            views = ['sat' if v == 'mono' else 'mono' for v in prompt_views]
        else:
            views = prompt_views
        
        # 快速路径：所有样本相同
        if all(v == 'mono' for v in views):
            return mono_features
        if all(v == 'sat' for v in views):
            return sat_features
        
        # 混合 batch
        selected = torch.zeros_like(mono_features)
        for i, v in enumerate(views):
            selected[i] = mono_features[i] if v == 'mono' else sat_features[i]
        return selected


def build_cross_view_localizer_pi3(
    pretrained_pi3: Optional[str] = None,
    freeze_backbone: bool = True,
    **kwargs
) -> CrossViewLocalizerPi3:
    """
    Build CrossViewLocalizerPi3 with optional pretrained Pi3 weights.
    
    Args:
        pretrained_pi3: Path to pretrained Pi3 checkpoint
        freeze_backbone: Whether to freeze Pi3 backbone
        **kwargs: Additional arguments for CrossViewLocalizerPi3
    
    Returns:
        CrossViewLocalizerPi3 model
    """
    model = CrossViewLocalizerPi3(freeze_backbone=freeze_backbone, **kwargs)
    
    if pretrained_pi3 is not None:
        load_pi3_weights(model.backbone, pretrained_pi3, strict=False)
    
    return model
