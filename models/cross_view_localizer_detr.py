# Cross-View Drone Localization System
# Integrates VGGT, DETR, and SAM for cross-view object detection and camera localization
#
# Architecture:
# 1. VGGT Backbone: Extract front-view and satellite features with cross-view fusion
# 2. SAM Prompt Encoder: Encode user prompts (points/boxes/masks)
# 3. Prompt Fusion: Fuse prompts with front-view features (SAM-style two-way transformer)
# 4. DETR Decoder (Unified): Single decoder with two types of queries
#    - Object Queries (N_obj): For bbox detection in satellite view
#    - Location Queries (G x G): For camera position heatmap
#    Both query types share the same decoder layers, then split for task-specific heads
# 5. Task Heads: BBox Head, Heatmap Head, Camera Head

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

from .vggt_aggregator import Aggregator
from .encoder import GeometryPromptEncoder, PositionEmbeddingSine
from .prompt_fusion import PromptFusionWithDense
from .decoder import TransformerDecoder, MLP
from .heads.yaw_head import CameraHead


class CrossViewLocalizerDETR(nn.Module):
    """
    Cross-View Drone Localization System with DETR-style architecture.
    
    Pipeline:
    1. Input: Front-view image, Satellite image, Prompts (point/bbox/mask)
    2. VGGT: Extract features F_f and F_s with cross-view attention
    3. Prompt Encoder: Encode prompts -> E_p (sparse) and E_d (dense)
    4. Prompt Fusion: Fuse E_p with F_f -> F_target (target-aware features)
    5. DETR Decoder (Unified):
       - Concatenate [Object Queries, Location Queries] as unified queries
       - All queries attend to satellite features F_s (guided by F_target)
       - Split decoder output back to object/location branches
       a. Object branch -> BBox Head -> BBox predictions
       b. Location branch -> Heatmap Head -> Position heatmap
    6. Camera Head: Predict yaw angle from camera tokens
    
    Args:
        img_size: Input image size (518 for DINOv2)
        patch_size: Patch size (14)
        embed_dim: VGGT embedding dimension (1024, output is 2*embed_dim=2048)
        vggt_depth: Number of VGGT blocks (24)
        num_heads: Number of attention heads
        num_decoder_layers: DETR decoder layers
        num_object_queries: Number of object queries for bbox detection
        location_grid_size: Grid size for location queries (e.g., 32x32)
        freeze_vggt: Freeze VGGT backbone
        use_prompt_fusion: Use SAM-style prompt fusion
    """
    
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        vggt_depth: int = 24,
        num_heads: int = 16,
        num_decoder_layers: int = 6,
        num_object_queries: int = 10,
        num_location_queries: int = 16,
        heatmap_size: int = 32,
        freeze_vggt: bool = False,
        patch_embed: str = "dinov2_vitl14_reg",
        use_prompt_fusion: bool = True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = 2 * embed_dim  # VGGT outputs 2*C
        
        # Calculate patch grid
        self.num_patches_per_side = img_size // patch_size  # 37
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
        self.patch_start_idx = self.vggt.patch_start_idx  # 5
        
        if freeze_vggt:
            self._freeze_vggt()
        
        # ============ 2. SAM Prompt Encoder ============
        self.prompt_encoder = GeometryPromptEncoder(
            embed_dim=self.output_dim,
            image_embedding_size=(self.num_patches_per_side, self.num_patches_per_side),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )
        
        # ============ 3. Prompt Fusion Module (SAM-style) ============
        self.use_prompt_fusion = use_prompt_fusion
        if use_prompt_fusion:
            self.prompt_fusion = PromptFusionWithDense(
                embedding_dim=self.output_dim,
                num_heads=num_heads // 2,
                depth=2,
                mlp_dim=self.output_dim,
                image_embedding_size=(self.num_patches_per_side, self.num_patches_per_side),
                activation=nn.ReLU,
                attention_downsample_rate=2,
            )
        
        # Target guidance projection (for adding F_target to queries)
        self.target_guidance_proj = nn.Linear(self.output_dim, self.output_dim)
        
        # ============ 4. Unified DETR Decoder ============
        # Two types of queries processed by the SAME decoder
        self.num_object_queries = num_object_queries
        self.num_location_queries = num_location_queries
        self.heatmap_size = heatmap_size
        
        # Object queries: learnable embeddings for bbox detection
        self.object_queries = nn.Embedding(num_object_queries, self.output_dim)
        self.object_query_pos = nn.Embedding(num_object_queries, self.output_dim)
        
        # Location queries: learnable embeddings (DETR-style)
        self.location_queries = nn.Embedding(num_location_queries, self.output_dim)
        self.location_query_pos = nn.Embedding(num_location_queries, self.output_dim)
        
        # Memory (satellite features) positional encoding - use sinusoidal (DETR-style)
        self.memory_pos_embed = PositionEmbeddingSine(
            num_pos_feats=self.output_dim // 2,
            normalize=True,
        )
        
        # Single unified decoder for both query types
        self.decoder = TransformerDecoder(
            d_model=self.output_dim,
            nhead=num_heads // 2,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=self.output_dim,
            dropout=0.1,
            normalize_before=False,
            return_intermediate=False,
        )
        
        # ============ 5. Task-Specific Heads ============
        # BBox prediction head (DETR-style 3-layer MLP)
        self.bbox_head = MLP(self.output_dim, self.output_dim, 4, 3)
        self.bbox_score_head = nn.Linear(self.output_dim, 1)
        
        # Heatmap prediction head using Mask2Former-style Dynamic Conv (dot product)
        # Each location query generates a C-dim vector for 1x1 conv with sat features
        self.mask_embed_dim = 256  # Dimension for mask embedding
        
        # Project location queries to mask embeddings
        self.loc_to_mask_embed = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.output_dim // 2, self.mask_embed_dim),
        )
        # Project location queries to weights (for combining multiple queries)
        self.loc_to_weight = nn.Linear(self.output_dim, 1)
        
        # Project satellite features to mask embedding dimension
        self.sat_to_mask_feat = nn.Sequential(
            nn.Linear(self.output_dim, self.mask_embed_dim),
            nn.LayerNorm(self.mask_embed_dim),
        )
        
        # ============ 6. Camera Head (Yaw prediction) ============
        self.camera_head = CameraHead(
            dim_in=self.output_dim,
            trunk_depth=4,
            num_heads=num_heads,
            mlp_ratio=4,
            init_values=0.01,
            num_iterations=4,
        )
    
    def _soft_argmax(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Differentiable soft-argmax to extract 2D coordinates from heatmap."""
        B, H, W = heatmap.shape
        device = heatmap.device
        
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        
        y_expected = (heatmap.sum(dim=2) * y_coords).sum(dim=1)
        x_expected = (heatmap.sum(dim=1) * x_coords).sum(dim=1)
        
        return torch.stack([x_expected, y_expected], dim=1)
    
    def _freeze_vggt(self):
        """Freeze VGGT backbone."""
        for param in self.vggt.parameters():
            param.requires_grad = False
    
    def unfreeze_vggt(self):
        """Unfreeze VGGT backbone."""
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
            front_view: [B, 3, H, W] Front-view image
            satellite_view: [B, 3, H, W] Satellite image
            points: Tuple of (coords [B, N, 2], labels [B, N])
            boxes: [B, M, 4] in (x1, y1, x2, y2) format
            masks: [B, 1, H, W] Binary masks
        
        Returns:
            Dict containing:
                - pred_boxes: [B, N_obj, 4] BBox predictions (cx, cy, w, h)
                - bbox_scores: [B, N_obj] BBox confidence scores
                - heatmap: [B, H, W] Camera position heatmap
                - position: [B, 2] Camera position from heatmap
                - yaw_radians: [B] Camera yaw angle
                - yaw_degrees: [B] Camera yaw in degrees
        """
        B = front_view.shape[0]
        
        # ============ Step 1: VGGT Feature Extraction ============
        # Stack views: [satellite, front] to match VGGT convention
        images = torch.stack([satellite_view, front_view], dim=1)  # [B, 2, 3, H, W]
        
        vggt_outputs, patch_start_idx = self.vggt(images)
        # vggt_outputs: List of [B, 2, P_total, 2*C]
        
        # Get last layer features
        features = vggt_outputs[-1]  # [B, 2, P_total, 2*C]
        
        # Split satellite and front features
        sat_features = features[:, 0]    # [B, P_total, 2*C]
        front_features = features[:, 1]  # [B, P_total, 2*C]
        
        # Extract patch tokens (remove camera and register tokens)
        front_patch_features = front_features[:, patch_start_idx:]  # [B, P, 2*C]
        sat_patch_features = sat_features[:, patch_start_idx:]      # [B, P, 2*C]
        
        # Extract camera tokens for CameraHead
        sat_camera_token = sat_features[:, 0]      # [B, 2*C]
        front_camera_token = front_features[:, 0]  # [B, 2*C]
        
        # ============ Step 2: Prompt Encoding ============
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        # sparse_embeddings: [B, N_sparse, 2*C]
        # dense_embeddings: [B, 2*C, H', W']
        
        # ============ Step 3: Prompt Fusion (SAM-style) ============
        if self.use_prompt_fusion:
            # Use PromptFusionWithDense for complete fusion
            # Pass dense_embeddings only if masks were provided
            dense_for_fusion = dense_embeddings if masks is not None else None
            
            fused_sparse, fused_front, target_guidance = self.prompt_fusion(
                image_features=front_patch_features,     # [B, P, 2*C]
                sparse_embeddings=sparse_embeddings,     # [B, N_sparse, 2*C]
                dense_embeddings=dense_for_fusion,       # [B, 2*C, H', W'] or None
            )
            # fused_sparse: [B, N_sparse, 2*C] - target-aware prompt embeddings
            # fused_front: [B, P, 2*C] - F_target (prompt-guided front features)
            # target_guidance: [B, 2*C] - pooled target guidance vector
        else:
            fused_sparse = sparse_embeddings
            fused_front = front_patch_features
            target_guidance = front_patch_features.mean(dim=1)  # [B, 2*C]
        
        # ============ Step 4: Unified DETR Decoder ============
        # Prepare object queries with positional encoding
        obj_queries = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N_obj, C]
        obj_query_pos = self.object_query_pos.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N_obj, C]
        
        # Prepare location queries with positional encoding
        loc_queries = self.location_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N_loc, C]
        loc_query_pos = self.location_query_pos.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N_loc, C]
        
        # Add target guidance to ALL queries (content, not position)
        target_proj = self.target_guidance_proj(target_guidance)  # [B, C]
        obj_queries = obj_queries + target_proj.unsqueeze(1)
        loc_queries = loc_queries + target_proj.unsqueeze(1)
        
        # Concatenate queries and their positional encodings
        unified_queries = torch.cat([obj_queries, loc_queries], dim=1)  # [B, N_total, C]
        unified_query_pos = torch.cat([obj_query_pos, loc_query_pos], dim=1)  # [B, N_total, C]
        
        # Memory positional encoding for satellite features
        # sat_patch_features: [B, P, C] where P = 37*37 = 1369
        device = sat_patch_features.device
        memory_pos = self.memory_pos_embed(
            (self.num_patches_per_side, self.num_patches_per_side), device=device
        )  # [C, H, W]
        memory_pos = memory_pos.flatten(1).permute(1, 0)  # [P, C]
        memory_pos = memory_pos.unsqueeze(0).expand(B, -1, -1)  # [B, P, C]
        
        # Single decoder pass for all queries
        decoder_out = self.decoder(
            tgt=unified_queries,
            memory=sat_patch_features,
            pos=memory_pos,
            query_pos=unified_query_pos,
        )  # [1, B, N_total, C] or [B, N_total, C]
        
        # Handle output shape
        if decoder_out.dim() == 4:
            decoder_out = decoder_out[-1]  # Take last layer: [B, N_total, C]
        
        # Split decoder output back to object and location branches
        obj_decoder_out = decoder_out[:, :self.num_object_queries, :]  # [B, N_obj, C]
        loc_decoder_out = decoder_out[:, self.num_object_queries:, :]  # [B, G*G, C]
        
        # ============ Step 5a: BBox Predictions (from object queries) ============
        pred_boxes = self.bbox_head(obj_decoder_out).sigmoid()  # [B, N_obj, 4]
        bbox_scores = self.bbox_score_head(obj_decoder_out).squeeze(-1).sigmoid()  # [B, N_obj]
        
        # ============ Step 5b: Heatmap Predictions (from location queries) ============
        # Use Mask2Former-style dot product: query embeddings dot product with sat features
        
        # Project satellite features to mask embedding space
        H_feat = W_feat = self.num_patches_per_side  # 37
        sat_mask_feat = self.sat_to_mask_feat(sat_patch_features)  # [B, P, mask_dim]
        sat_mask_feat = sat_mask_feat.permute(0, 2, 1).view(B, -1, H_feat, W_feat)  # [B, mask_dim, H, W]
        
        # Generate mask embeddings from location queries
        mask_embeds = self.loc_to_mask_embed(loc_decoder_out)  # [B, N_loc, mask_dim]
        query_weights = self.loc_to_weight(loc_decoder_out).softmax(dim=1)  # [B, N_loc, 1]
        
        # Weighted combination of mask embeddings
        combined_mask_embed = (mask_embeds * query_weights).sum(dim=1)  # [B, mask_dim]
        
        # Dot product: [B, mask_dim] x [B, mask_dim, H, W] -> [B, H, W]
        heatmap_grid = torch.einsum('bc,bchw->bhw', combined_mask_embed, sat_mask_feat)
        
        # Upsample to target heatmap size first, then to image size
        heatmap_upsampled = F.interpolate(
            heatmap_grid.unsqueeze(1),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=True
        ).squeeze(1)  # [B, H, W]
        
        # Apply softmax to get probability distribution
        heatmap_flat = heatmap_upsampled.view(B, -1)
        heatmap_prob = F.softmax(heatmap_flat, dim=-1).view(B, self.img_size, self.img_size)
        
        # Extract position using soft-argmax
        position = self._soft_argmax(heatmap_prob)  # [B, 2]
        
        # ============ Step 6: Camera Yaw Prediction ============
        camera_output = self.camera_head(
            front_camera_token=front_camera_token,
            sat_camera_token=sat_camera_token,
        )
        # camera_output: {yaw_radians, yaw_degrees, quaternion, ...}
        
        # ============ Combine Outputs ============
        outputs = {
            # BBox detection (from object queries)
            'pred_boxes': pred_boxes,
            'bbox_scores': bbox_scores,
            
            # Camera position (from location queries)
            'heatmap': heatmap_prob,
            'position': position,
            'heatmap_logits': heatmap_grid,
            
            # Camera angle
            'yaw_radians': camera_output['yaw_radians'],
            'yaw_degrees': camera_output['yaw_degrees'],
            'quaternion': camera_output['quaternion'],
            'pose_enc': camera_output['pose_enc'],
            
            # Features for visualization/debugging
            'front_features': front_patch_features,
            'sat_features': sat_patch_features,
            'sparse_embeddings': sparse_embeddings,
            'fused_front_features': fused_front if self.use_prompt_fusion else front_patch_features,
            'target_guidance': target_guidance,
        }
        
        return outputs


def build_cross_view_localizer_detr(
    pretrained_vggt: Optional[str] = None,
    freeze_vggt: bool = True,
    **kwargs
) -> CrossViewLocalizerDETR:
    """
    Build CrossViewLocalizerDETR with optional pretrained VGGT weights.
    
    Args:
        pretrained_vggt: Path to pretrained VGGT checkpoint
        freeze_vggt: Whether to freeze VGGT backbone
        **kwargs: Additional arguments for CrossViewLocalizerDETR
    
    Returns:
        CrossViewLocalizerDETR model
    """
    model = CrossViewLocalizerDETR(freeze_vggt=freeze_vggt, **kwargs)
    
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
