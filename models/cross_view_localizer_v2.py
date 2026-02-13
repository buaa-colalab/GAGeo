# Cross-View Drone Localization V2 — Unified Backbone Architecture
# 
# Key changes from V1:
# - Prompt tokens and learnable queries are injected directly into Pi3 Backbone
# - Custom attention masks control token interactions (see docs/ARCHITECTURE_V2.md)
# - Mask head (SAM-style) added alongside BBox head
# - Deep supervision at decoder layers 4, 11, 17
# - No separate PromptFusion / QueryDecoder stages; Pi3 backbone handles everything
# - Mask prompt fused via element-wise addition to front view tokens

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .backbone.pi3_backbone_v2 import Pi3BackboneV2
from .backbone import load_pi3_weights
from .encoder.prompt_encoder import GeometryPromptEncoder, load_sam_prompt_encoder_weights
from .heads import BBoxHead, HeatmapHead, Pi3CameraHead, CrossViewContrastiveHead
from .heads.mask_head import SAMMaskHead


class CrossViewLocalizerV2(nn.Module):
    """
    Cross-View Drone Localization V2 with Unified Backbone.
    
    Pipeline:
    1. Input: Front-view image, Satellite image, Prompts (point/bbox/mask)
    2. SAM Prompt Encoder: Encode prompts -> sparse + dense embeddings
    3. Pi3 Backbone V2: Process all tokens together with custom attn masks
       - Sate + Front patches as two views
       - Learnable queries + prompt tokens appended to front view
       - Dense mask embedding added to front view tokens
       - Deep supervision at layers 4, 11, 17
    4. Task Heads: BBox, Mask, Heatmap, Camera from backbone outputs
    
    Args:
        img_size: Input image size (default 518)
        patch_size: Patch size (default 14)
        decoder_size: Pi3 decoder size ('small', 'base', 'large')
        num_learnable_tokens: Number of learnable query tokens (default 2)
        supervision_layers: Layers for deep supervision (default [3, 10, 16])
        supervision_weights: Weights for deep supervision (default [0.1, 0.3, 0.6])
        dropout: Dropout rate
        freeze_backbone: Freeze Pi3 backbone
        sam_embed_dim: SAM internal dimension (256)
    """
    
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        decoder_size: str = 'large',
        num_learnable_tokens: int = 2,
        supervision_layers: List[int] = None,
        supervision_weights: List[float] = None,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        contrastive: bool = True,
        contrastive_proj_dim: int = 256,
        contrastive_queue_size: int = 16384,
        contrastive_momentum: float = 0.999,
        contrastive_temperature: float = 0.07,
        sam_embed_dim: int = None,
        num_mask_tokens: int = 1,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size  # 37
        self.supervision_layers = supervision_layers or [3, 10, 16]
        self.supervision_weights = supervision_weights or [0.1, 0.3, 0.6]
        
        # ============ 1. Pi3 Backbone V2 ============
        self.backbone = Pi3BackboneV2(
            pos_type='rope100',
            decoder_size=decoder_size,
            img_size=img_size,
            patch_size=patch_size,
            num_learnable_tokens=num_learnable_tokens,
            supervision_layers=self.supervision_layers,
        )
        self.output_dim = self.backbone.output_dim  # 2048 for large
        self.dec_embed_dim = self.backbone.dec_embed_dim  # 1024 for large
        
        if freeze_backbone:
            self._freeze_backbone()
        
        # ============ 2. SAM Prompt Encoder ============
        # Internal dim = sam_embed_dim (256), output projects to dec_embed_dim (1024)
        # Note: V2 uses dec_embed_dim (1024) not output_dim (2048) because tokens
        # are injected into the decoder which operates at dec_embed_dim
        self._sam_embed_dim = sam_embed_dim
        self.prompt_encoder = GeometryPromptEncoder(
            embed_dim=self.dec_embed_dim,  # Target: decoder dimension
            image_embedding_size=(self.num_patches_per_side, self.num_patches_per_side),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
            sam_embed_dim=sam_embed_dim,
        )
        
        # ============ 3. Task Heads ============
        # BBox Head (uses learnable query 0)
        self.bbox_head = BBoxHead(
            hidden_dim=self.output_dim,
            num_classes=1,
        )
        
        # Mask Head (uses learnable query 0 + sate spatial features)
        self.mask_head = SAMMaskHead(
            hidden_dim=self.output_dim,
            output_size=img_size,
            num_mask_tokens=num_mask_tokens,
        )
        
        # Heatmap Head (uses learnable query 1 + sate spatial features)
        self.heatmap_head = HeatmapHead(
            hidden_dim=self.output_dim,
            output_size=img_size,
        )
        
        # Camera Head (uses front + sate patch features)
        self.camera_head = Pi3CameraHead(
            in_dim=self.output_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            depth=5,
            patch_size=patch_size,
            rope_freq=100.0,
        )
        
        # Contrastive Head
        self.contrastive_head = None
        if contrastive:
            self.contrastive_head = CrossViewContrastiveHead(
                in_dim=self.output_dim,
                proj_dim=contrastive_proj_dim,
                queue_size=contrastive_queue_size,
                momentum=contrastive_momentum,
                temperature=contrastive_temperature,
            )
        
        # ============ 4. Intermediate BBox/Mask Heads for Deep Supervision ============
        self.inter_bbox_heads = nn.ModuleDict()
        self.inter_mask_heads = nn.ModuleDict()
        for layer_idx in self.supervision_layers:
            self.inter_bbox_heads[str(layer_idx)] = BBoxHead(
                hidden_dim=self.output_dim,
                num_classes=1,
            )
            self.inter_mask_heads[str(layer_idx)] = SAMMaskHead(
                hidden_dim=self.output_dim,
                output_size=img_size,
                num_mask_tokens=num_mask_tokens,
            )
    
    def _freeze_backbone(self):
        """Freeze Pi3 backbone (encoder + decoder, but NOT learnable queries or projections)."""
        for name, param in self.backbone.named_parameters():
            if 'learnable_queries' in name or 'intermediate_projs' in name or 'prompt_proj' in name:
                continue  # Keep these trainable
            param.requires_grad = False
    
    def _freeze_prompt_encoder(self):
        """Freeze SAM prompt encoder (keep projection layers trainable)."""
        for name, param in self.prompt_encoder.named_parameters():
            if 'sparse_proj' in name or 'dense_proj' in name:
                continue
            param.requires_grad = False
    
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
            front_view: [B, 3, H, W]
            satellite_view: [B, 3, H, W]
            points: Tuple of (coords [B, N, 2], labels [B, N])
            boxes: [B, M, 4] in (x, y, w, h) format
            masks: [B, 1, H, W] Binary prompt masks
            mono_mask: [B, 1, H, W] Front-view seg mask for contrastive
            sat_mask: [B, 1, H, W] Satellite seg mask for contrastive
        
        Returns:
            Dict with all predictions
        """
        B = front_view.shape[0]
        
        # Ensure dtype consistency
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
        
        # ============ Step 1: Prompt Encoding ============
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points, boxes=boxes, masks=masks,
        )
        sparse_embeddings = sparse_embeddings.to(target_dtype)
        dense_embeddings = dense_embeddings.to(target_dtype)
        
        # Build prompt coordinates for RoPE
        prompt_coords = self._build_prompt_coords(points, boxes, sparse_embeddings, B)
        
        # Dense embeddings only used when mask prompt is given
        dense_for_backbone = dense_embeddings if masks is not None else None
        
        # ============ Step 2: Pi3 Backbone V2 ============
        backbone_out = self.backbone(
            front_view=front_view,
            satellite_view=satellite_view,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_for_backbone,
            prompt_coords=prompt_coords,
        )
        
        sate_features = backbone_out['sate_features']     # [B, 1369, 2048]
        front_features = backbone_out['front_features']    # [B, 1369, 2048]
        learnable_out = backbone_out['learnable_out']      # [B, 2, 2048]
        
        # Split learnable queries
        bbox_query = learnable_out[:, 0]   # [B, 2048] - for bbox + mask
        heatmap_query = learnable_out[:, 1]  # [B, 2048] - for heatmap
        
        # ============ Step 3: Final Task Heads ============
        # BBox prediction
        bbox_outputs = self.bbox_head(bbox_query.unsqueeze(1))  # expects [B, N, C]
        
        # Mask prediction (SAM-style)
        mask_outputs = self.mask_head(
            query_token=bbox_query,
            spatial_features=sate_features,
            spatial_size=(self.num_patches_per_side, self.num_patches_per_side),
        )
        
        # Heatmap prediction
        heatmap_outputs = self.heatmap_head(
            query_features=heatmap_query.unsqueeze(1),  # [B, 1, C]
            spatial_features=sate_features,
            spatial_size=(self.num_patches_per_side, self.num_patches_per_side),
        )
        
        # Camera rotation (uses all patch features)
        camera_output = self.camera_head(
            front_patch_features=front_features,
            sat_patch_features=sate_features,
            img_size=self.img_size,
        )
        
        # Contrastive loss
        contrastive_loss = None
        if self.contrastive_head is not None and mono_mask is not None and sat_mask is not None:
            contrastive_loss = self.contrastive_head(
                mono_features=front_features,
                sat_features=sate_features,
                mono_mask=mono_mask,
                sat_mask=sat_mask,
            )
        
        # ============ Step 4: Intermediate Deep Supervision ============
        intermediate_preds = {}
        for layer_idx, inter_data in backbone_out['intermediate'].items():
            inter_learn = inter_data['learnable']     # [B, 2, 2048]
            inter_sate = inter_data['sate_patches']   # [B, 1369, 2048]
            
            inter_bbox_query = inter_learn[:, 0]  # [B, 2048]
            
            # Intermediate BBox
            inter_bbox = self.inter_bbox_heads[str(layer_idx)](inter_bbox_query.unsqueeze(1))
            
            # Intermediate Mask
            inter_mask = self.inter_mask_heads[str(layer_idx)](
                query_token=inter_bbox_query,
                spatial_features=inter_sate,
                spatial_size=(self.num_patches_per_side, self.num_patches_per_side),
            )
            
            intermediate_preds[layer_idx] = {
                'pred_boxes': inter_bbox['pred_boxes'],
                'class_logits': inter_bbox['class_logits'],
                'bbox_scores': inter_bbox['bbox_scores'],
                'mask_logits': inter_mask['mask_logits'],
                'mask_pred': inter_mask['mask_pred'],
            }
        
        # ============ Combine Outputs ============
        result = {
            # BBox detection
            'pred_boxes': bbox_outputs['pred_boxes'],
            'bbox_scores': bbox_outputs['bbox_scores'],
            'class_logits': bbox_outputs['class_logits'],
            
            # Mask prediction
            'mask_logits': mask_outputs['mask_logits'],
            'mask_pred': mask_outputs['mask_pred'],
            'iou_pred': mask_outputs['iou_pred'],
            
            # Heatmap (camera position)
            'heatmap': heatmap_outputs['heatmap'],
            'position': heatmap_outputs['position'],
            'heatmap_logits': heatmap_outputs['heatmap_logits'],
            
            # Camera rotation
            'rotation_matrix': camera_output['rotation_matrix'],
            'yaw': camera_output['yaw'],
            'pitch': camera_output['pitch'],
            'roll': camera_output['roll'],
            
            # Deep supervision intermediate predictions
            'intermediate_preds': intermediate_preds,
            
            # Features for debugging
            'front_features': front_features,
            'sat_features': sate_features,
            'sparse_embeddings': sparse_embeddings,
        }
        
        if contrastive_loss is not None:
            result['contrastive_loss'] = contrastive_loss
        
        return result
    
    def _build_prompt_coords(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        sparse_embeddings: torch.Tensor,
        B: int,
    ) -> Optional[torch.Tensor]:
        """
        Build normalized coordinates for prompt tokens (used for RoPE).
        
        Returns:
            prompt_coords: [B, K, 2] normalized [0,1] coordinates, or None
        """
        coords_list = []
        
        if points is not None:
            point_coords = points[0]  # [B, N, 2]
            # Normalize to [0, 1]
            normalized = point_coords / self.img_size
            if boxes is None:
                # Padding point was added, append (0,0) coord for it
                padding_coord = torch.zeros(B, 1, 2, device=point_coords.device, dtype=point_coords.dtype)
                normalized = torch.cat([normalized, padding_coord], dim=1)
            coords_list.append(normalized)
        
        if boxes is not None:
            # boxes: [B, M, 4] in (x, y, w, h)
            # Each box produces 2 corner tokens
            x1 = boxes[:, :, 0:1] / self.img_size
            y1 = boxes[:, :, 1:2] / self.img_size
            w = boxes[:, :, 2:3] / self.img_size
            h = boxes[:, :, 3:4] / self.img_size
            x2 = x1 + w
            y2 = y1 + h
            
            corner1 = torch.cat([x1, y1], dim=-1)  # [B, M, 2]
            corner2 = torch.cat([x2, y2], dim=-1)  # [B, M, 2]
            # Interleave corners
            corners = torch.stack([corner1, corner2], dim=2).reshape(B, -1, 2)  # [B, M*2, 2]
            coords_list.append(corners)
        
        if not coords_list:
            return None
        
        return torch.cat(coords_list, dim=1)  # [B, K, 2]


def build_cross_view_localizer_v2(
    pretrained_pi3: Optional[str] = None,
    freeze_backbone: bool = False,
    freeze_prompt_encoder: bool = True,
    load_camera_head_weights: bool = True,
    sam_weights: Optional[str] = None,
    **kwargs
) -> CrossViewLocalizerV2:
    """
    Build CrossViewLocalizerV2 with optional pretrained weights.
    
    Args:
        pretrained_pi3: Path to pretrained Pi3 checkpoint
        freeze_backbone: Whether to freeze Pi3 backbone
        freeze_prompt_encoder: Whether to freeze SAM prompt encoder
        load_camera_head_weights: Whether to load camera head weights from Pi3 checkpoint
        sam_weights: Path to SAM2 checkpoint for prompt encoder weights
        **kwargs: Additional arguments for CrossViewLocalizerV2
    
    Returns:
        CrossViewLocalizerV2 model
    """
    model = CrossViewLocalizerV2(freeze_backbone=freeze_backbone, **kwargs)
    
    if pretrained_pi3 is not None:
        _load_pi3_weights_v2(model.backbone, pretrained_pi3)
        
        if load_camera_head_weights:
            _load_camera_head_from_pi3(model.camera_head, pretrained_pi3)
    
    if sam_weights is not None:
        load_sam_prompt_encoder_weights(model.prompt_encoder, sam_weights)
    
    if freeze_prompt_encoder:
        model._freeze_prompt_encoder()
    
    return model


def _load_pi3_weights_v2(backbone: Pi3BackboneV2, checkpoint_path: str):
    """Load Pi3 pretrained weights into V2 backbone."""
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    # Filter to encoder, decoder, register_token weights
    filtered = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.') or k.startswith('decoder.') or k.startswith('register_token'):
            filtered[k] = v
        if k.startswith('rope.') or k.startswith('position_getter.'):
            filtered[k] = v
    
    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    
    print(f"Loaded Pi3 weights into V2 backbone from {checkpoint_path}")
    print(f"  Loaded keys: {len(filtered)}")
    # Expected missing: learnable_queries, intermediate_projs, masked_blocks (wrappers)
    new_keys = [k for k in missing if not k.startswith('masked_blocks.')]
    if new_keys:
        print(f"  New (uninitialized) keys: {len(new_keys)}")
        for k in new_keys[:10]:
            print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")


def _load_camera_head_from_pi3(camera_head, checkpoint_path: str):
    """Load camera_decoder and camera_head weights from Pi3 checkpoint."""
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
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
