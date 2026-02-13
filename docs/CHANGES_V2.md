# V2 Architecture Changes Summary

## Branch: `feature/unified-backbone-v2`

---

## Overview

V2 is a major architecture redesign that moves from a sequential pipeline (backbone → prompt fusion → decoder → heads) to a **unified backbone** where all tokens—including prompt embeddings and learnable queries—are injected directly into Pi3's decoder layers with custom attention masks.

---

## New Files Created

| File | Description |
|------|-------------|
| `models/backbone/pi3_backbone_v2.py` | Pi3BackboneV2 with token injection, attention masks, deep supervision |
| `models/heads/mask_head.py` | SAM-style mask prediction head with hypernetwork MLP |
| `models/cross_view_localizer_v2.py` | CrossViewLocalizerV2 main model + `build_cross_view_localizer_v2()` factory |
| `utils/losses_v2.py` | DETRCriterionV2 with mask loss, deep supervision, MSE heatmap |
| `configs/default_v2.yaml` | V2 configuration with new parameters |
| `train_detr_v2.py` | V2 training script with 3-group LR |
| `docs/ARCHITECTURE_V2.md` | Detailed architecture documentation |
| `docs/CHANGES_V2.md` | This file |

## Modified Files

| File | Changes |
|------|---------|
| `utils/weight_loader.py` | `get_param_groups()` now supports 3-group LR with `lr_new_tokens` parameter |
| `utils/__init__.py` | Added `DETRCriterionV2` export |
| `models/__init__.py` | Added V2 model exports |
| `models/backbone/__init__.py` | Added `Pi3BackboneV2` export |
| `models/heads/__init__.py` | Added `SAMMaskHead` export |

---

## Requirement Checklist

### ✅ 0. Architecture Analysis & Documentation
- Created `docs/ARCHITECTURE_V2.md` with token layout, attention mask rules, deep supervision

### ✅ 1. Git Branch
- Branch `feature/unified-backbone-v2` created from `stable-pi3-test`

### ✅ 2. SAM-style Mask Head + Loss
- `SAMMaskHead` in `models/heads/mask_head.py`:
  - ConvTranspose2d upscaling: 37→74→148 (spatial)
  - Hypernetwork MLP per mask token
  - IoU prediction head
  - Output interpolated to 518×518
- Mask loss in `losses_v2.py`:
  - BCE loss on logits + Dice loss on sigmoid probabilities
  - Weights: `weight_mask_bce=2.0`, `weight_mask_dice=5.0`

### ✅ 3. Deep Supervision at Layers 4, 11, 17
- `Pi3BackboneV2` collects intermediate outputs at configurable layers (default: [3, 10, 16] — 0-indexed)
- Each intermediate output projected from `dec_embed_dim` (1024) → `output_dim` (2048) via `intermediate_projs`
- Separate BBox + Mask heads per supervision layer (`inter_bbox_heads`, `inter_mask_heads`)
- Supervision weights: [0.1, 0.3, 0.6] (increasing with depth)
- `DETRCriterionV2` aggregates intermediate losses with layer weights × loss type weights

### ✅ 4. Differential Learning Rates (3 Groups)
- `get_param_groups()` updated with `lr_new_tokens` parameter:
  - **Backbone** (encoder + decoder): `lr_backbone = 1e-5`
  - **New tokens** (learnable queries, intermediate projections, prompt projections): `lr_new_tokens = 5e-4`
  - **Heads** (all task heads + intermediate supervision heads): `lr_heads = 1e-4`
- Pattern matching for new tokens: `backbone.learnable_queries`, `backbone.intermediate_projs.*`, `backbone.prompt_proj.*`, `prompt_encoder.sparse_proj*`, `prompt_encoder.dense_proj*`

### ✅ 5. SAM/Pi3 PE Alignment
- Prompt tokens receive RoPE positions from `_build_prompt_coords()` which maps prompt coordinates to the front-view spatial grid
- Pi3 backbone applies RoPE uniformly to all tokens (patches + prompts + learnable)
- Learnable queries get position (0.5, 0.5) — center of frame
- Point prompts use their actual coordinates; bbox prompts use corner coordinates
- No separate positional encoding from SAM is used in the backbone — pure RoPE

### ✅ 6. All Losses Checked
- **BBox**: L1 + GIoU with Hungarian matching ✓
- **Classification**: Sigmoid focal loss (α=0.25, γ=2.0), normalized by num_boxes ✓
- **Mask**: BCE + Dice (SAM-style) ✓
- **Heatmap**: Position MSE only (simplified) ✓
- **Rotation**: Geodesic loss on SO(3) with smooth option (1-cos(θ)) ✓
- **Contrastive**: MoCo cross-view loss (passthrough from model) ✓
- **Deep supervision**: Weighted intermediate BBox + Mask losses ✓

### ✅ 7. Heatmap Loss → MSE Only
- Removed KL divergence component
- Removed Gaussian target construction (`heatmap_sigma`, `heatmap_label_smooth` no longer needed)
- Simple `F.mse_loss(pred_pos, target_pos)` on normalized coordinates

### ✅ 8. SAM Mask Conv Weight Loading + Freeze
- `build_cross_view_localizer_v2()` loads SAM weights via `load_sam_prompt_encoder_weights()`
- `train_detr_v2.py` freezes mask downscaling layers when `freeze_mask_conv: true`:
  ```python
  for name, param in model.prompt_encoder.named_parameters():
      if 'mask_downscaling' in name:
          param.requires_grad = False
  ```
- SAM `mask_downscaling` operates at native 256-dim → projected to 1024 via `dense_proj`

### ✅ 9. Flash Attention Compatibility
- `MaskedFlashAttentionRope` in `pi3_backbone_v2.py`:
  - When `attn_mask=None`: uses `FLASH_ATTENTION` backend (fastest, used for encoder layers)
  - When `attn_mask` provided: falls back to `MATH` or `EFFICIENT_ATTENTION` backend (supports additive masks)
  - Fallback is automatic via `torch.backends.cuda.sdp_kernel()`
- Even-indexed decoder layers (local attn) have prompt self-block mask
- Odd-indexed decoder layers (global attn) have prompt↔sate mutual block + prompt self-block mask

### ✅ 10. Changes Summary
- This document (`docs/CHANGES_V2.md`)

---

## Token Layout (V2)

```
View 0 (Satellite): [register(5) | sate_patches(1369)]
View 1 (Front):     [register(5) | front_patches(1369) | learnable(2) | prompt(K)]
```

- K = number of prompt tokens (varies: 1 for point, 2 for bbox, 1 for each point in multi-point)
- Dense mask embedding is element-wise ADDED to front patch tokens (not separate tokens)
- After backbone:
  - Learnable token 0 → BBox head + Mask head
  - Learnable token 1 → Heatmap head (position prediction)

## Attention Mask Rules

**Local Attention (even layers — intra-view):**
- Prompt tokens can see front patches and themselves
- Prompt tokens CANNOT see each other
- All other tokens attend normally within their view

**Global Attention (odd layers — cross-view):**
- Prompt tokens CANNOT see sate tokens (mutual block)
- Sate tokens CANNOT see prompt tokens (mutual block)
- Prompt tokens CANNOT see each other (self-block)
- All other cross-view attention is normal

---

## Training Command

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
    train_detr_v2.py --config configs/default_v2.yaml
```
