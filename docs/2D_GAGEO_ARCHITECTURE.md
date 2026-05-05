# 2D-GAGeo Architecture

This document describes the current 2D-GAGeo design implemented in this repo.
It reflects the `DINOv2 encoder + joint ViT fusion backbone` that replaced the
older `2d_cva` adapter design.

Relevant code:

- [models/cross_view_localizer_v2.py](/mnt/data/wrp/location_v4/models/cross_view_localizer_v2.py)
- [models/backbone/dinov2_joint_vit_backbone.py](/mnt/data/wrp/location_v4/models/backbone/dinov2_joint_vit_backbone.py)
- [models/encoder/prompt_encoder.py](/mnt/data/wrp/location_v4/models/encoder/prompt_encoder.py)
- [models/heads/bbox_head.py](/mnt/data/wrp/location_v4/models/heads/bbox_head.py)
- [models/heads/mask_head.py](/mnt/data/wrp/location_v4/models/heads/mask_head.py)
- [models/heads/heatmap_head.py](/mnt/data/wrp/location_v4/models/heads/heatmap_head.py)
- [models/heads/pi3_camera_head.py](/mnt/data/wrp/location_v4/models/heads/pi3_camera_head.py)

## 1. Goal

The current `2D-GAGeo` keeps:

- GAGeo's prompt encoder
- GAGeo's task heads
- GAGeo's task-token decoding interface
- GAGeo's DINOv2-L/14 visual encoder

It changes only the cross-view fusion prior:

- remove Pi3's 3D alternating local/global decoder
- remove the older 2D-CVA `BlockRope` stack
- replace them with pretrained torchvision ViT encoder blocks

So the model now tests:

- whether a pure 2D pretrained fusion backbone can work inside the full GAGeo
  pipeline
- while keeping the same downstream decoding and supervision interface

In code, this means:

- `backbone_type="dinov2_joint_vit"`
- `Pi3BackboneV2` is not used
- `CrossViewAdapter2D` is not used by this design
- the active backbone is `DINOv2JointViTBackbone`

## 2. Active Configurations

There are two active joint-ViT variants.

### 2.1 GAGeo-DINOv2-ViT-B

Config: [configs/gageo_dinov2_vit_b16_joint.yaml](/mnt/data/wrp/location_v4/configs/gageo_dinov2_vit_b16_joint.yaml)

- input size: `518 x 518`
- patch size: `14`
- patches per side: `518 / 14 = 37`
- patch tokens per image: `37 x 37 = 1369`
- frozen encoder: `DINOv2-L/14-reg`
- encoder token dim: `1024`
- joint fusion backbone: torchvision `ViT-B/16`
- ViT hidden dim: `768`
- ViT depth: `12`
- final backbone output dim: `2048`
- learnable task tokens: `2`
- deep supervision: disabled
- total params: about `491.1M`
- trainable params with frozen DINO: about `177.3M`

### 2.2 GAGeo-DINOv2-ViT-H

Config: [configs/gageo_dinov2_vit_h14_joint.yaml](/mnt/data/wrp/location_v4/configs/gageo_dinov2_vit_h14_joint.yaml)

- input size: `518 x 518`
- patch size: `14`
- patches per side: `37`
- patch tokens per image: `1369`
- frozen encoder: `DINOv2-L/14-reg`
- encoder token dim: `1024`
- joint fusion backbone: torchvision `ViT-H/14`
- ViT hidden dim: `1280`
- ViT depth: `32`
- final backbone output dim: `2048`
- learnable task tokens: `2`
- deep supervision: disabled
- total params: about `1102.8M`
- trainable params with frozen DINO: about `789.0M`

## 3. High-Level Pipeline

The full model is:

1. Input `front_view` and `satellite_view`
2. Encode geometric prompts with the SAM-style prompt encoder
3. Encode each image with shared frozen DINOv2-L/14
4. Project DINO tokens into the ViT fusion hidden space
5. Concatenate:
   satellite tokens, front tokens, learnable task tokens, prompt tokens
6. Run pretrained ViT encoder blocks over the joint token sequence
7. Project final tokens to the `2048`-dim GAGeo head space
8. Decode:
   bbox, mask, heatmap position, camera rotation

The top-level forward in
[cross_view_localizer_v2.py](/mnt/data/wrp/location_v4/models/cross_view_localizer_v2.py)
returns:

- `pred_boxes`
- `bbox_scores`
- `class_logits`
- `mask_logits`
- `mask_pred`
- `iou_pred`
- `heatmap`
- `position`
- `rotation_matrix`
- `yaw`, `pitch`, `roll`

## 4. Input Tensors

The model forward signature is:

```python
model(
    front_view,
    satellite_view,
    points=None,
    boxes=None,
    masks=None,
    mono_mask=None,
    sat_mask=None,
)
```

Main tensor shapes:

- `front_view`: `[B, 3, H, W]`
- `satellite_view`: `[B, 3, H, W]`
- `points[0]`: `[B, N_point, 2]`
- `points[1]`: `[B, N_point]`
- `boxes`: `[B, N_box, 4]` in `(x, y, w, h)`
- `masks`: `[B, 1, H, W]`

Before entering the backbone, both views are resized to `518 x 518` if needed.

## 5. Prompt Encoder Data Flow

Prompt encoding is implemented by
[GeometryPromptEncoder](/mnt/data/wrp/location_v4/models/encoder/prompt_encoder.py).

The prompt encoder is created with:

- `embed_dim = dec_embed_dim`
- for ViT-B joint: `dec_embed_dim = 768`
- for ViT-H joint: `dec_embed_dim = 1280`
- `sam_embed_dim = 256`

So the SAM-style internal prompt representation stays at `256`, then is
projected into the ViT fusion token dimension.

### 5.1 Sparse prompt tokens

Sparse prompt output shape:

- ViT-B joint: `sparse_embeddings = [B, K, 768]`
- ViT-H joint: `sparse_embeddings = [B, K, 1280]`

Where `K` depends on prompt type:

- point only:
  `K = N_point + 1`
- box only:
  `K = 2 * N_box`
- point + box:
  `K = N_point + 2 * N_box`

For the common single-point prompt case:

- `points[0] = [B, 1, 2]`
- `points[1] = [B, 1]`
- `K = 2`

So:

- ViT-B joint: `[B, 2, 768]`
- ViT-H joint: `[B, 2, 1280]`

### 5.2 Dense prompt map

Dense prompt output shape:

- ViT-B joint: `[B, 768, 37, 37]`
- ViT-H joint: `[B, 1280, 37, 37]`

When no mask prompt is provided, the prompt encoder still produces a learned
default dense embedding, but the backbone uses dense prompt injection only when
`masks is not None`.

### 5.3 Prompt coordinates

Prompt coordinates are built in
[CrossViewLocalizerV2._build_prompt_coords](/mnt/data/wrp/location_v4/models/cross_view_localizer_v2.py).

Output shape:

- `prompt_coords`: `[B, K, 2]`

These are normalized image coordinates in `[0, 1]`. In the joint ViT backbone,
they are not converted into RoPE positions. Instead, they are passed through a
small MLP and added to sparse prompt tokens as an explicit geometric bias.

## 6. DINOv2 Encoder Stage

The joint ViT backbone uses shared DINOv2-L/14 for both views.

Inside
[dinov2_joint_vit_backbone.py](/mnt/data/wrp/location_v4/models/backbone/dinov2_joint_vit_backbone.py):

1. resize both views to `518 x 518`
2. stack them:
   `[B, 3, 518, 518] + [B, 3, 518, 518] -> [B, 2, 3, 518, 518]`
3. flatten the view dimension:
   `[B, 2, 3, 518, 518] -> [B * 2, 3, 518, 518]`
4. run shared DINOv2 encoder

DINOv2 output:

- raw patch tokens: `[B * 2, 1369, 1024]`

Then reshape back to two views:

- `[B * 2, 1369, 1024] -> [B, 2, 1369, 1024]`

Split by view:

- `sat_tokens_dino`: `[B, 1369, 1024]`
- `front_tokens_dino`: `[B, 1369, 1024]`

Important point:

- DINOv2 is the only part loaded from the Pi3 checkpoint
- only keys under `encoder.*` are loaded
- Pi3 decoder weights are ignored
- DINOv2 is frozen during training in the active configs

## 7. Projection Into ViT Fusion Space

The DINO token dim `1024` is projected into the ViT hidden dim.

### 7.1 ViT-B joint

- input: `[B, 2, 1369, 1024]`
- projection `1024 -> 768`
- output: `[B, 2, 1369, 768]`

### 7.2 ViT-H joint

- input: `[B, 2, 1369, 1024]`
- projection `1024 -> 1280`
- output: `[B, 2, 1369, 1280]`

This projection is done by `self.dino_to_vit`.

## 8. Position and View Embeddings

The fusion backbone does not use RoPE.

Instead it uses:

- interpolated pretrained ViT patch positional embeddings
- learned view embeddings

### 8.1 Patch positional embedding

The pretrained ViT patch positional embedding is stored as:

- ViT-B pretrained shape: `[1, 197, 768]`
- ViT-H pretrained shape: `[1, 257, 1280]`

The class token position is discarded. The patch part is interpolated to the
`37 x 37` DINO grid:

- interpolated patch position:
  - ViT-B: `[1, 1369, 768]`
  - ViT-H: `[1, 1369, 1280]`

### 8.2 View embedding

Learned view embedding shape:

- ViT-B: `[1, 2, 1, 768]`
- ViT-H: `[1, 2, 1, 1280]`

It is broadcast and added so the model can distinguish:

- satellite tokens
- front-view tokens

After adding patch position and view embedding:

- ViT-B:
  `[B, 2, 1369, 768]`
- ViT-H:
  `[B, 2, 1369, 1280]`

## 9. Token Layout Before Joint ViT

The token sequence passed into the pretrained ViT blocks is:

```text
[satellite patch tokens | front patch tokens | learnable task tokens | sparse prompt tokens]
```

### 9.1 Satellite patch tokens

- ViT-B: `[B, 1369, 768]`
- ViT-H: `[B, 1369, 1280]`

### 9.2 Front patch tokens

- ViT-B: `[B, 1369, 768]`
- ViT-H: `[B, 1369, 1280]`

### 9.3 Learnable task tokens

The model keeps two learnable task tokens:

- `1` bbox/mask query token
- `1` heatmap query token

Shape:

- ViT-B: `[B, 2, 768]`
- ViT-H: `[B, 2, 1280]`

### 9.4 Sparse prompt tokens

If present:

- ViT-B: `[B, K, 768]`
- ViT-H: `[B, K, 1280]`

### 9.5 Dense mask injection

If `masks is not None`, dense prompt tokens are flattened:

- ViT-B:
  `[B, 768, 37, 37] -> [B, 1369, 768]`
- ViT-H:
  `[B, 1280, 37, 37] -> [B, 1369, 1280]`

Then they are added only to the front-view patch tokens:

- `front_tokens += dense_flat`

So dense prompts bias the query/front branch before joint fusion.

### 9.6 Final sequence length

If `K` sparse prompt tokens are present:

- total tokens:
  `1369 + 1369 + 2 + K = 2740 + K`

So the final ViT input is:

- ViT-B: `[B, 2740 + K, 768]`
- ViT-H: `[B, 2740 + K, 1280]`

## 10. Joint ViT Fusion Stage

The pretrained ViT is used only as a transformer over feature tokens.

It does not use:

- ViT patch embedding convolution
- ViT class token
- ViT classification head

It does use:

- pretrained transformer encoder blocks
- pretrained layer norm
- pretrained patch positional embedding, after interpolation

### 10.1 ViT-B joint

- sequence input: `[B, 2740 + K, 768]`
- depth: `12`
- hidden stays: `[B, 2740 + K, 768]`

### 10.2 ViT-H joint

- sequence input: `[B, 2740 + K, 1280]`
- depth: `32`
- hidden stays: `[B, 2740 + K, 1280]`

If intermediate supervision layers are configured, the backbone can capture
intermediate outputs from selected ViT layers. In the current configs,
supervision is disabled, so `intermediate = {}`.

## 11. Projection Back To GAGeo Head Space

After the final ViT block, the sequence is normalized with `vit_norm`, then
split back by ownership:

- satellite patch segment
- front patch segment
- learnable task-token segment

Each segment is projected to `output_dim = 2048` through `final_proj`.

### 11.1 Final backbone outputs

For both ViT-B and ViT-H variants:

- `sate_features`: `[B, 1369, 2048]`
- `front_features`: `[B, 1369, 2048]`
- `learnable_out`: `[B, 2, 2048]`
- `features`: `[B, 2, 1369, 2048]`

Additional pooled camera tokens are exposed as:

- `sate_camera_token`: `[B, 2048]`
- `front_camera_token`: `[B, 2048]`

These are simple means over patch features, but the camera head mainly consumes
the full patch-feature tensors.

## 12. Task Head Inputs

The downstream GAGeo decoding logic is unchanged.

### 12.1 Query split

`learnable_out` is split into:

- `bbox_queries = learnable_out[:, :1]`
- `heatmap_queries = learnable_out[:, 1:]`

So:

- `bbox_queries`: `[B, 1, 2048]`
- `heatmap_queries`: `[B, 1, 2048]`

### 12.2 BBox head

Input:

- query tokens: `[B, 1, 2048]`
- satellite spatial features: `[B, 1369, 2048]`
- spatial size: `(37, 37)`

Output:

- `pred_boxes`: `[B, 1, 4]`
- `class_logits`: `[B, 1, 1]`
- `bbox_scores`: `[B, 1]`

### 12.3 Mask head

Input:

- query tokens: `[B, 1, 2048]`
- satellite spatial features: `[B, 1369, 2048]`
- spatial size: `(37, 37)`

Output:

- `mask_logits`: `[B, 1, 518, 518]`
- `mask_pred`: `[B, 1, 518, 518]`
- `iou_pred`: `[B, 1]`

### 12.4 Heatmap head

The heatmap branch averages heatmap queries across query count. Since there is
only one heatmap token, this is still shape-preserving.

Input:

- `heatmap_query.mean(dim=1)`: `[B, 2048]`
- unsqueezed query: `[B, 1, 2048]`
- satellite spatial features: `[B, 1369, 2048]`

Output:

- `heatmap_logits`: `[B, 1, 518, 518]`
- `heatmap`: `[B, 518, 518]`
- `position`: `[B, 2]`

### 12.5 Camera head

Input:

- `front_patch_features`: `[B, 1369, 2048]`
- `sat_patch_features`: `[B, 1369, 2048]`

Internal camera decoder dim matches the backbone token dim:

- ViT-B joint:
  `dec_embed_dim = 768`
- ViT-H joint:
  `dec_embed_dim = 1280`

Output:

- `rotation_matrix`: `[B, 4, 4]`
- `yaw`: `[B]`
- `pitch`: `[B]`
- `roll`: `[B]`

## 13. Training Behavior

The active configs match GAGeo's encoder-freezing policy:

- `freeze_encoder: true`
- `freeze_dinov2: true`
- `freeze_decoder: false`

So:

- DINOv2 encoder is frozen
- ViT fusion blocks are trainable
- task heads are trainable
- prompt projection layers are trainable
- SAM prompt encoder core is frozen

This makes the comparison cleaner:

- the visual patch extractor stays the same as GAGeo
- the changed variable is the cross-view fusion prior

## 14. What Changed Relative To The Older 2D-CVA Design

The older 2D-CVA design used:

- a 2D encoder
- a randomly initialized `BlockRope` cross-view adapter stack
- explicit local/global alternating layers
- RoPE prompt positions

The current design uses:

- GAGeo's DINOv2-L encoder
- pretrained ViT-B or ViT-H transformer blocks
- one joint token sequence for both views
- standard learned position/view embeddings
- prompt-coordinate additive bias instead of RoPE

So the current 2D-GAGeo is closer to:

- `shared DINO visual tokenizer`
- `joint pretrained 2D transformer fusion`
- `unchanged GAGeo downstream decoding`

than to the previous `2d_cva` adapter baseline.
