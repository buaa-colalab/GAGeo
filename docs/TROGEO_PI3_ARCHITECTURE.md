# TROGeo-Pi3 Architecture

This document describes the implemented `TROGeo-π³` variant from a data-flow
perspective, with explicit tensor shapes at each stage.

Relevant code:

- [CVOS-Code/model/TROGeo_pi3.py](/mnt/data/wrp/CVOS-Code/model/TROGeo_pi3.py)
- [CVOS-Code/model/pi3_feature_adapter.py](/mnt/data/wrp/CVOS-Code/model/pi3_feature_adapter.py)
- [CVOS-Code/model/TROGeo.py](/mnt/data/wrp/CVOS-Code/model/TROGeo.py)
- [CVOS-Code/model/attention.py](/mnt/data/wrp/CVOS-Code/model/attention.py)
- [CVOS-Code/train.py](/mnt/data/wrp/CVOS-Code/train.py)
- [CVOS-Code/dataset/eccv_loader.py](/mnt/data/wrp/CVOS-Code/dataset/eccv_loader.py)
- [location_v4/models/backbone/pi3_backbone.py](/mnt/data/wrp/location_v4/models/backbone/pi3_backbone.py)

## 1. Goal

`TROGeo-π³` is a strict backbone-swap experiment:

- keep the original TROGeo pipeline
- keep `CVOPM`
- keep the original bbox head
- keep the original mask head
- keep `SPS` as bbox-to-SAM post-processing
- replace only the visual backbone prior

The design constraint is stronger than “use Pi3 somewhere in the model”.

The implemented variant is:

- **original TROGeo conditioning and decoding**
- **joint-view Pi3 feature extraction**
- **a lightweight feature adapter back to the Swin-S interface**

It is intentionally **not**:

- GAGeo renamed as TROGeo
- Pi3 with GAGeo task tokens
- Pi3 with prompt-token decoding
- Pi3 with new bbox/mask/camera heads

## 2. Current Configuration

The implemented `TROGeo-π³` path is tied to the current `eccv_data` training
setup.

### 2.1 Data resolution

From the `eccv_data` pipeline:

- `front_view` is resized to `518 x 518`
- `satellite_view` is cropped or loaded at `518 x 518`

So the effective TROGeo training inputs are:

- `query_imgs`: `[B, 3, 518, 518]`
- `reference_imgs`: `[B, 3, 518, 518]`
- `mat_clickptns`: `[B, 518, 518]`

### 2.2 Pi3 backbone configuration

The adapter uses:

- `Pi3Backbone`
- `decoder_size="large"`
- `patch_size=14`
- `img_size=518`

This implies:

- patches per side: `518 / 14 = 37`
- patch tokens per image: `37 x 37 = 1369`
- register tokens per image: `5`
- total tokens per image inside Pi3 decoder: `1374`
- Pi3 output dim: `2048`

The current wrapper only supports `decoder_size="large"`, because the existing
`Pi3Backbone` implementation does not include an encoder-to-decoder projection
for `small` or `base`.

### 2.3 TROGeo downstream interface

The adapter is required to reproduce the feature interface expected by original
TROGeo on `518 x 518` input:

- `query_fvisu`: `[B, 768, 17, 17]`
- `reference_fvisu`: `[B, 768, 17, 17]`

That is the whole point of the adapter layer.

## 3. High-Level Pipeline

The full `TROGeo-π³` pipeline is:

1. Build a dense click heatmap from the monocular point
2. Inject the heatmap into the query image with the original TROGeo conv block
3. Run **joint-view** Pi3 over `[satellite, query]`
4. Extract patch tokens for both views
5. Project and resize Pi3 features back to the original Swin-S feature map
   interface
6. Feed those maps into the original TROGeo `CVOPM`
7. Decode with the original bbox head and mask head
8. In evaluation, pass the predicted bbox to `SPS` / SAM

The top-level forward in
[TROGeo_pi3.py](/mnt/data/wrp/CVOS-Code/model/TROGeo_pi3.py:52)
still returns:

- `outbox`
- `coodrs`

just like original TROGeo.

## 4. Input Tensors

The model forward signature is:

```python
model(query_imgs, reference_imgs, mat_clickptns)
```

Main tensor shapes:

- `query_imgs`: `[B, 3, 518, 518]`
- `reference_imgs`: `[B, 3, 518, 518]`
- `mat_clickptns`: `[B, 518, 518]`

In the `eccv_data` training path these come from
[eccv_loader.py](/mnt/data/wrp/CVOS-Code/dataset/eccv_loader.py):

- `query_imgs <- front_view`
- `reference_imgs <- satellite_view`
- `mat_clickptns <- mono_point` converted into a dense heatmap

Additional supervision tensors used by training:

- `ori_gt_bbox`: `[B, 4]` in absolute `xyxy` pixels on the `518 x 518`
  satellite crop
- `mask_rsimg`: `[B, 1, 518, 518]`

## 5. Data Adapter From `eccv_data`

The TROGeo training code expects tuple batches, while `eccv_data` provides a
dictionary with normalized boxes and segmentation masks. The adapter in
[eccv_loader.py](/mnt/data/wrp/CVOS-Code/dataset/eccv_loader.py)
bridges that mismatch.

### 5.1 Image normalization

`CrossViewDataset` returns tensors in `[0, 1]`.

The wrapper converts them to ImageNet-normalized tensors:

- input image: `[3, 518, 518]`
- normalized image: `[3, 518, 518]`

using:

- mean: `[0.485, 0.456, 0.406]`
- std: `[0.229, 0.224, 0.225]`

### 5.2 Click heatmap

`mono_point` is a single image-space point:

- `mono_point`: `[2] = [x, y]`

The wrapper constructs:

- `click_map`: `[518, 518]`

with the same dense radial heatmap pattern used by the baseline inference
scripts.

### 5.3 Bounding box conversion

`eccv_data` stores:

- `sat_bbox`: normalized `[cx, cy, w, h]`

The wrapper converts this to:

- absolute `xyxy` pixels: `[x1, y1, x2, y2]`

on the `518 x 518` crop, because TROGeo loss code expects absolute pixel boxes.

### 5.4 Returned training tuple

Each sample becomes:

```python
(query_img, sat_img, click_map, sat_bbox_xyxy, sat_mask)
```

with shapes:

- `query_img`: `[3, 518, 518]`
- `sat_img`: `[3, 518, 518]`
- `click_map`: `[518, 518]`
- `sat_bbox_xyxy`: `[4]`
- `sat_mask`: `[1, 518, 518]`

## 6. Query Conditioning Before Pi3

Original TROGeo injects the click map into the query RGB image before the
visual backbone. `TROGeo-π³` keeps that behavior exactly.

In [TROGeo_pi3.py](/mnt/data/wrp/CVOS-Code/model/TROGeo_pi3.py:53):

1. `mat_clickptns.unsqueeze(1)`
   - `[B, 518, 518] -> [B, 1, 518, 518]`
2. concatenate with query image
   - `[B, 3, 518, 518] + [B, 1, 518, 518] -> [B, 4, 518, 518]`
3. pass through `double_conv(4 -> 3)`
   - output: `[B, 3, 518, 518]`

This block is the original TROGeo query-side conditioning module.

It is **not** `SPS`.

`SPS` remains a later bbox-to-SAM stage used in evaluation and qualitative
comparison.

## 7. Joint-View Pi3 Stage

### 7.1 Why it is joint-view

The Pi3 backbone is not run as two independent single-image encoders.

Inside
[pi3_backbone.py](/mnt/data/wrp/location_v4/models/backbone/pi3_backbone.py:226),
the adapter stacks the two views:

- `images = torch.stack([satellite_view, front_view], dim=1)`
- shape: `[B, 2, 3, 518, 518]`

Then `Pi3Backbone.forward(images)` encodes both views jointly.

This matters because the Pi3 decoder alternates:

- local attention within each view
- global attention across views

So the backbone prior used here is genuinely multi-view.

### 7.2 Encoder output

For each image:

- DINOv2 encoder patch tokens: `1369`
- encoder channel dim: `1024`

After the Pi3 decoder:

- output tokens per image: `1374 = 5 register + 1369 patch`
- token dim: `2048`

So the full joint-view output shape is:

- `features`: `[B, 2, 1374, 2048]`

### 7.3 View split and patch extraction

`get_front_sat_features(...)` splits the joint output into two views:

- `sat_features`: `[B, 1374, 2048]`
- `front_features`: `[B, 1374, 2048]`

Then it removes the 5 register tokens:

- `sat_patch_features`: `[B, 1369, 2048]`
- `front_patch_features`: `[B, 1369, 2048]`

It also exposes one register token as a camera token:

- `sat_camera_token`: `[B, 2048]`
- `front_camera_token`: `[B, 2048]`

But `TROGeo-π³` does **not** use those camera tokens downstream.

Only patch features are consumed by the feature adapter.

## 8. Pi3-to-Swin Feature Adapter

The adapter in
[pi3_feature_adapter.py](/mnt/data/wrp/CVOS-Code/model/pi3_feature_adapter.py)
is intentionally small.

Its only job is to transform Pi3 patch features into the feature map interface
expected by original TROGeo.

### 8.1 Input to the adapter

For each branch:

- patch tokens: `[B, 1369, 2048]`

### 8.2 Reshape to 2D map

Because `1369 = 37 x 37`, the adapter reshapes tokens into:

- `[B, 2048, 37, 37]`

This is the native Pi3 patch grid for `518 x 518`, `patch_size=14`.

### 8.3 Channel projection

The adapter applies a `1 x 1` convolution:

- `Conv2d(2048 -> 768, kernel_size=1)`

Output:

- `[B, 768, 37, 37]`

This is the channel-alignment step back to TROGeo’s original embedding width.

### 8.4 Spatial resize

Original Swin-S on `518 x 518` produces a `17 x 17` feature map.

So the adapter performs:

- bilinear interpolate: `37 x 37 -> 17 x 17`

Final per-branch outputs:

- `query_fvisu`: `[B, 768, 17, 17]`
- `reference_fvisu`: `[B, 768, 17, 17]`

This is the exact downstream interface contract.

## 9. TROGeo Cross-View Fusion (`CVOPM`)

The original TROGeo `CVOPM` is implemented by
[SpatialTransformer](/mnt/data/wrp/CVOS-Code/model/attention.py:262).

`TROGeo-π³` keeps this module unchanged.

### 9.1 Query context sequence

After adapter output:

- `query_fvisu`: `[B, 768, 17, 17]`

The model flattens it into:

- `context = rearrange(query_fvisu, 'b c h w -> b (h w) c')`
- `context`: `[B, 289, 768]`

because:

- `17 x 17 = 289`

### 9.2 Reference stream

The reference branch remains a 2D map:

- `reference_fvisu`: `[B, 768, 17, 17]`

This is passed as `x` into `SpatialTransformer`.

### 9.3 Fused output

`CVOPM` returns:

- `fused_features`: `[B, 768, 17, 17]`

This output shape is the same as original TROGeo, which is why the original
heads can remain unchanged.

## 10. Prediction Heads

`TROGeo-π³` keeps the original two TROGeo heads.

### 10.1 Bounding box head

`fcn_out` is:

1. `ConvTranspose2d(768 -> 384, kernel=4, stride=2, padding=1)`
2. `ReLU`
3. `Conv2d(384 -> 45, kernel=1)`

Shape flow:

- input: `[B, 768, 17, 17]`
- after deconv: `[B, 384, 34, 34]`
- final bbox logits: `[B, 45, 34, 34]`

Interpretation:

- `45 = 9 anchors x 5 values`

The 5 values per anchor are:

- `tx`
- `ty`
- `tw`
- `th`
- `confidence`

During training the code reshapes this to:

- `[B, 9, 5, 34, 34]`

### 10.2 Mask / coordinate head

`coodrs_out` has the same stem:

1. `ConvTranspose2d(768 -> 384, kernel=4, stride=2, padding=1)`
2. `ReLU`
3. `Conv2d(384 -> 1, kernel=1)`

Shape flow:

- input: `[B, 768, 17, 17]`
- output: `[B, 1, 34, 34]`

This output is consumed by the original BCE + Dice mask loss.

## 11. Training Targets on `518 x 518`

The original official TROGeo code assumed a different input/output scale. For
`eccv_data`, the implementation keeps the same loss definitions but adapts
target construction to the actual output resolution.

### 11.1 BBox targets

`build_target(...)` already uses:

- `pred_anchor.shape[3]`

to derive the output grid size.

So with:

- `pred_anchor`: `[B, 9, 5, 34, 34]`
- `img_size = 518`

the bbox target path remains valid without changing the loss definition.

### 11.2 Mask targets

The original training code used a fixed `MaxPool2d(16, stride=16)` path.

That is not correct for the new `518 -> 34` output resolution.

So the implemented training path now does:

```python
coords_gt = F.adaptive_max_pool2d(mask_rsimg, output_size=pred_coords.shape[-2:])
```

Shape flow:

- `mask_rsimg`: `[B, 1, 518, 518]`
- `pred_coords`: `[B, 1, 34, 34]`
- `coords_gt`: `[B, 1, 34, 34]`

This keeps the original head and original loss, while aligning the target to
the actual model output size.

## 12. SPS Stage

`SPS` is not inside `TROGeo-π³` forward.

It remains an evaluation / post-processing stage:

1. run TROGeo to predict `raw_anchor`
2. decode the best bounding box
3. pass that bbox to SAM
4. get the final binary mask

So the semantic contract is still:

- `TROGeo-π³` predicts the coarse object box
- `SPS` refines that box into the final segmentation mask

This preserves the original paper pipeline.

## 13. End-to-End Shape Summary

For one training batch:

### 13.1 Inputs

- `query_imgs`: `[B, 3, 518, 518]`
- `reference_imgs`: `[B, 3, 518, 518]`
- `mat_clickptns`: `[B, 518, 518]`

### 13.2 Query conditioning

- click unsqueeze: `[B, 1, 518, 518]`
- concat with query: `[B, 4, 518, 518]`
- conditioned query RGB: `[B, 3, 518, 518]`

### 13.3 Joint-view Pi3

- stacked views: `[B, 2, 3, 518, 518]`
- Pi3 full tokens: `[B, 2, 1374, 2048]`
- patch tokens only: `[B, 2, 1369, 2048]`

### 13.4 Adapter outputs

- query map before resize: `[B, 2048, 37, 37]`
- reference map before resize: `[B, 2048, 37, 37]`
- projected maps: `[B, 768, 37, 37]`
- final TROGeo feature maps:
  - `query_fvisu`: `[B, 768, 17, 17]`
  - `reference_fvisu`: `[B, 768, 17, 17]`

### 13.5 CVOPM and heads

- query context: `[B, 289, 768]`
- fused map: `[B, 768, 17, 17]`
- bbox logits: `[B, 45, 34, 34]`
- mask logits: `[B, 1, 34, 34]`

### 13.6 Training targets

- bbox target: derived on `34 x 34` grid
- mask target: `[B, 1, 34, 34]` via adaptive pooling

## 14. What This Model Is, and Is Not

`TROGeo-π³` **is**:

- original TROGeo pipeline
- original `CVOPM`
- original bbox head
- original mask head
- original `SPS`
- stronger joint-view pretrained backbone prior

`TROGeo-π³` is **not**:

- a GAGeo decoder
- a task-token model
- a prompt-token model
- a new end-to-end segmentation architecture
- a new post-SAM refinement scheme

The only substantive architectural change is:

- **joint-view Pi3 patch features**
- followed by
- **a lightweight adapter back to the original Swin-S feature interface**

