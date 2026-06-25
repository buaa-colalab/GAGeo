# CMA-Loc Data Directory

This directory is reserved for local CMA-Loc annotations and images. The Python dataset loader lives in `datasets/`.

Expected annotation files:

```text
data/json/train_all.json
data/json/val_all.json
data/json/test_all.json
data/json/unseen_test.json
```

Expected image layout:

```text
data/urban/<city>/mono/<mono_filename>
data/urban/<city>/sate/<sat_filename>
data/urban/<city>/crop_sate/<sat_filename>
```

Each annotation sample contains front-view and satellite filenames, prompt annotations, target annotations, and optional pose fields:

```json
{
  "city": "London",
  "mono_filename": "front.jpg",
  "mono_point": [256.0, 256.0],
  "mono_bbox": [200.0, 210.0, 80.0, 90.0],
  "mono_segmentation": {"size": [518, 518], "counts": "..."},
  "sat_filename": "satellite.jpg",
  "sate_point": [300.0, 280.0],
  "sate_bbox": [260.0, 240.0, 70.0, 80.0],
  "sate_segmentation": {"size": [1280, 1280], "counts": "..."},
  "relative_yaw": 0.0,
  "relative_pitch": 90.0,
  "relative_roll": 0.0,
  "camera_position": [640.0, 640.0]
}
```

Training loads raw satellite images from `sate/` and samples a crop that keeps both the target and camera position visible. Evaluation loads pre-cropped satellite images from `crop_sate/`.

## Coordinate Conventions

Point and prompt boxes are pixel coordinates. Training targets are normalized to `[0, 1]` in `cx, cy, w, h` format after crop and resize transforms. Masks are COCO RLE binary masks.
