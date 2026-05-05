# Offline Rebuttal Training

These experiments can run on a machine without network access if the dataset
paths and checkpoint paths are provided in the YAML/config environment.

## Required Checkpoints

Prepare one checkpoint directory on a machine with network access:

```bash
cd /mnt/data/wrp/location_v4
bash scripts/download_required_checkpoints.sh --output_dir /mnt/data/wrp/checkpoints_offline
```

The directory must contain:

```text
pi3_model.safetensors
sam2.1_hiera_large.pt
vit_b_16_imagenet1k_v1.pth
vit_h_14_imagenet1k_swag_e2e_v1.pth
```

Copy this directory to the offline machine and set:

```bash
export CHECKPOINT_DIR=/path/to/checkpoints_offline
```

## Dataset Paths

GAGeo configs expand these environment variables:

```bash
export JSON_ROOT=/path/to/eccv_data/data/json
export DATA_ROOT=/path/to/eccv_data/data/urban
export OUTPUT_ROOT=/path/to/location_v4/output_v3
```

The JSON files can live separately from images. The image root only needs to
match the annotation layout used by the dataset loader:

```text
DATA_ROOT/<city>/mono/<image>
DATA_ROOT/<city>/sate/<image>
DATA_ROOT/<city>/crop_sate/<image>
```

You can also write absolute paths directly in each YAML:

```yaml
data:
  train_json: /path/to/train_all.json
  val_json: /path/to/val_all.json
  data_root: /path/to/images/urban
model:
  pi3_weights: /path/to/checkpoints/pi3_model.safetensors
  sam_weights: /path/to/checkpoints/sam2.1_hiera_large.pt
  joint_vit_weights: /path/to/checkpoints/vit_b_16_imagenet1k_v1.pth
checkpoint:
  output_dir: /path/to/output/experiment_name
```

## Experiment Scripts

GAGeo ViT-B:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29601 \
  bash scripts/train_gageo_dinov2_vit_b16_joint_ddp_terminal.sh
```

GAGeo ViT-H:

```bash
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29601 \
  bash scripts/train_gageo_dinov2_vit_h14_joint_ddp_terminal.sh
```

GAGeo MoCo queue size 4096:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29601 \
  bash scripts/train_gageo_moco_q4096_ddp_terminal.sh
```

GAGeo Pi3 frame-wise positional embedding:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29601 \
  bash scripts/train_gageo_pi3_frame_pos_cmaloc_ddp_terminal.sh
```

TROGeo-Pi3:

```bash
cd /mnt/data/wrp/CVOS-Code
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29601 \
  bash scripts/train_trogeo_pi3_offline_terminal.sh
```
