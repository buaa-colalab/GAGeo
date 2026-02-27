#!/bin/bash
#SBATCH --job-name=cvloc_v3_ab7_dshm
#SBATCH --output=logs/slurm_v3_ab7_dshm_%j.out
#SBATCH --error=logs/slurm_v3_ab7_dshm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --partition=vip_gpu_5090_scxi704

set -e

ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v3}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

TRAINING_CONFIG=${1:-"${WORKSPACE_DIR}/configs/default_v3.yaml"}
OUTPUT_DIR="${WORKSPACE_DIR}/output_v3/ablation_7_ds_heatmap"

EXPRIMENT_NAME="ablation_7_ds_heatmap"

bash "${WORKSPACE_DIR}/scripts/slurm_train_accelerate_v3.sh" \
  "$EXPRIMENT_NAME" \
  "$TRAINING_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --use_deep_supervision true \
  --use_contrastive_loss false \
  --use_rot_pos_supervision true \
  --use_heatmap_loss true
