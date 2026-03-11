#!/bin/bash
#SBATCH --job-name=cvloc_v3_ab13_nogmask
#SBATCH --output=logs/slurm_v3_ab13_nogmask_%j.out
#SBATCH --error=logs/slurm_v3_ab13_nogmask_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --partition=vip_gpu_5090_scxi704

set -e

ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v4}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

TRAINING_CONFIG=${1:-"${WORKSPACE_DIR}/configs/default_v3.yaml"}
OUTPUT_DIR="${WORKSPACE_DIR}/output_v3/ablation_13_global_attn_no_mask"

EXPRIMENT_NAME="ablation_13_global_attn_no_mask"

bash "${WORKSPACE_DIR}/scripts/slurm_train_accelerate_v3.sh" \
  "$EXPRIMENT_NAME" \
  "$TRAINING_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --use_deep_supervision true \
  --use_contrastive_loss true \
  --use_rot_pos_supervision true \
  --mask_inject_mode global_kv \
  --use_global_attn_mask false

