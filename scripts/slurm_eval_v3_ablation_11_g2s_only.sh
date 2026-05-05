#!/bin/bash
# Terminal grouped evaluation wrapper for ablation_11_g2s_only.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
SAM_CKPT="${1:-/mnt/data/wrp/GaGeo/ckpt/sam2.1_hiera_large.pt}"
GPU_IDS="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
EXPERIMENT_NAME="${3:-ablation_11_g2s_only}"
MODEL_DIR="${4:-output_v3/ablation_11_g2s_only}"
CHECKPOINT_NAME="${5:-best}"
VIEW_SUBSET="${6:-all}"

"${WORKSPACE_DIR}/scripts/slurm_eval_v2_grouped.sh"   "$SAM_CKPT"   "$GPU_IDS"   "$EXPERIMENT_NAME"   "$MODEL_DIR"   "$CHECKPOINT_NAME"   "$VIEW_SUBSET"
