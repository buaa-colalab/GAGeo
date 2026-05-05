#!/bin/bash

# ============================================
# terminal Visualization: worst-K zero-shot ground -> drone
# Visualize pred/gt bbox on drone image + point prompt on ground image
# ============================================
# Usage:
#   bash scripts/slurm_vis_zero_shot_ground_to_drone_worst50.sh \
#       [triplet_json] [root_dir] [checkpoint_dir] [gpu_id] [checkpoint_name] [worst_k]
#
# Example:
#   bash scripts/slurm_vis_zero_shot_ground_to_drone_worst50.sh \
#       ${ROOT_DIR}/University-Release/verified_triplets_sam2_masks.json \
#       ${ROOT_DIR}/University-Release \
#       ${ROOT_DIR}/${WORKSPACE_NAME}/output_v3/ablation_4_all_on/best \
#       0 \
#       best \
#       50
# ============================================

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"/mnt/data/wrp"}
WORKSPACE_NAME=${WORKSPACE_NAME:-"location_v4"}
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"
RUN_ROOT="${RUN_ROOT:-/mnt/data/wrp}"
CACHE_ROOT=${CACHE_ROOT:-"${ROOT_DIR}/.cache"}
EXPRIMENT_NAME="ablation_4_all_on"
CHECKPOINT_NAME="${5:-best}"

TRIPLET_JSON="${1:-${ROOT_DIR}/University-Release/verified_triplets_sam2_masks.json}"
ROOT_DIR_DATA="${2:-${ROOT_DIR}/University-Release}"
CKPT_DIR="${3:-${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/${CHECKPOINT_NAME}}"
GPU_ID="${4:-0}"
WORST_K="${6:-50}"

CONFIG_PATH="${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/config.yaml"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/vis_worst_${WORST_K}_${STAMP}"
OUT_JSON="${OUT_DIR}/worst_samples_summary.json"

if [[ ! -f "$TRIPLET_JSON" ]]; then
  echo "[ERROR] triplet json not found: $TRIPLET_JSON"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] config not found: $CONFIG_PATH"
  exit 1
fi

# conda env

# Cache dirs
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

cd "$WORKSPACE_DIR"
mkdir -p logs output_v3 "$OUT_DIR"

echo "=========================================="
echo "Visualize Worst Zero-shot Ground->Drone"
echo "=========================================="
echo "Experiment name: $EXPRIMENT_NAME"
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Triplet JSON: $TRIPLET_JSON"
echo "Root dir: $ROOT_DIR_DATA"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CKPT_DIR"
echo "GPU: $GPU_ID"
echo "Image size: 518"
echo "Worst K: $WORST_K"
echo "Output dir: $OUT_DIR"
echo "=========================================="

CUDA_VISIBLE_DEVICES="$GPU_ID" \
"${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}" run -n gageo --no-capture-output \
  python "${WORKSPACE_DIR}/visualize_zero_shot_ground_to_drone_worst50.py" \
    --triplet_json "$TRIPLET_JSON" \
    --root_dir "$ROOT_DIR_DATA" \
    --config "$CONFIG_PATH" \
    --checkpoint "$CKPT_DIR" \
    --img_size 518 \
    --batch_size 8 \
    --num_workers 8 \
    --gpu 0 \
    --worst_k "$WORST_K" \
    --out_dir "$OUT_DIR" \
    --save_json "$OUT_JSON"

echo "Done. Visualization dir: $OUT_DIR"
echo "Summary JSON: $OUT_JSON"

