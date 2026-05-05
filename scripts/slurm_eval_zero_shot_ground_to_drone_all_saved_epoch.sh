#!/bin/bash

set -euo pipefail

# =========================================================
# Zero-shot Ground -> Drone multi-epoch evaluation
# =========================================================

ROOT_DIR=${ROOT_DIR:-"/mnt/data/wrp"}
WORKSPACE_NAME=${WORKSPACE_NAME:-"location_v4"}
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"
RUN_ROOT="${RUN_ROOT:-/mnt/data/wrp}"
CACHE_ROOT=${CACHE_ROOT:-"${ROOT_DIR}/.cache"}

EXPRIMENT_NAME="ablation_9_wo_contrastive"

# ===============================
# ⭐ Epoch Range (可配置)
# ===============================
START_EPOCH=${START_EPOCH:-1}
END_EPOCH=${END_EPOCH:-30}
EPOCH_STEP=${EPOCH_STEP:-2}

# ===============================
# Inputs
# ===============================
TRIPLET_JSON="${1:-${ROOT_DIR}/University-Release/verified_triplets_sam2_masks.json}"
ROOT_DIR_DATA="${2:-${ROOT_DIR}/University-Release}"
CKPT_ROOT="${3:-${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}}"
GPU_ID="${4:-0}"

CONFIG_PATH="${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/config.yaml"

if [[ ! -f "$TRIPLET_JSON" ]]; then
  echo "[ERROR] triplet json not found: $TRIPLET_JSON"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] config not found: $CONFIG_PATH"
  exit 1
fi

# ===============================
# Conda
# ===============================

# ===============================
# Cache
# ===============================
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

cd "$WORKSPACE_DIR"
mkdir -p logs output_v3

echo "=========================================="
echo "Multi-epoch Zero-shot Ground->Drone Eval"
echo "=========================================="
echo "Experiment: $EXPRIMENT_NAME"
echo "Epoch range: ${START_EPOCH} -> ${END_EPOCH}"
echo "Checkpoint root: $CKPT_ROOT"
echo "=========================================="

# =========================================================
# 🔥 Loop over epochs
# =========================================================

for ((EPOCH=$START_EPOCH; EPOCH<=END_EPOCH; EPOCH+=EPOCH_STEP)); do

    CHECKPOINT_NAME="epoch_${EPOCH}"
    CKPT_DIR="${CKPT_ROOT}/${CHECKPOINT_NAME}"

    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "[WARNING] Skip missing checkpoint: $CKPT_DIR"
        continue
    fi

    OUT_JSON="${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/eval_zero_shot_ground_to_drone_epoch_${EPOCH}.json"

    echo ""
    echo "------------------------------------------"
    echo "Evaluating epoch ${EPOCH}"
    echo "Checkpoint: $CKPT_DIR"
    echo "Output: $OUT_JSON"
    echo "------------------------------------------"

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}" run -n gageo --no-capture-output \
      python "${WORKSPACE_DIR}/evaluate_zero_shot_ground_to_drone.py" \
        --triplet_json "$TRIPLET_JSON" \
        --root_dir "$ROOT_DIR_DATA" \
        --config "$CONFIG_PATH" \
        --checkpoint "$CKPT_DIR" \
        --img_size 518 \
        --batch_size 8 \
        --num_workers 8 \
        --gpu 0 \
        --save_json "$OUT_JSON"

done

echo "=========================================="
echo "All evaluations finished."
echo "=========================================="