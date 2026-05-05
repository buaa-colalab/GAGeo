#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/data/wrp}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
CONDA_BIN="${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gageo}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/data/wrp/.cache}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gageo_pi3_frame_pos_cmaloc}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best}"
GPU_ID="${GPU_ID:-0}"
ROOT_DIR_DATA="${ROOT_DIR_DATA:-${ROOT_DIR}/University-Release}"

SEEN_JSON="${1:-${ROOT_DIR_DATA}/verified_triplets_sam2_masks.json}"
UNSEEN_JSON="${2:-}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"

cd "$WORKSPACE_DIR"

CONFIG_PATH="${WORKSPACE_DIR}/output_v3/${EXPERIMENT_NAME}/config.yaml"
CKPT_DIR="${WORKSPACE_DIR}/output_v3/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"
OUT_DIR="${WORKSPACE_DIR}/output_v3/${EXPERIMENT_NAME}/zero_shot_g2d"
mkdir -p "$OUT_DIR"

run_split() {
  local split_name="$1"
  local triplet_json="$2"
  if [[ -z "$triplet_json" ]]; then
    echo "[SKIP] ${split_name}: no triplet JSON provided"
    return 0
  fi
  if [[ ! -f "$triplet_json" ]]; then
    echo "[ERROR] triplet JSON not found: $triplet_json"
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    python evaluate_zero_shot_ground_to_drone.py \
      --triplet_json "$triplet_json" \
      --root_dir "$ROOT_DIR_DATA" \
      --config "$CONFIG_PATH" \
      --checkpoint "$CKPT_DIR" \
      --img_size 518 \
      --batch_size 8 \
      --num_workers 8 \
      --gpu 0 \
      --save_json "${OUT_DIR}/g2d_${split_name}.json"
}

run_split "seen" "$SEEN_JSON"
run_split "unseen" "$UNSEEN_JSON"

echo "Saved zero-shot G->D outputs to: ${OUT_DIR}"
