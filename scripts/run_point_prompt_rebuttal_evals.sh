#!/bin/bash
# Run point-prompt rebuttal evaluations for one trained GAGeo experiment.
#
# D->S and G->S use CMA-Loc test/unseen_test splits. G->D needs explicit
# triplet JSONs because it is a separate zero-shot transfer benchmark.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/data/wrp}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v4}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
CONDA_BIN="${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gageo}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/data/wrp/.cache}"

EXPERIMENT_NAME="${1:?Usage: $0 <experiment_name> [model_dir] [checkpoint_name] [gpu_id] [g2d_seen_json] [g2d_unseen_json]}"
MODEL_DIR="${2:-output_v3/${EXPERIMENT_NAME}}"
CHECKPOINT_NAME="${3:-best}"
GPU_ID="${4:-0}"
G2D_SEEN_JSON="${5:-}"
G2D_UNSEEN_JSON="${6:-}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"

cd "$WORKSPACE_DIR"
OUT_DIR="${WORKSPACE_DIR}/${MODEL_DIR}/rebuttal_point"
mkdir -p "$OUT_DIR"

CONFIG="${WORKSPACE_DIR}/${MODEL_DIR}/config.yaml"
CKPT="${WORKSPACE_DIR}/${MODEL_DIR}/${CHECKPOINT_NAME}"

run_cmaloc_eval() {
  local subset="$1"
  local out_json="$2"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    python evaluate_custom_v2.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --splits test unseen_test \
    --prompt_types point \
    --batch_size 8 \
    --num_workers 8 \
    --gpu 0 \
    --view_subset "$subset" \
    --save_json "$out_json"
}

run_g2d_eval() {
  local split_name="$1"
  local triplet_json="$2"
  local out_json="$3"
  if [[ -z "$triplet_json" ]]; then
    echo "[SKIP] G->D ${split_name}: no triplet JSON provided"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    python evaluate_zero_shot_ground_to_drone.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --triplet_json "$triplet_json" \
    --gpu 0 \
    --save_json "$out_json"
}

run_cmaloc_eval "drone_to_satellite" "${OUT_DIR}/d2s_point_seen_unseen.json"
run_cmaloc_eval "ground_to_satellite" "${OUT_DIR}/g2s_point_seen_unseen.json"
run_g2d_eval "seen" "$G2D_SEEN_JSON" "${OUT_DIR}/g2d_point_seen.json"
run_g2d_eval "unseen" "$G2D_UNSEEN_JSON" "${OUT_DIR}/g2d_point_unseen.json"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
  python scripts/profile_gageo_model.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --device cuda \
  --batch_size 1 \
  --save_json "${OUT_DIR}/profile.json"

echo "Saved point-prompt rebuttal outputs to: ${OUT_DIR}"
