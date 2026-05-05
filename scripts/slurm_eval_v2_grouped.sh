#!/bin/bash
# Backward-compatible terminal grouped evaluation wrapper.
#
# This legacy filename now runs directly in the terminal
# with the `gageo` conda environment and local /mnt/data paths.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/data/wrp}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v4}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
CONDA_BIN="${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gageo}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/data/wrp/.cache}"

SAM_CKPT="${1:-/mnt/data/wrp/GaGeo/ckpt/sam2.1_hiera_large.pt}"
GPU_IDS="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
EXPRIMENT_NAME="${3:-gageo_pi3}"
MODEL_DIR="${4:-output_v3/${EXPRIMENT_NAME}}"
CHECKPOINT_NAME="${5:-best}"
VIEW_SUBSET="${6:-all}"
PROMPT_TYPES=(point bbox mask)

export ROOT_DIR WORKSPACE_NAME WORKSPACE_DIR
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"

cd "$WORKSPACE_DIR"
mkdir -p logs "${MODEL_DIR}"

OUT_JSON="${MODEL_DIR}/eval_grouped_$(date +%Y%m%d_%H%M%S).json"
TMP_JSON_DIR="${MODEL_DIR}/eval_grouped_parts_manual_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TMP_JSON_DIR"

echo "=========================================="
echo "Cross-View grouped evaluation (terminal)"
echo "=========================================="
echo "Workspace    : $WORKSPACE_DIR"
echo "Experiment   : $EXPRIMENT_NAME"
echo "Model dir    : $MODEL_DIR"
echo "Checkpoint   : $CHECKPOINT_NAME"
echo "SAM ckpt     : $SAM_CKPT"
echo "GPU ids      : $GPU_IDS"
echo "View subset  : $VIEW_SUBSET"
echo "Output JSON  : $OUT_JSON"
echo "=========================================="

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPUS=${#GPU_LIST[@]}
if [[ $NUM_GPUS -lt 1 ]]; then
  echo "[ERROR] No valid GPU ids parsed from: $GPU_IDS"
  exit 1
fi

pids=()

launch_eval() {
  local prompt_type="$1"
  local gpu_id="$2"
  local out_json="$3"

  echo "[Launch] prompt=${prompt_type} on GPU ${gpu_id} -> ${out_json}"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    python "${WORKSPACE_DIR}/evaluate_custom_v2.py" \
      --config "${WORKSPACE_DIR}/${MODEL_DIR}/config.yaml" \
      --checkpoint "${WORKSPACE_DIR}/${MODEL_DIR}/${CHECKPOINT_NAME}" \
      --splits test unseen_test \
      --prompt_types "$prompt_type" \
      --batch_size 8 \
      --num_workers 8 \
      --gpu 0 \
      --sam_checkpoint "$SAM_CKPT" \
      --sam_model_type vit_h \
      --view_subset "$VIEW_SUBSET" \
      --save_json "$out_json" &

  pids+=("$!")
}

for i in "${!PROMPT_TYPES[@]}"; do
  prompt="${PROMPT_TYPES[$i]}"
  gpu="${GPU_LIST[$((i % NUM_GPUS))]}"
  prompt_json="$TMP_JSON_DIR/${prompt}.json"

  while [[ $(jobs -rp | wc -l) -ge $NUM_GPUS ]]; do
    wait -n
  done
  launch_eval "$prompt" "$gpu" "$prompt_json"
done

echo "Waiting for prompt evaluations ..."
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output python - <<PY
import json
from pathlib import Path

tmp_dir = Path("${TMP_JSON_DIR}")
out_path = Path("${OUT_JSON}")
merged = {}
for jf in sorted(tmp_dir.glob("*.json")):
    with jf.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for split, prompt_dict in data.items():
        merged.setdefault(split, {})
        merged[split].update(prompt_dict)

with out_path.open("w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
print(f"Merged result saved to: {out_path}")
PY

echo "Evaluation completed."
