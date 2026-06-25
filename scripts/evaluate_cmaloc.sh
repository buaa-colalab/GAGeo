#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "${WORKSPACE_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${WORKSPACE_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

export ROOT_DIR="${ROOT_DIR:-$(dirname "$WORKSPACE_DIR")}"
export WORKSPACE_NAME="${WORKSPACE_NAME:-$(basename "$WORKSPACE_DIR")}"
export DATA_ROOT="${DATA_ROOT:-${WORKSPACE_DIR}/data/urban}"
export JSON_ROOT="${JSON_ROOT:-${WORKSPACE_DIR}/data/json}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORKSPACE_DIR}/checkpoints_offline}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_DIR}/outputs}"

CONFIG_PATH="${CONFIG_PATH:-${WORKSPACE_DIR}/configs/default.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${WORKSPACE_DIR}/GAGeo_ckpt/gageo/mp_rank_00_model_states.pt}"
SAVE_JSON="${SAVE_JSON:-${OUTPUT_ROOT}/cmaloc_metrics.json}"
mkdir -p "$(dirname "$SAVE_JSON")"

cd "$WORKSPACE_DIR"
"$PYTHON_BIN" evaluate_cmaloc.py \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --image_root "$DATA_ROOT" \
  --splits ${SPLITS:-test unseen_test} \
  --prompt_types ${PROMPT_TYPES:-point} \
  --batch_size "${BATCH_SIZE:-8}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --gpu "${GPU:-0}" \
  --view_subset "${VIEW_SUBSET:-all}" \
  --skip_sam \
  --save_json "$SAVE_JSON"
