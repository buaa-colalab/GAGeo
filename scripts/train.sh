#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

EXPERIMENT_NAME="${1:-gageo}"
CONFIG_PATH="${2:-${WORKSPACE_DIR}/configs/default.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

export ROOT_DIR="${ROOT_DIR:-$(dirname "$WORKSPACE_DIR")}"
export WORKSPACE_NAME="${WORKSPACE_NAME:-$(basename "$WORKSPACE_DIR")}"
export WORKSPACE_DIR
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORKSPACE_DIR}/checkpoints_offline}"
export DATA_ROOT="${DATA_ROOT:-${WORKSPACE_DIR}/data/urban}"
export JSON_ROOT="${JSON_ROOT:-${WORKSPACE_DIR}/data/json}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_DIR}/outputs}"
export WANDB_PROJECT="${WANDB_PROJECT:-gageo}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"

CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$OUTPUT_ROOT" "${WORKSPACE_DIR}/logs"

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

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local visible_devices="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
    if [[ -z "$visible_devices" ]]; then
      echo "1"
      return
    fi
    local without_commas="${visible_devices//,/}"
    echo $(( ${#visible_devices} - ${#without_commas} + 1 ))
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus | wc -l
    return
  fi
  echo "1"
}

pick_master_port() {
  local port="${MASTER_PORT:-29500}"
  while true; do
    if command -v ss >/dev/null 2>&1; then
      if ! ss -ltn "( sport = :$port )" | grep -q ":$port"; then
        echo "$port"
        return
      fi
    else
      echo "$port"
      return
    fi
    port=$((port + 1))
  done
}

VISIBLE_GPU_COUNT="$(count_visible_gpus)"
NUM_PROCESSES="${NUM_PROCESSES:-$VISIBLE_GPU_COUNT}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-auto}"
if [[ "$DISTRIBUTED_BACKEND" == "auto" ]]; then
  if [[ "$NUM_PROCESSES" -gt 1 ]]; then
    DISTRIBUTED_BACKEND="accelerate"
  else
    DISTRIBUTED_BACKEND="single"
  fi
fi
if [[ "$NUM_PROCESSES" -le 1 ]]; then
  DISTRIBUTED_BACKEND="single"
fi

cd "$WORKSPACE_DIR"

echo "=========================================="
echo "GAGeo training"
echo "=========================================="
echo "Workspace    : $WORKSPACE_DIR"
echo "Experiment   : $EXPERIMENT_NAME"
echo "Config       : $CONFIG_PATH"
echo "Python       : $PYTHON_BIN"
echo "Output root  : $OUTPUT_ROOT"
echo "JSON root    : $JSON_ROOT"
echo "Data root    : $DATA_ROOT"
echo "Checkpoint   : $CHECKPOINT_DIR"
echo "CUDA devices : ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "Launch mode  : $DISTRIBUTED_BACKEND"
echo "Processes    : $NUM_PROCESSES"
echo "=========================================="

"$PYTHON_BIN" scripts/preflight_gageo_config.py "$CONFIG_PATH"

if [[ "$DISTRIBUTED_BACKEND" == "accelerate" ]]; then
  "$PYTHON_BIN" -m accelerate.commands.launch \
    --config_file "${ACCELERATE_CONFIG:-${WORKSPACE_DIR}/configs/accelerate_deepspeed_zero2.yaml}" \
    --num_processes "$NUM_PROCESSES" \
    train.py \
    --config "$CONFIG_PATH" \
    "${EXTRA_ARGS[@]}"
elif [[ "$DISTRIBUTED_BACKEND" == "ddp" ]]; then
  "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node "$NUM_PROCESSES" \
    --master_port "$(pick_master_port)" \
    train_ddp.py \
    --config "$CONFIG_PATH" \
    "${EXTRA_ARGS[@]}"
elif [[ "$DISTRIBUTED_BACKEND" == "single" ]]; then
  "$PYTHON_BIN" train.py --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
else
  echo "Unsupported DISTRIBUTED_BACKEND=$DISTRIBUTED_BACKEND" >&2
  exit 1
fi
