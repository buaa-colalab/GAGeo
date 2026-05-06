#!/bin/bash
# Terminal training entrypoint for GAGeo experiments in this image.
#
# No scheduler is used. The script can run either:
# - a single local Python process
# - an Accelerate/DeepSpeed multi-GPU launch
# - a native PyTorch DDP multi-GPU launch

set -euo pipefail

ulimit -l unlimited || echo "[warn] ulimit -l unlimited failed; continue with current memlock limit"

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
CONDA_BIN="${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gageo}"
UV_BIN="${UV_BIN:-uv}"
UV_PROJECT_DIR="${UV_PROJECT_DIR:-${WORKSPACE_DIR}}"
ENV_MANAGER="${ENV_MANAGER:-auto}"
# Keep runtime caches inside the workspace by default so remote schedulers do
# not depend on a writable /mnt/data mount.
CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_DIR}/.cache}"
MASTER_PORT="${MASTER_PORT:-29500}"
WANDB_PROJECT="${WANDB_PROJECT:-location_v4}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"

EXPERIMENT_NAME="${1:?Usage: scripts/train_gageo_terminal.sh <experiment_name> <config_path> [extra train args...]}"
TRAINING_CONFIG="${2:?Usage: scripts/train_gageo_terminal.sh <experiment_name> <config_path> [extra train args...]}"
shift 2
EXTRA_ARGS=("$@")
EXTRA_ARG_COUNT="$#"
DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT:-${WORKSPACE_DIR}/output_v3}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-}"

export ROOT_DIR="${ROOT_DIR:-${WORKSPACE_DIR%/location_v4}}"
export WORKSPACE_NAME="${WORKSPACE_NAME:-location_v4}"
export WORKSPACE_DIR
export EXPRIMENT_NAME="$EXPERIMENT_NAME"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"

cd "$WORKSPACE_DIR"
mkdir -p logs output_v3

has_uv_runtime() {
  command -v "$UV_BIN" >/dev/null 2>&1 && [[ -f "${UV_PROJECT_DIR}/uv.lock" || -d "${UV_PROJECT_DIR}/.venv" ]]
}

resolve_env_manager() {
  case "$ENV_MANAGER" in
    uv|conda|current) echo "$ENV_MANAGER" ;;
    auto)
      if [[ -n "${VIRTUAL_ENV:-}" ]] || command -v python >/dev/null 2>&1; then
        echo "current"
      elif has_uv_runtime; then
        echo "uv"
      else
        echo "conda"
      fi
      ;;
    *)
      echo "Unsupported ENV_MANAGER=$ENV_MANAGER" >&2
      exit 1
      ;;
  esac
}

RUNTIME_MANAGER="$(resolve_env_manager)"

CURRENT_PYTHON_BIN="${CURRENT_PYTHON_BIN:-}"
if [[ -z "$CURRENT_PYTHON_BIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    CURRENT_PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "${WORKSPACE_DIR}/.venv/bin/python" ]]; then
    CURRENT_PYTHON_BIN="${WORKSPACE_DIR}/.venv/bin/python"
  else
    CURRENT_PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

validate_current_runtime() {
  if [[ "$RUNTIME_MANAGER" != "current" ]]; then
    return
  fi
  if [[ ! -x "$CURRENT_PYTHON_BIN" ]]; then
    echo "Missing executable Python for current runtime: $CURRENT_PYTHON_BIN" >&2
    exit 1
  fi
  "$CURRENT_PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' || {
    echo "Python >= 3.8 is required, got: $($CURRENT_PYTHON_BIN --version 2>&1)" >&2
    exit 1
  }
}

run_python_in_env() {
  if [[ "$RUNTIME_MANAGER" == "current" ]]; then
    "$CURRENT_PYTHON_BIN" "$@"
  elif [[ "$RUNTIME_MANAGER" == "uv" ]]; then
    "$UV_BIN" run --project "$UV_PROJECT_DIR" python "$@"
  else
    "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output python "$@"
  fi
}

run_tool_in_env() {
  local tool="$1"
  shift
  if [[ "$RUNTIME_MANAGER" == "current" ]]; then
    if [[ "$tool" == "torchrun" ]]; then
      # Avoid executing .venv/bin/torchrun directly: very long venv paths can
      # exceed the kernel shebang limit and fail with "bad interpreter".
      "$CURRENT_PYTHON_BIN" -m torch.distributed.run "$@"
    else
      "$tool" "$@"
    fi
  elif [[ "$RUNTIME_MANAGER" == "uv" ]]; then
    "$UV_BIN" run --project "$UV_PROJECT_DIR" "$tool" "$@"
  else
    "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output "$tool" "$@"
  fi
}

get_output_dir_override() {
  local idx=0
  local total="$EXTRA_ARG_COUNT"
  while [[ "$idx" -lt "$total" ]]; do
    if [[ "${EXTRA_ARGS[$idx]}" == "--output_dir" ]]; then
      if [[ $((idx + 1)) -ge "$total" ]]; then
        echo "Missing value after --output_dir" >&2
        exit 1
      fi
      echo "${EXTRA_ARGS[$((idx + 1))]}"
      return 0
    fi
    idx=$((idx + 1))
  done
  return 1
}

get_config_output_dir() {
  local config_path="$1"
  run_python_in_env "${WORKSPACE_DIR}/scripts/resolve_gageo_output_dir.py" "$config_path"
}

OUTPUT_DIR_OVERRIDE="$(get_output_dir_override || true)"
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  RUN_OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
elif [[ -n "$RUN_OUTPUT_DIR" ]]; then
  if [[ "$EXTRA_ARG_COUNT" -gt 0 ]]; then
    EXTRA_ARGS=(--output_dir "$RUN_OUTPUT_DIR" "${EXTRA_ARGS[@]}")
  else
    EXTRA_ARGS=(--output_dir "$RUN_OUTPUT_DIR")
  fi
  EXTRA_ARG_COUNT="${#EXTRA_ARGS[@]}"
else
  RUN_OUTPUT_DIR="$(get_config_output_dir "$TRAINING_CONFIG")"
  if [[ -z "$RUN_OUTPUT_DIR" ]]; then
    RUN_OUTPUT_DIR="${DEFAULT_OUTPUT_ROOT}/${EXPERIMENT_NAME}"
    if [[ "$EXTRA_ARG_COUNT" -gt 0 ]]; then
      EXTRA_ARGS=(--output_dir "$RUN_OUTPUT_DIR" "${EXTRA_ARGS[@]}")
    else
      EXTRA_ARGS=(--output_dir "$RUN_OUTPUT_DIR")
    fi
    EXTRA_ARG_COUNT="${#EXTRA_ARGS[@]}"
  fi
fi
LOG_DIR="${WORKSPACE_DIR}/logs"
RUN_LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"
WANDB_DIR="${WANDB_DIR:-${RUN_OUTPUT_DIR}/wandb}"
export WANDB_PROJECT
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"
export WANDB_DIR
if [[ -z "${WANDB_API_KEY:-}" && -n "$WANDB_API_KEY_FILE" && -f "$WANDB_API_KEY_FILE" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "$WANDB_API_KEY_FILE")"
fi
mkdir -p "$LOG_DIR" "$WANDB_DIR"
: > "$RUN_LOG_FILE"
exec > >(tee -a "$RUN_LOG_FILE") 2>&1

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

prefetch_pretrained_weights() {
  local config_path="$1"
  run_python_in_env "${WORKSPACE_DIR}/scripts/prefetch_gageo_pretrained.py" "$config_path"
}

pick_master_port() {
  local port="$1"
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
validate_current_runtime
USE_ACCELERATE_FLAG="${USE_ACCELERATE:-auto}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-}"
if [[ -z "$DISTRIBUTED_BACKEND" ]]; then
  if [[ "$USE_ACCELERATE_FLAG" == "1" || "$USE_ACCELERATE_FLAG" == "true" ]]; then
    DISTRIBUTED_BACKEND="accelerate"
  elif [[ "$USE_ACCELERATE_FLAG" == "0" || "$USE_ACCELERATE_FLAG" == "false" ]]; then
    DISTRIBUTED_BACKEND="single"
  elif [[ "$USE_ACCELERATE_FLAG" == "auto" && "$NUM_PROCESSES" -gt 1 ]]; then
    DISTRIBUTED_BACKEND="accelerate"
  else
    DISTRIBUTED_BACKEND="single"
  fi
fi

case "$DISTRIBUTED_BACKEND" in
  single|accelerate|ddp) ;;
  *)
    echo "Unsupported DISTRIBUTED_BACKEND=$DISTRIBUTED_BACKEND"
    echo "Use one of: single, accelerate, ddp"
    exit 1
    ;;
esac

if [[ "$NUM_PROCESSES" -le 1 ]]; then
  DISTRIBUTED_BACKEND="single"
fi

echo "=========================================="
echo "GAGeo terminal training"
echo "=========================================="
echo "Workspace    : $WORKSPACE_DIR"
echo "Experiment   : $EXPERIMENT_NAME"
echo "Config       : $TRAINING_CONFIG"
echo "Runtime      : $RUNTIME_MANAGER"
if [[ "$RUNTIME_MANAGER" == "current" ]]; then
  echo "Python       : $CURRENT_PYTHON_BIN"
  echo "Python ver   : $("$CURRENT_PYTHON_BIN" --version 2>&1)"
  echo "DDP launcher : $CURRENT_PYTHON_BIN -m torch.distributed.run"
fi
if [[ "$RUNTIME_MANAGER" == "uv" ]]; then
  echo "UV project   : $UV_PROJECT_DIR"
else
  echo "Conda env    : $CONDA_ENV"
fi
echo "Output dir   : $RUN_OUTPUT_DIR"
echo "Run log      : $RUN_LOG_FILE"
echo "W&B project  : $WANDB_PROJECT"
echo "W&B name     : ${WANDB_NAME}"
echo "W&B dir      : $WANDB_DIR"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "W&B auth     : configured"
else
  echo "W&B auth     : not configured (training will auto-disable W&B logging)"
fi
if [[ "$EXTRA_ARG_COUNT" -gt 0 ]]; then
  echo "Extra args   : ${EXTRA_ARGS[*]}"
else
  echo "Extra args   : <none>"
fi
echo "CUDA devices : ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "GPUs         : $VISIBLE_GPU_COUNT visible, $NUM_PROCESSES process(es)"
echo "Launch mode  : $DISTRIBUTED_BACKEND"
echo "=========================================="

echo "Running preflight checks ..."
run_python_in_env "${WORKSPACE_DIR}/scripts/preflight_gageo_config.py" "$TRAINING_CONFIG"

if [[ "$DISTRIBUTED_BACKEND" == "accelerate" ]]; then
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${WORKSPACE_DIR}/configs/accelerate_deepspeed_zero2.yaml}"
  echo "Prefetching pretrained weights into local cache ..."
  PYTHONPATH="$WORKSPACE_DIR" MPLCONFIGDIR="$MPLCONFIGDIR" prefetch_pretrained_weights "$TRAINING_CONFIG"
  if [[ "$EXTRA_ARG_COUNT" -gt 0 ]]; then
    run_tool_in_env accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_PROCESSES" \
        "${WORKSPACE_DIR}/train_detr_v2.py" \
        --config "$TRAINING_CONFIG" \
        "${EXTRA_ARGS[@]}"
  else
    run_tool_in_env accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_PROCESSES" \
        "${WORKSPACE_DIR}/train_detr_v2.py" \
        --config "$TRAINING_CONFIG"
  fi
elif [[ "$DISTRIBUTED_BACKEND" == "ddp" ]]; then
  DDP_MASTER_PORT="$(pick_master_port "$MASTER_PORT")"
  echo "Prefetching pretrained weights into local cache ..."
  PYTHONPATH="$WORKSPACE_DIR" MPLCONFIGDIR="$MPLCONFIGDIR" prefetch_pretrained_weights "$TRAINING_CONFIG"
  echo "DDP master   : 127.0.0.1:${DDP_MASTER_PORT}"
  if [[ "$EXTRA_ARG_COUNT" -gt 0 ]]; then
    run_tool_in_env torchrun \
        --nproc_per_node "$NUM_PROCESSES" \
        --master_port "$DDP_MASTER_PORT" \
        "${WORKSPACE_DIR}/train_detr_v2_ddp.py" \
        --config "$TRAINING_CONFIG" \
        "${EXTRA_ARGS[@]}"
  else
    run_tool_in_env torchrun \
        --nproc_per_node "$NUM_PROCESSES" \
        --master_port "$DDP_MASTER_PORT" \
        "${WORKSPACE_DIR}/train_detr_v2_ddp.py" \
        --config "$TRAINING_CONFIG"
  fi
else
  if [[ "$EXTRA_ARG_COUNT" -gt 0 ]]; then
    run_python_in_env "${WORKSPACE_DIR}/train_detr_v2.py" \
        --config "$TRAINING_CONFIG" \
        "${EXTRA_ARGS[@]}"
  else
    run_python_in_env "${WORKSPACE_DIR}/train_detr_v2.py" \
        --config "$TRAINING_CONFIG"
  fi
fi

echo "Training completed."
