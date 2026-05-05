#!/bin/bash
# Terminal training entrypoint for GAGeo experiments in this image.
#
# No scheduler is used. The script can run either:
# - a single local Python process
# - an Accelerate/DeepSpeed multi-GPU launch
# - a native PyTorch DDP multi-GPU launch

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
CONDA_BIN="${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gageo}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/data/wrp/.cache}"
START_TENSORBOARD="${START_TENSORBOARD:-1}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-0.0.0.0}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
MASTER_PORT="${MASTER_PORT:-29500}"
USER_TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-}"
USER_TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-}"
TB_PID=""
TB_PORT=""

EXPERIMENT_NAME="${1:?Usage: scripts/train_gageo_terminal.sh <experiment_name> <config_path> [extra train args...]}"
TRAINING_CONFIG="${2:?Usage: scripts/train_gageo_terminal.sh <experiment_name> <config_path> [extra train args...]}"
shift 2
EXTRA_ARGS=("$@")
DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT:-${WORKSPACE_DIR}/output_v3}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-}"

export ROOT_DIR="${ROOT_DIR:-/mnt/data/wrp}"
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

get_output_dir_override() {
  local idx=0
  local total="${#EXTRA_ARGS[@]}"
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
  "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output python - <<'PY' "$config_path"
import os
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

defaults = {
    "ROOT_DIR": os.environ.get("ROOT_DIR", "/mnt/data/wrp"),
    "WORKSPACE_NAME": os.environ.get("WORKSPACE_NAME", "location_v4"),
    "CHECKPOINT_DIR": os.environ.get("CHECKPOINT_DIR", "/mnt/data/wrp/checkpoints_offline"),
    "DATA_ROOT": os.environ.get("DATA_ROOT", "/mnt/data/wrp/eccv_data/data/urban"),
    "JSON_ROOT": os.environ.get("JSON_ROOT", "/mnt/data/wrp/eccv_data/data/json"),
}
defaults["WORKSPACE_DIR"] = os.environ.get(
    "WORKSPACE_DIR", f"{defaults['ROOT_DIR']}/{defaults['WORKSPACE_NAME']}"
)
defaults["OUTPUT_ROOT"] = os.environ.get("OUTPUT_ROOT", f"{defaults['WORKSPACE_DIR']}/output_v3")

value = str((cfg.get("checkpoint") or {}).get("output_dir") or "").strip()
value = os.path.expandvars(value)
for key, default in defaults.items():
    value = value.replace("${%s}" % key, default)
print(value)
PY
}

OUTPUT_DIR_OVERRIDE="$(get_output_dir_override || true)"
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  RUN_OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
elif [[ -n "$RUN_OUTPUT_DIR" ]]; then
  EXTRA_ARGS=(--output_dir "$RUN_OUTPUT_DIR" "${EXTRA_ARGS[@]}")
else
  RUN_OUTPUT_DIR="$(get_config_output_dir "$TRAINING_CONFIG")"
  if [[ -z "$RUN_OUTPUT_DIR" ]]; then
    RUN_OUTPUT_DIR="${DEFAULT_OUTPUT_ROOT}/${EXPERIMENT_NAME}"
    EXTRA_ARGS=(--output_dir "$RUN_OUTPUT_DIR" "${EXTRA_ARGS[@]}")
  fi
fi
TENSORBOARD_ROOT="${USER_TENSORBOARD_ROOT:-${RUN_OUTPUT_DIR}/tensorboard}"
TENSORBOARD_LOGDIR="${USER_TENSORBOARD_LOGDIR:-${TENSORBOARD_ROOT}}"
TENSORBOARD_PID_FILE="${RUN_OUTPUT_DIR}/tensorboard.pid"

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${#gpu_list[@]}"
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
  "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output python - <<'PY' "$config_path"
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mc = cfg.get("model", {})
if not mc.get("encoder_pretrained", True):
    raise SystemExit(0)

encoder_name = str(mc.get("encoder_name", "")).strip().lower()

backbone_type = str(mc.get("backbone_type", "")).strip().lower()
joint_vit_variant = str(mc.get("joint_vit_variant", encoder_name)).strip().lower()
encoder_weights = str(mc.get("encoder_weights", "")).strip()
joint_vit_weights = str(mc.get("joint_vit_weights", "") or "").strip()

if backbone_type in {"dinov2_joint_vit", "joint_vit", "dinov2_vit", "gageo_dinov2_vit"}:
    if joint_vit_weights:
        raise SystemExit(0)
    import torchvision.models as tv_models
    if joint_vit_variant in {"vit_h14", "vit-h14", "vit_h_14", "h14"}:
        weights = getattr(
            tv_models.ViT_H_14_Weights,
            encoder_weights or "IMAGENET1K_SWAG_E2E_V1",
            tv_models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
        )
        weights.get_state_dict(progress=True)
    else:
        weights = getattr(
            tv_models.ViT_B_16_Weights,
            encoder_weights or "IMAGENET1K_V1",
            tv_models.ViT_B_16_Weights.IMAGENET1K_V1,
        )
        weights.get_state_dict(progress=True)
elif encoder_name in {"vit_b16", "vit-b16", "vit_b_16", "imagenet_vit_b16"}:
    import torchvision.models as tv_models
    tv_models.ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
elif encoder_name in {"dinov2_g14", "dinov2-g14", "dinov2_vitg14", "dinov2_vitg14_reg"}:
    from models.dinov2.hub.utils import _DINOV2_BASE_URL
    import torch
    model_base_name = "dinov2_vitg14"
    model_full_name = "dinov2_vitg14"
    url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
    torch.hub.load_state_dict_from_url(url, map_location="cpu")
PY
}

pick_tensorboard_port() {
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

cleanup_tensorboard_server() {
  if [[ -f "$TENSORBOARD_PID_FILE" ]]; then
    rm -f "$TENSORBOARD_PID_FILE"
  fi

  if [[ -z "$TB_PID" ]]; then
    return
  fi

  if kill -0 "$TB_PID" >/dev/null 2>&1; then
    echo "Stopping TensorBoard (pid=${TB_PID}${TB_PORT:+, port=${TB_PORT}})"
    kill "$TB_PID" >/dev/null 2>&1 || true
    wait "$TB_PID" 2>/dev/null || true
  fi
}

trap cleanup_tensorboard_server EXIT INT TERM

start_tensorboard_server() {
  if [[ "$START_TENSORBOARD" == "0" || "$START_TENSORBOARD" == "false" ]]; then
    return
  fi

  local tb_port
  tb_port="$(pick_tensorboard_port "$TENSORBOARD_PORT")"
  mkdir -p "$(dirname "$TENSORBOARD_LOGDIR")" logs

  "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    tensorboard \
      --logdir "$TENSORBOARD_LOGDIR" \
      --host "$TENSORBOARD_HOST" \
      --port "$tb_port" \
      > "logs/tensorboard_${EXPERIMENT_NAME}_${tb_port}.log" 2>&1 &

  local tb_pid=$!
  TB_PID="$tb_pid"
  TB_PORT="$tb_port"
  printf '%s\n' "$TB_PID" > "$TENSORBOARD_PID_FILE"
  echo "TensorBoard  : http://127.0.0.1:${tb_port}"
  echo "TensorBoard  : pid=${tb_pid}, logdir=${TENSORBOARD_LOGDIR}"
}

VISIBLE_GPU_COUNT="$(count_visible_gpus)"
NUM_PROCESSES="${NUM_PROCESSES:-$VISIBLE_GPU_COUNT}"
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
echo "Conda env    : $CONDA_ENV"
echo "Output dir   : $RUN_OUTPUT_DIR"
echo "Extra args   : ${EXTRA_ARGS[*]:-<none>}"
echo "CUDA devices : ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "GPUs         : $VISIBLE_GPU_COUNT visible, $NUM_PROCESSES process(es)"
echo "Launch mode  : $DISTRIBUTED_BACKEND"
echo "TB logdir    : $TENSORBOARD_LOGDIR"
echo "=========================================="

start_tensorboard_server

if [[ "$DISTRIBUTED_BACKEND" == "accelerate" ]]; then
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${WORKSPACE_DIR}/configs/accelerate_deepspeed_zero2.yaml}"
  echo "Prefetching pretrained weights into local cache ..."
  PYTHONPATH="$WORKSPACE_DIR" MPLCONFIGDIR="$MPLCONFIGDIR" prefetch_pretrained_weights "$TRAINING_CONFIG"
  "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    accelerate launch \
      --config_file "$ACCELERATE_CONFIG" \
      --num_processes "$NUM_PROCESSES" \
      "${WORKSPACE_DIR}/train_detr_v2.py" \
      --config "$TRAINING_CONFIG" \
      "${EXTRA_ARGS[@]}"
elif [[ "$DISTRIBUTED_BACKEND" == "ddp" ]]; then
  DDP_MASTER_PORT="$(pick_master_port "$MASTER_PORT")"
  echo "Prefetching pretrained weights into local cache ..."
  PYTHONPATH="$WORKSPACE_DIR" MPLCONFIGDIR="$MPLCONFIGDIR" prefetch_pretrained_weights "$TRAINING_CONFIG"
  echo "DDP master   : 127.0.0.1:${DDP_MASTER_PORT}"
  "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    torchrun \
      --nproc_per_node "$NUM_PROCESSES" \
      --master_port "$DDP_MASTER_PORT" \
      "${WORKSPACE_DIR}/train_detr_v2_ddp.py" \
      --config "$TRAINING_CONFIG" \
      "${EXTRA_ARGS[@]}"
else
  "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    python "${WORKSPACE_DIR}/train_detr_v2.py" \
      --config "$TRAINING_CONFIG" \
      "${EXTRA_ARGS[@]}"
fi

echo "Training completed."
