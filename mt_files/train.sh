#!/bin/bash
set -e

ROOT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/colab"
WORKSPACE_NAME=${WORKSPACE_NAME:-"location_v3"}
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"
RUN_ROOT="$(dirname "$ROOT_DIR")"
CACHE_ROOT=${CACHE_ROOT:-"${ROOT_DIR}/.cache"}
export ROOT_DIR WORKSPACE_NAME WORKSPACE_DIR

source "${ROOT_DIR}/location_v1/.venv/bin/activate"

# Cache dirs (avoid home quota pressure)
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

# Configuration
EXPERIMENT_NAME="${1:-default}"
export EXPERIMENT_NAME
# Backward compatibility (legacy typo used in some scripts/configs)
export EXPRIMENT_NAME="$EXPERIMENT_NAME"
TRAINING_CONFIG=${2:-"${WORKSPACE_DIR}/configs/default_v3.yaml"}
ACCELERATE_CONFIG="${WORKSPACE_DIR}/configs/accelerate_deepspeed_zero2.yaml"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
EXTRA_ARGS=("${@:3}")

echo "=========================================="
echo "SLURM Accelerate Training (V3)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Training Config: $TRAINING_CONFIG"
echo "Extra Args: ${EXTRA_ARGS[*]}"
echo "GPUs: $NUM_GPUS"
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "Conda Env: filtre"
echo "=========================================="

cd "$WORKSPACE_DIR"
mkdir -p logs

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    "${WORKSPACE_DIR}/train_detr_v2.py" \
    --config "$TRAINING_CONFIG" \
    "${EXTRA_ARGS[@]}"

echo "Training completed!"