#!/bin/bash
set -e

ROOT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/EVA/yangheqing/workspace/colab"
WORKSPACE_NAME=${WORKSPACE_NAME:-"location_v3"}
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"
export ROOT_DIR WORKSPACE_NAME WORKSPACE_DIR

source "${ROOT_DIR}/location_v2/.venv/activate"

# Cache dirs (avoid home quota pressure)
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

# Configuration
TRAINING_CONFIG=${1:-"${WORKSPACE_DIR}/configs/default_v3.yaml"}
ACCELERATE_CONFIG="${WORKSPACE_DIR}/configs/accelerate_deepspeed_zero2.yaml"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
EXTRA_ARGS=("${@:2}")

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