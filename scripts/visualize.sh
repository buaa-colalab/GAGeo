#!/bin/bash
# Visualization Script
# Usage: bash scripts/visualize.sh [checkpoint] [config] [num_samples] [gpu_id]

set -e

# Workspace path config
ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v3}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

CHECKPOINT=${1:-"${WORKSPACE_DIR}/output/test/best.pth"}
CONFIG=${2:-"${WORKSPACE_DIR}/configs/test.yaml"}
NUM_SAMPLES=${3:-20}
GPU_ID=${4:-"7"}

echo "=========================================="
echo "Visualization"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Samples: $NUM_SAMPLES"
echo "GPU: $GPU_ID"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPU_ID
export OPENBLAS_NUM_THREADS=4

cd "$WORKSPACE_DIR"

python "${WORKSPACE_DIR}/vis_detr.py" \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --num_samples $NUM_SAMPLES \
    --gpu 0

echo "Visualizations saved!"
