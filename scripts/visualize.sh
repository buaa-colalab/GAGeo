#!/bin/bash
# Visualization Script
# Usage: bash scripts/visualize.sh [checkpoint] [config] [num_samples] [gpu_id]

set -e

CHECKPOINT=${1:-"./output/test/best.pth"}
CONFIG=${2:-"configs/test.yaml"}
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

python vis.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --num_samples $NUM_SAMPLES \
    --gpu 0

echo "Visualizations saved!"
