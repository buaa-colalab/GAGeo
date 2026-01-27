#!/bin/bash
# Single GPU Training Script
# Usage: bash scripts/train_single.sh [config_file] [gpu_id]

set -e

CONFIG=${1:-"configs/test.yaml"}
GPU_ID=${2:-"7"}

echo "=========================================="
echo "Single GPU Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "GPU: $GPU_ID"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPU_ID
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

python train.py --config $CONFIG

echo "Training completed!"
