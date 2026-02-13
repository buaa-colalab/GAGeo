#!/bin/bash
# DDP Training Script (supports single or multi-GPU)
# Usage: bash scripts/train_single.sh [config_file] [gpu_ids]

set -e

# Activate conda filtre environment
source ~/run/miniconda3/bin/activate
conda activate filtre

CONFIG=${1:-"configs/test.yaml"}
GPU_ID=${2:-"5,6,7"}
NUM_GPUS=$(echo $GPU_ID | tr ',' '\n' | wc -l)

echo "=========================================="
echo "DDP Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "GPU: $GPU_ID (${NUM_GPUS} GPUs)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPU_ID
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Use random port to avoid conflicts
MASTER_PORT=$((29500 + RANDOM % 1000))

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    python train_ddp.py --config $CONFIG
else
    # Multi-GPU DDP training
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_ddp.py --config $CONFIG
fi

echo "Training completed!"
