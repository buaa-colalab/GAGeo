#!/bin/bash
# Multi-GPU DDP Training Script
# Usage: bash scripts/train_ddp.sh [config_file] [gpu_ids]
# Example: bash scripts/train_ddp.sh configs/default.yaml "5,6,7"

set -e

# Activate conda filtre environment
source ~/run/miniconda3/bin/activate
conda activate filtre

CONFIG=${1:-"configs/test.yaml"}
GPU_IDS=${2:-"5,6,7"}

# Count number of GPUs
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

echo "=========================================="
echo "Multi-GPU DDP Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Use random port to avoid conflicts with other running processes
MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using master port: $MASTER_PORT"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_ddp.py --config $CONFIG

echo "Training completed!"
