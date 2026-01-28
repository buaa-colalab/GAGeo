#!/bin/bash
# Train with Hugging Face Accelerate + DeepSpeed
# Usage: bash scripts/train_accelerate.sh [TRAINING_CONFIG] [GPU_IDS]

set -e
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# ============================================
# Training Parameters (from config file)
# ============================================
TRAINING_CONFIG=${1:-"configs/default.yaml"}

# ============================================
# Accelerate Parameters (hardware setup)
# ============================================
GPUS=${2:-"0,1,2,3,4,5"}
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

# Accelerate config file
ACCELERATE_CONFIG="configs/accelerate_deepspeed_zero2.yaml"

# Auto-update num_processes in accelerate config to match GPU count
if [ -f "$ACCELERATE_CONFIG" ]; then
    # Use sed to update num_processes in-place
    sed -i "s/^num_processes: .*/num_processes: $NUM_GPUS/" "$ACCELERATE_CONFIG"
fi

echo "=========================================="
echo "Accelerate Training with DeepSpeed ZeRO-2"
echo "=========================================="
echo "Training Config: $TRAINING_CONFIG"
echo "GPUs: $GPUS (${NUM_GPUS} GPUs)"
echo "Accelerate Config: $ACCELERATE_CONFIG (num_processes auto-set to $NUM_GPUS)"
echo "=========================================="

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=$GPUS

# Run with accelerate
accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    train_accelerate.py \
    --config $TRAINING_CONFIG

echo "Training completed!"
