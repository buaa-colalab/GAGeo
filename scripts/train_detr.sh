#!/bin/bash
# Train with Hugging Face Accelerate + DeepSpeed
# Usage: bash scripts/train_accelerate.sh [TRAINING_CONFIG] [GPU_IDS]

set -e
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Workspace path config
ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_all_components}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

# ============================================
# Training Parameters (from config file)
# ============================================
TRAINING_CONFIG=${1:-"${WORKSPACE_DIR}/configs/default.yaml"}

# ============================================
# Accelerate Parameters (hardware setup)
# ============================================
GPUS=${2:-"4,5,6,7"}
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

# Accelerate config file
ACCELERATE_CONFIG="${WORKSPACE_DIR}/configs/accelerate_deepspeed_zero2.yaml"

# Auto-update num_processes in accelerate config to match GPU count
if [ -f "$ACCELERATE_CONFIG" ]; then
    # Use sed to update num_processes in-place
    sed -i "s/^num_processes: .*/num_processes: $NUM_GPUS/" "$ACCELERATE_CONFIG"
fi

# ============================================
# Logging Setup
# ============================================
LOG_DIR="${WORKSPACE_DIR}/output/logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/train2.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Accelerate Training with DeepSpeed ZeRO-2"
echo "=========================================="
echo "Training Config: $TRAINING_CONFIG"
echo "GPUs: $GPUS (${NUM_GPUS} GPUs)"
echo "Accelerate Config: $ACCELERATE_CONFIG (num_processes auto-set to $NUM_GPUS)"
echo "Log File: $LOG_FILE"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=$GPUS

cd "$WORKSPACE_DIR"

# Run with accelerate and redirect output to log file
{
    echo "=== Training Started at $(date) ==="
    accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        "${WORKSPACE_DIR}/train_detr.py" \
        --config "$TRAINING_CONFIG"
    echo "=== Training Completed at $(date) ==="
} 2>&1 | tee $LOG_FILE

echo "Training completed! Log saved to: $LOG_FILE"
