#!/bin/bash
#SBATCH --job-name=cvloc_accel        # Job name
#SBATCH --output=logs/slurm_accelerate_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/slurm_accelerate_%j.err     # Standard error log
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Tasks per node (1 for multi-GPU on single node)
#SBATCH --cpus-per-task=64             # CPU cores per task
#SBATCH --gres=gpu:8                   # Number of GPUs (change to 2, 4, 8 as needed)
#SBATCH --mem=512G                     # Memory per node
#SBATCH --partition=vip_gpu_5090_scxi704

# ============================================
# SLURM Accelerate Training Script for Cross-View Localization
# ============================================
# Usage: sbatch scripts/slurm_train_accelerate.sh [config_file]
# Example: sbatch scripts/slurm_train_accelerate.sh configs/default.yaml
#
# To request multiple GPUs, modify the #SBATCH --gres=gpu:N line above
# For example: --gres=gpu:4 for 4 GPUs with DeepSpeed ZeRO-2
# ============================================

set -e

# Workspace path config
ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v4}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

# Load your conda environment
source /data/home/scxi704/run/miniconda3/bin/activate

# Activate your environment
conda activate filtre

# Load CUDA module (required for DeepSpeed)
module load cuda

# 设置缓存目录到大硬盘，避免写满 /data/home/scxi704
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export TORCH_HOME="${ROOT_DIR}/.cache/torch"
export TMPDIR="${ROOT_DIR}/.cache/tmp"
export TRITON_CACHE_DIR="${ROOT_DIR}/.cache/triton"
mkdir -p $HF_HOME $TORCH_HOME $TMPDIR $TRITON_CACHE_DIR

# Configuration
TRAINING_CONFIG=${1:-"${WORKSPACE_DIR}/configs/default.yaml"}
ACCELERATE_CONFIG="${WORKSPACE_DIR}/configs/accelerate_deepspeed_zero2.yaml"

# Get GPU count from SLURM
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "=========================================="
echo "SLURM Accelerate Training with DeepSpeed ZeRO-2"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Training Config: $TRAINING_CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "=========================================="

# Change to project directory
cd "$WORKSPACE_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs


# Run with accelerate
srun accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    "${WORKSPACE_DIR}/train_detr.py" \
    --config "$TRAINING_CONFIG"

echo "Training completed!"
