#!/bin/bash
#SBATCH --job-name=cvloc_ddp          # Job name
#SBATCH --output=logs/slurm_ddp_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/slurm_ddp_%j.err     # Standard error log
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Tasks per node (1 for multi-GPU on single node)
#SBATCH --cpus-per-task=64            # CPU cores per task
#SBATCH --gres=gpu:8                   # Number of GPUs (change to 2, 4, 8 as needed)
#SBATCH --mem=512G                     # Memory per node
#SBATCH --partition=vip_gpu_5090_scxi704

# ============================================
# SLURM DDP Training Script for Cross-View Localization
# ============================================
# Usage: sbatch scripts/slurm_train_ddp.sh [config_file]
# Example: sbatch scripts/slurm_train_ddp.sh configs/default.yaml
#
# To request multiple GPUs, modify the #SBATCH --gres=gpu:N line above
# For example: --gres=gpu:4 for 4 GPUs
# ============================================

set -e

# Workspace path config
ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_all_components}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

# Load your conda environment
source /data/home/scxi704/run/miniconda3/bin/activate

# Activate your environment
conda activate filtre

# Configuration
CONFIG=${1:-"${WORKSPACE_DIR}/configs/default.yaml"}

# Get GPU count from SLURM
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "=========================================="
echo "SLURM DDP Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

cd "$WORKSPACE_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    "${WORKSPACE_DIR}/train_ddp.py" \
    --config "$CONFIG"

echo "Training completed!"
