#!/bin/bash
#SBATCH --job-name=cvloc_v2_accel
#SBATCH --output=logs/slurm_v2_accelerate_%j.out
#SBATCH --error=logs/slurm_v2_accelerate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --partition=vip_gpu_5090_scxi704

# ============================================
# SLURM Accelerate Training Script for Cross-View Localization V2
# Usage: sbatch scripts/slurm_train_accelerate_v2.sh [config_file]
# Example: sbatch scripts/slurm_train_accelerate_v2.sh configs/default_v2.yaml
# ============================================

set -e

# Workspace path config
ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_all_components}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

# Conda env: filtre
source /data/home/scxi704/run/miniconda3/bin/activate
conda activate filtre

# CUDA module
module load cuda

# Cache dirs (avoid home quota pressure)
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export TORCH_HOME="${ROOT_DIR}/.cache/torch"
export TMPDIR="${ROOT_DIR}/.cache/tmp"
export TRITON_CACHE_DIR="${ROOT_DIR}/.cache/triton"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

# Configuration
TRAINING_CONFIG=${1:-"${WORKSPACE_DIR}/configs/default_v2.yaml"}
ACCELERATE_CONFIG="${WORKSPACE_DIR}/configs/accelerate_deepspeed_zero2.yaml"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "=========================================="
echo "SLURM Accelerate Training (V2)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Training Config: $TRAINING_CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "Conda Env: filtre"
echo "=========================================="

cd "$WORKSPACE_DIR"
mkdir -p logs

srun accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    "${WORKSPACE_DIR}/train_detr_v2.py" \
    --config "$TRAINING_CONFIG"

echo "Training completed!"