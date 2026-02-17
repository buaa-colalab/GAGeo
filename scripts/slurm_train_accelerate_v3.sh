#!/bin/bash
#SBATCH --job-name=cvloc_v3_${1:-default}
#SBATCH --output=logs/slurm_v3_${1:-default}_%j.out
#SBATCH --error=logs/slurm_v3_${1:-default}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --partition=vip_gpu_5090_scxi704

# ============================================
# SLURM Accelerate Training Script for Cross-View Localization V3
# Usage: sbatch scripts/slurm_train_accelerate_v3.sh [experiment_name] [config_file]
# Example: sbatch scripts/slurm_train_accelerate_v3.sh configs/default_v3.yaml
# ============================================

set -e

# Workspace path config
ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v3}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"
export ROOT_DIR WORKSPACE_NAME WORKSPACE_DIR

# Conda env: filtre
source ~/run/miniconda3/etc/profile.d/conda.sh
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
EXPRIMENT_NAME="${1:-default}"
export EXPRIMENT_NAME
TRAINING_CONFIG=${2:-"${WORKSPACE_DIR}/configs/default_v3.yaml"}
ACCELERATE_CONFIG="${WORKSPACE_DIR}/configs/accelerate_deepspeed_zero2.yaml"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
EXTRA_ARGS=("${@:3}")

echo "=========================================="
echo "SLURM Accelerate Training (V3)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Training Config: $TRAINING_CONFIG"
echo "Extra Args: ${EXTRA_ARGS[*]}"
echo "GPUs: $NUM_GPUS"
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "Conda Env: filtre"
echo "=========================================="

cd "$WORKSPACE_DIR"
mkdir -p logs

srun accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    "${WORKSPACE_DIR}/train_detr_v2.py" \
    --config "$TRAINING_CONFIG" \
    "${EXTRA_ARGS[@]}"

echo "Training completed!"