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

# Conda env: filtre
source ~/run/miniconda3/bin/activate
conda activate filtre

# CUDA module
module load cuda

# Cache dirs (avoid home quota pressure)
export HF_HOME="/data/run01/scxi704/xhj/.cache/huggingface"
export TORCH_HOME="/data/run01/scxi704/xhj/.cache/torch"
export TMPDIR="/data/run01/scxi704/xhj/.cache/tmp"
export TRITON_CACHE_DIR="/data/run01/scxi704/xhj/.cache/triton"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

# Configuration
TRAINING_CONFIG=${1:-"configs/default_v2.yaml"}
ACCELERATE_CONFIG="configs/accelerate_deepspeed_zero2.yaml"
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

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

srun accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    train_detr_v2.py \
    --config "$TRAINING_CONFIG"

echo "Training completed!"