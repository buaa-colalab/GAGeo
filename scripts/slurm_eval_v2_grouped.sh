#!/bin/bash
#SBATCH --job-name=cvloc_v2_eval
#SBATCH --output=logs/slurm_v2_eval_%j.out
#SBATCH --error=logs/slurm_v2_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=vip_gpu_5090_scxi704

# ============================================
# SLURM Evaluate Script for Cross-View V2
# 评估 test + unseen_test，按 task/size/shape 分组
# ============================================
# Usage:
#   sbatch scripts/slurm_eval_v2_grouped.sh <sam_checkpoint> [gpu_id]
# Example:
#   sbatch scripts/slurm_eval_v2_grouped.sh /path/to/sam_vit_h_4b8939.pth 0
# ============================================

set -euo pipefail

SAM_CKPT="${1:-}"
GPU_ID="${2:-0}"

if [[ -z "$SAM_CKPT" ]]; then
  echo "[ERROR] Please provide SAM checkpoint path as first argument."
  echo "Usage: sbatch scripts/slurm_eval_v2_grouped.sh <sam_checkpoint> [gpu_id]"
  exit 1
fi

# conda env
source ~/run/miniconda3/bin/activate
conda activate filtre

module load cuda

# Cache dirs
export HF_HOME="/data/run01/scxi704/xhj/.cache/huggingface"
export TORCH_HOME="/data/run01/scxi704/xhj/.cache/torch"
export TMPDIR="/data/run01/scxi704/xhj/.cache/tmp"
export TRITON_CACHE_DIR="/data/run01/scxi704/xhj/.cache/triton"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

OUT_JSON="output_v2/eval_grouped_$(date +%Y%m%d_%H%M%S).json"

echo "=========================================="
echo "Cross-View V2 Grouped Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $GPU_ID"
echo "SAM checkpoint: $SAM_CKPT"
echo "Output JSON: $OUT_JSON"
echo "=========================================="

srun /data/home/scxi704/run/miniconda3/bin/conda run -n filtre --no-capture-output \
  python evaluate_custom_v2.py \
    --config /data/home/scxi704/run/xhj/location/output_v2/config.yaml \
    --checkpoint /data/home/scxi704/run/xhj/location/output_v2/best \
    --splits test unseen_test \
    --batch_size 8 \
    --num_workers 8 \
    --gpu "$GPU_ID" \
    --sam_checkpoint "$SAM_CKPT" \
    --sam_model_type vit_h \
    --save_json "$OUT_JSON"

echo "Evaluation completed!"