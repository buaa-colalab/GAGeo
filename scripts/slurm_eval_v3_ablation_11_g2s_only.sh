#!/bin/bash
#SBATCH --job-name=cvloc_eval_ab11_g2s
#SBATCH --output=/data/home/scxi704/run/eval_logs/slurm_v3_eval_ab11_g2s_%j.out
#SBATCH --error=/data/home/scxi704/run/eval_logs/slurm_v3_eval_ab11_g2s_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --partition=vip_gpu_5090_scxi704

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"/data/home/scxi704/run/xhj"}
WORKSPACE_NAME=${WORKSPACE_NAME:-"location_v4"}
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

SAM_CKPT="${1:-}"
GPU_IDS="${2:-${CUDA_VISIBLE_DEVICES:-0,1,2}}"
EXPRIMENT_NAME="${3:-ablation_11_g2s_only}"

bash "${WORKSPACE_DIR}/scripts/slurm_eval_v2_grouped.sh" \
  "$SAM_CKPT" \
  "$GPU_IDS" \
  "$EXPRIMENT_NAME" \
  "output_v3/${EXPRIMENT_NAME}" \
  "best" \
  "ground_to_satellite"

