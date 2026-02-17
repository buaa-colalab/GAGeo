#!/bin/bash
#SBATCH --job-name=cvloc_zs_g2d
#SBATCH --output=logs/slurm_zs_g2d_%j.out
#SBATCH --error=logs/slurm_zs_g2d_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=vip_gpu_5090_scxi704

# ============================================
# SLURM Zero-shot Eval: ground -> drone (point prompt)
# Metrics: mean IoU / ACC@25 / ACC@50
# ============================================
# Usage:
#   sbatch scripts/slurm_eval_zero_shot_ground_to_drone.sh \
#       [triplet_json] [root_dir] [checkpoint_dir] [gpu_id]
#
# Example:
#   sbatch scripts/slurm_eval_zero_shot_ground_to_drone.sh \
#       ${ROOT_DIR}/University-Release/verified_triplets.json \
#       ${ROOT_DIR}/University-Release \
#       ${ROOT_DIR}/${WORKSPACE_NAME}/output_v2/best \
#       0
# ============================================

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "$0")/../.." && pwd)"}
WORKSPACE_NAME=${WORKSPACE_NAME:-"location"}
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"
RUN_ROOT="$(dirname "$ROOT_DIR")"
CACHE_ROOT=${CACHE_ROOT:-"${ROOT_DIR}/.cache"}
EXPRIMENT_NAME="ablation_4_all_on"

TRIPLET_JSON="${1:-${ROOT_DIR}/University-Release/verified_triplets.json}"
ROOT_DIR_DATA="${2:-${ROOT_DIR}/University-Release}"
CKPT_DIR="${3:-${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/best}"
GPU_ID="${4:-0}"

CONFIG_PATH="${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/config.yaml"
OUT_JSON="${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/eval_zero_shot_ground_to_drone_$(date +%Y%m%d_%H%M%S).json"
if [[ ! -f "$TRIPLET_JSON" ]]; then
  echo "[ERROR] triplet json not found: $TRIPLET_JSON"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] config not found: $CONFIG_PATH"
  exit 1
fi

# conda env
source "${RUN_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate filtre

module load cuda

# Cache dirs
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

cd "$WORKSPACE_DIR"
mkdir -p logs output_v2

echo "=========================================="
echo "Zero-shot Ground->Drone Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Triplet JSON: $TRIPLET_JSON"
echo "Root dir: $ROOT_DIR_DATA"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CKPT_DIR"
echo "GPU: $GPU_ID"
echo "Image size: 518 (raw 512 resized before model)"
echo "Output JSON: $OUT_JSON"
echo "=========================================="

CUDA_VISIBLE_DEVICES="$GPU_ID" \
"${RUN_ROOT}/miniconda3/bin/conda" run -n filtre --no-capture-output \
  python "${WORKSPACE_DIR}/evaluate_zero_shot_ground_to_drone.py" \
    --triplet_json "$TRIPLET_JSON" \
    --root_dir "$ROOT_DIR_DATA" \
    --config "$CONFIG_PATH" \
    --checkpoint "$CKPT_DIR" \
    --img_size 518 \
    --batch_size 8 \
    --num_workers 8 \
    --gpu 0 \
    --save_json "$OUT_JSON"

echo "Done. Result JSON: $OUT_JSON"
