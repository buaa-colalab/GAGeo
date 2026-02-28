#!/bin/bash
#SBATCH --job-name=v3_base_qual
#SBATCH --output=/data/home/scxi704/run/eval_logs/v3_base_qual_%j.out
#SBATCH --error=/data/home/scxi704/run/eval_logs/v3_base_qual_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=vip_gpu_5090_scxi704

set -euo pipefail

# Usage:
#   sbatch scripts/slurm_qualitative_compare_v3_baselines.sh \
#     [v3_ckpt] [cvos_ckpt] [detgeo_ckpt] [sam_ckpt] [top_k] [vis_k] [gap_mode] [gpu_id]
#
# Example:
#   sbatch scripts/slurm_qualitative_compare_v3_baselines.sh \
#     /data/home/scxi704/run/xhj/location_v4/output_v3/ablation_4_all_on/best \
#     /data/home/scxi704/run/baseline/CVOS-Code/saved_models/customdata_model_best.pth.tar \
#     /data/home/scxi704/run/baseline/DetGeo/saved_models/customdata_model_best.pth.tar \
#     /data/home/scxi704/run/baseline/CVOS-Code/segment_anything/weights/sam_vit_h_4b8939.pth \
#     200 20 abs 0

RUN_ROOT="/data/home/scxi704/run"
WORKSPACE_DIR="${RUN_ROOT}/xhj/location_v4"
PY="${RUN_ROOT}/miniconda3/envs/filtre/bin/python"

V3_CONFIG="${WORKSPACE_DIR}/output_v3/ablation_4_all_on/config.yaml"
V3_CKPT="${1:-${WORKSPACE_DIR}/output_v3/ablation_4_all_on/best}"
CVOS_CKPT="${2:-${RUN_ROOT}/baseline/CVOS-Code/saved_models/customdata_model_best.pth.tar}"
DETGEO_CKPT="${3:-${RUN_ROOT}/baseline/DetGeo/saved_models/customdata_model_best.pth.tar}"
SAM_CKPT="${4:-${RUN_ROOT}/baseline/CVOS-Code/segment_anything/weights/sam_vit_h_4b8939.pth}"
TOP_K="${5:-200}"
VIS_K="${6:-20}"
GAP_MODE="${7:-abs}"
GPU_ID="${8:-0}"

JSON_PATH="${RUN_ROOT}/xhj/data/json/unseen_test.json"
IMAGE_ROOT="${RUN_ROOT}/xhj/data/urban"
OUTPUT_ROOT="${WORKSPACE_DIR}/vis"
SUMMARY_JSON="${OUTPUT_ROOT}/selection_summary_$(date +%Y%m%d_%H%M%S).json"

source "${RUN_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate /data/home/scxi704/run/miniconda3/envs/filtre

cd "${WORKSPACE_DIR}"

echo "=========================================="
echo "V3 vs Baseline Qualitative Compare"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "GPU ID: ${GPU_ID}"
echo "JSON: ${JSON_PATH}"
echo "Image root: ${IMAGE_ROOT}"
echo "V3 config: ${V3_CONFIG}"
echo "V3 ckpt: ${V3_CKPT}"
echo "CVOS ckpt: ${CVOS_CKPT}"
echo "DetGeo ckpt: ${DETGEO_CKPT}"
echo "SAM ckpt: ${SAM_CKPT}"
echo "top_k_candidates: ${TOP_K}"
echo "vis_k: ${VIS_K}"
echo "gap_mode: ${GAP_MODE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Summary json: ${SUMMARY_JSON}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PY}" "${WORKSPACE_DIR}/qualitative_compare_v3_baselines.py" \
  --json_path "${JSON_PATH}" \
  --image_root "${IMAGE_ROOT}" \
  --v3_root "${WORKSPACE_DIR}" \
  --v3_config "${V3_CONFIG}" \
  --v3_checkpoint "${V3_CKPT}" \
  --cvos_root "${RUN_ROOT}/baseline/CVOS-Code" \
  --cvos_checkpoint "${CVOS_CKPT}" \
  --detgeo_root "${RUN_ROOT}/baseline/DetGeo" \
  --detgeo_checkpoint "${DETGEO_CKPT}" \
  --sam_checkpoint "${SAM_CKPT}" \
  --sam_model_type vit_h \
  --top_k_candidates "${TOP_K}" \
  --vis_k "${VIS_K}" \
  --gap_mode "${GAP_MODE}" \
  --output_root "${OUTPUT_ROOT}" \
  --save_summary_json "${SUMMARY_JSON}" \
  --gpu 0

echo "Qualitative comparison completed."

