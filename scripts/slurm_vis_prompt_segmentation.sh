#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"/mnt/data/wrp"}
WORKSPACE_NAME=${WORKSPACE_NAME:-"location_v4"}
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

# ---------- Configurable arguments ----------
EXPERIMENT_NAME="${1:-ablation_4_all_on}"
SPLIT="${2:-unseen_test}"
NUM_SAMPLES="${3:-20}"
GPU_ID="${4:-0}"

MODEL_DIR="output_v3/${EXPERIMENT_NAME}"
CONFIG="${WORKSPACE_DIR}/${MODEL_DIR}/config.yaml"
CHECKPOINT="${WORKSPACE_DIR}/${MODEL_DIR}/best"
OUTPUT_DIR="${WORKSPACE_DIR}/vis_results/seg_prompt_compare/${EXPERIMENT_NAME}_${SPLIT}"

export ROOT_DIR WORKSPACE_NAME
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================================"
echo " Prompt Segmentation Visualisation"
echo " Experiment : ${EXPERIMENT_NAME}"
echo " Split      : ${SPLIT}"
echo " Num samples: ${NUM_SAMPLES}"
echo " Output dir : ${OUTPUT_DIR}"
echo "============================================================"

"${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}" run -n gageo --no-capture-output \
  python "${WORKSPACE_DIR}/visualize_prompt_segmentation.py" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --num_samples "${NUM_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    --gpu 0 \
    --ranking_metric mask_gap \
    --view_subsets drone_to_satellite ground_to_satellite \
    --img_gap 4 \
    --mask_alpha 0.45 \
    --color_point "0,255,0" \
    --color_bbox "0,128,255" \
    --color_mask "255,0,128" \
    --color_gt "255,255,0"

echo "Done. Results in: ${OUTPUT_DIR}"
