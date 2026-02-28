#!/bin/bash

set -e

ROOT_DIR="${ROOT_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/colab}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v4}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

TRAINING_CONFIG=${1:-"${WORKSPACE_DIR}/configs/default_v3.yaml"}

EXPRIMENT_NAME="ablation_2_contrastive_only"
OUTPUT_DIR="${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}"

bash "${WORKSPACE_DIR}/mt_files/train.sh" \
  "$EXPRIMENT_NAME" \
  "$TRAINING_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --use_deep_supervision false \
  --use_contrastive_loss true \
  --use_rot_pos_supervision false
