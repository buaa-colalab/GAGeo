#!/bin/bash

set -e

ROOT_DIR="${ROOT_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/colab}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v3}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

TRAINING_CONFIG=${1:-"${WORKSPACE_DIR}/configs/default_v3.yaml"}
OUTPUT_DIR="${WORKSPACE_DIR}/output_v3/ablation_5_all_off"

bash "${WORKSPACE_DIR}/mt_files/train.sh" \
  "$TRAINING_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --use_deep_supervision false \
  --use_contrastive_loss false \
  --use_rot_pos_supervision false
