#!/bin/bash
# Terminal wrapper for ablation_9_wo_contrastive. This keeps the old filename but runs locally.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
TRAINING_CONFIG="${1:-${WORKSPACE_DIR}/configs/default_v3.yaml}"
shift $(( $# >= 1 ? 1 : 0 ))
EXTRA_ARGS=("$@")
EXPERIMENT_NAME="ablation_9_wo_contrastive"
OUTPUT_DIR="${WORKSPACE_DIR}/output_v3/ablation_9_wo_contrastive"

"${WORKSPACE_DIR}/scripts/train_gageo_terminal.sh"   "$EXPERIMENT_NAME"   "$TRAINING_CONFIG"   --output_dir "$OUTPUT_DIR"   "--use_deep_supervision" "true" "--use_contrastive_loss" "false" "--use_rot_pos_supervision" "true" "--use_heatmap_loss" "true"   "${EXTRA_ARGS[@]}"
