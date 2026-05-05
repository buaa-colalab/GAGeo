#!/bin/bash
# Terminal wrapper for ablation_12_mask_inject_pre_backbone. This keeps the old filename but runs locally.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
TRAINING_CONFIG="${1:-${WORKSPACE_DIR}/configs/default_v3.yaml}"
shift $(( $# >= 1 ? 1 : 0 ))
EXTRA_ARGS=("$@")
EXPERIMENT_NAME="ablation_12_mask_inject_pre_backbone"
OUTPUT_DIR="${WORKSPACE_DIR}/output_v3/ablation_12_mask_inject_pre_backbone"

"${WORKSPACE_DIR}/scripts/train_gageo_terminal.sh"   "$EXPERIMENT_NAME"   "$TRAINING_CONFIG"   --output_dir "$OUTPUT_DIR"   "--use_deep_supervision" "true" "--use_contrastive_loss" "true" "--use_rot_pos_supervision" "true" "--mask_inject_mode" "pre_backbone" "--use_global_attn_mask" "true"   "${EXTRA_ARGS[@]}"
