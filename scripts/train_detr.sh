#!/bin/bash
# Terminal training wrapper for the current image.
# Usage: bash scripts/train_detr.sh [config_file] [experiment_name] [extra train args...]

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
CONFIG="${1:-${WORKSPACE_DIR}/configs/default_v3.yaml}"
EXPERIMENT_NAME="${2:-default_train_detr}"
shift $(( $# >= 2 ? 2 : $# ))

"${WORKSPACE_DIR}/scripts/train_gageo_terminal.sh" "$EXPERIMENT_NAME" "$CONFIG" "$@"
