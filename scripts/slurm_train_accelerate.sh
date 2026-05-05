#!/bin/bash
# Terminal replacement for the old terminal accelerate entrypoint.
# Usage: bash scripts/slurm_train_accelerate.sh [config_file] [experiment_name] [extra train args...]

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
CONFIG="${1:-${WORKSPACE_DIR}/configs/default_v3.yaml}"
EXPERIMENT_NAME="${2:-default_accelerate}"
shift $(( $# >= 2 ? 2 : $# ))
USE_ACCELERATE="${USE_ACCELERATE:-1}" "${WORKSPACE_DIR}/scripts/train_gageo_terminal.sh" "$EXPERIMENT_NAME" "$CONFIG" "$@"
