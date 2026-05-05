#!/bin/bash
# Backward-compatible terminal training wrapper.
# Accepts either: <experiment_name> <config_path> [extra args...]
# or the historical form: <config_path> [extra args...].

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="/mnt/data/wrp/location_v4/configs/default_v3.yaml"

if [[ $# -ge 2 && "$2" == *.yaml ]]; then
  EXPERIMENT_NAME="$1"
  CONFIG_PATH="$2"
  shift 2
elif [[ $# -ge 1 && "$1" == *.yaml ]]; then
  CONFIG_PATH="$1"
  EXPERIMENT_NAME="${EXPRIMENT_NAME:-$(basename "${CONFIG_PATH%.yaml}")}"
  shift 1
else
  EXPERIMENT_NAME="${1:-default_v3}"
  CONFIG_PATH="${2:-$DEFAULT_CONFIG}"
  shift $(( $# >= 2 ? 2 : $# ))
fi

"${SCRIPT_DIR}/train_gageo_terminal.sh" "$EXPERIMENT_NAME" "$CONFIG_PATH" "$@"
