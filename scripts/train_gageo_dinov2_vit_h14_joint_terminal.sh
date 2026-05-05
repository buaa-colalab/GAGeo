#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export WORKSPACE_DIR
"${SCRIPT_DIR}/train_gageo_terminal.sh" \
  "gageo_dinov2_vit_h14_joint" \
  "${WORKSPACE_DIR}/configs/gageo_dinov2_vit_h14_joint.yaml" \
  "$@"
