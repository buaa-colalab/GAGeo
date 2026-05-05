#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/train_gageo_terminal.sh" \
  "gageo_dinov2_g14_cva" \
  "/mnt/data/wrp/location_v4/configs/gageo_dinov2_g14_cva.yaml" \
  "$@"
