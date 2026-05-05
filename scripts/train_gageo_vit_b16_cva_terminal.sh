#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/train_gageo_terminal.sh" \
  "gageo_vit_b16_cva" \
  "/mnt/data/wrp/location_v4/configs/gageo_vit_b16_cva.yaml" \
  "$@"
