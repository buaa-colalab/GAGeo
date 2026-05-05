#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTRIBUTED_BACKEND=ddp "${SCRIPT_DIR}/train_gageo_vit_b16_cva_terminal.sh" "$@"
