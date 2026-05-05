#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTRIBUTED_BACKEND=ddp "${SCRIPT_DIR}/train_gageo_pi3_frame_pos_cmaloc_terminal.sh" "$@"
