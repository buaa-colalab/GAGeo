#!/bin/bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gageo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

"$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
  python "$WORKSPACE_DIR/scripts/download_required_checkpoints.py" "$@"
