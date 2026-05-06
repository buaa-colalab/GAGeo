#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_MANAGER="${ENV_MANAGER:-auto}"
CONDA_BIN="${CONDA_BIN:-/mnt/data/wrp/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gageo}"
UV_BIN="${UV_BIN:-uv}"
UV_PROJECT_DIR="${UV_PROJECT_DIR:-${WORKSPACE_DIR}}"

if [[ "$ENV_MANAGER" == "current" || ( "$ENV_MANAGER" == "auto" && -n "${VIRTUAL_ENV:-}" ) ]]; then
  python "$WORKSPACE_DIR/scripts/download_required_checkpoints.py" "$@"
elif [[ "$ENV_MANAGER" == "uv" || ( "$ENV_MANAGER" == "auto" && -x "${WORKSPACE_DIR}/.venv/bin/python" ) ]]; then
  if command -v "$UV_BIN" >/dev/null 2>&1; then
    "$UV_BIN" run --project "$UV_PROJECT_DIR" python "$WORKSPACE_DIR/scripts/download_required_checkpoints.py" "$@"
  else
    "${WORKSPACE_DIR}/.venv/bin/python" "$WORKSPACE_DIR/scripts/download_required_checkpoints.py" "$@"
  fi
else
  "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    python "$WORKSPACE_DIR/scripts/download_required_checkpoints.py" "$@"
fi
