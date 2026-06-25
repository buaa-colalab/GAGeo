#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Installing GAGeo dependencies ==="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "=== Verifying installation ==="
python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
PY

echo "Installation completed."
