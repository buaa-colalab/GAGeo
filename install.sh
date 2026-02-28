#!/bin/bash

echo "=== Installing Cross-View Localization Dependencies ==="

ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_v4}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

# 检测 CUDA 版本
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

echo "Detected CUDA version: $CUDA_VERSION"

# 根据 CUDA 版本选择 PyTorch 索引
case $CUDA_VERSION in
    "11.8.89")
        INDEX_URL="https://download.pytorch.org/whl/cu118"
        ;;
    "12.1")
        INDEX_URL="https://download.pytorch.org/whl/cu121"
        ;;
    "12.4")
        INDEX_URL="https://download.pytorch.org/whl/cu124"
        ;;
    *)
        echo "Unsupported CUDA version: $CUDA_VERSION"
        exit 1
        ;;
esac

echo "Using PyTorch index: $INDEX_URL"

# 升级 pip
echo "=== Upgrading pip ==="
pip install --upgrade pip

# 安装 Core Deep Learning 依赖
echo "=== Installing Core Deep Learning packages ==="
pip install torch torchvision torchaudio --index-url $INDEX_URL
pip install xformers --index-url $INDEX_URL

# 安装 Distributed Training & Optimization 依赖
echo "=== Installing Distributed Training & Optimization packages ==="
pip install accelerate transformers deepspeed

# 安装 Data Processing 依赖
echo "=== Installing Data Processing packages ==="
pip install numpy scipy Pillow opencv-python pycocotools

# 安装 Utilities 依赖
echo "=== Installing Utility packages ==="
pip install pyyaml tqdm matplotlib

# 安装 Logging 依赖
echo "=== Installing Logging packages ==="
pip install tensorboard

# 编译 curope (RoPE2D CUDA extension)
echo "=== Compiling curope (RoPE2D CUDA extension) ==="
if [ -d "${WORKSPACE_DIR}/curope" ]; then
    cd "${WORKSPACE_DIR}/curope"
    python setup.py build_ext --inplace
    cd "$WORKSPACE_DIR"
    echo "curope compilation completed!"
else
    echo "Warning: ${WORKSPACE_DIR}/curope directory not found. Skipping curope compilation."
    echo "Please ensure the curope source code is available and compile it manually."
fi

# 验证安装
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
"

python -c "
import xformers
print(f'xformers: {xformers.__version__}')
"

python -c "
from models.layers.pos_embed import RoPE2D
print(f'RoPE2D: {RoPE2D}')
"

echo "=== Installation completed successfully! ==="
echo "You can now run training with: python train_ddp.py"