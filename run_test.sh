#!/bin/bash
# Test script with thread limits for GPUs 4-7

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Activate conda environment
source ~/.bashrc
conda activate filtre

# Set GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "=========================================="
echo "Testing on GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Run quick forward test first
echo ""
echo "Running quick forward test..."
python test_model_forward.py

# If successful, run full test
if [ $? -eq 0 ]; then
    echo ""
    echo "Running full model test..."
    python test_model.py
else
    echo "Quick test failed, skipping full test"
    exit 1
fi
