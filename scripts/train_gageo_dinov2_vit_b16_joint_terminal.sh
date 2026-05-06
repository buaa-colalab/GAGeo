#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/EVA/yangheqing/workspace/location_v4"
WORKSPACE_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/EVA/yangheqing/workspace"
VENV_DIR="${WORKSPACE_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

EXPERIMENT_NAME="gageo_dinov2_vit_b16_joint"
CONFIG_PATH="${WORKSPACE_DIR}/configs/gageo_dinov2_vit_b16_joint.yaml"
OUTPUT_DIR="${WORKSPACE_DIR}/outputs/${EXPERIMENT_NAME}"
LOG_FILE="${WORKSPACE_DIR}/logs/${EXPERIMENT_NAME}.log"

export ROOT_DIR="${WORKSPACE_ROOT}"
export WORKSPACE_NAME="location_v4"
export WORKSPACE_DIR="${WORKSPACE_DIR}"
export CHECKPOINT_DIR="${WORKSPACE_DIR}/checkpoints_offline"
export JSON_ROOT="${WORKSPACE_ROOT}/CMA-Loc"
export DATA_ROOT="${WORKSPACE_ROOT}/CMA-Loc/data"
export OUTPUT_ROOT="${WORKSPACE_DIR}/outputs"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MASTER_PORT="29610"
export WANDB_PROJECT="location_v4"
export WANDB_NAME="${EXPERIMENT_NAME}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_API_KEY="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"
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export TORCH_HOME="${WORKSPACE_DIR}/.cache/torch"
export TMPDIR="${WORKSPACE_DIR}/.cache/tmp"
export TRITON_CACHE_DIR="${WORKSPACE_DIR}/.cache/triton"
export MPLCONFIGDIR="${WORKSPACE_DIR}/.cache/matplotlib"
export PYTHONPATH="${WORKSPACE_DIR}"
export http_proxy="http://10.217.142.137:8080"
export https_proxy="http://10.217.142.137:8080"
export HTTP_PROXY="http://10.217.142.137:8080"
export HTTPS_PROXY="http://10.217.142.137:8080"

mkdir -p "${WORKSPACE_DIR}/logs" "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
cd "${WORKSPACE_DIR}"
source "${VENV_DIR}/bin/activate"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Config    : ${CONFIG_PATH}"
echo "Output    : ${OUTPUT_DIR}"
echo "Python    : ${PYTHON_BIN}"
echo "Launcher  : ${PYTHON_BIN} -m torch.distributed.run"

"${PYTHON_BIN}" "${WORKSPACE_DIR}/scripts/preflight_gageo_config.py" "${CONFIG_PATH}"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node 8 --master_port "${MASTER_PORT}" \
  "${WORKSPACE_DIR}/train_detr_v2_ddp.py" \
  --config "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}"
