#!/bin/bash
#SBATCH --job-name=cvloc_v2_eval
#SBATCH --output=logs/slurm_v2_eval_%j.out
#SBATCH --error=logs/slurm_v2_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --partition=vip_gpu_5090_scxi704

# ============================================
# SLURM Evaluate Script for Cross-View V2
# 评估 test + unseen_test，按 task/size/shape 分组
# 并按 prompt 类型(point/bbox/mask)分别评估
# ============================================
# Usage:
#   # 单卡（串行）
#   sbatch scripts/slurm_eval_v2_grouped.sh <sam_checkpoint> [gpu_ids]
#
#   # 多卡（并行，建议按 prompt 次数申请）
#   sbatch --gres=gpu:3 scripts/slurm_eval_v2_grouped.sh <sam_checkpoint> 0,1,2
# Example:
#   sbatch --gres=gpu:3 scripts/slurm_eval_v2_grouped.sh /path/to/sam_vit_h_4b8939.pth 0,1,2
# ============================================

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"/data/home/scxi704/run/xhj"}
WORKSPACE_NAME=${WORKSPACE_NAME:-"location_v3"}
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"
RUN_ROOT="$(dirname "$ROOT_DIR")"
CACHE_ROOT=${CACHE_ROOT:-"${ROOT_DIR}/.cache"}

EXPRIMENT_NAME="ablation_4_all_on"

SAM_CKPT="${1:-}"
GPU_IDS="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
PROMPT_TYPES=(point bbox mask)

if [[ -z "$SAM_CKPT" ]]; then
  echo "[ERROR] Please provide SAM checkpoint path as first argument."
  echo "Usage: sbatch scripts/slurm_eval_v2_grouped.sh <sam_checkpoint> [gpu_id]"
  exit 1
fi

# conda env
source "${RUN_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate filtre

module load cuda

# Cache dirs
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

cd "$SLURM_SUBMIT_DIR"
if [[ -d "$WORKSPACE_DIR" ]]; then
  cd "$WORKSPACE_DIR"
fi
mkdir -p logs



OUT_JSON="output_v3/${EXPRIMENT_NAME}/eval_grouped_$(date +%Y%m%d_%H%M%S).json"

echo "=========================================="
echo "Cross-View V2 Grouped Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Requested GPU ids: $GPU_IDS"
echo "SAM checkpoint: $SAM_CKPT"
echo "Prompt types: point bbox mask"
echo "Output JSON: $OUT_JSON"
echo "=========================================="

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPUS=${#GPU_LIST[@]}
if [[ $NUM_GPUS -lt 1 ]]; then
  echo "[ERROR] No valid GPU ids parsed from: $GPU_IDS"
  exit 1
fi

TMP_JSON_DIR="output_v3/${EXPRIMENT_NAME}/eval_grouped_parts_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TMP_JSON_DIR"

echo "Run mode: ${#PROMPT_TYPES[@]} prompt(s) over $NUM_GPUS GPU(s)"

pids=()

launch_eval() {
  local prompt_type="$1"
  local gpu_id="$2"
  local out_json="$3"

  echo "[Launch] prompt=${prompt_type} on GPU ${gpu_id} -> ${out_json}"

  CUDA_VISIBLE_DEVICES="$gpu_id" \
  "${RUN_ROOT}/miniconda3/bin/conda" run -n filtre --no-capture-output \
    python "${WORKSPACE_DIR}/evaluate_custom_v2.py" \
      --config "${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/config.yaml" \
      --checkpoint "${WORKSPACE_DIR}/output_v3/${EXPRIMENT_NAME}/best" \
      --splits test unseen_test \
      --prompt_types "$prompt_type" \
      --batch_size 8 \
      --num_workers 8 \
      --gpu 0 \
      --sam_checkpoint "$SAM_CKPT" \
      --sam_model_type vit_h \
      --save_json "$out_json" &

  pids+=("$!")
}

for i in "${!PROMPT_TYPES[@]}"; do
  prompt="${PROMPT_TYPES[$i]}"
  gpu="${GPU_LIST[$((i % NUM_GPUS))]}"
  prompt_json="$TMP_JSON_DIR/${prompt}.json"

  # 若 prompt 数超过 GPU 数，限流到 NUM_GPUS 并发
  while [[ $(jobs -rp | wc -l) -ge $NUM_GPUS ]]; do
    wait -n
  done

  launch_eval "$prompt" "$gpu" "$prompt_json"
done

echo "Waiting all prompt evaluations to finish ..."
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Merging prompt JSONs -> $OUT_JSON"
python - <<PY
import json
from pathlib import Path

tmp_dir = Path("${TMP_JSON_DIR}")
out_path = Path("${OUT_JSON}")

merged = {}
for jf in sorted(tmp_dir.glob("*.json")):
    with jf.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for split, prompt_dict in data.items():
        merged.setdefault(split, {})
        for prompt_type, groups in prompt_dict.items():
            merged[split][prompt_type] = groups

with out_path.open("w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Merged result saved to: {out_path}")
PY

echo "Evaluation completed!"