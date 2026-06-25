#!/usr/bin/env bash
# Evaluate all rebuttal checkpoints once they are copied back.
#
# Override paths as needed, for example:
#   export OUTPUT_ROOT=/path/to/location_outputs
#   export TROGEO_OUTPUT_ROOT=/path/to/trogeo_outputs/trogeo_pi3_eccv
#   export JSON_ROOT=/path/to/CMA-Loc
#   export DATA_ROOT=/path/to/CMA-Loc/data
#   export G2D_UNSEEN_JSON=/path/to/g2d_unseen_triplets.json
#   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   bash scripts/run_rebuttal_checkpoint_evals.sh

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/data/wrp/location_v4}"
CVOS_ROOT="${CVOS_ROOT:-/mnt/data/wrp/CVOS-Code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_DIR}/output_v3}"
TROGEO_OUTPUT_ROOT="${TROGEO_OUTPUT_ROOT:-${CVOS_ROOT}/outputs}"
JSON_ROOT="${JSON_ROOT:-/mnt/data/wrp/eccv_data/data/json}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/wrp/eccv_data/data/urban}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/data/wrp/checkpoints_offline}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/mnt/data/wrp/rebuttal_eval_outputs}"
G2D_ROOT_DIR="${G2D_ROOT_DIR:-/mnt/data/wrp/University-Release}"
G2D_UNSEEN_JSON="${G2D_UNSEEN_JSON:-${G2D_ROOT_DIR}/verified_triplets_sam2_masks.json}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TROGEO_PYTHON_BIN="${TROGEO_PYTHON_BIN:-${PYTHON_BIN}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# Evaluation is often launched as many independent single-GPU jobs.  Keeping
# workers at 0 avoids Python multiprocessing socket/ulimit failures in locked
# down containers; override NUM_WORKERS manually on machines known to support it.
NUM_WORKERS="${NUM_WORKERS:-0}"
GPUS="${CUDA_VISIBLE_DEVICES:-auto}"

cd "${WORKSPACE_DIR}"

ARGS=(
  --location-root "${WORKSPACE_DIR}"
  --cvos-root "${CVOS_ROOT}"
  --gageo-output-root "${OUTPUT_ROOT}"
  --trogeo-output-root "${TROGEO_OUTPUT_ROOT}"
  --eval-root "${EVAL_OUTPUT_ROOT}"
  --json-root "${JSON_ROOT}"
  --data-root "${DATA_ROOT}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --g2d-root-dir "${G2D_ROOT_DIR}"
  --gpus "${GPUS}"
  --python "${PYTHON_BIN}"
  --trogeo-python "${TROGEO_PYTHON_BIN}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)

if [[ -f "${G2D_UNSEEN_JSON}" ]]; then
  ARGS+=(--g2d-unseen-json "${G2D_UNSEEN_JSON}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

if [[ "${BEST_ONLY:-0}" == "1" ]]; then
  ARGS+=(--best-only)
fi

"${PYTHON_BIN}" scripts/run_rebuttal_checkpoint_evals.py "${ARGS[@]}"
