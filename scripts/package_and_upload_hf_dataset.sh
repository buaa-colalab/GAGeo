#!/bin/bash
set -euo pipefail

# Workspace path config
ROOT_DIR="${ROOT_DIR:-/data/home/scxi704/run/xhj}"
WORKSPACE_NAME="${WORKSPACE_NAME:-location_all_components}"
WORKSPACE_DIR="${ROOT_DIR}/${WORKSPACE_NAME}"

# ========================================================
# Package large dataset into split tar parts, then upload
# to Hugging Face dataset repo.
#
# Default target:
#   /data/home/scxi704/run/xhj/data  ->  cipual/Urban-CVOGL
# ========================================================

SRC_DIR="${SRC_DIR:-/data/home/scxi704/run/xhj/data}"
REPO_ID="${REPO_ID:-cipual/Urban-CVOGL}"
REPO_TYPE="${REPO_TYPE:-dataset}"
WORK_DIR="${WORK_DIR:-/data/home/scxi704/run/xhj/data_hf_pack}"
PACK_NAME="${PACK_NAME:-Urban-CVOGL}"
PART_SIZE="${PART_SIZE:-50G}"   # e.g. 10G / 20G / 50G
KEEP_PACK="${KEEP_PACK:-1}"     # 1 keep tar parts, 0 remove after upload

# conda env requested by user
source /data/home/scxi704/run/miniconda3/bin/activate
conda activate filtre

mkdir -p "$WORK_DIR"
PACK_DIR="$WORK_DIR/pack_${PACK_NAME}"
mkdir -p "$PACK_DIR"

# echo "[1/4] Check tools"
# command -v tar >/dev/null || { echo "tar not found"; exit 1; }
# command -v split >/dev/null || { echo "split not found"; exit 1; }
# command -v sha256sum >/dev/null || { echo "sha256sum not found"; exit 1; }

# python - <<'PY'
# import importlib.util, sys
# mods = ["huggingface_hub"]
# missing = [m for m in mods if importlib.util.find_spec(m) is None]
# if missing:
#     print("Missing python packages:", ", ".join(missing))
#     sys.exit(1)
# print("Python deps OK")
# PY

# echo "[2/4] Packaging source dir into split tar parts"
# if [[ ! -d "$SRC_DIR" ]]; then
#   echo "Source dir does not exist: $SRC_DIR"
#   exit 1
# fi

# # Clean old pack with same prefix (optional)
# rm -f "$PACK_DIR/${PACK_NAME}.tar.part-"* || true

# SRC_PARENT="$(dirname "$SRC_DIR")"
# SRC_BASE="$(basename "$SRC_DIR")"

# # Stream tar -> split (avoid temporary giant tar file)
# tar -C "$SRC_PARENT" -cf - "$SRC_BASE" | split -d -a 5 -b "$PART_SIZE" - "$PACK_DIR/${PACK_NAME}.tar.part-"

# echo "[3/4] Writing manifests"
# (
#   cd "$PACK_DIR"
#   ls -lh > FILELIST.txt
#   sha256sum ${PACK_NAME}.tar.part-* > SHA256SUMS.txt
# )

# cat > "$PACK_DIR/README_UPLOAD.txt" <<EOF
# Packaged from: $SRC_DIR
# Repo target: $REPO_ID ($REPO_TYPE)
# Part size: $PART_SIZE
# Created at: $(date -Iseconds)

# Reconstruct command:
#   cat ${PACK_NAME}.tar.part-* > ${PACK_NAME}.tar
#   tar -xf ${PACK_NAME}.tar
# EOF

echo "[4/4] Upload to Hugging Face"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "Please export HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) before upload."
  echo "Example: export HF_TOKEN=hf_xxx"
  exit 1
fi

python "${WORKSPACE_DIR}/scripts/upload_hf_dataset.py" \
  --repo-id "$REPO_ID" \
  --repo-type "$REPO_TYPE" \
  --pack-dir "$PACK_DIR"

if [[ "$KEEP_PACK" == "0" ]]; then
  echo "Removing packaged parts: $PACK_DIR"
  rm -rf "$PACK_DIR"
fi

echo "Done."
