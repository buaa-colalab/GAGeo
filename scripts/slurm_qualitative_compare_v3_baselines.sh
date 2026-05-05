#!/bin/bash

set -euo pipefail

# Usage:
#   bash scripts/slurm_qualitative_compare_v3_baselines.sh \
#     [v3_ckpt] [cvos_ckpt] [detgeo_ckpt] [sam_ckpt] [top_k] [vis_k] [gap_mode] [gpu_id] \
#     [gt_mask_color] [gt_mask_alpha] [pred_mask_color] [pred_mask_alpha] \
#     [pose_point_color_pred] [pose_arrow_color_pred] [pose_point_color_gt] [pose_arrow_color_gt] \
#     [v3_mask_color] [v3_mask_alpha] [cvos_mask_color] [cvos_mask_alpha] [detgeo_mask_color] [detgeo_mask_alpha] \
#     [dataset]
#
# Example:
#   bash scripts/slurm_qualitative_compare_v3_baselines.sh \
#     /mnt/data/wrp/location_v4/output_v3/ablation_4_all_on/best \
#     /mnt/data/wrp/CVOS-Code/saved_models/customdata_model_best.pth.tar \
#     /mnt/data/wrp/DetGeo/saved_models/customdata_model_best.pth.tar \
#     /mnt/data/wrp/CVOS-Code/segment_anything/weights/sam_vit_h_4b8939.pth \
#     200 20 abs 0 \
#     255,255,255 1.0 255,255,255 1.0 255,255,255 255,255,255 255,255,255 255,255,255 \
#     241,216,167 1.0 218,134,129 1.0 230,200,200 1.0 \
#     unseen

RUN_ROOT="/mnt/data/wrp"
WORKSPACE_DIR="${RUN_ROOT}/location_v4"
PY="${RUN_ROOT}/miniconda3/envs/gageo/bin/python"

V3_CONFIG="${WORKSPACE_DIR}/output_v3/ablation_4_all_on/config.yaml"
V3_CKPT="${1:-${WORKSPACE_DIR}/output_v3/ablation_4_all_on/best}"
CVOS_CKPT="${2:-${RUN_ROOT}/CVOS-Code/saved_models/customdata_model_best.pth.tar}"
DETGEO_CKPT="${3:-${RUN_ROOT}/DetGeo/saved_models/customdata_model_best.pth.tar}"
SAM_CKPT="${4:-${RUN_ROOT}/CVOS-Code/segment_anything/weights/sam_vit_h_4b8939.pth}"
TOP_K="${5:-200}"
VIS_K="${6:-20}"
GAP_MODE="${7:-abs}"
GPU_ID="${8:-0}"
GT_MASK_COLOR="${9:-248,203,173}"
GT_MASK_ALPHA="${10:-1.0}"
PRED_MASK_COLOR="${11:-255,255,255}"
PRED_MASK_ALPHA="${12:-1.0}"
POSE_POINT_COLOR_PRED="${13:-255,255,255}"
POSE_ARROW_COLOR_PRED="${14:-255,255,255}"
POSE_POINT_COLOR_GT="${15:-255,255,255}"
POSE_ARROW_COLOR_GT="${16:-255,255,255}"
V3_MASK_COLOR="${17:-241,216,167}"
V3_MASK_ALPHA="${18:-1.0}"
CVOS_MASK_COLOR="${19:-218,134,129}"
CVOS_MASK_ALPHA="${20:-1.0}"
DETGEO_MASK_COLOR="${21:-230,200,200}"
DETGEO_MASK_ALPHA="${22:-1.0}"
DATASET="${23:-unseen}"   # unseen | university | cvoglseg

JSON_PATH="${RUN_ROOT}/eccv_data/data/json/unseen_test.json"
IMAGE_ROOT="${RUN_ROOT}/eccv_data/data/urban"
UNIV_JSON="${RUN_ROOT}/University-Release/verified_triplets_sam2_masks.json"
UNIV_ROOT="${RUN_ROOT}/University-Release"
CVOGL_ROOT="${RUN_ROOT}/CVOS-Code/dataset/CVOGL"
CVOGLSEG_ROOT="${RUN_ROOT}/CVOS-Code/dataset/CVOGL-Seg"

cd "${WORKSPACE_DIR}"

echo "=========================================="
echo "V3 vs Baseline Qualitative Compare"
echo "=========================================="
echo "GPU ID: ${GPU_ID}"
echo "Dataset: ${DATASET}"
echo "V3 config: ${V3_CONFIG}"
echo "V3 ckpt: ${V3_CKPT}"
echo "CVOS ckpt: ${CVOS_CKPT}"
echo "DetGeo ckpt: ${DETGEO_CKPT}"
echo "SAM ckpt: ${SAM_CKPT}"
echo "top_k_candidates: ${TOP_K}"
echo "vis_k: ${VIS_K}"
echo "gap_mode: ${GAP_MODE}"
echo "gt_mask_color: ${GT_MASK_COLOR} | gt_mask_alpha: ${GT_MASK_ALPHA}"
echo "pred_mask_color: ${PRED_MASK_COLOR} | pred_mask_alpha: ${PRED_MASK_ALPHA}"
echo "v3_mask_color: ${V3_MASK_COLOR} | v3_mask_alpha: ${V3_MASK_ALPHA}"
echo "cvos_mask_color: ${CVOS_MASK_COLOR} | cvos_mask_alpha: ${CVOS_MASK_ALPHA}"
echo "detgeo_mask_color: ${DETGEO_MASK_COLOR} | detgeo_mask_alpha: ${DETGEO_MASK_ALPHA}"
echo "pose colors (pred point/arrow): ${POSE_POINT_COLOR_PRED} / ${POSE_ARROW_COLOR_PRED}"
echo "pose colors (gt point/arrow): ${POSE_POINT_COLOR_GT} / ${POSE_ARROW_COLOR_GT}"
echo "=========================================="

case "${DATASET}" in
  unseen)
    OUTPUT_ROOT="${WORKSPACE_DIR}/vis"
    SUMMARY_JSON="${OUTPUT_ROOT}/selection_summary_$(date +%Y%m%d_%H%M%S).json"
    echo "JSON: ${JSON_PATH}"
    echo "Image root: ${IMAGE_ROOT}"
    echo "Output root: ${OUTPUT_ROOT}"
    echo "Summary json: ${SUMMARY_JSON}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PY}" "${WORKSPACE_DIR}/qualitative_compare_v3_baselines.py" \
      --json_path "${JSON_PATH}" \
      --image_root "${IMAGE_ROOT}" \
      --v3_root "${WORKSPACE_DIR}" \
      --v3_config "${V3_CONFIG}" \
      --v3_checkpoint "${V3_CKPT}" \
      --cvos_root "${RUN_ROOT}/CVOS-Code" \
      --cvos_checkpoint "${CVOS_CKPT}" \
      --detgeo_root "${RUN_ROOT}/DetGeo" \
      --detgeo_checkpoint "${DETGEO_CKPT}" \
      --sam_checkpoint "${SAM_CKPT}" \
      --sam_model_type vit_h \
      --top_k_candidates "${TOP_K}" \
      --vis_k "${VIS_K}" \
      --gap_mode "${GAP_MODE}" \
      --output_root "${OUTPUT_ROOT}" \
      --gt_mask_color "${GT_MASK_COLOR}" \
      --gt_mask_alpha "${GT_MASK_ALPHA}" \
      --pred_mask_color "${PRED_MASK_COLOR}" \
      --pred_mask_alpha "${PRED_MASK_ALPHA}" \
      --v3_mask_color "${V3_MASK_COLOR}" \
      --v3_mask_alpha "${V3_MASK_ALPHA}" \
      --cvos_mask_color "${CVOS_MASK_COLOR}" \
      --cvos_mask_alpha "${CVOS_MASK_ALPHA}" \
      --detgeo_mask_color "${DETGEO_MASK_COLOR}" \
      --detgeo_mask_alpha "${DETGEO_MASK_ALPHA}" \
      --pose_point_color_pred "${POSE_POINT_COLOR_PRED}" \
      --pose_arrow_color_pred "${POSE_ARROW_COLOR_PRED}" \
      --pose_point_color_gt "${POSE_POINT_COLOR_GT}" \
      --pose_arrow_color_gt "${POSE_ARROW_COLOR_GT}" \
      --save_summary_json "${SUMMARY_JSON}" \
      --gpu 0
    ;;
  university)
    OUTPUT_ROOT="${WORKSPACE_DIR}/vis_university"
    SUMMARY_JSON="${OUTPUT_ROOT}/selection_summary_$(date +%Y%m%d_%H%M%S).json"
    echo "Triplet json: ${UNIV_JSON}"
    echo "Root dir: ${UNIV_ROOT}"
    echo "Output root: ${OUTPUT_ROOT}"
    echo "Summary json: ${SUMMARY_JSON}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PY}" "${WORKSPACE_DIR}/qualitative_compare_university.py" \
      --triplet_json "${UNIV_JSON}" \
      --root_dir "${UNIV_ROOT}" \
      --v3_root "${WORKSPACE_DIR}" \
      --v3_config "${V3_CONFIG}" \
      --v3_checkpoint "${V3_CKPT}" \
      --cvos_root "${RUN_ROOT}/CVOS-Code" \
      --cvos_checkpoint "${CVOS_CKPT}" \
      --detgeo_root "${RUN_ROOT}/DetGeo" \
      --detgeo_checkpoint "${DETGEO_CKPT}" \
      --sam_checkpoint "${SAM_CKPT}" \
      --sam_model_type vit_h \
      --top_k_candidates "${TOP_K}" \
      --vis_k "${VIS_K}" \
      --gap_mode "${GAP_MODE}" \
      --output_root "${OUTPUT_ROOT}" \
      --gt_mask_color "${GT_MASK_COLOR}" \
      --gt_mask_alpha "${GT_MASK_ALPHA}" \
      --pred_mask_color "${PRED_MASK_COLOR}" \
      --pred_mask_alpha "${PRED_MASK_ALPHA}" \
      --v3_mask_color "${V3_MASK_COLOR}" \
      --v3_mask_alpha "${V3_MASK_ALPHA}" \
      --cvos_mask_color "${CVOS_MASK_COLOR}" \
      --cvos_mask_alpha "${CVOS_MASK_ALPHA}" \
      --detgeo_mask_color "${DETGEO_MASK_COLOR}" \
      --detgeo_mask_alpha "${DETGEO_MASK_ALPHA}" \
      --save_summary_json "${SUMMARY_JSON}" \
      --gpu 0
    ;;
  cvoglseg)
    OUTPUT_ROOT="${WORKSPACE_DIR}/vis_cvoglseg"
    SUMMARY_JSON="${OUTPUT_ROOT}/selection_summary_$(date +%Y%m%d_%H%M%S).json"
    echo "CVOGL root: ${CVOGL_ROOT}"
    echo "CVOGL-Seg root: ${CVOGLSEG_ROOT}"
    echo "Output root: ${OUTPUT_ROOT}"
    echo "Summary json: ${SUMMARY_JSON}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PY}" "${WORKSPACE_DIR}/qualitative_compare_cvoglseg.py" \
      --cvogl_root "${CVOGL_ROOT}" \
      --cvoglseg_root "${CVOGLSEG_ROOT}" \
      --split test \
      --subsets CVOGL_SVI CVOGL_DroneAerial \
      --v3_root "${WORKSPACE_DIR}" \
      --v3_config "${V3_CONFIG}" \
      --v3_checkpoint "${V3_CKPT}" \
      --cvos_root "${RUN_ROOT}/CVOS-Code" \
      --cvos_checkpoint "${CVOS_CKPT}" \
      --detgeo_root "${RUN_ROOT}/DetGeo" \
      --detgeo_checkpoint "${DETGEO_CKPT}" \
      --sam_checkpoint "${SAM_CKPT}" \
      --sam_model_type vit_h \
      --top_k_candidates "${TOP_K}" \
      --vis_k "${VIS_K}" \
      --gap_mode "${GAP_MODE}" \
      --output_root "${OUTPUT_ROOT}" \
      --gt_mask_color "${GT_MASK_COLOR}" \
      --gt_mask_alpha "${GT_MASK_ALPHA}" \
      --pred_mask_color "${PRED_MASK_COLOR}" \
      --pred_mask_alpha "${PRED_MASK_ALPHA}" \
      --v3_mask_color "${V3_MASK_COLOR}" \
      --v3_mask_alpha "${V3_MASK_ALPHA}" \
      --cvos_mask_color "${CVOS_MASK_COLOR}" \
      --cvos_mask_alpha "${CVOS_MASK_ALPHA}" \
      --detgeo_mask_color "${DETGEO_MASK_COLOR}" \
      --detgeo_mask_alpha "${DETGEO_MASK_ALPHA}" \
      --save_summary_json "${SUMMARY_JSON}" \
      --gpu 0
    ;;
  *)
    echo "[error] Unknown dataset='${DATASET}', expected one of: unseen | university | cvoglseg"
    exit 2
    ;;
esac

echo "Qualitative comparison completed."

