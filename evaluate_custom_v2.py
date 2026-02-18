#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-View Localizer V2 独立评测脚本（参考 baseline/evaluate_custom.py）

评估指标：
  检测: mean IoU · ACC@25 · ACC@50
  分割(模型 mask): mIoU · mDice · AAE · ME
  分割(SAM mask, bbox+point->SAM): mIoU · mDice · AAE · ME

支持：
    - split: test / unseen_test（默认都评估）
    - 分组: task_type / size_category / shape_category（并包含 task×size / task×shape）
    - prompt 可选: point / bbox / mask（可一次评估多个）
    - 默认加载 output_v2/best
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from models import build_cross_view_localizer_v2


def get_workspace_dir() -> Path:
    root_dir = os.environ.get("ROOT_DIR", "")
    workspace_name = os.environ.get("WORKSPACE_NAME", "")
    if root_dir and workspace_name:
        return Path(root_dir) / workspace_name
    return Path(__file__).resolve().parent


def load_cfg_with_env(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f) if config_path.endswith(".json") else __import__("yaml").safe_load(f)

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    return _expand(cfg)


# =========================
# 通用工具
# =========================

def decode_segmentation(segmentation, h: int, w: int) -> np.ndarray:
    """解码 COCO segmentation -> 二值 mask (H,W), uint8。"""
    if segmentation is None:
        return np.zeros((h, w), dtype=np.uint8)

    if isinstance(segmentation, list):
        # polygon
        if len(segmentation) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in segmentation:
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        return mask

    if isinstance(segmentation, dict) and "counts" in segmentation:
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, h, w)
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(np.uint8)

    return np.zeros((h, w), dtype=np.uint8)


def bbox_xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b.astype(np.float32)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def bbox_cxcywh_norm_to_xyxy_abs(b: torch.Tensor, img_size: int) -> np.ndarray:
    # b: [4], normalized cxcywh
    cx, cy, w, h = b.detach().cpu().numpy().astype(np.float32)
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def clip_bbox_xyxy(b: np.ndarray, size: int) -> np.ndarray:
    b = b.copy()
    b[0::2] = np.clip(b[0::2], 0, size - 1)
    b[1::2] = np.clip(b[1::2], 0, size - 1)
    return b


def bbox_iou_np(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, float(b1[2] - b1[0])) * max(0.0, float(b1[3] - b1[1]))
    a2 = max(0.0, float(b2[2] - b2[0])) * max(0.0, float(b2[3] - b2[1]))
    return inter / (a1 + a2 - inter + 1e-16)


def rotation_matrix_to_yaw_np(R: np.ndarray) -> np.ndarray:
    """Extract yaw (radians) from rotation matrix using ZYX convention.

    Args:
        R: [..., 3, 3] or [..., 4, 4]
    Returns:
        yaw: [...] in radians
    """
    if R.shape[-2:] == (4, 4):
        R = R[..., :3, :3]
    return np.arctan2(R[..., 1, 0], R[..., 0, 0]).astype(np.float32)


def maybe_deg_to_rad(x: float) -> np.float32:
    """Convert degree input to radians when value range indicates degrees."""
    x = float(x)
    x = np.deg2rad(x)
    return np.float32(x)


def euler_to_rotation_matrix_np(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def geodesic_rotation_error_deg_np(pred_R: np.ndarray, gt_R: np.ndarray) -> float:
    """SO(3) geodesic rotation error in degrees (same definition as training/val)."""
    if pred_R.shape == (4, 4):
        pred_R = pred_R[:3, :3]
    if gt_R.shape == (4, 4):
        gt_R = gt_R[:3, :3]

    R_diff = pred_R.T @ gt_R
    trace = float(R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2])
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7))
    return float(np.degrees(np.arccos(cos_angle)))


def build_gt_rotation_matrix(item: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    """Build GT relative rotation matrix with training-consistent fields/defaults."""
    if "relative_yaw" in item:
        yaw = np.deg2rad(float(item.get("relative_yaw", 0.0)))
        default_pitch = 45.0 if "drone" in str(item.get("mono_filename", "")).lower() else 90.0
        pitch = np.deg2rad(float(item.get("relative_pitch", default_pitch)))
        roll = np.deg2rad(float(item.get("relative_roll", 0.0)))
        return euler_to_rotation_matrix_np(yaw, pitch, roll), True

    # Fallback: only absolute/legacy yaw is available
    if "rotation" in item and item.get("rotation") is not None:
        yaw = float(maybe_deg_to_rad(item["rotation"]))
        return euler_to_rotation_matrix_np(yaw, 0.0, 0.0), True

    return np.eye(3, dtype=np.float32), False


class SegMetrics:
    @staticmethod
    def iou(p: np.ndarray, g: np.ndarray) -> float:
        p, g = p.astype(bool), g.astype(bool)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        return 1.0 if union == 0 else float(inter) / float(union)

    @staticmethod
    def dice(p: np.ndarray, g: np.ndarray) -> float:
        p, g = p.astype(bool), g.astype(bool)
        inter = np.logical_and(p, g).sum()
        total = p.sum() + g.sum()
        return 1.0 if total == 0 else 2.0 * float(inter) / float(total)

    @staticmethod
    def aae(p: np.ndarray, g: np.ndarray) -> float:
        return float(abs(int(p.astype(bool).sum()) - int(g.astype(bool).sum())))

    @staticmethod
    def me(p: np.ndarray, g: np.ndarray) -> float:
        pb, gb = p.astype(bool), g.astype(bool)
        if pb.sum() == 0 or gb.sum() == 0:
            return float(np.sqrt(p.shape[0] ** 2 + p.shape[1] ** 2))
        py, px = np.where(pb)
        gy, gx = np.where(gb)
        return float(np.sqrt((px.mean() - gx.mean()) ** 2 + (py.mean() - gy.mean()) ** 2))

    @classmethod
    def all(cls, p: np.ndarray, g: np.ndarray) -> Tuple[float, float, float, float]:
        return cls.iou(p, g), cls.dice(p, g), cls.aae(p, g), cls.me(p, g)


# =========================
# 分组评估器
# =========================

class GroupedEvaluator:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add(
        self,
        *,
        det_iou: float,
        model_seg: Tuple[float, float, float, float] | None,
        sam_seg: Tuple[float, float, float, float] | None,
        rotation_error_deg: float | None,
        pos_err_px: float | None,
        task_type: str,
        size_category: str,
        shape_category: str,
    ):
        self.records.append(
            {
                "det_iou": det_iou,
                "model_seg": model_seg,
                "sam_seg": sam_seg,
                "rotation_error_deg": rotation_error_deg,
                "pos_err_px": pos_err_px,
                "task_type": task_type,
                "size_category": size_category,
                "shape_category": shape_category,
            }
        )

    @staticmethod
    def _agg(records: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not records:
            return None

        det = [r["det_iou"] for r in records]
        out: Dict[str, Any] = {
            "count": len(records),
            "det_iou": float(np.mean(det)),
            "det_acc25": float(np.mean([1.0 if x > 0.25 else 0.0 for x in det])),
            "det_acc50": float(np.mean([1.0 if x > 0.50 else 0.0 for x in det])),
        }

        rot_vals = [r["rotation_error_deg"] for r in records if r.get("rotation_error_deg") is not None]
        pos_vals = [r["pos_err_px"] for r in records if r.get("pos_err_px") is not None]
        if rot_vals:
            out["rotation_error_deg"] = float(np.mean(rot_vals))
        if pos_vals:
            out["pos_err_px"] = float(np.mean(pos_vals))

        for key, prefix in (("model_seg", "model"), ("sam_seg", "sam")):
            vals = [r[key] for r in records if r[key] is not None]
            if vals:
                out[f"{prefix}_miou"] = float(np.mean([v[0] for v in vals]))
                out[f"{prefix}_mdice"] = float(np.mean([v[1] for v in vals]))
                out[f"{prefix}_aae"] = float(np.mean([v[2] for v in vals]))
                out[f"{prefix}_me"] = float(np.mean([v[3] for v in vals]))

        return out

    def summarize(self) -> OrderedDict:
        g = OrderedDict()
        g["overall"] = self._agg(self.records)

        for tt in ("drone", "ground"):
            s = [r for r in self.records if r["task_type"] == tt]
            if s:
                g[f"task/{tt}"] = self._agg(s)

        for sc in ("small", "medium", "large"):
            s = [r for r in self.records if r["size_category"] == sc]
            if s:
                g[f"size/{sc}"] = self._agg(s)

        for sh in ("regular", "irregular"):
            s = [r for r in self.records if r["shape_category"] == sh]
            if s:
                g[f"shape/{sh}"] = self._agg(s)

        for tt in ("drone", "ground"):
            for sc in ("small", "medium", "large"):
                s = [r for r in self.records if r["task_type"] == tt and r["size_category"] == sc]
                if s:
                    g[f"task×size/{tt}/{sc}"] = self._agg(s)
            for sh in ("regular", "irregular"):
                s = [r for r in self.records if r["task_type"] == tt and r["shape_category"] == sh]
                if s:
                    g[f"task×shape/{tt}/{sh}"] = self._agg(s)

        return g


# =========================
# 数据集
# =========================

class EvalDatasetV2(Dataset):
    """
    与训练解耦的评估数据集：
      - 只读取 test/unseen_test json
            - 支持 point / bbox / mask prompt
      - 返回分组字段 task_type/size_category/shape_category
    """

    def __init__(self, data_list: List[Dict[str, Any]], image_root: str, img_size: int = 518):
        self.data_list = data_list
        self.image_root = image_root
        self.img_size = img_size

    def __len__(self):
        return len(self.data_list)

    def _load_rgb(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        item = self.data_list[idx]
        city = item.get("city", "")

        mono_name = item["mono_filename"]
        sat_name = item.get("sat_filename") or item.get("sate_filename")
        if sat_name is None:
            raise KeyError("sample missing sat_filename/sate_filename")

        mono_path = os.path.join(self.image_root, city, "mono", mono_name)
        sat_path = os.path.join(self.image_root, city, "crop_sate", sat_name)

        mono = self._load_rgb(mono_path)
        sat = self._load_rgb(sat_path)

        # 原始尺寸
        h_m, w_m = mono.shape[:2]
        h_s, w_s = sat.shape[:2]

        # 点提示（mono坐标）
        mono_point = np.array(item["mono_point"][:2], dtype=np.float32)

        # bbox 提示（mono，通常为 cx,cy,w,h）
        mono_bbox = np.array(item.get("mono_bbox", [0, 0, 0, 0])[:4], dtype=np.float32)

        # mono mask 提示
        mono_mask = decode_segmentation(item.get("mono_segmentation"), h_m, w_m)

        # GT bbox: sate_bbox in xywh
        gt_bbox_xywh = np.array(item["sate_bbox"][:4], dtype=np.float32)

        # GT mask
        gt_mask = decode_segmentation(item.get("sate_segmentation"), h_s, w_s)

        # Pose GT (可选): 统一为旋转矩阵 + 位置
        gt_rotation_matrix, has_rotation = build_gt_rotation_matrix(item)
        gt_pos = item.get("camera_position", None)
        has_position = (gt_pos is not None) and (len(gt_pos) >= 2)
        if has_position:
            gt_pos = np.array(gt_pos[:2], dtype=np.float32)
            gt_pos_is_pixel = True
        else:
            gt_pos = np.zeros((2,), dtype=np.float32)
            gt_pos_is_pixel = False

        # resize 到 img_size 并同步坐标
        S = self.img_size

        if (h_m, w_m) != (S, S):
            sx_m, sy_m = S / w_m, S / h_m
            mono = cv2.resize(mono, (S, S), interpolation=cv2.INTER_LINEAR)
            mono_point = np.array([mono_point[0] * sx_m, mono_point[1] * sy_m], dtype=np.float32)
            mono_bbox = np.array(
                [
                    mono_bbox[0] * sx_m,
                    mono_bbox[1] * sy_m,
                    mono_bbox[2] * sx_m,
                    mono_bbox[3] * sy_m,
                ],
                dtype=np.float32,
            )
            mono_mask = cv2.resize(mono_mask.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)

        if (h_s, w_s) != (S, S):
            sx_s, sy_s = S / w_s, S / h_s
            sat = cv2.resize(sat, (S, S), interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
            gt_bbox_xywh = np.array(
                [
                    gt_bbox_xywh[0] * sx_s,
                    gt_bbox_xywh[1] * sy_s,
                    gt_bbox_xywh[2] * sx_s,
                    gt_bbox_xywh[3] * sy_s,
                ],
                dtype=np.float32,
            )
            if has_position and gt_pos_is_pixel:
                gt_pos = np.array([gt_pos[0] * sx_s, gt_pos[1] * sy_s], dtype=np.float32)

        gt_bbox_xyxy = bbox_xywh_to_xyxy(gt_bbox_xywh)

        # Normalize GT position to [0,1] to match model output `position`.
        if has_position:
            if gt_pos_is_pixel:
                gt_pos = np.array([gt_pos[0] / S, gt_pos[1] / S], dtype=np.float32)
            gt_pos = np.clip(gt_pos, 0.0, 1.0).astype(np.float32)

        mono_t = to_tensor(Image.fromarray(mono))  # [3,S,S], [0,1]
        sat_t = to_tensor(Image.fromarray(sat))

        return {
            "front_view": mono_t,
            "sat_view": sat_t,
            "mono_point": torch.from_numpy(mono_point),
            "mono_bbox": torch.from_numpy(mono_bbox),
            "mono_mask": torch.from_numpy((mono_mask > 0).astype(np.uint8)),
            "gt_bbox_xyxy": torch.from_numpy(gt_bbox_xyxy),
            "gt_mask": torch.from_numpy((gt_mask > 0).astype(np.uint8)),
            "gt_rotation_matrix": torch.from_numpy(gt_rotation_matrix),
            "gt_position": torch.from_numpy(gt_pos),
            "has_rotation": bool(has_rotation),
            "has_pose": bool(has_position),
            "sat_rgb": sat,
            "task_type": item.get("task_type", "unknown"),
            "size_category": item.get("size_category", "unknown"),
            "shape_category": item.get("shape_category", "unknown"),
            "index": idx,
        }


def collate_eval(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "front_view": torch.stack([x["front_view"] for x in batch], dim=0),
        "sat_view": torch.stack([x["sat_view"] for x in batch], dim=0),
        "mono_point": torch.stack([x["mono_point"] for x in batch], dim=0),
        "mono_bbox": torch.stack([x["mono_bbox"] for x in batch], dim=0),
        "mono_mask": torch.stack([x["mono_mask"] for x in batch], dim=0),
        "gt_bbox_xyxy": torch.stack([x["gt_bbox_xyxy"] for x in batch], dim=0),
        "gt_mask": torch.stack([x["gt_mask"] for x in batch], dim=0),
        "gt_rotation_matrix": torch.stack([x["gt_rotation_matrix"] for x in batch], dim=0),
        "gt_position": torch.stack([x["gt_position"] for x in batch], dim=0),
        "has_rotation": [x["has_rotation"] for x in batch],
        "has_pose": [x["has_pose"] for x in batch],
        "sat_rgb": [x["sat_rgb"] for x in batch],
        "task_type": [x["task_type"] for x in batch],
        "size_category": [x["size_category"] for x in batch],
        "shape_category": [x["shape_category"] for x in batch],
        "index": [x["index"] for x in batch],
    }


# =========================
# 模型加载
# =========================

def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = [
        path / "pytorch_model" / "mp_rank_00_model_states.pt",
        path / "mp_rank_00_model_states.pt",
        path / "pytorch_model.bin",
        path / "model.safetensors",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Cannot resolve checkpoint file from: {path}")


def extract_state_dict(obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for k in ["module", "model", "state_dict", "model_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                # strip possible 'module.' prefix
                if len(sd) > 0:
                    first_k = next(iter(sd.keys()))
                    if first_k.startswith("module."):
                        sd = {kk[len("module."):]: vv for kk, vv in sd.items()}
                return sd
        # root state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format")


def build_model_from_cfg(cfg: Dict[str, Any], device: torch.device):
    mc = cfg["model"]
    dc = cfg["data"]

    model = build_cross_view_localizer_v2(
        pretrained_pi3=None,  # 评测时由 checkpoint 覆盖
        freeze_backbone=False,
        freeze_prompt_encoder=False,
        load_camera_head_weights=False,
        sam_weights=None,
        img_size=dc.get("img_size", 518),
        decoder_size=mc.get("decoder_size", "large"),
        num_learnable_tokens=mc.get("num_learnable_tokens", 2),
        supervision_layers=mc.get("supervision_layers", [4, 11, 17]),
        supervision_weights=mc.get("supervision_weights", [0.1, 0.3, 0.6]),
        dropout=mc.get("dropout", 0.1),
        contrastive=mc.get("contrastive", True),
        contrastive_proj_dim=mc.get("contrastive_proj_dim", 256),
        contrastive_queue_size=mc.get("contrastive_queue_size", 16384),
        contrastive_momentum=mc.get("contrastive_momentum", 0.999),
        contrastive_temperature=mc.get("contrastive_temperature", 0.07),
        sam_embed_dim=mc.get("sam_embed_dim", 256),
        num_mask_tokens=mc.get("num_mask_tokens", 1),
    )

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# =========================
# 评测
# =========================

@torch.no_grad()
def evaluate_split(
    model,
    sam_predictor,
    loader: DataLoader,
    img_size: int,
    device: torch.device,
    use_sam: bool,
    prompt_type: str,
):
    evaluator = GroupedEvaluator()

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        front = batch["front_view"].to(device, non_blocking=True)
        sat = batch["sat_view"].to(device, non_blocking=True)
        mono_point = batch["mono_point"].to(device, non_blocking=True)  # [B,2]
        mono_bbox = batch["mono_bbox"].to(device, non_blocking=True)    # [B,4] cx,cy,w,h
        mono_mask = batch["mono_mask"].to(device, non_blocking=True).float().unsqueeze(1)  # [B,1,S,S]

        B = front.shape[0]
        points, boxes, masks = None, None, None
        if prompt_type == "point":
            point_coords = mono_point.unsqueeze(1)  # [B,1,2]
            point_labels = torch.ones(B, 1, device=device)
            points = (point_coords, point_labels)
        elif prompt_type == "bbox":
            # IMPORTANT:
            # V2 prompt encoder + prompt_coords path expects boxes in (x, y, w, h)
            # pixel space (top-left + size), not (x1, y1, x2, y2).
            # Wrong format may produce RoPE positions out-of-range and trigger CUDA assert.
            boxes = mono_bbox.clone()
            boxes[:, 0] = boxes[:, 0].clamp(0, img_size - 1)  # x
            boxes[:, 1] = boxes[:, 1].clamp(0, img_size - 1)  # y
            boxes[:, 2] = boxes[:, 2].clamp(min=1.0, max=img_size)  # w
            boxes[:, 3] = boxes[:, 3].clamp(min=1.0, max=img_size)  # h
            boxes = boxes.unsqueeze(1)  # [B,1,4] (x, y, w, h)
        elif prompt_type == "mask":
            masks = mono_mask
        else:
            raise ValueError(f"Unsupported prompt_type: {prompt_type}")

        outputs = model(
            front_view=front,
            satellite_view=sat,
            points=points,
            boxes=boxes,
            masks=masks,
            mono_mask=None,
            sat_mask=None,
        )

        pred_bbox_norm = outputs["pred_boxes"][:, 0]  # [B,4], cxcywh in [0,1]
        pred_mask = outputs["mask_pred"][:, 0].detach().cpu().numpy()  # [B,S,S]

        gt_bbox_xyxy = batch["gt_bbox_xyxy"].numpy()
        gt_mask = batch["gt_mask"].numpy().astype(np.uint8)
        gt_rotation = batch["gt_rotation_matrix"].numpy().astype(np.float32)
        gt_pos = batch["gt_position"].numpy().astype(np.float32)
        has_rotation = batch["has_rotation"]
        has_pose = batch["has_pose"]

        pred_R = outputs.get("rotation_matrix", None)
        pred_pos = outputs.get("position", None)
        if pred_R is not None:
            pred_R = pred_R.detach().cpu().numpy().astype(np.float32)
        if pred_pos is not None:
            pred_pos = pred_pos.detach().cpu().numpy().astype(np.float32)

        for i in range(B):
            # ---- Detection ----
            pb = clip_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_bbox_norm[i], img_size), img_size)
            gb = clip_bbox_xyxy(gt_bbox_xyxy[i].astype(np.float32), img_size)
            det_iou = bbox_iou_np(pb, gb)

            # ---- Model mask metrics ----
            pm = (pred_mask[i] > 0.5).astype(np.uint8)
            gm = (gt_mask[i] > 0).astype(np.uint8)
            model_seg = SegMetrics.all(pm, gm)

            # ---- SAM mask metrics: bbox + point(center) -> SAM ----
            sam_seg = None
            if use_sam and sam_predictor is not None:
                try:
                    sat_rgb = batch["sat_rgb"][i]  # uint8 RGB, SxS
                    sam_predictor.set_image(sat_rgb)
                    cx = (pb[0] + pb[2]) * 0.5
                    cy = (pb[1] + pb[3]) * 0.5
                    point = np.array([[cx, cy]], dtype=np.float32)
                    plabel = np.array([1], dtype=np.int32)
                    box = pb.astype(np.float32)
                    masks, _, _ = sam_predictor.predict(
                        point_coords=point,
                        point_labels=plabel,
                        box=box,
                        multimask_output=False,
                    )
                    sm = masks[0].astype(np.uint8)
                    sam_seg = SegMetrics.all(sm, gm)
                except Exception:
                    sam_seg = None

            # ---- Pose metrics ----
            rotation_error_deg = None
            pos_err_px = None
            if has_rotation[i] and pred_R is not None:
                rotation_error_deg = geodesic_rotation_error_deg_np(pred_R[i], gt_rotation[i])
            if has_pose[i] and pred_pos is not None:
                pos_err_px = float(np.linalg.norm(pred_pos[i] - gt_pos[i]) * img_size)

            evaluator.add(
                det_iou=det_iou,
                model_seg=model_seg,
                sam_seg=sam_seg,
                rotation_error_deg=rotation_error_deg,
                pos_err_px=pos_err_px,
                task_type=batch["task_type"][i],
                size_category=batch["size_category"][i],
                shape_category=batch["shape_category"][i],
            )

    return evaluator.summarize()


# =========================
# 打印与保存
# =========================

def _fmt(v, w=8, decimals=4):
    if v is None:
        return "-".center(w)
    if isinstance(v, (int, np.integer)):
        return str(int(v)).rjust(w)
    return f"{float(v):.{decimals}f}".rjust(w)


def print_grouped(results: OrderedDict, split_name: str, prompt_type: str, has_model_mask=True, has_sam=True):
    print("\n" + "=" * 120)
    print(f"  Cross-View V2 Evaluation — {split_name} | prompt={prompt_type}")
    print("=" * 120)

    hdr_det = f'{"Count":>7} {"Det_mIoU":>9} {"ACC@25":>8} {"ACC@50":>8} {"RotErr":>8} {"PosErrPx":>9}'
    hdr_model = f' │ {"M_mIoU":>8} {"M_mDice":>8} {"M_AAE":>9} {"M_ME":>8}' if has_model_mask else ""
    hdr_sam = f' │ {"S_mIoU":>8} {"S_mDice":>8} {"S_AAE":>9} {"S_ME":>8}' if has_sam else ""

    print(f'  {"Group":<30} {hdr_det}{hdr_model}{hdr_sam}')
    print(f'  {"─" * 30} {"─" * 54}{"─" * 40 if has_model_mask else ""}{"─" * 40 if has_sam else ""}')

    for name, r in results.items():
        if r is None:
            continue
        line = f'  {name:<30} '
        line += (
            f'{_fmt(r["count"], 7, 0)} {_fmt(r["det_iou"], 9)} {_fmt(r["det_acc25"], 8)} {_fmt(r["det_acc50"], 8)} '
            f'{_fmt(r.get("rotation_error_deg"), 8, 2)} {_fmt(r.get("pos_err_px"), 9, 2)}'
        )
        if has_model_mask:
            line += f' │ {_fmt(r.get("model_miou"), 8)} {_fmt(r.get("model_mdice"), 8)} {_fmt(r.get("model_aae"), 9, 1)} {_fmt(r.get("model_me"), 8, 2)}'
        if has_sam:
            line += f' │ {_fmt(r.get("sam_miou"), 8)} {_fmt(r.get("sam_mdice"), 8)} {_fmt(r.get("sam_aae"), 9, 1)} {_fmt(r.get("sam_me"), 8, 2)}'
        print(line)


def save_results_json(all_results: Dict[str, Dict[str, OrderedDict]], path: str):
    out = {}
    for split, prompt_dict in all_results.items():
        out[split] = {}
        for prompt_type, groups in prompt_dict.items():
            out[split][prompt_type] = {}
            for gname, gval in groups.items():
                if gval is None:
                    continue
                out[split][prompt_type][gname] = {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in gval.items()
                }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {path}")


# =========================
# Main
# =========================

def parse_args():
    ws_dir = get_workspace_dir()
    output_dir = ws_dir / "output_v3"
    p = argparse.ArgumentParser(description="Evaluate Cross-View Localizer V3 (grouped metrics)")
    p.add_argument("--config", type=str, default=str(output_dir / "config.yaml"))
    p.add_argument("--checkpoint", type=str, default=str(output_dir / "best"))
    p.add_argument("--image_root", type=str, default="")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--splits", nargs="+", default=["test", "unseen_test"])
    p.add_argument("--prompt_types", nargs="+", default=["point", "bbox", "mask"],
                   choices=["point", "bbox", "mask"])
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--sam_checkpoint", type=str, required=True, help="segment-anything checkpoint path")
    p.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    ws_dir = get_workspace_dir()
    os.environ.setdefault("ROOT_DIR", str(ws_dir.parent))
    os.environ.setdefault("WORKSPACE_NAME", ws_dir.name)
    # IMPORTANT for multi-process multi-GPU launch:
    # If launcher already sets CUDA_VISIBLE_DEVICES per process, do NOT override it here,
    # otherwise all workers may collapse to the same physical GPU and trigger OOM.
    preset_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not preset_cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        preset_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    else:
        print(f"Using preset CUDA_VISIBLE_DEVICES={preset_cvd}; ignore --gpu={args.gpu}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA_VISIBLE_DEVICES={preset_cvd}")

    cfg = load_cfg_with_env(args.config)

    data_root = cfg["data"]["data_root"]
    image_root = args.image_root or data_root
    img_size = int(cfg["data"].get("img_size", 518))

    # build/load model
    print("\n[1/3] Loading model ...")
    model = build_model_from_cfg(cfg, device)

    ckpt_file = resolve_checkpoint(Path(args.checkpoint).resolve())
    print(f"Checkpoint file: {ckpt_file}")
    obj = torch.load(str(ckpt_file), map_location="cpu")
    sd = extract_state_dict(obj)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded state_dict keys: {len(sd)}")
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # load SAM
    print("\n[2/3] Loading SAM ...")
    _cvos_dir = str(ws_dir.parents[1] / "baseline" / "CVOS-Code")
    import sys
    if os.path.isdir(_cvos_dir):
        sys.path.insert(0, _cvos_dir)
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad = False
    sam_predictor = SamPredictor(sam)

    split_to_json = {
        "test": "test_all.json",
        "unseen_test": "unseen_test.json",
    }

    all_results: Dict[str, Dict[str, OrderedDict]] = OrderedDict()

    print("\n[3/3] Evaluating ...")
    for split in args.splits:
        if split not in split_to_json:
            print(f"Skip unknown split: {split}")
            continue

        json_path = ws_dir / "data" / split_to_json[split]
        if not json_path.exists():
            print(f"Skip missing file: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        ds = EvalDatasetV2(data_list, image_root=image_root, img_size=img_size)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_eval,
        )

        all_results[split] = OrderedDict()
        for prompt_type in args.prompt_types:
            t0 = time.time()
            results = evaluate_split(
                model=model,
                sam_predictor=sam_predictor,
                loader=loader,
                img_size=img_size,
                device=device,
                use_sam=True,
                prompt_type=prompt_type,
            )
            dt = time.time() - t0

            print_grouped(results, split_name=split, prompt_type=prompt_type, has_model_mask=True, has_sam=True)
            print(f"Split {split} | prompt={prompt_type} done in {dt:.1f}s")
            all_results[split][prompt_type] = results

    if args.save_json:
        save_results_json(all_results, args.save_json)

    print("\n" + "=" * 120)
    print("All evaluations completed!")
    print("=" * 120)


if __name__ == "__main__":
    main()
