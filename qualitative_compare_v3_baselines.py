#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative comparison for V3 vs baselines (point prompt).

Workflow:
1) Run V3 + SPS on unseen_test (point prompt), select top-K samples per task by V3 mask mIoU.
2) Evaluate selected candidates with:
   - V3 + SPS (SAM Prompt Stage)
   - CVOS + SPS (SAM Prompt Stage)
   - DetGeo + SPS (SAM Prompt Stage)
3) For each task, pick top-N samples with largest (configurable) gap between CVOS and V3.
4) Save required visualizations and a 6-panel combined image.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


def parse_rgb_color(s: str) -> Tuple[int, int, int]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid RGB color string: {s}")
    vals = tuple(int(x) for x in parts)
    for v in vals:
        if v < 0 or v > 255:
            raise ValueError(f"RGB value out of range [0,255]: {s}")
    return vals


def decode_segmentation(segmentation: Any, h: int, w: int) -> np.ndarray:
    if segmentation is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(segmentation, list):
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


def iou_np(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def make_click_heatmap(click_y: float, click_x: float, size: int) -> np.ndarray:
    rows = np.arange(size, dtype=np.float64)
    norm = np.sqrt(float(size * size + size * size))
    dh = (rows - click_y) ** 2
    dw = (rows - click_x) ** 2
    dist = np.sqrt(dh[:, None] + dw[None, :])
    val = 1.0 - dist / norm
    return (val * val).astype(np.float32)


def bbox_xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b.astype(np.float32)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def bbox_cxcywh_norm_to_xyxy_abs(b: torch.Tensor, img_size: int) -> np.ndarray:
    cx, cy, w, h = b.detach().cpu().numpy().astype(np.float32)
    x1 = (cx - w / 2.0) * img_size
    y1 = (cy - h / 2.0) * img_size
    x2 = (cx + w / 2.0) * img_size
    y2 = (cy + h / 2.0) * img_size
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def sanitize_bbox_xyxy(b: np.ndarray, size: int) -> np.ndarray:
    out = b.copy().astype(np.float32)
    out[0::2] = np.clip(out[0::2], 0, size - 1)
    out[1::2] = np.clip(out[1::2], 0, size - 1)
    x1, y1, x2, y2 = [float(v) for v in out]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1:
        x2 = min(float(size - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(size - 1), y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def rotation_matrix_to_yaw_np(rotation_matrix: np.ndarray) -> float:
    rot = rotation_matrix
    if rot.shape == (4, 4):
        rot = rot[:3, :3]
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def euler_to_rotation_matrix_np(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def build_gt_rotation_matrix(item: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    if "relative_yaw" in item:
        yaw = np.deg2rad(float(item.get("relative_yaw", 0.0)))
        default_pitch = 45.0 if "drone" in str(item.get("mono_filename", "")).lower() else 90.0
        pitch = np.deg2rad(float(item.get("relative_pitch", default_pitch)))
        roll = np.deg2rad(float(item.get("relative_roll", 0.0)))
        return euler_to_rotation_matrix_np(yaw, pitch, roll), True
    if "rotation" in item and item.get("rotation") is not None:
        yaw = np.deg2rad(float(item["rotation"]))
        return euler_to_rotation_matrix_np(yaw, 0.0, 0.0), True
    return np.eye(3, dtype=np.float32), False


def bbox_to_mask(bbox_xyxy: np.ndarray, size: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    x1 = int(np.clip(np.floor(x1), 0, size - 1))
    y1 = int(np.clip(np.floor(y1), 0, size - 1))
    x2 = int(np.clip(np.ceil(x2), 0, size))
    y2 = int(np.clip(np.ceil(y2), 0, size))
    m = np.zeros((size, size), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    return m


def resolve_image_paths(item: Dict[str, Any], image_root: str) -> Tuple[str, str]:
    city = item.get("city", "")
    mono_name = item["mono_filename"]
    sat_name = item.get("sat_filename") or item.get("sate_filename")
    if sat_name is None:
        raise KeyError("sample missing sat_filename/sate_filename")
    mono_path = os.path.join(image_root, city, "mono", mono_name)
    sat_path = os.path.join(image_root, city, "crop_sate", sat_name)
    return mono_path, sat_path


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_sample_for_size(item: Dict[str, Any], image_root: str, size: int) -> Dict[str, Any]:
    mono_path, sat_path = resolve_image_paths(item, image_root)
    mono = read_rgb(mono_path)
    sat = read_rgb(sat_path)
    h_m, w_m = mono.shape[:2]
    h_s, w_s = sat.shape[:2]

    mono_point = np.array(item["mono_point"][:2], dtype=np.float32)
    mono_bbox_xywh = np.array(item.get("mono_bbox", [0, 0, 0, 0])[:4], dtype=np.float32)
    mono_mask = decode_segmentation(item.get("mono_segmentation"), h_m, w_m)
    gt_mask = decode_segmentation(item.get("sate_segmentation"), h_s, w_s)
    gt_bbox_xywh = np.array(item["sate_bbox"][:4], dtype=np.float32)
    gt_rotation_matrix, has_gt_rotation = build_gt_rotation_matrix(item)
    gt_pos_raw = item.get("camera_position", None)
    has_gt_position = (gt_pos_raw is not None) and (len(gt_pos_raw) >= 2)
    if has_gt_position:
        gt_pos = np.array(gt_pos_raw[:2], dtype=np.float32)
    else:
        gt_pos = np.zeros((2,), dtype=np.float32)

    if (h_m, w_m) != (size, size):
        sx_m, sy_m = size / w_m, size / h_m
        mono = cv2.resize(mono, (size, size), interpolation=cv2.INTER_LINEAR)
        mono_point = np.array([mono_point[0] * sx_m, mono_point[1] * sy_m], dtype=np.float32)
        mono_bbox_xywh = np.array(
            [
                mono_bbox_xywh[0] * sx_m,
                mono_bbox_xywh[1] * sy_m,
                mono_bbox_xywh[2] * sx_m,
                mono_bbox_xywh[3] * sy_m,
            ],
            dtype=np.float32,
        )
        mono_mask = cv2.resize(mono_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)

    if (h_s, w_s) != (size, size):
        sx_s, sy_s = size / w_s, size / h_s
        sat = cv2.resize(sat, (size, size), interpolation=cv2.INTER_LINEAR)
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        gt_bbox_xywh = np.array(
            [
                gt_bbox_xywh[0] * sx_s,
                gt_bbox_xywh[1] * sy_s,
                gt_bbox_xywh[2] * sx_s,
                gt_bbox_xywh[3] * sy_s,
            ],
            dtype=np.float32,
        )
        if has_gt_position and np.max(np.abs(gt_pos)) > 1.5:
            gt_pos = np.array([gt_pos[0] * sx_s, gt_pos[1] * sy_s], dtype=np.float32)

    if has_gt_position:
        # Normalize position to [0,1] to align with V3 model output convention.
        if np.max(np.abs(gt_pos)) > 1.5:
            gt_pos = np.array([gt_pos[0] / float(size), gt_pos[1] / float(size)], dtype=np.float32)
        gt_pos = np.clip(gt_pos, 0.0, 1.0).astype(np.float32)

    return {
        "mono_rgb": mono,
        "sat_rgb": sat,
        "mono_point": mono_point,
        "mono_bbox_xywh": mono_bbox_xywh,
        "mono_mask": (mono_mask > 0).astype(np.uint8),
        "gt_mask": (gt_mask > 0).astype(np.uint8),
        "gt_bbox_xyxy": bbox_xywh_to_xyxy(gt_bbox_xywh),
        "gt_position": gt_pos,
        "gt_rotation_matrix": gt_rotation_matrix,
        "has_gt_position": bool(has_gt_position),
        "has_gt_rotation": bool(has_gt_rotation),
    }


def import_model_class(project_root: str, module_name: str, class_name: str, clear_prefixes: List[str] | None = None):
    old_sys_path = list(sys.path)
    try:
        sys.path.insert(0, project_root)
        prefixes = clear_prefixes or ["model"]
        for key in list(sys.modules.keys()):
            for p in prefixes:
                if key == p or key.startswith(f"{p}."):
                    del sys.modules[key]
                    break
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    finally:
        sys.path[:] = old_sys_path


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
        for key in ["module", "model", "state_dict", "model_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                if len(sd) > 0:
                    first_k = next(iter(sd.keys()))
                    if first_k.startswith("module."):
                        sd = {k[len("module."):]: v for k, v in sd.items()}
                return sd
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format")


def load_cfg_with_env(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".json"):
            cfg = json.load(f)
        else:
            import yaml

            cfg = yaml.safe_load(f)

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(x) for x in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    return _expand(cfg)


def load_v3_model(v3_root: str, cfg: Dict[str, Any], checkpoint: str, device: torch.device):
    old_sys_path = list(sys.path)
    try:
        sys.path.insert(0, v3_root)
        build_fn = importlib.import_module("models").build_cross_view_localizer_v2
    finally:
        sys.path[:] = old_sys_path

    mc = cfg["model"]
    dc = cfg["data"]
    model = build_fn(
        pretrained_pi3=None,
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
    ckpt_file = resolve_checkpoint(Path(checkpoint).resolve())
    obj = torch.load(str(ckpt_file), map_location="cpu")
    sd = extract_state_dict(obj)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_cvos_model(cvos_root: str, checkpoint: str, device: torch.device):
    TROGeo = import_model_class(cvos_root, "model.TROGeo", "TROGeo", clear_prefixes=["model", "utils"])
    model = TROGeo()
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_sd = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in model_sd}
    if len(sd) == 0:
        raise RuntimeError("No matching keys for CVOS checkpoint")
    model_sd.update(sd)
    model.load_state_dict(model_sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_detgeo_model(detgeo_root: str, checkpoint: str, device: torch.device):
    DetGeo = import_model_class(detgeo_root, "model.DetGeo", "DetGeo", clear_prefixes=["model", "utils"])
    model = DetGeo()
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_sd = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in model_sd}
    if len(sd) == 0:
        raise RuntimeError("No matching keys for DetGeo checkpoint")
    model_sd.update(sd)
    model.load_state_dict(model_sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_sam(cvos_root: str, sam_checkpoint: str, sam_model_type: str, device: torch.device):
    old_sys_path = list(sys.path)
    try:
        sys.path.insert(0, cvos_root)
        sam_mod = importlib.import_module("segment_anything")
    finally:
        sys.path[:] = old_sys_path
    sam = sam_mod.sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device).eval()
    for p in sam.parameters():
        p.requires_grad = False
    return sam


class V3PointDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, Any]], image_root: str, img_size: int):
        self.data_list = data_list
        self.image_root = image_root
        self.img_size = img_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        item = self.data_list[idx]
        prep = resize_sample_for_size(item, self.image_root, self.img_size)
        return {
            "front": to_tensor(Image.fromarray(prep["mono_rgb"])),
            "sat": to_tensor(Image.fromarray(prep["sat_rgb"])),
            "point": torch.from_numpy(prep["mono_point"]),
            "gt_mask": torch.from_numpy(prep["gt_mask"]),
            "task_type": item.get("task_type", "unknown"),
            "index": idx,
        }


def collate_v3_point(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "front": torch.stack([x["front"] for x in batch], dim=0),
        "sat": torch.stack([x["sat"] for x in batch], dim=0),
        "point": torch.stack([x["point"] for x in batch], dim=0),
        "gt_mask": torch.stack([x["gt_mask"] for x in batch], dim=0),
        "task_type": [x["task_type"] for x in batch],
        "index": [x["index"] for x in batch],
    }


@torch.no_grad()
def run_v3_point_all(
    model,
    sam,
    data_list: List[Dict[str, Any]],
    image_root: str,
    img_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[int, Dict[str, Any]]:
    ds = V3PointDataset(data_list, image_root=image_root, img_size=img_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_v3_point,
    )
    out: Dict[int, Dict[str, Any]] = {}
    for batch in tqdm(loader, desc="V3+SPS on full unseen_test"):
        front = batch["front"].to(device, non_blocking=True)
        sat = batch["sat"].to(device, non_blocking=True)
        point = batch["point"].to(device, non_blocking=True)
        point_coords = point.unsqueeze(1)
        point_labels = torch.ones(front.shape[0], 1, device=device)
        outputs = model(
            front_view=front,
            satellite_view=sat,
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
            mono_mask=None,
            sat_mask=None,
        )
        pred_boxes = outputs["pred_boxes"]  # [B, Q, 4]
        bbox_scores = outputs.get("bbox_scores", None)
        gt = batch["gt_mask"].numpy().astype(np.uint8)
        bsz = front.shape[0]
        for i, idx in enumerate(batch["index"]):
            if pred_boxes.shape[1] > 1 and bbox_scores is not None:
                q_idx = int(bbox_scores[i].argmax().item())
            else:
                q_idx = 0
            pb = sanitize_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_boxes[i, q_idx], img_size), img_size)
            bb = torch.from_numpy(pb).to(device=device, dtype=torch.float32).unsqueeze(0)
            pred_bin = sam_sps_from_bbox(sam=sam, ref_img=sat[i:i + 1], bb=bb, size=img_size, device=device)
            miou = iou_np(pred_bin, gt[i])
            out[int(idx)] = {
                "task_type": batch["task_type"][i],
                "v3_miou": miou,
            }
    return out


@torch.no_grad()
def infer_v3_one(
    model, sam, item: Dict[str, Any], image_root: str, size: int, device: torch.device
) -> Tuple[np.ndarray, float, Dict[str, Any], Dict[str, Any]]:
    prep = resize_sample_for_size(item, image_root, size)
    front = to_tensor(Image.fromarray(prep["mono_rgb"])).unsqueeze(0).to(device)
    sat = to_tensor(Image.fromarray(prep["sat_rgb"])).unsqueeze(0).to(device)
    point = torch.from_numpy(prep["mono_point"]).unsqueeze(0).to(device)
    point_coords = point.unsqueeze(1)
    point_labels = torch.ones(1, 1, device=device)
    outputs = model(
        front_view=front,
        satellite_view=sat,
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
        mono_mask=None,
        sat_mask=None,
    )
    pred_boxes = outputs["pred_boxes"]  # [1, Q, 4]
    bbox_scores = outputs.get("bbox_scores", None)
    if pred_boxes.shape[1] > 1 and bbox_scores is not None:
        q_idx = int(bbox_scores[0].argmax().item())
    else:
        q_idx = 0
    pb = sanitize_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_boxes[0, q_idx], size), size)
    bb = torch.from_numpy(pb).to(device=device, dtype=torch.float32).unsqueeze(0)
    pred_bin = sam_sps_from_bbox(sam=sam, ref_img=sat, bb=bb, size=size, device=device)
    miou = iou_np(pred_bin, prep["gt_mask"])
    pred_pos = outputs.get("position", None)
    pred_rot = outputs.get("rotation_matrix", None)
    pose = {
        "pred_position": None if pred_pos is None else pred_pos[0].detach().cpu().numpy().astype(np.float32),
        "pred_rotation_matrix": None if pred_rot is None else pred_rot[0].detach().cpu().numpy().astype(np.float32),
        "gt_position": prep["gt_position"].astype(np.float32),
        "gt_rotation_matrix": prep["gt_rotation_matrix"].astype(np.float32),
        "has_gt_position": prep["has_gt_position"],
        "has_gt_rotation": prep["has_gt_rotation"],
    }
    return pred_bin, miou, prep, pose


@torch.no_grad()
def decode_best_anchor_bbox(raw_anchor: torch.Tensor, size: int, device: torch.device) -> torch.Tensor:
    if raw_anchor.ndim == 4:
        b, c, gh, gw = raw_anchor.shape
        if c % 5 != 0:
            raise RuntimeError(f"Unexpected raw_anchor shape: {tuple(raw_anchor.shape)}")
        raw_anchor = raw_anchor.view(b, c // 5, 5, gh, gw)
    elif raw_anchor.ndim == 5:
        b, _, _, gh, gw = raw_anchor.shape
    else:
        raise RuntimeError(f"Unexpected raw_anchor ndim={raw_anchor.ndim}, shape={tuple(raw_anchor.shape)}")

    conf = raw_anchor[:, :, 4, :, :]
    anchors = np.array(
        [44, 41, 85, 85, 143, 130, 266, 153, 182, 235, 187, 444, 467, 194, 321, 299, 440, 433],
        dtype=np.float32,
    ).reshape(-1, 2)[::-1].copy()
    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    stride = size / float(gh)
    sa = anchors_t / stride
    flat = conf[0].reshape(-1)
    best = int(flat.argmax().item())
    n = best // (gh * gw)
    gj = (best % (gh * gw)) // gw
    gi = (best % (gh * gw)) % gw
    x = (raw_anchor[0, n, 0, gj, gi].sigmoid() + gi) * stride
    y = (raw_anchor[0, n, 1, gj, gi].sigmoid() + gj) * stride
    w = torch.exp(raw_anchor[0, n, 2, gj, gi]) * sa[n, 0] * stride
    h = torch.exp(raw_anchor[0, n, 3, gj, gi]) * sa[n, 1] * stride
    return torch.tensor([x - w / 2, y - h / 2, x + w / 2, y + h / 2], device=device).unsqueeze(0)


@torch.no_grad()
def sam_sps_from_bbox(sam, ref_img: torch.Tensor, bb: torch.Tensor, size: int, device: torch.device) -> np.ndarray:
    ref_img_sam = F.interpolate(ref_img, size=(1024, 1024), mode="bilinear", align_corners=False)
    ref_emb = sam.image_encoder(ref_img_sam)
    bb_t_sam = bb * (1024.0 / size)
    cx = (bb_t_sam[:, 0] + bb_t_sam[:, 2]) / 2.0
    cy = (bb_t_sam[:, 1] + bb_t_sam[:, 3]) / 2.0
    pts = torch.stack([cx, cy], dim=-1).unsqueeze(1)
    lbls = torch.ones(1, 1, device=device)
    sparse, dense = sam.prompt_encoder(points=(pts, lbls), boxes=bb_t_sam, masks=None)
    lo, _ = sam.mask_decoder(
        image_embeddings=ref_emb,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    sm = F.interpolate(lo, size=(size, size), mode="bilinear", align_corners=False)
    sm = sm.squeeze().detach().cpu().numpy()
    return (sm > 0.5).astype(np.uint8)


@torch.no_grad()
def infer_cvos_sps_one(
    model,
    sam,
    item: Dict[str, Any],
    image_root: str,
    size: int,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    prep = resize_sample_for_size(item, image_root, size)
    transform = Compose(
        [
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    q = transform(prep["mono_rgb"]).unsqueeze(0).to(device)
    r = transform(prep["sat_rgb"]).unsqueeze(0).to(device)
    click = prep["mono_point"]
    mat = make_click_heatmap(float(click[1]), float(click[0]), size)
    click_t = torch.from_numpy(mat).unsqueeze(0).to(device)

    raw_anchor, _ = model(q, r, click_t)
    bb = decode_best_anchor_bbox(raw_anchor, size=size, device=device)
    pred_bin = sam_sps_from_bbox(sam=sam, ref_img=r, bb=bb, size=size, device=device)
    miou = iou_np(pred_bin, prep["gt_mask"])
    return pred_bin, miou


@torch.no_grad()
def infer_detgeo_sps_one(
    model,
    sam,
    item: Dict[str, Any],
    image_root: str,
    size: int,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    prep = resize_sample_for_size(item, image_root, size)
    transform = Compose(
        [
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    q = transform(prep["mono_rgb"]).unsqueeze(0).to(device)
    r = transform(prep["sat_rgb"]).unsqueeze(0).to(device)
    click = prep["mono_point"]
    mat = make_click_heatmap(float(click[1]), float(click[0]), size)
    click_t = torch.from_numpy(mat).unsqueeze(0).to(device)

    raw_anchor, _ = model(q, r, click_t)
    bb = decode_best_anchor_bbox(raw_anchor, size=size, device=device)
    pred_bin = sam_sps_from_bbox(sam=sam, ref_img=r, bb=bb, size=size, device=device)
    miou = iou_np(pred_bin, prep["gt_mask"])
    return pred_bin, miou


def blend_mask(base_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = base_rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[m] = out[m] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_pred_mask_like_visualize(
    base_rgb: np.ndarray,
    pred_mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.55,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Follow visualize.py style:
    1) threshold prediction to binary mask first
    2) overlay only on masked pixels
    This avoids visually amplifying interpolation artifacts from soft masks.
    """
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    out = base_rgb.copy().astype(np.float32)
    m = pred_bin.astype(bool)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[m] = out[m] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_pose_on_satellite(
    sat_rgb: np.ndarray,
    pos_norm: np.ndarray | None,
    yaw_rad: float | None,
    point_color: Tuple[int, int, int],
    arrow_color: Tuple[int, int, int],
    point_radius: int = 6,
    arrow_len_ratio: float = 0.08,
    arrow_thickness: int = 3,
) -> np.ndarray:
    out = sat_rgb.copy()
    if pos_norm is None:
        return out
    h, w = out.shape[:2]
    px = int(np.clip(round(float(pos_norm[0]) * w), 0, w - 1))
    py = int(np.clip(round(float(pos_norm[1]) * h), 0, h - 1))
    cv2.circle(out, (px, py), point_radius, point_color, thickness=-1, lineType=cv2.LINE_AA)
    if yaw_rad is None:
        return out
    arrow_len = int(round(min(h, w) * arrow_len_ratio))
    dx = int(round(np.cos(float(yaw_rad)) * arrow_len))
    dy = int(round(np.sin(float(yaw_rad)) * arrow_len))
    cv2.arrowedLine(
        out,
        (px, py),
        (int(np.clip(px + dx, 0, w - 1)), int(np.clip(py + dy, 0, h - 1))),
        arrow_color,
        thickness=arrow_thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.28,
    )
    return out


def draw_point(rgb: np.ndarray, point_xy: np.ndarray, radius: int = 6) -> np.ndarray:
    out = rgb.copy()
    x, y = int(round(float(point_xy[0]))), int(round(float(point_xy[1])))
    cv2.circle(out, (x, y), radius, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    return out


def draw_bbox_xywh(rgb: np.ndarray, bbox_xywh: np.ndarray, thickness: int = 3) -> np.ndarray:
    out = rgb.copy()
    x, y, w, h = bbox_xywh.astype(np.float32)
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)
    return out


def ensure_dirs(base_vis_root: Path):
    for task_root, prompt_prefix in [("D2S", "D"), ("G2S", "G")]:
        for name in [
            f"{prompt_prefix}withPoint",
            f"{prompt_prefix}withBbox",
            f"{prompt_prefix}withMask",
            "gtSate",
            "detgeoSate",
            "cvosSate",
            "v3Sate",
            "v3Pose",
            "gtPose",
            "combine",
        ]:
            (base_vis_root / task_root / name).mkdir(parents=True, exist_ok=True)


def clear_vis_outputs(base_vis_root: Path):
    for task_root in ["D2S", "G2S"]:
        task_dir = base_vis_root / task_root
        if task_dir.exists():
            shutil.rmtree(task_dir)


def combine_row(images: List[np.ndarray], panel_size: int) -> np.ndarray:
    resized = [cv2.resize(im, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR) for im in images]
    return np.concatenate(resized, axis=1)


@dataclass
class CandidateMetrics:
    idx: int
    task_type: str
    v3_miou: float
    cvos_sps_miou: float
    detgeo_miou: float

    @property
    def gap_abs(self) -> float:
        return float(abs(self.cvos_sps_miou - self.v3_miou))

    @property
    def gap_cvos_minus_v3(self) -> float:
        return float(self.cvos_sps_miou - self.v3_miou)

    @property
    def gap_v3_minus_cvos(self) -> float:
        return float(self.v3_miou - self.cvos_sps_miou)


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative comparison: V3 vs CVOS+SPS vs DetGeo+SPS (point prompt)")
    p.add_argument("--json_path", type=str, default="/data/home/scxi704/run/xhj/data/json/unseen_test.json")
    p.add_argument("--image_root", type=str, default="/data/home/scxi704/run/xhj/data")

    p.add_argument("--v3_root", type=str, default="/data/home/scxi704/run/xhj/location_v4")
    p.add_argument("--v3_config", type=str, default="/data/home/scxi704/run/xhj/location_v4/output_v3/ablation_4_all_on/config.yaml")
    p.add_argument("--v3_checkpoint", type=str, default="/data/home/scxi704/run/xhj/location_v4/output_v3/ablation_4_all_on/best")

    p.add_argument("--cvos_root", type=str, default="/data/home/scxi704/run/baseline/CVOS-Code")
    p.add_argument("--cvos_checkpoint", type=str, required=True)
    p.add_argument("--detgeo_root", type=str, default="/data/home/scxi704/run/baseline/DetGeo")
    p.add_argument("--detgeo_checkpoint", type=str, required=True)

    p.add_argument(
        "--sam_checkpoint",
        type=str,
        default="/data/home/scxi704/run/baseline/CVOS-Code/segment_anything/weights/sam_vit_h_4b8939.pth",
    )
    p.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])

    p.add_argument("--top_k_candidates", type=int, default=200)
    p.add_argument("--vis_k", type=int, default=20)
    p.add_argument("--gap_mode", type=str, default="abs", choices=["abs", "cvos_minus_v3", "v3_minus_cvos"])

    p.add_argument("--v3_batch_size", type=int, default=8)
    p.add_argument("--v3_num_workers", type=int, default=8)
    p.add_argument("--cvos_img_size", type=int, default=512)
    p.add_argument("--detgeo_img_size", type=int, default=512)
    p.add_argument("--panel_size", type=int, default=512)
    p.add_argument("--output_root", type=str, default="/data/home/scxi704/run/xhj/location_v4/vis")
    p.add_argument("--gt_mask_color", type=str, default="255,0,0")
    p.add_argument("--gt_mask_alpha", type=float, default=0.55)
    # Legacy shared prediction mask style (kept for backward compatibility).
    p.add_argument("--pred_mask_color", type=str, default="0,255,0")
    p.add_argument("--pred_mask_alpha", type=float, default=0.55)
    # Per-model prediction mask style (if omitted, fallback to legacy shared style above).
    p.add_argument("--v3_mask_color", type=str, default="")
    p.add_argument("--v3_mask_alpha", type=float, default=-1.0)
    p.add_argument("--cvos_mask_color", type=str, default="")
    p.add_argument("--cvos_mask_alpha", type=float, default=-1.0)
    p.add_argument("--detgeo_mask_color", type=str, default="")
    p.add_argument("--detgeo_mask_alpha", type=float, default=-1.0)
    p.add_argument("--pose_point_color_pred", type=str, default="0,255,0")
    p.add_argument("--pose_arrow_color_pred", type=str, default="0,255,0")
    p.add_argument("--pose_point_color_gt", type=str, default="255,0,0")
    p.add_argument("--pose_arrow_color_gt", type=str, default="255,0,0")
    p.add_argument("--clean_output_each_run", action="store_true", default=True)
    p.add_argument("--no_clean_output_each_run", action="store_false", dest="clean_output_each_run")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--save_summary_json", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    gt_mask_color = parse_rgb_color(args.gt_mask_color)
    pred_mask_color = parse_rgb_color(args.pred_mask_color)
    v3_mask_color = parse_rgb_color(args.v3_mask_color) if args.v3_mask_color else pred_mask_color
    cvos_mask_color = parse_rgb_color(args.cvos_mask_color) if args.cvos_mask_color else pred_mask_color
    detgeo_mask_color = parse_rgb_color(args.detgeo_mask_color) if args.detgeo_mask_color else pred_mask_color
    v3_mask_alpha = args.v3_mask_alpha if args.v3_mask_alpha >= 0.0 else args.pred_mask_alpha
    cvos_mask_alpha = args.cvos_mask_alpha if args.cvos_mask_alpha >= 0.0 else args.pred_mask_alpha
    detgeo_mask_alpha = args.detgeo_mask_alpha if args.detgeo_mask_alpha >= 0.0 else args.pred_mask_alpha
    pose_point_color_pred = parse_rgb_color(args.pose_point_color_pred)
    pose_arrow_color_pred = parse_rgb_color(args.pose_arrow_color_pred)
    pose_point_color_gt = parse_rgb_color(args.pose_point_color_gt)
    pose_arrow_color_gt = parse_rgb_color(args.pose_arrow_color_gt)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    with open(args.json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    print(f"Loaded unseen_test samples: {len(data_list)}")

    cfg = load_cfg_with_env(args.v3_config)
    v3_img_size = int(cfg["data"].get("img_size", 518))

    print("[1/5] Loading models ...")
    v3_model = load_v3_model(args.v3_root, cfg, args.v3_checkpoint, device)
    cvos_model = load_cvos_model(args.cvos_root, args.cvos_checkpoint, device)
    detgeo_model = load_detgeo_model(args.detgeo_root, args.detgeo_checkpoint, device)
    sam_model = load_sam(args.cvos_root, args.sam_checkpoint, args.sam_model_type, device)

    print("[2/5] Running V3 on full unseen_test (point prompt) ...")
    v3_full = run_v3_point_all(
        model=v3_model,
        sam=sam_model,
        data_list=data_list,
        image_root=args.image_root,
        img_size=v3_img_size,
        device=device,
        batch_size=args.v3_batch_size,
        num_workers=args.v3_num_workers,
    )

    task_to_candidates: Dict[str, List[int]] = {"drone": [], "ground": []}
    for idx, rec in v3_full.items():
        tt = rec["task_type"]
        if tt in task_to_candidates:
            task_to_candidates[tt].append(idx)
    for tt in task_to_candidates:
        task_to_candidates[tt].sort(key=lambda x: v3_full[x]["v3_miou"], reverse=True)
        task_to_candidates[tt] = task_to_candidates[tt][: max(0, args.top_k_candidates)]
        print(f"Top-{args.top_k_candidates} candidates | task={tt}: {len(task_to_candidates[tt])}")

    print("[3/5] Evaluating CVOS + DetGeo+SPS on candidates ...")
    all_candidate_metrics: Dict[str, List[CandidateMetrics]] = {"drone": [], "ground": []}
    for tt in ["drone", "ground"]:
        for idx in tqdm(task_to_candidates[tt], desc=f"Baselines on {tt} candidates"):
            item = data_list[idx]
            _, cvos_miou = infer_cvos_sps_one(cvos_model, sam_model, item, args.image_root, args.cvos_img_size, device)
            _, detgeo_miou = infer_detgeo_sps_one(
                detgeo_model, sam_model, item, args.image_root, args.detgeo_img_size, device
            )
            all_candidate_metrics[tt].append(
                CandidateMetrics(
                    idx=idx,
                    task_type=tt,
                    v3_miou=float(v3_full[idx]["v3_miou"]),
                    cvos_sps_miou=float(cvos_miou),
                    detgeo_miou=float(detgeo_miou),
                )
            )

    def _gap_value(m: CandidateMetrics) -> float:
        if args.gap_mode == "abs":
            return m.gap_abs
        if args.gap_mode == "cvos_minus_v3":
            return m.gap_cvos_minus_v3
        return m.gap_v3_minus_cvos

    selected_for_vis: Dict[str, List[CandidateMetrics]] = {}
    for tt in ["drone", "ground"]:
        sorted_metrics = sorted(all_candidate_metrics[tt], key=_gap_value, reverse=True)
        selected_for_vis[tt] = sorted_metrics[: max(0, args.vis_k)]
        print(f"Selected vis samples | task={tt}: {len(selected_for_vis[tt])}")

    vis_root = Path(args.output_root)
    if args.clean_output_each_run:
        clear_vis_outputs(vis_root)
    ensure_dirs(vis_root)

    print("[4/5] Saving visualizations ...")
    summary = {"args": vars(args), "selected": {"drone": [], "ground": []}}

    for tt in ["drone", "ground"]:
        is_drone = tt == "drone"
        task_folder = "D2S" if is_drone else "G2S"
        prompt_prefix = "D" if is_drone else "G"
        for rank, m in enumerate(tqdm(selected_for_vis[tt], desc=f"Visualize {tt}"), start=1):
            item = data_list[m.idx]

            v3_pred, v3_miou_check, prep_v3, pose_info = infer_v3_one(
                v3_model, sam_model, item, args.image_root, v3_img_size, device
            )
            cvos_pred, cvos_miou_check = infer_cvos_sps_one(
                cvos_model, sam_model, item, args.image_root, args.cvos_img_size, device
            )
            detgeo_pred, detgeo_miou_check = infer_detgeo_sps_one(
                detgeo_model, sam_model, item, args.image_root, args.detgeo_img_size, device
            )

            # Prompt visualizations use V3-resized front image and annotations.
            front_rgb = prep_v3["mono_rgb"]
            point_img = draw_point(front_rgb, prep_v3["mono_point"])
            bbox_img = draw_bbox_xywh(front_rgb, prep_v3["mono_bbox_xywh"])
            mask_img = blend_mask(front_rgb, prep_v3["mono_mask"], (255, 0, 0), alpha=0.45)

            # Satellite visualizations: GT mask saved independently; each model panel uses prediction-only overlay.
            # For V3, use threshold-first overlay (visualize.py style) to reduce soft-mask artifacts.
            gt_sat = blend_mask(prep_v3["sat_rgb"], prep_v3["gt_mask"], gt_mask_color, alpha=args.gt_mask_alpha)
            v3_sat = overlay_pred_mask_like_visualize(
                prep_v3["sat_rgb"], v3_pred, v3_mask_color, alpha=v3_mask_alpha, threshold=0.5
            )
            prep_cvos = resize_sample_for_size(item, args.image_root, args.cvos_img_size)
            cvos_sat = blend_mask(prep_cvos["sat_rgb"], cvos_pred, cvos_mask_color, alpha=cvos_mask_alpha)
            prep_det = resize_sample_for_size(item, args.image_root, args.detgeo_img_size)
            detgeo_sat = blend_mask(prep_det["sat_rgb"], detgeo_pred, detgeo_mask_color, alpha=detgeo_mask_alpha)

            pred_pose_yaw = None
            if pose_info["pred_rotation_matrix"] is not None:
                pred_pose_yaw = rotation_matrix_to_yaw_np(pose_info["pred_rotation_matrix"])
            gt_pose_yaw = None
            if pose_info["has_gt_rotation"]:
                gt_pose_yaw = rotation_matrix_to_yaw_np(pose_info["gt_rotation_matrix"])
            pred_pose_pos = pose_info["pred_position"]
            gt_pose_pos = pose_info["gt_position"] if pose_info["has_gt_position"] else None
            v3_pose_img = draw_pose_on_satellite(
                prep_v3["sat_rgb"],
                pred_pose_pos,
                pred_pose_yaw,
                point_color=pose_point_color_pred,
                arrow_color=pose_arrow_color_pred,
            )
            gt_pose_img = draw_pose_on_satellite(
                prep_v3["sat_rgb"],
                gt_pose_pos,
                gt_pose_yaw,
                point_color=pose_point_color_gt,
                arrow_color=pose_arrow_color_gt,
            )

            key = f"rank{rank:03d}_idx{m.idx:05d}_gap{_gap_value(m):.4f}"

            p_point = vis_root / task_folder / f"{prompt_prefix}withPoint" / f"{key}.png"
            p_bbox = vis_root / task_folder / f"{prompt_prefix}withBbox" / f"{key}.png"
            p_mask = vis_root / task_folder / f"{prompt_prefix}withMask" / f"{key}.png"
            p_gt = vis_root / task_folder / "gtSate" / f"{key}.png"
            p_det = vis_root / task_folder / "detgeoSate" / f"{key}.png"
            p_cvos = vis_root / task_folder / "cvosSate" / f"{key}.png"
            p_v3 = vis_root / task_folder / "v3Sate" / f"{key}.png"
            p_v3_pose = vis_root / task_folder / "v3Pose" / f"{key}.png"
            p_gt_pose = vis_root / task_folder / "gtPose" / f"{key}.png"
            p_combine = vis_root / task_folder / "combine" / f"{key}.png"

            Image.fromarray(point_img).save(p_point)
            Image.fromarray(bbox_img).save(p_bbox)
            Image.fromarray(mask_img).save(p_mask)
            Image.fromarray(gt_sat).save(p_gt)
            Image.fromarray(detgeo_sat).save(p_det)
            Image.fromarray(cvos_sat).save(p_cvos)
            Image.fromarray(v3_sat).save(p_v3)
            Image.fromarray(v3_pose_img).save(p_v3_pose)
            Image.fromarray(gt_pose_img).save(p_gt_pose)

            combined = combine_row([point_img, detgeo_sat, cvos_sat, v3_sat, gt_sat], panel_size=args.panel_size)
            Image.fromarray(combined).save(p_combine)

            summary["selected"][tt].append(
                {
                    "rank": rank,
                    "idx": m.idx,
                    "task_type": tt,
                    "mono_filename": item.get("mono_filename", ""),
                    "sat_filename": item.get("sat_filename") or item.get("sate_filename", ""),
                    "v3_miou": float(v3_miou_check),
                    "cvos_miou": float(cvos_miou_check),
                    "cvos_sps_miou": float(cvos_miou_check),
                    "detgeo_sps_miou": float(detgeo_miou_check),
                    "gap_abs": float(abs(cvos_miou_check - v3_miou_check)),
                    "gap_cvos_minus_v3": float(cvos_miou_check - v3_miou_check),
                }
            )

    summary_path = Path(args.save_summary_json) if args.save_summary_json else (vis_root / "selection_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[5/5] Done.")
    print(f"Visualization root: {vis_root}")
    print(f"Summary json: {summary_path}")


if __name__ == "__main__":
    main()

