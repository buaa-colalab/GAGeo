#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative comparison: V3 vs CVOS+SPS vs DetGeo+SPS on CVOGL-Seg dataset.

Subsets:
  SVI         — mono (from panorama) -> satellite  (point prompt on mono image)
  DroneAerial — drone -> satellite                 (point prompt on drone image)

Note: For SVI, we visualize the *monocular* crop (not the raw panorama)
      as the query image, matching the online evaluation pipeline.

Panel layout (6-panel combined):
  [query+point | sat_raw | sat+GT_mask | DetGeo+SPS | CVOS+SPS | V3]

Workflow:
  1) Load data via CVOGLSegOnlineDataset (handles panorama→mono conversion).
  2) Run V3 on all samples (point prompt) → compute mask mIoU.
  3) Select top-K candidates per subset (sorted by V3 mIoU).
  4) Run CVOS+SPS / DetGeo+SPS on candidates.
  5) Pick top-N by gap metric.
  6) Save 6-panel visualizations and summary JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

# Import the online dataset module (same directory)
from cvoglseg_online_data import CVOGLSegOnlineDataset


# ──────────────────────────── Utility functions ────────────────────────────


def parse_rgb_color(s: str) -> Tuple[int, int, int]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid RGB color string: {s}")
    vals = tuple(int(x) for x in parts)
    for v in vals:
        if v < 0 or v > 255:
            raise ValueError(f"RGB value out of range [0,255]: {s}")
    return vals


def iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def make_click_heatmap(click_y: float, click_x: float, size: int) -> np.ndarray:
    rows = np.arange(size, dtype=np.float64)
    norm = np.sqrt(float(size * size + size * size))
    dh = (rows - click_y) ** 2
    dw = (rows - click_x) ** 2
    dist = np.sqrt(dh[:, None] + dw[None, :])
    val = 1.0 - dist / norm
    return (val * val).astype(np.float32)


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


# ─────────────────────────── Model loading helpers ──────────────────────────


def import_model_class(project_root, module_name, class_name, clear_prefixes=None):
    old = list(sys.path)
    try:
        sys.path.insert(0, project_root)
        for k in list(sys.modules.keys()):
            for p in (clear_prefixes or ["model"]):
                if k == p or k.startswith(f"{p}."):
                    del sys.modules[k]
                    break
        return getattr(importlib.import_module(module_name), class_name)
    finally:
        sys.path[:] = old


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    for c in [
        path / "pytorch_model" / "mp_rank_00_model_states.pt",
        path / "mp_rank_00_model_states.pt",
        path / "pytorch_model.bin",
        path / "model.safetensors",
    ]:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot resolve checkpoint: {path}")


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["module", "model", "state_dict", "model_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                if sd and next(iter(sd.keys())).startswith("module."):
                    sd = {k[len("module."):]: v for k, v in sd.items()}
                return sd
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format")


def load_cfg_with_env(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f) if path.endswith(".json") else __import__("yaml").safe_load(f)

    def _e(o):
        if isinstance(o, dict):
            return {k: _e(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_e(v) for v in o]
        return os.path.expandvars(o) if isinstance(o, str) else o

    return _e(cfg)


def load_v3_model(v3_root, cfg, checkpoint, device):
    old = list(sys.path)
    try:
        sys.path.insert(0, v3_root)
        build_fn = importlib.import_module("models").build_cross_view_localizer_v2
    finally:
        sys.path[:] = old
    mc, dc = cfg["model"], cfg["data"]
    model = build_fn(
        pretrained_pi3=None,
        freeze_backbone=False,
        freeze_prompt_encoder=False,
        load_camera_head_weights=False,
        sam_weights=None,
        img_size=dc.get("img_size", 518),
        patch_size=mc.get("patch_size", 14),
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
        backbone_type=mc.get("backbone_type", "pi3"),
        encoder_name=mc.get("encoder_name", "vit_b16"),
        encoder_pretrained=False,
        encoder_weights=mc.get("encoder_weights", "LVD142M"),
        joint_vit_variant=mc.get("joint_vit_variant"),
        joint_vit_weights=mc.get("joint_vit_weights"),
        adapter_dim=mc.get("adapter_dim", 1024),
        adapter_depth=mc.get("adapter_depth", 36),
        adapter_num_heads=mc.get("adapter_num_heads", 16),
        use_frame_pos_embed=mc.get("use_frame_pos_embed", False),
    )
    ckpt = resolve_checkpoint(Path(checkpoint).resolve())
    sd = extract_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_cvos_model(root, ckpt, device):
    cls = import_model_class(root, "model.TROGeo", "TROGeo", ["model", "utils"])
    model = torch.nn.DataParallel(cls())
    obj = torch.load(ckpt, map_location="cpu")
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    ms = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in ms}
    if len(sd) == 0:
        raise RuntimeError("No matching keys for CVOS checkpoint")
    ms.update(sd)
    model.load_state_dict(ms)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_detgeo_model(root, ckpt, device):
    cls = import_model_class(root, "model.DetGeo", "DetGeo", ["model", "utils"])
    model = torch.nn.DataParallel(cls())
    obj = torch.load(ckpt, map_location="cpu")
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    ms = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in ms}
    if len(sd) == 0:
        raise RuntimeError("No matching keys for DetGeo checkpoint")
    ms.update(sd)
    model.load_state_dict(ms)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_sam(cvos_root, ckpt, model_type, device):
    old = list(sys.path)
    try:
        sys.path.insert(0, cvos_root)
        sm = importlib.import_module("segment_anything")
    finally:
        sys.path[:] = old
    sam = sm.sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device).eval()
    for p in sam.parameters():
        p.requires_grad = False
    return sam


# ──────────────────────── Anchor bbox & SAM SPS ────────────────────────


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
        raise RuntimeError(f"Unexpected raw_anchor ndim={raw_anchor.ndim}")

    conf = raw_anchor[:, :, 4, :, :]
    anchors = np.array(
        [44, 41, 85, 85, 143, 130, 266, 153, 182, 235, 187, 444, 467, 194, 321, 299, 440, 433],
        dtype=np.float32,
    ).reshape(-1, 2)[::-1].copy()
    at = torch.tensor(anchors, dtype=torch.float32, device=device)
    stride = size / float(gh)
    sa = at / stride
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
    ref_sam = F.interpolate(ref_img, size=(1024, 1024), mode="bilinear", align_corners=False)
    ref_emb = sam.image_encoder(ref_sam)
    bb_sam = bb * (1024.0 / size)
    cx = (bb_sam[:, 0] + bb_sam[:, 2]) / 2.0
    cy = (bb_sam[:, 1] + bb_sam[:, 3]) / 2.0
    pts = torch.stack([cx, cy], dim=-1).unsqueeze(1)
    lbls = torch.ones(1, 1, device=device)
    sparse, dense = sam.prompt_encoder(points=(pts, lbls), boxes=bb_sam, masks=None)
    lo, _ = sam.mask_decoder(
        image_embeddings=ref_emb,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    sm = F.interpolate(lo, size=(size, size), mode="bilinear", align_corners=False)
    return (sm.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8)


# ──────────────────────── Visualization helpers ────────────────────────


def blend_mask(rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[m] = out[m] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_gt_pred(base: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """GT=red, pred=green, overlap=blue."""
    g, p = gt.astype(bool), pred.astype(bool)
    overlap = g & p
    gt_only = g & ~p
    pred_only = p & ~g
    out = base.copy().astype(np.float32)
    out[gt_only] = out[gt_only] * 0.45 + np.array([255, 0, 0], dtype=np.float32) * 0.55
    out[pred_only] = out[pred_only] * 0.45 + np.array([0, 255, 0], dtype=np.float32) * 0.55
    out[overlap] = out[overlap] * 0.35 + np.array([0, 0, 255], dtype=np.float32) * 0.65
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_pred_mask_like_visualize(
    base_rgb: np.ndarray,
    pred_mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.55,
    threshold: float = 0.5,
) -> np.ndarray:
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    out = base_rgb.copy().astype(np.float32)
    m = pred_bin.astype(bool)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[m] = out[m] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_point(rgb: np.ndarray, pt: np.ndarray, radius: int = 6) -> np.ndarray:
    out = rgb.copy()
    x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
    cv2.circle(out, (x, y), radius, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    return out


def combine_row(images: List[np.ndarray], panel_size: int) -> np.ndarray:
    return np.concatenate(
        [cv2.resize(im, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR) for im in images],
        axis=1,
    )


# ──────────────────────── CVOGL-Seg data wrapper ────────────────────────


def load_cvoglseg_items(
    cvogl_root: str,
    cvoglseg_root: str,
    split_name: str,
    img_size: int,
    subset: str,
) -> List[Dict[str, Any]]:
    """
    Load all samples for a given subset using CVOGLSegOnlineDataset
    and return a list of dicts with pre-processed images / annotations.
    """
    ds = CVOGLSegOnlineDataset(
        cvogl_root=cvogl_root,
        cvoglseg_root=cvoglseg_root,
        split_name=split_name,
        img_size=img_size,
    )
    items: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        rec = ds.records[i]
        if rec.dataset_name != subset:
            continue
        sample = ds[i]
        items.append({
            "query_img": sample["query_img"],        # uint8, (img_size, img_size, 3)
            "sat_img": sample["sat_img"],            # uint8, (img_size, img_size, 3)
            "point_xy": sample["point_xy"],          # float32, (2,)
            "gt_bbox_xyxy": sample["gt_bbox_xyxy"],  # float32, (4,)
            "gt_mask": sample["gt_mask"],            # uint8, (img_size, img_size)
            "dataset_name": sample["dataset_name"],
            "class_name": sample["class_name"],
            "query_path": sample["query_path"],
            "sat_path": sample["sat_path"],
            "global_index": i,
        })
    return items


def resize_sample_to(query_img: np.ndarray, sat_img: np.ndarray,
                     point_xy: np.ndarray, gt_mask: np.ndarray,
                     src_size: int, dst_size: int):
    """Resize a sample from src_size to dst_size, scaling point and mask."""
    if src_size == dst_size:
        return query_img.copy(), sat_img.copy(), point_xy.copy(), gt_mask.copy()
    s = dst_size / src_size
    q = cv2.resize(query_img, (dst_size, dst_size), interpolation=cv2.INTER_LINEAR)
    r = cv2.resize(sat_img, (dst_size, dst_size), interpolation=cv2.INTER_LINEAR)
    pt = (point_xy * s).astype(np.float32)
    m = cv2.resize(gt_mask, (dst_size, dst_size), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.uint8)
    return q, r, pt, m


# ──────────────────────── V3 batch dataset ────────────────────────


class V3CVOGLSegDataset(Dataset):
    """Wraps pre-loaded items for batched V3 inference."""

    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return {
            "front": to_tensor(Image.fromarray(it["query_img"])),
            "sat": to_tensor(Image.fromarray(it["sat_img"])),
            "point": torch.from_numpy(it["point_xy"].copy()),
            "gt_mask": torch.from_numpy(it["gt_mask"].copy()),
            "index": i,
        }


def collate_v3(batch):
    return {
        "front": torch.stack([x["front"] for x in batch]),
        "sat": torch.stack([x["sat"] for x in batch]),
        "point": torch.stack([x["point"] for x in batch]),
        "gt_mask": torch.stack([x["gt_mask"] for x in batch]),
        "index": [x["index"] for x in batch],
    }


# ──────────────────────── Inference functions ────────────────────────


@torch.no_grad()
def run_v3_batch(model, sam, items, device, batch_size, num_workers):
    """Run V3 on all items, return {local_idx: v3_miou}."""
    ds = V3CVOGLSegDataset(items)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_v3)
    out: Dict[int, float] = {}
    for batch in tqdm(loader, desc="V3 batch"):
        front = batch["front"].to(device, non_blocking=True)
        sat = batch["sat"].to(device, non_blocking=True)
        point = batch["point"].to(device, non_blocking=True)
        B = front.shape[0]
        outputs = model(
            front_view=front,
            satellite_view=sat,
            points=(point.unsqueeze(1), torch.ones(B, 1, device=device)),
            boxes=None, masks=None, mono_mask=None, sat_mask=None,
        )
        pred_boxes = outputs["pred_boxes"]
        bbox_scores = outputs.get("bbox_scores", None)
        gt = batch["gt_mask"].numpy()
        for j, idx in enumerate(batch["index"]):
            if pred_boxes.shape[1] > 1 and bbox_scores is not None:
                q_idx = int(bbox_scores[j].argmax().item())
            else:
                q_idx = 0
            size = sat.shape[-1]
            pb = sanitize_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_boxes[j, q_idx], size), size)
            bb = torch.from_numpy(pb).to(device=device, dtype=torch.float32).unsqueeze(0)
            pred_bin = sam_sps_from_bbox(sam=sam, ref_img=sat[j:j + 1], bb=bb, size=size, device=device)
            out[int(idx)] = float(iou_np(pred_bin, gt[j]))
    return out


@torch.no_grad()
def infer_v3_one(model, sam, query_img, sat_img, point_xy, gt_mask, device):
    """Single-sample V3 inference. Returns (pred_bin, miou)."""
    front = to_tensor(Image.fromarray(query_img)).unsqueeze(0).to(device)
    sat = to_tensor(Image.fromarray(sat_img)).unsqueeze(0).to(device)
    pt = torch.from_numpy(point_xy).unsqueeze(0).to(device)
    outputs = model(
        front_view=front, satellite_view=sat,
        points=(pt.unsqueeze(1), torch.ones(1, 1, device=device)),
        boxes=None, masks=None, mono_mask=None, sat_mask=None,
    )
    pred_boxes = outputs["pred_boxes"]
    bbox_scores = outputs.get("bbox_scores", None)
    if pred_boxes.shape[1] > 1 and bbox_scores is not None:
        q_idx = int(bbox_scores[0].argmax().item())
    else:
        q_idx = 0
    size = sat.shape[-1]
    pb = sanitize_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_boxes[0, q_idx], size), size)
    bb = torch.from_numpy(pb).to(device=device, dtype=torch.float32).unsqueeze(0)
    pred_bin = sam_sps_from_bbox(sam=sam, ref_img=sat, bb=bb, size=size, device=device)
    miou = iou_np(pred_bin, gt_mask)
    return pred_bin, miou


@torch.no_grad()
def infer_baseline_sps_one(model, sam, query_img, sat_img, point_xy, gt_mask, img_size, device):
    """Run CVOS/DetGeo + SAM SPS on one sample. Returns (pred_bin, miou)."""
    transform = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    q = transform(query_img).unsqueeze(0).to(device)
    r = transform(sat_img).unsqueeze(0).to(device)
    mat = make_click_heatmap(float(point_xy[1]), float(point_xy[0]), img_size)
    click_t = torch.from_numpy(mat).unsqueeze(0).to(device)

    raw_anchor, _ = model(q, r, click_t)
    bb = decode_best_anchor_bbox(raw_anchor, size=img_size, device=device)
    pred_bin = sam_sps_from_bbox(sam=sam, ref_img=r, bb=bb, size=img_size, device=device)
    miou = iou_np(pred_bin, gt_mask)
    return pred_bin, miou


# ──────────────────────── Candidate metrics ────────────────────────


@dataclass
class CandidateMetrics:
    idx: int
    subset: str
    v3_miou: float
    cvos_sps_miou: float
    detgeo_sps_miou: float

    @property
    def gap_abs(self) -> float:
        return float(abs(self.cvos_sps_miou - self.v3_miou))

    @property
    def gap_cvos_minus_v3(self) -> float:
        return float(self.cvos_sps_miou - self.v3_miou)

    @property
    def gap_v3_minus_cvos(self) -> float:
        return float(self.v3_miou - self.cvos_sps_miou)


# ──────────────────────── Folder layout ────────────────────────


def ensure_dirs(vis_root: Path, subsets: List[str]):
    for subset in subsets:
        for name in ["QwithPoint", "gtSate", "detgeoSate", "cvosSate", "v3Sate", "combine"]:
            (vis_root / subset / name).mkdir(parents=True, exist_ok=True)


# ──────────────────────── CLI ────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative comparison on CVOGL-Seg (V3 vs baselines)")

    p.add_argument("--cvogl_root", type=str,
                    default="/data/home/scxi704/run/baseline/CVOS-Code/dataset/CVOGL")
    p.add_argument("--cvoglseg_root", type=str,
                    default="/data/home/scxi704/run/baseline/CVOS-Code/dataset/CVOGL-Seg")
    p.add_argument("--split", type=str, default="test", help="CVOGL split name")
    p.add_argument("--subsets", nargs="+", default=["CVOGL_SVI", "CVOGL_DroneAerial"])

    p.add_argument("--v3_root", type=str, default="/data/home/scxi704/run/xhj/location_v4")
    p.add_argument("--v3_config", type=str,
                    default="/data/home/scxi704/run/xhj/location_v4/output_v3/ablation_4_all_on/config.yaml")
    p.add_argument("--v3_checkpoint", type=str,
                    default="/data/home/scxi704/run/xhj/location_v4/output_v3/ablation_4_all_on/best")

    p.add_argument("--cvos_root", type=str, default="/data/home/scxi704/run/baseline/CVOS-Code")
    p.add_argument("--cvos_checkpoint", type=str, required=True)
    p.add_argument("--detgeo_root", type=str, default="/data/home/scxi704/run/baseline/DetGeo")
    p.add_argument("--detgeo_checkpoint", type=str, required=True)

    p.add_argument("--sam_checkpoint", type=str,
                    default="/data/home/scxi704/run/baseline/CVOS-Code/segment_anything/weights/sam_vit_h_4b8939.pth")
    p.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])

    p.add_argument("--top_k_candidates", type=int, default=200)
    p.add_argument("--vis_k", type=int, default=20)
    p.add_argument("--gap_mode", type=str, default="abs", choices=["abs", "cvos_minus_v3", "v3_minus_cvos"])

    p.add_argument("--v3_img_size", type=int, default=0, help="0 = read from config")
    p.add_argument("--baseline_img_size", type=int, default=512)
    p.add_argument("--v3_batch_size", type=int, default=8)
    p.add_argument("--v3_num_workers", type=int, default=8)
    p.add_argument("--panel_size", type=int, default=512)
    p.add_argument("--output_root", type=str,
                    default="/data/home/scxi704/run/xhj/location_v4/vis_cvoglseg")
    p.add_argument("--gt_mask_color", type=str, default="255,255,255")
    p.add_argument("--gt_mask_alpha", type=float, default=1.0)
    p.add_argument("--pred_mask_color", type=str, default="255,255,255")
    p.add_argument("--pred_mask_alpha", type=float, default=1.0)
    p.add_argument("--v3_mask_color", type=str, default="")
    p.add_argument("--v3_mask_alpha", type=float, default=-1.0)
    p.add_argument("--cvos_mask_color", type=str, default="")
    p.add_argument("--cvos_mask_alpha", type=float, default=-1.0)
    p.add_argument("--detgeo_mask_color", type=str, default="")
    p.add_argument("--detgeo_mask_alpha", type=float, default=-1.0)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--save_summary_json", type=str, default="")
    return p.parse_args()


# ──────────────────────── Main ────────────────────────


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
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    # V3 config & img size
    cfg = load_cfg_with_env(args.v3_config)
    v3_img_size = args.v3_img_size if args.v3_img_size > 0 else int(cfg["data"].get("img_size", 518))
    bl_img_size = args.baseline_img_size

    # ──── Load models ────
    print("[1/5] Loading models ...")
    v3_model = load_v3_model(args.v3_root, cfg, args.v3_checkpoint, device)
    cvos_model = load_cvos_model(args.cvos_root, args.cvos_checkpoint, device)
    detgeo_model = load_detgeo_model(args.detgeo_root, args.detgeo_checkpoint, device)
    sam_model = load_sam(args.cvos_root, args.sam_checkpoint, args.sam_model_type, device)

    vis_root = Path(args.output_root)
    ensure_dirs(vis_root, args.subsets)

    summary: Dict[str, Any] = {"args": vars(args), "selected": {}}

    for subset in args.subsets:
        print(f"\n{'='*72}\n  Subset: {subset}\n{'='*72}")

        # ──── Load data ────
        print(f"[2/5] Loading CVOGL-Seg data ({subset}, split={args.split}) ...")
        items = load_cvoglseg_items(
            cvogl_root=args.cvogl_root,
            cvoglseg_root=args.cvoglseg_root,
            split_name=args.split,
            img_size=v3_img_size,
            subset=subset,
        )
        print(f"Loaded samples: {len(items)}")
        if not items:
            print(f"[warn] No samples for {subset}, skip.")
            continue

        # ──── V3 batch pass ────
        print(f"[3/5] Running V3 on all {len(items)} samples ({subset}) ...")
        v3_mious = run_v3_batch(v3_model, sam_model, items, device, args.v3_batch_size, args.v3_num_workers)

        # Sort by V3 mIoU, take top-K
        sorted_idx = sorted(v3_mious.keys(), key=lambda x: v3_mious[x], reverse=True)
        candidates = sorted_idx[: min(len(sorted_idx), args.top_k_candidates)]
        print(f"Top-{args.top_k_candidates} candidates: {len(candidates)}")

        # ──── Run baselines on candidates ────
        print(f"[4/5] Running CVOS+SPS and DetGeo+SPS on {len(candidates)} candidates ({subset}) ...")
        all_cm: List[CandidateMetrics] = []
        for lidx in tqdm(candidates, desc=f"Baselines ({subset})"):
            it = items[lidx]
            # Resize to baseline size
            q_bl, s_bl, pt_bl, m_bl = resize_sample_to(
                it["query_img"], it["sat_img"], it["point_xy"], it["gt_mask"],
                src_size=v3_img_size, dst_size=bl_img_size,
            )
            _, cvos_miou = infer_baseline_sps_one(
                cvos_model, sam_model, q_bl, s_bl, pt_bl, m_bl, bl_img_size, device,
            )
            _, detgeo_miou = infer_baseline_sps_one(
                detgeo_model, sam_model, q_bl, s_bl, pt_bl, m_bl, bl_img_size, device,
            )
            all_cm.append(CandidateMetrics(
                idx=lidx, subset=subset,
                v3_miou=v3_mious[lidx],
                cvos_sps_miou=cvos_miou,
                detgeo_sps_miou=detgeo_miou,
            ))

        # Sort by gap, pick top-N
        def _gap(m: CandidateMetrics) -> float:
            if args.gap_mode == "abs":
                return m.gap_abs
            if args.gap_mode == "cvos_minus_v3":
                return m.gap_cvos_minus_v3
            return m.gap_v3_minus_cvos

        selected = sorted(all_cm, key=_gap, reverse=True)[: min(len(all_cm), args.vis_k)]
        print(f"Selected for visualization: {len(selected)}")

        # ──── Visualize ────
        print(f"[5/5] Saving visualizations ({subset}) ...")
        summary["selected"][subset] = []

        for rank, m in enumerate(tqdm(selected, desc=f"Vis ({subset})"), start=1):
            it = items[m.idx]

            # V3 inference at V3 resolution
            v3_pred, v3_miou = infer_v3_one(
                v3_model, sam_model, it["query_img"], it["sat_img"], it["point_xy"], it["gt_mask"], device,
            )

            # Baseline inference at baseline resolution
            q_bl, s_bl, pt_bl, m_bl = resize_sample_to(
                it["query_img"], it["sat_img"], it["point_xy"], it["gt_mask"],
                src_size=v3_img_size, dst_size=bl_img_size,
            )
            cvos_pred, cvos_miou = infer_baseline_sps_one(
                cvos_model, sam_model, q_bl, s_bl, pt_bl, m_bl, bl_img_size, device,
            )
            detgeo_pred, detgeo_miou = infer_baseline_sps_one(
                detgeo_model, sam_model, q_bl, s_bl, pt_bl, m_bl, bl_img_size, device,
            )

            # Panel 1: query (mono/drone) with point
            point_img = draw_point(it["query_img"], it["point_xy"])

            sat_gt = blend_mask(it["sat_img"], it["gt_mask"], gt_mask_color, alpha=args.gt_mask_alpha)

            # Panel 4: DetGeo+SPS overlay (baseline resolution)
            detgeo_vis = blend_mask(s_bl, detgeo_pred, detgeo_mask_color, alpha=detgeo_mask_alpha)

            # Panel 5: CVOS+SPS overlay (baseline resolution)
            cvos_vis = blend_mask(s_bl, cvos_pred, cvos_mask_color, alpha=cvos_mask_alpha)

            # Panel 6: V3 overlay (V3 resolution)
            v3_vis = overlay_pred_mask_like_visualize(
                it["sat_img"], v3_pred, v3_mask_color, alpha=v3_mask_alpha, threshold=0.5
            )

            # Save individual panels
            key = f"rank{rank:03d}_idx{m.idx:05d}_gap{_gap(m):.4f}"

            panels = {
                "QwithPoint": point_img,
                "gtSate": sat_gt,
                "detgeoSate": detgeo_vis,
                "cvosSate": cvos_vis,
                "v3Sate": v3_vis,
            }
            for folder, img in panels.items():
                p = vis_root / subset / folder / f"{key}.png"
                Image.fromarray(img).save(p)

            # Combined 6-panel
            combined = combine_row(
                [point_img, detgeo_vis, cvos_vis, v3_vis, sat_gt],
                panel_size=args.panel_size,
            )
            Image.fromarray(combined).save(vis_root / subset / "combine" / f"{key}.png")

            summary["selected"][subset].append({
                "rank": rank,
                "local_idx": m.idx,
                "global_idx": it["global_index"],
                "subset": subset,
                "class_name": it["class_name"],
                "query_path": it["query_path"],
                "sat_path": it["sat_path"],
                "v3_miou": float(v3_miou),
                "cvos_sps_miou": float(cvos_miou),
                "detgeo_sps_miou": float(detgeo_miou),
                "gap_abs": float(abs(cvos_miou - v3_miou)),
                "gap_cvos_minus_v3": float(cvos_miou - v3_miou),
            })

    # ──── Save summary ────
    summary_path = Path(args.save_summary_json) if args.save_summary_json else (vis_root / "selection_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone.")
    print(f"Visualization root: {vis_root}")
    print(f"Summary json: {summary_path}")


if __name__ == "__main__":
    main()
