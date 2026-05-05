#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative comparison: V3 vs CVOS+SPS vs DetGeo+SPS on University-Release dataset.

Tasks:
  G2D — ground image -> drone image (point prompt on ground)
  G2S — ground image -> satellite image (point prompt on ground)

Panel layout (6-panel combined):
  [query+point | target_raw | target+GT_mask | DetGeo+SPS | CVOS+SPS | V3]

Workflow:
  1) Run V3 on all triplets (point prompt) -> compute mask mIoU per sample.
  2) Select top-K candidates per task (sorted by V3 mIoU).
  3) Run CVOS+SPS / DetGeo+SPS on selected candidates.
  4) Pick top-N by gap metric.
  5) Save 6-panel visualizations and summary JSON.
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
import pycocotools.mask as mask_utils
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


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


def decode_rle_mask(seg: Any, H: int, W: int) -> np.ndarray:
    """Decode RLE segmentation dict -> binary mask."""
    if not isinstance(seg, dict) or "counts" not in seg:
        return np.zeros((H, W), dtype=np.uint8)
    rle = dict(seg)
    if isinstance(rle["counts"], list):
        rle = mask_utils.frPyObjects(rle, H, W)
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


def draw_bbox_xyxy(rgb: np.ndarray, bb: np.ndarray, color=(0, 255, 0), thickness: int = 3) -> np.ndarray:
    out = rgb.copy()
    x1, y1, x2, y2 = bb.astype(np.int32)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def combine_row(images: List[np.ndarray], panel_size: int) -> np.ndarray:
    return np.concatenate(
        [cv2.resize(im, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR) for im in images],
        axis=1,
    )


# ──────────────────────── Data preparation ────────────────────────


TASK_CFG = {
    "G2D": {
        "target_key": "drone_image",
        "bbox_key": "drone_image_bbox",
        "seg_key": "drone_segmentation",
        "label": "Drone",
    },
    "G2S": {
        "target_key": "satellite_image",
        "bbox_key": "satellite_image_bbox",
        "seg_key": "satellite_segmentation",
        "label": "Sat",
    },
}


def filter_valid_triplets(triplets: List[Dict], task: str) -> List[int]:
    """Return indices of triplets valid for a given task."""
    tc = TASK_CFG[task]
    valid = []
    for i, t in enumerate(triplets):
        if not isinstance(t, dict):
            continue
        if not t.get("ground_image") or not t.get(tc["target_key"]):
            continue
        gp = t.get("ground_image_point")
        if not isinstance(gp, dict) or "x" not in gp or "y" not in gp:
            continue
        bb = t.get(tc["bbox_key"])
        if not isinstance(bb, (list, tuple)) or len(bb) < 4:
            continue
        valid.append(i)
    return valid


def prepare_sample(
    triplet: Dict[str, Any],
    root_dir: str,
    task: str,
    size: int,
) -> Dict[str, Any]:
    """Load images, resize to *size*, and return all fields needed for inference & vis."""
    tc = TASK_CFG[task]

    ground = read_rgb(os.path.join(root_dir, triplet["ground_image"]))
    target = read_rgb(os.path.join(root_dir, triplet[tc["target_key"]]))

    Hg, Wg = ground.shape[:2]
    Ht, Wt = target.shape[:2]

    point = np.array(
        [float(triplet["ground_image_point"]["x"]), float(triplet["ground_image_point"]["y"])],
        dtype=np.float32,
    )
    gt_bbox_xywh = np.array(triplet[tc["bbox_key"]][:4], dtype=np.float32)

    seg = triplet.get(tc["seg_key"])
    gt_mask = decode_rle_mask(seg, Ht, Wt) if seg else np.zeros((Ht, Wt), dtype=np.uint8)

    S = size
    if (Hg, Wg) != (S, S):
        sxg, syg = S / Wg, S / Hg
        ground = cv2.resize(ground, (S, S), interpolation=cv2.INTER_LINEAR)
        point = np.array([point[0] * sxg, point[1] * syg], dtype=np.float32)

    if (Ht, Wt) != (S, S):
        sxt, syt = S / Wt, S / Ht
        target = cv2.resize(target, (S, S), interpolation=cv2.INTER_LINEAR)
        gt_bbox_xywh = np.array(
            [gt_bbox_xywh[0] * sxt, gt_bbox_xywh[1] * syt, gt_bbox_xywh[2] * sxt, gt_bbox_xywh[3] * syt],
            dtype=np.float32,
        )
        gt_mask = cv2.resize(gt_mask, (S, S), interpolation=cv2.INTER_NEAREST)

    gt_mask = (gt_mask > 0).astype(np.uint8)
    gt_bbox_xyxy = bbox_xywh_to_xyxy(gt_bbox_xywh)

    return {
        "ground_rgb": ground,
        "target_rgb": target,
        "point_xy": point,
        "gt_bbox_xywh": gt_bbox_xywh,
        "gt_bbox_xyxy": gt_bbox_xyxy,
        "gt_mask": gt_mask,
    }


# ──────────────────────── V3 batch dataset ────────────────────────


class V3BatchDataset(Dataset):
    """Wraps triplets for batched V3 point-prompt inference."""

    def __init__(self, triplets: List[Dict], indices: List[int], root_dir: str, task: str, img_size: int):
        self.triplets = triplets
        self.indices = indices
        self.root_dir = root_dir
        self.task = task
        self.img_size = img_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        prep = prepare_sample(self.triplets[idx], self.root_dir, self.task, self.img_size)
        return {
            "front": to_tensor(Image.fromarray(prep["ground_rgb"])),
            "sat": to_tensor(Image.fromarray(prep["target_rgb"])),
            "point": torch.from_numpy(prep["point_xy"]),
            "gt_mask": torch.from_numpy(prep["gt_mask"]),
            "index": idx,
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
def run_v3_batch(model, sam, triplets, indices, root_dir, task, img_size, device, batch_size, num_workers):
    """Run V3 on all valid samples, return {idx: {"v3_miou": float}}."""
    ds = V3BatchDataset(triplets, indices, root_dir, task, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, collate_fn=collate_v3)
    out: Dict[int, Dict[str, Any]] = {}
    for batch in tqdm(loader, desc=f"V3 batch ({task})"):
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
            pb = sanitize_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_boxes[j, q_idx], img_size), img_size)
            bb = torch.from_numpy(pb).to(device=device, dtype=torch.float32).unsqueeze(0)
            pred_bin = sam_sps_from_bbox(sam=sam, ref_img=sat[j:j + 1], bb=bb, size=img_size, device=device)
            out[int(idx)] = {"v3_miou": float(iou_np(pred_bin, gt[j]))}
    return out


@torch.no_grad()
def infer_v3_one(model, sam, triplet, root_dir, task, img_size, device):
    """Single-sample V3 inference. Returns (pred_bin, miou, prep_dict)."""
    prep = prepare_sample(triplet, root_dir, task, img_size)
    front = to_tensor(Image.fromarray(prep["ground_rgb"])).unsqueeze(0).to(device)
    sat = to_tensor(Image.fromarray(prep["target_rgb"])).unsqueeze(0).to(device)
    pt = torch.from_numpy(prep["point_xy"]).unsqueeze(0).to(device)
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
    pb = sanitize_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_boxes[0, q_idx], img_size), img_size)
    bb = torch.from_numpy(pb).to(device=device, dtype=torch.float32).unsqueeze(0)
    pred_bin = sam_sps_from_bbox(sam=sam, ref_img=sat, bb=bb, size=img_size, device=device)
    miou = iou_np(pred_bin, prep["gt_mask"])
    return pred_bin, miou, prep


@torch.no_grad()
def infer_baseline_sps_one(model, sam, triplet, root_dir, task, img_size, device):
    """Run a CVOS/DetGeo baseline + SAM SPS on one sample. Returns (pred_bin, miou)."""
    prep = prepare_sample(triplet, root_dir, task, img_size)
    transform = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    q = transform(prep["ground_rgb"]).unsqueeze(0).to(device)
    r = transform(prep["target_rgb"]).unsqueeze(0).to(device)
    click = prep["point_xy"]
    mat = make_click_heatmap(float(click[1]), float(click[0]), img_size)
    click_t = torch.from_numpy(mat).unsqueeze(0).to(device)

    raw_anchor, _ = model(q, r, click_t)
    bb = decode_best_anchor_bbox(raw_anchor, size=img_size, device=device)
    pred_bin = sam_sps_from_bbox(sam=sam, ref_img=r, bb=bb, size=img_size, device=device)
    miou = iou_np(pred_bin, prep["gt_mask"])
    return pred_bin, miou


# ──────────────────────── Candidate selection ────────────────────────


@dataclass
class CandidateMetrics:
    idx: int
    task: str
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


def ensure_dirs(vis_root: Path, tasks: List[str]):
    for task in tasks:
        tc = TASK_CFG[task]
        label = tc["label"]
        for name in [
            "GwithPoint",
            "gtSate",
            f"detgeo{label}",
            f"cvos{label}",
            f"v3{label}",
            "combine",
        ]:
            (vis_root / task / name).mkdir(parents=True, exist_ok=True)


# ──────────────────────── CLI ────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative comparison on University-Release (V3 vs baselines)")

    p.add_argument("--triplet_json", type=str,
                    default="/data/home/scxi704/run/xhj/University-Release/verified_triplets_sam2_masks.json")
    p.add_argument("--root_dir", type=str,
                    default="/data/home/scxi704/run/xhj/University-Release")

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

    p.add_argument("--tasks", nargs="+", default=["G2D", "G2S"], choices=["G2D", "G2S"])
    p.add_argument("--top_k_candidates", type=int, default=200)
    p.add_argument("--vis_k", type=int, default=20)
    p.add_argument("--gap_mode", type=str, default="abs", choices=["abs", "cvos_minus_v3", "v3_minus_cvos"])

    p.add_argument("--v3_img_size", type=int, default=0, help="0 = read from config")
    p.add_argument("--baseline_img_size", type=int, default=512)
    p.add_argument("--v3_batch_size", type=int, default=8)
    p.add_argument("--v3_num_workers", type=int, default=8)
    p.add_argument("--panel_size", type=int, default=512)
    p.add_argument("--output_root", type=str,
                    default="/data/home/scxi704/run/xhj/location_v4/vis_university")
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

    # Load triplets
    with open(args.triplet_json, "r", encoding="utf-8") as f:
        triplets: List[Dict] = json.load(f)
    print(f"Loaded triplets: {len(triplets)}")

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
    ensure_dirs(vis_root, args.tasks)

    summary: Dict[str, Any] = {"args": vars(args), "selected": {}}

    for task in args.tasks:
        tc = TASK_CFG[task]
        label = tc["label"]
        print(f"\n{'='*72}\n  Task: {task}  ({label})\n{'='*72}")

        # ──── Filter valid triplets for task ────
        valid_idx = filter_valid_triplets(triplets, task)
        print(f"Valid triplets for {task}: {len(valid_idx)}")
        if not valid_idx:
            continue

        # ──── V3 batch pass ────
        print(f"[2/5] Running V3 on all valid triplets ({task}) ...")
        v3_results = run_v3_batch(
            v3_model, sam_model, triplets, valid_idx, args.root_dir, task,
            v3_img_size, device, args.v3_batch_size, args.v3_num_workers,
        )

        # Sort by V3 mIoU, take top-K
        sorted_by_v3 = sorted(v3_results.keys(), key=lambda x: v3_results[x]["v3_miou"], reverse=True)
        candidates = sorted_by_v3[: min(len(sorted_by_v3), args.top_k_candidates)]
        print(f"Top-{args.top_k_candidates} candidates: {len(candidates)}")

        # ──── Run baselines on candidates ────
        print(f"[3/5] Running CVOS+SPS and DetGeo+SPS on {len(candidates)} candidates ({task}) ...")
        all_cm: List[CandidateMetrics] = []
        for idx in tqdm(candidates, desc=f"Baselines ({task})"):
            t = triplets[idx]
            _, cvos_miou = infer_baseline_sps_one(cvos_model, sam_model, t, args.root_dir, task, bl_img_size, device)
            _, detgeo_miou = infer_baseline_sps_one(detgeo_model, sam_model, t, args.root_dir, task, bl_img_size, device)
            all_cm.append(CandidateMetrics(
                idx=idx, task=task,
                v3_miou=v3_results[idx]["v3_miou"],
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
        print(f"[4/5] Saving visualizations ({task}) ...")
        summary["selected"][task] = []

        for rank, m in enumerate(tqdm(selected, desc=f"Vis ({task})"), start=1):
            t = triplets[m.idx]

            # V3 inference (at V3 resolution)
            v3_pred, v3_miou, prep_v3 = infer_v3_one(v3_model, sam_model, t, args.root_dir, task, v3_img_size, device)

            # Baseline inference (at baseline resolution)
            cvos_pred, cvos_miou = infer_baseline_sps_one(cvos_model, sam_model, t, args.root_dir, task, bl_img_size, device)
            detgeo_pred, detgeo_miou = infer_baseline_sps_one(detgeo_model, sam_model, t, args.root_dir, task, bl_img_size, device)

            # Prepare panels at each model's resolution
            prep_bl = prepare_sample(t, args.root_dir, task, bl_img_size)

            # Panel 1: ground with point (V3 resolution)
            point_img = draw_point(prep_v3["ground_rgb"], prep_v3["point_xy"])

            target_gt = blend_mask(prep_v3["target_rgb"], prep_v3["gt_mask"], gt_mask_color, alpha=args.gt_mask_alpha)

            # Panel 4: DetGeo+SPS overlay on target (baseline resolution)
            detgeo_vis = blend_mask(prep_bl["target_rgb"], detgeo_pred, detgeo_mask_color, alpha=detgeo_mask_alpha)

            # Panel 5: CVOS+SPS overlay on target (baseline resolution)
            cvos_vis = blend_mask(prep_bl["target_rgb"], cvos_pred, cvos_mask_color, alpha=cvos_mask_alpha)

            # Panel 6: V3 overlay on target (V3 resolution)
            v3_vis = overlay_pred_mask_like_visualize(
                prep_v3["target_rgb"], v3_pred, v3_mask_color, alpha=v3_mask_alpha, threshold=0.5
            )

            # Save individual panels
            key = f"rank{rank:03d}_idx{m.idx:05d}_gap{_gap(m):.4f}"

            paths = {
                "GwithPoint": point_img,
                "gtSate": target_gt,
                f"detgeo{label}": detgeo_vis,
                f"cvos{label}": cvos_vis,
                f"v3{label}": v3_vis,
            }
            for folder, img in paths.items():
                p = vis_root / task / folder / f"{key}.png"
                Image.fromarray(img).save(p)

            # Combined 6-panel
            combined = combine_row(
                [point_img, detgeo_vis, cvos_vis, v3_vis, target_gt],
                panel_size=args.panel_size,
            )
            Image.fromarray(combined).save(vis_root / task / "combine" / f"{key}.png")

            summary["selected"][task].append({
                "rank": rank,
                "idx": m.idx,
                "task": task,
                "ground_image": t.get("ground_image", ""),
                "target_image": t.get(tc["target_key"], ""),
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

    print(f"\n[5/5] Done.")
    print(f"Visualization root: {vis_root}")
    print(f"Summary json: {summary_path}")


if __name__ == "__main__":
    main()
