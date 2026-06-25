#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot ground->drone evaluation for Cross-View Localizer V2.

Task:
- Input: ground image + point prompt (on ground image) + drone image
- Model output: bbox (and optional mask) on drone image

Detection metrics:
- A@0.5:0.95 · ACC@0.5 · ACC@0.75

Segmentation metrics (if triplet JSON 含有无人机分割标注 `drone_segmentation`):
- 分割 (模型 mask — mask_pred): mIoU · mDice · AAE · ME

Triplet JSON format (per item):
{
  "drone_image": "train/drone/xxxx/image-01.jpeg",
  "ground_image": "train/street/xxxx/1.jpg",
  "drone_image_bbox": [x, y, w, h],                    # coco xywh
  "ground_image_point": {"x": ..., "y": ...},
  "drone_segmentation": { "size": [H, W], "counts": "..." }  # optional, COCO RLE
}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from models import build_cross_view_localizer_v2


DET_THRESHOLDS = np.arange(0.5, 1.0, 0.05, dtype=np.float32)


class SegMetrics:
    """单样本分割指标: mIoU / mDice / AAE / ME"""

    @staticmethod
    def iou(p: np.ndarray, g: np.ndarray) -> float:
        p, g = p.astype(bool), g.astype(bool)
        inter = np.logical_and(p, g).sum()
        uni = np.logical_or(p, g).sum()
        return 1.0 if uni == 0 else float(inter) / float(uni)

    @staticmethod
    def dice(p: np.ndarray, g: np.ndarray) -> float:
        p, g = p.astype(bool), g.astype(bool)
        inter = np.logical_and(p, g).sum()
        total = p.sum() + g.sum()
        return 1.0 if total == 0 else 2.0 * float(inter) / float(total)

    @staticmethod
    def aae(p: np.ndarray, g: np.ndarray) -> float:
        return abs(int(p.astype(bool).sum()) - int(g.astype(bool).sum()))

    @staticmethod
    def me(p: np.ndarray, g: np.ndarray) -> float:
        pb, gb = p.astype(bool), g.astype(bool)
        if pb.sum() == 0 or gb.sum() == 0:
            h, w = p.shape[:2]
            return float(np.sqrt(h * h + w * w))
        py, px = np.where(pb)
        gy, gx = np.where(gb)
        return float(np.sqrt((px.mean() - gx.mean()) ** 2 + (py.mean() - gy.mean()) ** 2))

    @classmethod
    def all(cls, p: np.ndarray, g: np.ndarray):
        return cls.iou(p, g), cls.dice(p, g), cls.aae(p, g), cls.me(p, g)


def get_workspace_dir() -> Path:
    root_dir = os.environ.get("ROOT_DIR", "")
    workspace_name = os.environ.get("WORKSPACE_NAME", "")
    if root_dir and workspace_name:
        return Path(root_dir) / workspace_name
    return Path(__file__).resolve().parent


def load_cfg_with_env(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f) if config_path.endswith(".json") else __import__("yaml").safe_load(f)

    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "/mnt/data/wrp/checkpoints_offline")).expanduser()

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            value = os.path.expandvars(obj)
            # Configs saved on a training machine can contain absolute
            # checkpoint paths.  Re-root checkpoint filenames to the current
            # offline checkpoint directory for portable evaluation.
            if "/checkpoints_offline/" in value:
                return str(checkpoint_dir / value.rsplit("/checkpoints_offline/", 1)[1])
            if "/GaGeo/ckpt/" in value:
                return str(checkpoint_dir / value.rsplit("/GaGeo/ckpt/", 1)[1])
            return value
        return obj

    return _expand(cfg)


def bbox_xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b.astype(np.float32)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def bbox_cxcywh_norm_to_xyxy_abs(b: torch.Tensor, img_size: int) -> np.ndarray:
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
                if len(sd) > 0:
                    first_k = next(iter(sd.keys()))
                    if first_k.startswith("module."):
                        sd = {kk[len("module."):]: vv for kk, vv in sd.items()}
                return sd
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format")


def remap_legacy_mask_head_keys(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Backward-compat key remap for old checkpoints:
      *.output_hypernetworks_mlps.0.*  ->  *.output_hypernetwork_mlp.*

    参考 evaluate_custom_v2.py:
    我们将 mask head 从 ModuleList(single token) 改成了共享 MLP，对于旧 checkpoint，
    如果不做 key 映射，mask head 会以随机初始化加载，导致分割指标异常。
    """
    remapped: Dict[str, torch.Tensor] = {}
    num_renamed = 0
    needle = ".output_hypernetworks_mlps.0."
    repl = ".output_hypernetwork_mlp."
    for k, v in state_dict.items():
        if needle in k:
            remapped[k.replace(needle, repl)] = v
            num_renamed += 1
        else:
            remapped[k] = v
    return remapped, num_renamed


def build_model_from_cfg(cfg: Dict[str, Any], device: torch.device):
    mc = cfg["model"]
    dc = cfg["data"]

    model = build_cross_view_localizer_v2(
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
        # Evaluation immediately loads the experiment checkpoint, so avoid
        # re-fetching external ImageNet/DINO weights here.
        encoder_pretrained=False,
        encoder_weights=mc.get("encoder_weights", "LVD142M"),
        joint_vit_variant=mc.get("joint_vit_variant"),
        joint_vit_weights=mc.get("joint_vit_weights"),
        adapter_dim=mc.get("adapter_dim", 1024),
        adapter_depth=mc.get("adapter_depth", 36),
        adapter_num_heads=mc.get("adapter_num_heads", 16),
        use_frame_pos_embed=mc.get("use_frame_pos_embed", False),
        use_spatial_bbox_head=mc.get("use_spatial_bbox_head", False),
    )

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class GroundDroneTripletDataset(Dataset):
    def __init__(self, triplet_json: str, root_dir: str, input_size: int = 518):
        self.root_dir = Path(root_dir)
        self.input_size = input_size

        with open(triplet_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples: List[Dict[str, Any]] = []
        for x in raw:
            if not isinstance(x, dict):
                continue
            if not x.get("drone_image") or not x.get("ground_image"):
                continue
            gp = x.get("ground_image_point", None)
            db = x.get("drone_image_bbox", None)
            if not isinstance(gp, dict) or ("x" not in gp or "y" not in gp):
                continue
            if not isinstance(db, (list, tuple)) or len(db) < 4:
                continue
            self.samples.append(x)

        if len(self.samples) == 0:
            raise ValueError("No valid triplets found in JSON")

        print(f"Loaded valid triplets: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _resolve_path(self, rel_path: str) -> Path:
        path = self.root_dir / rel_path
        if path.exists():
            return path
        fallback = str(rel_path)
        replacements = {
            "test/drone/": "test/gallery_drone/",
            "test/street/": "test/query_street/",
        }
        for old, new in replacements.items():
            if fallback.startswith(old):
                candidate = self.root_dir / fallback.replace(old, new, 1)
                if candidate.exists():
                    return candidate
        return path

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        ground_path = self._resolve_path(item["ground_image"])
        drone_path = self._resolve_path(item["drone_image"])

        ground = self._load_rgb(ground_path)
        drone = self._load_rgb(drone_path)

        Hg, Wg = ground.shape[:2]
        Hd, Wd = drone.shape[:2]

        point = np.array(
            [
                float(item["ground_image_point"]["x"]),
                float(item["ground_image_point"]["y"]),
            ],
            dtype=np.float32,
        )

        gt_bbox_xywh = np.array(item["drone_image_bbox"][:4], dtype=np.float32)

        # 可选: 无人机图分割 GT, 由 SAM2 标注脚本产生
        gt_mask = None
        if "drone_segmentation" in item:
            seg = item["drone_segmentation"]
            if isinstance(seg, dict) and "counts" in seg:
                rle = dict(seg)
                if isinstance(rle["counts"], list):
                    rle = mask_utils.frPyObjects(rle, Hd, Wd)
                mask = mask_utils.decode(rle)
                gt_mask = mask.astype(np.float32)

        # Model expects patch-multiple size (V2 uses 518, divisible by patch size 14).
        # Raw triplet images may be 512x512; we always remap image/point/bbox/mask to S.
        S = self.input_size

        if (Hg, Wg) != (S, S):
            sxg, syg = S / Wg, S / Hg
            ground = cv2.resize(ground, (S, S), interpolation=cv2.INTER_LINEAR)
            point = np.array([point[0] * sxg, point[1] * syg], dtype=np.float32)

        if (Hd, Wd) != (S, S):
            sxd, syd = S / Wd, S / Hd
            drone = cv2.resize(drone, (S, S), interpolation=cv2.INTER_LINEAR)
            gt_bbox_xywh = np.array(
                [
                    gt_bbox_xywh[0] * sxd,
                    gt_bbox_xywh[1] * syd,
                    gt_bbox_xywh[2] * sxd,
                    gt_bbox_xywh[3] * syd,
                ],
                dtype=np.float32,
            )
            if gt_mask is not None:
                gt_mask = cv2.resize(gt_mask, (S, S), interpolation=cv2.INTER_NEAREST)

        gt_bbox_xyxy = bbox_xywh_to_xyxy(gt_bbox_xywh)

        out: Dict[str, Any] = {
            "front_view": to_tensor(Image.fromarray(ground)),
            "sat_view": to_tensor(Image.fromarray(drone)),
            "mono_point": torch.from_numpy(point),
            "gt_bbox_xyxy": torch.from_numpy(gt_bbox_xyxy),
            "sample_id": idx,
            "ground_image": item["ground_image"],
            "drone_image": item["drone_image"],
        }
        if gt_mask is not None:
            out["gt_mask"] = torch.from_numpy((gt_mask > 0.5).astype(np.float32))
        return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "front_view": torch.stack([x["front_view"] for x in batch], dim=0),
        "sat_view": torch.stack([x["sat_view"] for x in batch], dim=0),
        "mono_point": torch.stack([x["mono_point"] for x in batch], dim=0),
        "gt_bbox_xyxy": torch.stack([x["gt_bbox_xyxy"] for x in batch], dim=0),
        "sample_id": [x["sample_id"] for x in batch],
        "ground_image": [x["ground_image"] for x in batch],
        "drone_image": [x["drone_image"] for x in batch],
    }
    if "gt_mask" in batch[0]:
        out["gt_mask"] = torch.stack([x["gt_mask"] for x in batch], dim=0)
    return out


@torch.no_grad()
def evaluate(model, loader: DataLoader, img_size: int, device: torch.device):
    ious: List[float] = []
    per_sample: List[Dict[str, Any]] = []
    seg_values: List[Tuple[float, float, float, float]] = []

    for batch in tqdm(loader, desc="Evaluating ground->drone zero-shot"):
        front = batch["front_view"].to(device, non_blocking=True)
        sat = batch["sat_view"].to(device, non_blocking=True)
        mono_point = batch["mono_point"].to(device, non_blocking=True)

        B = front.shape[0]
        point_coords = mono_point.unsqueeze(1)  # [B,1,2]
        point_labels = torch.ones(B, 1, device=device)

        outputs = model(
            front_view=front,
            satellite_view=sat,
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
            mono_mask=None,
            sat_mask=None,
        )

        pred_boxes_all = outputs["pred_boxes"]  # [B, Q, 4]
        pred_masks_all = outputs.get("mask_pred", None)
        has_seg = (pred_masks_all is not None) and ("gt_mask" in batch)

        if pred_boxes_all.shape[1] > 1 and "bbox_scores" in outputs:
            bbox_scores = outputs["bbox_scores"]  # [B, Q]
            best_idx = bbox_scores.argmax(dim=1)  # [B]
            pred_bbox_norm = pred_boxes_all[torch.arange(B), best_idx]
            if has_seg:
                pred_masks = pred_masks_all[torch.arange(B), best_idx].detach().cpu().numpy()
        else:
            pred_bbox_norm = pred_boxes_all[:, 0]
            if has_seg and pred_masks_all is not None:
                pred_masks = pred_masks_all[:, 0].detach().cpu().numpy()

        gt_bbox_xyxy = batch["gt_bbox_xyxy"].numpy().astype(np.float32)

        for i in range(B):
            pb = clip_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_bbox_norm[i], img_size), img_size)
            gb = clip_bbox_xyxy(gt_bbox_xyxy[i], img_size)
            iou = bbox_iou_np(pb, gb)
            ious.append(iou)

            rec: Dict[str, Any] = {
                "sample_id": int(batch["sample_id"][i]),
                "ground_image": batch["ground_image"][i],
                "drone_image": batch["drone_image"][i],
                "iou": float(iou),
                "pred_bbox_xyxy": [float(x) for x in pb.tolist()],
                "gt_bbox_xyxy": [float(x) for x in gb.tolist()],
            }

            if has_seg:
                gm = batch["gt_mask"][i].numpy()
                pm = pred_masks[i]
                if pm.shape != gm.shape:
                    pm = cv2.resize(pm, (gm.shape[1], gm.shape[0]), interpolation=cv2.INTER_LINEAR)
                pm_bin = (pm > 0.5).astype(np.uint8)
                gm_bin = (gm > 0.5).astype(np.uint8)
                mv = SegMetrics.all(pm_bin, gm_bin)
                seg_values.append(mv)
                rec["seg_model_miou"] = float(mv[0])
                rec["seg_model_mdice"] = float(mv[1])
                rec["seg_model_aae"] = float(mv[2])
                rec["seg_model_me"] = float(mv[3])

            per_sample.append(rec)

    avg_acc_50_95 = float(
        np.mean([np.mean([1.0 if x >= t else 0.0 for t in DET_THRESHOLDS]) for x in ious])
    ) if ious else 0.0
    acc50 = float(np.mean([1.0 if x >= 0.50 else 0.0 for x in ious])) if ious else 0.0
    acc75 = float(np.mean([1.0 if x >= 0.75 else 0.0 for x in ious])) if ious else 0.0

    metrics: Dict[str, Any] = {
        "count": len(ious),
        "avg_acc_50_95": avg_acc_50_95,
        "acc50": acc50,
        "acc75": acc75,
    }

    if seg_values:
        seg_arr = np.asarray(seg_values, dtype=np.float32)
        metrics.update(
            {
                "seg_model_miou": float(seg_arr[:, 0].mean()),
                "seg_model_mdice": float(seg_arr[:, 1].mean()),
                "seg_model_aae": float(seg_arr[:, 2].mean()),
                "seg_model_me": float(seg_arr[:, 3].mean()),
            }
        )

    return metrics, per_sample


def parse_args():
    ws_dir = get_workspace_dir()
    root_dir = os.environ.get("ROOT_DIR", "/mnt/data/wrp")
    output_dir = ws_dir / "output_v3"
    p = argparse.ArgumentParser(description="Zero-shot ground->drone evaluation (point prompt only)")
    p.add_argument("--triplet_json", type=str, default=str(Path(root_dir) / "University-Release" / "verified_triplets.json"))
    p.add_argument("--root_dir", type=str, default=str(Path(root_dir) / "University-Release"))
    p.add_argument("--config", type=str, default=str(output_dir / "config.yaml"))
    p.add_argument("--checkpoint", type=str, default=str(output_dir / "best"))
    p.add_argument(
        "--img_size",
        type=int,
        default=518,
        help="Model input size (raw 512x512 will be resized to this size; point and bbox are scaled accordingly)",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    ws_dir = get_workspace_dir()
    os.environ.setdefault("ROOT_DIR", str(ws_dir.parent))
    os.environ.setdefault("WORKSPACE_NAME", ws_dir.name)

    preset_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not preset_cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        preset_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    else:
        print(f"Using preset CUDA_VISIBLE_DEVICES={preset_cvd}; ignore --gpu={args.gpu}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA_VISIBLE_DEVICES={preset_cvd}")

    cfg = load_cfg_with_env(args.config)

    # override input size by CLI for zero-shot test setup
    cfg.setdefault("data", {})["img_size"] = int(args.img_size)

    print("\n[1/3] Loading model ...")
    model = build_model_from_cfg(cfg, device)

    ckpt_file = resolve_checkpoint(Path(args.checkpoint).resolve())
    print(f"Checkpoint file: {ckpt_file}")
    obj = torch.load(str(ckpt_file), map_location="cpu")
    sd = extract_state_dict(obj)
    # 适配旧的 mask head key（与 evaluate_custom_v2 保持一致）
    sd, renamed = remap_legacy_mask_head_keys(sd)
    if renamed > 0:
        print(f"Applied legacy mask-head key remap: {renamed} tensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded state_dict keys: {len(sd)}")
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    print("\n[2/3] Building dataset ...")
    ds = GroundDroneTripletDataset(
        triplet_json=args.triplet_json,
        root_dir=args.root_dir,
        input_size=int(args.img_size),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print("\n[3/3] Evaluating ...")
    metrics, per_sample = evaluate(model, loader, int(args.img_size), device)

    print("\n" + "=" * 72)
    print("Zero-shot Ground->Drone (point prompt) Metrics")
    print("=" * 72)
    print(f"Count    : {metrics['count']}")
    print(f"A@0.5:0.95: {metrics['avg_acc_50_95']:.4f}")
    print(f"ACC@0.5  : {metrics['acc50']:.4f}")
    print(f"ACC@0.75 : {metrics['acc75']:.4f}")
    if "seg_model_miou" in metrics:
        print("-" * 72)
        print("Segmentation (model mask — mask_pred vs drone_segmentation)")
        print(
            f"mIoU={metrics['seg_model_miou']:.4f}, "
            f"mDice={metrics['seg_model_mdice']:.4f}, "
            f"AAE={metrics['seg_model_aae']:.2f}, "
            f"ME={metrics['seg_model_me']:.2f}"
        )
    print("=" * 72)

    if args.save_json:
        out = {
            "metrics": metrics,
            "setting": {
                "triplet_json": args.triplet_json,
                "root_dir": args.root_dir,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "img_size": int(args.img_size),
                "task": "ground_to_drone_zero_shot",
                "prompt": "point",
                "bbox_format": "xywh(coco) -> converted to xyxy for IoU",
            },
            "per_sample": per_sample,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()
