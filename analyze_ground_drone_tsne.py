#!/usr/bin/env python3
"""Plot paper-quality satellite-anchor t-SNE for ground/drone features.

The script compares two checkpoints with and without contrastive learning using
the same balanced sample set and the same t-SNE embedding. It evaluates the
contrastive-learning expectation in the satellite-anchor space:

1. paired mono-satellite positive cosine distance,
2. joint t-SNE of normalize(mono_feature) and normalize(satellite_feature).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from data.dataset import CrossViewDataset
from evaluate_zero_shot_ground_to_drone import extract_state_dict, remap_legacy_mask_head_keys
from models import build_cross_view_localizer_v2
from utils.prompt_utils import prepare_single_prompt


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_ENV = {
    "JSON_ROOT": str(WORKSPACE_ROOT / "eccv_data" / "data" / "json"),
    "DATA_ROOT": str(WORKSPACE_ROOT / "eccv_data" / "data" / "urban"),
    "CHECKPOINT_DIR": str(WORKSPACE_ROOT / "checkpoints_offline"),
    "OUTPUT_ROOT": str(REPO_ROOT / "outputs" / "tsne_default_v3"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ground/drone features in satellite-anchor space."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "default_v3.yaml"),
        help="V3 model/data config. Environment variables are expanded.",
    )
    parser.add_argument(
        "--with-cl-ckpt",
        type=str,
        default=str(REPO_ROOT / "GAGeo_ckpt" / "gageo" / "mp_rank_00_model_states.pt"),
    )
    parser.add_argument(
        "--without-cl-ckpt",
        type=str,
        default=str(REPO_ROOT / "GAGeo_ckpt" / "no_cl" / "mp_rank_00_model_states.pt"),
    )
    parser.add_argument("--json", type=str, default="", help="Dataset json; defaults to cfg data.val_json.")
    parser.add_argument("--data-root", type=str, default="", help="Image root; defaults to cfg data.data_root.")
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["urban", "triplet"],
        default="urban",
        help="urban uses CrossViewDataset; triplet uses University ground-drone-satellite triplets.",
    )
    parser.add_argument(
        "--triplet-json",
        type=str,
        default=str(WORKSPACE_ROOT / "University-Release" / "verified_triplets_sam2_masks.json"),
        help="Triplet JSON used when --input-format triplet.",
    )
    parser.add_argument(
        "--triplet-root-dir",
        type=str,
        default=str(WORKSPACE_ROOT / "University-Release"),
        help="Triplet image root used when --input-format triplet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "ground_drone_tsne_default_v3"),
    )
    parser.add_argument("--split-name", type=str, default="val")
    parser.add_argument("--samples-per-class", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["point", "bbox", "mask", "none"],
        default="point",
        help="Prompt used during feature extraction.",
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=["masked_mean", "mean", "cls", "token_sample"],
        default="masked_mean",
        help="How to pool front/satellite patch features.",
    )
    parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=8,
        help="Patch tokens sampled per image when --pool token_sample.",
    )
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--tsne-iter", type=int, default=1500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU even when CUDA is available.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Skip model forward and only redraw from existing features.npz.",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(path: str) -> Dict[str, Any]:
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def expand(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    return expand(cfg)


def build_model_from_cfg(cfg: Dict[str, Any], ckpt_path: str, device: torch.device) -> torch.nn.Module:
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
        num_bbox_mask_queries=mc.get("num_bbox_mask_queries"),
        num_heatmap_queries=mc.get("num_heatmap_queries", 1),
        supervision_layers=mc.get("supervision_layers", [4, 11, 17]),
        supervision_weights=mc.get("supervision_weights", [0.1, 0.3, 0.6]),
        dropout=mc.get("dropout", 0.1),
        contrastive=mc.get("contrastive", True),
        contrastive_proj_dim=mc.get("contrastive_proj_dim", 256),
        contrastive_queue_size=mc.get("contrastive_queue_size", 16384),
        contrastive_momentum=mc.get("contrastive_momentum", 0.999),
        contrastive_temperature=mc.get("contrastive_temperature", 0.07),
        sam_embed_dim=mc.get("sam_embed_dim", 256),
        backbone_type=mc.get("backbone_type", "pi3"),
        encoder_name=mc.get("encoder_name", "vit_b16"),
        encoder_pretrained=False,
        encoder_weights=mc.get("encoder_weights", "LVD142M"),
        joint_vit_variant=mc.get("joint_vit_variant"),
        joint_vit_weights=mc.get("joint_vit_weights"),
        adapter_dim=mc.get("adapter_dim", 1024),
        adapter_depth=mc.get("adapter_depth", 36),
        adapter_num_heads=mc.get("adapter_num_heads", 16),
        mask_inject_mode=mc.get("mask_inject_mode", "global_kv"),
        use_global_attn_mask=mc.get("use_global_attn_mask", True),
        use_frame_pos_embed=mc.get("use_frame_pos_embed", False),
        use_spatial_bbox_head=mc.get("use_spatial_bbox_head", False),
    )

    ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(ckpt_obj)
    state_dict, renamed = remap_legacy_mask_head_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded {Path(ckpt_path).parent.name}: keys={len(state_dict)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, remapped={renamed}"
    )
    bad_unexpected = [k for k in unexpected if not k.startswith("contrastive_head.")]
    if bad_unexpected:
        print(f"  unexpected sample: {bad_unexpected[:8]}")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def balanced_indices(dataset: CrossViewDataset, samples_per_class: int, seed: int) -> Tuple[List[int], np.ndarray]:
    ground = []
    drone = []
    for idx, item in enumerate(dataset.data):
        target = drone if "drone" in str(item.get("mono_filename", "")).lower() else ground
        target.append(idx)

    rng = random.Random(seed)
    rng.shuffle(ground)
    rng.shuffle(drone)
    n = min(samples_per_class, len(ground), len(drone))
    selected = ground[:n] + drone[:n]
    labels = np.array(["ground"] * n + ["drone"] * n)
    order = list(range(len(selected)))
    rng.shuffle(order)
    selected = [selected[i] for i in order]
    labels = labels[order]
    print(f"Selected {n} ground + {n} drone samples from {len(dataset.data)} records.")
    return selected, labels


class UniversityTripletPairDataset(Dataset):
    """Flatten University triplets into paired ground-satellite and drone-satellite rows."""

    def __init__(self, triplet_json: str, root_dir: str, input_size: int = 518):
        self.root_dir = Path(root_dir)
        self.input_size = int(input_size)
        with open(triplet_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.rows: List[Dict[str, Any]] = []
        for triplet_id, item in enumerate(raw):
            if not all(item.get(k) for k in ["ground_image", "drone_image", "satellite_image"]):
                continue
            if "ground_image_point" not in item or "satellite_image_bbox" not in item:
                continue
            self.rows.append({"triplet_id": triplet_id, "view": "ground", "item": item})
            self.rows.append({"triplet_id": triplet_id, "view": "drone", "item": item})
        if not self.rows:
            raise ValueError(f"No valid triplet rows found in {triplet_json}")
        print(f"Loaded triplet rows: {len(self.rows)} ({len(self.rows) // 2} triplets)")

    def __len__(self) -> int:
        return len(self.rows)

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
            "test/satellite/": "test/gallery_satellite/",
        }
        for old, new in replacements.items():
            if fallback.startswith(old):
                candidate = self.root_dir / fallback.replace(old, new, 1)
                if candidate.exists():
                    return candidate
        return path

    @staticmethod
    def _decode_rle(seg: Optional[Dict[str, Any]], height: int, width: int) -> Optional[np.ndarray]:
        if not isinstance(seg, dict) or "counts" not in seg:
            return None
        rle = dict(seg)
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle).astype(np.float32)

    @staticmethod
    def _bbox_xywh_to_cxcywh_norm(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
        x, y, w, h = bbox.astype(np.float32)
        return np.array([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height], dtype=np.float32)

    @staticmethod
    def _point_disk_mask(point: np.ndarray, height: int, width: int, radius: int = 18) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(mask, (int(round(point[0])), int(round(point[1]))), radius, 1.0, thickness=-1)
        return mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        item = row["item"]
        view = row["view"]
        triplet_id = int(row["triplet_id"])
        size = self.input_size

        sat = self._load_rgb(self._resolve_path(item["satellite_image"]))
        hs0, ws0 = sat.shape[:2]
        sat_bbox = np.array(item["satellite_image_bbox"][:4], dtype=np.float32)
        sat_mask = self._decode_rle(item.get("satellite_segmentation"), hs0, ws0)
        if sat_mask is None:
            sat_mask = np.zeros((hs0, ws0), dtype=np.float32)
            x, y, w, h = sat_bbox.astype(np.int32)
            sat_mask[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = 1.0

        if view == "ground":
            mono_rel = item["ground_image"]
            mono = self._load_rgb(self._resolve_path(mono_rel))
            hm0, wm0 = mono.shape[:2]
            point = np.array([float(item["ground_image_point"]["x"]), float(item["ground_image_point"]["y"])], dtype=np.float32)
            mono_mask = self._point_disk_mask(point, hm0, wm0)
            box_size = max(24.0, min(hm0, wm0) * 0.08)
            mono_bbox = np.array([point[0] - box_size / 2, point[1] - box_size / 2, box_size, box_size], dtype=np.float32)
        else:
            mono_rel = item["drone_image"]
            mono = self._load_rgb(self._resolve_path(mono_rel))
            hm0, wm0 = mono.shape[:2]
            mono_bbox = np.array(item["drone_image_bbox"][:4], dtype=np.float32)
            point = np.array([mono_bbox[0] + mono_bbox[2] / 2, mono_bbox[1] + mono_bbox[3] / 2], dtype=np.float32)
            mono_mask = self._decode_rle(item.get("drone_segmentation"), hm0, wm0)
            if mono_mask is None:
                mono_mask = np.zeros((hm0, wm0), dtype=np.float32)
                x, y, w, h = mono_bbox.astype(np.int32)
                mono_mask[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = 1.0

        if (hm0, wm0) != (size, size):
            sx, sy = size / wm0, size / hm0
            mono = cv2.resize(mono, (size, size), interpolation=cv2.INTER_LINEAR)
            mono_mask = cv2.resize(mono_mask, (size, size), interpolation=cv2.INTER_NEAREST)
            point = np.array([point[0] * sx, point[1] * sy], dtype=np.float32)
            mono_bbox = np.array([mono_bbox[0] * sx, mono_bbox[1] * sy, mono_bbox[2] * sx, mono_bbox[3] * sy], dtype=np.float32)

        if (hs0, ws0) != (size, size):
            sxs, sys = size / ws0, size / hs0
            sat = cv2.resize(sat, (size, size), interpolation=cv2.INTER_LINEAR)
            sat_mask = cv2.resize(sat_mask, (size, size), interpolation=cv2.INTER_NEAREST)
            sat_bbox = np.array([sat_bbox[0] * sxs, sat_bbox[1] * sys, sat_bbox[2] * sxs, sat_bbox[3] * sys], dtype=np.float32)

        mono_bbox[0] = np.clip(mono_bbox[0], 0, size - 1)
        mono_bbox[1] = np.clip(mono_bbox[1], 0, size - 1)
        mono_bbox[2] = np.clip(mono_bbox[2], 1, size)
        mono_bbox[3] = np.clip(mono_bbox[3], 1, size)
        sat_bbox[0] = np.clip(sat_bbox[0], 0, size - 1)
        sat_bbox[1] = np.clip(sat_bbox[1], 0, size - 1)
        sat_bbox[2] = np.clip(sat_bbox[2], 1, size)
        sat_bbox[3] = np.clip(sat_bbox[3], 1, size)

        return {
            "front_view": to_tensor(Image.fromarray(mono)),
            "satellite_view": to_tensor(Image.fromarray(sat)),
            "mono_point": torch.from_numpy(point.astype(np.float32)),
            "mono_bbox": torch.from_numpy(self._bbox_xywh_to_cxcywh_norm(mono_bbox, size, size)),
            "mono_mask": torch.from_numpy((mono_mask > 0.5).astype(np.float32)).unsqueeze(0),
            "sat_mask": torch.from_numpy((sat_mask > 0.5).astype(np.float32)).unsqueeze(0),
            "sat_bbox": torch.from_numpy(self._bbox_xywh_to_cxcywh_norm(sat_bbox, size, size)),
            "mono_filename": mono_rel,
            "sat_filename": item["satellite_image"],
            "view_label": view,
            "triplet_id": triplet_id,
        }


def triplet_indices(dataset: UniversityTripletPairDataset, samples_per_class: int, seed: int) -> Tuple[List[int], np.ndarray, np.ndarray]:
    groups: Dict[int, Dict[str, int]] = {}
    for idx, row in enumerate(dataset.rows):
        groups.setdefault(int(row["triplet_id"]), {})[str(row["view"])] = idx
    pair_ids = [pid for pid, views in groups.items() if "ground" in views and "drone" in views]
    rng = random.Random(seed)
    rng.shuffle(pair_ids)
    n = min(int(samples_per_class), len(pair_ids))
    selected: List[int] = []
    labels: List[str] = []
    triplet_ids: List[int] = []
    for pid in pair_ids[:n]:
        for view in ["ground", "drone"]:
            selected.append(groups[pid][view])
            labels.append(view)
            triplet_ids.append(pid)
    order = list(range(len(selected)))
    rng.shuffle(order)
    selected = [selected[i] for i in order]
    labels_arr = np.asarray([labels[i] for i in order])
    triplet_arr = np.asarray([triplet_ids[i] for i in order], dtype=np.int64)
    print(f"Selected {n} ground-drone-satellite triplets ({len(selected)} rows).")
    return selected, labels_arr, triplet_arr


def masked_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bsz, num_tokens, _ = features.shape
    side = int(num_tokens**0.5)
    if side * side != num_tokens:
        return features.mean(dim=1)
    patch_mask = F.adaptive_avg_pool2d(mask.float(), (side, side)).reshape(bsz, -1)
    patch_mask = (patch_mask > 0.5).to(dtype=features.dtype)
    denom = patch_mask.sum(dim=1, keepdim=True)
    weights = torch.where(
        denom > 0,
        patch_mask / denom.clamp(min=1.0),
        torch.full_like(patch_mask, 1.0 / float(num_tokens)),
    )
    return (features * weights.unsqueeze(-1)).sum(dim=1)


def pool_features(
    features: torch.Tensor,
    mask: torch.Tensor,
    pool: str,
    cls_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    features = torch.nan_to_num(features.float())
    if pool == "masked_mean":
        pooled = masked_pool(features, mask)
    elif pool == "cls" and cls_token is not None:
        pooled = torch.nan_to_num(cls_token.float())
    else:
        pooled = features.mean(dim=1)
    return torch.nan_to_num(F.normalize(pooled, dim=-1))


def token_sample_features(
    features: torch.Tensor,
    mask: torch.Tensor,
    tokens_per_sample: int,
    seed: int,
) -> torch.Tensor:
    """Sample patch tokens directly, preferring mask-positive tokens."""
    features = torch.nan_to_num(features.float())
    bsz, num_tokens, dim = features.shape
    side = int(num_tokens**0.5)
    if side * side == num_tokens:
        patch_mask = F.adaptive_avg_pool2d(mask.float(), (side, side)).reshape(bsz, -1) > 0.5
    else:
        patch_mask = torch.ones(bsz, num_tokens, device=features.device, dtype=torch.bool)

    k = max(1, int(tokens_per_sample))
    sampled = []
    for i in range(bsz):
        candidates = torch.where(patch_mask[i])[0]
        if candidates.numel() == 0:
            candidates = torch.arange(num_tokens, device=features.device)
        gen = torch.Generator(device=features.device)
        gen.manual_seed(int(seed) + i)
        if candidates.numel() >= k:
            chosen = candidates[torch.randperm(candidates.numel(), device=features.device, generator=gen)[:k]]
        else:
            extra = candidates[torch.randint(candidates.numel(), (k - candidates.numel(),), device=features.device, generator=gen)]
            chosen = torch.cat([candidates, extra], dim=0)
        sampled.append(features[i, chosen])
    return torch.nan_to_num(F.normalize(torch.stack(sampled, dim=0).reshape(bsz * k, dim), dim=-1))


@torch.inference_mode()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    prompt_type: str,
    pool: str,
    desc: str,
    seed: int,
    tokens_per_sample: int,
) -> Dict[str, np.ndarray]:
    mono_chunks: List[np.ndarray] = []
    sat_chunks: List[np.ndarray] = []
    align_mono_chunks: List[np.ndarray] = []
    align_sat_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    sample_offset = 0
    for batch in tqdm(loader, desc=desc):
        front = batch["front_view"].to(device, non_blocking=True)
        sat = batch["satellite_view"].to(device, non_blocking=True)

        if prompt_type == "none":
            points, boxes, masks = None, None, None
        else:
            points, boxes, masks = prepare_single_prompt(batch, device, prompt_type=prompt_type)

        target_dtype = model.backbone.image_mean.dtype
        if front.dtype != target_dtype:
            front = front.to(target_dtype)
            sat = sat.to(target_dtype)
            if points is not None:
                points = (points[0].to(target_dtype), points[1])
            if boxes is not None:
                boxes = boxes.to(target_dtype)
            if masks is not None:
                masks = masks.to(target_dtype)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        sparse_embeddings = sparse_embeddings.to(target_dtype)
        dense_embeddings = dense_embeddings.to(target_dtype)
        prompt_coords = model._build_prompt_coords(points, boxes, sparse_embeddings, front.shape[0])
        backbone_out = model.backbone(
            front_view=front,
            satellite_view=sat,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings if masks is not None else None,
            prompt_coords=prompt_coords,
        )
        mono_mask = batch["mono_mask"].to(device, non_blocking=True)
        sat_mask = batch["sat_mask"].to(device, non_blocking=True)
        if pool == "token_sample":
            mono_tokens = token_sample_features(
                backbone_out["front_features"],
                mono_mask,
                tokens_per_sample=tokens_per_sample,
                seed=seed + sample_offset,
            )
            sat_tokens = token_sample_features(
                backbone_out["sate_features"],
                sat_mask,
                tokens_per_sample=tokens_per_sample,
                seed=seed + 1000003 + sample_offset,
            )
            mono_chunks.append(mono_tokens.cpu().numpy().astype(np.float32))
            sat_chunks.append(sat_tokens.cpu().numpy().astype(np.float32))
            batch_labels = np.asarray(
                ["drone" if "drone" in str(name).lower() else "ground" for name in batch["mono_filename"]]
            )
            label_chunks.append(np.repeat(batch_labels, tokens_per_sample))

            # Keep paired-alignment metrics image-level so the positive pair
            # still has a clear mono-satellite correspondence.
            align_mono = pool_features(backbone_out["front_features"], mono_mask, pool="masked_mean")
            align_sat = pool_features(backbone_out["sate_features"], sat_mask, pool="masked_mean")
            align_mono_chunks.append(align_mono.cpu().numpy().astype(np.float32))
            align_sat_chunks.append(align_sat.cpu().numpy().astype(np.float32))
        else:
            mono_pooled = pool_features(
                backbone_out["front_features"],
                mono_mask,
                pool=pool,
                cls_token=backbone_out.get("front_camera_token"),
            )
            sat_pooled = pool_features(
                backbone_out["sate_features"],
                sat_mask,
                pool=pool,
                cls_token=backbone_out.get("sate_camera_token"),
            )
            mono_chunks.append(mono_pooled.cpu().numpy().astype(np.float32))
            sat_chunks.append(sat_pooled.cpu().numpy().astype(np.float32))
        sample_offset += front.shape[0]
    mono = np.concatenate(mono_chunks, axis=0)
    sat = np.concatenate(sat_chunks, axis=0)
    residual = normalize_np(mono) - normalize_np(sat)
    out = {"mono": mono, "sat": sat, "residual": residual.astype(np.float32)}
    if pool == "token_sample":
        out["align_mono"] = np.concatenate(align_mono_chunks, axis=0)
        out["align_sat"] = np.concatenate(align_sat_chunks, axis=0)
        out["token_labels"] = np.concatenate(label_chunks, axis=0).astype(str)
    return out


def normalize_np(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(normalize_np(a) * normalize_np(b), axis=1)


def compute_embeddings(features: Dict[str, Dict[str, np.ndarray]], labels: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    names = ["without_cl", "with_cl"]
    plot_labels = features["without_cl"].get("token_labels", labels)
    all_features = np.concatenate(
        [
            features["without_cl"]["mono"],
            features["without_cl"]["sat"],
            features["with_cl"]["mono"],
            features["with_cl"]["sat"],
        ],
        axis=0,
    )
    nonfinite = int((~np.isfinite(all_features)).sum())
    if nonfinite:
        print(f"Warning: replacing {nonfinite} non-finite feature values before PCA/t-SNE.")
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
    all_features = StandardScaler().fit_transform(all_features)
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanstd(all_features)) < 1e-8:
        rng = np.random.default_rng(args.seed)
        all_features = all_features + rng.normal(0.0, 1e-6, size=all_features.shape)
    pca_dim = min(args.pca_dim, all_features.shape[0] - 1, all_features.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    all_pca = pca.fit_transform(all_features)
    # t-SNE requires perplexity < n_samples. Keep the default paper setting for
    # normal runs, but clamp aggressively so tiny smoke tests still work.
    perplexity = min(args.perplexity, max(1.0, (all_pca.shape[0] - 1) / 3.0))
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=args.tsne_iter,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
            metric="euclidean",
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=args.tsne_iter,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
            metric="euclidean",
        )
    embedding_all = tsne.fit_transform(all_pca).astype(np.float32)

    n = len(plot_labels)
    embeddings = {
        "without_cl": {
            "mono": embedding_all[:n],
            "satellite": embedding_all[n : 2 * n],
        },
        "with_cl": {
            "mono": embedding_all[2 * n : 3 * n],
            "satellite": embedding_all[3 * n : 4 * n],
        },
    }
    metrics = {}
    y = (plot_labels == "drone").astype(np.int64)
    y_align = (labels == "drone").astype(np.int64)
    for name in names:
        feat = features[name]["mono"]
        emb = embeddings[name]["mono"]
        align_mono = features[name].get("align_mono", features[name]["mono"])
        align_sat = features[name].get("align_sat", features[name]["sat"])
        pos_dist = cosine_distance(align_mono, align_sat)
        metrics[name] = {
            "paired_cosine_distance_mean": float(pos_dist.mean()),
            "paired_cosine_distance_std": float(pos_dist.std()),
            "paired_cosine_distance_ground_mean": float(pos_dist[y_align == 0].mean()),
            "paired_cosine_distance_ground_std": float(pos_dist[y_align == 0].std()),
            "paired_cosine_distance_drone_mean": float(pos_dist[y_align == 1].mean()),
            "paired_cosine_distance_drone_std": float(pos_dist[y_align == 1].std()),
            "positive_cosine_similarity_mean": float(1.0 - pos_dist.mean()),
            "mono_silhouette_feature": safe_silhouette(feat, y),
            "mono_silhouette_tsne": safe_silhouette(emb, y),
            "mono_knn_modal_purity_k10": float(knn_modal_purity(feat, y, k=10)),
            "mono_centroid_distance_feature": float(np.linalg.norm(feat[y == 0].mean(0) - feat[y == 1].mean(0))),
            "mono_centroid_distance_tsne": float(np.linalg.norm(emb[y == 0].mean(0) - emb[y == 1].mean(0))),
            "satellite_silhouette_feature": safe_silhouette(features[name]["sat"], y),
            "satellite_silhouette_tsne": safe_silhouette(embeddings[name]["satellite"], y),
        }
    return {
        "embeddings": embeddings,
        "metrics": metrics,
        "plot_labels": plot_labels,
        "pca_explained_variance": float(np.nan_to_num(pca.explained_variance_ratio_).sum()),
        "perplexity": float(perplexity),
    }


def knn_modal_purity(features: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    k = min(k + 1, len(y))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(features)
    indices = nn.kneighbors(features, return_distance=False)[:, 1:]
    return float((y[indices] == y[:, None]).mean())


def safe_silhouette(values: np.ndarray, y: np.ndarray) -> float:
    n_labels = len(np.unique(y))
    if values.shape[0] <= n_labels or n_labels < 2:
        return float("nan")
    if float(np.nanstd(values)) < 1e-12:
        return float("nan")
    try:
        return float(silhouette_score(np.nan_to_num(values), y))
    except ValueError:
        return float("nan")


def save_npz(
    path: Path,
    labels: np.ndarray,
    features: Dict[str, Dict[str, np.ndarray]],
    embeddings: Dict[str, Dict[str, np.ndarray]],
    triplet_ids: Optional[np.ndarray] = None,
) -> None:
    arrays = {
        "labels": labels,
        "mono_without_cl": features["without_cl"]["mono"],
        "sat_without_cl": features["without_cl"]["sat"],
        "residual_without_cl": features["without_cl"]["residual"],
        "mono_with_cl": features["with_cl"]["mono"],
        "sat_with_cl": features["with_cl"]["sat"],
        "residual_with_cl": features["with_cl"]["residual"],
        "mono_tsne_without_cl": embeddings["without_cl"]["mono"],
        "satellite_tsne_without_cl": embeddings["without_cl"]["satellite"],
        "mono_tsne_with_cl": embeddings["with_cl"]["mono"],
        "satellite_tsne_with_cl": embeddings["with_cl"]["satellite"],
    }
    for name in ["without_cl", "with_cl"]:
        if "align_mono" in features[name]:
            arrays[f"align_mono_{name}"] = features[name]["align_mono"]
            arrays[f"align_sat_{name}"] = features[name]["align_sat"]
        if "token_labels" in features[name]:
            arrays[f"token_labels_{name}"] = features[name]["token_labels"]
    if triplet_ids is not None:
        arrays["triplet_ids"] = triplet_ids
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Optional[np.ndarray]]:
    obj = np.load(path)
    labels = obj["labels"]
    features = {
        "without_cl": {
            "mono": obj["mono_without_cl"],
            "sat": obj["sat_without_cl"],
            "residual": obj["residual_without_cl"],
        },
        "with_cl": {
            "mono": obj["mono_with_cl"],
            "sat": obj["sat_with_cl"],
            "residual": obj["residual_with_cl"],
        },
    }
    for name in ["without_cl", "with_cl"]:
        if f"align_mono_{name}" in obj.files:
            features[name]["align_mono"] = obj[f"align_mono_{name}"]
            features[name]["align_sat"] = obj[f"align_sat_{name}"]
        if f"token_labels_{name}" in obj.files:
            features[name]["token_labels"] = obj[f"token_labels_{name}"].astype(str)
    embeddings = {
        "without_cl": {
            "mono": obj["mono_tsne_without_cl"] if "mono_tsne_without_cl" in obj.files else obj["residual_tsne_without_cl"],
            "satellite": obj["satellite_tsne_without_cl"] if "satellite_tsne_without_cl" in obj.files else obj["residual_tsne_without_cl"],
        },
        "with_cl": {
            "mono": obj["mono_tsne_with_cl"] if "mono_tsne_with_cl" in obj.files else obj["residual_tsne_with_cl"],
            "satellite": obj["satellite_tsne_with_cl"] if "satellite_tsne_with_cl" in obj.files else obj["residual_tsne_with_cl"],
        },
    }
    triplet_ids = obj["triplet_ids"] if "triplet_ids" in obj.files else None
    return labels, features, embeddings, triplet_ids


def plot_figure(
    output_dir: Path,
    labels: np.ndarray,
    embeddings: Dict[str, Dict[str, np.ndarray]],
    metrics: Dict[str, Dict[str, float]],
    meta: Dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )

    colors = {"ground": "#0072B2", "drone": "#D55E00"}
    names = [("without_cl", "w/o contrastive learning"), ("with_cl", "with contrastive learning")]
    all_xy = np.concatenate(
        [embeddings[name]["mono"] for name, _ in names]
        + [embeddings[name]["satellite"] for name, _ in names],
        axis=0,
    )
    pad = 0.06 * (all_xy.max(axis=0) - all_xy.min(axis=0) + 1e-6)
    xlim = (float(all_xy[:, 0].min() - pad[0]), float(all_xy[:, 0].max() + pad[0]))
    ylim = (float(all_xy[:, 1].min() - pad[1]), float(all_xy[:, 1].max() + pad[1]))

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), constrained_layout=True)
    for ax, (name, _) in zip(axes, names):
        xy = embeddings[name]["mono"]
        sat_xy = embeddings[name]["satellite"]
        ax.scatter(
            sat_xy[:, 0],
            sat_xy[:, 1],
            s=8,
            c="#595959",
            marker="o",
            alpha=0.28,
            linewidths=0,
            rasterized=True,
        )
        for label in ["ground", "drone"]:
            mask = labels == label
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=8,
                c=colors[label],
                alpha=0.72,
                linewidths=0,
                rasterized=True,
            )
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(length=2.5, width=0.7)
        ax.grid(True, color="#E6E6E6", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ext in ["pdf", "svg", "png"]:
        path = output_dir / f"satellite_mono_joint_tsne_default_v3.{ext}"
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {path}")
        compat_path = output_dir / f"satellite_anchor_residual_tsne_default_v3.{ext}"
        fig.savefig(compat_path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {compat_path}")
    plt.close(fig)


def plot_alignment_figure(
    output_dir: Path,
    labels: np.ndarray,
    features: Dict[str, Dict[str, np.ndarray]],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    colors = {"ground": "#0072B2", "drone": "#D55E00"}
    y = (labels == "drone").astype(np.int64)
    names = [("without_cl", "w/o CL"), ("with_cl", "with CL")]
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.1), constrained_layout=True)

    positions = []
    values = []
    facecolors = []
    tick_positions = []
    tick_labels = []
    pos = 1.0
    for name, model_label in names:
        dist = cosine_distance(
            features[name].get("align_mono", features[name]["mono"]),
            features[name].get("align_sat", features[name]["sat"]),
        )
        for class_value, class_label in [(0, "Ground"), (1, "Drone")]:
            vals = dist[y == class_value]
            positions.append(pos)
            values.append(vals)
            facecolors.append(colors[class_label.lower()])
            tick_positions.append(pos)
            tick_labels.append(f"{model_label}\n{class_label}")
            pos += 0.75
        pos += 0.35

    parts = ax.violinplot(values, positions=positions, widths=0.55, showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"], facecolors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.25)

    for x, vals, color in zip(positions, values, facecolors):
        rng = np.random.default_rng(int(x * 1000))
        sample = vals if len(vals) <= 220 else rng.choice(vals, size=220, replace=False)
        jitter = rng.normal(0.0, 0.035, size=len(sample))
        ax.scatter(np.full_like(sample, x) + jitter, sample, s=5, c=color, alpha=0.35, linewidths=0, rasterized=True)
        ax.plot([x - 0.18, x + 0.18], [np.mean(vals), np.mean(vals)], color=color, linewidth=1.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ["pdf", "svg", "png"]:
        path = output_dir / f"paired_satellite_alignment_default_v3.{ext}"
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def paired_ground_drone_distances(
    features: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
    triplet_ids: Optional[np.ndarray],
    name: str,
) -> Optional[np.ndarray]:
    if triplet_ids is None:
        return None
    mono = features[name].get("align_mono", features[name]["mono"])
    if len(mono) != len(labels) or len(triplet_ids) != len(labels):
        return None
    rows: Dict[int, Dict[str, int]] = {}
    for idx, (label, tid) in enumerate(zip(labels.astype(str), triplet_ids.astype(np.int64))):
        if label in {"ground", "drone"}:
            rows.setdefault(int(tid), {})[label] = idx
    values = []
    for pair in rows.values():
        if "ground" in pair and "drone" in pair:
            g = mono[pair["ground"] : pair["ground"] + 1]
            d = mono[pair["drone"] : pair["drone"] + 1]
            values.append(float(cosine_distance(g, d)[0]))
    if not values:
        return None
    return np.asarray(values, dtype=np.float32)


def add_ground_drone_metrics(
    metrics: Dict[str, Dict[str, float]],
    features: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
    triplet_ids: Optional[np.ndarray],
) -> None:
    for name in ["without_cl", "with_cl"]:
        dist = paired_ground_drone_distances(features, labels, triplet_ids, name)
        if dist is None:
            continue
        metrics[name]["paired_ground_drone_cosine_distance_mean"] = float(dist.mean())
        metrics[name]["paired_ground_drone_cosine_distance_std"] = float(dist.std())
        metrics[name]["paired_ground_drone_cosine_distance_median"] = float(np.median(dist))
        metrics[name]["paired_ground_drone_pair_count"] = int(len(dist))


def plot_ground_drone_alignment_figure(
    output_dir: Path,
    labels: np.ndarray,
    features: Dict[str, Dict[str, np.ndarray]],
    triplet_ids: Optional[np.ndarray],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    d0 = paired_ground_drone_distances(features, labels, triplet_ids, "without_cl")
    d1 = paired_ground_drone_distances(features, labels, triplet_ids, "with_cl")
    if d0 is None or d1 is None:
        print("Skip paired drone-ground distance figure: no paired triplet_id data.")
        return

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.1), constrained_layout=True)
    values = [d0, d1]
    positions = [1.0, 1.8]
    colors = ["#8C8C8C", "#009E73"]
    parts = ax.violinplot(values, positions=positions, widths=0.52, showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.25)
    for x, vals, color in zip(positions, values, colors):
        rng = np.random.default_rng(int(x * 1000))
        sample = vals if len(vals) <= 260 else rng.choice(vals, size=260, replace=False)
        jitter = rng.normal(0.0, 0.035, size=len(sample))
        ax.scatter(np.full_like(sample, x) + jitter, sample, s=5, c=color, alpha=0.35, linewidths=0, rasterized=True)
        ax.plot([x - 0.17, x + 0.17], [np.mean(vals), np.mean(vals)], color=color, linewidth=1.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(["w/o CL", "with CL"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for ext in ["pdf", "svg", "png"]:
        path = output_dir / f"paired_drone_ground_alignment_default_v3.{ext}"
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = output_dir / "satellite_anchor_features.npz"

    if args.cache_only:
        labels, features, embeddings, triplet_ids = load_npz(feature_cache)
        tsne_result = compute_embeddings(features, labels, args)
        embeddings = tsne_result["embeddings"]
        metrics = tsne_result["metrics"]
        add_ground_drone_metrics(metrics, features, labels, triplet_ids)
        plot_labels = tsne_result["plot_labels"]
        meta = vars(args)
        meta.update(
            {
                "n_samples": int(len(labels)),
                "n_tsne_points_per_modality": int(len(plot_labels)),
                "labels": {"ground": int((labels == "ground").sum()), "drone": int((labels == "drone").sum())},
                "pca_explained_variance": tsne_result["pca_explained_variance"],
                "actual_perplexity": tsne_result["perplexity"],
                "cache_only_recomputed_tsne": True,
            }
        )
        save_npz(feature_cache, labels, features, embeddings, triplet_ids=triplet_ids)
        plot_figure(output_dir, plot_labels, embeddings, metrics, meta)
        plot_alignment_figure(output_dir, labels, features, metrics)
        plot_ground_drone_alignment_figure(output_dir, labels, features, triplet_ids, metrics)
        write_json(output_dir / "satellite_anchor_metrics.json", {"metrics": metrics, "meta": meta})
        return

    cfg = load_cfg(args.config)
    json_path = args.json or cfg["data"]["val_json"]
    data_root = args.data_root or cfg["data"]["data_root"]
    device = torch.device("cpu" if args.force_cpu else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Dataset: {json_path}")

    triplet_ids: Optional[np.ndarray] = None
    if args.input_format == "triplet":
        dataset = UniversityTripletPairDataset(
            triplet_json=args.triplet_json,
            root_dir=args.triplet_root_dir,
            input_size=cfg["data"].get("img_size", 518),
        )
        indices, labels, triplet_ids = triplet_indices(dataset, args.samples_per_class, args.seed)
        json_path = args.triplet_json
        data_root = args.triplet_root_dir
    else:
        dataset = CrossViewDataset(
            json_path=json_path,
            data_root=data_root,
            mono_size=cfg["data"].get("img_size", 518),
            sat_size=1280,
            crop_sat=False,
            crop_size=cfg["data"].get("crop_size", cfg["data"].get("img_size", 518)),
            view_subset="all",
        )
        indices, labels = balanced_indices(dataset, args.samples_per_class, args.seed)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    features: Dict[str, Dict[str, np.ndarray]] = {}
    for name, ckpt in [
        ("without_cl", args.without_cl_ckpt),
        ("with_cl", args.with_cl_ckpt),
    ]:
        print(f"\nExtracting {name} features from {ckpt}")
        model = build_model_from_cfg(cfg, ckpt, device)
        features[name] = extract_features(
            model=model,
            loader=loader,
            device=device,
            prompt_type=args.prompt,
            pool=args.pool,
            desc=f"{name}",
            seed=args.seed,
            tokens_per_sample=args.tokens_per_sample,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tsne_result = compute_embeddings(features, labels, args)
    embeddings = tsne_result["embeddings"]
    metrics = tsne_result["metrics"]
    add_ground_drone_metrics(metrics, features, labels, triplet_ids)
    plot_labels = tsne_result["plot_labels"]
    save_npz(feature_cache, labels, features, embeddings, triplet_ids=triplet_ids)

    meta = vars(args)
    meta.update(
        {
            "json_path": json_path,
            "data_root": data_root,
            "input_format": args.input_format,
            "n_samples": int(len(labels)),
            "n_triplet_pairs": int(len(np.unique(triplet_ids))) if triplet_ids is not None else 0,
            "n_tsne_points_per_modality": int(len(plot_labels)),
            "labels": {"ground": int((labels == "ground").sum()), "drone": int((labels == "drone").sum())},
            "pca_explained_variance": tsne_result["pca_explained_variance"],
            "actual_perplexity": tsne_result["perplexity"],
        }
    )
    write_json(output_dir / "satellite_anchor_metrics.json", {"metrics": metrics, "meta": meta})
    plot_figure(output_dir, plot_labels, embeddings, metrics, meta)
    plot_alignment_figure(output_dir, labels, features, metrics)
    plot_ground_drone_alignment_figure(output_dir, labels, features, triplet_ids, metrics)


def compute_metrics_from_cache(
    features: Dict[str, Dict[str, np.ndarray]],
    embeddings: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    y = (labels == "drone").astype(np.int64)
    metrics = {}
    for name in ["without_cl", "with_cl"]:
        mono = features[name]["mono"]
        pos_dist = cosine_distance(features[name]["mono"], features[name]["sat"])
        metrics[name] = {
            "paired_cosine_distance_mean": float(pos_dist.mean()),
            "paired_cosine_distance_std": float(pos_dist.std()),
            "paired_cosine_distance_ground_mean": float(pos_dist[y == 0].mean()),
            "paired_cosine_distance_ground_std": float(pos_dist[y == 0].std()),
            "paired_cosine_distance_drone_mean": float(pos_dist[y == 1].mean()),
            "paired_cosine_distance_drone_std": float(pos_dist[y == 1].std()),
            "positive_cosine_similarity_mean": float(1.0 - pos_dist.mean()),
            "mono_silhouette_feature": safe_silhouette(mono, y),
            "mono_silhouette_tsne": safe_silhouette(embeddings[name]["mono"], y),
            "mono_knn_modal_purity_k10": float(knn_modal_purity(mono, y, k=10)),
            "mono_centroid_distance_feature": float(np.linalg.norm(mono[y == 0].mean(0) - mono[y == 1].mean(0))),
            "mono_centroid_distance_tsne": float(
                np.linalg.norm(embeddings[name]["mono"][y == 0].mean(0) - embeddings[name]["mono"][y == 1].mean(0))
            ),
            "satellite_silhouette_feature": safe_silhouette(features[name]["sat"], y),
            "satellite_silhouette_tsne": safe_silhouette(embeddings[name]["satellite"], y),
        }
    return metrics


if __name__ == "__main__":
    main()
