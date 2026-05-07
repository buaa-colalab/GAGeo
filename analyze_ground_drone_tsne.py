#!/usr/bin/env python3
"""Plot paper-quality t-SNE for ground/drone front-view features.

The script compares two checkpoints with and without contrastive learning using
the same balanced sample set and the same t-SNE embedding.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
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
        description="Compare ground/drone feature distributions by t-SNE."
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
        choices=["masked_mean", "mean", "cls"],
        default="masked_mean",
        help="How to pool front-view patch features.",
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


@torch.inference_mode()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    prompt_type: str,
    pool: str,
    desc: str,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
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
        front_features = torch.nan_to_num(backbone_out["front_features"].float())
        if pool == "masked_mean":
            pooled = masked_pool(front_features, batch["mono_mask"].to(device, non_blocking=True))
        elif pool == "cls":
            pooled = backbone_out.get("front_camera_token", front_features.mean(dim=1)).float()
        else:
            pooled = front_features.mean(dim=1)
        pooled = F.normalize(pooled, dim=-1)
        pooled = torch.nan_to_num(pooled)
        chunks.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def compute_embeddings(features: Dict[str, np.ndarray], labels: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    names = ["without_cl", "with_cl"]
    all_features = np.concatenate([features[name] for name in names], axis=0)
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

    n = len(labels)
    embeddings = {
        "without_cl": embedding_all[:n],
        "with_cl": embedding_all[n:],
    }
    metrics = {}
    y = (labels == "drone").astype(np.int64)
    for name in names:
        feat = features[name]
        emb = embeddings[name]
        metrics[name] = {
            "silhouette_feature": safe_silhouette(feat, y),
            "silhouette_tsne": safe_silhouette(emb, y),
            "knn_modal_purity_k10": float(knn_modal_purity(feat, y, k=10)),
            "centroid_distance_feature": float(np.linalg.norm(feat[y == 0].mean(0) - feat[y == 1].mean(0))),
            "centroid_distance_tsne": float(np.linalg.norm(emb[y == 0].mean(0) - emb[y == 1].mean(0))),
        }
    return {
        "embeddings": embeddings,
        "metrics": metrics,
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


def save_npz(path: Path, labels: np.ndarray, features: Dict[str, np.ndarray], embeddings: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        labels=labels,
        features_without_cl=features["without_cl"],
        features_with_cl=features["with_cl"],
        tsne_without_cl=embeddings["without_cl"],
        tsne_with_cl=embeddings["with_cl"],
    )


def load_npz(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    obj = np.load(path)
    labels = obj["labels"]
    features = {
        "without_cl": obj["features_without_cl"],
        "with_cl": obj["features_with_cl"],
    }
    embeddings = {
        "without_cl": obj["tsne_without_cl"],
        "with_cl": obj["tsne_with_cl"],
    }
    return labels, features, embeddings


def plot_figure(
    output_dir: Path,
    labels: np.ndarray,
    embeddings: Dict[str, np.ndarray],
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
    names = [("without_cl", "(a) w/o contrastive learning"), ("with_cl", "(b) with contrastive learning")]
    all_xy = np.concatenate([embeddings[name] for name, _ in names], axis=0)
    pad = 0.06 * (all_xy.max(axis=0) - all_xy.min(axis=0) + 1e-6)
    xlim = (float(all_xy[:, 0].min() - pad[0]), float(all_xy[:, 0].max() + pad[0]))
    ylim = (float(all_xy[:, 1].min() - pad[1]), float(all_xy[:, 1].max() + pad[1]))

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), constrained_layout=True)
    for ax, (name, title) in zip(axes, names):
        xy = embeddings[name]
        for label in ["ground", "drone"]:
            mask = labels == label
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=8,
                c=colors[label],
                label=label.capitalize(),
                alpha=0.72,
                linewidths=0,
                rasterized=True,
            )
        sil = metrics.get(name, {}).get("silhouette_feature", float("nan"))
        purity = metrics.get(name, {}).get("knn_modal_purity_k10", float("nan"))
        ax.set_title(f"{title}\nSilhouette={sil:.3f}, kNN purity={purity:.3f}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(length=2.5, width=0.7)
        ax.grid(True, color="#E6E6E6", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        handletextpad=0.3,
        columnspacing=1.2,
    )
    fig.suptitle(
        f"Ground vs. drone front-view features ({meta['pool']}, {meta['split_name']}, n={len(labels)})",
        y=1.02,
        fontsize=10.5,
    )

    for ext in ["pdf", "svg", "png"]:
        path = output_dir / f"ground_drone_tsne_default_v3.{ext}"
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
    feature_cache = output_dir / "ground_drone_tsne_features.npz"

    if args.cache_only:
        labels, features, embeddings = load_npz(feature_cache)
        metrics = compute_metrics_from_cache(features, embeddings, labels)
        meta = vars(args)
        plot_figure(output_dir, labels, embeddings, metrics, meta)
        write_json(output_dir / "ground_drone_tsne_metrics.json", {"metrics": metrics, "meta": meta})
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

    features: Dict[str, np.ndarray] = {}
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
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tsne_result = compute_embeddings(features, labels, args)
    embeddings = tsne_result["embeddings"]
    metrics = tsne_result["metrics"]
    save_npz(feature_cache, labels, features, embeddings)

    meta = vars(args)
    meta.update(
        {
            "json_path": json_path,
            "data_root": data_root,
            "n_samples": int(len(labels)),
            "labels": {"ground": int((labels == "ground").sum()), "drone": int((labels == "drone").sum())},
            "pca_explained_variance": tsne_result["pca_explained_variance"],
            "actual_perplexity": tsne_result["perplexity"],
        }
    )
    write_json(output_dir / "ground_drone_tsne_metrics.json", {"metrics": metrics, "meta": meta})
    plot_figure(output_dir, labels, embeddings, metrics, meta)


def compute_metrics_from_cache(
    features: Dict[str, np.ndarray],
    embeddings: Dict[str, np.ndarray],
    labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    y = (labels == "drone").astype(np.int64)
    metrics = {}
    for name in ["without_cl", "with_cl"]:
        metrics[name] = {
            "silhouette_feature": safe_silhouette(features[name], y),
            "silhouette_tsne": safe_silhouette(embeddings[name], y),
            "knn_modal_purity_k10": float(knn_modal_purity(features[name], y, k=10)),
            "centroid_distance_feature": float(
                np.linalg.norm(features[name][y == 0].mean(0) - features[name][y == 1].mean(0))
            ),
            "centroid_distance_tsne": float(
                np.linalg.norm(embeddings[name][y == 0].mean(0) - embeddings[name][y == 1].mean(0))
            ),
        }
    return metrics


if __name__ == "__main__":
    main()
