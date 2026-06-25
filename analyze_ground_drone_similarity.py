#!/usr/bin/env python3
"""Plot ground-drone cross-view feature similarity for CL ablation.

The figure directly measures whether matched University ground/drone object
features are closer than mismatched pairs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from analyze_ground_drone_tsne import (
    DEFAULT_ENV,
    UniversityTripletPairDataset,
    build_model_from_cfg,
    extract_features,
    load_cfg,
    triplet_indices,
)


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground-drone similarity matrix for GAGeo CL ablation.")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "default_v3.yaml"))
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
    parser.add_argument(
        "--triplet-json",
        type=str,
        default=str(WORKSPACE_ROOT / "University-Release" / "verified_triplets_sam2_masks.json"),
    )
    parser.add_argument("--triplet-root-dir", type=str, default=str(WORKSPACE_ROOT / "University-Release"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "ground_drone_similarity_cl_ablation"),
    )
    parser.add_argument("--num-triplets", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, choices=["point", "bbox", "mask", "none"], default="point")
    parser.add_argument("--pool", type=str, choices=["masked_mean", "mean", "cls"], default="masked_mean")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-only", action="store_true", help="Only redraw from cached similarity_features.npz.")
    return parser.parse_args()


def normalize_np(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


def pair_ground_drone_features(
    mono_features: np.ndarray,
    labels: np.ndarray,
    triplet_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: Dict[int, Dict[str, int]] = {}
    for idx, (label, tid) in enumerate(zip(labels.astype(str), triplet_ids.astype(np.int64))):
        rows.setdefault(int(tid), {})[label] = idx

    paired_ids: List[int] = []
    ground: List[np.ndarray] = []
    drone: List[np.ndarray] = []
    for tid in sorted(rows):
        pair = rows[tid]
        if "ground" not in pair or "drone" not in pair:
            continue
        paired_ids.append(tid)
        ground.append(mono_features[pair["ground"]])
        drone.append(mono_features[pair["drone"]])

    if not paired_ids:
        raise ValueError("No paired ground/drone rows found.")
    return (
        np.stack(ground, axis=0).astype(np.float32),
        np.stack(drone, axis=0).astype(np.float32),
        np.asarray(paired_ids, dtype=np.int64),
    )


def similarity_and_metrics(ground: np.ndarray, drone: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    ground = normalize_np(ground)
    drone = normalize_np(drone)
    sim = ground @ drone.T
    n = sim.shape[0]
    diag = np.diag(sim)
    off_mask = ~np.eye(n, dtype=bool)
    off = sim[off_mask]
    g_rank = np.argsort(-sim, axis=1)
    d_rank = np.argsort(-sim, axis=0)
    gt = np.arange(n)
    metrics = {
        "num_pairs": int(n),
        "all_similarity_mean": float(sim.mean()),
        "all_similarity_std": float(sim.std()),
        "positive_similarity_mean": float(diag.mean()),
        "positive_similarity_std": float(diag.std()),
        "negative_similarity_mean": float(off.mean()),
        "negative_similarity_std": float(off.std()),
        "positive_negative_gap": float(diag.mean() - off.mean()),
        "g2d_r1": float(np.mean(g_rank[:, 0] == gt)),
        "g2d_r5": float(np.mean([i in g_rank[i, : min(5, n)] for i in range(n)])),
        "d2g_r1": float(np.mean(d_rank[0, :] == gt)),
        "d2g_r5": float(np.mean([i in d_rank[: min(5, n), i] for i in range(n)])),
    }
    return sim.astype(np.float32), metrics


def plot_two_panel(
    output_dir: Path,
    sims: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )

    all_values = np.concatenate([sims["without_cl"].reshape(-1), sims["with_cl"].reshape(-1)])
    vmin = float(np.percentile(all_values, 2.0))
    vmax = float(np.percentile(all_values, 98.0))
    if vmax <= vmin:
        vmin, vmax = float(all_values.min()), float(all_values.max())

    fig = plt.figure(figsize=(6.8, 2.85))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.045],
        left=0.035,
        right=0.985,
        bottom=0.08,
        top=0.88,
        wspace=0.12,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    titles = {"without_cl": "(a) GAGeo w/o CL", "with_cl": "(b) GAGeo w/ CL"}
    image = None
    for ax, name in zip(axes, ["without_cl", "with_cl"]):
        sim = sims[name]
        image = ax.imshow(sim, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest", rasterized=True)
        ax.set_title(titles[name])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        m = metrics[name]
        text = (
            f"Mean {m['all_similarity_mean']:.3f}\n"
            f"Std {m['all_similarity_std']:.3f}"
        )
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=7.2,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2.2},
        )
    if image is not None:
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("Cosine", fontsize=7.0)
        cbar.ax.tick_params(labelsize=7.0, length=2.0, width=0.6)

    for ext in ["pdf", "svg", "png"]:
        path = output_dir / f"ground_drone_similarity_matrix_cl_ablation.{ext}"
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def write_json(path: Path, obj: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_cache(path: Path) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    obj = np.load(path, allow_pickle=False)
    labels = obj["labels"].astype(str)
    features = {
        "without_cl": {"mono": obj["mono_without_cl"]},
        "with_cl": {"mono": obj["mono_with_cl"]},
    }
    return labels, features


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "similarity_features.npz"

    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
    cfg = load_cfg(args.config)

    if args.cache_only:
        labels, features = load_cache(cache_path)
        triplet_ids = np.load(cache_path, allow_pickle=False)["triplet_ids"].astype(np.int64)
    else:
        device = torch.device(args.device)
        dataset = UniversityTripletPairDataset(
            triplet_json=args.triplet_json,
            root_dir=args.triplet_root_dir,
            input_size=cfg["data"].get("img_size", 518),
        )
        indices, labels, triplet_ids = triplet_indices(dataset, args.num_triplets, args.seed)
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
            print(f"\nExtracting {name} object features from {ckpt}")
            model = build_model_from_cfg(cfg, ckpt, device)
            features[name] = extract_features(
                model=model,
                loader=loader,
                device=device,
                prompt_type=args.prompt,
                pool=args.pool,
                desc=name,
                seed=args.seed,
                tokens_per_sample=1,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        np.savez_compressed(
            cache_path,
            labels=labels.astype(str),
            triplet_ids=triplet_ids.astype(np.int64),
            mono_without_cl=features["without_cl"]["mono"],
            mono_with_cl=features["with_cl"]["mono"],
        )
        print(f"Saved {cache_path}")

    sims: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    paired_ids: Optional[np.ndarray] = None
    for name in ["without_cl", "with_cl"]:
        ground, drone, ids = pair_ground_drone_features(features[name]["mono"], labels, triplet_ids)
        if paired_ids is None:
            paired_ids = ids
        elif not np.array_equal(paired_ids, ids):
            raise ValueError("Paired triplet order differs between models.")
        sims[name], metrics[name] = similarity_and_metrics(ground, drone)

    np.savez_compressed(
        output_dir / "similarity_matrices.npz",
        triplet_ids=paired_ids,
        sim_without_cl=sims["without_cl"],
        sim_with_cl=sims["with_cl"],
    )
    meta = {
        "config": args.config,
        "triplet_json": args.triplet_json,
        "triplet_root_dir": args.triplet_root_dir,
        "with_cl_ckpt": args.with_cl_ckpt,
        "without_cl_ckpt": args.without_cl_ckpt,
        "num_requested_triplets": int(args.num_triplets),
        "num_pairs": int(len(paired_ids)) if paired_ids is not None else 0,
        "prompt": args.prompt,
        "pool": args.pool,
        "seed": int(args.seed),
    }
    write_json(output_dir / "ground_drone_similarity_metrics.json", {"metrics": metrics, "meta": meta})
    plot_two_panel(output_dir, sims, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
