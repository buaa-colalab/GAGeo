#!/usr/bin/env python3
"""Collect all pretrained checkpoints needed by rebuttal training scripts.

The training configs in this repo expect these filenames under CHECKPOINT_DIR:
  - pi3_model.safetensors
  - sam2.1_hiera_large.pt
  - vit_b_16_imagenet1k_v1.pth
  - vit_h_14_imagenet1k_swag_e2e_v1.pth

All downloads go through Hugging Face Hub. For Pi3 and SAM2.1 we use public
upstream HF repos by default. For torchvision ViT-B/H weights, provide a HF
mirror repo containing the exact `.pth` files used by the training configs, or
put all four expected files in one repo and pass `--hf_repo`.
"""

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_PI3_HF_REPO = "yyfz233/Pi3"
DEFAULT_SAM_HF_REPO = "facebook/sam2.1-hiera-large"
DEFAULT_SAM_HF_FILENAME = "sam2.1_hiera_large.pt"
DEFAULT_PI3_HF_CANDIDATES = (
    "model.safetensors",
    "pi3_model.safetensors",
    "pytorch_model.safetensors",
)


def hf_download_file(name, dst, repo_id, filename, repo_type="model"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {name}: {dst}")
        return

    if not repo_id or not filename:
        raise ValueError(
            f"Missing HF repo/filename for {name}. "
            "Pass --hf_repo or the corresponding --*_hf_repo/--*_hf_filename."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for checkpoint download.") from exc

    print(f"[download] {name} from HF {repo_id}/{filename} -> {dst}")
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    shutil.copy2(cached, dst)


def maybe_download_pi3_from_hf(dst, repo_id, filename, repo_type="model"):
    """Download Pi3 weights from Hugging Face Hub.

    The Pi3 repo layout may change over time, so we first try the explicitly
    requested filename, then fall back to common safetensors names, and finally
    inspect the repo file list to find a suitable `.safetensors` checkpoint.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("[warn] huggingface_hub is not installed; cannot download Pi3 from HF.")
        return False

    repo = str(repo_id or DEFAULT_PI3_HF_REPO).strip()
    candidates: list[str] = []
    if filename:
        candidates.append(str(filename).strip())
    for candidate in DEFAULT_PI3_HF_CANDIDATES:
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        try:
            print(f"[download] Pi3 from HF {repo}/{candidate} -> {dst}")
            cached = hf_hub_download(repo_id=repo, filename=candidate, repo_type=repo_type)
            shutil.copy2(cached, dst)
            return True
        except Exception as exc:
            print(f"[warn] Pi3 HF candidate not available: {candidate} ({exc})")

    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo, repo_type=repo_type)
    except Exception as exc:
        print(f"[warn] Failed to inspect HF repo {repo}: {exc}")
        return False

    safetensor_files = [name for name in repo_files if name.endswith(".safetensors")]
    preferred = sorted(
        safetensor_files,
        key=lambda name: (
            0 if Path(name).name in DEFAULT_PI3_HF_CANDIDATES else 1,
            0 if "model" in Path(name).name.lower() else 1,
            len(name),
            name,
        ),
    )
    if not preferred:
        print(f"[warn] No .safetensors checkpoint found in HF repo {repo}.")
        return False

    chosen = preferred[0]
    print(f"[download] Pi3 from HF {repo}/{chosen} -> {dst}")
    cached = hf_hub_download(repo_id=repo, filename=chosen, repo_type=repo_type)
    shutil.copy2(cached, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="/mnt/data/wrp/checkpoints_offline")
    parser.add_argument(
        "--hf_repo",
        default="",
        help=(
            "Optional all-in-one HF repo containing the exact expected files: "
            "pi3_model.safetensors, sam2.1_hiera_large.pt, "
            "vit_b_16_imagenet1k_v1.pth, vit_h_14_imagenet1k_swag_e2e_v1.pth."
        ),
    )
    parser.add_argument(
        "--hf_repo_type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Repo type for --hf_repo.",
    )
    parser.add_argument(
        "--pi3_hf_repo",
        default="",
        help=f"HF repo id for Pi3 checkpoint. Defaults to {DEFAULT_PI3_HF_REPO} when --hf_repo is not set.",
    )
    parser.add_argument(
        "--pi3_hf_filename",
        default="",
        help="Optional exact Pi3 filename inside the HF repo; auto-discovered when empty",
    )
    parser.add_argument("--pi3_hf_repo_type", default="model", choices=("model", "dataset", "space"))
    parser.add_argument("--sam_hf_repo", default="", help=f"Defaults to {DEFAULT_SAM_HF_REPO} when --hf_repo is not set.")
    parser.add_argument("--sam_hf_filename", default=DEFAULT_SAM_HF_FILENAME)
    parser.add_argument("--sam_hf_repo_type", default="model", choices=("model", "dataset", "space"))
    parser.add_argument("--vit_b_hf_repo", default="", help="HF repo containing exact ViT-B torchvision .pth")
    parser.add_argument(
        "--vit_b_hf_filename",
        default="vit_b_16_imagenet1k_v1.pth",
        help="Filename inside --vit_b_hf_repo or --hf_repo.",
    )
    parser.add_argument("--vit_b_hf_repo_type", default="model", choices=("model", "dataset", "space"))
    parser.add_argument("--vit_h_hf_repo", default="", help="HF repo containing exact ViT-H torchvision .pth")
    parser.add_argument(
        "--vit_h_hf_filename",
        default="vit_h_14_imagenet1k_swag_e2e_v1.pth",
        help="Filename inside --vit_h_hf_repo or --hf_repo.",
    )
    parser.add_argument("--vit_h_hf_repo_type", default="model", choices=("model", "dataset", "space"))
    parser.add_argument("--skip_vit_h", action="store_true", help="Skip the large ViT-H download")
    args = parser.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    pi3_dst = out / "pi3_model.safetensors"
    pi3_repo = args.pi3_hf_repo or args.hf_repo or DEFAULT_PI3_HF_REPO
    pi3_repo_type = (
        args.pi3_hf_repo_type
        if args.pi3_hf_repo
        else (args.hf_repo_type if args.hf_repo else "model")
    )
    if args.hf_repo:
        hf_download_file("Pi3", pi3_dst, pi3_repo, args.pi3_hf_filename or "pi3_model.safetensors", pi3_repo_type)
    elif not maybe_download_pi3_from_hf(pi3_dst, pi3_repo, args.pi3_hf_filename, pi3_repo_type):
        print(
            "[error] Pi3 checkpoint was not prepared. Re-run with --pi3_hf_repo/--pi3_hf_filename.",
            file=sys.stderr,
        )
        return 2

    sam_repo = args.sam_hf_repo or args.hf_repo or DEFAULT_SAM_HF_REPO
    sam_repo_type = (
        args.sam_hf_repo_type
        if args.sam_hf_repo
        else (args.hf_repo_type if args.hf_repo else "model")
    )
    sam_filename = args.sam_hf_filename or DEFAULT_SAM_HF_FILENAME
    hf_download_file("SAM2.1 Hiera-L", out / "sam2.1_hiera_large.pt", sam_repo, sam_filename, sam_repo_type)

    vit_b_repo = args.vit_b_hf_repo or args.hf_repo
    vit_b_repo_type = args.vit_b_hf_repo_type if args.vit_b_hf_repo else args.hf_repo_type
    hf_download_file(
        "ViT-B/16 ImageNet-1K",
        out / "vit_b_16_imagenet1k_v1.pth",
        vit_b_repo,
        args.vit_b_hf_filename,
        vit_b_repo_type,
    )

    if not args.skip_vit_h:
        vit_h_repo = args.vit_h_hf_repo or args.hf_repo
        vit_h_repo_type = args.vit_h_hf_repo_type if args.vit_h_hf_repo else args.hf_repo_type
        hf_download_file(
            "ViT-H/14 SWAG E2E",
            out / "vit_h_14_imagenet1k_swag_e2e_v1.pth",
            vit_h_repo,
            args.vit_h_hf_filename,
            vit_h_repo_type,
        )

    manifest = out / "offline_paths.yaml"
    manifest.write_text(
        "\n".join(
            [
                f"CHECKPOINT_DIR: {out}",
                f"pi3_weights: {pi3_dst}",
                f"sam_weights: {out / 'sam2.1_hiera_large.pt'}",
                f"vit_b_weights: {out / 'vit_b_16_imagenet1k_v1.pth'}",
                f"vit_h_weights: {out / 'vit_h_14_imagenet1k_swag_e2e_v1.pth'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[done] checkpoint directory: {out}")
    print(f"[done] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
