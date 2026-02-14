#!/usr/bin/env python3
"""
Upload pre-packaged dataset files to a Hugging Face dataset repository.

Designed for very large datasets (hundreds of GB) that were split into parts.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload packaged dataset parts to Hugging Face")
    p.add_argument("--repo-id", default="cipual/Urban-CVOGL", help="HF dataset repo id")
    p.add_argument("--pack-dir", required=True, help="Directory containing packaged files to upload")
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    p.add_argument("--private", action="store_true", help="Create repo as private if it doesn't exist")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--sleep-base", type=float, default=2.0)
    return p.parse_args()


def gather_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files.append(p)
    return files


def upload_one(api: HfApi, file_path: Path, root: Path, repo_id: str, repo_type: str, max_retries: int, sleep_base: float) -> None:
    rel = file_path.relative_to(root).as_posix()
    for i in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=rel,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Upload {rel}",
            )
            print(f"[OK] {rel}")
            return
        except Exception as e:  # noqa: BLE001
            wait_s = sleep_base * (2 ** i)
            print(f"[RETRY {i+1}/{max_retries}] {rel} failed: {e}. sleep={wait_s:.1f}s")
            time.sleep(wait_s)
    raise RuntimeError(f"Failed to upload after retries: {rel}")


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Missing token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")

    pack_dir = Path(args.pack_dir).resolve()
    if not pack_dir.exists():
        raise SystemExit(f"pack dir not found: {pack_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    files = gather_files(pack_dir)
    if not files:
        raise SystemExit(f"No files found in {pack_dir}")

    print(f"Repo: {args.repo_id} ({args.repo_type})")
    print(f"Pack dir: {pack_dir}")
    print(f"Files to upload: {len(files)}")

    for f in files:
        upload_one(api, f, pack_dir, args.repo_id, args.repo_type, args.max_retries, args.sleep_base)

    print("Upload finished.")


if __name__ == "__main__":
    main()
