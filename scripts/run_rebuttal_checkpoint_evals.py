#!/usr/bin/env python3
"""Run rebuttal checkpoint evaluations across all available GPUs.

Planned evaluations:
- TROGeo-Pi3: D->S and G->S on unseen_test, reporting mAcc(%) and mIoU(%).
- GAGeo+ViT-B / GAGeo+ViT-H: D->S and G->S on unseen_test.
- GAGeo+PE / MoCo |Q|=4096: zero-shot G->D unseen.

The script discovers every checkpoint under each experiment directory and runs
one evaluation job per GPU. It writes per-job JSON/log files and a summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LOCATION_ROOT = Path(__file__).resolve().parents[1]
WRP_ROOT = LOCATION_ROOT.parent
CVOS_ROOT = WRP_ROOT / "CVOS-Code"


@dataclass(frozen=True)
class EvalJob:
    experiment: str
    checkpoint: Path
    family: str
    setting: str
    command: List[str]
    env: Dict[str, str]
    output_json: Path
    log_file: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--location-root", type=Path, default=LOCATION_ROOT)
    p.add_argument("--cvos-root", type=Path, default=CVOS_ROOT)
    p.add_argument("--gageo-output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", LOCATION_ROOT / "output_v3")))
    p.add_argument("--trogeo-output-root", type=Path, default=Path(os.environ.get("TROGEO_OUTPUT_ROOT", CVOS_ROOT / "saved_models")))
    p.add_argument("--eval-root", type=Path, default=Path(os.environ.get("EVAL_OUTPUT_ROOT", WRP_ROOT / "rebuttal_eval_outputs")))
    p.add_argument("--json-root", type=Path, default=Path(os.environ.get("JSON_ROOT", WRP_ROOT / "eccv_data" / "data" / "json")))
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", WRP_ROOT / "eccv_data" / "data" / "urban")))
    p.add_argument("--checkpoint-dir", type=Path, default=Path(os.environ.get("CHECKPOINT_DIR", WRP_ROOT / "checkpoints_offline")))
    p.add_argument("--trogeo-sam-checkpoint", type=Path, default=Path(os.environ["TROGEO_SAM_CHECKPOINT"]) if os.environ.get("TROGEO_SAM_CHECKPOINT") else None)
    p.add_argument("--g2d-root-dir", type=Path, default=Path(os.environ.get("G2D_ROOT_DIR", WRP_ROOT / "University-Release")))
    p.add_argument("--g2d-unseen-json", type=Path, default=Path(os.environ.get("G2D_UNSEEN_JSON", "")) if os.environ.get("G2D_UNSEEN_JSON") else None)
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "auto"), help="'auto' or comma-separated GPU ids.")
    p.add_argument("--python", default=sys.executable, help="Python executable for GAGeo evaluations.")
    p.add_argument("--trogeo-python", default=sys.executable, help="Python executable for TROGeo evaluations.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--progress-interval",
        type=float,
        default=float(os.environ.get("EVAL_PROGRESS_INTERVAL", "30")),
        help="Seconds between per-job progress summaries printed to the terminal.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--best-only", action="store_true", help="Evaluate only best checkpoints for each experiment.")
    p.add_argument("--include", nargs="*", default=None, help="Optional experiment-name filter.")
    return p.parse_args()


def visible_gpus(spec: str) -> List[str]:
    if spec and spec != "auto":
        return [x.strip() for x in spec.split(",") if x.strip()]
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        ids = [line.strip() for line in proc.stdout.splitlines() if line.strip().isdigit()]
        if ids:
            return ids
    except Exception:
        pass
    return ["0"]


def checkpoint_sort_key(path: Path) -> tuple:
    name = path.name
    nums = [int(x) for x in re.findall(r"\d+", name)]
    rank = 0 if name == "best" or "best" in name else 1
    return rank, nums[-1] if nums else -1, name


def is_accelerate_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    candidates = [
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "pytorch_model" / "mp_rank_00_model_states.pt",
        path / "mp_rank_00_model_states.pt",
    ]
    return any(x.exists() for x in candidates)


def discover_gageo_checkpoints(exp_dir: Path) -> List[Path]:
    if not exp_dir.exists():
        return []
    found: List[Path] = []
    for child in exp_dir.iterdir():
        if child.is_dir() and is_accelerate_checkpoint_dir(child):
            found.append(child)
        elif child.is_file() and child.suffix in {".pt", ".pth", ".bin", ".safetensors"}:
            found.append(child)
    for pattern in ("epoch_*", "checkpoint_*", "step_*", "best"):
        for child in exp_dir.glob(pattern):
            if child.is_dir() and is_accelerate_checkpoint_dir(child) and child not in found:
                found.append(child)
    return sorted(found, key=checkpoint_sort_key)


def discover_trogeo_checkpoints(exp_dir: Path) -> List[Path]:
    if not exp_dir.exists():
        return []
    files: List[Path] = []
    for pattern in ("*.pth.tar", "*.pth", "*.pt"):
        files.extend(exp_dir.glob(pattern))
    if not files:
        for pattern in ("*.pth.tar", "*.pth", "*.pt"):
            files.extend(exp_dir.rglob(pattern))
    return sorted(set(files), key=checkpoint_sort_key)


def filter_best_checkpoints(checkpoints: List[Path]) -> List[Path]:
    """Keep only checkpoints whose filename/directory name identifies a best model."""
    best = [ckpt for ckpt in checkpoints if "best" in ckpt.name.lower()]
    return best if best else checkpoints[:1]


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name)


def gageo_config(args: argparse.Namespace, exp_name: str, exp_dir: Path) -> Path:
    local = exp_dir / "config.yaml"
    if local.exists():
        return local
    fallback = args.location_root / "configs" / f"{exp_name}.yaml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No config.yaml or configs/{exp_name}.yaml for {exp_name}")


def base_env(args: argparse.Namespace, gpu: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "JSON_ROOT": str(args.json_root),
            "DATA_ROOT": str(args.data_root),
            "CHECKPOINT_DIR": str(args.checkpoint_dir),
            "OUTPUT_ROOT": str(args.gageo_output_root),
            "MPLCONFIGDIR": str(args.eval_root / "_matplotlib"),
            "HF_HOME": env.get("HF_HOME", str(WRP_ROOT / ".cache" / "huggingface")),
            "TORCH_HOME": env.get("TORCH_HOME", str(WRP_ROOT / ".cache" / "torch")),
        }
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def make_gageo_cmaloc_job(
    args: argparse.Namespace,
    exp_name: str,
    checkpoint: Path,
    view_subset: str,
    setting: str,
) -> EvalJob:
    exp_dir = args.gageo_output_root / exp_name
    out_dir = args.eval_root / exp_name / safe_name(checkpoint) / setting
    out_json = out_dir / "metrics.json"
    log_file = out_dir / "eval.log"
    cmd = [
        args.python,
        str(args.location_root / "evaluate_custom_v2.py"),
        "--config",
        str(gageo_config(args, exp_name, exp_dir)),
        "--checkpoint",
        str(checkpoint),
        "--image_root",
        str(args.data_root),
        "--splits",
        "unseen_test",
        "--prompt_types",
        "point",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--skip_sam",
        "--gpu",
        "0",
        "--view_subset",
        view_subset,
        "--save_json",
        str(out_json),
    ]
    return EvalJob(exp_name, checkpoint, "gageo_cmaloc", setting, cmd, base_env(args), out_json, log_file)


def make_gageo_g2d_job(args: argparse.Namespace, exp_name: str, checkpoint: Path) -> Optional[EvalJob]:
    if args.g2d_unseen_json is None:
        return None
    exp_dir = args.gageo_output_root / exp_name
    out_dir = args.eval_root / exp_name / safe_name(checkpoint) / "g2d_unseen"
    out_json = out_dir / "metrics.json"
    log_file = out_dir / "eval.log"
    cmd = [
        args.python,
        str(args.location_root / "evaluate_zero_shot_ground_to_drone.py"),
        "--triplet_json",
        str(args.g2d_unseen_json),
        "--root_dir",
        str(args.g2d_root_dir),
        "--config",
        str(gageo_config(args, exp_name, exp_dir)),
        "--checkpoint",
        str(checkpoint),
        "--img_size",
        "518",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--gpu",
        "0",
        "--save_json",
        str(out_json),
    ]
    return EvalJob(exp_name, checkpoint, "gageo_g2d", "g2d_unseen", cmd, base_env(args), out_json, log_file)


def make_trogeo_job(args: argparse.Namespace, checkpoint: Path, view_subset: str, setting: str) -> EvalJob:
    exp_name = "trogeo_pi3"
    out_dir = args.eval_root / exp_name / safe_name(checkpoint) / setting
    out_json = out_dir / "metrics.json"
    log_file = out_dir / "eval.log"
    sam_checkpoint = resolve_trogeo_sam_checkpoint(args)
    cmd = [
        args.trogeo_python,
        str(args.cvos_root / "scripts" / "evaluate_trogeo_pi3_rebuttal.py"),
        "--checkpoint",
        str(checkpoint),
        "--json",
        str(args.json_root / "unseen_test.json"),
        "--data_root",
        str(args.data_root),
        "--view_subset",
        view_subset,
        "--backbone_type",
        "pi3",
        "--pi3_pretrain",
        str(args.checkpoint_dir / "pi3_model.safetensors"),
        "--sam_checkpoint",
        str(sam_checkpoint),
        "--img_size",
        "518",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--save_json",
        str(out_json),
    ]
    env = base_env(args)
    env["LOCATION_DIR"] = str(args.location_root)
    return EvalJob(exp_name, checkpoint, "trogeo", setting, cmd, env, out_json, log_file)


def resolve_trogeo_sam_checkpoint(args: argparse.Namespace) -> Path:
    """Resolve the SAM1 ViT-H checkpoint used by TROGeo's SPS stage."""
    candidates = []
    if args.trogeo_sam_checkpoint is not None:
        candidates.append(args.trogeo_sam_checkpoint)
    candidates.extend(
        [
            args.checkpoint_dir / "sam_vit_h_4b8939.pth",
            args.cvos_root / "segment_anything" / "weights" / "sam_vit_h_4b8939.pth",
            WRP_ROOT / "baseline" / "CVOS-Code" / "segment_anything" / "weights" / "sam_vit_h_4b8939.pth",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "TROGeo SPS evaluation needs SAM1 ViT-H weights. Set TROGEO_SAM_CHECKPOINT "
        "or place sam_vit_h_4b8939.pth under CHECKPOINT_DIR."
    )


def collect_jobs(args: argparse.Namespace) -> List[EvalJob]:
    jobs: List[EvalJob] = []
    include = set(args.include) if args.include else None

    gageo_cmaloc = ["gageo_dinov2_vit_b16_joint", "gageo_dinov2_vit_h14_joint"]
    for exp in gageo_cmaloc:
        if include and exp not in include:
            continue
        ckpts = discover_gageo_checkpoints(args.gageo_output_root / exp)
        if args.best_only:
            ckpts = filter_best_checkpoints(ckpts)
        for ckpt in ckpts:
            jobs.append(make_gageo_cmaloc_job(args, exp, ckpt, "drone_to_satellite", "d2s_unseen"))
            jobs.append(make_gageo_cmaloc_job(args, exp, ckpt, "ground_to_satellite", "g2s_unseen"))

    gageo_g2d = ["gageo_pi3_frame_pos_cmaloc", "gageo_moco_q4096"]
    for exp in gageo_g2d:
        if include and exp not in include:
            continue
        ckpts = discover_gageo_checkpoints(args.gageo_output_root / exp)
        if args.best_only:
            ckpts = filter_best_checkpoints(ckpts)
        for ckpt in ckpts:
            job = make_gageo_g2d_job(args, exp, ckpt)
            if job is not None:
                jobs.append(job)

    if include is None or "trogeo_pi3" in include:
        ckpts = discover_trogeo_checkpoints(args.trogeo_output_root)
        if args.best_only:
            ckpts = filter_best_checkpoints(ckpts)
        for ckpt in ckpts:
            jobs.append(make_trogeo_job(args, ckpt, "drone_to_satellite", "d2s_unseen"))
            jobs.append(make_trogeo_job(args, ckpt, "ground_to_satellite", "g2s_unseen"))

    return jobs


def latest_progress_line(log_file: Path) -> str:
    """Return the newest useful progress/error line from a running eval log."""
    if not log_file.exists():
        return ""
    try:
        with log_file.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 20000))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""

    # tqdm usually rewrites one carriage-returned line; split on both forms.
    text = text.replace("\r", "\n")
    ansi_re = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    lines = [ansi_re.sub("", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    progress_tokens = (
        "Evaluating:",
        "TROGeo eval",
        "Val Epoch",
        "Zero-shot",
        "Loaded checkpoint",
        "Traceback",
        "RuntimeError",
        "Error",
    )
    command_tokens = (
        "CUDA_VISIBLE_DEVICES=",
        " evaluate_custom_v2.py ",
        " evaluate_zero_shot_ground_to_drone.py ",
        " evaluate_trogeo_pi3_rebuttal.py ",
    )
    for line in reversed(lines):
        if any(token in line for token in command_tokens):
            continue
        if any(token in line for token in progress_tokens) or "%" in line:
            return line[-220:]
    return ""


def checkpoint_label(path: Path) -> str:
    parent = path.parent.name if path.parent.name else ""
    return f"{parent}/{path.name}" if parent else path.name


def run_job(job: EvalJob, gpu: str, dry_run: bool, progress_interval: float = 30.0) -> Dict[str, Any]:
    job.output_json.parent.mkdir(parents=True, exist_ok=True)
    env = dict(job.env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd_str = " ".join(str(x) for x in job.command)
    if dry_run:
        return {
            "experiment": job.experiment,
            "checkpoint": str(job.checkpoint),
            "setting": job.setting,
            "status": "dry_run",
            "gpu": gpu,
            "command": cmd_str,
            "output_json": str(job.output_json),
        }
    with job.log_file.open("w", encoding="utf-8") as log:
        log.write(f"CUDA_VISIBLE_DEVICES={gpu} {cmd_str}\n\n")
        log.flush()
        proc = subprocess.Popen(
            job.command,
            cwd=str(LOCATION_ROOT if job.family.startswith("gageo") else CVOS_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        last_report = 0.0
        while True:
            returncode = proc.poll()
            now = time.time()
            if returncode is not None or now - last_report >= progress_interval:
                progress = latest_progress_line(job.log_file)
                if progress:
                    print(
                        f"[GPU {gpu}] {job.experiment} {job.setting} "
                        f"{checkpoint_label(job.checkpoint)} | {progress}",
                        flush=True,
                    )
                last_report = now
            if returncode is not None:
                break
            time.sleep(min(max(progress_interval / 3.0, 2.0), 10.0))
    row = {
        "experiment": job.experiment,
        "checkpoint": str(job.checkpoint),
        "setting": job.setting,
        "status": "ok" if proc.returncode == 0 else f"failed:{proc.returncode}",
        "gpu": gpu,
        "log": str(job.log_file),
        "output_json": str(job.output_json),
    }
    if proc.returncode == 0 and job.output_json.exists():
        row.update(extract_metrics(job))
    return row


def extract_metrics(job: EvalJob) -> Dict[str, Any]:
    data = json.loads(job.output_json.read_text(encoding="utf-8"))
    if job.family == "gageo_cmaloc":
        overall = data["unseen_test"]["point"]["overall"]
        miou = overall.get("model_miou", overall.get("det_miou"))
        return {
            "mAcc_percent": overall.get("det_avg_acc", 0.0) * 100.0,
            "mIoU_percent": None if miou is None else miou * 100.0,
            "count": overall.get("count"),
        }
    if job.family == "gageo_g2d":
        metrics = data["metrics"]
        miou = metrics.get("seg_model_miou")
        return {
            "mAcc_percent": metrics.get("avg_acc_50_95", 0.0) * 100.0,
            "mIoU_percent": None if miou is None else miou * 100.0,
            "count": metrics.get("count"),
        }
    if job.family == "trogeo":
        metrics = data["metrics"]
        return {
            "mAcc_percent": metrics.get("macc_percent"),
            "mIoU_percent": metrics.get("miou_percent"),
            "count": metrics.get("count"),
        }
    return {}


def write_summary(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [
        "experiment",
        "setting",
        "checkpoint",
        "status",
        "gpu",
        "mAcc_percent",
        "mIoU_percent",
        "count",
        "output_json",
        "log",
        "command",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV : {csv_path}")


def main() -> None:
    args = parse_args()
    jobs = collect_jobs(args)
    gpus = visible_gpus(args.gpus)
    print(f"Discovered {len(jobs)} eval jobs; GPUs={','.join(gpus)}")
    if not jobs:
        print("No checkpoints found yet. Re-run after checkpoint directories are copied back.")
        write_summary([], args.eval_root)
        return

    if args.dry_run:
        rows = [run_job(job, gpus[i % len(gpus)], dry_run=True) for i, job in enumerate(jobs)]
        for row in rows:
            print(f"[DRY] GPU {row['gpu']} {row['experiment']} {row['setting']} {row['checkpoint']}")
        write_summary(rows, args.eval_root)
        return

    work_q: queue.Queue[EvalJob] = queue.Queue()
    for job in jobs:
        work_q.put(job)
    rows: List[Dict[str, Any]] = []
    rows_lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                job = work_q.get_nowait()
            except queue.Empty:
                return
            print(f"[GPU {gpu}] {job.experiment} {job.setting} {job.checkpoint}")
            row = run_job(job, gpu, dry_run=False, progress_interval=args.progress_interval)
            with rows_lock:
                rows.append(row)
                write_summary(rows, args.eval_root)
                metric_msg = ""
                if row.get("mAcc_percent") is not None and row.get("mIoU_percent") is not None:
                    metric_msg = f" mAcc={row['mAcc_percent']:.2f} mIoU={row['mIoU_percent']:.2f}"
                print(
                    f"[DONE GPU {gpu}] {row['status']} {job.experiment} {job.setting} "
                    f"{checkpoint_label(job.checkpoint)}{metric_msg}",
                    flush=True,
                )
            work_q.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows.sort(key=lambda x: (x.get("experiment", ""), x.get("checkpoint", ""), x.get("setting", "")))
    write_summary(rows, args.eval_root)


if __name__ == "__main__":
    main()
