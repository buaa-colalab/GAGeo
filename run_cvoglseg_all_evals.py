#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _run(cmd):
    print("[CMD]", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def _safe_load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ws = Path("/data/home/scxi704/run/xhj/location_v3")
    p = argparse.ArgumentParser("Run v3 + baselines on CVOGL-Seg (online)")
    p.add_argument("--output_dir", type=str, default=str(ws / "output_v3" / "cvoglseg_eval"))
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--sam_checkpoint", type=str, default="")
    p.add_argument("--cvogl_root", type=str, default="/data/home/scxi704/run/baseline/CVOS-Code/dataset/CVOGL")
    p.add_argument("--cvoglseg_root", type=str, default="/data/home/scxi704/run/baseline/CVOS-Code/dataset/CVOGL-Seg")
    p.add_argument("--v3_config", type=str, default=str(ws / "output_v3" / "ablation_4_all_on" / "config.yaml"))
    p.add_argument("--v3_checkpoint", type=str, default=str(ws / "output_v3" / "ablation_4_all_on" / "best"))
    p.add_argument("--ckpt_transgeo", type=str, default="")
    p.add_argument("--ckpt_l2ltr", type=str, default="")
    p.add_argument("--ckpt_safa", type=str, default="")
    p.add_argument("--ckpt_rknet", type=str, default="")
    p.add_argument("--ckpt_sample4geo", type=str, default="")
    p.add_argument("--ckpt_cvos", type=str, default="")
    p.add_argument("--ckpt_detgeo", type=str, default="")
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    # 1) v3
    v3_json = out_dir / "v3_online.json"
    rc = _run(
        [
            "python",
            str(ws / "evaluate_cvoglseg_online_v3.py"),
            "--config",
            args.v3_config,
            "--checkpoint",
            args.v3_checkpoint,
            "--gpu",
            args.gpu,
            "--cvogl_root",
            args.cvogl_root,
            "--cvoglseg_root",
            args.cvoglseg_root,
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--sam_checkpoint",
            args.sam_checkpoint,
            "--save_json",
            str(v3_json),
        ]
    )
    summary["v3"] = {"return_code": rc, "result_file": str(v3_json), "results": _safe_load_json(v3_json)}

    # 2) retrieval-like baselines in custom_detseg
    retrieval_models = {
        "transgeo": args.ckpt_transgeo,
        "l2ltr": args.ckpt_l2ltr,
        "safa": args.ckpt_safa,
        "rknet": args.ckpt_rknet,
        "sample4geo": args.ckpt_sample4geo,
    }
    eval_detseg = "/data/home/scxi704/run/baseline/custom_detseg/eval_baseline_detseg.py"
    for model_name, ckpt in retrieval_models.items():
        if not ckpt:
            summary[f"baseline/{model_name}"] = {"skipped": True, "reason": "checkpoint not provided"}
            continue
        out_json = out_dir / f"baseline_{model_name}.json"
        rc = _run(
            [
                "python",
                eval_detseg,
                "--baseline",
                model_name,
                "--checkpoint",
                ckpt,
                "--img_size",
                str(args.img_size),
                "--gpu",
                args.gpu,
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--dataset_mode",
                "cvogl_seg_online",
                "--cvogl_root",
                args.cvogl_root,
                "--cvoglseg_root",
                args.cvoglseg_root,
                "--sam_checkpoint",
                args.sam_checkpoint,
                "--save_json",
                str(out_json),
            ]
        )
        summary[f"baseline/{model_name}"] = {
            "return_code": rc,
            "result_file": str(out_json),
            "results": _safe_load_json(out_json),
        }

    # 3) detector baselines
    det_models = {"cvos": args.ckpt_cvos, "detgeo": args.ckpt_detgeo}
    eval_det = "/data/home/scxi704/run/baseline/custom_detseg/eval_detector_baselines_online.py"
    for model_name, ckpt in det_models.items():
        if not ckpt:
            summary[f"baseline/{model_name}"] = {"skipped": True, "reason": "checkpoint not provided"}
            continue
        out_json = out_dir / f"baseline_{model_name}.json"
        rc = _run(
            [
                "python",
                eval_det,
                "--baseline",
                model_name,
                "--checkpoint",
                ckpt,
                "--img_size",
                str(args.img_size),
                "--gpu",
                args.gpu,
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--cvogl_root",
                args.cvogl_root,
                "--cvoglseg_root",
                args.cvoglseg_root,
                "--sam_checkpoint",
                args.sam_checkpoint,
                "--save_json",
                str(out_json),
            ]
        )
        summary[f"baseline/{model_name}"] = {
            "return_code": rc,
            "result_file": str(out_json),
            "results": _safe_load_json(out_json),
        }

    summary_file = out_dir / "summary_all_models.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()

