#!/usr/bin/env python3
"""
统计已保存模型的参数量与磁盘大小。

支持：
1) 普通 checkpoint 文件（.pt/.pth/.bin/.safetensors）
2) DeepSpeed/Accelerate 保存目录（如 output_v2/best）

示例：
  python scripts/inspect_saved_model_params.py \
        --checkpoint ${ROOT_DIR}/${WORKSPACE_NAME}/output_v2/best
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
import os
import torch


ROOT_DIR = os.environ.get("ROOT_DIR", "/data/home/scxi704/run/xhj")
WORKSPACE_NAME = os.environ.get("WORKSPACE_NAME", "location_v4")
WORKSPACE_DIR = Path(ROOT_DIR) / WORKSPACE_NAME


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{n} B"


def resolve_checkpoint(path: Path) -> Path:
    """将目录解析到真实模型文件。"""
    if path.is_file():
        return path

    # DeepSpeed/Accelerate 常见路径
    candidates = [
        path / "pytorch_model" / "mp_rank_00_model_states.pt",
        path / "mp_rank_00_model_states.pt",
        path / "pytorch_model.bin",
        path / "model.safetensors",
    ]
    for c in candidates:
        if c.exists():
            return c

    # 兜底：找目录下第一个可能的模型文件
    for p in path.rglob("*"):
        if p.suffix in {".pt", ".pth", ".bin", ".safetensors"}:
            return p

    raise FileNotFoundError(f"未在 {path} 下找到模型文件")


def extract_state_dict(obj):
    """从常见 checkpoint 结构提取 state_dict。"""
    if isinstance(obj, dict):
        for k in ["module", "model", "state_dict", "model_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k], k
        # 有些文件本身就是 state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, "root"
    raise ValueError("无法识别 checkpoint 结构，请检查文件内容")


def load_any(path: Path):
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("读取 .safetensors 需要安装 safetensors") from e
        return load_file(str(path))

    return torch.load(str(path), map_location="cpu")


def count_params(state_dict: dict[str, torch.Tensor]):
    total = 0
    trainable_like = 0  # checkpoint里无 requires_grad，这里仅等于 total
    dtype_counter = Counter()
    for _, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        n = v.numel()
        total += n
        trainable_like += n
        dtype_counter[str(v.dtype)] += n
    return total, trainable_like, dtype_counter


def main():
    ap = argparse.ArgumentParser(description="统计保存模型参数量")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=str(WORKSPACE_DIR / "output_v2" / "best"),
        help="checkpoint 文件或目录路径",
    )
    args = ap.parse_args()

    ckpt_input = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_input.exists():
        raise FileNotFoundError(f"路径不存在: {ckpt_input}")

    ckpt_file = resolve_checkpoint(ckpt_input)
    obj = load_any(ckpt_file)
    state_dict, source_key = extract_state_dict(obj)

    total, trainable_like, dtype_counter = count_params(state_dict)
    file_size = ckpt_file.stat().st_size

    print("=" * 70)
    print(f"Checkpoint 输入路径 : {ckpt_input}")
    print(f"实际读取文件       : {ckpt_file}")
    print(f"提取字典来源       : {source_key}")
    print("-" * 70)
    print(f"参数总量 (numel)    : {total:,}  ({total/1e6:.3f} M)")
    print(f"可训练参数(近似)    : {trainable_like:,}  ({trainable_like/1e6:.3f} M)")
    print(f"模型文件大小        : {file_size:,} bytes ({human_bytes(file_size)})")
    print("-" * 70)
    print("dtype 分布（按参数个数）:")
    for dt, n in dtype_counter.most_common():
        print(f"  {dt:>14}: {n:,} ({n/1e6:.3f} M)")
    print("=" * 70)


if __name__ == "__main__":
    main()
