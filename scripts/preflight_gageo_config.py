#!/usr/bin/env python3
"""Fail fast on missing files before launching a GAGeo training job."""

import os
import sys

import yaml


def expand_value(value, defaults):
    value = os.path.expandvars(str(value or "").strip())
    for key, default in defaults.items():
        value = value.replace("${%s}" % key, default)
    return value


def require_file(path, label, errors):
    if path and not os.path.isfile(path):
        errors.append("%s missing: %s" % (label, path))


def require_dir(path, label, errors):
    if path and not os.path.isdir(path):
        errors.append("%s missing: %s" % (label, path))


def main():
    if len(sys.argv) != 2:
        print("Usage: preflight_gageo_config.py <config_path>", file=sys.stderr)
        return 1

    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = {
        "ROOT_DIR": os.environ.get("ROOT_DIR", "/mnt/data/wrp"),
        "WORKSPACE_NAME": os.environ.get("WORKSPACE_NAME", "location_v4"),
        "WORKSPACE_DIR": os.environ.get("WORKSPACE_DIR", "/mnt/data/wrp/location_v4"),
        "CHECKPOINT_DIR": os.environ.get("CHECKPOINT_DIR", "/mnt/data/wrp/checkpoints_offline"),
        "DATA_ROOT": os.environ.get("DATA_ROOT", "/mnt/data/wrp/eccv_data/data/urban"),
        "JSON_ROOT": os.environ.get("JSON_ROOT", "/mnt/data/wrp/eccv_data/data/json"),
        "OUTPUT_ROOT": os.environ.get("OUTPUT_ROOT", "/mnt/data/wrp/location_v4/output_v3"),
    }

    data = cfg.get("data", {})
    model = cfg.get("model", {})
    checkpoint = cfg.get("checkpoint", {})
    errors = []

    train_json = expand_value(data.get("train_json"), defaults)
    val_json = expand_value(data.get("val_json"), defaults)
    data_root = expand_value(data.get("data_root"), defaults)
    output_dir = expand_value(checkpoint.get("output_dir"), defaults)

    require_file(cfg_path, "config", errors)
    require_file(train_json, "train_json", errors)
    require_file(val_json, "val_json", errors)
    require_dir(data_root, "data_root", errors)

    for key in ("pi3_weights", "sam_weights", "joint_vit_weights"):
        value = expand_value(model.get(key), defaults)
        if value:
            require_file(value, key, errors)

    if output_dir:
        parent = os.path.dirname(output_dir.rstrip("/")) or output_dir
        if parent and not os.path.isdir(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as exc:
                errors.append("output parent not writable: %s (%s)" % (parent, exc))

    if errors:
        print("Preflight failed:", file=sys.stderr)
        for err in errors:
            print("  - " + err, file=sys.stderr)
        return 1

    print("Preflight OK: config/data/checkpoints are reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
