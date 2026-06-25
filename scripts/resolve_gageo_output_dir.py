#!/usr/bin/env python3
"""Resolve the output directory from a GAGeo YAML config.

This is a standalone helper instead of `python - <<'PY'` so remote job
launchers do not need to preserve stdin for shell heredocs.
"""

import os
import sys
from pathlib import Path

import yaml


def main():
    if len(sys.argv) != 2:
        print("Usage: resolve_gageo_output_dir.py <config_path>", file=sys.stderr)
        return 1

    cfg_path = Path(sys.argv[1])
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = {
        "ROOT_DIR": os.environ.get("ROOT_DIR", str(Path.cwd().parent)),
        "WORKSPACE_NAME": os.environ.get("WORKSPACE_NAME", Path.cwd().name),
        "CHECKPOINT_DIR": os.environ.get("CHECKPOINT_DIR", str(Path.cwd() / "checkpoints_offline")),
    }
    defaults["WORKSPACE_DIR"] = os.environ.get(
        "WORKSPACE_DIR", "{}/{}".format(defaults["ROOT_DIR"], defaults["WORKSPACE_NAME"])
    )
    defaults["DATA_ROOT"] = os.environ.get("DATA_ROOT", "{}/data/urban".format(defaults["WORKSPACE_DIR"]))
    defaults["JSON_ROOT"] = os.environ.get("JSON_ROOT", "{}/data/json".format(defaults["WORKSPACE_DIR"]))
    defaults["OUTPUT_ROOT"] = os.environ.get(
        "OUTPUT_ROOT", "{}/outputs".format(defaults["WORKSPACE_DIR"])
    )

    value = str((cfg.get("checkpoint") or {}).get("output_dir") or "").strip()
    value = os.path.expandvars(value)
    for key, default in defaults.items():
        value = value.replace("${%s}" % key, default)
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
