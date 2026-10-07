#!/usr/bin/env python3
"""
Copy a training YAML and override values with dotted keys.

  python scripts/make_run_config.py BASE.yaml OUT.yaml model.model_path=G:/x.safetensors optimization.steps=3000

Values are parsed as YAML (numbers, lists, null, true/false work). Missing keys are created.
The result is validated against LtxTrainerConfig before it is written.
"""

import sys
from pathlib import Path

import yaml


def _set(d: dict | list, dotted: str, value: object) -> None:
    """Set a dotted key; numeric parts index into lists (validation.samples.0.prompt)."""
    keys = dotted.split(".")
    for i, k in enumerate(keys):
        last = i == len(keys) - 1
        if isinstance(d, list):
            idx = int(k)
            if idx >= len(d):
                raise KeyError(f"{dotted}: index {idx} out of range (list has {len(d)} items)")
            if last:
                d[idx] = value
            else:
                d = d[idx]
        else:
            if last:
                d[k] = value
            else:
                if not isinstance(d.get(k), (dict, list)):
                    d[k] = {}
                d = d[k]


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    base, out, *pairs = sys.argv[1:]
    with open(base, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for p in pairs:
        if "=" not in p:
            print(f"Bad override (expected key=value): {p}")
            sys.exit(1)
        k, v = p.split("=", 1)
        _set(cfg, k.strip(), yaml.safe_load(v) if v.strip() != "" else None)

    from ltx_trainer.config import LtxTrainerConfig  # noqa: PLC0415

    LtxTrainerConfig(**cfg)  # raises with a readable message if something is wrong

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    print(f"Run config written: {out}")


if __name__ == "__main__":
    main()
