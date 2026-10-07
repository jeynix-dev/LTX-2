#!/usr/bin/env python3
"""
Feed train_log.csv into TensorBoard, live.

The trainer writes both TensorBoard events and a CSV. If the `tensorboard` package was not
installed when a run started, that run has only the CSV -- this script replays it into event
files so the curves show up without restarting training, and keeps following the file while
training continues.

  python scripts/csv_to_tensorboard.py G:/Lora/LTX-2/outputs          # all runs, keep watching
  python scripts/csv_to_tensorboard.py G:/Lora/LTX-2/outputs --once   # convert and exit

For each <outputs>/<run>/train_log.csv it writes <outputs>/<run>/tensorboard_csv/.
TensorBoard pointed at <outputs> then shows "<run>/tensorboard_csv" next to the trainer's own
"<run>/tensorboard" (if any).
"""

import argparse
import csv
import time
from pathlib import Path

SCALARS = {
    "loss": "train/loss",
    "loss_ema": "train/loss_ema",
    "sigma": "train/sigma",
    "lr": "train/learning_rate",
    "step_time": "train/step_time",
}


def _rows(csv_path: Path, skip: int) -> tuple[list[dict], int]:
    """Read rows past the first `skip` data rows. Returns (new_rows, total_rows_seen)."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError):
        return [], skip
    return rows[skip:], len(rows)


def _write(writer, rows: list[dict]) -> int:  # noqa: ANN001
    written = 0
    for row in rows:
        try:
            step = int(float(row["step"]))
        except (KeyError, TypeError, ValueError):
            continue
        for col, tag in SCALARS.items():
            raw = row.get(col)
            if raw in (None, "", "None"):
                continue
            try:
                writer.add_scalar(tag, float(raw), step)
            except ValueError:
                continue
        written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", help="Folder holding the run folders (e.g. G:/Lora/LTX-2/outputs)")
    ap.add_argument("--once", action="store_true", help="Convert what exists and exit")
    ap.add_argument("--interval", type=float, default=10.0, help="Seconds between polls (default 10)")
    args = ap.parse_args()

    root = Path(args.outputs)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

    writers: dict[Path, object] = {}
    seen: dict[Path, int] = {}

    try:
        while True:
            for csv_path in sorted(root.glob("*/train_log.csv")):
                if csv_path not in writers:
                    out = csv_path.parent / "tensorboard_csv"
                    writers[csv_path] = SummaryWriter(log_dir=str(out), flush_secs=10)
                    seen[csv_path] = 0
                    print(f"watching {csv_path} -> {out}", flush=True)
                new, total = _rows(csv_path, seen[csv_path])
                if new:
                    n = _write(writers[csv_path], new)
                    seen[csv_path] = total
                    writers[csv_path].flush()
                    print(f"{csv_path.parent.name}: +{n} points (total {total})", flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        for w in writers.values():
            w.close()


if __name__ == "__main__":
    main()
