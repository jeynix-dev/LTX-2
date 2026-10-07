#!/usr/bin/env python3
"""
Compare step times of the two benchmark runs.

Reads the last N rows of each run's train_log.csv and prints per-step times plus a summary.
train_log.csv is append-only, so re-running a benchmark just adds rows - only the last N count.

  python scripts/bench_report.py G:/Lora/LTX-2/outputs --steps 30
"""

import argparse
import csv
import statistics as st
from pathlib import Path

RUNS = [("_bench_nocompile", "baseline"), ("_bench_compile", "torch.compile")]


def load(path: Path, n: int) -> list[tuple[int, float]]:
    if not path.is_file():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(float(r["step"])), float(r["step_time"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows[-n:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", help="Folder with the run folders, e.g. G:/Lora/LTX-2/outputs")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10, help="Steps ignored in the steady-state average")
    args = ap.parse_args()

    root = Path(args.outputs)
    data = {}
    for folder, label in RUNS:
        rows = load(root / folder / "train_log.csv", args.steps)
        if rows:
            data[label] = rows
        else:
            print(f"!! no data for '{label}' ({root / folder / 'train_log.csv'})")

    if not data:
        raise SystemExit("Nothing to report - run the benchmarks first.")

    # per-step table
    width = max(len(lbl) for lbl in data) + 2
    print("\nper-step time, seconds")
    print("step".rjust(5) + "".join(lbl.rjust(width) for lbl in data))
    n = max(len(v) for v in data.values())
    for i in range(n):
        line = str(i + 1).rjust(5)
        for lbl in data:
            v = data[lbl]
            line += (f"{v[i][1]:.2f}" if i < len(v) else "-").rjust(width)
        print(line)

    print("\nsummary")
    base = None
    for lbl, rows in data.items():
        t = [x[1] for x in rows]
        steady = t[args.warmup :] if len(t) > args.warmup else t
        mean_s = st.mean(steady)
        print(
            f"  {lbl:<14} all {len(t):>3} steps: mean {st.mean(t):6.2f}s  median {st.median(t):6.2f}s  "
            f"min {min(t):6.2f}s  max {max(t):6.2f}s  |  after warm-up ({len(steady)} steps): "
            f"mean {mean_s:6.2f}s  median {st.median(steady):6.2f}s"
        )
        if lbl == "baseline":
            base = mean_s

    if base and "torch.compile" in data:
        t = [x[1] for x in data["torch.compile"]]
        steady = t[args.warmup :] if len(t) > args.warmup else t
        comp = st.mean(steady)
        diff = (base - comp) / base * 100
        verb = "FASTER" if diff > 0 else "SLOWER"
        print(f"\n  torch.compile is {abs(diff):.1f}% {verb} than baseline (steady-state mean)")
        print(f"  on a 6000-step run that is {abs(base - comp) * 6000 / 3600:.1f} h difference")


if __name__ == "__main__":
    main()
