"""Live view of a training run: reads <run>/progress.jsonl (written at every report) and prints status, ETA and alarms.

    python3 -m h.monitor h/runs/run1            # once
    python3 -m h.monitor h/runs/run1 --watch    # refresh every 10 s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROWS = 8
ZERO_COLLAPSE_NORM = 0.5
REFRESH_SECONDS = 10


def status(run: Path) -> str:
    path = run / "progress.jsonl"
    if not path.exists():
        return f"{run}: no progress.jsonl yet (phases 0-2 print to the terminal; phase 3 writes here at every report)"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    last = rows[-1]
    done, total = last["epoch"], last["total_epochs"]
    eta = last["seconds"] / max(done, 1) * (total - done)
    best = min(rows, key=lambda r: r["rel_error"])
    lines = [f"{run}  epoch {done}/{total}  elapsed {last['seconds'] / 60:.1f} min  ETA {eta / 60:.1f} min",
             f"{'epoch':>7} {'rel_err':>9} {'norm':>6} {'herm':>8} {'delta':>8} {'lambda':>9}"]
    for r in rows[-ROWS:]:
        lines.append(f"{r['epoch']:>7} {r['rel_error']:>9.4f} {r['norm_ratio']:>6.3f} {r['hermiticity_defect']:>8.4f} {r['delta']:>8.4f} {r['lambda']:>9.1e}")
    lines.append(f"best rel_error {best['rel_error']:.4f} at epoch {best['epoch']}")
    if last["norm_ratio"] < ZERO_COLLAPSE_NORM:
        lines.append("ALARM: output norm collapsed, the network is drifting to the zero map")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)
    while True:
        print(status(Path(args.run)), flush=True)
        if not args.watch:
            return 0
        time.sleep(REFRESH_SECONDS)
        print()


if __name__ == "__main__":
    sys.exit(main())
