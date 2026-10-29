"""Full protocol: phase 0 (GOE/GUE ratio) -> 1 (batch size) -> 2 (seed mining) -> 3 (long training with lambda pressure).

    python3 -m h.train --out h/runs/run1                       # defaults: a long run
    python3 -m h.train --out h/runs/quick --quick               # minutes, for smoke tests
    python3 -m h.train --out h/runs/x --skip-mining --seed 7 --batch 64 --epochs 4000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from .config import HConfig
from .ensembles import scan_imaginary_ratio
from .mining import mine_seeds, prospect_batch_size
from .trainer import fit


def quick(config: HConfig) -> HConfig:
    return replace(config, epochs=300, batch_candidates=[32, 64], batch_prospect_epochs=40, mining_attempts=3,
                   mining_prospect_epochs=40, mining_report_interval=20, lambda_interval_epochs=40, checkpoint_every_epochs=100,
                   log_every_epochs=20, train_samples=1024, val_samples=256, ensemble_size=96, ensemble_matrices=4)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--target-ensemble", choices=["goe", "gue"], default="gue")
    p.add_argument("--ratio", type=float)
    p.add_argument("--batch", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--skip-mining", action="store_true")
    p.add_argument("--no-lambda", action="store_true")
    p.add_argument("--export", help="also copy best.pth here (e.g. hamiltonian.pth for quantum_computer)")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    config = HConfig()
    if args.quick:
        config = quick(config)
    if args.epochs:
        config = replace(config, epochs=args.epochs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"config": json.loads(config.to_json())}

    ratio = args.ratio
    if ratio is None:
        print("Phase 0: GOE/GUE imaginary ratio")
        scan = scan_imaginary_ratio(config, args.target_ensemble)
        ratio = scan["best_ratio"]
        report["phase0"] = scan
        print(f"  best ratio {ratio} (target {args.target_ensemble})")
    batch = args.batch
    if batch is None and not args.skip_mining:
        print("Phase 1: batch size prospecting")
        found = prospect_batch_size(config, ratio)
        batch, report["phase1"] = found["best_batch"], found
    batch = batch or 64
    seed = args.seed
    if seed is None and not args.skip_mining:
        print("Phase 2: seed mining")
        found = mine_seeds(config, batch, ratio)
        seed, report["phase2"] = found["best_seed"], found
    seed = seed or config.mining_start_seed
    print(f"Phase 3: long training  seed={seed} batch={batch} ratio={ratio} epochs={config.epochs} lambda={'off' if args.no_lambda else 'adaptive'}")
    result = fit(config, seed, batch, ratio, config.epochs, out_dir=out, use_lambda=not args.no_lambda, log=print)
    report["phase3"] = {"seed": seed, "batch": batch, "ratio": ratio, "final": result["final"], "best": {k: v for k, v in result["best"].items() if k != "state"},
                        "lambda": result["lambda"], "seconds": result["seconds"]}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if args.export and (out / "best.pth").exists():
        shutil.copy(out / "best.pth", args.export)
        print(f"exported best.pth -> {args.export}")
    print(f"done: best rel error {result['best']['rel_error']:.4f} at epoch {result['best']['epoch']}  (report: {out / 'report.json'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
