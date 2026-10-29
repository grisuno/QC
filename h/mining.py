"""Maxwell-style protocol: phase 1 batch-size prospecting and phase 2 seed mining (phase 0 lives in ensembles.py)."""

from __future__ import annotations

from typing import Callable, Dict

from .config import HConfig
from .trainer import fit


def prospect_batch_size(config: HConfig, imaginary_ratio: float, log: Callable[[str], None] = print) -> Dict[str, object]:
    """Train briefly at every candidate batch size with the same seed; best = lowest relative error, ties broken by delta."""
    rows = {}
    for batch in config.batch_candidates:
        result = fit(config, config.batch_prospect_seed, batch, imaginary_ratio, config.batch_prospect_epochs,
                     report_every=config.batch_prospect_epochs)
        last = result["trajectory"][-1]
        rows[batch] = {"rel_error": last["rel_error"], "delta": last["delta"], "seconds": result["seconds"],
                       "hermiticity_defect": last["hermiticity_defect"]}
        log(f"  batch {batch:>4}: rel={last['rel_error']:.4f} delta={last['delta']:.4f} herm={last['hermiticity_defect']:.3f} [{result['seconds']:.0f}s]")
    best = min(rows, key=lambda b: (round(rows[b]["rel_error"], 3), rows[b]["delta"]))
    return {"best_batch": best, "rows": rows}


def mine_seeds(config: HConfig, batch_size: int, imaginary_ratio: float, log: Callable[[str], None] = print) -> Dict[str, object]:
    """Short probe per seed. A seed is 'cooling' when its error keeps falling; best = cooling seed with the lowest error."""
    rows = {}
    for seed in range(config.mining_start_seed, config.mining_start_seed + config.mining_attempts):
        result = fit(config, seed, batch_size, imaginary_ratio, config.mining_prospect_epochs, report_every=config.mining_report_interval)
        errors = [t["rel_error"] for t in result["trajectory"]]
        velocity = (errors[-1] - errors[0]) / max(len(errors) - 1, 1) if len(errors) > 1 else 0.0
        rows[seed] = {"rel_error": errors[-1], "velocity": velocity, "cooling": velocity < 0,
                      "delta": result["trajectory"][-1]["delta"], "errors": errors}
        log(f"  seed {seed:>4}: rel={errors[-1]:.4f} v={velocity:+.4f} {'COOLING' if velocity < 0 else 'warming'} delta={rows[seed]['delta']:.4f}")
    pool = {s: r for s, r in rows.items() if r["cooling"]} or rows
    best = min(pool, key=lambda s: pool[s]["rel_error"])
    return {"best_seed": best, "rows": rows}
