"""Training loop shared by prospecting, seed mining and the long run. Checkpoints match the project's loader."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .config import HConfig
from .data import HamiltonianDataset, validation_set
from .lambda_pressure import AdaptiveLambda
from .metrics import evaluate, relative_loss
from .model import HamiltonianTorusNet


def build(config: HConfig, seed: int, imaginary_ratio: float) -> HamiltonianTorusNet:
    torch.manual_seed(seed)
    return HamiltonianTorusNet(config, imaginary_ratio).to(config.device)


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, config: HConfig, extra: dict):
    payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
               "metrics": metrics, "config": asdict(config), **extra}
    torch.save(payload, path)


def fit(config: HConfig, seed: int, batch_size: int, imaginary_ratio: float, epochs: int, out_dir: Optional[Path] = None,
        use_lambda: bool = False, report_every: int = 0, log: Callable[[str], None] = lambda _: None) -> Dict[str, object]:
    """Train from scratch; returns the final metrics, the trajectory and the best state."""
    model = build(config, seed, imaginary_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * config.cosine_floor)
    data = HamiltonianDataset(config, seed)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_x, val_y = validation_set(config)
    pressure = AdaptiveLambda(config) if use_lambda else None
    trajectory: List[dict] = []
    best = {"rel_error": float("inf"), "state": None, "epoch": 0, "metrics": None}
    stable_state = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoints: List[Path] = []
    started = time.time()
    interval = report_every or config.log_every_epochs
    metrics = evaluate(model, val_x, val_y, config)
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = relative_loss(model(x).reshape_as(y), y)
            if pressure is not None:
                loss = loss + pressure.penalty(model).to(loss.dtype)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        scheduler.step()
        if epoch % interval == 0 or epoch == epochs:
            model.eval()
            metrics = evaluate(model, val_x, val_y, config)
            with torch.no_grad():
                delta = float((pressure or AdaptiveLambda(config)).discretisation(model))
            row = {"epoch": epoch, "delta": delta, "lambda": float(pressure.value) if pressure else 0.0, **metrics}
            trajectory.append(row)
            sane = abs(metrics["norm_ratio"] - 1.0) < config.norm_ratio_tolerance
            if sane and metrics["rel_error"] < best["rel_error"]:
                best = {"rel_error": metrics["rel_error"], "epoch": epoch, "metrics": metrics,
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}
            if pressure is not None:
                before = float(pressure.value)
                action = pressure.update(epoch, metrics["rel_error"])
                if action == "backoff":
                    model.load_state_dict(stable_state)
                elif metrics["rel_error"] <= config.accuracy_tolerance:
                    stable_state = {k: v.clone() for k, v in model.state_dict().items()}
                if action != "hold":
                    log(f"  lambda {action}: {before:.3e} -> {float(pressure.value):.3e}")
            if out_dir is not None:
                with open(out_dir / "progress.jsonl", "a") as handle:
                    handle.write(json.dumps({**row, "seconds": time.time() - started, "total_epochs": epochs}) + "\n")
            log(f"  ep {epoch:>6} rel={metrics['rel_error']:.4f} norm={metrics['norm_ratio']:.3f} herm={metrics['hermiticity_defect']:.3f} "
                f"delta={delta:.4f} lambda={row['lambda']:.2e} [{time.time() - started:.0f}s]")
        if out_dir is not None and epoch % config.checkpoint_every_epochs == 0:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"checkpoint_epoch_{epoch}_{stamp}.pth"
            save_checkpoint(path, model, optimizer, epoch, metrics, config,
                            {"seed": seed, "batch_size": batch_size, "imaginary_ratio": imaginary_ratio,
                             "lambda_pressure": float(pressure.value) if pressure else 0.0})
            checkpoints.append(path)
            while len(checkpoints) > config.max_checkpoints:
                checkpoints.pop(0).unlink(missing_ok=True)
            shutil_copy = out_dir / "latest.pth"
            torch.save(torch.load(path, weights_only=False), shutil_copy)
    final_state = {k: v.clone() for k, v in model.state_dict().items()}
    if out_dir is not None:
        extra = {"seed": seed, "batch_size": batch_size, "imaginary_ratio": imaginary_ratio,
                 "lambda_pressure": float(pressure.value) if pressure else 0.0}
        save_checkpoint(out_dir / "latest.pth", model, optimizer, epochs, metrics, config, extra)
        if best["state"] is not None:
            model.load_state_dict(best["state"])
            save_checkpoint(out_dir / "best.pth", model, optimizer, best["epoch"], best["metrics"], config, extra)
        (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=1))
    return {"final": metrics, "best": best, "trajectory": trajectory, "final_state": final_state,
            "lambda": float(pressure.value) if pressure else 0.0, "seconds": time.time() - started}
