"""Adaptive lambda pressure.

The Maxwell/HPU protocol adds lambda * delta to the loss, delta being the distance of the parameters to the nearest
lattice point, with lambda growing by decades. With a plain MSE that pressure wins against a loss the zero map already
satisfies, and the weights collapse. Here lambda only grows while the *relative* operator error stays below the accuracy
target; if it exceeds the tolerance, lambda backs off. The final lambda is the largest pressure compatible with a
network that still computes H.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import HConfig


class AdaptiveLambda:
    def __init__(self, config: HConfig):
        self.config = config
        self.value = np.float64(config.lambda_initial)
        self.history: list[tuple[int, float, float]] = []

    def discretisation(self, model: nn.Module) -> torch.Tensor:
        step = self.config.lattice_step
        total = torch.zeros((), dtype=torch.float64)
        count = 0
        for p in model.parameters():
            if p.numel():
                q = p.double() / step
                total = total + ((q - q.round()) ** 2).sum()
                count += p.numel()
        return total / max(count, 1)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        return self.discretisation(model) * float(self.value)

    def update(self, epoch: int, rel_error: float) -> str:
        """Back off immediately when accuracy is lost; grow at interval boundaries only while accuracy is high."""
        cfg = self.config
        action = "hold"
        if rel_error > cfg.accuracy_tolerance:
            self.value = max(np.float64(cfg.lambda_floor), self.value / np.float64(cfg.lambda_backoff))
            action = "backoff"
        elif epoch % cfg.lambda_interval_epochs == 0 and rel_error < cfg.accuracy_target:
            self.value = min(np.float64(cfg.lambda_max), self.value * np.float64(cfg.lambda_growth))
            action = "grow"
        if action != "hold":
            self.history.append((epoch, float(self.value), rel_error))
        return action
