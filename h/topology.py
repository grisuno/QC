"""Topological diagnostics of the spectral kernels (definitions taken from the HPU-Core / Schrodinger tools).

- winding: phase circulation of the channel-averaged complex kernel around the boundary of the torus frequency
  grid, as in check_fase_berry.py (unwrapped, divided by 2 pi).
- berry trajectory: discrete Berry connection along the checkpoint sequence in parameter space,
  A_n = arg <theta_{n-1} | theta_n> with theta the normalised flattened complex kernels, as in
  berry_phase_calculator.py. Needs at least two checkpoints.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch


def _kernels(state: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("spectral_layers.")})
    return [torch.complex(state[f"spectral_layers.{i}.kernel_real"], state[f"spectral_layers.{i}.kernel_imag"]) for i in layers]


def boundary_winding(kernel: torch.Tensor) -> float:
    phase = torch.angle(kernel.mean(dim=(0, 1))).numpy()
    loop = np.concatenate([phase[0, :], phase[1:-1, -1], phase[-1, ::-1], phase[-2:0:-1, 0]])
    steps = np.diff(np.append(loop, loop[0]))
    steps = (steps + math.pi) % (2 * math.pi) - math.pi
    return float(steps.sum() / (2 * math.pi))


def kernel_report(state: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
    out = []
    for index, kernel in enumerate(_kernels(state)):
        magnitude = kernel.abs()
        out.append({
            "layer": index,
            "winding": boundary_winding(kernel),
            "mean_magnitude": float(magnitude.mean()),
            "near_zero_fraction": float((magnitude < 1e-3).float().mean()),
            "phase_coherence": float(torch.abs(torch.exp(1j * torch.angle(kernel)).mean())),
        })
    return out


def berry_trajectory(states: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    flats = [torch.cat([k.flatten() for k in _kernels(s)]) for s in states]
    phases = []
    for prev, cur in zip(flats[:-1], flats[1:]):
        a = prev / (prev.norm() + 1e-12)
        b = cur / (cur.norm() + 1e-12)
        phases.append(float(torch.angle((a.conj() * b).sum())))
    total = float(np.sum(phases)) if phases else 0.0
    return {"total_berry_phase": total, "mod_2pi": total % (2 * math.pi), "winding": total / (2 * math.pi), "steps": len(phases)}
