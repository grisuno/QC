"""Phase 0: GOE <-> GUE imaginary-ratio search (Maxwell protocol).

Hermitian matrices H = A + i r B (A symmetric, B antisymmetric Gaussian) interpolate between the orthogonal (r = 0)
and unitary (r = 1) ensembles. The ratio whose level statistics best match a target ensemble sets how complex the
spectral kernels are at initialisation.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

from .config import HConfig

GOE_R_MEAN = 0.5307
GUE_R_MEAN = 0.5996
SPACING_BINS = 40
SPACING_RANGE = 4.0


def interpolated_matrix(size: int, imaginary_ratio: float, generator: torch.Generator) -> torch.Tensor:
    real = torch.randn(size, size, generator=generator)
    imag = torch.randn(size, size, generator=generator)
    real = (real + real.T) / math.sqrt(2)
    imag = (imag - imag.T) / math.sqrt(2)
    return torch.complex(real, imag * imaginary_ratio)


def unfolded_spacings(eigenvalues: np.ndarray) -> np.ndarray:
    eig = np.sort(eigenvalues.real)
    n = len(eig)
    unfolded = np.polyval(np.polyfit(eig, np.arange(1, n + 1, dtype=float), 3), eig)
    spacing = np.diff(unfolded)
    return spacing / max(spacing.mean(), 1e-12)


def wigner_surmise(s: np.ndarray, target: str) -> np.ndarray:
    if target == "goe":
        return (math.pi / 2) * s * np.exp(-math.pi * s**2 / 4)
    return (32 / math.pi**2) * s**2 * np.exp(-4 * s**2 / math.pi)


def spacing_loss(spacings: np.ndarray, target: str) -> float:
    hist, edges = np.histogram(spacings, bins=SPACING_BINS, range=(0, SPACING_RANGE), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return float(np.mean((hist - wigner_surmise(centers, target)) ** 2))


def mean_gap_ratio(eigenvalues: np.ndarray) -> float:
    """<min(s_i, s_i+1) / max(s_i, s_i+1)>: 0.386 Poisson, 0.531 GOE, 0.600 GUE. Needs no unfolding."""
    gaps = np.diff(np.sort(eigenvalues.real))
    ratio = np.minimum(gaps[1:], gaps[:-1]) / np.maximum(np.maximum(gaps[1:], gaps[:-1]), 1e-12)
    return float(ratio.mean())


def scan_imaginary_ratio(config: HConfig, target: str = "gue", seed: int = 0) -> Dict[str, object]:
    """Evaluate every candidate ratio and return the one closest to the target ensemble."""
    gen = torch.Generator().manual_seed(seed)
    reference = GUE_R_MEAN if target == "gue" else GOE_R_MEAN
    rows: List[dict] = []
    for ratio in config.ratio_candidates:
        spacings, gap_ratios = [], []
        for _ in range(config.ensemble_matrices):
            eig = torch.linalg.eigvalsh(interpolated_matrix(config.ensemble_size, ratio, gen)).numpy()
            core = eig[len(eig) // 8 : -len(eig) // 8]
            spacings.append(unfolded_spacings(core))
            gap_ratios.append(mean_gap_ratio(core))
        s_loss = spacing_loss(np.concatenate(spacings), target)
        r_loss = abs(float(np.mean(gap_ratios)) - reference)
        rows.append({"ratio": ratio, "spacing_loss": s_loss, "gap_ratio": float(np.mean(gap_ratios)),
                     "combined": config.spacing_weight * s_loss + config.correlation_weight * r_loss})
    best = min(rows, key=lambda row: row["combined"])
    return {"target": target, "best_ratio": best["ratio"], "rows": rows}
