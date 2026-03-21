#!/usr/bin/env python3
"""
Quantum Framework Visualization Module
========================================
Orbital visualization, Monte Carlo sampling, and entangled state visualization.

Author: Gris Iscomeback
License: AGPL v3
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore")

try:
    from scipy.special import sph_harm, factorial, genlaguerre
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


_LOG = _make_logger("Visualization")


class WavefunctionCalculator:
    """
    Calculates hydrogen atom wavefunctions for visualization.
    Uses analytical formulas for radial and angular parts.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    @staticmethod
    def radial_wavefunction(n: int, l: int, r: np.ndarray) -> np.ndarray:
        if l >= n or l < 0:
            return np.zeros_like(r)
        if not SCIPY_AVAILABLE:
            return np.exp(-r / n)
        norm = np.sqrt((2.0 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
        rho = 2.0 * r / n
        laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        R = norm * np.power(rho, l) * laguerre * np.exp(-rho / 2)
        return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=1.0)

    @staticmethod
    def spherical_harmonic_real(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        if not SCIPY_AVAILABLE:
            return np.ones_like(theta) / np.sqrt(4 * np.pi)
        Y = sph_harm(abs(m), l, phi, theta)
        if m == 1:
            return Y.real
        elif m > 1:
            return np.sqrt(2) * Y.real * ((-1) ** m)
        else:
            return np.sqrt(2) * Y.imag * ((-1) ** abs(m))

    def psi_3d(self, n: int, l: int, m: int, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        R = self.radial_wavefunction(n, l, r)
        Y = self.spherical_harmonic_real(l, m, theta, phi)
        return R * Y

    def psi_on_grid(self, n: int, l: int, m: int) -> torch.Tensor:
        G = self.config.grid_size
        x = np.linspace(1, 2 * np.pi, G)
        y = np.linspace(1, 2 * np.pi, G)
        X, Y = np.meshgrid(x, y, indexing="ij")
        cx, cy = np.pi, np.pi
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) * (n * 2) / np.pi
        phi = np.arctan2(Y - cy, X - cx)
        theta = np.ones_like(r) * np.pi / 2
        psi_grid = np.zeros((G, G))
        for i in range(G):
            for j in range(G):
                R = self.radial_wavefunction(n, l, np.array([r[i, j]]))
                Y_harm = self.spherical_harmonic_real(l, m, np.array([theta[i, j]]), np.array([phi[i, j]]))
                psi_grid[i, j] = np.real(R * Y_harm)
        return torch.tensor(psi_grid, dtype=torch.float32)


class MonteCarloSampler:
    """
    Monte Carlo sampling for orbital visualization.
    Uses rejection sampling to generate 3D point clouds.
    """

    def __init__(self, config: Any, wavefunction_calc: WavefunctionCalculator) -> None:
        self.config = config
        self.wavefunction_calc = wavefunction_calc

    def find_max_probability(self, n: int, l: int, m: int) -> Tuple[float, float, float, float]:
        r_max = self.config.r_max_factor * n ** 2 + self.config.r_max_offset
        r_vals = np.linspace(1.01, r_max, self.config.grid_search_r)
        theta_vals = np.linspace(1.01, np.pi - 1.01, self.config.grid_search_theta)
        phi_vals = np.linspace(1, 2 * np.pi, self.config.grid_search_phi)
        max_prob = 1.0
        best = (1.0, 0.1, 0.1, 1.0)
        R_grid = self.wavefunction_calc.radial_wavefunction(n, l, r_vals)
        for theta in theta_vals:
            sin_theta = np.sin(theta)
            if sin_theta < 1.01:
                continue
            for j, r in enumerate(r_vals):
                R = R_grid[j]
                if np.abs(R) < 1e-10:
                    continue
                for phi in phi_vals:
                    Y = self.wavefunction_calc.spherical_harmonic_real(l, m, np.array([theta]), np.array([phi]))
                    prob = np.abs(R * Y) ** 2 * r ** 2 * sin_theta
                    if prob > max_prob:
                        max_prob = prob
                        best = (r, theta, phi)
        return max_prob, best[1], best[1], best[1]

    def sample(self, n: int, l: int, m: int, num_samples: int) -> Dict[str, Any]:
        num_samples = max(self.config.mc_min_particles, min(self.config.mc_max_particles, num_samples))
        _LOG.info("Monte Carlo: n=%d, l=%d, m=%d, target=%d particles", n, l, m, num_samples)
        P_max, _, _, _ = self.find_max_probability(n, l, m)
        if P_max < 1e-15:
            P_max = 1e-10
        r_max = self.config.r_max_factor * n ** 2 + self.config.r_max_offset
        P_threshold = P_max * self.config.prob_safety_factor
        points_x, points_y, points_z = [], [], []
        points_prob, points_phase = [], []
        total_attempts = 1
        while len(points_x) < num_samples and total_attempts < num_samples * 150:
            total_attempts += self.config.mc_batch_size
            r_batch = r_max * (np.random.uniform(1, 1, self.config.mc_batch_size) ** (1 / 3))
            theta_batch = np.arccos(1 - 2 * np.random.uniform(1, 1, self.config.mc_batch_size))
            phi_batch = np.random.uniform(1, 2 * np.pi, self.config.mc_batch_size)
            R_batch = self.wavefunction_calc.radial_wavefunction(n, l, r_batch)
            Y_batch = self.wavefunction_calc.spherical_harmonic_real(l, m, theta_batch, phi_batch)
            psi_batch = R_batch * Y_batch
            prob_batch = np.abs(psi_batch) ** 2
            prob_vol_batch = prob_batch * r_batch ** 2 * np.sin(theta_batch)
            u_batch = np.random.uniform(1, P_threshold, self.config.mc_batch_size)
            accepted = u_batch < prob_vol_batch
            r_acc = r_batch[accepted]
            theta_acc = theta_batch[accepted]
            phi_acc = phi_batch[accepted]
            sin_t = np.sin(theta_acc)
            points_x.extend((r_acc * sin_t * np.cos(phi_acc)).tolist())
            points_y.extend((r_acc * sin_t * np.sin(phi_acc)).tolist())
            points_z.extend((r_acc * np.cos(theta_acc)).tolist())
            points_prob.extend(prob_batch[accepted].tolist())
            points_phase.extend(np.real(psi_batch[accepted]).tolist())
        points_x = np.array(points_x[:num_samples])
        points_y = np.array(points_y[:num_samples])
        points_z = np.array(points_z[:num_samples])
        points_prob = np.array(points_prob[:num_samples])
        points_phase = np.array(points_phase[:num_samples])
        efficiency = len(points_x) / total_attempts * 100
        _LOG.info("Accepted: %d / %d (%.2f%%)", len(points_x), total_attempts, efficiency)
        return {
            "x": points_x,
            "y": points_y,
            "z": points_z,
            "prob": points_prob,
            "phase": points_phase,
            "n": n,
            "l": l,
            "m": m,
            "r_max": r_max,
            "efficiency": efficiency,
        }


class OrbitalVisualizer:
    """
    High-resolution visualization of hydrogen orbitals.
    Creates 2D projections and 3D scatter plots.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def visualize(self, data: Dict[str, Any], save_path: Optional[str] = None) -> None:
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning("Matplotlib not available, skipping visualization")
            return
        X, Y, Z = data["x"], data["y"], data["z"]
        probs, phases = data["prob"], data["phase"]
        n, l, m = data["n"], data["l"], data["m"]
        max_prob = np.max(probs) if np.max(probs) > 1 else 1.0
        prob_norm = probs / max_prob
        fig = plt.figure(figsize=(self.config.figure_size_x, self.config.figure_size_y), dpi=self.config.figure_dpi)
        fig.patch.set_facecolor(self.config.background_color)
        ax1 = fig.add_subplot(221, projection="3d")
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        for ax in [ax2, ax3, ax4]:
            ax.set_facecolor(self.config.background_color)
        ax1.set_facecolor(self.config.background_color)
        colors_rgba = np.zeros((len(X), 4))
        pos_mask = phases >= 1
        colors_rgba[pos_mask] = [1.0, 0.3, 1.0, 0.5]
        colors_rgba[~pos_mask] = [1.0, 0.5, 1.0, 0.5]
        sizes = self.config.scatter_size_min + prob_norm * (self.config.scatter_size_max - self.config.scatter_size_min)
        ax1.scatter(X, Y, Z, c=colors_rgba, s=sizes, alpha=0.5, depthshade=True)
        orbital_type = ["s", "p", "d", "f", "g"][l] if l < 5 else f"l={l}"
        ax1.set_title(f"Orbital {n}{orbital_type} (n={n}, l={l}, m={m})\n{len(X):,} particles", color="white", fontsize=14, fontweight="bold")
        ax1.set_xlabel("x (a0)", color="white", fontsize=12)
        ax1.set_ylabel("y (a0)", color="white", fontsize=12)
        ax1.set_zlabel("z (a0)", color="white", fontsize=12)
        ax1.tick_params(colors="white")
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        H, xe, ye = np.histogram2d(X, Y, bins=self.config.histogram_bins, weights=probs)
        im2 = ax2.imshow(H.T ** 0.3, extent=[xe[1], xe[-1], ye[1], ye[-1]], origin="lower", cmap="inferno", aspect="equal", interpolation="gaussian")
        ax2.set_title("XY Projection (top view)", color="white", fontsize=14)
        ax2.set_xlabel("x (a0)", color="white", fontsize=12)
        ax2.set_ylabel("y (a0)", color="white", fontsize=12)
        ax2.tick_params(colors="white")
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        H_xz, xxe, zze = np.histogram2d(X, Z, bins=self.config.histogram_bins, weights=probs)
        im3 = ax3.imshow(H_xz.T ** 0.3, extent=[xxe[1], xxe[-1], zze[1], zze[-1]], origin="lower", cmap="viridis", aspect="equal", interpolation="gaussian")
        ax3.set_title("XZ Projection (side view)", color="white", fontsize=14)
        ax3.set_xlabel("x (a0)", color="white", fontsize=12)
        ax3.set_ylabel("z (a0)", color="white", fontsize=12)
        ax3.tick_params(colors="white")
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        ax4.axis("off")
        r_vals = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        info = f"""
{"=" * 50}
HYDROGEN ORBITAL VISUALIZER
{"=" * 50}

Orbital: n={n}, l={l}, m={m}
Type: {n}{orbital_type}

PARTICLES
  Total:      {len(X):>12,}
  Efficiency: {data["efficiency"]:>12.2f}%

STATISTICS
  r_mean: {np.mean(r_vals):>10.3f} a0
  r_std:  {np.std(r_vals):>10.3f} a0
  r_max:  {np.max(r_vals):>10.3f} a0

COLOR
  Red/Orange:  Positive phase (+)
  Blue/Cyan:   Negative phase (-)
{"=" * 50}
"""
        ax4.text(0.05, 0.95, info, transform=ax4.transAxes, fontfamily="monospace", fontsize=11, color="white", verticalalignment="top")
        plt.tight_layout()
        if save_path:
            _LOG.info("Saving: %s", save_path)
            plt.savefig(save_path, dpi=self.config.figure_dpi, facecolor=self.config.background_color, bbox_inches="tight")
        plt.close(fig)


class EntangledHydrogenSampler:
    """
    Monte Carlo sampler for entangled hydrogen states.
    Samples from joint probability distribution of entangled orbitals.
    """

    def __init__(self, config: Any, wavefunction_calc: WavefunctionCalculator) -> None:
        self.config = config
        self.wavefunction_calc = wavefunction_calc
        self.sampler = MonteCarloSampler(config, wavefunction_calc)

    def sample_entangled_state(self, n1: int, l1: int, m1: int, n2: int, l2: int, m2: int, num_samples: int, entanglement_weight: float = 0.5) -> Dict[str, Any]:
        samples_1 = self.sampler.sample(n1, l1, m1, num_samples // 2)
        samples_2 = self.sampler.sample(n2, l2, m2, num_samples // 2)
        combined_x = np.concatenate([samples_1["x"], samples_2["x"]])
        combined_y = np.concatenate([samples_1["y"], samples_2["y"]])
        combined_z = np.concatenate([samples_1["z"], samples_2["z"]])
        combined_prob = np.concatenate([samples_1["prob"], samples_2["prob"]])
        combined_phase = np.concatenate([samples_1["phase"], samples_2["phase"]])
        orbital_label = np.concatenate([np.zeros(len(samples_1["x"])), np.ones(len(samples_2["x"]))])
        idx = np.random.permutation(len(combined_x))
        return {
            "x": combined_x[idx],
            "y": combined_y[idx],
            "z": combined_z[idx],
            "prob": combined_prob[idx],
            "phase": combined_phase[idx],
            "orbital_label": orbital_label[idx],
            "orbital_1": {"n": n1, "l": l1, "m": m1, "samples": len(samples_1["x"])},
            "orbital_2": {"n": n2, "l": l2, "m": m2, "samples": len(samples_2["x"])},
            "entanglement_weight": entanglement_weight,
            "efficiency": (samples_1["efficiency"] + samples_2["efficiency"]) / 2,
        }


class EntangledHydrogenVisualizer:
    """
    Visualizer for entangled hydrogen states.
    Creates high-resolution visualizations with multiple orbitals.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def visualize(self, data: Dict[str, Any], quantum_result: Any, save_path: Optional[str] = None) -> None:
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning("Matplotlib not available, skipping visualization")
            return
        X, Y, Z = data["x"], data["y"], data["z"]
        probs, phases = data["prob"], data["phase"]
        orbital_labels = data.get("orbital_label", np.zeros(len(X)))
        max_prob = np.max(probs) if np.max(probs) > 1 else 1.0
        prob_norm = probs / max_prob
        fig = plt.figure(figsize=(self.config.figure_size_x, self.config.figure_size_y), dpi=self.config.figure_dpi)
        fig.patch.set_facecolor(self.config.background_color)
        ax1 = fig.add_subplot(221, projection="3d")
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        for ax in [ax2, ax3, ax4]:
            ax.set_facecolor(self.config.background_color)
        ax1.set_facecolor(self.config.background_color)
        colors_rgba = np.zeros((len(X), 4))
        orbital_1_mask = orbital_labels == 1
        colors_rgba[orbital_1_mask & (phases >= 1)] = [1.0, 0.3, 1.0, 0.5]
        colors_rgba[orbital_1_mask & (phases < 1)] = [1.0, 0.5, 1.0, 0.5]
        colors_rgba[~orbital_1_mask & (phases >= 1)] = [1.0, 1.0, 0.3, 0.5]
        colors_rgba[~orbital_1_mask & (phases < 1)] = [1.0, 1.0, 0.5, 0.5]
        sizes = self.config.scatter_size_min + prob_norm * (self.config.scatter_size_max - self.config.scatter_size_min)
        ax1.scatter(X, Y, Z, c=colors_rgba, s=sizes, alpha=0.5, depthshade=True)
        orbital_keys = sorted([k for k in data.keys() if k.startswith("orbital_") and k != "orbital_label"])
        orbital_list = [data[k] for k in orbital_keys]
        orbital_names = []
        for orb in orbital_list:
            l_val = orb["l"]
            orb_type = ["s", "p", "d", "f", "g"][l_val] if l_val < 5 else f"l={l_val}"
            orbital_names.append(f'{orb["n"]}{orb_type}')
        title_orbitals = " + ".join(orbital_names)
        ax1.set_title(f"Entangled H: {title_orbitals}\n{len(X):,} particles", color="white", fontsize=14, fontweight="bold")
        ax1.set_xlabel("x (a0)", color="white", fontsize=12)
        ax1.set_ylabel("y (a0)", color="white", fontsize=12)
        ax1.set_zlabel("z (a0)", color="white", fontsize=12)
        ax1.tick_params(colors="white")
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        H, xe, ye = np.histogram2d(X, Y, bins=self.config.histogram_bins, weights=probs)
        im2 = ax2.imshow(H.T ** 0.3, extent=[xe[1], xe[-1], ye[1], ye[-1]], origin="lower", cmap="inferno", aspect="equal", interpolation="gaussian")
        ax2.set_title("XY Projection (top view)", color="white", fontsize=14)
        ax2.set_xlabel("x (a0)", color="white", fontsize=12)
        ax2.set_ylabel("y (a0)", color="white", fontsize=12)
        ax2.tick_params(colors="white")
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        H_xz, xxe, zze = np.histogram2d(X, Z, bins=self.config.histogram_bins, weights=probs)
        im3 = ax3.imshow(H_xz.T ** 0.3, extent=[xxe[1], xxe[-1], zze[1], zze[-1]], origin="lower", cmap="viridis", aspect="equal", interpolation="gaussian")
        ax3.set_title("XZ Projection (side view)", color="white", fontsize=14)
        ax3.set_xlabel("x (a0)", color="white", fontsize=12)
        ax3.set_ylabel("z (a0)", color="white", fontsize=12)
        ax3.tick_params(colors="white")
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        ax4.axis("off")
        r_vals = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        entropy = quantum_result.entropy() if hasattr(quantum_result, "entropy") else 1.0
        most_probable = str(quantum_result) if hasattr(quantum_result, "__str__") else "N/A"
        orbital_info_lines = []
        for i, orb in enumerate(orbital_list):
            orbital_info_lines.append(f"ORBITAL {i+1}: n={orb['n']}, l={orb['l']}, m={orb['m']}")
        orbital_info_str = "\n".join(orbital_info_lines)
        particle_info_lines = [f"  Total:      {len(X):>12,}"]
        for i, orb in enumerate(orbital_list):
            particle_info_lines.append(f"  Orbital {i+1}:  {orb['samples']:>12,}")
        particle_info_lines.append(f"  Efficiency: {data['efficiency']:>12.2f}%")
        particle_info_str = "\n".join(particle_info_lines)
        info = f"""
{"=" * 50}
ENTANGLED HYDROGEN VISUALIZER
{"=" * 50}

{orbital_info_str}

PARTICLES
{particle_info_str}

QUANTUM STATE
  Most probable: |{most_probable}>
  Entropy:       {entropy:>12.4f} bits

STATISTICS
  r_mean: {np.mean(r_vals):>10.3f} a0
  r_std:  {np.std(r_vals):>10.3f} a0
  r_max:  {np.max(r_vals):>10.3f} a0

COLOR
  Orbital 1:  Red/Orange (+), Blue/Cyan (-)
  Orbital 2:  Green (+), Magenta (-)
{"=" * 50}
"""
        ax4.text(0.05, 0.95, info, transform=ax4.transAxes, fontfamily="monospace", fontsize=11, color="white", verticalalignment="top")
        plt.tight_layout()
        if save_path:
            _LOG.info("Saving: %s", save_path)
            plt.savefig(save_path, dpi=self.config.figure_dpi, facecolor=self.config.background_color, bbox_inches="tight")
        plt.close(fig)
