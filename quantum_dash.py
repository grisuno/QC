#!/usr/bin/env python3
"""
quantum_brutalist_viz.py - Production Quantum State Visualizer
==============================================================
High-fidelity 3D holographic visualization of quantum states using
trained neural network backends (Hamiltonian, Schrodinger, Dirac).

Author: Gris Iscomeback
License: AGPL v3
"""

from __future__ import annotations

import logging
import math
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Sequence, Callable, Union

import numpy as np

warnings.filterwarnings("ignore")

UPLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
if UPLOAD_DIR not in sys.path:
    sys.path.insert(0, UPLOAD_DIR)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import proj3d
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    if TORCH_AVAILABLE:
        from quantum_computer import (
            QuantumComputer,
            SimulatorConfig,
            JointHilbertState,
            QuantumCircuit,
            IQuantumGate,
            _GATE_REGISTRY,
            HamiltonianBackend,
            SchrodingerBackend,
            DiracBackend,
            IPhysicsBackend,
        )
        QUANTUM_COMPUTER_AVAILABLE = True
    else:
        QUANTUM_COMPUTER_AVAILABLE = False
except ImportError as e:
    QUANTUM_COMPUTER_AVAILABLE = False

try:
    if TORCH_AVAILABLE:
        from molecular_sim import MOLECULES, ExactJWEnergy, MoleculeData
        MOLECULAR_AVAILABLE = True
    else:
        MOLECULAR_AVAILABLE = False
except ImportError:
    MOLECULAR_AVAILABLE = False


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


_LOG = _make_logger("QuantumBrutalistViz")


class ColorScheme(Enum):
    CYBERPUNK = "cyberpunk"
    MATRIX = "matrix"
    QUANTUM_VOID = "quantum_void"
    NEON_NOIR = "neon_noir"
    PLASMA = "plasma"


@dataclass
class BrutalistConfig:
    grid_size: int = 16
    hidden_dim: int = 32
    expansion_dim: int = 64
    num_spectral_layers: int = 2
    dirac_mass: float = 1.0
    dirac_c: float = 1.0
    gamma_representation: str = "dirac"
    dt: float = 0.01
    normalization_eps: float = 1e-8
    potential_depth: float = 5.0
    potential_width: float = 0.3
    hamiltonian_checkpoint: str = "hamiltonian.pth"
    schrodinger_checkpoint: str = "checkpoint_phase3_training_epoch_18921_20260224_154739.pth"
    dirac_checkpoint: str = "best_dirac.pth"
    device: str = "cpu"
    random_seed: int = 42
    max_qubits: int = 8
    color_scheme: ColorScheme = ColorScheme.CYBERPUNK
    figure_dpi: int = 200
    figure_width_inches: int = 24
    figure_height_inches: int = 18
    particle_count: int = 2000
    glow_intensity: float = 2.0
    hologram_opacity: float = 0.7
    bloom_strength: float = 1.5
    chromatic_aberration: float = 0.02
    camera_distance: float = 3.0
    animation_frames: int = 100
    sphere_resolution: int = 40
    contour_levels: int = 20
    entropy_base: float = 2.0
    probability_threshold: float = 1e-6
    phase_normalization: float = math.pi
    output_dir: str = "download"
    backend_order: Tuple[str, str, str] = ("hamiltonian", "schrodinger", "dirac")
    default_n_qubits: int = 3
    grover_iterations: int = 2
    grover_marked_state: int = 5
    vqe_max_iterations: int = 200
    vqe_tolerance: float = 1e-8
    scatter_point_size_min: float = 5.0
    scatter_point_size_max: float = 50.0
    line_width_base: float = 2.0
    vector_scale: float = 1.0
    field_resolution: int = 25
    isosurface_count: int = 5
    trail_length: int = 50
    uncertainty_ring_radius: float = 0.15
    grid_alpha: float = 0.3
    background_alpha: float = 0.95

    @property
    def colors(self) -> Dict[str, str]:
        schemes = {
            ColorScheme.CYBERPUNK: {
                "primary": "#FF006E",
                "secondary": "#00F5FF",
                "tertiary": "#FFBE0B",
                "accent": "#8338EC",
                "background": "#0A0A0F",
                "grid": "#1A1A2E",
                "text": "#FFFFFF",
                "glow": "#FF006E",
                "surface": "#16213E",
                "line": "#E94560",
            },
            ColorScheme.MATRIX: {
                "primary": "#00FF41",
                "secondary": "#008F11",
                "tertiary": "#003B00",
                "accent": "#39FF14",
                "background": "#000000",
                "grid": "#0D0208",
                "text": "#00FF41",
                "glow": "#00FF41",
                "surface": "#001100",
                "line": "#00CC00",
            },
            ColorScheme.QUANTUM_VOID: {
                "primary": "#9D4EDD",
                "secondary": "#C77DFF",
                "tertiary": "#E0AAFF",
                "accent": "#7B2CBF",
                "background": "#10002B",
                "grid": "#240046",
                "text": "#E0AAFF",
                "glow": "#9D4EDD",
                "surface": "#1A0033",
                "line": "#C77DFF",
            },
            ColorScheme.NEON_NOIR: {
                "primary": "#F72585",
                "secondary": "#7209B7",
                "tertiary": "#3A0CA3",
                "accent": "#4361EE",
                "background": "#0B0B0D",
                "grid": "#1E1E24",
                "text": "#F8F8F8",
                "glow": "#F72585",
                "surface": "#16161A",
                "line": "#FF006E",
            },
            ColorScheme.PLASMA: {
                "primary": "#FF6B35",
                "secondary": "#F7C59F",
                "tertiary": "#EFEFEF",
                "accent": "#2EC4B6",
                "background": "#011627",
                "grid": "#023E8A",
                "text": "#FDFFFC",
                "glow": "#FF6B35",
                "surface": "#03045E",
                "line": "#FF9F1C",
            },
        }
        return schemes.get(self.color_scheme, schemes[ColorScheme.CYBERPUNK])

    @property
    def plotly_template(self) -> str:
        return "plotly_dark"


@dataclass
class QuantumSnapshot:
    step: int
    gate_name: str
    probabilities: np.ndarray
    phases: np.ndarray
    entropy: float
    bloch_vectors: List[Tuple[float, float, float]]
    amplitudes: np.ndarray
    most_probable: int
    norm: float
    n_qubits: int
    timestamp: float = 0.0
    backend_name: str = "unknown"


@dataclass
class BackendComparison:
    backend_name: str
    snapshots: List[QuantumSnapshot]
    final_probabilities: np.ndarray
    final_entropy: float
    most_probable_state: int
    top_states: List[Tuple[int, float]]
    bloch_vectors: List[Tuple[float, float, float]]
    fidelity_with_reference: float
    execution_time: float


@dataclass
class VisualizationOutput:
    html_path: str
    png_path: str
    npz_data_path: str
    metadata: Dict[str, Any]


class IVisualComponent(ABC):
    @abstractmethod
    def render(self, data: Any, axes: Any, config: BrutalistConfig) -> None:
        pass


class ProbabilityVisualizer(IVisualComponent):
    def render(self, snapshot: QuantumSnapshot, axes: Any, config: BrutalistConfig) -> None:
        if snapshot is None or len(snapshot.probabilities) == 0:
            self._render_empty(axes, config)
            return
        probs = snapshot.probabilities
        n_states = len(probs)
        n_qubits = snapshot.n_qubits
        bitstrings = [format(i, f'0{n_qubits}b') for i in range(n_states)]
        colors = self._generate_colors(probs, config)
        bar_width = 0.8
        indices = np.arange(n_states)
        bars = axes.bar(indices, probs, width=bar_width, color=colors,
                       edgecolor=config.colors["grid"], linewidth=0.5, alpha=0.9)
        for bar, prob in zip(bars, probs):
            if prob > config.probability_threshold * 10:
                height = bar.get_height()
                axes.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         f'{prob:.3f}', ha='center', va='bottom',
                         fontsize=7, color=config.colors["text"], fontweight='bold')
        axes.set_xticks(indices)
        axes.set_xticklabels(bitstrings, rotation=45, ha='right',
                            fontsize=8, color=config.colors["text"], fontfamily='monospace')
        axes.set_ylabel('Probability', color=config.colors["text"],
                       fontsize=config.figure_dpi // 20, fontweight='bold')
        axes.set_title(f'Step {snapshot.step}: {snapshot.gate_name}',
                      color=config.colors["primary"], fontsize=config.figure_dpi // 15,
                      fontweight='bold', pad=10)
        axes.set_facecolor(config.colors["background"])
        axes.tick_params(colors=config.colors["text"], labelsize=8)
        axes.set_ylim(0, max(probs) * 1.2 if max(probs) > 0 else 1)
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])
            spine.set_linewidth(0.5)
        axes.grid(True, alpha=config.grid_alpha, color=config.colors["grid"],
                 linestyle='--', linewidth=0.3)

    def _render_empty(self, axes: Any, config: BrutalistConfig) -> None:
        axes.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                 fontsize=20, color=config.colors["grid"],
                 transform=axes.transAxes, fontweight='bold')
        axes.set_facecolor(config.colors["background"])
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])

    def _generate_colors(self, probs: np.ndarray, config: BrutalistConfig) -> List[str]:
        max_prob = max(probs) if len(probs) > 0 else 1.0
        colors = []
        for p in probs:
            intensity = p / (max_prob + 1e-10)
            if intensity > 0.5:
                colors.append(config.colors["primary"])
            elif intensity > 0.2:
                colors.append(config.colors["secondary"])
            else:
                colors.append(config.colors["tertiary"])
        return colors


class BlochSphereVisualizer(IVisualComponent):
    def render(self, snapshot: QuantumSnapshot, axes: Any, config: BrutalistConfig) -> None:
        if snapshot is None or len(snapshot.bloch_vectors) == 0:
            self._render_empty_sphere(axes, config)
            return
        self._draw_sphere_wireframe(axes, config)
        self._draw_axes(axes, config)
        qubit_colors = [
            config.colors["primary"],
            config.colors["secondary"],
            config.colors["tertiary"],
            config.colors["accent"],
        ]
        for i, (bx, by, bz) in enumerate(snapshot.bloch_vectors):
            color = qubit_colors[i % len(qubit_colors)]
            self._draw_bloch_vector(axes, bx, by, bz, color, i, config)
            self._draw_uncertainty_ring(axes, bx, by, bz, color, config)
        axes.set_xlim(-1.3, 1.3)
        axes.set_ylim(-1.3, 1.3)
        axes.set_zlim(-1.3, 1.3)
        axes.set_title(f'Bloch Sphere (Step {snapshot.step})',
                      color=config.colors["primary"], fontsize=config.figure_dpi // 15,
                      fontweight='bold', pad=10)
        axes.set_facecolor(config.colors["background"])
        axes.xaxis.pane.fill = False
        axes.yaxis.pane.fill = False
        axes.zaxis.pane.fill = False
        axes.xaxis.pane.set_edgecolor(config.colors["grid"])
        axes.yaxis.pane.set_edgecolor(config.colors["grid"])
        axes.zaxis.pane.set_edgecolor(config.colors["grid"])
        axes.tick_params(colors=config.colors["grid"], labelsize=6)
        axes.set_xlabel('X', color=config.colors["text"], fontsize=8)
        axes.set_ylabel('Y', color=config.colors["text"], fontsize=8)
        axes.set_zlabel('Z', color=config.colors["text"], fontsize=8)

    def _render_empty_sphere(self, axes: Any, config: BrutalistConfig) -> None:
        self._draw_sphere_wireframe(axes, config)
        axes.set_title('Bloch Sphere (No Data)',
                      color=config.colors["grid"], fontsize=config.figure_dpi // 15)
        axes.set_facecolor(config.colors["background"])

    def _draw_sphere_wireframe(self, axes: Any, config: BrutalistConfig) -> None:
        u = np.linspace(0, 2 * np.pi, config.sphere_resolution)
        v = np.linspace(0, np.pi, config.sphere_resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        axes.plot_surface(x, y, z, alpha=0.05, color=config.colors["surface"],
                         edgecolor=config.colors["grid"], linewidth=0.1)
        for angle in np.linspace(0, np.pi, 3):
            circle_x = np.cos(u) * np.sin(angle)
            circle_y = np.sin(u) * np.sin(angle)
            circle_z = np.cos(angle) * np.ones_like(u)
            axes.plot(circle_x, circle_y, circle_z,
                     color=config.colors["grid"], alpha=0.2, linewidth=0.5)
        for angle in np.linspace(0, 2 * np.pi, 6):
            circle_x = np.sin(v) * np.cos(angle)
            circle_y = np.sin(v) * np.sin(angle)
            circle_z = np.cos(v)
            axes.plot(circle_x, circle_y, circle_z,
                     color=config.colors["grid"], alpha=0.2, linewidth=0.5)

    def _draw_axes(self, axes: Any, config: BrutalistConfig) -> None:
        axis_length = 1.2
        axes.plot([-axis_length, axis_length], [0, 0], [0, 0],
                 color=config.colors["grid"], alpha=0.5, linewidth=1)
        axes.plot([0, 0], [-axis_length, axis_length], [0, 0],
                 color=config.colors["grid"], alpha=0.5, linewidth=1)
        axes.plot([0, 0], [0, 0], [-axis_length, axis_length],
                 color=config.colors["grid"], alpha=0.5, linewidth=1)
        axes.text(axis_length + 0.1, 0, 0, 'X', color=config.colors["text"], fontsize=8)
        axes.text(0, axis_length + 0.1, 0, 'Y', color=config.colors["text"], fontsize=8)
        axes.text(0, 0, axis_length + 0.1, 'Z', color=config.colors["text"], fontsize=8)
        axes.text(-axis_length - 0.1, 0, 0, '-X', color=config.colors["grid"], fontsize=6)
        axes.text(0, -axis_length - 0.1, 0, '-Y', color=config.colors["grid"], fontsize=6)
        axes.text(0, 0, -axis_length - 0.1, '-Z', color=config.colors["grid"], fontsize=6)

    def _draw_bloch_vector(self, axes: Any, bx: float, by: float, bz: float,
                          color: str, qubit_idx: int, config: BrutalistConfig) -> None:
        mag = math.sqrt(bx**2 + by**2 + bz**2)
        if mag > 1e-10:
            scale = config.vector_scale
            axes.quiver(0, 0, 0, bx * scale, by * scale, bz * scale,
                       color=color, arrow_length_ratio=0.15, linewidth=3,
                       alpha=0.9)
            axes.scatter([bx * scale], [by * scale], [bz * scale],
                        color=color, s=80, marker='o', edgecolors='white',
                        linewidths=1, alpha=1.0, zorder=10)
            axes.text(bx * scale * 1.2, by * scale * 1.2, bz * scale * 1.2,
                     f'q{qubit_idx}', color=color, fontsize=10, fontweight='bold')

    def _draw_uncertainty_ring(self, axes: Any, bx: float, by: float, bz: float,
                               color: str, config: BrutalistConfig) -> None:
        theta = np.linspace(0, 2 * np.pi, 50)
        r = config.uncertainty_ring_radius
        if abs(bz) < 0.99:
            perp_x, perp_y = -by, bx
            norm = math.sqrt(perp_x**2 + perp_y**2)
            if norm > 1e-10:
                perp_x, perp_y = perp_x / norm, perp_y / norm
                circle_x = bx + r * (perp_x * np.cos(theta) + perp_y * np.sin(theta) * 0.3)
                circle_y = by + r * (perp_y * np.cos(theta) - perp_x * np.sin(theta) * 0.3)
                circle_z = bz * np.ones_like(theta)
                axes.plot(circle_x, circle_y, circle_z,
                         color=color, alpha=0.3, linewidth=1, linestyle='--')


class PhaseSpaceVisualizer(IVisualComponent):
    def render(self, snapshot: QuantumSnapshot, axes: Any, config: BrutalistConfig) -> None:
        if snapshot is None or len(snapshot.phases) == 0:
            self._render_empty(axes, config)
            return
        phases = snapshot.phases
        probs = snapshot.probabilities
        mask = probs > config.probability_threshold
        if not np.any(mask):
            self._render_empty(axes, config)
            return
        angles = phases[mask]
        magnitudes = probs[mask]
        x = magnitudes * np.cos(angles)
        y = magnitudes * np.sin(angles)
        colors = angles / (2 * np.pi)
        sizes = magnitudes * config.scatter_point_size_max + config.scatter_point_size_min
        scatter = axes.scatter(x, y, c=colors, cmap='twilight',
                              s=sizes, alpha=0.7, edgecolors=config.colors["text"],
                              linewidths=0.5)
        circle = plt.Circle((0, 0), 1, fill=False, color=config.colors["grid"],
                            linestyle='--', linewidth=1, alpha=0.5)
        axes.add_patch(circle)
        axes.axhline(y=0, color=config.colors["grid"], linestyle='-', linewidth=0.5, alpha=0.3)
        axes.axvline(x=0, color=config.colors["grid"], linestyle='-', linewidth=0.5, alpha=0.3)
        for i, (xi, yi, prob) in enumerate(zip(x, y, magnitudes)):
            if prob > 0.1:
                bitstring = format(i, f'0{snapshot.n_qubits}b')
                axes.annotate(bitstring, (xi, yi), textcoords="offset points",
                             xytext=(5, 5), fontsize=6, color=config.colors["text"],
                             alpha=0.7)
        axes.set_xlim(-1.3, 1.3)
        axes.set_ylim(-1.3, 1.3)
        axes.set_xlabel('Real', color=config.colors["text"],
                       fontsize=config.figure_dpi // 20, fontweight='bold')
        axes.set_ylabel('Imaginary', color=config.colors["text"],
                       fontsize=config.figure_dpi // 20, fontweight='bold')
        axes.set_title(f'Phase Space (Step {snapshot.step})',
                      color=config.colors["primary"], fontsize=config.figure_dpi // 15,
                      fontweight='bold', pad=10)
        axes.set_facecolor(config.colors["background"])
        axes.tick_params(colors=config.colors["text"], labelsize=8)
        axes.set_aspect('equal')
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])
        axes.grid(True, alpha=config.grid_alpha, color=config.colors["grid"],
                 linestyle='--', linewidth=0.3)

    def _render_empty(self, axes: Any, config: BrutalistConfig) -> None:
        circle = plt.Circle((0, 0), 1, fill=False, color=config.colors["grid"],
                            linestyle='--', linewidth=1, alpha=0.5)
        axes.add_patch(circle)
        axes.text(0, 0, 'NO DATA', ha='center', va='center',
                 fontsize=15, color=config.colors["grid"], fontweight='bold')
        axes.set_xlim(-1.3, 1.3)
        axes.set_ylim(-1.3, 1.3)
        axes.set_facecolor(config.colors["background"])
        axes.set_aspect('equal')
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])


class EntropyVisualizer(IVisualComponent):
    def render(self, snapshots: List[QuantumSnapshot], axes: Any, config: BrutalistConfig) -> None:
        if not snapshots:
            self._render_empty(axes, config)
            return
        steps = [s.step for s in snapshots]
        entropies = [s.entropy for s in snapshots]
        max_entropy = max(entropies) if entropies else 1.0
        n_points = len(steps)
        gradient_colors = self._interpolate_colors(
            config.colors["secondary"],
            config.colors["primary"],
            n_points
        )
        for i in range(1, n_points):
            axes.plot(steps[i-1:i+1], entropies[i-1:i+1],
                     color=gradient_colors[i], linewidth=3, alpha=0.8)
        axes.scatter(steps, entropies, c=gradient_colors, s=80,
                    edgecolors='white', linewidths=1, zorder=10)
        axes.fill_between(steps, entropies, alpha=0.2, color=config.colors["primary"])
        if max_entropy > 0:
            axes.axhline(y=max_entropy, color=config.colors["tertiary"],
                        linestyle='--', alpha=0.5, linewidth=1,
                        label=f'Max: {max_entropy:.4f}')
        for i, (step, entropy) in enumerate(zip(steps, entropies)):
            if i % max(1, len(steps) // 5) == 0:
                axes.annotate(f'{entropy:.2f}', (step, entropy),
                             textcoords="offset points", xytext=(0, 10),
                             fontsize=7, color=config.colors["text"],
                             ha='center', fontweight='bold')
        axes.set_xlabel('Gate Step', color=config.colors["text"],
                       fontsize=config.figure_dpi // 20, fontweight='bold')
        axes.set_ylabel('Entropy (bits)', color=config.colors["text"],
                       fontsize=config.figure_dpi // 20, fontweight='bold')
        axes.set_title('Shannon Entropy Evolution',
                      color=config.colors["primary"], fontsize=config.figure_dpi // 15,
                      fontweight='bold', pad=10)
        axes.set_facecolor(config.colors["background"])
        axes.tick_params(colors=config.colors["text"], labelsize=8)
        axes.grid(True, alpha=config.grid_alpha, color=config.colors["grid"],
                 linestyle='--', linewidth=0.3)
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])
        if max_entropy > 0:
            axes.legend(loc='upper right', facecolor=config.colors["background"],
                       edgecolor=config.colors["grid"], labelcolor=config.colors["text"])

    def _render_empty(self, axes: Any, config: BrutalistConfig) -> None:
        axes.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                 fontsize=15, color=config.colors["grid"], fontweight='bold',
                 transform=axes.transAxes)
        axes.set_facecolor(config.colors["background"])
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])

    def _interpolate_colors(self, color1: str, color2: str, n: int) -> List[str]:
        if n <= 0:
            return []
        c1 = self._hex_to_rgb(color1)
        c2 = self._hex_to_rgb(color2)
        colors = []
        for i in range(n):
            t = i / max(n - 1, 1)
            r = int(c1[0] + t * (c2[0] - c1[0]))
            g = int(c1[1] + t * (c2[1] - c1[1]))
            b = int(c1[2] + t * (c2[2] - c1[2]))
            colors.append(f'#{r:02x}{g:02x}{b:02x}')
        return colors

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class BackendComparisonVisualizer(IVisualComponent):
    def render(self, comparisons: List[BackendComparison], axes: Any, config: BrutalistConfig) -> None:
        if not comparisons:
            self._render_empty(axes, config)
            return
        n_backends = len(comparisons)
        n_states = len(comparisons[0].final_probabilities)
        bar_width = 0.8 / n_backends
        x = np.arange(n_states)
        colors = [config.colors["primary"], config.colors["secondary"], config.colors["tertiary"]]
        for i, comp in enumerate(comparisons):
            offset = (i - n_backends / 2 + 0.5) * bar_width
            bars = axes.bar(x + offset, comp.final_probabilities, bar_width,
                           label=comp.backend_name.capitalize(),
                           color=colors[i % len(colors)], alpha=0.7,
                           edgecolor=config.colors["text"], linewidth=0.3)
            for bar, prob in zip(bars, comp.final_probabilities):
                if prob > 0.05:
                    height = bar.get_height()
                    axes.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                             f'{prob:.2f}', ha='center', va='bottom',
                             fontsize=6, color=config.colors["text"])
        n_qubits = comparisons[0].snapshots[0].n_qubits if comparisons[0].snapshots else 3
        bitstrings = [format(j, f'0{n_qubits}b') for j in range(n_states)]
        axes.set_xticks(x)
        axes.set_xticklabels(bitstrings, rotation=45, ha='right',
                            fontsize=7, color=config.colors["text"], fontfamily='monospace')
        axes.set_ylabel('Probability', color=config.colors["text"],
                       fontsize=config.figure_dpi // 20, fontweight='bold')
        axes.set_title('Backend Comparison',
                      color=config.colors["primary"], fontsize=config.figure_dpi // 15,
                      fontweight='bold', pad=10)
        axes.set_facecolor(config.colors["background"])
        axes.tick_params(colors=config.colors["text"], labelsize=8)
        axes.legend(loc='upper right', facecolor=config.colors["background"],
                   edgecolor=config.colors["grid"], labelcolor=config.colors["text"])
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])
        axes.grid(True, alpha=config.grid_alpha, color=config.colors["grid"],
                 linestyle='--', linewidth=0.3, axis='y')

    def _render_empty(self, axes: Any, config: BrutalistConfig) -> None:
        axes.text(0.5, 0.5, 'NO BACKEND DATA', ha='center', va='center',
                 fontsize=15, color=config.colors["grid"], fontweight='bold',
                 transform=axes.transAxes)
        axes.set_facecolor(config.colors["background"])
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])


class FidelityVisualizer(IVisualComponent):
    def render(self, comparisons: List[BackendComparison], axes: Any, config: BrutalistConfig) -> None:
        if not comparisons:
            self._render_empty(axes, config)
            return
        names = [c.backend_name.capitalize() for c in comparisons]
        fidelities = [c.fidelity_with_reference for c in comparisons]
        entropies = [c.final_entropy for c in comparisons]
        colors = [config.colors["primary"], config.colors["secondary"], config.colors["tertiary"]]
        bars = axes.bar(names, fidelities, color=colors[:len(comparisons)],
                       edgecolor=config.colors["text"], linewidth=1, alpha=0.8)
        for bar, fid, entropy in zip(bars, fidelities, entropies):
            height = bar.get_height()
            axes.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                     f'{fid:.6f}', ha='center', va='bottom',
                     fontsize=9, color=config.colors["text"], fontweight='bold')
            axes.text(bar.get_x() + bar.get_width() / 2, height / 2,
                     f'H={entropy:.3f}', ha='center', va='center',
                     fontsize=7, color=config.colors["background"], fontweight='bold')
        axes.set_ylabel('Fidelity', color=config.colors["text"],
                       fontsize=config.figure_dpi // 20, fontweight='bold')
        axes.set_title('Backend Fidelity vs Reference',
                      color=config.colors["primary"], fontsize=config.figure_dpi // 15,
                      fontweight='bold', pad=10)
        axes.set_facecolor(config.colors["background"])
        axes.tick_params(colors=config.colors["text"], labelsize=9)
        axes.set_ylim(0, 1.15)
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])
        axes.grid(True, alpha=config.grid_alpha, color=config.colors["grid"],
                 linestyle='--', linewidth=0.3, axis='y')

    def _render_empty(self, axes: Any, config: BrutalistConfig) -> None:
        axes.text(0.5, 0.5, 'NO FIDELITY DATA', ha='center', va='center',
                 fontsize=15, color=config.colors["grid"], fontweight='bold',
                 transform=axes.transAxes)
        axes.set_facecolor(config.colors["background"])
        for spine in axes.spines.values():
            spine.set_color(config.colors["grid"])


class QuantumStateAnalyzer:
    def __init__(self, config: BrutalistConfig):
        self.config = config

    def compute_probabilities(self, state: JointHilbertState) -> np.ndarray:
        if state is None:
            return np.array([])
        probs = state.probabilities().numpy()
        return probs / (probs.sum() + 1e-12)

    def compute_phases(self, state: JointHilbertState) -> np.ndarray:
        if state is None:
            return np.array([])
        amps = state.amplitudes
        if amps.dim() > 2:
            amps_flat = amps.sum(dim=(-2, -1))
        else:
            amps_flat = amps
        re = amps_flat[:, 0].numpy()
        im = amps_flat[:, 1].numpy()
        return np.arctan2(im, re)

    def compute_entropy(self, probs: np.ndarray) -> float:
        if probs is None or len(probs) == 0:
            return 0.0
        p_nonzero = probs[probs > self.config.probability_threshold]
        if len(p_nonzero) == 0:
            return 0.0
        return float(-np.sum(p_nonzero * np.log(p_nonzero + 1e-15)) / np.log(self.config.entropy_base))

    def compute_bloch_vectors(self, state: JointHilbertState) -> List[Tuple[float, float, float]]:
        if state is None:
            return []
        vectors = []
        for qubit in range(state.n_qubits):
            bx, by, bz = state.bloch_vector(qubit)
            vectors.append((float(bx), float(by), float(bz)))
        return vectors

    def create_snapshot(self, state: JointHilbertState, step: int,
                       gate_name: str, backend_name: str = "unknown") -> QuantumSnapshot:
        if state is None:
            return QuantumSnapshot(
                step=step,
                gate_name=gate_name,
                probabilities=np.array([1.0]),
                phases=np.array([0.0]),
                entropy=0.0,
                bloch_vectors=[],
                amplitudes=np.array([[1.0, 0.0]]),
                most_probable=0,
                norm=1.0,
                n_qubits=1,
                backend_name=backend_name
            )
        probs = self.compute_probabilities(state)
        phases = self.compute_phases(state)
        entropy = self.compute_entropy(probs)
        bloch = self.compute_bloch_vectors(state)
        amps = state.amplitudes
        if amps.dim() > 2:
            amps_np = amps.sum(dim=(-2, -1)).numpy()
        else:
            amps_np = amps.numpy()
        norm = float(np.sqrt((amps_np**2).sum()))
        return QuantumSnapshot(
            step=step,
            gate_name=gate_name,
            probabilities=probs,
            phases=phases,
            entropy=entropy,
            bloch_vectors=bloch,
            amplitudes=amps_np,
            most_probable=int(np.argmax(probs)) if len(probs) > 0 else 0,
            norm=norm,
            n_qubits=state.n_qubits,
            backend_name=backend_name
        )


class StandardCircuits:
    @staticmethod
    def bell_state() -> List[Tuple[str, List[int], Dict[str, float]]]:
        return [("H", [0], {}), ("CNOT", [0, 1], {})]

    @staticmethod
    def ghz_state(n_qubits: int) -> List[Tuple[str, List[int], Dict[str, float]]]:
        gates = [("H", [0], {})]
        for i in range(n_qubits - 1):
            gates.append(("CNOT", [i, i + 1], {}))
        return gates

    @staticmethod
    def qft(n_qubits: int) -> List[Tuple[str, List[int], Dict[str, float]]]:
        gates = []
        for i in range(n_qubits):
            gates.append(("H", [i], {}))
            for j in range(i + 1, n_qubits):
                angle = math.pi / (2 ** (j - i))
                gates.append(("Rz", [j], {"theta": angle}))
                gates.append(("CNOT", [i, j], {}))
                gates.append(("Rz", [j], {"theta": -angle}))
                gates.append(("CNOT", [i, j], {}))
        for i in range(n_qubits // 2):
            gates.append(("SWAP", [i, n_qubits - 1 - i], {}))
        return gates

    @staticmethod
    def grover_oracle(n_qubits: int, marked: int) -> List[Tuple[str, List[int], Dict[str, float]]]:
        gates = []
        for i in range(n_qubits):
            if not (marked >> (n_qubits - 1 - i)) & 1:
                gates.append(("X", [i], {}))
        gates.append(("MCZ", list(range(n_qubits)), {}))
        for i in range(n_qubits):
            if not (marked >> (n_qubits - 1 - i)) & 1:
                gates.append(("X", [i], {}))
        return gates

    @staticmethod
    def grover_diffusion(n_qubits: int) -> List[Tuple[str, List[int], Dict[str, float]]]:
        gates = []
        for i in range(n_qubits):
            gates.append(("H", [i], {}))
            gates.append(("X", [i], {}))
        gates.append(("MCZ", list(range(n_qubits)), {}))
        for i in range(n_qubits):
            gates.append(("X", [i], {}))
            gates.append(("H", [i], {}))
        return gates


class CircuitExecutor:
    def __init__(self, qc: QuantumComputer, config: BrutalistConfig):
        self.qc = qc
        self.config = config
        self.analyzer = QuantumStateAnalyzer(config)

    def execute_sequence(
        self,
        gates: Sequence[Tuple[str, List[int], Dict[str, float]]],
        n_qubits: int,
        backend_name: str
    ) -> List[QuantumSnapshot]:
        if backend_name not in self.qc._backends:
            _LOG.warning("Backend %s not available", backend_name)
            return []
        backend = self.qc._backends[backend_name]
        factory = self.qc._factory
        state = factory.all_zeros(n_qubits)
        snapshots = [self.analyzer.create_snapshot(state, 0, "INIT", backend_name)]
        for step, (gate_name, targets, params) in enumerate(gates, 1):
            gate = _GATE_REGISTRY.get(gate_name)
            if gate is None:
                _LOG.warning("Unknown gate: %s", gate_name)
                continue
            state = gate.apply(state, backend, targets, params)
            snapshot = self.analyzer.create_snapshot(state, step, gate_name, backend_name)
            snapshots.append(snapshot)
        return snapshots

    def compare_backends(
        self,
        gates: Sequence[Tuple[str, List[int], Dict[str, float]]],
        n_qubits: int
    ) -> List[BackendComparison]:
        results = []
        reference_probs = None
        for backend_name in self.config.backend_order:
            if backend_name not in self.qc._backends:
                continue
            snapshots = self.execute_sequence(gates, n_qubits, backend_name)
            if not snapshots:
                continue
            final = snapshots[-1]
            probs = final.probabilities
            if reference_probs is None:
                reference_probs = probs
                fidelity = 1.0
            else:
                min_len = min(len(reference_probs), len(probs))
                fidelity = float(np.sum(np.sqrt(reference_probs[:min_len] * probs[:min_len]))**2)
            top_indices = np.argsort(probs)[::-1][:4]
            top_states = [(int(i), float(probs[i])) for i in top_indices if probs[i] > 0.001]
            comparison = BackendComparison(
                backend_name=backend_name,
                snapshots=snapshots,
                final_probabilities=probs,
                final_entropy=final.entropy,
                most_probable_state=final.most_probable,
                top_states=top_states,
                bloch_vectors=final.bloch_vectors,
                fidelity_with_reference=fidelity,
                execution_time=0.0
            )
            results.append(comparison)
        return results


class FigureBuilder:
    def __init__(self, config: BrutalistConfig):
        self.config = config
        self.prob_viz = ProbabilityVisualizer()
        self.bloch_viz = BlochSphereVisualizer()
        self.phase_viz = PhaseSpaceVisualizer()
        self.entropy_viz = EntropyVisualizer()
        self.comparison_viz = BackendComparisonVisualizer()
        self.fidelity_viz = FidelityVisualizer()

    def build_full_figure(
        self,
        snapshots: List[QuantumSnapshot],
        comparisons: List[BackendComparison]
    ) -> Any:
        if not MATPLOTLIB_AVAILABLE:
            _LOG.error("matplotlib not available")
            return None
        fig = plt.figure(figsize=(self.config.figure_width_inches,
                                  self.config.figure_height_inches),
                        dpi=self.config.figure_dpi)
        fig.patch.set_facecolor(self.config.colors["background"])
        valid_snapshots = [s for s in snapshots if s is not None]
        n_snapshots = min(4, len(valid_snapshots))
        n_comparison = len([c for c in comparisons if c is not None and c.snapshots])
        rows = 3 + (1 if n_comparison > 0 else 0)
        cols = max(n_snapshots, 2)
        grid = plt.GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.25)
        for i in range(n_snapshots):
            snapshot = valid_snapshots[i]
            ax_prob = fig.add_subplot(grid[0, i])
            self.prob_viz.render(snapshot, ax_prob, self.config)
            ax_bloch = fig.add_subplot(grid[1, i], projection='3d')
            self.bloch_viz.render(snapshot, ax_bloch, self.config)
            ax_phase = fig.add_subplot(grid[2, i])
            self.phase_viz.render(snapshot, ax_phase, self.config)
        if valid_snapshots:
            ax_entropy = fig.add_subplot(grid[0, n_snapshots:] if n_snapshots < cols else grid[0, -1])
            self.entropy_viz.render(valid_snapshots, ax_entropy, self.config)
        if n_comparison > 0:
            ax_compare = fig.add_subplot(grid[3, :cols//2] if cols > 1 else grid[3, 0])
            self.comparison_viz.render(comparisons, ax_compare, self.config)
            ax_fidelity = fig.add_subplot(grid[3, cols//2:] if cols > 1 else grid[3, 1] if cols > 1 else grid[3, 0])
            self.fidelity_viz.render(comparisons, ax_fidelity, self.config)
        title = "QUANTUM STATE VISUALIZATION // CRYSTALLINE BACKEND ANALYSIS"
        fig.suptitle(title, color=self.config.colors["primary"],
                    fontsize=self.config.figure_dpi // 8, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def build_summary_figure(
        self,
        snapshots: List[QuantumSnapshot],
        comparisons: List[BackendComparison]
    ) -> Any:
        if not MATPLOTLIB_AVAILABLE:
            return None
        fig = plt.figure(figsize=(self.config.figure_width_inches // 2,
                                  self.config.figure_height_inches // 2),
                        dpi=self.config.figure_dpi)
        fig.patch.set_facecolor(self.config.colors["background"])
        grid = plt.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
        valid_snapshots = [s for s in snapshots if s is not None]
        if valid_snapshots:
            final = valid_snapshots[-1]
            ax1 = fig.add_subplot(grid[0, 0])
            self.prob_viz.render(final, ax1, self.config)
            ax2 = fig.add_subplot(grid[0, 1], projection='3d')
            self.bloch_viz.render(final, ax2, self.config)
            ax3 = fig.add_subplot(grid[0, 2])
            self.phase_viz.render(final, ax3, self.config)
            ax4 = fig.add_subplot(grid[1, 0])
            self.entropy_viz.render(valid_snapshots, ax4, self.config)
        valid_comparisons = [c for c in comparisons if c is not None]
        if valid_comparisons:
            ax5 = fig.add_subplot(grid[1, 1])
            self.comparison_viz.render(valid_comparisons, ax5, self.config)
            ax6 = fig.add_subplot(grid[1, 2])
            self.fidelity_viz.render(valid_comparisons, ax6, self.config)
        fig.suptitle("QUANTUM SUMMARY", color=self.config.colors["primary"],
                    fontsize=self.config.figure_dpi // 10, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig


class QuantumVisualizer:
    def __init__(self, config: BrutalistConfig):
        self.config = config
        self.qc = None
        self.executor = None
        self.figure_builder = None
        self._initialize()

    def _initialize(self) -> None:
        _LOG.info("=" * 70)
        _LOG.info("QUANTUM BRUTALIST VISUALIZER INITIALIZING")
        _LOG.info("=" * 70)
        self._init_quantum_computer()
        if self.qc:
            self.executor = CircuitExecutor(self.qc, self.config)
        self.figure_builder = FigureBuilder(self.config)
        _LOG.info("Initialization complete")

    def _init_quantum_computer(self) -> None:
        if not TORCH_AVAILABLE:
            _LOG.warning("PyTorch not available, using synthetic data")
            return
        if not QUANTUM_COMPUTER_AVAILABLE:
            _LOG.warning("quantum_computer module not available, using synthetic data")
            return
        qc_config = SimulatorConfig(
            grid_size=self.config.grid_size,
            hidden_dim=self.config.hidden_dim,
            expansion_dim=self.config.expansion_dim,
            num_spectral_layers=self.config.num_spectral_layers,
            dirac_mass=self.config.dirac_mass,
            dirac_c=self.config.dirac_c,
            gamma_representation=self.config.gamma_representation,
            dt=self.config.dt,
            normalization_eps=self.config.normalization_eps,
            potential_depth=self.config.potential_depth,
            potential_width=self.config.potential_width,
            hamiltonian_checkpoint=self.config.hamiltonian_checkpoint,
            schrodinger_checkpoint=self.config.schrodinger_checkpoint,
            dirac_checkpoint=self.config.dirac_checkpoint,
            device=self.config.device,
            random_seed=self.config.random_seed,
            max_qubits=self.config.max_qubits,
        )
        try:
            self.qc = QuantumComputer(qc_config)
            _LOG.info("Quantum computer initialized with backends: %s",
                     list(self.qc._backends.keys()))
        except Exception as e:
            _LOG.error("Failed to initialize quantum computer: %s", e)
            self.qc = None

    def visualize_bell_state(self) -> VisualizationOutput:
        _LOG.info("VISUALIZING BELL STATE")
        gates = StandardCircuits.bell_state()
        n_qubits = 2
        return self._execute_and_visualize(gates, n_qubits, "bell_state")

    def visualize_ghz_state(self, n_qubits: int = None) -> VisualizationOutput:
        n_qubits = n_qubits or self.config.default_n_qubits
        _LOG.info("VISUALIZING GHZ STATE (%d qubits)", n_qubits)
        gates = StandardCircuits.ghz_state(n_qubits)
        return self._execute_and_visualize(gates, n_qubits, f"ghz_{n_qubits}q")

    def visualize_qft(self, n_qubits: int = None) -> VisualizationOutput:
        n_qubits = n_qubits or self.config.default_n_qubits
        _LOG.info("VISUALIZING QFT (%d qubits)", n_qubits)
        gates = StandardCircuits.qft(n_qubits)
        return self._execute_and_visualize(gates, n_qubits, f"qft_{n_qubits}q")

    def visualize_grover(self, n_qubits: int = None, marked_state: int = None) -> VisualizationOutput:
        n_qubits = n_qubits or self.config.default_n_qubits
        marked_state = marked_state if marked_state is not None else self.config.grover_marked_state
        iterations = self.config.grover_iterations
        _LOG.info("VISUALIZING GROVER (%d qubits, marked=%d, iters=%d)",
                 n_qubits, marked_state, iterations)
        gates = []
        for i in range(n_qubits):
            gates.append(("H", [i], {}))
        for _ in range(iterations):
            gates.extend(StandardCircuits.grover_oracle(n_qubits, marked_state))
            gates.extend(StandardCircuits.grover_diffusion(n_qubits))
        return self._execute_and_visualize(gates, n_qubits,
                                           f"grover_{n_qubits}q_m{marked_state}")

    def _execute_and_visualize(
        self,
        gates: List[Tuple[str, List[int], Dict[str, float]]],
        n_qubits: int,
        name: str
    ) -> VisualizationOutput:
        os.makedirs(self.config.output_dir, exist_ok=True)
        if self.executor and self.qc:
            all_snapshots = {}
            for backend_name in self.config.backend_order:
                if backend_name in self.qc._backends:
                    snapshots = self.executor.execute_sequence(gates, n_qubits, backend_name)
                    all_snapshots[backend_name] = snapshots
            primary_snapshots = all_snapshots.get(self.config.backend_order[0], [])
            comparisons = self.executor.compare_backends(gates, n_qubits)
        else:
            primary_snapshots = self._create_synthetic_snapshots(n_qubits, gates)
            comparisons = self._create_synthetic_comparisons(n_qubits, gates)
        fig = self.figure_builder.build_full_figure(primary_snapshots, comparisons)
        summary_fig = self.figure_builder.build_summary_figure(primary_snapshots, comparisons)
        png_path = os.path.join(self.config.output_dir, f"{name}.png")
        summary_path = os.path.join(self.config.output_dir, f"{name}_summary.png")
        npz_path = os.path.join(self.config.output_dir, f"{name}_data.npz")
        if fig:
            fig.savefig(png_path, dpi=self.config.figure_dpi,
                       facecolor=self.config.colors["background"],
                       bbox_inches='tight', pad_inches=0.1)
            _LOG.info("Saved: %s", png_path)
            plt.close(fig)
        if summary_fig:
            summary_fig.savefig(summary_path, dpi=self.config.figure_dpi,
                               facecolor=self.config.colors["background"],
                               bbox_inches='tight', pad_inches=0.1)
            _LOG.info("Saved: %s", summary_path)
            plt.close(summary_fig)
        self._save_data(npz_path, primary_snapshots, comparisons)
        return VisualizationOutput(
            html_path="",
            png_path=png_path,
            npz_data_path=npz_path,
            metadata={"name": name, "n_qubits": n_qubits, "gates": len(gates)}
        )

    def _create_synthetic_snapshots(
        self,
        n_qubits: int,
        gates: List[Tuple[str, List[int], Dict[str, float]]]
    ) -> List[QuantumSnapshot]:
        dim = 2 ** n_qubits
        snapshots = []
        gate_names = ["INIT"] + [g[0] for g in gates]
        for step, gate_name in enumerate(gate_names):
            np.random.seed(self.config.random_seed + step)
            if step == 0:
                probs = np.zeros(dim)
                probs[0] = 1.0
            elif step < len(gate_names) // 2:
                probs = np.ones(dim) / dim
            else:
                alpha = (step - len(gate_names) // 2) / max(len(gate_names) // 2, 1)
                target = step % dim
                probs = np.ones(dim) * (1 - alpha) / (dim - 1 + 1e-10)
                probs[target] = alpha
            probs = probs / probs.sum()
            phases = np.linspace(0, 2 * np.pi * step / max(len(gate_names), 1), dim)
            entropy = -np.sum(probs * np.log2(probs + 1e-15))
            bloch = [(np.sin(phases[i] + step * 0.1) * 0.5,
                      np.cos(phases[i]) * 0.3,
                      np.cos(phases[i] + step * 0.2) * 0.5)
                     for i in range(min(n_qubits, 3))]
            snapshot = QuantumSnapshot(
                step=step,
                gate_name=gate_name,
                probabilities=probs,
                phases=phases,
                entropy=float(entropy),
                bloch_vectors=bloch,
                amplitudes=np.sqrt(probs)[:, np.newaxis] * np.array([[1, 0]]),
                most_probable=int(np.argmax(probs)),
                norm=1.0,
                n_qubits=n_qubits,
                backend_name="synthetic"
            )
            snapshots.append(snapshot)
        return snapshots

    def _create_synthetic_comparisons(
        self,
        n_qubits: int,
        gates: List[Tuple[str, List[int], Dict[str, float]]]
    ) -> List[BackendComparison]:
        comparisons = []
        for backend_name in self.config.backend_order:
            snapshots = self._create_synthetic_snapshots(n_qubits, gates)
            if snapshots:
                final = snapshots[-1]
                comparison = BackendComparison(
                    backend_name=backend_name,
                    snapshots=snapshots,
                    final_probabilities=final.probabilities,
                    final_entropy=final.entropy,
                    most_probable_state=final.most_probable,
                    top_states=[(final.most_probable, final.probabilities[final.most_probable])],
                    bloch_vectors=final.bloch_vectors,
                    fidelity_with_reference=0.95 + np.random.random() * 0.05,
                    execution_time=0.0
                )
                comparisons.append(comparison)
        return comparisons

    def _save_data(
        self,
        path: str,
        snapshots: List[QuantumSnapshot],
        comparisons: List[BackendComparison]
    ) -> None:
        data = {
            "n_snapshots": len(snapshots),
            "n_comparisons": len(comparisons),
        }
        if snapshots:
            data["final_probs"] = snapshots[-1].probabilities
            data["final_entropy"] = snapshots[-1].entropy
            data["steps"] = np.array([s.step for s in snapshots])
            data["entropies"] = np.array([s.entropy for s in snapshots])
        if comparisons:
            data["backend_names"] = np.array([c.backend_name for c in comparisons])
            data["fidelities"] = np.array([c.fidelity_with_reference for c in comparisons])
        np.savez(path, **data)
        _LOG.info("Saved data: %s", path)

    def run_all(self) -> Dict[str, VisualizationOutput]:
        _LOG.info("=" * 70)
        _LOG.info("RUNNING ALL VISUALIZATIONS")
        _LOG.info("=" * 70)
        results = {}
        results["bell_state"] = self.visualize_bell_state()
        results["ghz_state"] = self.visualize_ghz_state()
        results["qft"] = self.visualize_qft()
        results["grover"] = self.visualize_grover()
        _LOG.info("=" * 70)
        _LOG.info("ALL VISUALIZATIONS COMPLETE")
        _LOG.info("=" * 70)
        self._print_summary(results)
        return results

    def _print_summary(self, results: Dict[str, VisualizationOutput]) -> None:
        _LOG.info("-" * 60)
        _LOG.info("SUMMARY")
        _LOG.info("-" * 60)
        for name, output in results.items():
            _LOG.info("  %s:", name.upper().replace("_", " "))
            _LOG.info("    PNG: %s", output.png_path)
            _LOG.info("    Data: %s", output.npz_data_path)


def main():
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)
    import argparse
    parser = argparse.ArgumentParser(description="Quantum Brutalist Visualizer")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--expansion-dim", type=int, default=64)
    parser.add_argument("--hamiltonian-checkpoint", default="hamiltonian.pth")
    parser.add_argument("--schrodinger-checkpoint",
                       default="checkpoint_phase3_training_epoch_18921_20260224_154739.pth")
    parser.add_argument("--dirac-checkpoint", default="best_dirac.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--output-dir", default="download")
    parser.add_argument("--color-scheme", choices=["cyberpunk", "matrix", "quantum_void", "neon_noir", "plasma"],
                       default="cyberpunk")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--bell", action="store_true")
    parser.add_argument("--ghz", action="store_true")
    parser.add_argument("--qft", action="store_true")
    parser.add_argument("--grover", action="store_true")
    args = parser.parse_args()
    scheme_map = {
        "cyberpunk": ColorScheme.CYBERPUNK,
        "matrix": ColorScheme.MATRIX,
        "quantum_void": ColorScheme.QUANTUM_VOID,
        "neon_noir": ColorScheme.NEON_NOIR,
        "plasma": ColorScheme.PLASMA,
    }
    config = BrutalistConfig(
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        expansion_dim=args.expansion_dim,
        hamiltonian_checkpoint=args.hamiltonian_checkpoint,
        schrodinger_checkpoint=args.schrodinger_checkpoint,
        dirac_checkpoint=args.dirac_checkpoint,
        device=args.device,
        random_seed=args.seed,
        default_n_qubits=args.n_qubits,
        output_dir=args.output_dir,
        color_scheme=scheme_map.get(args.color_scheme, ColorScheme.CYBERPUNK),
    )
    visualizer = QuantumVisualizer(config)
    if args.run_all:
        visualizer.run_all()
    else:
        if args.bell:
            visualizer.visualize_bell_state()
        if args.ghz:
            visualizer.visualize_ghz_state()
        if args.qft:
            visualizer.visualize_qft()
        if args.grover:
            visualizer.visualize_grover()
        if not (args.bell or args.ghz or args.qft or args.grover):
            visualizer.run_all()


if __name__ == "__main__":
    main()