#!/usr/bin/env python3
"""
quantum_visualizer.py - Production Quantum State Visualizer
============================================================
Brutal real-time visualization of quantum state evolution using
trained neural network backends (Hamiltonian, Schrodinger, Dirac).

Imports and uses existing quantum_computer.py, molecular_sim.py,
and advanced_experiments.py infrastructure.

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
from typing import Dict, List, Optional, Tuple, Any, Sequence, Callable

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

warnings.filterwarnings("ignore")

UPLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
if UPLOAD_DIR not in sys.path:
    sys.path.insert(0, UPLOAD_DIR)

try:
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for quantum_computer")
    from quantum_computer import (
        QuantumComputer,
        SimulatorConfig,
        JointHilbertState,
        QuantumCircuit,
        IQuantumGate,
        _GATE_REGISTRY,
        _single_qubit_unitary,
        _two_qubit_unitary,
        HamiltonianBackend,
        SchrodingerBackend,
        DiracBackend,
        IPhysicsBackend,
        JointStateFactory,
    )
    QUANTUM_COMPUTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import quantum_computer: {e}")
    QUANTUM_COMPUTER_AVAILABLE = False

try:
    from molecular_sim import (
        MoleculeData,
        ExactJWEnergy,
        VQESolver,
        VQEResult,
        MOLECULES,
        build_jw_hamiltonian_of,
    )
    MOLECULAR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: molecular_sim not available: {e}")
    MOLECULAR_AVAILABLE = False

try:
    from advanced_experiments import (
        GroverConfig,
        GroverSearch,
        QEDConfig,
        QEDEffectsExperiment,
        LambShiftCalculator,
        AnomalousMagneticMoment,
    )
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: advanced_experiments not available: {e}")
    ADVANCED_AVAILABLE = False


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


_LOG = _make_logger("QuantumVisualizer")


@dataclass
class VisualizerConfig:
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
    figure_dpi: int = 150
    figure_size_x: int = 24
    figure_size_y: int = 20
    colormap_primary: str = "inferno"
    colormap_secondary: str = "viridis"
    colormap_phase: str = "twilight"
    background_color: str = "#000008"
    text_color: str = "white"
    grid_color: str = "#333333"
    axis_label_size: int = 12
    title_size: int = 14
    annotation_size: int = 11
    animation_frames: int = 100
    animation_interval: int = 50
    bloch_sphere_resolution: int = 50
    histogram_bins: int = 100
    entropy_base: float = 2.0
    probability_threshold: float = 1e-6
    phase_normalization: float = math.pi
    output_dir: str = "download"
    backend_order: Tuple[str, str, str] = ("hamiltonian", "schrodinger", "dirac")
    gate_sequence: Tuple[str, ...] = ("H", "CNOT", "Z", "H")
    n_qubits_default: int = 3
    grover_iterations: int = 2
    grover_marked_state: int = 5
    vqe_max_iterations: int = 200
    vqe_tolerance: float = 1e-8


@dataclass
class QuantumStateSnapshot:
    step: int
    gate_name: str
    probabilities: np.ndarray
    phases: np.ndarray
    entropy: float
    bloch_vectors: List[Tuple[float, float, float]]
    amplitudes: np.ndarray
    most_probable: int
    norm: float


@dataclass
class BackendComparisonResult:
    backend_name: str
    probabilities: np.ndarray
    entropy: float
    most_probable: int
    top_states: List[Tuple[int, float]]
    bloch_vectors: List[Tuple[float, float, float]]
    fidelity_with_reference: float


@dataclass
class VisualizationResult:
    snapshots: List[QuantumStateSnapshot]
    backend_results: List[BackendComparisonResult]
    circuit_diagram: str
    execution_time: float
    total_steps: int
    output_path: str


class IVisualizationComponent(ABC):
    @abstractmethod
    def render(self, data: Any, axes: Any, config: VisualizerConfig) -> None:
        pass


class ProbabilityBarRenderer(IVisualizationComponent):
    def render(self, data: QuantumStateSnapshot, axes: Any, config: VisualizerConfig) -> None:
        probs = data.probabilities
        n_states = len(probs)
        indices = np.arange(n_states)
        bitstrings = [format(i, f'0{int(np.log2(n_states))}b') for i in range(n_states)]
        colors = self._get_colors(probs, config)
        bars = axes.bar(indices, probs, color=colors, edgecolor=config.grid_color, linewidth=0.5)
        axes.set_xticks(indices)
        axes.set_xticklabels(bitstrings, rotation=45, ha='right', fontsize=8, color=config.text_color)
        axes.set_ylabel('Probability', color=config.text_color, fontsize=config.axis_label_size)
        axes.set_title(f'Step {data.step}: {data.gate_name}', color=config.text_color, fontsize=config.title_size)
        axes.set_facecolor(config.background_color)
        axes.tick_params(colors=config.text_color)
        for spine in axes.spines.values():
            spine.set_color(config.grid_color)
        for bar, prob in zip(bars, probs):
            if prob > config.probability_threshold:
                axes.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{prob:.3f}', ha='center', va='bottom',
                         fontsize=7, color=config.text_color)

    def _get_colors(self, probs: np.ndarray, config: VisualizerConfig) -> List[str]:
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(config.colormap_primary)
            return [cmap(p) for p in probs]
        except ImportError:
            return ['#FF6600' if p > 0.5 else '#333333' for p in probs]


class BlochSphereRenderer(IVisualizationComponent):
    def render(self, data: QuantumStateSnapshot, axes: Any, config: VisualizerConfig) -> None:
        u = np.linspace(0, 2 * np.pi, config.bloch_sphere_resolution)
        v = np.linspace(0, np.pi, config.bloch_sphere_resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        axes.plot_surface(x, y, z, alpha=0.1, color='gray', edgecolor=config.grid_color, linewidth=0.2)
        axes.plot([-1.2, 1.2], [0, 0], [0, 0], 'w-', alpha=0.3, linewidth=0.5)
        axes.plot([0, 0], [-1.2, 1.2], [0, 0], 'w-', alpha=0.3, linewidth=0.5)
        axes.plot([0, 0], [0, 0], [-1.2, 1.2], 'w-', alpha=0.3, linewidth=0.5)
        colors = ['#FF6600', '#00FF66', '#6600FF', '#FFFF00', '#FF00FF', '#00FFFF']
        for i, (bx, by, bz) in enumerate(data.bloch_vectors):
            color = colors[i % len(colors)]
            axes.quiver(0, 0, 0, bx, by, bz, color=color, arrow_length_ratio=0.15, linewidth=2)
            axes.scatter([bx], [by], [bz], color=color, s=30, marker='o')
            axes.text(bx*1.15, by*1.15, bz*1.15, f'q{i}', color=color, fontsize=8)
        axes.set_xlim([-1.3, 1.3])
        axes.set_ylim([-1.3, 1.3])
        axes.set_zlim([-1.3, 1.3])
        axes.set_title(f'Bloch Sphere (Step {data.step})', color=config.text_color, fontsize=config.title_size)
        axes.set_facecolor(config.background_color)
        axes.xaxis.pane.fill = False
        axes.yaxis.pane.fill = False
        axes.zaxis.pane.fill = False
        axes.tick_params(colors=config.grid_color, labelsize=6)
        axes.set_xlabel('X', color=config.text_color, fontsize=8)
        axes.set_ylabel('Y', color=config.text_color, fontsize=8)
        axes.set_zlabel('Z', color=config.text_color, fontsize=8)


class PhasePlotRenderer(IVisualizationComponent):
    def render(self, data: QuantumStateSnapshot, axes: Any, config: VisualizerConfig) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            return
        phases = data.phases
        probs = data.probabilities
        non_zero = probs > config.probability_threshold
        angles = phases[non_zero]
        magnitudes = probs[non_zero]
        x = magnitudes * np.cos(angles)
        y = magnitudes * np.sin(angles)
        colors = np.angle(angles)
        scatter = axes.scatter(x, y, c=colors, cmap=config.colormap_phase,
                              s=magnitudes*500+20, alpha=0.7, edgecolors=config.text_color, linewidth=0.5)
        circle = Circle((0, 0), 1, fill=False, color=config.grid_color, linestyle='--', linewidth=0.5)
        axes.add_patch(circle)
        axes.axhline(y=0, color=config.grid_color, linestyle='-', linewidth=0.5, alpha=0.3)
        axes.axvline(x=0, color=config.grid_color, linestyle='-', linewidth=0.5, alpha=0.3)
        axes.set_xlim([-1.2, 1.2])
        axes.set_ylim([-1.2, 1.2])
        axes.set_xlabel('Real', color=config.text_color, fontsize=config.axis_label_size)
        axes.set_ylabel('Imaginary', color=config.text_color, fontsize=config.axis_label_size)
        axes.set_title(f'Phase Space (Step {data.step})', color=config.text_color, fontsize=config.title_size)
        axes.set_facecolor(config.background_color)
        axes.tick_params(colors=config.text_color)
        axes.set_aspect('equal')
        for spine in axes.spines.values():
            spine.set_color(config.grid_color)


class EntropyPlotRenderer(IVisualizationComponent):
    def render(self, snapshots: List[QuantumStateSnapshot], axes: Any, config: VisualizerConfig) -> None:
        steps = [s.step for s in snapshots]
        entropies = [s.entropy for s in snapshots]
        axes.plot(steps, entropies, 'o-', color='#FF6600', linewidth=2, markersize=6, markerfacecolor='#FF9933')
        axes.fill_between(steps, entropies, alpha=0.3, color='#FF6600')
        if len(entropies) > 0:
            max_entropy = max(entropies)
            axes.axhline(y=max_entropy, color='#00FF66', linestyle='--', alpha=0.5, label=f'Max: {max_entropy:.3f}')
        axes.set_xlabel('Gate Step', color=config.text_color, fontsize=config.axis_label_size)
        axes.set_ylabel('Entropy (bits)', color=config.text_color, fontsize=config.axis_label_size)
        axes.set_title('Shannon Entropy Evolution', color=config.text_color, fontsize=config.title_size)
        axes.set_facecolor(config.background_color)
        axes.tick_params(colors=config.text_color)
        axes.grid(True, alpha=0.3, color=config.grid_color)
        for spine in axes.spines.values():
            spine.set_color(config.grid_color)
        if len(entropies) > 0 and max(entropies) > 0:
            axes.legend(loc='upper right', facecolor=config.background_color,
                       edgecolor=config.grid_color, labelcolor=config.text_color)


class BackendComparisonRenderer(IVisualizationComponent):
    def render(self, results: List[BackendComparisonResult], axes: Any, config: VisualizerConfig) -> None:
        n_backends = len(results)
        n_states = len(results[0].probabilities) if results else 0
        if n_backends == 0 or n_states == 0:
            return
        width = 1.0 / (n_backends + 1)
        x = np.arange(n_states)
        colors = ['#FF6600', '#00FF66', '#6600FF']
        for i, result in enumerate(results):
            offset = (i - n_backends/2 + 0.5) * width
            axes.bar(x + offset, result.probabilities, width,
                    label=f'{result.backend_name.capitalize()}',
                    color=colors[i % len(colors)], alpha=0.7,
                    edgecolor=config.text_color, linewidth=0.3)
        bitstrings = [format(j, f'0{int(np.log2(n_states))}b') for j in range(n_states)]
        axes.set_xticks(x)
        axes.set_xticklabels(bitstrings, rotation=45, ha='right', fontsize=7, color=config.text_color)
        axes.set_ylabel('Probability', color=config.text_color, fontsize=config.axis_label_size)
        axes.set_title('Backend Comparison', color=config.text_color, fontsize=config.title_size)
        axes.set_facecolor(config.background_color)
        axes.tick_params(colors=config.text_color)
        axes.legend(loc='upper right', facecolor=config.background_color,
                   edgecolor=config.grid_color, labelcolor=config.text_color)
        for spine in axes.spines.values():
            spine.set_color(config.grid_color)


class QuantumStateAnalyzer:
    def __init__(self, config: VisualizerConfig):
        self.config = config

    def compute_probabilities(self, state: JointHilbertState) -> np.ndarray:
        probs = state.probabilities().numpy()
        return probs / (probs.sum() + 1e-12)

    def compute_phases(self, state: JointHilbertState) -> np.ndarray:
        amps = state.amplitudes
        if amps.dim() > 2:
            amps_flat = amps.sum(dim=(-2, -1))
        else:
            amps_flat = amps
        re = amps_flat[:, 0].numpy()
        im = amps_flat[:, 1].numpy()
        phases = np.arctan2(im, re)
        return phases

    def compute_entropy(self, probs: np.ndarray) -> float:
        p_nonzero = probs[probs > self.config.probability_threshold]
        if len(p_nonzero) == 0:
            return 0.0
        entropy = -np.sum(p_nonzero * np.log(p_nonzero + 1e-15)) / np.log(self.config.entropy_base)
        return float(entropy)

    def compute_bloch_vectors(self, state: JointHilbertState) -> List[Tuple[float, float, float]]:
        vectors = []
        for qubit in range(state.n_qubits):
            bx, by, bz = state.bloch_vector(qubit)
            vectors.append((float(bx), float(by), float(bz)))
        return vectors

    def create_snapshot(self, state: JointHilbertState, step: int, gate_name: str) -> QuantumStateSnapshot:
        probs = self.compute_probabilities(state)
        phases = self.compute_phases(state)
        entropy = self.compute_entropy(probs)
        bloch = self.compute_bloch_vectors(state)
        if state.amplitudes.dim() > 2:
            amps = state.amplitudes.sum(dim=(-2, -1)).numpy()
        else:
            amps = state.amplitudes.numpy()
        norm = float(np.sqrt((amps**2).sum()))
        return QuantumStateSnapshot(
            step=step,
            gate_name=gate_name,
            probabilities=probs,
            phases=phases,
            entropy=entropy,
            bloch_vectors=bloch,
            amplitudes=amps,
            most_probable=int(np.argmax(probs)),
            norm=norm
        )


class CircuitExecutor:
    def __init__(self, qc: QuantumComputer, config: VisualizerConfig):
        self.qc = qc
        self.config = config
        self.analyzer = QuantumStateAnalyzer(config)

    def execute_sequence(
        self,
        gates: Sequence[Tuple[str, List[int], Dict[str, float]]],
        n_qubits: int,
        backend_name: str
    ) -> List[QuantumStateSnapshot]:
        backend = self.qc._backends[backend_name]
        factory = self.qc._factory
        state = factory.all_zeros(n_qubits)
        snapshots = [self.analyzer.create_snapshot(state, 0, "INIT")]
        for step, (gate_name, targets, params) in enumerate(gates, 1):
            gate = _GATE_REGISTRY.get(gate_name)
            if gate is None:
                _LOG.warning("Unknown gate: %s", gate_name)
                continue
            state = gate.apply(state, backend, targets, params)
            snapshot = self.analyzer.create_snapshot(state, step, gate_name)
            snapshots.append(snapshot)
        return snapshots

    def compare_backends(
        self,
        gates: Sequence[Tuple[str, List[int], Dict[str, float]]],
        n_qubits: int,
        reference_backend: str = "hamiltonian"
    ) -> List[BackendComparisonResult]:
        results = []
        reference_probs = None
        for backend_name in self.config.backend_order:
            snapshots = self.execute_sequence(gates, n_qubits, backend_name)
            if not snapshots:
                continue
            final = snapshots[-1]
            probs = final.probabilities
            if reference_probs is None:
                reference_probs = probs
                fidelity = 1.0
            else:
                fidelity = float(np.sum(np.sqrt(reference_probs * probs))**2)
            top_indices = np.argsort(probs)[::-1][:4]
            top_states = [(int(i), float(probs[i])) for i in top_indices]
            result = BackendComparisonResult(
                backend_name=backend_name,
                probabilities=probs,
                entropy=final.entropy,
                most_probable=final.most_probable,
                top_states=top_states,
                bloch_vectors=final.bloch_vectors,
                fidelity_with_reference=fidelity
            )
            results.append(result)
        return results


class StandardCircuits:
    @staticmethod
    def bell_state() -> List[Tuple[str, List[int], Dict[str, float]]]:
        return [
            ("H", [0], {}),
            ("CNOT", [0, 1], {}),
        ]

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
        targets = list(range(n_qubits))
        gates.append(("MCZ", targets, {}))
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
        targets = list(range(n_qubits))
        gates.append(("MCZ", targets, {}))
        for i in range(n_qubits):
            gates.append(("X", [i], {}))
            gates.append(("H", [i], {}))
        return gates

    @staticmethod
    def custom_sequence(sequence: Sequence[str]) -> List[Tuple[str, List[int], Dict[str, float]]]:
        gates = []
        for i, gate_name in enumerate(sequence):
            if gate_name in ("H", "X", "Y", "Z", "S", "T"):
                gates.append((gate_name, [i % 2], {}))
            elif gate_name == "CNOT":
                gates.append((gate_name, [i % 2, (i + 1) % 2], {}))
            elif gate_name == "CZ":
                gates.append((gate_name, [i % 2, (i + 1) % 2], {}))
            else:
                gates.append((gate_name, [i % 2], {}))
        return gates


class FigureBuilder:
    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.prob_renderer = ProbabilityBarRenderer()
        self.bloch_renderer = BlochSphereRenderer()
        self.phase_renderer = PhasePlotRenderer()
        self.entropy_renderer = EntropyPlotRenderer()
        self.comparison_renderer = BackendComparisonRenderer()

    def build_evolution_figure(
        self,
        snapshots: List[QuantumStateSnapshot],
        backend_results: List[BackendComparisonResult]
    ) -> Any:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            _LOG.error("matplotlib not available")
            return None
        n_snapshots = min(4, len(snapshots))
        fig = plt.figure(figsize=(self.config.figure_size_x, self.config.figure_size_y),
                        dpi=self.config.figure_dpi)
        fig.patch.set_facecolor(self.config.background_color)
        for i in range(n_snapshots):
            snapshot = snapshots[i]
            ax1 = fig.add_subplot(3, n_snapshots, i + 1)
            self.prob_renderer.render(snapshot, ax1, self.config)
            ax2 = fig.add_subplot(3, n_snapshots, n_snapshots + i + 1, projection='3d')
            self.bloch_renderer.render(snapshot, ax2, self.config)
            ax3 = fig.add_subplot(3, n_snapshots, 2 * n_snapshots + i + 1)
            self.phase_renderer.render(snapshot, ax3, self.config)
        plt.tight_layout()
        return fig

    def build_summary_figure(
        self,
        snapshots: List[QuantumStateSnapshot],
        backend_results: List[BackendComparisonResult]
    ) -> Any:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            _LOG.error("matplotlib not available")
            return None
        fig = plt.figure(figsize=(self.config.figure_size_x, self.config.figure_size_y),
                        dpi=self.config.figure_dpi)
        fig.patch.set_facecolor(self.config.background_color)
        ax1 = fig.add_subplot(231)
        self.entropy_renderer.render(snapshots, ax1, self.config)
        ax2 = fig.add_subplot(232)
        self.comparison_renderer.render(backend_results, ax2, self.config)
        if snapshots:
            final = snapshots[-1]
            ax3 = fig.add_subplot(233)
            self.prob_renderer.render(final, ax3, self.config)
            ax4 = fig.add_subplot(234, projection='3d')
            self.bloch_renderer.render(final, ax4, self.config)
            ax5 = fig.add_subplot(235)
            self.phase_renderer.render(final, ax5, self.config)
        ax6 = fig.add_subplot(236)
        self._render_backend_fidelity(backend_results, ax6)
        plt.tight_layout()
        return fig

    def _render_backend_fidelity(self, results: List[BackendComparisonResult], axes: Any) -> None:
        names = [r.backend_name.capitalize() for r in results]
        fidelities = [r.fidelity_with_reference for r in results]
        colors = ['#FF6600', '#00FF66', '#6600FF'][:len(results)]
        bars = axes.bar(names, fidelities, color=colors, edgecolor=self.config.text_color, linewidth=0.5)
        axes.set_ylabel('Fidelity', color=self.config.text_color, fontsize=self.config.axis_label_size)
        axes.set_title('Backend Fidelity vs Reference', color=self.config.text_color, fontsize=self.config.title_size)
        axes.set_facecolor(self.config.background_color)
        axes.tick_params(colors=self.config.text_color)
        axes.set_ylim([0, 1.1])
        for bar, fid in zip(bars, fidelities):
            axes.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{fid:.6f}', ha='center', va='bottom',
                     fontsize=9, color=self.config.text_color)
        for spine in axes.spines.values():
            spine.set_color(self.config.grid_color)


class QuantumVisualizer:
    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.qc = None
        self.executor = None
        self.figure_builder = None
        self._initialize()

    def _initialize(self) -> None:
        _LOG.info("=" * 70)
        _LOG.info("QUANTUM STATE VISUALIZER INITIALIZING")
        _LOG.info("=" * 70)
        self._init_quantum_computer()
        self.executor = CircuitExecutor(self.qc, self.config)
        self.figure_builder = FigureBuilder(self.config)
        _LOG.info("Initialization complete")

    def _init_quantum_computer(self) -> None:
        if not QUANTUM_COMPUTER_AVAILABLE:
            raise RuntimeError("quantum_computer module not available")
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
        self.qc = QuantumComputer(qc_config)
        _LOG.info("Quantum computer initialized with backends: %s", list(self.qc._backends.keys()))

    def visualize_bell_state(self) -> VisualizationResult:
        _LOG.info("\n" + "=" * 60)
        _LOG.info("VISUALIZING BELL STATE")
        _LOG.info("=" * 60)
        gates = StandardCircuits.bell_state()
        n_qubits = 2
        all_snapshots = {}
        all_results = {}
        for backend_name in self.config.backend_order:
            snapshots = self.executor.execute_sequence(gates, n_qubits, backend_name)
            all_snapshots[backend_name] = snapshots
            _LOG.info("  %s backend: %d steps, final entropy=%.4f bits",
                     backend_name, len(snapshots), snapshots[-1].entropy if snapshots else 0)
        comparison = self.executor.compare_backends(gates, n_qubits)
        for result in comparison:
            _LOG.info("  %s: most_probable=|%s> entropy=%.4f fidelity=%.6f",
                     result.backend_name, format(result.most_probable, f'0{n_qubits}b'),
                     result.entropy, result.fidelity_with_reference)
        fig = self.figure_builder.build_summary_figure(all_snapshots["hamiltonian"], comparison)
        output_path = self._save_figure(fig, "bell_state_visualization")
        return VisualizationResult(
            snapshots=all_snapshots["hamiltonian"],
            backend_results=comparison,
            circuit_diagram="H(0) -> CNOT(0,1)",
            execution_time=0.0,
            total_steps=len(gates),
            output_path=output_path
        )

    def visualize_ghz_state(self, n_qubits: int = None) -> VisualizationResult:
        n_qubits = n_qubits or self.config.n_qubits_default
        _LOG.info("\n" + "=" * 60)
        _LOG.info("VISUALIZING GHZ STATE (%d qubits)", n_qubits)
        _LOG.info("=" * 60)
        gates = StandardCircuits.ghz_state(n_qubits)
        all_snapshots = {}
        for backend_name in self.config.backend_order:
            snapshots = self.executor.execute_sequence(gates, n_qubits, backend_name)
            all_snapshots[backend_name] = snapshots
            _LOG.info("  %s backend: %d steps, final entropy=%.4f bits",
                     backend_name, len(snapshots), snapshots[-1].entropy if snapshots else 0)
        comparison = self.executor.compare_backends(gates, n_qubits)
        for result in comparison:
            _LOG.info("  %s: most_probable=|%s> entropy=%.4f fidelity=%.6f",
                     result.backend_name, format(result.most_probable, f'0{n_qubits}b'),
                     result.entropy, result.fidelity_with_reference)
        fig = self.figure_builder.build_summary_figure(all_snapshots["hamiltonian"], comparison)
        output_path = self._save_figure(fig, f"ghz_{n_qubits}q_visualization")
        return VisualizationResult(
            snapshots=all_snapshots["hamiltonian"],
            backend_results=comparison,
            circuit_diagram=f"H(0) -> CNOT chain ({n_qubits} qubits)",
            execution_time=0.0,
            total_steps=len(gates),
            output_path=output_path
        )

    def visualize_qft(self, n_qubits: int = None) -> VisualizationResult:
        n_qubits = n_qubits or self.config.n_qubits_default
        _LOG.info("\n" + "=" * 60)
        _LOG.info("VISUALIZING QFT (%d qubits)", n_qubits)
        _LOG.info("=" * 60)
        gates = StandardCircuits.qft(n_qubits)
        all_snapshots = {}
        for backend_name in self.config.backend_order:
            snapshots = self.executor.execute_sequence(gates, n_qubits, backend_name)
            all_snapshots[backend_name] = snapshots
            _LOG.info("  %s backend: %d steps, final entropy=%.4f bits",
                     backend_name, len(snapshots), snapshots[-1].entropy if snapshots else 0)
        comparison = self.executor.compare_backends(gates, n_qubits)
        for result in comparison:
            _LOG.info("  %s: most_probable=|%s> entropy=%.4f fidelity=%.6f",
                     result.backend_name, format(result.most_probable, f'0{n_qubits}b'),
                     result.entropy, result.fidelity_with_reference)
        fig = self.figure_builder.build_summary_figure(all_snapshots["hamiltonian"], comparison)
        output_path = self._save_figure(fig, f"qft_{n_qubits}q_visualization")
        return VisualizationResult(
            snapshots=all_snapshots["hamiltonian"],
            backend_results=comparison,
            circuit_diagram=f"QFT ({n_qubits} qubits)",
            execution_time=0.0,
            total_steps=len(gates),
            output_path=output_path
        )

    def visualize_grover(self, n_qubits: int = None, marked_state: int = None) -> VisualizationResult:
        n_qubits = n_qubits or self.config.n_qubits_default
        marked_state = marked_state if marked_state is not None else self.config.grover_marked_state
        iterations = self.config.grover_iterations
        _LOG.info("\n" + "=" * 60)
        _LOG.info("VISUALIZING GROVER ALGORITHM (%d qubits, marked=|%s>, iters=%d)",
                 n_qubits, format(marked_state, f'0{n_qubits}b'), iterations)
        _LOG.info("=" * 60)
        gates = []
        for i in range(n_qubits):
            gates.append(("H", [i], {}))
        for _ in range(iterations):
            gates.extend(StandardCircuits.grover_oracle(n_qubits, marked_state))
            gates.extend(StandardCircuits.grover_diffusion(n_qubits))
        all_snapshots = {}
        for backend_name in self.config.backend_order:
            snapshots = self.executor.execute_sequence(gates, n_qubits, backend_name)
            all_snapshots[backend_name] = snapshots
            if snapshots:
                final_prob = snapshots[-1].probabilities[marked_state]
                _LOG.info("  %s backend: P(|%s>)=%.4f entropy=%.4f bits",
                         backend_name, format(marked_state, f'0{n_qubits}b'),
                         final_prob, snapshots[-1].entropy)
        comparison = self.executor.compare_backends(gates, n_qubits)
        for result in comparison:
            _LOG.info("  %s: most_probable=|%s> entropy=%.4f fidelity=%.6f",
                     result.backend_name, format(result.most_probable, f'0{n_qubits}b'),
                     result.entropy, result.fidelity_with_reference)
        fig = self.figure_builder.build_summary_figure(all_snapshots["hamiltonian"], comparison)
        output_path = self._save_figure(fig, f"grover_{n_qubits}q_m{marked_state}_visualization")
        return VisualizationResult(
            snapshots=all_snapshots["hamiltonian"],
            backend_results=comparison,
            circuit_diagram=f"Grover (n={n_qubits}, marked={marked_state}, iter={iterations})",
            execution_time=0.0,
            total_steps=len(gates),
            output_path=output_path
        )

    def visualize_custom_circuit(
        self,
        gates: List[Tuple[str, List[int], Dict[str, float]]],
        n_qubits: int,
        name: str = "custom"
    ) -> VisualizationResult:
        _LOG.info("\n" + "=" * 60)
        _LOG.info("VISUALIZING CUSTOM CIRCUIT: %s (%d qubits)", name, n_qubits)
        _LOG.info("=" * 60)
        all_snapshots = {}
        for backend_name in self.config.backend_order:
            snapshots = self.executor.execute_sequence(gates, n_qubits, backend_name)
            all_snapshots[backend_name] = snapshots
            _LOG.info("  %s backend: %d steps, final entropy=%.4f bits",
                     backend_name, len(snapshots), snapshots[-1].entropy if snapshots else 0)
        comparison = self.executor.compare_backends(gates, n_qubits)
        for result in comparison:
            _LOG.info("  %s: most_probable=|%s> entropy=%.4f fidelity=%.6f",
                     result.backend_name, format(result.most_probable, f'0{n_qubits}b'),
                     result.entropy, result.fidelity_with_reference)
        fig = self.figure_builder.build_summary_figure(all_snapshots["hamiltonian"], comparison)
        safe_name = name.replace(" ", "_").replace("/", "_")
        output_path = self._save_figure(fig, f"{safe_name}_visualization")
        return VisualizationResult(
            snapshots=all_snapshots["hamiltonian"],
            backend_results=comparison,
            circuit_diagram=name,
            execution_time=0.0,
            total_steps=len(gates),
            output_path=output_path
        )

    def run_all_visualizations(self) -> Dict[str, VisualizationResult]:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("RUNNING ALL QUANTUM STATE VISUALIZATIONS")
        _LOG.info("=" * 70)
        results = {}
        results["bell_state"] = self.visualize_bell_state()
        results["ghz_state"] = self.visualize_ghz_state()
        results["qft"] = self.visualize_qft()
        results["grover"] = self.visualize_grover()
        _LOG.info("\n" + "=" * 70)
        _LOG.info("ALL VISUALIZATIONS COMPLETE")
        _LOG.info("=" * 70)
        self._print_summary(results)
        return results

    def _save_figure(self, fig: Any, name: str) -> str:
        if fig is None:
            return ""
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, f"{name}.png")
        fig.savefig(output_path, dpi=self.config.figure_dpi,
                   facecolor=self.config.background_color, bbox_inches='tight')
        _LOG.info("Saved: %s (%dx%d pixels)", output_path,
                 self.config.figure_size_x * self.config.figure_dpi,
                 self.config.figure_size_y * self.config.figure_dpi)
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        return output_path

    def _print_summary(self, results: Dict[str, VisualizationResult]) -> None:
        _LOG.info("\n" + "-" * 60)
        _LOG.info("VISUALIZATION SUMMARY")
        _LOG.info("-" * 60)
        for name, result in results.items():
            _LOG.info("  %s:", name.replace("_", " ").upper())
            _LOG.info("    Output: %s", result.output_path)
            _LOG.info("    Steps: %d", result.total_steps)
            if result.backend_results:
                for br in result.backend_results:
                    _LOG.info("    %s: entropy=%.4f, fidelity=%.6f",
                             br.backend_name, br.entropy, br.fidelity_with_reference)


def main():
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    if not QUANTUM_COMPUTER_AVAILABLE:
        print("ERROR: quantum_computer module is required.")
        sys.exit(1)
    import argparse
    parser = argparse.ArgumentParser(description="Quantum State Visualizer")
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
    parser.add_argument("--run-all", action="store_true", help="Run all visualizations")
    parser.add_argument("--bell", action="store_true", help="Visualize Bell state")
    parser.add_argument("--ghz", action="store_true", help="Visualize GHZ state")
    parser.add_argument("--qft", action="store_true", help="Visualize QFT")
    parser.add_argument("--grover", action="store_true", help="Visualize Grover algorithm")
    args = parser.parse_args()
    config = VisualizerConfig(
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        expansion_dim=args.expansion_dim,
        hamiltonian_checkpoint=args.hamiltonian_checkpoint,
        schrodinger_checkpoint=args.schrodinger_checkpoint,
        dirac_checkpoint=args.dirac_checkpoint,
        device=args.device,
        random_seed=args.seed,
        n_qubits_default=args.n_qubits,
        output_dir=args.output_dir,
    )
    visualizer = QuantumVisualizer(config)
    if args.run_all:
        visualizer.run_all_visualizations()
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
            visualizer.run_all_visualizations()


if __name__ == "__main__":
    main()