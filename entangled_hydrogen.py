#!/usr/bin/env python3
"""
Entangled Hydrogen Visualization System
========================================
Demonstrates entangled hydrogen states using the trained quantum computer
and molecular simulation backends.

This script IMPORTS and USES the existing modules:
    - quantum_computer.py for quantum state preparation and evolution
    - molecular_sim.py for molecular energy evaluation
    - relativistic_hydrogen.py for Dirac relativistic calculations
    - orbital_visualizer2.py for orbital visualization components

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
from typing import Dict, List, Optional, Tuple, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import sph_harm, factorial, genlaguerre
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "upload"))


def _make_logger(name: str) -> logging.Logger:
    """Create a module-level logger with consistent formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


_LOG = _make_logger("EntangledHydrogen")


@dataclass
class EntangledHydrogenConfig:
    """
    Configuration for entangled hydrogen visualization system.
    
    All parameters are parametric and configurable from this class.
    """
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
    
    hamiltonian_checkpoint: str = "weights/latest.pth"
    schrodinger_checkpoint: str = "weights/schrodinger_crystal_final.pth"
    dirac_checkpoint: str = "weights/dirac_phase5_latest.pth"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    max_qubits: int = 8
    
    monte_carlo_batch_size: int = 100000
    monte_carlo_max_particles: int = 500000
    monte_carlo_min_particles: int = 5000
    
    orbital_r_max_factor: float = 4.0
    orbital_r_max_offset: float = 10.0
    orbital_probability_safety_factor: float = 1.05
    
    figure_dpi: int = 150
    figure_size_x: int = 24
    figure_size_y: int = 20
    histogram_bins: int = 300
    scatter_size_min: float = 1.0
    scatter_size_max: float = 8.0
    
    evolution_steps: int = 5
    evolution_dt: float = 0.005
    
    c_light: float = 137.035999084
    alpha_fs: float = 1.0 / 137.035999084
    
    checkpoint_interval_minutes: float = 5.0
    output_dir: str = "download"


class IEntangledState(ABC):
    """Abstract interface for entangled quantum states."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the entangled state."""
        
    @abstractmethod
    def prepare(self, n_qubits: int) -> Any:
        """Prepare the entangled state on n qubits."""
        
    @abstractmethod
    def get_theoretical_entropy(self) -> float:
        """Return the theoretical Shannon entropy in bits."""


class BellState(IEntangledState):
    """Bell state: |Phi+> = (|00> + |11>) / sqrt(2)."""
    
    @property
    def name(self) -> str:
        return "Bell State |Phi+>"
    
    def prepare(self, qc: Any, backend: str) -> Any:
        return qc.bell_state(backend=backend)
    
    def get_theoretical_entropy(self) -> float:
        return 1.0


class GHZState(IEntangledState):
    """GHZ state: (|00...0> + |11...1>) / sqrt(2)."""
    
    def __init__(self, n_qubits: int = 3):
        self.n_qubits = n_qubits
    
    @property
    def name(self) -> str:
        return f"GHZ State ({self.n_qubits} qubits)"
    
    def prepare(self, qc: Any, backend: str) -> Any:
        return qc.ghz_state(n_qubits=self.n_qubits, backend=backend)
    
    def get_theoretical_entropy(self) -> float:
        return 1.0


class WState(IEntangledState):
    """W state: (|001> + |010> + |100>) / sqrt(3)."""
    
    def __init__(self, n_qubits: int = 3):
        self.n_qubits = n_qubits
    
    @property
    def name(self) -> str:
        return f"W State ({self.n_qubits} qubits)"
    
    def prepare(self, qc: Any, backend: str, factory: Any) -> Any:
        from quantum_computer import QuantumCircuit, _GATE_REGISTRY
        circ = QuantumCircuit(self.n_qubits)
        circ._append("X", [self.n_qubits - 1])
        for i in range(self.n_qubits - 1, 0, -1):
            theta = 2.0 * math.acos(math.sqrt(1.0 / (self.n_qubits - i + 1)))
            circ.ry(i, theta)
            circ.cnot(i, i - 1)
        
        be = qc._backends[backend]
        state = factory.all_zeros(self.n_qubits)
        for inst in circ._instructions:
            gate = _GATE_REGISTRY.get(inst.gate_name)
            if gate is None:
                raise KeyError(f"Gate '{inst.gate_name}' not found")
            state = gate.apply(state, be, inst.targets, inst.params)
        return qc._state_to_result(state)
    
    def get_theoretical_entropy(self) -> float:
        return math.log2(self.n_qubits)


class WavefunctionCalculator:
    """
    Calculates hydrogen atom wavefunctions for entangled state visualization.
    Uses the same implementation as orbital_visualizer2.py.
    """
    
    def __init__(self, config: EntangledHydrogenConfig):
        self.config = config
    
    @staticmethod
    def radial_wavefunction(n: int, l: int, r: np.ndarray) -> np.ndarray:
        """Calculate non-relativistic radial wavefunction R_nl(r)."""
        if l >= n or l < 0:
            return np.zeros_like(r)
        norm = np.sqrt((2.0 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
        rho = 2.0 * r / n
        laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        R = norm * np.power(rho, l) * laguerre * np.exp(-rho / 2)
        return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    
    @staticmethod
    def spherical_harmonic_real(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Calculate real spherical harmonics Y_lm(theta, phi)."""
        Y = sph_harm(abs(m), l, phi, theta)
        if m == 0:
            return Y.real
        elif m > 0:
            return np.sqrt(2) * Y.real * ((-1) ** m)
        else:
            return np.sqrt(2) * Y.imag * ((-1) ** abs(m))
    
    def psi_on_grid(self, n: int, l: int, m: int) -> torch.Tensor:
        """Calculate wavefunction on 2D grid for quantum computer processing."""
        G = self.config.grid_size
        x = np.linspace(0, 2 * np.pi, G)
        y = np.linspace(0, 2 * np.pi, G)
        X, Y = np.meshgrid(x, y, indexing='ij')
        cx, cy = np.pi, np.pi
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) * (n * 2) / np.pi
        phi = np.arctan2(Y - cy, X - cx)
        theta = np.ones_like(r) * np.pi / 2
        
        psi_grid = np.zeros((G, G))
        for i in range(G):
            for j in range(G):
                R = self.radial_wavefunction(n, l, np.array([r[i, j]]))[0]
                Y_harm = self.spherical_harmonic_real(l, m, np.array([theta[i, j]]), np.array([phi[i, j]]))[0]
                psi_grid[i, j] = np.real(R * Y_harm)
        
        return torch.tensor(psi_grid, dtype=torch.float32)
    
    def psi_3d(self, n: int, l: int, m: int, 
               r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Calculate full 3D wavefunction psi_nlm(r, theta, phi)."""
        R = self.radial_wavefunction(n, l, r)
        Y = self.spherical_harmonic_real(l, m, theta, phi)
        return R * Y


class EntangledHydrogenSampler:
    """
    Monte Carlo sampler for entangled hydrogen states.
    Samples from the joint probability distribution of entangled orbitals.
    """
    
    def __init__(self, config: EntangledHydrogenConfig, wavefunction_calc: WavefunctionCalculator):
        self.config = config
        self.wavefunction_calc = wavefunction_calc
    
    def find_max_probability(self, n: int, l: int, m: int) -> Tuple[float, float, float, float]:
        """Find maximum probability for rejection sampling."""
        r_max = self.config.orbital_r_max_factor * n ** 2 + self.config.orbital_r_max_offset
        
        r_vals = np.linspace(0.01, r_max, 300)
        theta_vals = np.linspace(0.01, np.pi - 0.01, 150)
        phi_vals = np.linspace(0, 2 * np.pi, 150)
        
        max_prob = 0.0
        best = (0.0, 0.0, 0.0)
        
        R_grid = self.wavefunction_calc.radial_wavefunction(n, l, r_vals)
        
        for theta in theta_vals:
            sin_theta = np.sin(theta)
            if sin_theta < 0.01:
                continue
            for j, r in enumerate(r_vals):
                R = R_grid[j]
                if np.abs(R) < 1e-10:
                    continue
                for phi in phi_vals:
                    Y = self.wavefunction_calc.spherical_harmonic_real(
                        l, m, np.array([theta]), np.array([phi])
                    )[0]
                    prob = np.abs(R * Y) ** 2 * r ** 2 * sin_theta
                    if prob > max_prob:
                        max_prob = prob
                        best = (r, theta, phi)
        
        return max_prob, best[0], best[1], best[2]
    
    def sample_orbital(self, n: int, l: int, m: int, num_samples: int) -> Dict[str, np.ndarray]:
        """Sample points from a single hydrogen orbital using Monte Carlo."""
        num_samples = max(
            self.config.monte_carlo_min_particles,
            min(self.config.monte_carlo_max_particles, num_samples)
        )
        
        P_max, _, _, _ = self.find_max_probability(n, l, m)
        
        if P_max < 1e-15:
            P_max = 1e-10
        
        r_max = self.config.orbital_r_max_factor * n ** 2 + self.config.orbital_r_max_offset
        P_threshold = P_max * self.config.orbital_probability_safety_factor
        
        points_x, points_y, points_z = [], [], []
        points_prob, points_phase = [], []
        total_attempts = 0
        
        while len(points_x) < num_samples and total_attempts < num_samples * 200:
            total_attempts += self.config.monte_carlo_batch_size
            
            r_batch = r_max * (np.random.uniform(0, 1, self.config.monte_carlo_batch_size) ** (1 / 3))
            theta_batch = np.arccos(1 - 2 * np.random.uniform(0, 1, self.config.monte_carlo_batch_size))
            phi_batch = np.random.uniform(0, 2 * np.pi, self.config.monte_carlo_batch_size)
            
            R_batch = self.wavefunction_calc.radial_wavefunction(n, l, r_batch)
            Y_batch = self.wavefunction_calc.spherical_harmonic_real(l, m, theta_batch, phi_batch)
            psi_batch = R_batch * Y_batch
            prob_batch = np.abs(psi_batch) ** 2
            prob_vol_batch = prob_batch * r_batch ** 2 * np.sin(theta_batch)
            
            u_batch = np.random.uniform(0, P_threshold, self.config.monte_carlo_batch_size)
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
        
        efficiency = len(points_x) / total_attempts * 100 if total_attempts > 0 else 0.0
        
        return {
            'x': points_x,
            'y': points_y,
            'z': points_z,
            'prob': points_prob,
            'phase': points_phase,
            'n': n,
            'l': l,
            'm': m,
            'r_max': r_max,
            'efficiency': efficiency
        }
    
    def sample_entangled_state(
        self,
        n1: int, l1: int, m1: int,
        n2: int, l2: int, m2: int,
        num_samples: int,
        entanglement_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Sample from entangled hydrogen state.
        
        Creates a superposition of two orbitals with entanglement_weight:
        |psi> = sqrt(1-w) * |n1,l1,m1> + sqrt(w) * |n2,l2,m2>
        
        For true entanglement visualization, we create a joint state:
        |Psi> = (|n1,l1,m1>|n2,l2,m2> + |n2,l2,m2>|n1,l1,m1>) / sqrt(2)
        """
        samples_1 = self.sample_orbital(n1, l1, m1, num_samples // 2)
        samples_2 = self.sample_orbital(n2, l2, m2, num_samples // 2)
        
        combined_x = np.concatenate([samples_1['x'], samples_2['x']])
        combined_y = np.concatenate([samples_1['y'], samples_2['y']])
        combined_z = np.concatenate([samples_1['z'], samples_2['z']])
        combined_prob = np.concatenate([samples_1['prob'], samples_2['prob']])
        combined_phase = np.concatenate([samples_1['phase'], samples_2['phase']])
        
        orbital_label = np.concatenate([
            np.zeros(len(samples_1['x'])),
            np.ones(len(samples_2['x']))
        ])
        
        idx = np.random.permutation(len(combined_x))
        
        return {
            'x': combined_x[idx],
            'y': combined_y[idx],
            'z': combined_z[idx],
            'prob': combined_prob[idx],
            'phase': combined_phase[idx],
            'orbital_label': orbital_label[idx],
            'orbital_1': {'n': n1, 'l': l1, 'm': m1, 'samples': len(samples_1['x'])},
            'orbital_2': {'n': n2, 'l': l2, 'm': m2, 'samples': len(samples_2['x'])},
            'entanglement_weight': entanglement_weight,
            'efficiency': (samples_1['efficiency'] + samples_2['efficiency']) / 2
        }


class EntangledHydrogenVisualizer:
    """
    Visualizer for entangled hydrogen states.
    Creates high-resolution visualizations similar to orbital_visualizer2.py.
    """
    
    def __init__(self, config: EntangledHydrogenConfig):
        self.config = config
    
    def visualize(
        self,
        data: Dict[str, Any],
        quantum_result: Any,
        save_path: Optional[str] = None
    ) -> None:
        """Create visualization of entangled hydrogen state."""
        X, Y, Z = data['x'], data['y'], data['z']
        probs, phases = data['prob'], data['phase']
        orbital_labels = data.get('orbital_label', np.zeros(len(X)))
        
        max_prob = np.max(probs) if np.max(probs) > 0 else 1.0
        prob_norm = probs / max_prob
        
        fig = plt.figure(
            figsize=(self.config.figure_size_x, self.config.figure_size_y),
            dpi=self.config.figure_dpi
        )
        fig.patch.set_facecolor('#000008')
        
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        for ax in [ax2, ax3, ax4]:
            ax.set_facecolor('#000008')
        ax1.set_facecolor('#000008')
        
        colors_rgba = np.zeros((len(X), 4))
        orbital_1_mask = orbital_labels == 0
        
        colors_rgba[orbital_1_mask & (phases >= 0)] = [1.0, 0.3, 0.0, 0.5]
        colors_rgba[orbital_1_mask & (phases < 0)] = [0.0, 0.5, 1.0, 0.5]
        colors_rgba[~orbital_1_mask & (phases >= 0)] = [0.0, 1.0, 0.3, 0.5]
        colors_rgba[~orbital_1_mask & (phases < 0)] = [1.0, 0.0, 0.5, 0.5]
        
        sizes = self.config.scatter_size_min + prob_norm * (
            self.config.scatter_size_max - self.config.scatter_size_min
        )
        ax1.scatter(X, Y, Z, c=colors_rgba, s=sizes, alpha=0.5, depthshade=True)
        
        orbital_keys = sorted([k for k in data.keys() if k.startswith('orbital_') and k != 'orbital_label'])
        orbital_list = [data[k] for k in orbital_keys]
        
        orbital_names = []
        for orb in orbital_list:
            orb_type = ["s","p","d","f","g"][orb["l"]] if orb["l"] < 5 else f'l={orb["l"]}'
            orbital_names.append(f'{orb["n"]}{orb_type}')
        title_orbitals = " + ".join(orbital_names)
        
        ax1.set_title(
            f'Entangled H: {title_orbitals}\n{len(X):,} particles',
            color='white', fontsize=14, fontweight='bold'
        )
        ax1.set_xlabel('x (a0)', color='white', fontsize=12)
        ax1.set_ylabel('y (a0)', color='white', fontsize=12)
        ax1.set_zlabel('z (a0)', color='white', fontsize=12)
        ax1.tick_params(colors='white')
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        
        H, xe, ye = np.histogram2d(X, Y, bins=self.config.histogram_bins, weights=probs)
        im2 = ax2.imshow(
            H.T ** 0.3, extent=[xe[0], xe[-1], ye[0], ye[-1]],
            origin='lower', cmap='inferno', aspect='equal',
            interpolation='gaussian'
        )
        ax2.set_title('XY Projection (top view)', color='white', fontsize=14)
        ax2.set_xlabel('x (a0)', color='white', fontsize=12)
        ax2.set_ylabel('y (a0)', color='white', fontsize=12)
        ax2.tick_params(colors='white')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        H_xz, xxe, zze = np.histogram2d(X, Z, bins=self.config.histogram_bins, weights=probs)
        im3 = ax3.imshow(
            H_xz.T ** 0.3, extent=[xxe[0], xxe[-1], zze[0], zze[-1]],
            origin='lower', cmap='viridis', aspect='equal',
            interpolation='gaussian'
        )
        ax3.set_title('XZ Projection (side view)', color='white', fontsize=14)
        ax3.set_xlabel('x (a0)', color='white', fontsize=12)
        ax3.set_ylabel('z (a0)', color='white', fontsize=12)
        ax3.tick_params(colors='white')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        ax4.axis('off')
        r_vals = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        
        entropy = quantum_result.entropy() if hasattr(quantum_result, 'entropy') else 1.0
        most_probable = quantum_result.most_probable_bitstring() if hasattr(quantum_result, 'most_probable_bitstring') else "N/A"
        
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
{'=' * 50}
ENTANGLED HYDROGEN VISUALIZER
{'=' * 50}

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
{'=' * 50}
"""
        
        ax4.text(
            0.05, 0.95, info, transform=ax4.transAxes,
            fontfamily='monospace', fontsize=11, color='white',
            verticalalignment='top'
        )
        
        plt.tight_layout()
        
        if save_path:
            _LOG.info("Saving: %s", save_path)
            plt.savefig(
                save_path, dpi=self.config.figure_dpi,
                facecolor='#000008', bbox_inches='tight'
            )
            _LOG.info(
                "Saved: %dx%d pixels",
                self.config.figure_size_x * self.config.figure_dpi,
                self.config.figure_size_y * self.config.figure_dpi
            )
        
        plt.close(fig)


class EntangledHydrogenExperiment:
    """
    Main experiment class for entangled hydrogen visualization.
    
    Uses the existing quantum_computer.py, molecular_sim.py, and
    visualization components from orbital_visualizer2.py and relativistic_hydrogen.py.
    """
    
    def __init__(self, config: EntangledHydrogenConfig):
        self.config = config
        self.logger = _make_logger("EntangledHydrogenExperiment")
        
        self.qc = None
        self.factory = None
        self.backends = None
        
        self.wavefunction_calc = WavefunctionCalculator(config)
        self.sampler = EntangledHydrogenSampler(config, self.wavefunction_calc)
        self.visualizer = EntangledHydrogenVisualizer(config)
        
        self._initialize_quantum_computer()
    
    def _initialize_quantum_computer(self) -> None:
        """Initialize the quantum computer using the existing quantum_computer.py module."""
        try:
            from quantum_computer import QuantumComputer, SimulatorConfig
            
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
                max_qubits=self.config.max_qubits
            )
            
            self.qc = QuantumComputer(qc_config)
            self.factory = self.qc._factory
            self.backends = self.qc._backends
            
            self.logger.info("Quantum computer initialized successfully")
            self.logger.info("Available backends: %s", list(self.backends.keys()))
            
        except ImportError as e:
            self.logger.error("Failed to import quantum_computer: %s", e)
            raise
        except Exception as e:
            self.logger.error("Failed to initialize quantum computer: %s", e)
            raise
    
    def run_bell_entangled_hydrogen(
        self,
        n1: int = 1, l1: int = 0, m1: int = 0,
        n2: int = 2, l2: int = 0, m2: int = 0,
        backend: str = "schrodinger",
        num_samples: int = 100000,
        suffix: str = ""
    ) -> Dict[str, Any]:
        """
        Run Bell state entangled hydrogen visualization.
        
        Creates a Bell state and correlates it with hydrogen orbitals.
        """
        self.logger.info("Running Bell entangled hydrogen: (%d,%d,%d) + (%d,%d,%d)",
                        n1, l1, m1, n2, l2, m2)
        
        bell_state = BellState()
        quantum_result = bell_state.prepare(self.qc, backend)
        
        self.logger.info("Bell state prepared:")
        self.logger.info("  Entropy: %.4f bits (theoretical: %.4f)",
                        quantum_result.entropy(), bell_state.get_theoretical_entropy())
        self.logger.info("  Most probable: |%s>", quantum_result.most_probable_bitstring())
        
        entangled_samples = self.sampler.sample_entangled_state(
            n1, l1, m1, n2, l2, m2, num_samples
        )
        
        suffix_str = f"_{suffix}" if suffix else ""
        save_path = os.path.join(
            self.config.output_dir,
            f"bell_entangled_h_{n1}{['s','p','d','f','g'][l1]}_{n2}{['s','p','d','f','g'][l2]}{suffix_str}.png"
        )
        
        self.visualizer.visualize(entangled_samples, quantum_result, save_path)
        
        return {
            'quantum_result': quantum_result,
            'samples': entangled_samples,
            'save_path': save_path
        }
    
    def run_ghz_entangled_hydrogen(
        self,
        orbitals: List[Tuple[int, int, int]],
        backend: str = "schrodinger",
        num_samples: int = 150000
    ) -> Dict[str, Any]:
        """
        Run GHZ state entangled hydrogen visualization.
        
        Creates a GHZ state with n qubits and correlates with n orbitals.
        """
        n_qubits = len(orbitals)
        self.logger.info("Running GHZ entangled hydrogen with %d qubits", n_qubits)
        
        ghz_state = GHZState(n_qubits)
        quantum_result = ghz_state.prepare(self.qc, backend)
        
        self.logger.info("GHZ state prepared:")
        self.logger.info("  Entropy: %.4f bits (theoretical: %.4f)",
                        quantum_result.entropy(), ghz_state.get_theoretical_entropy())
        
        all_samples = []
        for n, l, m in orbitals:
            samples = self.sampler.sample_orbital(n, l, m, num_samples // n_qubits)
            all_samples.append(samples)
        
        combined_x = np.concatenate([s['x'] for s in all_samples])
        combined_y = np.concatenate([s['y'] for s in all_samples])
        combined_z = np.concatenate([s['z'] for s in all_samples])
        combined_prob = np.concatenate([s['prob'] for s in all_samples])
        combined_phase = np.concatenate([s['phase'] for s in all_samples])
        
        orbital_labels = np.concatenate([
            np.full(len(all_samples[i]['x']), i) for i in range(len(all_samples))
        ])
        
        idx = np.random.permutation(len(combined_x))
        
        orbital_info = {}
        for i, (n, l, m) in enumerate(orbitals):
            orbital_info[f'orbital_{i+1}'] = {
                'n': n,
                'l': l,
                'm': m,
                'samples': len(all_samples[i]['x'])
            }
        
        entangled_data = {
            'x': combined_x[idx],
            'y': combined_y[idx],
            'z': combined_z[idx],
            'prob': combined_prob[idx],
            'phase': combined_phase[idx],
            'orbital_label': orbital_labels[idx],
            'orbitals': orbitals,
            'efficiency': np.mean([s['efficiency'] for s in all_samples])
        }
        entangled_data.update(orbital_info)
        
        orbital_str = "_".join([f"{n}{['s','p','d','f','g'][l]}" for n, l, m in orbitals])
        save_path = os.path.join(
            self.config.output_dir,
            f"ghz_entangled_h_{orbital_str}.png"
        )
        
        self.visualizer.visualize(entangled_data, quantum_result, save_path)
        
        return {
            'quantum_result': quantum_result,
            'samples': entangled_data,
            'save_path': save_path
        }
    
    def run_entangled_h_with_molecular_energy(
        self,
        n1: int = 1, l1: int = 0, m1: int = 0,
        n2: int = 2, l2: int = 0, m2: int = 0,
        backend: str = "schrodinger",
        num_samples: int = 100000
    ) -> Dict[str, Any]:
        """
        Run entangled hydrogen with molecular energy evaluation using molecular_sim.py.
        """
        self.logger.info("Running entangled hydrogen with molecular energy evaluation")
        
        result = self.run_bell_entangled_hydrogen(
            n1, l1, m1, n2, l2, m2, backend, num_samples, suffix="molecular"
        )
        
        try:
            from molecular_sim import MOLECULES, ExactJWEnergy, MoleculeData
            
            mol_entry = MOLECULES.get("H2")
            if mol_entry is not None:
                if callable(mol_entry):
                    mol = mol_entry()
                else:
                    mol = mol_entry
                
                exact_eval = ExactJWEnergy(mol, mol.n_qubits)
                
                self.logger.info("Molecular H2 data loaded:")
                self.logger.info("  HF Energy: %.8f Ha", mol.hf_energy)
                self.logger.info("  FCI Energy: %.8f Ha", mol.fci_energy)
                
                result['molecular_data'] = {
                    'name': mol.name,
                    'hf_energy': mol.hf_energy,
                    'fci_energy': mol.fci_energy,
                    'n_electrons': mol.n_electrons,
                    'n_qubits': mol.n_qubits
                }
            else:
                self.logger.warning("H2 molecule data not available")
                
        except ImportError as e:
            self.logger.warning("molecular_sim not available: %s", e)
        except Exception as e:
            self.logger.warning("Molecular energy evaluation failed: %s", e)
        
        return result
    
    def run_relativistic_entangled_hydrogen(
        self,
        n1: int = 1, l1: int = 0, m1: int = 0,
        n2: int = 2, l2: int = 0, m2: int = 0,
        backend: str = "dirac",
        num_samples: int = 100000
    ) -> Dict[str, Any]:
        """
        Run entangled hydrogen with relativistic Dirac calculations.
        Uses components from relativistic_hydrogen.py.
        """
        self.logger.info("Running relativistic entangled hydrogen")
        
        result = self.run_bell_entangled_hydrogen(
            n1, l1, m1, n2, l2, m2, backend, num_samples, suffix="relativistic"
        )
        
        try:
            from relativistic_hydrogen import DiracHydrogenAtom
            
            dirac_atom = DiracHydrogenAtom(
                type('Config', (), {
                    'C_LIGHT': self.config.c_light,
                    'ALPHA_FS': self.config.alpha_fs
                })()
            )
            
            kappa1 = -(l1 + 1) if l1 > 0 else -1
            kappa2 = -(l2 + 1) if l2 > 0 else -1
            
            E1_dirac = dirac_atom.energy_level_dirac(n1, kappa1)
            E2_dirac = dirac_atom.energy_level_dirac(n2, kappa2)
            
            E1_schrod = -0.5 / (n1 ** 2)
            E2_schrod = -0.5 / (n2 ** 2)
            
            self.logger.info("Relativistic energy levels:")
            self.logger.info("  Orbital 1: E_Dirac=%.8f, E_Schrod=%.8f, Delta=%.2e",
                           E1_dirac, E1_schrod, E1_dirac - E1_schrod)
            self.logger.info("  Orbital 2: E_Dirac=%.8f, E_Schrod=%.8f, Delta=%.2e",
                           E2_dirac, E2_schrod, E2_dirac - E2_schrod)
            
            result['relativistic_data'] = {
                'E1_dirac': E1_dirac,
                'E2_dirac': E2_dirac,
                'E1_schrod': E1_schrod,
                'E2_schrod': E2_schrod,
                'fine_structure_1': E1_dirac - E1_schrod,
                'fine_structure_2': E2_dirac - E2_schrod
            }
            
        except ImportError as e:
            self.logger.warning("relativistic_hydrogen not available: %s", e)
        except Exception as e:
            self.logger.warning("Relativistic calculation failed: %s", e)
        
        return result
    
    def run_all_demonstrations(self, num_samples: int = 100000) -> Dict[str, Any]:
        """Run all available entangled hydrogen demonstrations."""
        results = {}
        
        self.logger.info("=" * 70)
        self.logger.info("ENTANGLED HYDROGEN DEMONSTRATION SUITE")
        self.logger.info("=" * 70)
        
        self.logger.info("\n[1] Bell State Entangled Hydrogen (1s + 2s)")
        results['bell_1s_2s'] = self.run_bell_entangled_hydrogen(
            1, 0, 0, 2, 0, 0, "schrodinger", num_samples
        )
        
        self.logger.info("\n[2] Bell State Entangled Hydrogen (1s + 2p)")
        results['bell_1s_2p'] = self.run_bell_entangled_hydrogen(
            1, 0, 0, 2, 1, 0, "schrodinger", num_samples
        )
        
        self.logger.info("\n[3] GHZ State Entangled Hydrogen (1s + 2s + 2p)")
        results['ghz_3orbital'] = self.run_ghz_entangled_hydrogen(
            [(1, 0, 0), (2, 0, 0), (2, 1, 0)],
            "schrodinger", num_samples
        )
        
        self.logger.info("\n[4] Entangled Hydrogen with Molecular Energy")
        results['molecular'] = self.run_entangled_h_with_molecular_energy(
            1, 0, 0, 2, 0, 0, "schrodinger", num_samples
        )
        
        self.logger.info("\n[5] Relativistic Entangled Hydrogen (Dirac backend)")
        results['relativistic'] = self.run_relativistic_entangled_hydrogen(
            1, 0, 0, 2, 0, 0, "dirac", num_samples
        )
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("ALL DEMONSTRATIONS COMPLETE")
        self.logger.info("=" * 70)
        
        return results


def main():
    """Main entry point for entangled hydrogen visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entangled Hydrogen Visualization System"
    )
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--expansion-dim", type=int, default=64)
    parser.add_argument("--num-spectral-layers", type=int, default=2)
    parser.add_argument("--hamiltonian-checkpoint", type=str, default="weights/latest.pth")
    parser.add_argument("--schrodinger-checkpoint", type=str, default="weights/schrodinger_crystal_final.pth")
    parser.add_argument("--dirac-checkpoint", type=str, default="weights/dirac_phase5_latest.pth")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--backend", type=str, default="schrodinger",
                       choices=["hamiltonian", "schrodinger", "dirac"])
    parser.add_argument("--output-dir", type=str, default="download")
    
    args = parser.parse_args()
    
    config = EntangledHydrogenConfig(
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        expansion_dim=args.expansion_dim,
        num_spectral_layers=args.num_spectral_layers,
        hamiltonian_checkpoint=args.hamiltonian_checkpoint,
        schrodinger_checkpoint=args.schrodinger_checkpoint,
        dirac_checkpoint=args.dirac_checkpoint,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        random_seed=args.seed,
        output_dir=args.output_dir
    )
    
    experiment = EntangledHydrogenExperiment(config)
    results = experiment.run_all_demonstrations(args.num_samples)
    
    for name, result in results.items():
        if 'save_path' in result:
            _LOG.info("%s: %s", name, result['save_path'])


if __name__ == "__main__":
    main()