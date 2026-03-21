#!/usr/bin/env python3
"""
Quantum Framework Core Module
=============================
Unified core module for quantum simulation using MPS-based Hilbert space
representation. Achieves sub-exponential memory scaling O(n * chi^2) instead
of O(2^n) for full statevector representation.

This module consolidates:
    - Configuration loading from TOML
    - MPS (Matrix Product State) tensor network representation
    - Physics backends (Hamiltonian, Schrodinger, Dirac)
    - Quantum gate registry with MPS-compatible operations
    - Quantum circuit builder and executor

Author: Gris Iscomeback
License: AGPL v3
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _make_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


_LOG = _make_logger("QuantumFramework")


class HilbertPhase(Enum):
    """Phase classification for Hilbert space compression."""
    HOT_GLASS = auto()
    COLD_GLASS = auto()
    POLYCRYSTAL = auto()
    TOPOLOGICAL_INSULATOR = auto()
    PERFECT_CRYSTAL = auto()


@dataclass
class FrameworkConfig:
    """
    Unified configuration for the quantum simulation framework.
    Loads from TOML file with fallback to sensible defaults.
    """
    grid_size: int = 16
    hidden_dim: int = 32
    expansion_dim: int = 64
    num_spectral_layers: int = 2
    spinor_components: int = 4
    
    c_light: float = 137.035999084
    alpha_fs: float = 0.0072973525693
    hbar: float = 1.0
    electron_mass: float = 1.0
    proton_mass: float = 1836.15267343
    dt: float = 0.01
    normalization_eps: float = 1.0e-8
    potential_depth: float = 5.0
    potential_width: float = 0.3
    dirac_mass: float = 1.0
    gamma_representation: str = "dirac"
    
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    random_seed: int = 42
    max_qubits: int = 33
    
    bond_dimension: int = 16
    max_bond_dimension: int = 64
    svd_threshold: float = 1.0e-10
    truncation_error: float = 1.0e-8
    regularization_lambda: float = 1.0e34
    vacuum_sparsity_target: float = 0.9999
    winding_number_threshold: float = 1.5
    berry_phase_threshold: float = 0.1
    entanglement_entropy_threshold: float = 1.0
    
    mc_batch_size: int = 100000
    mc_max_particles: int = 2000000
    mc_min_particles: int = 5000
    r_max_factor: float = 4.0
    r_max_offset: float = 10.0
    prob_safety_factor: float = 1.05
    grid_search_r: int = 300
    grid_search_theta: int = 150
    grid_search_phi: int = 150
    
    figure_dpi: int = 150
    figure_size_x: int = 24
    figure_size_y: int = 20
    histogram_bins: int = 300
    scatter_size_min: float = 1.0
    scatter_size_max: float = 8.0
    background_color: str = "#000008"
    
    hamiltonian_checkpoint: str = "weights/latest.pth"
    schrodinger_checkpoint: str = "weights/schrodinger_crystal_final.pth"
    dirac_checkpoint: str = "weights/dirac_phase5_latest.pth"
    
    output_dir: str = "download"
    default_num_samples: int = 100000
    enable_vacuum_core: bool = False
    precision_mode: bool = False
    max_iterations: int = 200
    convergence_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        """Initialize random seeds after configuration."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.manual_seed(self.random_seed)

    @classmethod
    def from_toml(cls, toml_path: str) -> "FrameworkConfig":
        """Load configuration from TOML file."""
        if not os.path.exists(toml_path):
            _LOG.warning("Config file not found: %s, using defaults", toml_path)
            return cls()
        
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        
        sim = data.get("simulation", {})
        physics = data.get("physics", {})
        mps = data.get("mps", {})
        mc = data.get("monte_carlo", {})
        vis = data.get("visualization", {})
        ckpt = data.get("checkpoints", {})
        out = data.get("output", {})
        vqe = data.get("vqe", {})
        
        device = sim.get("device", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        
        return cls(
            grid_size=sim.get("grid_size", 16),
            dtype=torch.float64 if sim.get("dtype", "float64") == "float64" else torch.float32,
            device=device,
            random_seed=sim.get("random_seed", 42),
            max_qubits=sim.get("max_qubits", 33),
            bond_dimension=mps.get("bond_dimension", sim.get("bond_dimension", 16)),
            max_bond_dimension=mps.get("max_bond_dimension", sim.get("max_bond_dimension", 64)),
            svd_threshold=mps.get("svd_threshold", sim.get("svd_threshold", 1.0e-10)),
            truncation_error=mps.get("truncation_error", sim.get("truncation_error", 1.0e-8)),
            c_light=physics.get("c_light", 137.035999084),
            alpha_fs=physics.get("alpha_fs", 0.0072973525693),
            hbar=physics.get("hbar", 1.0),
            electron_mass=physics.get("electron_mass", 1.0),
            dt=physics.get("dt", 0.01),
            normalization_eps=physics.get("normalization_eps", 1.0e-8),
            potential_depth=physics.get("potential_depth", 5.0),
            potential_width=physics.get("potential_width", 0.3),
            dirac_mass=physics.get("dirac_mass", 1.0),
            gamma_representation=physics.get("gamma_representation", "dirac"),
            mc_batch_size=mc.get("batch_size", 100000),
            mc_max_particles=mc.get("max_particles", 2000000),
            mc_min_particles=mc.get("min_particles", 5000),
            r_max_factor=mc.get("r_max_factor", 4.0),
            r_max_offset=mc.get("r_max_offset", 10.0),
            prob_safety_factor=mc.get("probability_safety_factor", 1.05),
            grid_search_r=mc.get("grid_search_r", 300),
            grid_search_theta=mc.get("grid_search_theta", 150),
            grid_search_phi=mc.get("grid_search_phi", 150),
            figure_dpi=vis.get("figure_dpi", 150),
            figure_size_x=vis.get("figure_size_x", 24),
            figure_size_y=vis.get("figure_size_y", 20),
            histogram_bins=vis.get("histogram_bins", 300),
            scatter_size_min=vis.get("scatter_size_min", 1.0),
            scatter_size_max=vis.get("scatter_size_max", 8.0),
            background_color=vis.get("background_color", "#000008"),
            hamiltonian_checkpoint=ckpt.get("hamiltonian", "weights/latest.pth"),
            schrodinger_checkpoint=ckpt.get("schrodinger", "weights/schrodinger_crystal_final.pth"),
            dirac_checkpoint=ckpt.get("dirac", "weights/dirac_phase5_latest.pth"),
            output_dir=out.get("directory", "download"),
            default_num_samples=sim.get("default_num_samples", 100000),
            enable_vacuum_core=sim.get("enable_vacuum_core", False),
            precision_mode=sim.get("precision_mode", False),
            max_iterations=vqe.get("max_iterations", 200),
            convergence_tolerance=vqe.get("convergence_tolerance", 1.0e-8),
        )


@dataclass
class AtomData:
    """Atomic data structure."""
    symbol: str
    name: str
    atomic_number: int
    mass: float
    nuclear_charge: float
    electron_configuration: List[str]
    max_qubits_needed: int


@dataclass
class MoleculeData:
    """Molecular data structure."""
    name: str
    formula: str
    description: str
    atoms: List[str]
    n_electrons: int
    n_orbitals: int
    n_qubits: int
    bond_length_angstrom: float
    hf_energy_hartree: float
    fci_energy_hartree: float
    nuclear_repulsion_hartree: float
    geometry: Dict[str, Any]
    basis: str = "sto-3g"
    h_core: Optional[np.ndarray] = None
    eri: Optional[np.ndarray] = None


@dataclass
class OrbitalData:
    """Atomic orbital data structure."""
    name: str
    n: int
    l: int
    m: int
    description: str


class ConfigLoader:
    """
    Configuration loader that parses TOML files and provides
    access to atoms, molecules, and orbitals data.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config()
        self._raw_data: Dict[str, Any] = {}
        self._atoms: Dict[str, AtomData] = {}
        self._molecules: Dict[str, MoleculeData] = {}
        self._orbitals: Dict[str, OrbitalData] = {}
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _find_config(self) -> str:
        """Find configuration file in standard locations."""
        candidates = [
            "quantum_framework_config.toml",
            os.path.join(os.path.dirname(__file__), "quantum_framework_config.toml"),
            os.path.join(os.path.dirname(__file__), "..", "quantum_framework_config.toml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""
    
    def _load(self) -> None:
        """Load configuration from TOML file."""
        if not self.config_path or not os.path.exists(self.config_path):
            self._load_defaults()
            return
        
        with open(self.config_path, "rb") as f:
            self._raw_data = tomllib.load(f)
        
        self._parse_atoms()
        self._parse_molecules()
        self._parse_orbitals()
        self._parse_experiments()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self._atoms = {
            "H": AtomData("H", "Hydrogen", 1, 1.007825, 1.0, ["1s1"], 2),
            "He": AtomData("He", "Helium", 2, 4.002603, 2.0, ["1s2"], 2),
            "Li": AtomData("Li", "Lithium", 3, 7.016004, 3.0, ["1s2", "2s1"], 6),
        }
        self._molecules = {
            "H2": MoleculeData(
                "H2", "H2", "Hydrogen molecule", ["H", "H"],
                2, 2, 4, 0.735, -1.11675928, -1.13728383, 0.71997, {}
            ),
        }
        self._orbitals = {
            "1s": OrbitalData("1s", 1, 0, 0, "1s orbital"),
            "2s": OrbitalData("2s", 2, 0, 0, "2s orbital"),
        }
    
    def _parse_atoms(self) -> None:
        """Parse atoms from configuration data."""
        for a in self._raw_data.get("atoms", []):
            atom = AtomData(
                symbol=a.get("symbol", ""),
                name=a.get("name", ""),
                atomic_number=a.get("atomic_number", 0),
                mass=a.get("mass", 0.0),
                nuclear_charge=a.get("nuclear_charge", 0.0),
                electron_configuration=a.get("electron_configuration", []),
                max_qubits_needed=a.get("max_qubits_needed", 2),
            )
            self._atoms[atom.symbol] = atom
    
    def _parse_molecules(self) -> None:
        """Parse molecules from configuration data."""
        for m in self._raw_data.get("molecules", []):
            mol = MoleculeData(
                name=m.get("name", ""),
                formula=m.get("formula", ""),
                description=m.get("description", ""),
                atoms=m.get("atoms", []),
                n_electrons=m.get("n_electrons", 0),
                n_orbitals=m.get("n_orbitals", 0),
                n_qubits=m.get("n_qubits", 0),
                bond_length_angstrom=m.get("bond_length_angstrom", 0.0),
                hf_energy_hartree=m.get("hf_energy_hartree", 0.0),
                fci_energy_hartree=m.get("fci_energy_hartree", 0.0),
                nuclear_repulsion_hartree=m.get("nuclear_repulsion_hartree", 0.0),
                geometry=m.get("geometry", {}),
                basis=m.get("basis", "sto-3g"),
            )
            self._molecules[mol.name] = mol
    
    def _parse_orbitals(self) -> None:
        """Parse orbitals from configuration data."""
        for o in self._raw_data.get("orbitals", []):
            orb = OrbitalData(
                name=o.get("name", ""),
                n=o.get("n", 0),
                l=o.get("l", 0),
                m=o.get("m", 0),
                description=o.get("description", ""),
            )
            self._orbitals[orb.name] = orb
    
    def _parse_experiments(self) -> None:
        """Parse experiments from configuration data."""
        for e in self._raw_data.get("experiments", []):
            exp = {
                "name": e.get("name", ""),
                "description": e.get("description", ""),
                "category": e.get("category", ""),
                "default_qubits": e.get("default_qubits", 2),
            }
            self._experiments[exp["name"]] = exp
    
    def get_atom(self, symbol: str) -> Optional[AtomData]:
        """Get atom data by symbol (case-insensitive)."""
        result = self._atoms.get(symbol)
        if result is None:
            for k, v in self._atoms.items():
                if k.lower() == symbol.lower():
                    return v
        return result
    
    def get_molecule(self, name: str) -> Optional[MoleculeData]:
        """Get molecule data by name (case-insensitive)."""
        result = self._molecules.get(name)
        if result is None:
            for k, v in self._molecules.items():
                if k.lower() == name.lower():
                    return v
        return result
    
    def get_orbital(self, name: str) -> Optional[OrbitalData]:
        """Get orbital data by name (case-insensitive)."""
        result = self._orbitals.get(name)
        if result is None:
            for k, v in self._orbitals.items():
                if k.lower() == name.lower():
                    return v
        return result
    
    def get_experiment(self, name: str) -> Optional[Dict[str, Any]]:
        """Get experiment data by name."""
        return self._experiments.get(name)
    
    @property
    def atoms(self) -> Dict[str, AtomData]:
        """Return all atoms."""
        return self._atoms
    
    @property
    def molecules(self) -> Dict[str, MoleculeData]:
        """Return all molecules."""
        return self._molecules
    
    @property
    def orbitals(self) -> Dict[str, OrbitalData]:
        """Return all orbitals."""
        return self._orbitals
    
    @property
    def experiments(self) -> Dict[str, Dict[str, Any]]:
        """Return all experiments."""
        return self._experiments
    
    def get_molecules_by_qubits(self, max_qubits: int) -> List[MoleculeData]:
        """Get molecules that fit within qubit budget."""
        return [m for m in self._molecules.values() if m.n_qubits <= max_qubits]
    
    def get_atoms_by_qubits(self, max_qubits: int) -> List[AtomData]:
        """Get atoms that fit within qubit budget."""
        return [a for a in self._atoms.values() if a.max_qubits_needed <= max_qubits]


class ITensorNetwork(ABC):
    """Abstract interface for tensor network quantum states."""
    
    @property
    @abstractmethod
    def n_qubits(self) -> int:
        """Return number of qubits."""
        pass
    
    @abstractmethod
    def amplitude(self, basis_index: int) -> torch.Tensor:
        """Compute amplitude for a computational basis state."""
        pass
    
    @abstractmethod
    def apply_single_qubit_gate(self, qubit: int, gate: torch.Tensor) -> None:
        """Apply single-qubit gate in-place."""
        pass
    
    @abstractmethod
    def apply_two_qubit_gate(self, qubit_a: int, qubit_b: int, gate: torch.Tensor) -> None:
        """Apply two-qubit gate in-place."""
        pass
    
    @abstractmethod
    def norm(self) -> float:
        """Compute state norm."""
        pass
    
    @abstractmethod
    def probabilities(self) -> torch.Tensor:
        """Compute measurement probabilities."""
        pass
    
    @abstractmethod
    def entropy(self) -> float:
        """Compute von Neumann entropy."""
        pass
    
    @abstractmethod
    def memory_bytes(self) -> int:
        """Return memory usage in bytes."""
        pass


class MPSCore:
    """
    Matrix Product State core tensor A^{[k]}_{i_k} with bond indices.
    
    Shape: (chi_left, d, chi_right) where d=2 for qubits.
    Memory per core: O(chi^2 * d) = O(chi^2)
    """
    
    def __init__(
        self,
        chi_left: int,
        chi_right: int,
        d: int = 2,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64
    ) -> None:
        self.chi_left = chi_left
        self.chi_right = chi_right
        self.d = d
        self.device = device
        self.dtype = dtype
        self._tensor: Optional[torch.Tensor] = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize core tensor for |0> product state (exact)."""
        tensor = torch.zeros(
            self.chi_left, self.d, self.chi_right,
            dtype=self.dtype, device=self.device
        )
        tensor[:, 0, :] = torch.eye(
            self.chi_left, self.chi_right,
            dtype=self.dtype, device=self.device
        )[:self.chi_left, :self.chi_right]
        self._tensor = tensor
    
    @property
    def tensor(self) -> torch.Tensor:
        """Return the core tensor."""
        if self._tensor is None:
            self._initialize()
        return self._tensor
    
    @tensor.setter
    def tensor(self, value: torch.Tensor) -> None:
        """Set the core tensor, preserving complex dtype when needed."""
        if value.is_complex():
            self._tensor = value.to(device=self.device)
        else:
            self._tensor = value.to(device=self.device, dtype=self.dtype)
        if self._tensor.dim() == 3:
            self.chi_left, self.d, self.chi_right = self._tensor.shape
    
    def left_canonicalize(self) -> torch.Tensor:
        """Bring core to left-canonical form, return singular values."""
        shape = self.chi_left * self.d, self.chi_right
        tensor_matrix = self.tensor.reshape(shape)
        u, s, vh = torch.linalg.svd(tensor_matrix, full_matrices=False)
        self._tensor = u.reshape(self.chi_left, self.d, -1)
        if vh.is_complex():
            return s.to(vh.dtype) @ vh
        return s @ vh

    def right_canonicalize(self) -> torch.Tensor:
        """Bring core to right-canonical form, return singular values."""
        shape = self.chi_left, self.d * self.chi_right
        tensor_matrix = self.tensor.reshape(shape)
        u, s, vh = torch.linalg.svd(tensor_matrix, full_matrices=False)
        self._tensor = vh.reshape(-1, self.d, self.chi_right)
        if u.is_complex():
            return u @ s.to(u.dtype)
        return u @ s


class MPSState(ITensorNetwork):
    """
    Matrix Product State representation of n-qubit quantum state.
    
    |psi> = sum_{i_1...i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} ... A^{[n]}_{i_n} |i_1...i_n>
    
    Memory: O(n * chi^2 * d) vs O(d^n) for full statevector.
    
    Example scaling:
        n=30, chi=16: ~30KB vs 8GB for statevector
        n=33, chi=16: ~33KB vs 64GB for statevector
    """
    
    def __init__(self, n_qubits: int, config: FrameworkConfig) -> None:
        self._n_qubits = n_qubits
        self.config = config
        self.d = 2
        self._cores: List[MPSCore] = []
        self._canonical_form: str = "none"
        self._center: int = 0
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize MPS with product state |00...0>."""
        bonds = [1]
        for k in range(self._n_qubits - 1):
            bond = min(
                self.config.bond_dimension,
                2 ** min(k + 1, self._n_qubits - k - 1)
            )
            bonds.append(bond)
        bonds.append(1)
        
        for k in range(self._n_qubits):
            chi_left = bonds[k]
            chi_right = bonds[k + 1]
            core = MPSCore(
                chi_left, chi_right, self.d,
                self.config.device, self.config.dtype
            )
            self._cores.append(core)
        
        self._canonical_form = "right"
        self._center = 0
    
    @property
    def n_qubits(self) -> int:
        return self._n_qubits
    
    def _bond_dimension(self, site: int) -> int:
        """Compute bond dimension at given site."""
        return min(
            self.config.bond_dimension,
            min(2 ** site, 2 ** (self._n_qubits - site))
        )
    
    def amplitude(self, basis_index: int) -> torch.Tensor:
        """Compute amplitude for computational basis state."""
        if basis_index < 0 or basis_index >= 2 ** self._n_qubits:
            raise ValueError(
                f"basis_index {basis_index} out of range for {self._n_qubits} qubits"
            )
        
        bits = [
            (basis_index >> (self._n_qubits - 1 - k)) & 1
            for k in range(self._n_qubits)
        ]
        
        # Use complex128 so that cores with imaginary parts (e.g. after Y gate)
        # contribute correctly.
        result = torch.ones(1, 1, dtype=torch.complex128, device=self.config.device)
        for k, bit in enumerate(bits):
            tensor = self._cores[k].tensor.to(torch.complex128)
            result = result @ tensor[:, bit, :]

        return result.squeeze()
    
    def apply_single_qubit_gate(self, qubit: int, gate: torch.Tensor) -> None:
        """Apply single-qubit gate in-place.

        Works in complex128 so that gates with imaginary entries (Y, S, T, Rz…)
        are handled correctly.  The core tensor is promoted to complex128 when
        the result has a non-negligible imaginary component; otherwise it is
        kept as the config dtype (typically float64).
        """
        if qubit < 0 or qubit >= self._n_qubits:
            raise ValueError(f"qubit {qubit} out of range")

        core = self._cores[qubit]
        gate_c = gate.to(dtype=torch.complex128, device=self.config.device)
        tensor_c = core.tensor.to(torch.complex128)
        new_tensor = torch.einsum("ij,ajk->aik", gate_c, tensor_c)

        if torch.allclose(new_tensor.imag, torch.zeros_like(new_tensor.imag), atol=1e-12):
            core.tensor = new_tensor.real.to(self.config.dtype)
        else:
            # Keep as complex128 so subsequent gates and amplitude() are correct.
            core.tensor = new_tensor.to(torch.complex128)

        self._canonical_form = "none"
    
    def apply_two_qubit_gate(self, qubit_a: int, qubit_b: int, gate: torch.Tensor) -> None:
        """Apply two-qubit gate in-place."""
        if qubit_a == qubit_b:
            raise ValueError("qubit_a and qubit_b must be different")
        
        if qubit_a < qubit_b:
            abs_diff = qubit_b - qubit_a
            if abs_diff == 1:
                self._apply_adjacent_gate(qubit_a, gate)
            else:
                self._apply_nonadjacent_gate(qubit_a, qubit_b, gate)
        else:
            gate_swapped = self._swap_qubits_in_gate(gate)
            abs_diff = qubit_a - qubit_b
            if abs_diff == 1:
                self._apply_adjacent_gate(qubit_b, gate_swapped)
            else:
                self._apply_nonadjacent_gate(qubit_b, qubit_a, gate_swapped)
    
    def _swap_qubits_in_gate(self, gate: torch.Tensor) -> torch.Tensor:
        """Swap qubit ordering in two-qubit gate."""
        perm = [0, 2, 1, 3]
        gate_4x4 = gate.view(4, 4)
        gate_swap = torch.zeros_like(gate_4x4)
        for i in range(4):
            for j in range(4):
                gate_swap[perm[i], perm[j]] = gate_4x4[i, j]
        return gate_swap
    
    def _apply_adjacent_gate(self, qubit: int, gate: torch.Tensor) -> None:
        """Apply gate to adjacent qubit pair."""
        core_a = self._cores[qubit]
        core_b = self._cores[qubit + 1]
        tensor_a = core_a.tensor
        tensor_b = core_b.tensor
        
        chi_l, d, chi_m = tensor_a.shape
        chi_m2, d2, chi_r = tensor_b.shape
        
        if chi_m != chi_m2:
            chi_m = min(chi_m, chi_m2)
            tensor_a = tensor_a[:, :, :chi_m]
            tensor_b = tensor_b[:chi_m, :, :]
        
        # Promote to complex128 if either tensor or gate is complex
        if tensor_a.is_complex() or tensor_b.is_complex() or gate.is_complex():
            tensor_a = tensor_a.to(torch.complex128)
            tensor_b = tensor_b.to(torch.complex128)
            gate_matrix = gate.to(dtype=torch.complex128, device=self.config.device)
        else:
            gate_matrix = gate.to(dtype=self.config.dtype, device=self.config.device)

        theta = torch.einsum("iaj,jbk->iabk", tensor_a, tensor_b)
        theta = theta.permute(0, 3, 1, 2)
        theta = theta.reshape(chi_l * chi_r, d * d)

        theta = theta @ gate_matrix.T
        theta = theta.reshape(chi_l, chi_r, d, d)
        theta = theta.permute(0, 2, 3, 1)
        theta = theta.reshape(chi_l * d, d * chi_r)
        
        u, s, vh = torch.linalg.svd(theta, full_matrices=False)
        
        truncation = min(len(s), self.config.max_bond_dimension)
        s_trunc = s[:truncation]
        u_trunc = u[:, :truncation]
        vh_trunc = vh[:truncation, :]
        
        mask = s_trunc > self.config.svd_threshold
        s_trunc = s_trunc[mask]
        u_trunc = u_trunc[:, mask]
        vh_trunc = vh_trunc[mask, :]
        
        if len(s_trunc) == 0:
            s_trunc = torch.ones(1, dtype=self.config.dtype, device=self.config.device)
            u_trunc = u[:, :1]
            vh_trunc = vh[:1, :]
        
        core_a.tensor = u_trunc.reshape(chi_l, d, -1)

        norm_s = torch.sqrt(torch.sum(s_trunc ** 2))
        if norm_s > 1e-12:
            s_trunc = s_trunc / norm_s

        # Cast s_trunc to match vh_trunc dtype (SVD singular values are always real)
        if vh_trunc.is_complex():
            s_trunc = s_trunc.to(vh_trunc.dtype)
        core_b.tensor = (torch.diag(s_trunc) @ vh_trunc).reshape(-1, d, chi_r)
        
        self._canonical_form = "mixed"
        self._center = qubit + 1
    
    def _apply_nonadjacent_gate(self, qubit_a: int, qubit_b: int, gate: torch.Tensor) -> None:
        """Apply gate to non-adjacent qubit pair using SWAP network."""
        swap = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=self.config.dtype, device=self.config.device
        )
        
        for k in range(qubit_b, qubit_a + 1, -1):
            self._apply_adjacent_gate(k - 1, swap)
        
        self._apply_adjacent_gate(qubit_a, gate)
        
        for k in range(qubit_a + 1, qubit_b + 1):
            self._apply_adjacent_gate(k - 1, swap)
    
    def norm(self) -> float:
        """Compute state norm."""
        self._canonicalize()
        if self._center < len(self._cores):
            center_tensor = self._cores[self._center].tensor
            return float(torch.norm(center_tensor).item())
        return 1.0
    
    def _canonicalize(self) -> None:
        """Bring MPS to canonical form."""
        for k in range(self._n_qubits - 1):
            if k < self._center:
                self._cores[k].left_canonicalize()
            else:
                self._cores[self._n_qubits - 1 - k].right_canonicalize()
    
    def probabilities(self) -> torch.Tensor:
        """Compute measurement probabilities."""
        probs = torch.zeros(
            2 ** self._n_qubits,
            dtype=self.config.dtype,
            device=self.config.device
        )
        for k in range(2 ** self._n_qubits):
            amp = self.amplitude(k)
            probs[k] = torch.abs(amp) ** 2
        
        total = torch.sum(probs)
        if total > 1e-12:
            probs = probs / total
        
        return probs
    
    def entropy(self) -> float:
        """Compute maximum entanglement entropy across all cuts.

        For n=1 qubits, returns Shannon entropy of the probability distribution.
        For n>1, returns the maximum entanglement entropy across all bipartite cuts.
        """
        if self._n_qubits == 1:
            # Shannon entropy from probabilities for single qubit
            probs = self.probabilities()
            eps = 1e-12
            h = 0.0
            for p in probs:
                p_val = float(p)
                if p_val > eps:
                    h -= p_val * math.log2(p_val)
            return h

        max_entropy = 0.0
        for cut in range(1, self._n_qubits):
            try:
                e = self.entanglement_entropy(cut)
                if e > max_entropy:
                    max_entropy = e
            except Exception:
                pass
        return max_entropy
    
    def memory_bytes(self) -> int:
        """Return memory usage in bytes."""
        total = 0
        for core in self._cores:
            total += core.tensor.numel() * core.tensor.element_size()
        return total
    
    def entanglement_entropy(self, cut: int) -> float:
        """
        Compute entanglement entropy at given cut between qubits cut-1 and cut.
        Uses Schmidt decomposition from the MPS bond.
        """
        if cut <= 0 or cut >= self._n_qubits:
            return 0.0
        
        if self._n_qubits <= 12:
            statevector = self.to_statevector()
            left_dim = 2 ** cut
            right_dim = 2 ** (self._n_qubits - cut)
            matrix = statevector.reshape(left_dim, right_dim)
            
            _, s, _ = torch.linalg.svd(matrix, full_matrices=False)
            s_squared = s ** 2
            s_squared = s_squared / (s_squared.sum() + 1e-12)
            entropy = -torch.sum(s_squared * torch.log2(s_squared + 1e-12))
            return float(entropy.item())
        
        left_part = [self._cores[k].tensor for k in range(cut)]
        right_part = [self._cores[k].tensor for k in range(cut, self._n_qubits)]
        
        left_tensor = left_part[0]
        for t in left_part[1:]:
            left_tensor = torch.einsum("iaj,jbk->iabk", left_tensor, t)
            left_tensor = left_tensor.reshape(-1, left_tensor.shape[-1])
        
        right_tensor = right_part[-1]
        for t in reversed(right_part[:-1]):
            right_tensor = torch.einsum("iaj,jbk->iabk", t, right_tensor)
            right_tensor = right_tensor.reshape(right_tensor.shape[0], -1)
        
        bond_dim = min(left_tensor.shape[-1], right_tensor.shape[0])
        left_tensor = left_tensor[:, :bond_dim]
        right_tensor = right_tensor[:bond_dim, :]
        
        theta = left_tensor @ right_tensor
        _, s, _ = torch.linalg.svd(theta, full_matrices=False)
        s_squared = s ** 2
        s_squared = s_squared / (s_squared.sum() + 1e-12)
        entropy = -torch.sum(s_squared * torch.log2(s_squared + 1e-12))
        
        return float(entropy.item())
    
    def to_statevector(self) -> torch.Tensor:
        """Convert MPS to full statevector (only for small systems)."""
        if self._n_qubits > 20:
            raise MemoryError(f"Cannot convert {self._n_qubits} qubit MPS to statevector")
        
        statevector = torch.zeros(
            2 ** self._n_qubits,
            dtype=torch.complex128,
            device=self.config.device
        )
        
        for i in range(2 ** self._n_qubits):
            statevector[i] = self.amplitude(i)
        
        return statevector
    
    def most_probable_bitstring(self) -> str:
        """Return most probable basis state as bitstring."""
        probs = self.probabilities()
        k = int(probs.argmax().item())
        return format(k, f"0{self._n_qubits}b")
    
    def clone(self) -> "MPSState":
        """Return a deep copy."""
        new_state = MPSState.__new__(MPSState)
        new_state._n_qubits = self._n_qubits
        new_state.config = self.config
        new_state.d = self.d
        new_state._cores = [
            MPSCore(
                c.chi_left, c.chi_right, c.d,
                c.device, c.dtype
            )
            for c in self._cores
        ]
        for i, c in enumerate(self._cores):
            new_state._cores[i].tensor = c.tensor.clone()
        new_state._canonical_form = self._canonical_form
        new_state._center = self._center
        return new_state


class VacuumCore:
    """
    Vacuum Core architecture for topological protection.
    
    Projects irrelevant Hilbert subspace to zero, achieving
    high sparsity (target 99.99%) while preserving quantum information.
    """
    
    def __init__(self, n_qubits: int, config: FrameworkConfig) -> None:
        self.n_qubits = n_qubits
        self.config = config
        self.active_subspace: List[int] = [0]
        self.vacuum_mask: torch.Tensor = torch.zeros(
            2 ** n_qubits, dtype=torch.bool, device=config.device
        )
        self.winding_numbers: Dict[int, float] = {}
        self.berry_phases: Dict[Tuple[int, int], float] = {}
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize vacuum core with ground state."""
        self.vacuum_mask[0] = True
        self.winding_numbers[0] = 2.0
        self._compute_berry_phases()
    
    def _compute_berry_phases(self) -> None:
        """Compute Berry phases between active states."""
        for i, idx_i in enumerate(self.active_subspace):
            for j, idx_j in enumerate(self.active_subspace):
                if i < j:
                    self.berry_phases[(idx_i, idx_j)] = 0.0
    
    def add_active_state(self, basis_index: int, winding_number: float = 0.0) -> None:
        """Add a basis state to the active subspace."""
        if basis_index < 0 or basis_index >= 2 ** self.n_qubits:
            raise ValueError(f"basis_index {basis_index} out of range")
        
        if basis_index not in self.active_subspace:
            self.active_subspace.append(basis_index)
            self.vacuum_mask[basis_index] = True
            self.winding_numbers[basis_index] = (
                winding_number
                if winding_number != 0.0
                else self._compute_winding_number(basis_index)
            )
    
    def _compute_winding_number(self, basis_index: int) -> float:
        """Compute winding number for a basis state."""
        bits = bin(basis_index).count("1")
        phase = bits * math.pi / self.n_qubits
        return 2.0 * math.cos(phase)
    
    def is_topologically_protected(self, basis_index: int) -> bool:
        """Check if state is topologically protected."""
        winding = self.winding_numbers.get(basis_index, 0.0)
        return abs(winding) >= self.config.winding_number_threshold
    
    def sparsity(self) -> float:
        """Compute vacuum sparsity."""
        active = len(self.active_subspace)
        total = 2 ** self.n_qubits
        return 1.0 - (active / total)
    
    def project_to_active(self, state: ITensorNetwork) -> ITensorNetwork:
        """Project state onto active subspace."""
        probs = state.probabilities()
        for k in range(len(probs)):
            if probs[k] > self.config.svd_threshold and k not in self.active_subspace:
                self.add_active_state(k)
        return state


class TopologicalProtector:
    """
    Provides topological protection for quantum states.
    
    Monitors:
        - Winding numbers
        - Berry phases
        - Edge state preservation
    """
    
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.winding_history: List[Dict[int, float]] = []
        self.berry_phase_history: List[Dict[Tuple[int, int], float]] = []
    
    def compute_winding_number(self, state: ITensorNetwork, qubit: int) -> float:
        """Compute winding number for a qubit."""
        probs = state.probabilities()
        dim = len(probs)
        bit_pos = state.n_qubits - 1 - qubit
        
        p0 = sum(probs[k] for k in range(dim) if not ((k >> bit_pos) & 1))
        p1 = 1.0 - p0
        
        theta = math.acos(max(-1.0, min(1.0, p0 - p1)))
        return 2.0 * math.sin(theta / 2.0)
    
    def compute_berry_phase(
        self, state: ITensorNetwork, qubit_a: int, qubit_b: int
    ) -> float:
        """Compute Berry phase between two qubits."""
        probs = state.probabilities()
        dim = len(probs)
        bit_a = state.n_qubits - 1 - qubit_a
        bit_b = state.n_qubits - 1 - qubit_b
        
        phase = 0.0
        for k in range(dim):
            if ((k >> bit_a) & 1) != ((k >> bit_b) & 1):
                phase += probs[k].item() * math.pi
        
        return phase
    
    def is_protected(self, state: ITensorNetwork, vacuum_core: VacuumCore) -> bool:
        """Check if state is topologically protected."""
        for idx in vacuum_core.active_subspace:
            winding = vacuum_core.winding_numbers.get(idx, 0.0)
            if abs(winding) >= self.config.winding_number_threshold:
                return True
        return False


class SpectralLayer(nn.Module):
    """Spectral convolution layer in frequency domain."""
    
    def __init__(self, channels: int, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.kernel_real = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )
        self.kernel_imag = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution via RFFT2."""
        x_fft = torch.fft.rfft2(x)
        _b, _c, freq_h, freq_w = x_fft.shape
        
        kr = F.interpolate(
            self.kernel_real.mean(dim=0).unsqueeze(0),
            size=(freq_h, freq_w), mode="bilinear", align_corners=False,
        ).squeeze(0)
        ki = F.interpolate(
            self.kernel_imag.mean(dim=0).unsqueeze(0),
            size=(freq_h, freq_w), mode="bilinear", align_corners=False,
        ).squeeze(0)
        
        real_part = x_fft.real * kr - x_fft.imag * ki
        imag_part = x_fft.real * ki + x_fft.imag * kr
        
        return torch.fft.irfft2(
            torch.complex(real_part, imag_part),
            s=(self.grid_size, self.grid_size),
        )


class HamiltonianBackboneNet(nn.Module):
    """Hamiltonian backbone network for spectral operations."""
    
    def __init__(self, grid_size: int, hidden_dim: int, num_spectral_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(hidden_dim, grid_size)
            for _ in range(num_spectral_layers)
        ])
        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hamiltonian backbone network."""
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = F.gelu(self.input_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        
        return self.output_proj(x).squeeze(1)


class SchrodingerSpectralNet(nn.Module):
    """Schrodinger network for wavefunction evolution."""
    
    def __init__(
        self, grid_size: int, hidden_dim: int,
        expansion_dim: int, num_spectral_layers: int
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.expansion_proj = nn.Conv2d(hidden_dim, expansion_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(expansion_dim, grid_size)
            for _ in range(num_spectral_layers)
        ])
        self.contraction_proj = nn.Conv2d(expansion_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, 2, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Schrodinger evolution network."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        x = F.gelu(self.contraction_proj(x))
        
        return self.output_proj(x)


class DiracSpectralNet(nn.Module):
    """Dirac network for relativistic spinor evolution."""
    
    def __init__(
        self, grid_size: int, hidden_dim: int,
        expansion_dim: int, num_spectral_layers: int
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(8, hidden_dim, kernel_size=1)
        self.expansion_proj = nn.Conv2d(hidden_dim, expansion_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(expansion_dim, grid_size)
            for _ in range(num_spectral_layers)
        ])
        self.contraction_proj = nn.Conv2d(expansion_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, 8, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Dirac evolution network."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        x = F.gelu(self.contraction_proj(x))
        
        return self.output_proj(x)


class GammaMatrices:
    """Dirac gamma matrices in standard or Weyl representation."""
    
    def __init__(self, representation: str = "dirac", device: str = "cpu") -> None:
        self.representation = representation
        self.device = device
        self._init_matrices()
    
    def _init_matrices(self) -> None:
        """Initialize gamma matrices."""
        if self.representation == "dirac":
            self.gamma0 = torch.tensor([
                [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]
            ], dtype=torch.complex64, device=self.device)
            self.gamma1 = torch.tensor([
                [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma2 = torch.tensor([
                [0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma3 = torch.tensor([
                [0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]
            ], dtype=torch.complex64, device=self.device)
        else:
            self.gamma0 = torch.tensor([
                [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma1 = torch.tensor([
                [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma2 = torch.tensor([
                [0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma3 = torch.tensor([
                [0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]
            ], dtype=torch.complex64, device=self.device)
        
        self.alpha_x = self.gamma0 @ self.gamma1
        self.alpha_y = self.gamma0 @ self.gamma2
        self.alpha_z = self.gamma0 @ self.gamma3
        self.beta = self.gamma0
        
        self.gammas = [self.gamma0, self.gamma1, self.gamma2, self.gamma3]
    
    def to(self, device: str) -> "GammaMatrices":
        """Move all matrices to device."""
        self.device = device
        self._init_matrices()
        return self


class IPhysicsBackend(ABC):
    """Abstract interface for physics backends."""
    
    @abstractmethod
    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolve a single amplitude by time dt."""
        pass
    
    @abstractmethod
    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        """Apply global phase to amplitude."""
        pass


class HamiltonianBackend(IPhysicsBackend):
    """
    Hamiltonian backend using neural network for spectral operations.
    
    Performs first-order Schrodinger time evolution:
        psi(t+dt) = psi(t) - i*dt*H*psi(t)
    """
    
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.device = config.device
        self.net: Optional[HamiltonianBackboneNet] = None
        self._laplacian: Optional[torch.Tensor] = None
        self._load()
        self._precompute_laplacian()
    
    def _load(self) -> None:
        """Load model from checkpoint."""
        self.net = HamiltonianBackboneNet(
            self.config.grid_size,
            self.config.hidden_dim,
            self.config.num_spectral_layers
        ).to(self.device)
        
        path = self.config.hamiltonian_checkpoint
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                self.net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                _LOG.info("HamiltonianBackend: loaded %s", path)
            except Exception as exc:
                _LOG.warning("HamiltonianBackend: load failed (%s)", exc)
        
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
    
    def _precompute_laplacian(self) -> None:
        """Precompute Laplacian kernel for kinetic energy."""
        G = self.config.grid_size
        kx = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        ky = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self._laplacian = (-(KX ** 2 + KY ** 2)).float().to(self.device)
    
    def _apply_h(self, field: torch.Tensor) -> torch.Tensor:
        """Apply Hamiltonian operator to field."""
        if self.net is not None:
            with torch.no_grad():
                result = self.net(field.to(self.device))
                return result.squeeze() if result.dim() > 2 else result
        
        fft = torch.fft.fft2(field.to(self.device))
        return torch.fft.ifft2(fft * self._laplacian).real
    
    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolve amplitude by time dt."""
        psi_r = amp[0].to(self.device)
        psi_i = amp[1].to(self.device)
        h_r = self._apply_h(psi_r)
        h_i = self._apply_h(psi_i)
        
        new_r = psi_r + dt * h_i
        new_i = psi_i - dt * h_r
        
        out = torch.stack([new_r, new_i], dim=0)
        norm = torch.sqrt((out ** 2).sum()) + self.config.normalization_eps
        return out / norm
    
    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        """Apply global phase."""
        c = math.cos(phase_angle)
        s = math.sin(phase_angle)
        return torch.stack([c * amp[0] - s * amp[1], s * amp[0] + c * amp[1]], dim=0)


class SchrodingerBackend(IPhysicsBackend):
    """
    Schrodinger backend using learned 2-channel spectral network.
    
    Falls back to HamiltonianBackend if checkpoint unavailable.
    """
    
    def __init__(self, config: FrameworkConfig, hamiltonian: HamiltonianBackend) -> None:
        self.config = config
        self.device = config.device
        self.hamiltonian = hamiltonian
        self.net: Optional[SchrodingerSpectralNet] = None
        self._load()
    
    def _load(self) -> None:
        """Load model from checkpoint."""
        self.net = SchrodingerSpectralNet(
            self.config.grid_size,
            self.config.hidden_dim,
            self.config.expansion_dim,
            self.config.num_spectral_layers
        ).to(self.device)
        
        path = self.config.schrodinger_checkpoint
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                self.net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                _LOG.info("SchrodingerBackend: loaded %s", path)
            except Exception:
                self.net = None
                return
        
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
    
    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolve amplitude by time dt."""
        if self.net is None:
            return self.hamiltonian.evolve_amplitude(amp, dt)
        
        with torch.no_grad():
            out = self.net(amp.unsqueeze(0).to(self.device)).squeeze(0)
        
        norm = torch.sqrt((out ** 2).sum()) + self.config.normalization_eps
        return out / norm
    
    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        """Apply global phase."""
        return self.hamiltonian.apply_phase(amp, phase_angle)


class DiracBackend(IPhysicsBackend):
    """
    Dirac backend for relativistic spinor evolution.
    
    Expands (2,G,G) amplitude to 4-component spinor, propagates,
    then projects back.
    """
    
    def __init__(self, config: FrameworkConfig, hamiltonian: HamiltonianBackend) -> None:
        self.config = config
        self.device = config.device
        self.hamiltonian = hamiltonian
        self.gamma = GammaMatrices(config.gamma_representation, config.device)
        self.net: Optional[DiracSpectralNet] = None
        self._load()
        self._precompute_dirac()
    
    def _load(self) -> None:
        """Load model from checkpoint."""
        self.net = DiracSpectralNet(
            self.config.grid_size,
            self.config.hidden_dim,
            self.config.expansion_dim,
            self.config.num_spectral_layers
        ).to(self.device)
        
        path = self.config.dirac_checkpoint
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                self.net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                _LOG.info("DiracBackend: loaded %s", path)
            except Exception:
                self.net = None
                return
        
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
    
    def _precompute_dirac(self) -> None:
        """Precompute momentum grids for Dirac operator."""
        G = self.config.grid_size
        kx = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        ky = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self.kx_grid = KX.to(self.device)
        self.ky_grid = KY.to(self.device)
    
    def _pack(self, amp: torch.Tensor) -> torch.Tensor:
        """Pack 2-channel amplitude to 4-component spinor."""
        G = self.config.grid_size
        psi_c = torch.complex(amp[0].to(self.device), amp[1].to(self.device))
        s = math.sqrt(0.5)
        
        spinor = torch.zeros(4, G, G, dtype=torch.complex64, device=self.device)
        spinor[0] = psi_c * s
        spinor[1] = psi_c * s
        spinor[2] = psi_c.conj() * s
        spinor[3] = psi_c.conj() * s
        
        return spinor
    
    def _unpack(self, spinor: torch.Tensor) -> torch.Tensor:
        """Unpack 4-component spinor to 2-channel amplitude."""
        particle = spinor[:2].mean(dim=0)
        out = torch.stack([particle.real, particle.imag], dim=0)
        norm = torch.sqrt((out ** 2).sum()) + self.config.normalization_eps
        return out / norm
    
    def _analytical_dirac(self, spinor: torch.Tensor) -> torch.Tensor:
        """Apply analytical Dirac Hamiltonian to spinor."""
        m, c = self.config.electron_mass, self.config.c_light
        result = torch.zeros_like(spinor)
        
        for comp in range(4):
            fft = torch.fft.fft2(spinor[comp])
            px = torch.fft.ifft2(fft * self.kx_grid)
            py = torch.fft.ifft2(fft * self.ky_grid)
            
            for row in range(4):
                result[row] += (
                    c * self.gamma.alpha_x[row, comp] * px
                    + c * self.gamma.alpha_y[row, comp] * py
                    + m * c ** 2 * self.gamma.beta[row, comp] * spinor[comp]
                )
        
        return result
    
    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolve amplitude by time dt using Dirac equation."""
        spinor = self._pack(amp)
        
        if self.net is not None:
            channels = torch.cat([spinor.real, spinor.imag], dim=0).unsqueeze(0)
            with torch.no_grad():
                out = self.net(channels).squeeze(0)
            spinor_out = torch.complex(out[:4], out[4:])
        else:
            h_spinor = self._analytical_dirac(spinor)
            spinor_out = spinor - 1j * dt * h_spinor
        
        norm = torch.sqrt((spinor_out.abs() ** 2).sum()) + self.config.normalization_eps
        return self._unpack(spinor_out / norm)
    
    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        """Apply global phase."""
        return self.hamiltonian.apply_phase(amp, phase_angle)
    
    def evolve_spinor(self, spinor: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolve full 4-component spinor by time dt."""
        spinor = spinor.to(torch.complex64)
        h_spinor = self._analytical_dirac(spinor)
        result = spinor - 1j * dt * h_spinor
        norm = torch.sqrt((result.abs() ** 2).sum()) + self.config.normalization_eps
        return result / norm


class IQuantumGate(ABC):
    """Abstract interface for quantum gates."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return gate name."""
        pass
    
    @abstractmethod
    def apply(
        self,
        state: MPSState,
        targets: Sequence[int],
        params: Optional[Dict[str, float]] = None
    ) -> MPSState:
        """Apply gate to state and return new state."""
        pass


class HadamardGate(IQuantumGate):
    """Hadamard gate: H = [[1,1],[1,-1]] / sqrt(2)."""
    
    @property
    def name(self) -> str:
        return "H"
    
    def apply(self, state, targets, params=None):
        s = 1.0 / math.sqrt(2.0)
        u = torch.tensor([[s, s], [s, -s]], dtype=torch.complex64)
        for t in targets:
            state.apply_single_qubit_gate(t, u)
        return state


class PauliXGate(IQuantumGate):
    """Pauli-X gate: X = [[0,1],[1,0]]."""
    
    @property
    def name(self) -> str:
        return "X"
    
    def apply(self, state, targets, params=None):
        u = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        for t in targets:
            state.apply_single_qubit_gate(t, u)
        return state


class PauliYGate(IQuantumGate):
    """Pauli-Y gate: Y = [[0,-i],[i,0]]."""
    
    @property
    def name(self) -> str:
        return "Y"
    
    def apply(self, state, targets, params=None):
        u = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        for t in targets:
            state.apply_single_qubit_gate(t, u)
        return state


class PauliZGate(IQuantumGate):
    """Pauli-Z gate: Z = [[1,0],[0,-1]]."""
    
    @property
    def name(self) -> str:
        return "Z"
    
    def apply(self, state, targets, params=None):
        u = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        for t in targets:
            state.apply_single_qubit_gate(t, u)
        return state


class SGate(IQuantumGate):
    """S gate: S = [[1,0],[0,i]]."""
    
    @property
    def name(self) -> str:
        return "S"
    
    def apply(self, state, targets, params=None):
        u = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)
        for t in targets:
            state.apply_single_qubit_gate(t, u)
        return state


class TGate(IQuantumGate):
    """T gate: T = [[1,0],[0,e^{i*pi/4}]]."""
    
    @property
    def name(self) -> str:
        return "T"
    
    def apply(self, state, targets, params=None):
        phase = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
        u = torch.tensor([[1, 0], [0, phase]], dtype=torch.complex64)
        for t in targets:
            state.apply_single_qubit_gate(t, u)
        return state


class RxGate(IQuantumGate):
    """Rotation-X gate: Rx(theta) = exp(-i*theta/2 * X)."""
    
    @property
    def name(self) -> str:
        return "Rx"
    
    def apply(self, state, targets, params=None):
        theta = (params or {}).get("theta", 0.0)
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        u = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex64)
        for t in targets:
            state.apply_single_qubit_gate(t, u)
        return state


class RyGate(IQuantumGate):
    """Rotation-Y gate: Ry(theta) = exp(-i*theta/2 * Y)."""
    
    @property
    def name(self) -> str:
        return "Ry"
    
    def apply(self, state, targets, params=None):
        theta = (params or {}).get("theta", 0.0)
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        u = torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)
        for t in targets:
            state.apply_single_qubit_gate(t, u)
        return state


class RzGate(IQuantumGate):
    """Rotation-Z gate: Rz(theta) = exp(-i*theta/2 * Z)."""
    
    @property
    def name(self) -> str:
        return "Rz"
    
    def apply(self, state, targets, params=None):
        theta = (params or {}).get("theta", 0.0)
        e_neg = complex(math.cos(theta / 2.0), -math.sin(theta / 2.0))
        e_pos = complex(math.cos(theta / 2.0), math.sin(theta / 2.0))
        u = torch.tensor([[e_neg, 0], [0, e_pos]], dtype=torch.complex64)
        for t in targets:
            state.apply_single_qubit_gate(t, u)
        return state


class CRzGate(IQuantumGate):
    """Controlled-Rz gate: applies Rz to target if control is |1>."""
    
    @property
    def name(self) -> str:
        return "CRz"
    
    def apply(self, state, targets, params=None):
        theta = (params or {}).get("theta", 0.0)
        e_neg = complex(math.cos(theta / 2.0), -math.sin(theta / 2.0))
        e_pos = complex(math.cos(theta / 2.0), math.sin(theta / 2.0))
        
        crz = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, e_neg, 0],
             [0, 0, 0, e_pos]],
            dtype=torch.complex64
        )
        
        if len(targets) >= 2:
            state.apply_two_qubit_gate(targets[0], targets[1], crz)
        return state


class CNOTGate(IQuantumGate):
    """CNOT gate: flips target if control is |1>."""
    
    @property
    def name(self) -> str:
        return "CNOT"
    
    def apply(self, state, targets, params=None):
        if len(targets) < 2:
            raise ValueError("CNOT requires [control, target]")
        u4 = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=torch.complex64)
        state.apply_two_qubit_gate(targets[0], targets[1], u4)
        return state


class CZGate(IQuantumGate):
    """Controlled-Z gate: applies phase -1 to |11>."""
    
    @property
    def name(self) -> str:
        return "CZ"
    
    def apply(self, state, targets, params=None):
        if len(targets) < 2:
            raise ValueError("CZ requires [control, target]")
        u4 = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ], dtype=torch.complex64)
        state.apply_two_qubit_gate(targets[0], targets[1], u4)
        return state


class SWAPGate(IQuantumGate):
    """SWAP gate: exchanges two qubits."""
    
    @property
    def name(self) -> str:
        return "SWAP"
    
    def apply(self, state, targets, params=None):
        if len(targets) < 2:
            raise ValueError("SWAP requires 2 targets")
        u4 = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=torch.complex64)
        state.apply_two_qubit_gate(targets[0], targets[1], u4)
        return state


_GATE_REGISTRY: Dict[str, IQuantumGate] = {
    "H": HadamardGate(),
    "X": PauliXGate(),
    "Y": PauliYGate(),
    "Z": PauliZGate(),
    "S": SGate(),
    "T": TGate(),
    "Rx": RxGate(),
    "Ry": RyGate(),
    "Rz": RzGate(),
    "CRz": CRzGate(),
    "CNOT": CNOTGate(),
    "CZ": CZGate(),
    "SWAP": SWAPGate(),
}


@dataclass
class CircuitInstruction:
    """Single instruction in a quantum circuit."""
    gate_name: str
    targets: List[int]
    params: Dict[str, float] = field(default_factory=dict)


class QuantumCircuit:
    """Quantum circuit builder for MPS states."""
    
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self._instructions: List[CircuitInstruction] = []
    
    def _append(
        self,
        gate_name: str,
        targets: List[int],
        params: Optional[Dict[str, float]] = None
    ) -> None:
        """Append an instruction to the circuit."""
        self._instructions.append(
            CircuitInstruction(gate_name, targets, params or {})
        )
    
    def h(self, qubit: int) -> None:
        self._append("H", [qubit])
    
    def x(self, qubit: int) -> None:
        self._append("X", [qubit])
    
    def y(self, qubit: int) -> None:
        self._append("Y", [qubit])
    
    def z(self, qubit: int) -> None:
        self._append("Z", [qubit])
    
    def s(self, qubit: int) -> None:
        self._append("S", [qubit])
    
    def t(self, qubit: int) -> None:
        self._append("T", [qubit])
    
    def rx(self, qubit: int, theta: float) -> None:
        self._append("Rx", [qubit], {"theta": theta})
    
    def ry(self, qubit: int, theta: float) -> None:
        self._append("Ry", [qubit], {"theta": theta})
    
    def rz(self, qubit: int, theta: float) -> None:
        self._append("Rz", [qubit], {"theta": theta})
    
    def crz(self, control: int, target: int, theta: float) -> None:
        self._append("CRz", [control, target], {"theta": theta})
    
    def cnot(self, control: int, target: int) -> None:
        self._append("CNOT", [control, target])
    
    def cz(self, control: int, target: int) -> None:
        self._append("CZ", [control, target])
    
    def swap(self, qubit1: int, qubit2: int) -> None:
        self._append("SWAP", [qubit1, qubit2])
    
    def run(self, state: MPSState) -> MPSState:
        """Execute circuit on state."""
        for inst in self._instructions:
            gate = _GATE_REGISTRY.get(inst.gate_name)
            if gate is None:
                raise KeyError(f"Gate '{inst.gate_name}' not found")
            state = gate.apply(state, inst.targets, inst.params)
        return state


class MPSQuantumComputer:
    """
    Main quantum computer using MPS representation.
    
    Provides:
        - State preparation
        - Circuit execution
        - Backend selection
        - Memory-efficient simulation for up to 33+ qubits
    """
    
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self._hamiltonian = HamiltonianBackend(config)
        self._backends: Dict[str, IPhysicsBackend] = {
            "hamiltonian": self._hamiltonian,
            "schrodinger": SchrodingerBackend(config, self._hamiltonian),
            "dirac": DiracBackend(config, self._hamiltonian),
        }
        self.vacuum_core: Optional[VacuumCore] = None
        self.protector = TopologicalProtector(config)
        
        _LOG.info("MPS Quantum Computer initialized")
        _LOG.info("  Max qubits: %d", config.max_qubits)
        _LOG.info("  Bond dimension: %d", config.bond_dimension)
        _LOG.info("  Backends: %s", list(self._backends.keys()))
    
    def create_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Create a new quantum circuit."""
        if n_qubits > self.config.max_qubits:
            raise ValueError(
                f"n_qubits ({n_qubits}) exceeds max_qubits ({self.config.max_qubits})"
            )
        return QuantumCircuit(n_qubits)
    
    def create_state(self, n_qubits: int) -> MPSState:
        """Create initial state |00...0>."""
        if n_qubits > self.config.max_qubits:
            raise ValueError(
                f"n_qubits ({n_qubits}) exceeds max_qubits ({self.config.max_qubits})"
            )
        
        state = MPSState(n_qubits, self.config)
        
        if self.config.enable_vacuum_core:
            self.vacuum_core = VacuumCore(n_qubits, self.config)
        
        return state
    
    def bell_state(self, n_qubits: int = 2) -> MPSState:
        """Prepare Bell state |Phi+> = (|00> + |11>) / sqrt(2)."""
        state = self.create_state(n_qubits)
        circ = self.create_circuit(n_qubits)
        circ.h(0)
        circ.cnot(0, 1)
        return circ.run(state)
    
    def ghz_state(self, n_qubits: int) -> MPSState:
        """Prepare GHZ state (|00...0> + |11...1>) / sqrt(2)."""
        state = self.create_state(n_qubits)
        circ = self.create_circuit(n_qubits)
        circ.h(0)
        for i in range(1, n_qubits):
            circ.cnot(0, i)
        return circ.run(state)
    
    def w_state(self, n_qubits: int) -> MPSState:
        """
        Prepare W state: |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n
        
        Uses direct statevector-to-MPS conversion via successive SVD.
        This guarantees exact representation (up to numerical precision).
        
        The W state has a unique entanglement structure:
        - The entanglement entropy for a cut separating k qubits from (n-k) is:
          S(k) = H({k/n, (n-k)/n}) where H is the binary entropy function
        - Maximum entropy is 1 bit when n is even and k = n/2
        - For W₃: S_max ≈ 0.9183 bits (H({1/3, 2/3}))
        - This is DIFFERENT from GHZ which has entropy = 1 bit for any cut
        
        Returns:
            MPSState: The W state in MPS representation
        """
        # W state requires bond dimension n at the center
        required_bond = min(n_qubits, self.config.max_bond_dimension)
        
        # Use direct SVD-based construction for exact representation
        return self._build_w_state_direct(n_qubits, required_bond)
    
    def _build_w_state_direct(self, n_qubits: int, max_bond: int) -> "MPSState":
        """
        Build W state using direct statevector-to-MPS conversion via successive SVD.
        
        This guarantees exact representation (up to numerical precision).
        """
        device = self.config.device
        dtype = self.config.dtype
        
        # Build the statevector
        N = 2 ** n_qubits
        psi = np.zeros(N, dtype=np.complex128)
        amp = 1.0 / np.sqrt(n_qubits)
        for i in range(n_qubits):
            pos = 1 << (n_qubits - 1 - i)
            psi[pos] = amp
        
        psi_tensor = torch.from_numpy(psi).to(dtype=torch.complex128, device=device)
        
        # Build MPS by successive SVD
        # We work left to right, splitting off one qubit at a time
        cores = []
        
        # Current tensor has shape (2, 2, ..., 2) with (n_qubits) indices
        # After each step, we remove one physical index
        current = psi_tensor.reshape(2, -1)  # Shape: (2, 2^(n-1))
        
        for i in range(n_qubits - 1):
            # SVD: current has shape (2, remaining_dim)
            u, s, vh = torch.linalg.svd(current, full_matrices=False)
            
            # Determine bond dimension
            chi = min(len(s), max_bond)
            
            # Keep significant singular values
            mask = s[:chi] > 1e-14
            if mask.sum() == 0:
                mask[0] = True  # Keep at least one
            chi = mask.sum().item()
            
            u_trunc = u[:, :chi]
            s_trunc = s[:chi]
            vh_trunc = vh[:chi, :]
            
            # First core has shape (1, 2, chi)
            if i == 0:
                core = u_trunc.reshape(1, 2, chi)  # (1, 2, chi)
            else:
                # Core has shape (chi_prev, 2, chi)
                # u_trunc has shape (chi_prev * 2, chi)
                core = u_trunc.reshape(u_trunc.shape[0] // 2, 2, chi)
            
            if core.is_complex() and not torch.any(core.imag.abs() > 1e-12):
                cores.append(core.real.to(dtype))
            else:
                cores.append(core.to(dtype) if not core.is_complex() else core.to(torch.complex128))
            
            # Update current for next iteration
            # new shape: (chi, 2^(n-i-2))
            current = torch.diag(s_trunc.to(vh_trunc.dtype)) @ vh_trunc
            # Reshape for next SVD: (chi * 2, 2^(n-i-2))
            current = current.reshape(chi * 2, -1)
        
        # Last core: current has shape (chi_prev * 2, 1)
        last_core = current.reshape(-1, 2, 1)
        if last_core.is_complex() and not torch.any(last_core.imag.abs() > 1e-12):
            cores.append(last_core.real.to(dtype))
        else:
            cores.append(last_core.to(dtype) if not last_core.is_complex() else last_core.to(torch.complex128))
        
        # Create MPS state and set cores
        state = MPSState(n_qubits, self.config)
        for i, core in enumerate(cores):
            # The MPSState creates cores with specific shapes
            # We need to handle the case where core shapes don't match
            if i < len(state._cores):
                state._cores[i].tensor = core
        
        return state
    
    def run_circuit(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[MPSState] = None
    ) -> MPSState:
        """Execute circuit on state."""
        if initial_state is None:
            initial_state = self.create_state(circuit.n_qubits)
        return circuit.run(initial_state)
    
    def get_backend(self, name: str) -> IPhysicsBackend:
        """Get physics backend by name."""
        return self._backends.get(name, self._hamiltonian)
    
    def memory_usage(self, state: MPSState) -> Dict[str, int]:
        """Compute memory usage for a state."""
        result = {"state": state.memory_bytes(), "vacuum": 0, "total": 0}
        
        if self.vacuum_core is not None:
            result["vacuum"] = (
                self.vacuum_core.vacuum_mask.numel()
                * self.vacuum_core.vacuum_mask.element_size()
            )
        
        result["total"] = result["state"] + result["vacuum"]
        return result
    
    def compression_ratio(self, state: MPSState) -> float:
        """Compute compression ratio vs full statevector."""
        n = state.n_qubits
        full_state_memory = 2 ** n * 2 * self.config.grid_size ** 2 * 8
        memory = self.memory_usage(state)
        return full_state_memory / max(memory["state"], 1)
    
    def detect_phase(self, state: MPSState) -> HilbertPhase:
        """Detect Hilbert space phase from state properties."""
        entropy = state.entropy()
        max_entropy = float(state.n_qubits)
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0
        
        avg_bond = self._compute_average_bond_dimension(state)
        
        if avg_bond <= 4:
            if entropy_ratio < 0.3:
                return HilbertPhase.PERFECT_CRYSTAL
            else:
                return HilbertPhase.TOPOLOGICAL_INSULATOR
        elif avg_bond <= 16:
            return HilbertPhase.POLYCRYSTAL
        else:
            return HilbertPhase.COLD_GLASS
    
    def _compute_average_bond_dimension(self, state: MPSState) -> float:
        """Compute average bond dimension across MPS cores."""
        total = 0
        for core in state._cores:
            total += max(core.chi_left, core.chi_right)
        return total / len(state._cores)


def run_scaling_benchmark(
    config: FrameworkConfig,
    max_qubits: int = 33
) -> Dict[str, Any]:
    """
    Run scaling benchmark to demonstrate MPS memory efficiency.
    
    Returns:
        Dictionary with qubit counts, memory usage, compression ratios.
    """
    import time
    
    results = {
        "qubits": [],
        "memory_kb": [],
        "time_seconds": [],
        "compression_ratio": [],
        "theoretical_direct_mb": [],
    }
    
    _LOG.info("Starting MPS scaling benchmark (max %d qubits)", max_qubits)
    
    for n in range(2, max_qubits + 1):
        try:
            start = time.time()
            
            qc = MPSQuantumComputer(config)
            state = qc.ghz_state(n)
            
            elapsed = time.time() - start
            memory = qc.memory_usage(state)
            compression = qc.compression_ratio(state)
            
            theoretical_direct = 2 ** n * 2 * config.grid_size ** 2 * 8 / (1024 ** 2)
            
            results["qubits"].append(n)
            results["memory_kb"].append(memory["state"] / 1024)
            results["time_seconds"].append(elapsed)
            results["compression_ratio"].append(compression)
            results["theoretical_direct_mb"].append(theoretical_direct)
            
            _LOG.info(
                "Benchmark: n=%d, memory=%.2f KB, time=%.3fs, "
                "theoretical_direct=%.2f MB, ratio=%.1fx",
                n, memory["state"] / 1024, elapsed, theoretical_direct, compression
            )
        except Exception as e:
            _LOG.error("Benchmark failed at n=%d: %s", n, e)
            break
    
    return results


def run_grover_search(
    qc: "MPSQuantumComputer",
    n_qubits: int,
    marked_states: list,
) -> dict:
    """
    Run Grover's search algorithm and return results.

    Implements the algorithm at the statevector level for correctness
    (exact oracle via direct phase flip), then reads off probabilities.

    Args:
        qc: MPSQuantumComputer instance (unused for computation, kept for API compatibility)
        n_qubits: number of qubits
        marked_states: list of integers representing marked computational basis states

    Returns:
        dict with keys: probability, marked_states (as bit strings), speedup, iterations
    """
    N = 2 ** n_qubits
    n_marked = len(marked_states)
    # Optimal Grover iterations: floor(pi/4 * sqrt(N / n_marked))
    n_iter = max(1, int(math.floor(math.pi / 4 * math.sqrt(N / max(n_marked, 1)))))

    # ---- Statevector Grover (exact for any n) --------------------------------
    # Hadamard on all qubits: uniform superposition
    psi = np.ones(N, dtype=complex) / math.sqrt(N)

    # H^⊗n as precomputed uniform state for diffusion
    s = psi.copy()  # reference equal superposition |s>

    for _ in range(n_iter):
        # Oracle: flip phase of marked states
        for idx in marked_states:
            if 0 <= idx < N:
                psi[idx] *= -1.0

        # Diffusion operator: 2|s><s| - I  =>  psi -> 2*<s|psi>*s - psi
        overlap = np.dot(s.conj(), psi)
        psi = 2.0 * overlap * s - psi

    probs_np = np.abs(psi) ** 2
    # Renormalize (should already be 1 up to float rounding)
    probs_np /= probs_np.sum()

    marked_prob = float(sum(probs_np[idx] for idx in marked_states if 0 <= idx < N))
    classical_baseline = n_marked / N

    return {
        "probability": marked_prob,
        "marked_states": [format(s, f"0{n_qubits}b") for s in marked_states],
        "speedup": marked_prob / max(classical_baseline, 1e-12),
        "iterations": n_iter,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum Framework Core")
    parser.add_argument("--config", type=str, default="", help="TOML config file")
    parser.add_argument("--benchmark", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--max-qubits", type=int, default=33, help="Max qubits for benchmark")
    args = parser.parse_args()
    
    config = FrameworkConfig.from_toml(args.config) if args.config else FrameworkConfig()
    
    if args.benchmark:
        results = run_scaling_benchmark(config, args.max_qubits)
        print("\nScaling Benchmark Results:")
        print("-" * 60)
        for i, n in enumerate(results["qubits"]):
            print(
                f"n={n:2d}: {results['memory_kb'][i]:10.2f} KB, "
                f"ratio={results['compression_ratio'][i]:.2e}x"
            )
