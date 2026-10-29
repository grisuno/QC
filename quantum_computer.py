#!/usr/bin/env python3
"""
quantum_computer.py

Author: Gris Iscomeback
License: AGPL v3

Collapse-Free Quantum Computer Simulator on Classical Hardware.

The state of n qubits lives in the JOINT Hilbert space C^(2^n).
The state vector has 2^n complex amplitudes: one per computational basis state.
This is the only representation that correctly supports entanglement.

Each amplitude alpha_k (k in {0,...,2^n - 1}) is encoded as a 2D spatial
wavefunction on a (G, G) grid, using the neural physics backends as the
time-evolution engine. The joint state tensor has shape:

    amplitudes: (2^n, 2, G, G)
        dim 0 : computational basis index  (2^n states)
        dim 1 : real / imaginary channel   (2 channels)
        dim 2 : spatial x                  (G points)
        dim 3 : spatial y                  (G points)

Single-qubit gates act via einsum on the qubit index within dim 0.
Two-qubit gates (CNOT, CZ, SWAP) permute and mix amplitude pairs in dim 0.
Measurement reads Born probabilities from norm-squared without collapsing.

Architecture (SOLID):
    - IQuantumGate         : gate abstraction (Interface Segregation)
    - IPhysicsBackend      : physics engine abstraction (Dependency Inversion)
    - JointHilbertState    : joint 2^n amplitude state (Single Responsibility)
    - HamiltonianBackend   : H(nn) spectral operator backend
    - SchrodingerBackend   : Schrodinger network backend
    - DiracBackend         : Dirac relativistic spinor backend
    - QuantumCircuit       : circuit builder (Open/Closed via IQuantumGate)
    - QuantumComputer      : top-level orchestrator (Single Responsibility)
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Canonical neural physics stack (single source of truth lives in
# quantum_framework_physics.py; re-exported here for backward compatibility).
from quantum_framework_physics import (
    SpectralLayer,
    HamiltonianBackboneNet,
    SchrodingerSpectralNet,
    DiracSpectralNet,
    GammaMatrices,
    IPhysicsBackend,
    HamiltonianBackend,
    SchrodingerBackend,
    DiracBackend,
    load_checkpoint,
)

warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("once", category=FutureWarning)
warnings.filterwarnings("module")

def _make_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a consistent formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


_LOG = _make_logger("QuantumComputer")

@dataclass
class SimulatorConfig:
    """
    Global configuration for the quantum computer simulator.

    grid_size, hidden_dim, expansion_dim, num_spectral_layers must match
    the values used when training the checkpoint files.
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

    hamiltonian_checkpoint: str = "hamiltonian.pth"
    schrodinger_checkpoint: str = "checkpoint_phase3_training_epoch_18921_20260224_154739.pth"
    dirac_checkpoint: str = "best_dirac.pth"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    max_qubits: int = 8
    propagator_dt: float = 0.01


class JointHilbertState:
    """
    Joint quantum state of n qubits in the full 2^n dimensional Hilbert space.

    The state is stored as a tensor of shape (2^n, 2, G, G):
        - dim 0: computational basis index k in {0, ..., 2^n - 1}
                 bit j of k is the state of qubit j (MSB = qubit 0)
        - dim 1: channel 0 = real part, channel 1 = imaginary part
        - dim 2: spatial x (G grid points)
        - dim 3: spatial y (G grid points)

    Each amplitude alpha_k is a spatial wavefunction. The overall quantum
    amplitude for basis state |k> is the complex field alpha_k(x,y).
    The Born probability of measuring |k> is:

        P(k) = integral |alpha_k(x,y)|^2 dx dy
             = sum_{x,y} (alpha_k_real^2 + alpha_k_imag^2)

    normalized so that sum_k P(k) = 1.

    This representation correctly supports:
        - Superposition: multiple k indices have non-zero amplitude
        - Entanglement: amplitudes do not factorize across qubits
        - Coherent multi-qubit gates: exact permutation and mixing of amplitudes
    """

    def __init__(self, amplitudes: torch.Tensor, n_qubits: int) -> None:
        expected_dim0 = 2 ** n_qubits
        if amplitudes.shape[0] != expected_dim0:
            raise ValueError(
                f"amplitudes.shape[0]={amplitudes.shape[0]} but 2^n_qubits={expected_dim0}"
            )
        self.amplitudes = amplitudes
        self.n_qubits = n_qubits
        self.dim = expected_dim0
        self.device = amplitudes.device
        self.G = amplitudes.shape[-1]

    def normalize_(self) -> None:
        """In-place normalization: sum_k P(k) = 1."""
        probs = (self.amplitudes[:, 0] ** 2 + self.amplitudes[:, 1] ** 2).sum(dim=(-2, -1))
        total = probs.sum() + 1e-12
        self.amplitudes = self.amplitudes / (total ** 0.5)

    def probabilities(self) -> torch.Tensor:
        """Return (2^n,) tensor of Born probabilities P(k) for each basis state."""
        probs = (self.amplitudes[:, 0] ** 2 + self.amplitudes[:, 1] ** 2).sum(dim=(-2, -1))
        return probs / (probs.sum() + 1e-12)

    def marginal_probability_one(self, qubit: int) -> float:
        """
        Marginal Born probability P(qubit_j = |1>).

        Sums P(k) over all basis states k where bit j == 1.
        Bit ordering: qubit 0 is the MSB of k.
        """
        probs = self.probabilities()
        bit_pos = self.n_qubits - 1 - qubit
        mask = torch.zeros(self.dim, dtype=torch.bool, device=self.device)
        for k in range(self.dim):
            if (k >> bit_pos) & 1:
                mask[k] = True
        return float(probs[mask].sum().clamp(0.0, 1.0))

    def most_probable_basis_state(self) -> int:
        """Return the index k with the highest probability."""
        return int(self.probabilities().argmax().item())

    def bloch_vector(self, qubit: int) -> Tuple[float, float, float]:
        """
        Compute the reduced Bloch vector for qubit j by partial trace.

        rho_j[0,0] = P(qubit=0), rho_j[1,1] = P(qubit=1)
        rho_j[0,1] = sum_{pairs} alpha_{k0}^* alpha_{k1} (off-diagonal coherence)
        bx = 2 Re(rho_j[0,1]), by = -2 Im(rho_j[0,1]), bz = P(0) - P(1)
        """
        probs = self.probabilities()
        bit_pos = self.n_qubits - 1 - qubit
        p0 = float(sum(probs[k] for k in range(self.dim) if not ((k >> bit_pos) & 1)))
        p1 = 1.0 - p0
        bz = p0 - p1

        re_offdiag = 0.0
        im_offdiag = 0.0
        for k0 in range(self.dim):
            if (k0 >> bit_pos) & 1:
                continue
            k1 = k0 | (1 << bit_pos)
            a0r = self.amplitudes[k0, 0]
            a0i = self.amplitudes[k0, 1]
            a1r = self.amplitudes[k1, 0]
            a1i = self.amplitudes[k1, 1]
            re_offdiag += float((a0r * a1r + a0i * a1i).sum())
            im_offdiag += float((a0r * a1i - a0i * a1r).sum())

        total_weight = float(
            (self.amplitudes[:, 0] ** 2 + self.amplitudes[:, 1] ** 2).sum()
        ) + 1e-12
        bx = 2.0 * re_offdiag / total_weight
        by = -2.0 * im_offdiag / total_weight

        mag = math.sqrt(bx ** 2 + by ** 2 + bz ** 2) + 1e-12
        if mag > 1.0:
            bx, by, bz = bx / mag, by / mag, bz / mag
        return bx, by, bz

    def clone(self) -> "JointHilbertState":
        """Return a deep copy."""
        return JointHilbertState(self.amplitudes.clone(), self.n_qubits)


class PotentialGenerator:
    """Spatial potentials for eigenstate initialization."""

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.G = config.grid_size

    def _grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(0, 2 * math.pi, self.G)
        y = torch.linspace(0, 2 * math.pi, self.G)
        return torch.meshgrid(x, y, indexing="ij")

    def harmonic(self) -> torch.Tensor:
        """V = k/2 * r^2."""
        X, Y = self._grid()
        cx, cy = math.pi, math.pi
        return 0.5 * self.config.potential_depth * (
            (X - cx) ** 2 + (Y - cy) ** 2
        ) / (math.pi ** 2)

    def double_well(self) -> torch.Tensor:
        """Double-well along x."""
        X, _ = self._grid()
        cx = math.pi
        w = self.config.potential_width * math.pi
        return self.config.potential_depth * ((X - cx) ** 2 / w ** 2 - 1.0) ** 2

    def coulomb(self) -> torch.Tensor:
        """Coulomb-like V ~ -1/r."""
        X, Y = self._grid()
        cx, cy = math.pi, math.pi
        r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2) + self.config.potential_width
        return -self.config.potential_depth / r

    def periodic_lattice(self) -> torch.Tensor:
        """Periodic cosine lattice."""
        X, Y = self._grid()
        return self.config.potential_depth * (torch.cos(2.0 * X) + torch.cos(2.0 * Y))

    def mixed(self, seed: int) -> torch.Tensor:
        """Dirichlet-weighted mixture of all four potentials."""
        rng = np.random.RandomState(seed)
        weights = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
        parts = [self.harmonic(), self.double_well(), self.coulomb(), self.periodic_lattice()]
        result = torch.zeros(self.G, self.G)
        for w, v in zip(weights, parts):
            result += float(w) * v
        return result


def _solve_eigenstate(
    config: SimulatorConfig,
    potential: torch.Tensor,
    n: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Solve the 1D marginal Hamiltonian and return the n-th eigenstate
    as a normalized 2-channel (2, G, G) real tensor.

    Uses a caller-provided torch.Generator for the random phase so the
    global RNG state is never touched (reproducibility).
    """
    G = config.grid_size
    kx = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
    h_matrix = torch.diag(-0.5 * (-(kx ** 2))) + torch.diag(potential.mean(dim=1))
    try:
        _, eigenvectors = torch.linalg.eigh(h_matrix.float())
    except Exception:
        eigenvectors = torch.eye(G)
    n_clamped = min(n, G - 1)
    psi_1d = eigenvectors[:, n_clamped]
    psi_2d = psi_1d.unsqueeze(1).expand(-1, G).clone()
    phase = torch.randn(G, G, generator=generator) * 0.1
    psi = torch.stack([psi_2d * torch.cos(phase), psi_2d * torch.sin(phase)], dim=0)
    norm = torch.sqrt((psi ** 2).sum()) + 1e-8
    return psi / norm


def _build_basis_amplitude(
    config: SimulatorConfig,
    basis_idx: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Build the (2, G, G) spatial wavefunction for amplitude at basis index basis_idx.

    Each computational basis state gets its own spatial eigenstate profile.
    The excitation level is proportional to the popcount of the basis index.
    """
    pot_gen = PotentialGenerator(config)
    popcount = bin(basis_idx).count("1")
    potential = pot_gen.mixed(seed=basis_idx * 17 + 3)
    return _solve_eigenstate(config, potential, n=popcount, generator=generator)


class JointStateFactory:
    """Builds JointHilbertState tensors for common initial conditions."""

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        # Dedicated generator: reproducible without touching the global RNG.
        self._generator = torch.Generator().manual_seed(config.random_seed)

    def _empty(self, n_qubits: int) -> torch.Tensor:
        return torch.zeros(2 ** n_qubits, 2, self.config.grid_size, self.config.grid_size,
                           device=self.config.device)

    def all_zeros(self, n_qubits: int) -> JointHilbertState:
        """Initialize register in |00...0>."""
        amps = self._empty(n_qubits)
        amps[0] = _build_basis_amplitude(self.config, 0, self._generator).to(self.config.device)
        state = JointHilbertState(amps, n_qubits)
        state.normalize_()
        return state

    def basis_state(self, n_qubits: int, k: int) -> JointHilbertState:
        """Initialize register in computational basis state |k>."""
        if k < 0 or k >= 2 ** n_qubits:
            raise ValueError(f"k={k} out of range for {n_qubits} qubits")
        amps = self._empty(n_qubits)
        amps[k] = _build_basis_amplitude(self.config, k, self._generator).to(self.config.device)
        state = JointHilbertState(amps, n_qubits)
        state.normalize_()
        return state

    def from_bitstring(self, bitstring: str) -> JointHilbertState:
        """Initialize in the basis state given by binary string."""
        return self.basis_state(len(bitstring), int(bitstring, 2))


def _single_qubit_unitary(
    state: JointHilbertState,
    qubit: int,
    u: torch.Tensor,
    backend: IPhysicsBackend,
) -> JointHilbertState:
    """
    Apply a 2x2 unitary u to qubit j in the joint Hilbert space.

    For each pair of basis states (k0, k1) that differ only in bit j:
        alpha_{k0}' = u[0,0]*alpha_{k0} + u[0,1]*alpha_{k1}
        alpha_{k1}' = u[1,0]*alpha_{k0} + u[1,1]*alpha_{k1}

    Complex scalar * (2,G,G) amplitude:
        (a+ib)(psi_r + i*psi_i) = (a*psi_r - b*psi_i) + i*(a*psi_i + b*psi_r)

    This is exact, preserves unitarity, and correctly creates superpositions.
    """
    n = state.n_qubits
    bit_pos = n - 1 - qubit
    new_amps = state.amplitudes.clone()

    u00r, u00i = float(u[0, 0].real), float(u[0, 0].imag)
    u01r, u01i = float(u[0, 1].real), float(u[0, 1].imag)
    u10r, u10i = float(u[1, 0].real), float(u[1, 0].imag)
    u11r, u11i = float(u[1, 1].real), float(u[1, 1].imag)

    processed = set()
    for k0 in range(2 ** n):
        if (k0 >> bit_pos) & 1:
            continue
        k1 = k0 | (1 << bit_pos)
        if k0 in processed:
            continue
        processed.add(k0)
        processed.add(k1)

        a0r = state.amplitudes[k0, 0]
        a0i = state.amplitudes[k0, 1]
        a1r = state.amplitudes[k1, 0]
        a1i = state.amplitudes[k1, 1]

        new_amps[k0, 0] = u00r*a0r - u00i*a0i + u01r*a1r - u01i*a1i
        new_amps[k0, 1] = u00r*a0i + u00i*a0r + u01r*a1i + u01i*a1r
        new_amps[k1, 0] = u10r*a0r - u10i*a0i + u11r*a1r - u11i*a1i
        new_amps[k1, 1] = u10r*a0i + u10i*a0r + u11r*a1i + u11i*a1r

    return JointHilbertState(new_amps, n)


def _two_qubit_unitary(
    state: JointHilbertState,
    ctrl: int,
    tgt: int,
    u4: torch.Tensor,
) -> JointHilbertState:
    """
    Apply a 4x4 unitary in the {|00>,|01>,|10>,|11>} subspace of (ctrl, tgt).

    For each group of 4 basis states sharing all bits except ctrl and tgt,
    apply the 4x4 unitary to the amplitude quadruplet.

    Ordering within the 4x4 block: |00>=0, |01>=1, |10>=2, |11>=3
    (first bit = ctrl, second bit = tgt).

    This correctly implements CNOT, CZ, SWAP and any 2-qubit gate.
    """
    n = state.n_qubits
    ctrl_bit = n - 1 - ctrl
    tgt_bit = n - 1 - tgt
    new_amps = state.amplitudes.clone()

    processed = set()
    for base in range(2 ** n):
        if (base >> ctrl_bit) & 1:
            continue
        if (base >> tgt_bit) & 1:
            continue
        k00 = base
        k01 = base | (1 << tgt_bit)
        k10 = base | (1 << ctrl_bit)
        k11 = base | (1 << ctrl_bit) | (1 << tgt_bit)
        key = (k00, k01, k10, k11)
        if key in processed:
            continue
        processed.add(key)

        indices = [k00, k01, k10, k11]
        old_r = [state.amplitudes[k, 0] for k in indices]
        old_i = [state.amplitudes[k, 1] for k in indices]

        for row, k_out in enumerate(indices):
            new_r = torch.zeros_like(old_r[0])
            new_i = torch.zeros_like(old_i[0])
            for col in range(4):
                ur = float(u4[row, col].real)
                ui = float(u4[row, col].imag)
                new_r = new_r + ur * old_r[col] - ui * old_i[col]
                new_i = new_i + ur * old_i[col] + ui * old_r[col]
            new_amps[k_out, 0] = new_r
            new_amps[k_out, 1] = new_i

    # Unitary gates preserve total norm; do not normalize here.
    return JointHilbertState(new_amps, n)

class IQuantumGate(ABC):
    """Abstract quantum gate operating on the joint Hilbert space."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Gate identifier."""

    @abstractmethod
    def apply(
        self,
        state: JointHilbertState,
        backend: IPhysicsBackend,
        targets: Sequence[int],
        params: Optional[Dict[str, float]],
    ) -> JointHilbertState:
        """Apply gate to joint state, return new joint state."""


class HadamardGate(IQuantumGate):
    """H = [[1,1],[1,-1]] / sqrt(2)."""

    @property
    def name(self) -> str:
        return "H"

    def apply(self, state, backend, targets, params):
        s = 1.0 / math.sqrt(2.0)
        u = torch.tensor([[s, s], [s, -s]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class PauliXGate(IQuantumGate):
    """X = [[0,1],[1,0]]."""

    @property
    def name(self) -> str:
        return "X"

    def apply(self, state, backend, targets, params):
        u = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class PauliYGate(IQuantumGate):
    """Y = [[0,-i],[i,0]]."""

    @property
    def name(self) -> str:
        return "Y"

    def apply(self, state, backend, targets, params):
        u = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class PauliZGate(IQuantumGate):
    """Z = [[1,0],[0,-1]]."""

    @property
    def name(self) -> str:
        return "Z"

    def apply(self, state, backend, targets, params):
        u = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class SGate(IQuantumGate):
    """S = [[1,0],[0,i]]."""

    @property
    def name(self) -> str:
        return "S"

    def apply(self, state, backend, targets, params):
        u = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class TGate(IQuantumGate):
    """T = [[1,0],[0,e^{i*pi/4}]]."""

    @property
    def name(self) -> str:
        return "T"

    def apply(self, state, backend, targets, params):
        phase = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
        u = torch.tensor([[1, 0], [0, phase]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class RxGate(IQuantumGate):
    """Rx(theta) = exp(-i*theta/2 * X)."""

    @property
    def name(self) -> str:
        return "Rx"

    def apply(self, state, backend, targets, params):
        theta = (params or {}).get("theta", 0.0)
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        u = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class RyGate(IQuantumGate):
    """Ry(theta) = exp(-i*theta/2 * Y)."""

    @property
    def name(self) -> str:
        return "Ry"

    def apply(self, state, backend, targets, params):
        theta = (params or {}).get("theta", 0.0)
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        u = torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class RzGate(IQuantumGate):
    """Rz(theta) = exp(-i*theta/2 * Z)."""

    @property
    def name(self) -> str:
        return "Rz"

    def apply(self, state, backend, targets, params):
        theta = (params or {}).get("theta", 0.0)
        e_neg = complex(math.cos(theta / 2.0), -math.sin(theta / 2.0))
        e_pos = complex(math.cos(theta / 2.0), math.sin(theta / 2.0))
        u = torch.tensor([[e_neg, 0], [0, e_pos]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class CNOTGate(IQuantumGate):
    """
    CNOT: |ctrl tgt> -> |ctrl, ctrl XOR tgt>.

    4x4 matrix (|00>,|01>,|10>,|11>):
        |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
    """

    @property
    def name(self) -> str:
        return "CNOT"

    def apply(self, state, backend, targets, params):
        if len(targets) < 2:
            raise ValueError("CNOT requires [control, target]")
        u4 = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=torch.complex64)
        return _two_qubit_unitary(state, targets[0], targets[1], u4)


class CZGate(IQuantumGate):
    """
    CZ: applies phase -1 to |11>.

    4x4 matrix: diag(1, 1, 1, -1).
    """

    @property
    def name(self) -> str:
        return "CZ"

    def apply(self, state, backend, targets, params):
        if len(targets) < 2:
            raise ValueError("CZ requires [control, target]")
        u4 = torch.tensor([
            [1, 0, 0,  0],
            [0, 1, 0,  0],
            [0, 0, 1,  0],
            [0, 0, 0, -1],
        ], dtype=torch.complex64)
        return _two_qubit_unitary(state, targets[0], targets[1], u4)


class SWAPGate(IQuantumGate):
    """
    SWAP: exchanges two qubits.

    4x4 matrix: |01>->|10>, |10>->|01>, others unchanged.
    """

    @property
    def name(self) -> str:
        return "SWAP"

    def apply(self, state, backend, targets, params):
        if len(targets) < 2:
            raise ValueError("SWAP requires 2 target indices")
        u4 = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=torch.complex64)
        return _two_qubit_unitary(state, targets[0], targets[1], u4)


class ToffoliGate(IQuantumGate):
    """
    Toffoli (CCX): flips target iff both controls are |1>.

    Exact amplitude permutation in the 8-element 3-qubit subspace.
    """

    @property
    def name(self) -> str:
        return "CCX"

    def apply(self, state, backend, targets, params):
        if len(targets) < 3:
            raise ValueError("Toffoli requires [ctrl0, ctrl1, target]")
        c0, c1, tgt = targets[0], targets[1], targets[2]
        n = state.n_qubits
        c0_bit = n - 1 - c0
        c1_bit = n - 1 - c1
        tgt_bit = n - 1 - tgt
        new_amps = state.amplitudes.clone()
        for k in range(2 ** n):
            if ((k >> c0_bit) & 1) and ((k >> c1_bit) & 1) and not ((k >> tgt_bit) & 1):
                k_flip = k | (1 << tgt_bit)
                new_amps[k] = state.amplitudes[k_flip].clone()
                new_amps[k_flip] = state.amplitudes[k].clone()
        # Exact permutation: unitary by construction, preserves total norm.
        return JointHilbertState(new_amps, n)


class MCZGate(IQuantumGate):
    """
    Multi-Controlled Z gate: applies phase -1 to the single basis state
    where ALL qubits in targets are |1>.

    This is the exact oracle primitive needed by Grover's algorithm.
    For n target qubits it marks the state |11...1> with a global phase of -1
    and leaves all other basis states unchanged.

    Implementation: iterate over all basis states k; if every bit
    corresponding to a qubit in targets is set to 1, negate that amplitude
    (multiply real and imaginary parts by -1).

    targets: list of qubit indices that must all be |1> for the phase flip.
    """

    @property
    def name(self) -> str:
        return "MCZ"

    def apply(self, state, backend, targets, params):
        if len(targets) < 1:
            raise ValueError("MCZ requires at least one target qubit")
        n = state.n_qubits
        bit_positions = [n - 1 - t for t in targets]
        new_amps = state.amplitudes.clone()
        for k in range(2 ** n):
            if all((k >> bp) & 1 for bp in bit_positions):
                new_amps[k] = -state.amplitudes[k]
        return JointHilbertState(new_amps, n)


class EvolveGate(IQuantumGate):
    """
    Free Hamiltonian evolution applied to every amplitude in the joint state.

    Uses the active physics backend.  The evolution preserves each
    amplitude's norm (unitary dynamics keeps ||U psi|| = ||psi||), so the
    Born weights of a superposition are conserved -- previously each
    amplitude was renormalized to 1, which silently flattened any
    non-uniform superposition into a uniform one.

    Note: in the joint Hilbert space representation every computational
    basis amplitude is a full spatial wavefunction, so the evolution is
    inherently global; ``targets`` is accepted for API compatibility.
    ``dt`` defaults to the backend configuration value.

    params: {"dt": float, "steps": int}
    """

    @property
    def name(self) -> str:
        return "Evolve"

    def apply(self, state, backend, targets, params):
        p = params or {}
        default_dt = getattr(getattr(backend, "config", None), "dt", 0.01)
        dt = p.get("dt") if p.get("dt") is not None else default_dt
        steps = int(p.get("steps", 1))
        new_amps = state.amplitudes.clone()
        for k in range(state.dim):
            amp = new_amps[k]
            if (amp ** 2).sum() < 1e-20:
                continue
            for _ in range(steps):
                amp = backend.evolve_amplitude(amp, dt)
            new_amps[k] = amp
        # Per-amplitude norm preservation implies global norm preservation;
        # no renormalization is applied (that would mask physics errors).
        return JointHilbertState(new_amps, state.n_qubits)


_GATE_REGISTRY: Dict[str, IQuantumGate] = {
    "H":      HadamardGate(),
    "X":      PauliXGate(),
    "Y":      PauliYGate(),
    "Z":      PauliZGate(),
    "S":      SGate(),
    "T":      TGate(),
    "Rx":     RxGate(),
    "Ry":     RyGate(),
    "Rz":     RzGate(),
    "CNOT":   CNOTGate(),
    "CX":     CNOTGate(),
    "CZ":     CZGate(),
    "SWAP":   SWAPGate(),
    "CCX":    ToffoliGate(),
    "MCZ":    MCZGate(),
    "Evolve": EvolveGate(),
}


def register_gate(name: str, gate: IQuantumGate) -> None:
    """Register a custom gate without modifying existing code (Open/Closed Principle)."""
    _GATE_REGISTRY[name] = gate



@dataclass
class CircuitInstruction:
    """Single gate instruction."""
    gate_name: str
    targets: List[int]
    params: Optional[Dict[str, float]] = None


class QuantumCircuit:
    """
    Ordered sequence of quantum gate instructions.

    Pure data structure: stores instructions, does not execute them.
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self._instructions: List[CircuitInstruction] = []

    def h(self, q: int) -> "QuantumCircuit":
        return self._append("H", [q])

    def x(self, q: int) -> "QuantumCircuit":
        return self._append("X", [q])

    def y(self, q: int) -> "QuantumCircuit":
        return self._append("Y", [q])

    def z(self, q: int) -> "QuantumCircuit":
        return self._append("Z", [q])

    def s(self, q: int) -> "QuantumCircuit":
        return self._append("S", [q])

    def t(self, q: int) -> "QuantumCircuit":
        return self._append("T", [q])

    def rx(self, q: int, theta: float) -> "QuantumCircuit":
        return self._append("Rx", [q], {"theta": theta})

    def ry(self, q: int, theta: float) -> "QuantumCircuit":
        return self._append("Ry", [q], {"theta": theta})

    def rz(self, q: int, theta: float) -> "QuantumCircuit":
        return self._append("Rz", [q], {"theta": theta})

    def cnot(self, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self._append("CNOT", [ctrl, tgt])

    def cx(self, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self.cnot(ctrl, tgt)

    def cz(self, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self._append("CZ", [ctrl, tgt])

    def swap(self, a: int, b: int) -> "QuantumCircuit":
        return self._append("SWAP", [a, b])

    def toffoli(self, c0: int, c1: int, tgt: int) -> "QuantumCircuit":
        return self._append("CCX", [c0, c1, tgt])

    def ccx(self, c0: int, c1: int, tgt: int) -> "QuantumCircuit":
        return self.toffoli(c0, c1, tgt)

    def evolve(self, qubits: List[int], dt: Optional[float] = None, steps: int = 1) -> "QuantumCircuit":
        return self._append("Evolve", qubits, {"dt": dt, "steps": steps})

    def barrier(self) -> "QuantumCircuit":
        return self

    def _append(self, gate_name: str, targets: List[int],
                params: Optional[Dict[str, float]] = None) -> "QuantumCircuit":
        for idx in targets:
            if idx < 0 or idx >= self.n_qubits:
                raise IndexError(f"Qubit index {idx} out of range for {self.n_qubits} qubits")
        self._instructions.append(CircuitInstruction(gate_name, targets, params))
        return self

    def depth(self) -> int:
        return len(self._instructions)

    def __len__(self) -> int:
        return len(self._instructions)

    def __repr__(self) -> str:
        lines = [f"QuantumCircuit({self.n_qubits} qubits, depth={self.depth()})"]
        for i, inst in enumerate(self._instructions):
            lines.append(f"  [{i:03d}] {inst.gate_name} q{inst.targets}")
        return "\n".join(lines)


@dataclass
class MeasurementResult:
    """
    Non-destructive Born-rule measurement of the full register.

    Contains the complete probability distribution over all 2^n basis states,
    per-qubit marginals, and Bloch vectors. The state is never modified.
    """

    full_distribution: Dict[str, float]
    marginal_p1: Dict[int, float]
    bloch_vectors: Dict[int, Tuple[float, float, float]]
    n_qubits: int

    @property
    def probabilities(self) -> Dict[int, float]:
        """Alias: marginal P(|1>) per qubit index."""
        return self.marginal_p1

    def most_probable_bitstring(self) -> str:
        """Return the bitstring with the highest probability."""
        return max(self.full_distribution, key=self.full_distribution.get)

    def expectation_z(self, qubit: int) -> float:
        """<Z>_j = P(0) - P(1) in [-1, +1]."""
        return 1.0 - 2.0 * self.marginal_p1[qubit]

    def entropy(self) -> float:
        """Shannon entropy of the full probability distribution in bits."""
        h = 0.0
        for p in self.full_distribution.values():
            if p > 1e-15:
                h -= p * math.log2(p)
        return h

    def __repr__(self) -> str:
        lines = [f"MeasurementResult ({self.n_qubits} qubits)"]
        bs = self.most_probable_bitstring()
        lines.append(f"  Most probable: |{bs}>  P={self.full_distribution[bs]:.4f}")
        lines.append(f"  Shannon entropy: {self.entropy():.4f} bits")
        top4 = sorted(self.full_distribution.items(), key=lambda x: -x[1])[:4]
        lines.append("  Top states:")
        for s, p in top4:
            if p > 1e-6:
                lines.append(f"    |{s}>  P={p:.4f}")
        lines.append("  Per-qubit marginals:")
        for i in range(self.n_qubits):
            bx, by, bz = self.bloch_vectors[i]
            lines.append(
                f"    q{i}: P(|1>)={self.marginal_p1[i]:.4f}  "
                f"<Z>={self.expectation_z(i):+.4f}  "
                f"Bloch=({bx:+.3f},{by:+.3f},{bz:+.3f})"
            )
        return "\n".join(lines)

class QuantumComputer:
    """
    Collapse-free quantum computer simulator with joint Hilbert space.

    The n-qubit state is stored as a (2^n, 2, G, G) tensor. All gates
    operate via exact unitary transformations on amplitude pairs. Measurement
    is non-destructive Born-rule readout — the state is never collapsed.

    Backends:
        "hamiltonian" : Hamiltonian NN spectral operator
        "schrodinger" : Schrodinger evolution network
        "dirac"       : Dirac relativistic spinor network

    Usage:
        config = SimulatorConfig(
            hamiltonian_checkpoint="weights/latest.pth",
            schrodinger_checkpoint="weights/schrodinger_crystal_final.pth",
            dirac_checkpoint="weights/dirac_phase5_latest.pth",
        )
        qc = QuantumComputer(config)
        circuit = QuantumCircuit(2)
        circuit.h(0).cnot(0, 1)
        result = qc.run(circuit, backend="schrodinger")
        print(result)
        # Most probable: |00> P=0.5  or  |11> P=0.5
        # Shannon entropy: 1.0000 bits  <- true entanglement
    """

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        self.config = config or SimulatorConfig()
        # No global RNG seeding: randomness flows through the dedicated
        # generator owned by JointStateFactory (see _build_basis_amplitude).
        self._h_backend = HamiltonianBackend(self.config)
        self._s_backend = SchrodingerBackend(self.config, self._h_backend)
        self._d_backend = DiracBackend(self.config, self._h_backend)
        self._backends: Dict[str, IPhysicsBackend] = {
            "hamiltonian": self._h_backend,
            "schrodinger": self._s_backend,
            "dirac": self._d_backend,
        }
        self._factory = JointStateFactory(self.config)

    def _select_backend(self, name: str) -> IPhysicsBackend:
        if name not in self._backends:
            raise ValueError(f"Unknown backend '{name}'. Options: {list(self._backends.keys())}")
        return self._backends[name]

    def _state_to_result(self, state: JointHilbertState) -> MeasurementResult:
        probs = state.probabilities()
        n = state.n_qubits
        full_dist = {format(k, f"0{n}b"): float(probs[k]) for k in range(state.dim)}
        marginal = {j: state.marginal_probability_one(j) for j in range(n)}
        bloch = {j: state.bloch_vector(j) for j in range(n)}
        return MeasurementResult(full_dist, marginal, bloch, n)

    def run(
        self,
        circuit: QuantumCircuit,
        backend: str = "schrodinger",
        initial_states: Optional[Dict[int, str]] = None,
    ) -> MeasurementResult:
        """
        Execute a quantum circuit on the joint Hilbert space.

        Args:
            circuit:        The QuantumCircuit to execute.
            backend:        Physics backend name.
            initial_states: Optional {qubit_idx: "0" or "1"}.

        Returns:
            Non-destructive MeasurementResult with full distribution.
        """
        be = self._select_backend(backend)
        state = self._factory.all_zeros(circuit.n_qubits)
        if initial_states:
            for qubit_idx, val in initial_states.items():
                if val not in ("0", "1"):
                    raise ValueError(f"initial_states value must be '0' or '1', got '{val}'")
                if val == "1":
                    state = PauliXGate().apply(state, be, [qubit_idx], None)
        for inst in circuit._instructions:
            gate = _GATE_REGISTRY.get(inst.gate_name)
            if gate is None:
                raise KeyError(f"Gate '{inst.gate_name}' not found in registry")
            state = gate.apply(state, be, inst.targets, inst.params)
        return self._state_to_result(state)

    def run_with_state_snapshots(
        self,
        circuit: QuantumCircuit,
        backend: str = "schrodinger",
        snapshot_after: Optional[List[int]] = None,
    ) -> Tuple[MeasurementResult, List[Dict[str, float]]]:
        """
        Execute circuit with non-destructive probability snapshots.

        The state is never collapsed between snapshots.
        """
        be = self._select_backend(backend)
        state = self._factory.all_zeros(circuit.n_qubits)
        snapshots: List[Dict[str, float]] = []
        snap_set = set(
            snapshot_after if snapshot_after is not None
            else range(len(circuit._instructions))
        )
        for step_idx, inst in enumerate(circuit._instructions):
            gate = _GATE_REGISTRY.get(inst.gate_name)
            if gate is None:
                raise KeyError(f"Gate '{inst.gate_name}' not found in registry")
            state = gate.apply(state, be, inst.targets, inst.params)
            if step_idx in snap_set:
                probs = state.probabilities()
                n = state.n_qubits
                snapshots.append({format(k, f"0{n}b"): float(probs[k]) for k in range(state.dim)})
        return self._state_to_result(state), snapshots

    def bell_state(self, backend: str = "schrodinger") -> MeasurementResult:
        """
        |Phi+> = (|00> + |11>) / sqrt(2).

        Expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit.
        """
        c = QuantumCircuit(2)
        c.h(0).cnot(0, 1)
        return self.run(c, backend=backend)

    def ghz_state(self, n_qubits: int = 3, backend: str = "schrodinger") -> MeasurementResult:
        """
        (|00...0> + |11...1>) / sqrt(2).

        Expected: P(|00...0>)=P(|11...1>)=0.5, all others 0.
        """
        c = QuantumCircuit(n_qubits)
        c.h(0)
        for i in range(n_qubits - 1):
            c.cnot(i, i + 1)
        return self.run(c, backend=backend)

    def quantum_fourier_transform(self, n_qubits: int,
                                   backend: str = "schrodinger") -> MeasurementResult:
        """QFT on |00...0>. Standard H + controlled-Rz decomposition."""
        c = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            c.h(i)
            for j in range(i + 1, n_qubits):
                c.rz(j, math.pi / (2 ** (j - i)))
        return self.run(c, backend=backend)

    def grover_oracle_search(self, n_qubits: int, target_bitstring: str,
                              backend: str = "schrodinger",
                              n_iterations: Optional[int] = None) -> MeasurementResult:
        """
        Grover's search algorithm with correct phase oracle and diffusion operator.

        The optimal number of iterations is floor(pi/4 * sqrt(2^n)) which gives
        the highest probability of measuring the target state.

        Oracle construction for target |t_0 t_1 ... t_{n-1}>:
            1. Apply X to every qubit i where t_i == '0'.
               This maps the target bitstring to |11...1>.
            2. Apply MCZ on all n qubits.
               MCZ flips the phase of |11...1> -> exactly the target state
               (after the X conjugation) gets phase -1.
            3. Undo the X gates from step 1.

        Diffusion operator (inversion about the mean):
            H^n  X^n  MCZ  X^n  H^n

        Both steps use MCZGate which applies phase -1 to the unique basis state
        where all specified qubits are |1>. This is exact for any n.

        Args:
            n_qubits:        Number of qubits.
            target_bitstring: Binary string of length n_qubits.
            backend:         Physics backend name.
            n_iterations:    Number of Grover iterations. Defaults to
                             max(1, round(pi/4 * sqrt(2^n_qubits))).

        Returns:
            MeasurementResult. The target bitstring should have the highest
            probability after the optimal number of iterations.
        """
        if len(target_bitstring) != n_qubits:
            raise ValueError(f"target_bitstring length must equal n_qubits={n_qubits}")

        optimal = max(1, round(math.pi / 4.0 * math.sqrt(2 ** n_qubits)))
        iters = n_iterations if n_iterations is not None else optimal

        c = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))

        # Initial equal superposition
        for i in range(n_qubits):
            c.h(i)

        for _ in range(iters):
            # --- Phase oracle: flip phase of target_bitstring ---
            # Step 1: X on qubits where target bit is '0'
            for i, bit in enumerate(target_bitstring):
                if bit == "0":
                    c.x(i)
            # Step 2: MCZ on all qubits -> phase -1 on |11...1> = target
            c._append("MCZ", all_qubits)
            # Step 3: undo X
            for i, bit in enumerate(target_bitstring):
                if bit == "0":
                    c.x(i)

            # --- Diffusion operator: inversion about the mean ---
            # H^n
            for i in range(n_qubits):
                c.h(i)
            # X^n
            for i in range(n_qubits):
                c.x(i)
            # MCZ on all qubits -> phase -1 on |11...1> = |00...0> after H+X
            c._append("MCZ", all_qubits)
            # X^n
            for i in range(n_qubits):
                c.x(i)
            # H^n
            for i in range(n_qubits):
                c.h(i)

        return self.run(c, backend=backend)

    def variational_ansatz(self, n_qubits: int, n_layers: int, thetas: List[float],
                            backend: str = "schrodinger") -> MeasurementResult:
        """Hardware-efficient ansatz: Ry layers + CNOT chain. len(thetas)=n_qubits*n_layers."""
        if len(thetas) != n_qubits * n_layers:
            raise ValueError(f"Expected {n_qubits * n_layers} thetas, got {len(thetas)}")
        c = QuantumCircuit(n_qubits)
        for layer in range(n_layers):
            for q in range(n_qubits):
                c.ry(q, thetas[layer * n_qubits + q])
            for q in range(n_qubits - 1):
                c.cnot(q, q + 1)
        return self.run(c, backend=backend)

    def teleportation(self, backend: str = "schrodinger") -> MeasurementResult:
        """
        3-qubit teleportation protocol.

        q0 prepared in Ry(pi/3). q2 should match q0's state after corrections.
        """
        c = QuantumCircuit(3)
        c.ry(0, math.pi / 3.0)
        c.h(1).cnot(1, 2)
        c.cnot(0, 1).h(0)
        c.cx(1, 2).cz(0, 2)
        return self.run(c, backend=backend)

    def deutsch_jozsa(self, n_input_qubits: int, is_constant: bool,
                       backend: str = "schrodinger") -> MeasurementResult:
        """
        Deutsch-Jozsa: constant -> all inputs |0>, balanced -> at least one |1>.
        """
        n = n_input_qubits + 1
        ancilla = n - 1
        c = QuantumCircuit(n)
        c.x(ancilla)
        for i in range(n):
            c.h(i)
        if not is_constant:
            c.cnot(0, ancilla)
        for i in range(n_input_qubits):
            c.h(i)
        return self.run(c, backend=backend)

_CHECK_LOG: List[bool] = []


def _check(label: str, condition: bool) -> bool:
    """Print PASS/FAIL for a single assertion. Returns True if passed."""
    _CHECK_LOG.append(condition)
    status = "PASS" if condition else "FAIL"
    _LOG.info("  [%s] %s", status, label)
    return condition


def run_phase_tests(config: SimulatorConfig) -> int:
    """
    Property-based test suite for quantum phase coherence and unitarity.

    Each test has a known exact analytic answer derived from the unitary
    algebra. Tests are designed to be sensitive to phase errors — circuits
    where wrong relative phases produce DIFFERENT probabilities, not just
    different phases that cancel out at measurement.

    Returns the number of failed tests.
    """
    qc = QuantumComputer(config)
    _LOG.info("")
    _LOG.info("=" * 70)
    _LOG.info("PHASE COHERENCE & UNITARITY TEST SUITE")
    _LOG.info("=" * 70)
    _CHECK_LOG.clear()
    failures = 0
    tol = 0.05  
    _LOG.info("\n--- Group 1: Single-qubit phase algebra ---")

    # HZH = X  =>  |0> -> |1>  (P(1) = 1.0)
    c = QuantumCircuit(1)
    c.h(0).z(0).h(0)
    r = qc.run(c)
    p1 = r.marginal_p1[0]
    ok = abs(p1 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"HZH = X  (|0>->|1>):  P(|1>)={p1:.4f}  expected=1.0", ok)

    # H X H = Z  =>  |0> stays |0>  (P(1) = 0.0)
    c = QuantumCircuit(1)
    c.h(0).x(0).h(0)
    r = qc.run(c)
    p1 = r.marginal_p1[0]
    ok = abs(p1 - 0.0) < tol
    if not ok:
        failures += 1
    _check(f"HXH = Z  (|0>->|0>):  P(|1>)={p1:.4f}  expected=0.0", ok)

    # H S H applied twice = H S^2 H = H Z H = X  =>  |0> -> |1>  (P(1) = 1.0)
    
    c = QuantumCircuit(1)
    c.h(0).s(0).s(0).h(0)   # H S S H = H Z H = X
    r = qc.run(c)
    p1 = r.marginal_p1[0]
    ok = abs(p1 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"HSSH = HZH = X  (P(|1>)={p1:.4f}  expected=1.0)", ok)

    # Rz(pi) = Z up to global phase  =>  H Rz(pi) H = HZH = X
    c = QuantumCircuit(1)
    c.h(0).rz(0, math.pi).h(0)
    r = qc.run(c)
    p1 = r.marginal_p1[0]
    ok = abs(p1 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"H Rz(pi) H = X  (P(|1>)={p1:.4f}  expected=1.0)", ok)

    # Ry(pi) = Y up to phase  =>  |0> -> |1>
    c = QuantumCircuit(1)
    c.ry(0, math.pi)
    r = qc.run(c)
    p1 = r.marginal_p1[0]
    ok = abs(p1 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"Ry(pi)|0> = |1>  (P(|1>)={p1:.4f}  expected=1.0)", ok)

    # X X = I  =>  |0> -> |0>
    c = QuantumCircuit(1)
    c.x(0).x(0)
    r = qc.run(c)
    p1 = r.marginal_p1[0]
    ok = abs(p1 - 0.0) < tol
    if not ok:
        failures += 1
    _check(f"XX = I  (|0>->|0>):  P(|1>)={p1:.4f}  expected=0.0", ok)

    # Z Z = I  =>  H |0> -> H|0> = equal super; Z Z H|0> = H|0>; H Z Z H|0> = |0>
    c = QuantumCircuit(1)
    c.h(0).z(0).z(0).h(0)
    r = qc.run(c)
    p1 = r.marginal_p1[0]
    ok = abs(p1 - 0.0) < tol
    if not ok:
        failures += 1
    _check(f"HZZH = H I H = I  (P(|1>)={p1:.4f}  expected=0.0)", ok)

    # Rx(pi) ≈ -iX  =>  |0> -> -i|1>, P(1) = 1.0
    c = QuantumCircuit(1)
    c.rx(0, math.pi)
    r = qc.run(c)
    p1 = r.marginal_p1[0]
    ok = abs(p1 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"Rx(pi)|0> = |1>  (P(|1>)={p1:.4f}  expected=1.0)", ok)

    
    _LOG.info("\n--- Group 2: Two-qubit phase-sensitive interference ---")

    # H(0) CNOT(0,1) CNOT(0,1) H(0) = identity  =>  |00> -> |00>
    c = QuantumCircuit(2)
    c.h(0).cnot(0, 1).cnot(0, 1).h(0)
    r = qc.run(c)
    p00 = r.full_distribution.get("00", 0.0)
    ok = abs(p00 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"H CNOT CNOT H = I  (P(|00>)={p00:.4f}  expected=1.0)", ok)

    # CZ CZ = I  =>  Bell state created and uncreated
    # H(0) CNOT(0,1) CZ(0,1) CZ(0,1) CNOT(0,1) H(0) |00> = |00>
    c = QuantumCircuit(2)
    c.h(0).cnot(0, 1).cz(0, 1).cz(0, 1).cnot(0, 1).h(0)
    r = qc.run(c)
    p00 = r.full_distribution.get("00", 0.0)
    ok = abs(p00 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"H CNOT CZ CZ CNOT H = I  (P(|00>)={p00:.4f}  expected=1.0)", ok)

    # Phase interference: H(0) CNOT(0,1) Z(0) CNOT(0,1) H(0)
    # = H Z_ctrl CNOT back = should give |10> (ctrl gets Z, then undone CNOT)
    # Exact algebra: HZH=X on qubit 0 after round-trip, so |00> -> |10>
    c = QuantumCircuit(2)
    c.h(0).cnot(0, 1).z(0).cnot(0, 1).h(0)
    r = qc.run(c)
    p10 = r.full_distribution.get("10", 0.0)
    ok = abs(p10 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"H CNOT Z(ctrl) CNOT H = X(0)  (P(|10>)={p10:.4f}  expected=1.0)", ok)

    # SWAP SWAP = I  =>  |01> -> |01>
    c = QuantumCircuit(2)
    c.x(1)  # prepare |01>
    c.swap(0, 1).swap(0, 1)
    r = qc.run(c)
    p01 = r.full_distribution.get("01", 0.0)
    ok = abs(p01 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"X(1) SWAP SWAP = I  (P(|01>)={p01:.4f}  expected=1.0)", ok)

    # SWAP  =>  |01> -> |10>
    c = QuantumCircuit(2)
    c.x(1)  # prepare |01>
    c.swap(0, 1)
    r = qc.run(c)
    p10 = r.full_distribution.get("10", 0.0)
    ok = abs(p10 - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"SWAP |01> = |10>  (P(|10>)={p10:.4f}  expected=1.0)", ok)
    _LOG.info("\n--- Group 3: Norm preservation (unitarity) ---")

    circuits_to_check = [
        ("H",          lambda c: c.h(0),                    1),
        ("X",          lambda c: c.x(0),                    1),
        ("HXH",        lambda c: c.h(0).x(0).h(0),          1),
        ("Bell",       lambda c: c.h(0).cnot(0, 1),         2),
        ("GHZ",        lambda c: [c.h(0)] + [c.cnot(i, i+1) for i in range(2)], 3),
        ("QFT-3",      lambda c: [c.h(i) for i in range(3)], 3),
    ]
    for name, builder, nq in circuits_to_check:
        c = QuantumCircuit(nq)
        builder(c)
        result = qc.run(c)
        total_prob = sum(result.full_distribution.values())
        ok = abs(total_prob - 1.0) < 1e-4
        if not ok:
            failures += 1
        _check(f"Norm preserved after {name}: sum(P)={total_prob:.8f}  expected=1.0", ok)


    _LOG.info("\n--- Group 4: Entanglement (Shannon entropy) ---")


    r = qc.bell_state()
    ent = r.entropy()
    ok = abs(ent - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"Bell state entropy = 1 bit  (got {ent:.4f})", ok)

    r = qc.ghz_state(3)
    ent = r.entropy()
    ok = abs(ent - 1.0) < tol
    if not ok:
        failures += 1
    _check(f"GHZ-3 entropy = 1 bit  (got {ent:.4f})", ok)

    # QFT on 3 qubits: all 8 states equiprobable => entropy = 3 bits
    r = qc.quantum_fourier_transform(3)
    ent = r.entropy()
    ok = abs(ent - 3.0) < tol
    if not ok:
        failures += 1
    _check(f"QFT-3 entropy = 3 bits  (got {ent:.4f})", ok)

    # |0> state has 0 entropy
    c = QuantumCircuit(1)
    r = qc.run(c)
    ent = r.entropy()
    ok = abs(ent - 0.0) < tol
    if not ok:
        failures += 1
    _check(f"|0> entropy = 0 bits  (got {ent:.4f})", ok)

    
    # Summary (counted programmatically from the checks actually executed)
    total = len(_CHECK_LOG)
    failures = _CHECK_LOG.count(False)
    passed = total - failures
    _LOG.info("")
    _LOG.info("=" * 70)
    if failures == 0:
        _LOG.info("ALL TESTS PASSED  (%d/%d)", passed, total)
    else:
        _LOG.info("TESTS COMPLETE: %d passed, %d FAILED  (%d/%d)", passed, failures, passed, total)
    _LOG.info("=" * 70)
    return failures


def _demo(config: SimulatorConfig) -> None:
    """Run demo suite validating entanglement on all backends."""
    qc = QuantumComputer(config)
    _LOG.info("=" * 70)
    _LOG.info("QUANTUM COMPUTER SIMULATOR - JOINT HILBERT SPACE")
    _LOG.info("=" * 70)

    for backend_name in ("hamiltonian", "schrodinger", "dirac"):
        _LOG.info("\n--- Backend: %s ---", backend_name.upper())

        _LOG.info("[Bell State]  expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit")
        _LOG.info("%s", qc.bell_state(backend=backend_name))

        _LOG.info("[GHZ 3q]  expected: P(|000>)=0.5, P(|111>)=0.5")
        _LOG.info("%s", qc.ghz_state(n_qubits=3, backend=backend_name))

        _LOG.info("[Deutsch-Jozsa constant]  expected: input qubits -> |0>")
        _LOG.info("%s", qc.deutsch_jozsa(2, is_constant=True, backend=backend_name))

        _LOG.info("[Deutsch-Jozsa balanced]  expected: input NOT all |0>")
        _LOG.info("%s", qc.deutsch_jozsa(2, is_constant=False, backend=backend_name))

        _LOG.info("[QFT 3q]")
        _LOG.info("%s", qc.quantum_fourier_transform(3, backend=backend_name))

        _LOG.info("[Grover |101>]  expected: |101> amplified ~94%")
        _LOG.info("%s", qc.grover_oracle_search(3, "101", backend=backend_name))

        _LOG.info("[Teleportation]  expected: q2 matches q0 initial state")
        _LOG.info("%s", qc.teleportation(backend=backend_name))

        _LOG.info("[Snapshots: H-CNOT-Z-H]")
        _LOG.info("  (Note: uniform distribution at step 3 is mathematically correct --")
        _LOG.info("   Z introduces phase -1 on |11>, then H(0) produces (+,−,+,+)/2.")
        _LOG.info("   All four |P|²=0.25, phases differ but are unobservable in Born rule.)")
        circ = QuantumCircuit(2)
        circ.h(0).cnot(0, 1).z(1).h(0)
        final, snaps = qc.run_with_state_snapshots(
            circ, backend=backend_name, snapshot_after=[0, 1, 2, 3]
        )
        for i, snap in enumerate(snaps):
            top = sorted(snap.items(), key=lambda x: -x[1])[:2]
            _LOG.info("  step %d: %s", i,
                      "  ".join(f"|{s}> {p:.3f}" for s, p in top if p > 0.01))
        _LOG.info("  final:\n%s", final)


    run_phase_tests(config)

    _LOG.info("=" * 70)
    _LOG.info("DEMO COMPLETE")
    _LOG.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collapse-Free Neural Quantum Computer Simulator")
    parser.add_argument("--grid-size",              type=int,   default=16)
    parser.add_argument("--hidden-dim",             type=int,   default=32)
    parser.add_argument("--expansion-dim",          type=int,   default=64)
    parser.add_argument("--num-spectral-layers",    type=int,   default=2)
    parser.add_argument("--hamiltonian-checkpoint", type=str,   default="hamiltonian.pth")
    parser.add_argument("--schrodinger-checkpoint", type=str,   default="checkpoint_phase3_training_epoch_18921_20260224_154739.pth")
    parser.add_argument("--dirac-checkpoint",       type=str,   default="best_dirac.pth")
    parser.add_argument("--dirac-mass",             type=float, default=1.0)
    parser.add_argument("--seed",                   type=int,   default=42)
    parser.add_argument("--device",                 type=str,   default=None)
    args = parser.parse_args()

    cfg = SimulatorConfig(
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        expansion_dim=args.expansion_dim,
        num_spectral_layers=args.num_spectral_layers,
        hamiltonian_checkpoint=args.hamiltonian_checkpoint,
        schrodinger_checkpoint=args.schrodinger_checkpoint,
        dirac_checkpoint=args.dirac_checkpoint,
        dirac_mass=args.dirac_mass,
        random_seed=args.seed,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    _demo(cfg)