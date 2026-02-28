#!/usr/bin/env python3
"""
Quantum Simulation Framework - Complete Edition
================================================
Self-contained quantum simulation framework with:
- Quantum circuit simulation (Hamiltonian, Schrodinger, Dirac backends)
- Atomic orbital visualization with Monte Carlo sampling
- Molecular VQE/UCCSD simulation
- Relativistic hydrogen with fine structure and Zitterbewegung
- Entangled state visualization
- Multi-atom orbital support

All configuration via TOML file - no hardcoded values.
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

try:
    from scipy.special import factorial, genlaguerre, sph_harm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _make_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


_LOG = _make_logger("QuantumFramework")


@dataclass
class FrameworkConfig:
    grid_size: int = 16
    hidden_dim: int = 32
    expansion_dim: int = 64
    num_spectral_layers: int = 2
    spinor_components: int = 4
    c_light: float = 137.035999084
    alpha_fs: float = 0.0072973525693
    hbar: float = 1.0
    electron_mass: float = 1.0
    dt: float = 0.01
    normalization_eps: float = 1.0e-8
    potential_depth: float = 5.0
    potential_width: float = 0.3
    device: str = "cpu"
    random_seed: int = 42
    max_qubits: int = 8
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
    zbw_time_steps: int = 1000
    zbw_dt: float = 0.001
    zbw_packet_width: float = 0.1
    fine_structure_orders: int = 4

    @classmethod
    def from_toml(cls, toml_path: str) -> "FrameworkConfig":
        if not os.path.exists(toml_path):
            return cls()
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        sim = data.get("simulation", {})
        grid = data.get("grid", {})
        physics = data.get("physics", {})
        mc = data.get("monte_carlo", {})
        vis = data.get("visualization", {})
        ckpt = data.get("checkpoints", {})
        out = data.get("output", {})
        return cls(
            grid_size=grid.get("size", 16),
            hidden_dim=grid.get("hidden_dim", 32),
            expansion_dim=grid.get("expansion_dim", 64),
            num_spectral_layers=grid.get("num_spectral_layers", 2),
            c_light=physics.get("c_light", 137.035999084),
            alpha_fs=physics.get("alpha_fs", 0.0072973525693),
            hbar=physics.get("hbar", 1.0),
            electron_mass=physics.get("electron_mass", 1.0),
            dt=physics.get("dt", 0.01),
            normalization_eps=physics.get("normalization_eps", 1.0e-8),
            potential_depth=physics.get("potential_depth", 5.0),
            potential_width=physics.get("potential_width", 0.3),
            random_seed=sim.get("random_seed", 42),
            max_qubits=sim.get("max_qubits", 8),
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
        )


@dataclass
class AtomData:
    symbol: str
    name: str
    atomic_number: int
    mass: float
    electron_configuration: List[str]
    nuclear_charge: float


@dataclass
class MoleculeData:
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
    h_core: Optional[np.ndarray] = None
    eri: Optional[np.ndarray] = None


@dataclass
class OrbitalData:
    name: str
    n: int
    l: int
    m: int
    description: str


class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config()
        self._raw_data: Dict[str, Any] = {}
        self._atoms: Dict[str, AtomData] = {}
        self._molecules: Dict[str, MoleculeData] = {}
        self._orbitals: Dict[str, OrbitalData] = {}
        self._load()

    def _find_config(self) -> str:
        candidates = [
            "quantum_simulator_config.toml",
            os.path.join(os.path.dirname(__file__), "quantum_simulator_config.toml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""

    def _load(self) -> None:
        if not self.config_path or not os.path.exists(self.config_path):
            self._load_defaults()
            return
        with open(self.config_path, "rb") as f:
            self._raw_data = tomllib.load(f)
        self._parse_atoms()
        self._parse_molecules()
        self._parse_orbitals()

    def _load_defaults(self) -> None:
        self._atoms = {"H": AtomData("H", "Hydrogen", 1, 1.007825, ["1s1"], 1.0)}
        self._molecules = {
            "H2": MoleculeData("H2", "H2", "Hydrogen molecule", ["H", "H"], 2, 2, 4, 0.735, -1.11675928, -1.13728383, 0.71997, {})
        }
        self._orbitals = {"1s": OrbitalData("1s", 1, 0, 0, "1s orbital")}

    def _parse_atoms(self) -> None:
        for a in self._raw_data.get("atoms", []):
            atom = AtomData(a.get("symbol", ""), a.get("name", ""), a.get("atomic_number", 0), a.get("mass", 0.0), a.get("electron_configuration", []), a.get("nuclear_charge", 0.0))
            self._atoms[atom.symbol] = atom

    def _parse_molecules(self) -> None:
        for m in self._raw_data.get("molecules", []):
            mol = MoleculeData(m.get("name", ""), m.get("formula", ""), m.get("description", ""), m.get("atoms", []), m.get("n_electrons", 0), m.get("n_orbitals", 0), m.get("n_qubits", 0), m.get("bond_length_angstrom", 0.0), m.get("hf_energy_hartree", 0.0), m.get("fci_energy_hartree", 0.0), m.get("nuclear_repulsion_hartree", 0.0), m.get("geometry", {}))
            self._molecules[mol.name] = mol

    def _parse_orbitals(self) -> None:
        for o in self._raw_data.get("orbitals", []):
            orb = OrbitalData(o.get("name", ""), o.get("n", 0), o.get("l", 0), o.get("m", 0), o.get("description", ""))
            self._orbitals[orb.name] = orb

    def get_atom(self, symbol: str) -> Optional[AtomData]:
        result = self._atoms.get(symbol)
        if result is None:
            for k, v in self._atoms.items():
                if k.lower() == symbol.lower():
                    return v
        return result

    def get_molecule(self, name: str) -> Optional[MoleculeData]:
        result = self._molecules.get(name)
        if result is None:
            for k, v in self._molecules.items():
                if k.lower() == name.lower():
                    return v
        return result

    def get_orbital(self, name: str) -> Optional[OrbitalData]:
        result = self._orbitals.get(name)
        if result is None:
            for k, v in self._orbitals.items():
                if k.lower() == name.lower():
                    return v
        return result

    @property
    def atoms(self) -> Dict[str, AtomData]:
        return self._atoms

    @property
    def molecules(self) -> Dict[str, MoleculeData]:
        return self._molecules

    @property
    def orbitals(self) -> Dict[str, OrbitalData]:
        return self._orbitals


class SpectralLayer(nn.Module):
    def __init__(self, channels: int, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.kernel_real = nn.Parameter(torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1)
        self.kernel_imag = nn.Parameter(torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft2(x)
        freq_h, freq_w = x_fft.shape[-2], x_fft.shape[-1]
        kr = F.interpolate(self.kernel_real.mean(dim=0).unsqueeze(0), size=(freq_h, freq_w), mode="bilinear", align_corners=False).squeeze(0)
        ki = F.interpolate(self.kernel_imag.mean(dim=0).unsqueeze(0), size=(freq_h, freq_w), mode="bilinear", align_corners=False).squeeze(0)
        return torch.fft.irfft2(torch.complex(x_fft.real * kr - x_fft.imag * ki, x_fft.real * ki + x_fft.imag * kr), s=(self.grid_size, self.grid_size))


class HamiltonianBackboneNet(nn.Module):
    def __init__(self, grid_size: int, hidden_dim: int, num_spectral_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([SpectralLayer(hidden_dim, grid_size) for _ in range(num_spectral_layers)])
        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.gelu(self.input_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        return self.output_proj(x).squeeze(1)


class SchrodingerSpectralNet(nn.Module):
    def __init__(self, grid_size: int, hidden_dim: int, expansion_dim: int, num_spectral_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.expansion_proj = nn.Conv2d(hidden_dim, expansion_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([SpectralLayer(expansion_dim, grid_size) for _ in range(num_spectral_layers)])
        self.contraction_proj = nn.Conv2d(expansion_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        x = F.gelu(self.contraction_proj(x))
        return self.output_proj(x)


class DiracSpectralNet(nn.Module):
    def __init__(self, grid_size: int, hidden_dim: int, expansion_dim: int, num_spectral_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(8, hidden_dim, kernel_size=1)
        self.expansion_proj = nn.Conv2d(hidden_dim, expansion_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([SpectralLayer(expansion_dim, grid_size) for _ in range(num_spectral_layers)])
        self.contraction_proj = nn.Conv2d(expansion_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        x = F.gelu(self.contraction_proj(x))
        return self.output_proj(x)


class GammaMatrices:
    def __init__(self, representation: str, device: str) -> None:
        self.device = device
        if representation == "dirac":
            self.gamma0 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=torch.complex64, device=device)
            self.gamma1 = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]], dtype=torch.complex64, device=device)
            self.gamma2 = torch.tensor([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]], dtype=torch.complex64, device=device)
            self.gamma3 = torch.tensor([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.complex64, device=device)
        else:
            self.gamma0 = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.complex64, device=device)
            self.gamma1 = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]], dtype=torch.complex64, device=device)
            self.gamma2 = torch.tensor([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]], dtype=torch.complex64, device=device)
            self.gamma3 = torch.tensor([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.complex64, device=device)
        self.alpha_x = self.gamma0 @ self.gamma1
        self.alpha_y = self.gamma0 @ self.gamma2
        self.alpha_z = self.gamma0 @ self.gamma3
        self.beta = self.gamma0


class JointHilbertState:
    def __init__(self, amplitudes: torch.Tensor, n_qubits: int) -> None:
        self.amplitudes = amplitudes
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.device = amplitudes.device
        self.G = amplitudes.shape[-1]

    def normalize_(self) -> None:
        probs = (self.amplitudes[:, 0] ** 2 + self.amplitudes[:, 1] ** 2).sum(dim=(-2, -1))
        total = probs.sum() + 1e-12
        self.amplitudes = self.amplitudes / (total ** 0.5)

    def probabilities(self) -> torch.Tensor:
        probs = (self.amplitudes[:, 0] ** 2 + self.amplitudes[:, 1] ** 2).sum(dim=(-2, -1))
        return probs / (probs.sum() + 1e-12)

    def entropy(self) -> float:
        probs = self.probabilities()
        probs = probs[probs > 1e-12]
        return float(-torch.sum(probs * torch.log2(probs)))

    def most_probable_bitstring(self) -> str:
        k = int(self.probabilities().argmax().item())
        return format(k, f"0{self.n_qubits}b")

    def clone(self) -> "JointHilbertState":
        return JointHilbertState(self.amplitudes.clone(), self.n_qubits)


class IPhysicsBackend(ABC):
    @abstractmethod
    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        pass

    @abstractmethod
    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        pass


class HamiltonianBackend(IPhysicsBackend):
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.device = config.device
        self.net: Optional[HamiltonianBackboneNet] = None
        self._laplacian: Optional[torch.Tensor] = None
        self._load()
        self._precompute_laplacian()

    def _load(self) -> None:
        self.net = HamiltonianBackboneNet(self.config.grid_size, self.config.hidden_dim, self.config.num_spectral_layers).to(self.device)
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
        G = self.config.grid_size
        kx = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        ky = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self._laplacian = (-(KX ** 2 + KY ** 2)).float().to(self.device)

    def _apply_h(self, field: torch.Tensor) -> torch.Tensor:
        if self.net is not None:
            with torch.no_grad():
                result = self.net(field.to(self.device))
                return result.squeeze() if result.dim() > 2 else result
        fft = torch.fft.fft2(field.to(self.device))
        return torch.fft.ifft2(fft * self._laplacian).real

    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        psi_r, psi_i = amp[0].to(self.device), amp[1].to(self.device)
        h_r, h_i = self._apply_h(psi_r), self._apply_h(psi_i)
        out = torch.stack([psi_r + dt * h_i, psi_i - dt * h_r], dim=0)
        norm = torch.sqrt((out ** 2).sum()) + self.config.normalization_eps
        return out / norm

    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        c, s = math.cos(phase_angle), math.sin(phase_angle)
        return torch.stack([c * amp[0] - s * amp[1], s * amp[0] + c * amp[1]], dim=0)


class SchrodingerBackend(IPhysicsBackend):
    def __init__(self, config: FrameworkConfig, hamiltonian: HamiltonianBackend) -> None:
        self.config = config
        self.device = config.device
        self.hamiltonian = hamiltonian
        self.net: Optional[SchrodingerSpectralNet] = None
        self._load()

    def _load(self) -> None:
        self.net = SchrodingerSpectralNet(self.config.grid_size, self.config.hidden_dim, self.config.expansion_dim, self.config.num_spectral_layers).to(self.device)
        path = self.config.schrodinger_checkpoint
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                self.net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                _LOG.info("SchrodingerBackend: loaded %s", path)
            except Exception:
                self.net = None
        if self.net is not None:
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad_(False)

    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        if self.net is None:
            return self.hamiltonian.evolve_amplitude(amp, dt)
        with torch.no_grad():
            out = self.net(amp.unsqueeze(0).to(self.device)).squeeze(0)
        norm = torch.sqrt((out ** 2).sum()) + self.config.normalization_eps
        return out / norm

    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        return self.hamiltonian.apply_phase(amp, phase_angle)


class DiracBackend(IPhysicsBackend):
    def __init__(self, config: FrameworkConfig, hamiltonian: HamiltonianBackend) -> None:
        self.config = config
        self.device = config.device
        self.hamiltonian = hamiltonian
        self.gamma = GammaMatrices("dirac", config.device)
        self.net: Optional[DiracSpectralNet] = None
        self._load()
        self._precompute_dirac()

    def _load(self) -> None:
        self.net = DiracSpectralNet(self.config.grid_size, self.config.hidden_dim, self.config.expansion_dim, self.config.num_spectral_layers).to(self.device)
        path = self.config.dirac_checkpoint
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                self.net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                _LOG.info("DiracBackend: loaded %s", path)
            except Exception:
                self.net = None
        if self.net is not None:
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad_(False)

    def _precompute_dirac(self) -> None:
        G = self.config.grid_size
        kx = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        ky = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self.kx_grid, self.ky_grid = KX.to(self.device), KY.to(self.device)

    def _pack(self, amp: torch.Tensor) -> torch.Tensor:
        psi_c = torch.complex(amp[0].to(self.device), amp[1].to(self.device))
        s = math.sqrt(0.5)
        spinor = torch.zeros(4, self.config.grid_size, self.config.grid_size, dtype=torch.complex64, device=self.device)
        spinor[0] = psi_c * s
        spinor[1] = psi_c * s
        spinor[2] = psi_c.conj() * s
        spinor[3] = psi_c.conj() * s
        return spinor

    def _unpack(self, spinor: torch.Tensor) -> torch.Tensor:
        particle = spinor[:2].mean(dim=0)
        out = torch.stack([particle.real, particle.imag], dim=0)
        norm = torch.sqrt((out ** 2).sum()) + self.config.normalization_eps
        return out / norm

    def _analytical_dirac(self, spinor: torch.Tensor) -> torch.Tensor:
        m, c = self.config.electron_mass, self.config.c_light
        result = torch.zeros_like(spinor)
        for comp in range(4):
            fft = torch.fft.fft2(spinor[comp])
            px = torch.fft.ifft2(fft * self.kx_grid)
            py = torch.fft.ifft2(fft * self.ky_grid)
            for row in range(4):
                result[row] += c * self.gamma.alpha_x[row, comp] * px + c * self.gamma.alpha_y[row, comp] * py + m * c ** 2 * self.gamma.beta[row, comp] * spinor[comp]
        return result

    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
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
        return self.hamiltonian.apply_phase(amp, phase_angle)

    def evolve_spinor(self, spinor: torch.Tensor, dt: float) -> torch.Tensor:
        h_spinor = self._analytical_dirac(spinor)
        result = spinor - 1j * dt * h_spinor
        norm = torch.sqrt((result.abs() ** 2).sum()) + self.config.normalization_eps
        return result / norm


def _single_qubit_unitary(state: JointHilbertState, qubit: int, u: torch.Tensor, backend: IPhysicsBackend) -> JointHilbertState:
    n = state.n_qubits
    bit_pos = n - 1 - qubit
    new_amps = state.amplitudes.clone()
    u00r, u00i = float(u[0, 0].real), float(u[0, 0].imag)
    u01r, u01i = float(u[0, 1].real), float(u[0, 1].imag)
    u10r, u10i = float(u[1, 0].real), float(u[1, 0].imag)
    u11r, u11i = float(u[1, 1].real), float(u[1, 1].imag)
    processed = set()
    for k0 in range(2 ** n):
        if (k0 >> bit_pos) & 1 or k0 in processed:
            continue
        k1 = k0 | (1 << bit_pos)
        processed.add(k0)
        processed.add(k1)
        a0r, a0i = state.amplitudes[k0, 0], state.amplitudes[k0, 1]
        a1r, a1i = state.amplitudes[k1, 0], state.amplitudes[k1, 1]
        new_amps[k0, 0] = u00r * a0r - u00i * a0i + u01r * a1r - u01i * a1i
        new_amps[k0, 1] = u00r * a0i + u00i * a0r + u01r * a1i + u01i * a1r
        new_amps[k1, 0] = u10r * a0r - u10i * a0i + u11r * a1r - u11i * a1i
        new_amps[k1, 1] = u10r * a0i + u10i * a0r + u11r * a1i + u11i * a1r
    return JointHilbertState(new_amps, n)


def _two_qubit_unitary(state: JointHilbertState, ctrl: int, tgt: int, u4: torch.Tensor) -> JointHilbertState:
    n = state.n_qubits
    ctrl_bit, tgt_bit = n - 1 - ctrl, n - 1 - tgt
    new_amps = state.amplitudes.clone()
    processed = set()
    for base in range(2 ** n):
        if (base >> ctrl_bit) & 1 or (base >> tgt_bit) & 1:
            continue
        k00, k01, k10, k11 = base, base | (1 << tgt_bit), base | (1 << ctrl_bit), base | (1 << ctrl_bit) | (1 << tgt_bit)
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
                ur, ui = float(u4[row, col].real), float(u4[row, col].imag)
                new_r = new_r + ur * old_r[col] - ui * old_i[col]
                new_i = new_i + ur * old_i[col] + ui * old_r[col]
            new_amps[k_out, 0], new_amps[k_out, 1] = new_r, new_i
    return JointHilbertState(new_amps, n)


class IQuantumGate(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, state: JointHilbertState, backend: IPhysicsBackend, targets: Sequence[int], params: Optional[Dict[str, float]]) -> JointHilbertState:
        pass


class HadamardGate(IQuantumGate):
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
    @property
    def name(self) -> str:
        return "X"

    def apply(self, state, backend, targets, params):
        u = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class PauliYGate(IQuantumGate):
    @property
    def name(self) -> str:
        return "Y"

    def apply(self, state, backend, targets, params):
        u = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class PauliZGate(IQuantumGate):
    @property
    def name(self) -> str:
        return "Z"

    def apply(self, state, backend, targets, params):
        u = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class SGate(IQuantumGate):
    @property
    def name(self) -> str:
        return "S"

    def apply(self, state, backend, targets, params):
        u = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)
        for t in targets:
            state = _single_qubit_unitary(state, t, u, backend)
        return state


class TGate(IQuantumGate):
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
    @property
    def name(self) -> str:
        return "CNOT"

    def apply(self, state, backend, targets, params):
        if len(targets) < 2:
            raise ValueError("CNOT requires [control, target]")
        u4 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex64)
        return _two_qubit_unitary(state, targets[0], targets[1], u4)


class CZGate(IQuantumGate):
    @property
    def name(self) -> str:
        return "CZ"

    def apply(self, state, backend, targets, params):
        if len(targets) < 2:
            raise ValueError("CZ requires [control, target]")
        u4 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=torch.complex64)
        return _two_qubit_unitary(state, targets[0], targets[1], u4)


class SWAPGate(IQuantumGate):
    @property
    def name(self) -> str:
        return "SWAP"

    def apply(self, state, backend, targets, params):
        if len(targets) < 2:
            raise ValueError("SWAP requires 2 targets")
        u4 = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.complex64)
        return _two_qubit_unitary(state, targets[0], targets[1], u4)


class ToffoliGate(IQuantumGate):
    @property
    def name(self) -> str:
        return "CCX"

    def apply(self, state, backend, targets, params):
        if len(targets) < 3:
            raise ValueError("Toffoli requires [ctrl0, ctrl1, target]")
        c0, c1, tgt = targets[0], targets[1], targets[2]
        n = state.n_qubits
        c0_bit, c1_bit, tgt_bit = n - 1 - c0, n - 1 - c1, n - 1 - tgt
        new_amps = state.amplitudes.clone()
        for k in range(2 ** n):
            if ((k >> c0_bit) & 1) and ((k >> c1_bit) & 1) and not ((k >> tgt_bit) & 1):
                k_flip = k | (1 << tgt_bit)
                new_amps[k] = state.amplitudes[k_flip].clone()
                new_amps[k_flip] = state.amplitudes[k].clone()
        new_state = JointHilbertState(new_amps, n)
        new_state.normalize_()
        return new_state


_GATE_REGISTRY: Dict[str, IQuantumGate] = {
    "H": HadamardGate(), "X": PauliXGate(), "Y": PauliYGate(), "Z": PauliZGate(),
    "S": SGate(), "T": TGate(), "Rx": RxGate(), "Ry": RyGate(), "Rz": RzGate(),
    "CNOT": CNOTGate(), "CZ": CZGate(), "SWAP": SWAPGate(), "CCX": ToffoliGate(),
}


@dataclass
class CircuitInstruction:
    gate_name: str
    targets: List[int]
    params: Dict[str, float] = field(default_factory=dict)


class QuantumCircuit:
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self._instructions: List[CircuitInstruction] = []

    def _append(self, gate_name: str, targets: List[int], params: Optional[Dict[str, float]] = None) -> None:
        self._instructions.append(CircuitInstruction(gate_name, targets, params or {}))

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

    def cnot(self, control: int, target: int) -> None:
        self._append("CNOT", [control, target])

    def cz(self, control: int, target: int) -> None:
        self._append("CZ", [control, target])

    def swap(self, qubit1: int, qubit2: int) -> None:
        self._append("SWAP", [qubit1, qubit2])

    def ccx(self, ctrl0: int, ctrl1: int, target: int) -> None:
        self._append("CCX", [ctrl0, ctrl1, target])

    def run(self, state: JointHilbertState, backend: IPhysicsBackend) -> JointHilbertState:
        for inst in self._instructions:
            gate = _GATE_REGISTRY.get(inst.gate_name)
            if gate is None:
                raise KeyError(f"Gate '{inst.gate_name}' not found")
            state = gate.apply(state, backend, inst.targets, inst.params)
        return state


class QuantumResult:
    def __init__(self, state: JointHilbertState) -> None:
        self.state = state

    def entropy(self) -> float:
        return self.state.entropy()

    def most_probable_bitstring(self) -> str:
        return self.state.most_probable_bitstring()

    def probabilities(self) -> torch.Tensor:
        return self.state.probabilities()


class PotentialGenerator:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.G = config.grid_size

    def _grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(0, 2 * math.pi, self.G)
        y = torch.linspace(0, 2 * math.pi, self.G)
        return torch.meshgrid(x, y, indexing="ij")

    def harmonic(self) -> torch.Tensor:
        X, Y = self._grid()
        cx, cy = math.pi, math.pi
        return 0.5 * self.config.potential_depth * ((X - cx) ** 2 + (Y - cy) ** 2) / (math.pi ** 2)

    def double_well(self) -> torch.Tensor:
        X, _ = self._grid()
        cx = math.pi
        w = self.config.potential_width * math.pi
        return self.config.potential_depth * ((X - cx) ** 2 / w ** 2 - 1.0) ** 2

    def coulomb(self) -> torch.Tensor:
        X, Y = self._grid()
        cx, cy = math.pi, math.pi
        r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2) + self.config.potential_width
        return -self.config.potential_depth / r

    def periodic_lattice(self) -> torch.Tensor:
        X, Y = self._grid()
        return self.config.potential_depth * (torch.cos(2.0 * X) + torch.cos(2.0 * Y))

    def mixed(self, seed: int) -> torch.Tensor:
        rng = np.random.RandomState(seed)
        weights = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
        parts = [self.harmonic(), self.double_well(), self.coulomb(), self.periodic_lattice()]
        result = torch.zeros(self.G, self.G)
        for w, v in zip(weights, parts):
            result += float(w) * v
        return result


def _solve_eigenstate(config: FrameworkConfig, potential: torch.Tensor, n: int) -> torch.Tensor:
    G = config.grid_size
    kx = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
    h_matrix = torch.diag(-0.5 * (-(kx ** 2))) + torch.diag(potential.mean(dim=1))
    try:
        _, eigenvectors = torch.linalg.eigh(h_matrix.float())
    except Exception:
        eigenvectors = torch.eye(G)
    psi_1d = eigenvectors[:, min(n, G - 1)]
    psi_2d = psi_1d.unsqueeze(1).expand(-1, G).clone()
    phase = torch.randn(G, G) * 0.1
    psi = torch.stack([psi_2d * torch.cos(phase), psi_2d * torch.sin(phase)], dim=0)
    norm = torch.sqrt((psi ** 2).sum()) + config.normalization_eps
    return psi / norm


def _build_basis_amplitude(config: FrameworkConfig, basis_idx: int) -> torch.Tensor:
    pot_gen = PotentialGenerator(config)
    popcount = bin(basis_idx).count("1")
    potential = pot_gen.mixed(seed=basis_idx * 17 + 3)
    return _solve_eigenstate(config, potential, n=popcount)


class JointStateFactory:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config

    def _empty(self, n_qubits: int) -> torch.Tensor:
        return torch.zeros(2 ** n_qubits, 2, self.config.grid_size, self.config.grid_size, device=self.config.device)

    def all_zeros(self, n_qubits: int) -> JointHilbertState:
        amps = self._empty(n_qubits)
        amps[0] = _build_basis_amplitude(self.config, 0).to(self.config.device)
        state = JointHilbertState(amps, n_qubits)
        state.normalize_()
        return state

    def basis_state(self, n_qubits: int, k: int) -> JointHilbertState:
        if k < 0 or k >= 2 ** n_qubits:
            raise ValueError(f"k={k} out of range")
        amps = self._empty(n_qubits)
        amps[k] = _build_basis_amplitude(self.config, k).to(self.config.device)
        state = JointHilbertState(amps, n_qubits)
        state.normalize_()
        return state

    def from_bitstring(self, bitstring: str) -> JointHilbertState:
        return self.basis_state(len(bitstring), int(bitstring, 2))


class QuantumComputer:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self._factory = JointStateFactory(config)
        self._hamiltonian = HamiltonianBackend(config)
        self._backends: Dict[str, IPhysicsBackend] = {
            "hamiltonian": self._hamiltonian,
            "schrodinger": SchrodingerBackend(config, self._hamiltonian),
            "dirac": DiracBackend(config, self._hamiltonian),
        }
        _LOG.info("Quantum computer initialized with backends: %s", list(self._backends.keys()))

    def create_circuit(self, n_qubits: int) -> QuantumCircuit:
        return QuantumCircuit(n_qubits)

    def run_circuit(self, circuit: QuantumCircuit, initial_state: Optional[JointHilbertState] = None, backend: str = "schrodinger") -> QuantumResult:
        if initial_state is None:
            initial_state = self._factory.all_zeros(circuit.n_qubits)
        be = self._backends.get(backend, self._backends["schrodinger"])
        final_state = circuit.run(initial_state, be)
        return QuantumResult(final_state)

    def bell_state(self, backend: str = "schrodinger") -> QuantumResult:
        circ = self.create_circuit(2)
        circ.h(0)
        circ.cnot(0, 1)
        return self.run_circuit(circ, backend=backend)

    def ghz_state(self, n_qubits: int = 3, backend: str = "schrodinger") -> QuantumResult:
        circ = self.create_circuit(n_qubits)
        circ.h(0)
        for i in range(n_qubits - 1):
            circ.cnot(i, i + 1)
        return self.run_circuit(circ, backend=backend)

    @property
    def factory(self) -> JointStateFactory:
        return self._factory

    @property
    def backends(self) -> Dict[str, IPhysicsBackend]:
        return self._backends


class WavefunctionCalculator:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config

    @staticmethod
    def radial_wavefunction(n: int, l: int, r: np.ndarray) -> np.ndarray:
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy required")
        if l >= n or l < 0:
            return np.zeros_like(r)
        norm = np.sqrt((2.0 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
        rho = 2.0 * r / n
        laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        R = norm * np.power(rho, l) * laguerre * np.exp(-rho / 2)
        return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def spherical_harmonic_real(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy required")
        Y = sph_harm(abs(m), l, phi, theta)
        if m == 0:
            return Y.real
        elif m > 0:
            return np.sqrt(2) * Y.real * ((-1) ** m)
        else:
            return np.sqrt(2) * Y.imag * ((-1) ** abs(m))

    def psi_3d(self, n: int, l: int, m: int, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        R = self.radial_wavefunction(n, l, r)
        Y = self.spherical_harmonic_real(l, m, theta, phi)
        return R * Y

    def energy_analytical(self, n: int) -> float:
        return -0.5 / (n ** 2)


class MonteCarloSampler:
    def __init__(self, config: FrameworkConfig, wavefunction_calc: WavefunctionCalculator) -> None:
        self.config = config
        self.wavefunction_calc = wavefunction_calc

    def find_max_probability(self, n: int, l: int, m: int) -> Tuple[float, float, float, float]:
        r_max = self.config.r_max_factor * n ** 2 + self.config.r_max_offset
        r_vals = np.linspace(0.01, r_max, self.config.grid_search_r)
        theta_vals = np.linspace(0.01, np.pi - 0.01, self.config.grid_search_theta)
        phi_vals = np.linspace(0, 2 * np.pi, self.config.grid_search_phi)
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
                    Y = self.wavefunction_calc.spherical_harmonic_real(l, m, np.array([theta]), np.array([phi]))[0]
                    prob = np.abs(R * Y) ** 2 * r ** 2 * sin_theta
                    if prob > max_prob:
                        max_prob = prob
                        best = (r, theta, phi)
        return max_prob, best[0], best[1], best[2]

    def sample_orbital(self, n: int, l: int, m: int, num_samples: int, Z: float = 1.0) -> Dict[str, Any]:
        num_samples = max(self.config.mc_min_particles, min(self.config.mc_max_particles, num_samples))
        P_max, _, _, _ = self.find_max_probability(n, l, m)
        if P_max < 1e-15:
            P_max = 1e-10
        r_max = self.config.r_max_factor * n ** 2 / Z + self.config.r_max_offset
        P_threshold = P_max * self.config.prob_safety_factor
        points_x, points_y, points_z = [], [], []
        points_prob, points_phase = [], []
        total_attempts = 0
        while len(points_x) < num_samples and total_attempts < num_samples * 200:
            total_attempts += self.config.mc_batch_size
            r_batch = r_max * (np.random.uniform(0, 1, self.config.mc_batch_size) ** (1 / 3))
            theta_batch = np.arccos(1 - 2 * np.random.uniform(0, 1, self.config.mc_batch_size))
            phi_batch = np.random.uniform(0, 2 * np.pi, self.config.mc_batch_size)
            R_batch = self.wavefunction_calc.radial_wavefunction(n, l, r_batch)
            Y_batch = self.wavefunction_calc.spherical_harmonic_real(l, m, theta_batch, phi_batch)
            psi_batch = R_batch * Y_batch
            prob_batch = np.abs(psi_batch) ** 2
            prob_vol_batch = prob_batch * r_batch ** 2 * np.sin(theta_batch)
            u_batch = np.random.uniform(0, P_threshold, self.config.mc_batch_size)
            accepted = u_batch < prob_vol_batch
            r_acc, theta_acc, phi_acc = r_batch[accepted], theta_batch[accepted], phi_batch[accepted]
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
        return {"x": points_x, "y": points_y, "z": points_z, "prob": points_prob, "phase": points_phase, "n": n, "l": l, "m": m, "r_max": r_max, "efficiency": efficiency}

    def sample_entangled_state(self, n1: int, l1: int, m1: int, n2: int, l2: int, m2: int, num_samples: int) -> Dict[str, Any]:
        samples_1 = self.sample_orbital(n1, l1, m1, num_samples // 2)
        samples_2 = self.sample_orbital(n2, l2, m2, num_samples // 2)
        combined_x = np.concatenate([samples_1["x"], samples_2["x"]])
        combined_y = np.concatenate([samples_1["y"], samples_2["y"]])
        combined_z = np.concatenate([samples_1["z"], samples_2["z"]])
        combined_prob = np.concatenate([samples_1["prob"], samples_2["prob"]])
        combined_phase = np.concatenate([samples_1["phase"], samples_2["phase"]])
        orbital_label = np.concatenate([np.zeros(len(samples_1["x"])), np.ones(len(samples_2["x"]))])
        idx = np.random.permutation(len(combined_x))
        return {"x": combined_x[idx], "y": combined_y[idx], "z": combined_z[idx], "prob": combined_prob[idx], "phase": combined_phase[idx], "orbital_label": orbital_label[idx], "orbital_1": {"n": n1, "l": l1, "m": m1, "samples": len(samples_1["x"])}, "orbital_2": {"n": n2, "l": l2, "m": m2, "samples": len(samples_2["x"])}, "efficiency": (samples_1["efficiency"] + samples_2["efficiency"]) / 2}


class DiracHydrogenAtom:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.c = config.c_light
        self.alpha_fs = config.alpha_fs

    def energy_level_dirac(self, n: int, kappa: int, Z: float = 1.0) -> float:
        alpha = self.alpha_fs * Z
        kappa_abs = abs(kappa)
        sqrt_term = np.sqrt(kappa_abs ** 2 - alpha ** 2)
        denominator = n - kappa_abs + sqrt_term
        E = 1.0 / np.sqrt(1.0 + (alpha / denominator) ** 2)
        return (E - 1.0) * self.c ** 2

    def energy_schrodinger(self, n: int, Z: float = 1.0) -> float:
        return -0.5 * Z ** 2 / n ** 2

    def fine_structure_splitting(self, n: int, l: int, Z: float = 1.0) -> Dict[str, float]:
        alpha = self.alpha_fs * Z
        if l == 0:
            return {"j_1_2": self.energy_level_dirac(n, -1, Z), "E_schrodinger": self.energy_schrodinger(n, Z), "fine_structure": self.energy_level_dirac(n, -1, Z) - self.energy_schrodinger(n, Z)}
        E_j_upper = self.energy_level_dirac(n, -(l + 1), Z)
        E_j_lower = self.energy_level_dirac(n, l, Z)
        return {"j_upper": l + 0.5, "j_lower": l - 0.5, "E_j_upper": E_j_upper, "E_j_lower": E_j_lower, "E_schrodinger": self.energy_schrodinger(n, Z), "splitting": E_j_upper - E_j_lower}

    def energy_spectrum(self, n_max: int = 4, Z: float = 1.0) -> List[Dict]:
        spectrum = []
        orbital_names = ["s", "p", "d", "f", "g"]
        for n in range(1, n_max + 1):
            for l in range(n):
                if l == 0:
                    E = self.energy_level_dirac(n, -1, Z)
                    spectrum.append({"n": n, "l": l, "j": 0.5, "notation": f"{n}s_{{1/2}}", "energy": E, "E_schrodinger": self.energy_schrodinger(n, Z), "degeneracy": 2})
                else:
                    E1 = self.energy_level_dirac(n, l, Z)
                    spectrum.append({"n": n, "l": l, "j": l - 0.5, "notation": f"{n}{orbital_names[l]}_{{{l-0.5}}}", "energy": E1, "E_schrodinger": self.energy_schrodinger(n, Z), "degeneracy": 2 * (l - 0.5) + 1})
                    E2 = self.energy_level_dirac(n, -(l + 1), Z)
                    spectrum.append({"n": n, "l": l, "j": l + 0.5, "notation": f"{n}{orbital_names[l]}_{{{l+0.5}}}", "energy": E2, "E_schrodinger": self.energy_schrodinger(n, Z), "degeneracy": 2 * (l + 0.5) + 1})
        return spectrum


class ZitterbewegungSimulator:
    def __init__(self, config: FrameworkConfig, dirac_backend: DiracBackend) -> None:
        self.config = config
        self.dirac = dirac_backend
        self.gamma = dirac_backend.gamma

    def create_gaussian_wave_packet(self, sigma: float = 0.1, momentum: float = 0.0) -> torch.Tensor:
        G = self.config.grid_size
        x = torch.linspace(-np.pi, np.pi, G, device=self.config.device)
        y = torch.linspace(-np.pi, np.pi, G, device=self.config.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        gaussian = torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        spinor = torch.zeros((4, G, G), dtype=torch.complex64, device=self.config.device)
        spinor[0] = gaussian
        spinor[1] = gaussian * 0.5j
        spinor[2] = gaussian * 0.01
        spinor[3] = gaussian * 0.01j
        if momentum != 0:
            phase = torch.exp(1j * momentum * X)
            for i in range(4):
                spinor[i] = spinor[i] * phase
        norm = torch.sqrt(torch.sum(torch.abs(spinor) ** 2)) + self.config.normalization_eps
        return spinor / norm

    def compute_position_expectation(self, spinor: torch.Tensor) -> Tuple[float, float]:
        G = self.config.grid_size
        x = torch.linspace(-np.pi, np.pi, G, device=self.config.device)
        y = torch.linspace(-np.pi, np.pi, G, device=self.config.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        prob = torch.zeros((G, G), device=self.config.device)
        for c in range(4):
            prob += torch.abs(spinor[c]) ** 2
        prob_sum = prob.sum() + self.config.normalization_eps
        return float((X * prob).sum() / prob_sum), float((Y * prob).sum() / prob_sum)

    def compute_velocity_expectation(self, spinor: torch.Tensor) -> Tuple[float, float]:
        vx_sum, vy_sum = 0.0, 0.0
        spinor_flat = spinor.view(4, -1)
        for c in range(4):
            for d in range(4):
                ax = self.gamma.alpha_x[c, d].item()
                ay = self.gamma.alpha_y[c, d].item()
                vx_sum += ax * float(torch.sum(torch.conj(spinor_flat[c]) * spinor_flat[d]).real)
                vy_sum += ay * float(torch.sum(torch.conj(spinor_flat[c]) * spinor_flat[d]).real)
        return self.config.c_light * vx_sum, self.config.c_light * vy_sum

    def simulate(self, duration: float = 1.0, dt: float = 0.001, sigma: float = 0.1) -> Dict:
        num_steps = int(duration / dt)
        spinor = self.create_gaussian_wave_packet(sigma=sigma)
        positions_x, positions_y, velocities_x, velocities_y, times = [], [], [], [], []
        x0, y0 = self.compute_position_expectation(spinor)
        positions_x.append(x0)
        positions_y.append(y0)
        vx, vy = self.compute_velocity_expectation(spinor)
        velocities_x.append(vx)
        velocities_y.append(vy)
        times.append(0.0)
        for step in range(num_steps):
            spinor = self.dirac.evolve_spinor(spinor, dt)
            x, y = self.compute_position_expectation(spinor)
            positions_x.append(x)
            positions_y.append(y)
            vx, vy = self.compute_velocity_expectation(spinor)
            velocities_x.append(vx)
            velocities_y.append(vy)
            times.append((step + 1) * dt)
        times = np.array(times)
        positions_x = np.array(positions_x)
        positions_y = np.array(positions_y)
        velocities_x = np.array(velocities_x)
        velocities_y = np.array(velocities_y)
        zbw_freq_expected = 2 * self.config.c_light ** 2
        if len(positions_x) > 10:
            fft_x = np.fft.fft(positions_x - positions_x.mean())
            freqs = np.fft.fftfreq(len(positions_x), dt)
            positive_freq_mask = freqs > 0
            dominant_freq = freqs[positive_freq_mask][np.argmax(np.abs(fft_x[positive_freq_mask]))] if positive_freq_mask.sum() > 0 else 0.0
        else:
            dominant_freq = 0.0
        return {"times": times, "positions_x": positions_x, "positions_y": positions_y, "velocities_x": velocities_x, "velocities_y": velocities_y, "zitterbewegung_frequency": dominant_freq, "zitterbewegung_frequency_expected": zbw_freq_expected, "zitterbewegung_amplitude": float(np.std(positions_x - np.linspace(positions_x[0], positions_x[-1], len(positions_x))))}


class OrbitalVisualizer:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config

    def visualize(self, data: Dict[str, Any], save_path: Optional[str] = None, title_suffix: str = "") -> None:
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning("matplotlib not available")
            return
        X, Y, Z = data["x"], data["y"], data["z"]
        probs, phases = data["prob"], data["phase"]
        n, l, m = data["n"], data["l"], data["m"]
        max_prob = np.max(probs) if np.max(probs) > 0 else 1.0
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
        pos_mask = phases >= 0
        colors_rgba[pos_mask] = [1.0, 0.3, 0.0, 0.5]
        colors_rgba[~pos_mask] = [0.0, 0.5, 1.0, 0.5]
        sizes = self.config.scatter_size_min + prob_norm * (self.config.scatter_size_max - self.config.scatter_size_min)
        ax1.scatter(X, Y, Z, c=colors_rgba, s=sizes, alpha=0.5, depthshade=True)
        orbital_type = ["s", "p", "d", "f", "g"][l] if l < 5 else f"l={l}"
        ax1.set_title(f"Orbital {n}{orbital_type} (n={n}, l={l}, m={m}){title_suffix}\n{len(X):,} particles", color="white", fontsize=14, fontweight="bold")
        ax1.set_xlabel("x (a0)", color="white", fontsize=12)
        ax1.set_ylabel("y (a0)", color="white", fontsize=12)
        ax1.set_zlabel("z (a0)", color="white", fontsize=12)
        ax1.tick_params(colors="white")
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        H, xe, ye = np.histogram2d(X, Y, bins=self.config.histogram_bins, weights=probs)
        ax2.imshow(H.T ** 0.3, extent=[xe[0], xe[-1], ye[0], ye[-1]], origin="lower", cmap="inferno", aspect="equal", interpolation="gaussian")
        ax2.set_title("XY Projection (top view)", color="white", fontsize=14)
        ax2.set_xlabel("x (a0)", color="white")
        ax2.set_ylabel("y (a0)", color="white")
        ax2.tick_params(colors="white")
        H_xz, xxe, zze = np.histogram2d(X, Z, bins=self.config.histogram_bins, weights=probs)
        ax3.imshow(H_xz.T ** 0.3, extent=[xxe[0], xxe[-1], zze[0], zze[-1]], origin="lower", cmap="viridis", aspect="equal", interpolation="gaussian")
        ax3.set_title("XZ Projection (side view)", color="white", fontsize=14)
        ax3.set_xlabel("x (a0)", color="white")
        ax3.set_ylabel("z (a0)", color="white")
        ax3.tick_params(colors="white")
        ax4.axis("off")
        r_vals = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        info = f"\n{'=' * 50}\nORBITAL VISUALIZER\n{'=' * 50}\n\nOrbital: n={n}, l={l}, m={m}\nType: {n}{orbital_type}{title_suffix}\n\nPARTICLES\n  Total:      {len(X):>12,}\n  Efficiency: {data['efficiency']:>12.2f}%\n\nSTATISTICS\n  r_mean: {np.mean(r_vals):>10.3f} a0\n  r_std:  {np.std(r_vals):>10.3f} a0\n  r_max:  {np.max(r_vals):>10.3f} a0\n\nCOLOR\n  Red/Orange:  Positive phase (+)\n  Blue/Cyan:   Negative phase (-)\n{'=' * 50}\n"
        ax4.text(0.05, 0.95, info, transform=ax4.transAxes, fontfamily="monospace", fontsize=11, color="white", verticalalignment="top")
        plt.tight_layout()
        if save_path:
            _LOG.info("Saving: %s", save_path)
            plt.savefig(save_path, dpi=self.config.figure_dpi, facecolor=self.config.background_color, bbox_inches="tight")
            _LOG.info("Saved: %dx%d pixels", self.config.figure_size_x * self.config.figure_dpi, self.config.figure_size_y * self.config.figure_dpi)
        plt.close(fig)


class EntangledVisualizer:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config

    def visualize(self, data: Dict[str, Any], quantum_result: QuantumResult, save_path: Optional[str] = None) -> None:
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning("matplotlib not available")
            return
        X, Y, Z = data["x"], data["y"], data["z"]
        probs, phases = data["prob"], data["phase"]
        orbital_labels = data.get("orbital_label", np.zeros(len(X)))
        max_prob = np.max(probs) if np.max(probs) > 0 else 1.0
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
        orbital_1_mask = orbital_labels == 0
        colors_rgba[orbital_1_mask & (phases >= 0)] = [1.0, 0.3, 0.0, 0.5]
        colors_rgba[orbital_1_mask & (phases < 0)] = [0.0, 0.5, 1.0, 0.5]
        colors_rgba[~orbital_1_mask & (phases >= 0)] = [0.0, 1.0, 0.3, 0.5]
        colors_rgba[~orbital_1_mask & (phases < 0)] = [1.0, 0.0, 0.5, 0.5]
        sizes = self.config.scatter_size_min + prob_norm * (self.config.scatter_size_max - self.config.scatter_size_min)
        ax1.scatter(X, Y, Z, c=colors_rgba, s=sizes, alpha=0.5, depthshade=True)
        orbital_keys = sorted([k for k in data.keys() if k.startswith("orbital_") and k != "orbital_label"])
        orbital_list = [data[k] for k in orbital_keys]
        orbital_names = []
        for orb in orbital_list:
            orb_type = ["s", "p", "d", "f", "g"][orb["l"]] if orb["l"] < 5 else f'l={orb["l"]}'
            orbital_names.append(f'{orb["n"]}{orb_type}')
        title_orbitals = " + ".join(orbital_names)
        ax1.set_title(f"Entangled State: {title_orbitals}\n{len(X):,} particles", color="white", fontsize=14, fontweight="bold")
        ax1.set_xlabel("x (a0)", color="white")
        ax1.set_ylabel("y (a0)", color="white")
        ax1.set_zlabel("z (a0)", color="white")
        ax1.tick_params(colors="white")
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        H, xe, ye = np.histogram2d(X, Y, bins=self.config.histogram_bins, weights=probs)
        ax2.imshow(H.T ** 0.3, extent=[xe[0], xe[-1], ye[0], ye[-1]], origin="lower", cmap="inferno", aspect="equal", interpolation="gaussian")
        ax2.set_title("XY Projection", color="white", fontsize=14)
        ax2.tick_params(colors="white")
        H_xz, xxe, zze = np.histogram2d(X, Z, bins=self.config.histogram_bins, weights=probs)
        ax3.imshow(H_xz.T ** 0.3, extent=[xxe[0], xxe[-1], zze[0], zze[-1]], origin="lower", cmap="viridis", aspect="equal", interpolation="gaussian")
        ax3.set_title("XZ Projection", color="white", fontsize=14)
        ax3.tick_params(colors="white")
        ax4.axis("off")
        r_vals = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        entropy = quantum_result.entropy()
        most_probable = quantum_result.most_probable_bitstring()
        info = f"\n{'=' * 50}\nENTANGLED STATE VISUALIZER\n{'=' * 50}\n\nOrbitals: {title_orbitals}\n\nQUANTUM STATE\n  Most probable: |{most_probable}>\n  Entropy:       {entropy:>12.4f} bits\n\nSTATISTICS\n  r_mean: {np.mean(r_vals):>10.3f} a0\n  r_std:  {np.std(r_vals):>10.3f} a0\n\nCOLOR\n  Orbital 1: Red/Orange (+), Blue/Cyan (-)\n  Orbital 2: Green (+), Magenta (-)\n{'=' * 50}\n"
        ax4.text(0.05, 0.95, info, transform=ax4.transAxes, fontfamily="monospace", fontsize=11, color="white", verticalalignment="top")
        plt.tight_layout()
        if save_path:
            _LOG.info("Saving: %s", save_path)
            plt.savefig(save_path, dpi=self.config.figure_dpi, facecolor=self.config.background_color, bbox_inches="tight")
        plt.close(fig)


class QuantumSimulationFramework:
    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = FrameworkConfig.from_toml(config_path) if config_path else FrameworkConfig()
        self.config_loader = ConfigLoader(config_path)
        self.quantum_computer = QuantumComputer(self.config)
        self.wavefunction_calc = WavefunctionCalculator(self.config)
        self.monte_carlo_sampler = MonteCarloSampler(self.config, self.wavefunction_calc)
        self.orbital_visualizer = OrbitalVisualizer(self.config)
        self.entangled_visualizer = EntangledVisualizer(self.config)
        self.dirac_atom = DiracHydrogenAtom(self.config)
        self.zitterbewegung = ZitterbewegungSimulator(self.config, self.quantum_computer.backends["dirac"])
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)

    def list_available_atoms(self) -> List[str]:
        return list(self.config_loader.atoms.keys())

    def list_available_molecules(self) -> List[str]:
        return list(self.config_loader.molecules.keys())

    def list_available_orbitals(self) -> List[str]:
        return list(self.config_loader.orbitals.keys())

    def get_atom(self, symbol: str) -> Optional[AtomData]:
        return self.config_loader.get_atom(symbol)

    def get_molecule(self, name: str) -> Optional[MoleculeData]:
        return self.config_loader.get_molecule(name)

    def get_orbital(self, name: str) -> Optional[OrbitalData]:
        return self.config_loader.get_orbital(name)

    def run_quantum_circuit(self, circuit: QuantumCircuit, backend: str = "schrodinger") -> QuantumResult:
        return self.quantum_computer.run_circuit(circuit, backend=backend)

    def visualize_orbital(self, orbital_name: str, num_samples: Optional[int] = None, save: bool = False, Z: float = 1.0, title_suffix: str = "") -> Dict[str, Any]:
        orbital = self.get_orbital(orbital_name)
        if orbital is None:
            raise ValueError(f"Orbital '{orbital_name}' not found")
        n_samples = num_samples or self.config.default_num_samples
        data = self.monte_carlo_sampler.sample_orbital(orbital.n, orbital.l, orbital.m, n_samples, Z)
        save_path = None
        if save:
            save_path = os.path.join(self.config.output_dir, f"orbital_{orbital_name}_{n_samples}.png")
        self.orbital_visualizer.visualize(data, save_path, title_suffix)
        return data

    def visualize_atom_orbitals(self, atom_symbol: str, num_samples: Optional[int] = None, save: bool = False) -> List[Dict[str, Any]]:
        atom = self.get_atom(atom_symbol)
        if atom is None:
            raise ValueError(f"Atom '{atom_symbol}' not found")
        results = []
        electron_config = atom.electron_configuration
        print(f"\n  Visualizing orbitals for {atom.name} (Z={atom.atomic_number}):")
        print(f"  Electron configuration: {', '.join(electron_config)}")
        for config in electron_config:
            n = int(config[0])
            l_char = config[1] if len(config) > 1 else "s"
            l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
            l = l_map.get(l_char, 0)
            for m in range(-l, l + 1):
                orbital_name = f"{n}{l_char}" + ("" if l == 0 else ["", "_z", "_x", "_y", "_xy", "_x2-y2"][m + l] if l <= 2 else f"_m{m}")
                print(f"\n  Sampling orbital: n={n}, l={l}, m={m}")
                data = self.monte_carlo_sampler.sample_orbital(n, l, m, num_samples or self.config.default_num_samples, atom.nuclear_charge)
                save_path = None
                if save:
                    save_path = os.path.join(self.config.output_dir, f"atom_{atom_symbol}_orbital_{n}{l_char}_m{m}.png")
                self.orbital_visualizer.visualize(data, save_path, f" ({atom.name})")
                results.append(data)
        return results

    def visualize_entangled_state(self, orbital1: str, orbital2: str, num_samples: Optional[int] = None, save: bool = False) -> Dict[str, Any]:
        orb1 = self.get_orbital(orbital1)
        orb2 = self.get_orbital(orbital2)
        if orb1 is None or orb2 is None:
            raise ValueError(f"Orbital not found: {orbital1 if orb1 is None else orbital2}")
        n_samples = num_samples or self.config.default_num_samples
        quantum_result = self.quantum_computer.bell_state()
        data = self.monte_carlo_sampler.sample_entangled_state(orb1.n, orb1.l, orb1.m, orb2.n, orb2.l, orb2.m, n_samples)
        save_path = None
        if save:
            save_path = os.path.join(self.config.output_dir, f"entangled_{orbital1}_{orbital2}_{n_samples}.png")
        self.entangled_visualizer.visualize(data, quantum_result, save_path)
        return data

    def compute_relativistic_energy(self, n: int, l: int, Z: float = 1.0) -> Dict[str, float]:
        return self.dirac_atom.fine_structure_splitting(n, l, Z)

    def compute_energy_spectrum(self, n_max: int = 4, Z: float = 1.0) -> List[Dict]:
        return self.dirac_atom.energy_spectrum(n_max, Z)

    def run_zitterbewegung_simulation(self, duration: float = 1.0, dt: float = 0.001, sigma: float = 0.1) -> Dict:
        return self.zitterbewegung.simulate(duration, dt, sigma)

    def run_all_demonstrations(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        results = {}
        n_samples = num_samples or self.config.default_num_samples
        _LOG.info("=" * 70)
        _LOG.info("QUANTUM SIMULATION FRAMEWORK - ALL DEMONSTRATIONS")
        _LOG.info("=" * 70)
        _LOG.info("\n[1] Bell State")
        result = self.quantum_computer.bell_state()
        results["bell"] = {"bitstring": result.most_probable_bitstring(), "entropy": result.entropy()}
        _LOG.info("  Most probable: |%s>, Entropy: %.4f", result.most_probable_bitstring(), result.entropy())
        _LOG.info("\n[2] GHZ State (3 qubits)")
        result = self.quantum_computer.ghz_state(3)
        results["ghz"] = {"bitstring": result.most_probable_bitstring(), "entropy": result.entropy()}
        _LOG.info("  Most probable: |%s>, Entropy: %.4f", result.most_probable_bitstring(), result.entropy())
        _LOG.info("\n[3] Relativistic Energy Spectrum")
        spectrum = self.compute_energy_spectrum(3)
        results["spectrum"] = spectrum
        for level in spectrum[:6]:
            _LOG.info("  %s: E=%.10f (Schrod: %.10f, FS: %.2e)", level["notation"], level["energy"], level["E_schrodinger"], level["energy"] - level["E_schrodinger"])
        _LOG.info("\n[4] Zitterbewegung Simulation")
        zbw = self.run_zitterbewegung_simulation(duration=0.5, dt=0.001)
        results["zitterbewegung"] = zbw
        _LOG.info("  ZBW frequency: %.2f (expected: %.2f)", zbw["zitterbewegung_frequency"], zbw["zitterbewegung_frequency_expected"])
        _LOG.info("  ZBW amplitude: %.6f", zbw["zitterbewegung_amplitude"])
        _LOG.info("\n" + "=" * 70)
        _LOG.info("DEMONSTRATIONS COMPLETE")
        _LOG.info("=" * 70)
        return results


class InteractiveMenu:
    def __init__(self, framework: QuantumSimulationFramework) -> None:
        self.framework = framework
        self.running = True

    def display_header(self) -> None:
        print("\n" + "=" * 70)
        print("          QUANTUM SIMULATION FRAMEWORK v2.0.0")
        print("=" * 70)
        print("  Complete quantum simulation platform")
        print("  Orbitals | Molecules | Quantum Circuits | Relativistic | ZBW")
        print("=" * 70)

    def display_main_menu(self) -> None:
        print("\n" + "-" * 70)
        print("  MAIN MENU")
        print("-" * 70)
        print("  [1] Orbital Visualization")
        print("  [2] Atom Orbital Visualization (multi-electron)")
        print("  [3] Entangled State Visualization")
        print("  [4] Quantum Circuit Simulation")
        print("  [5] Relativistic Hydrogen Calculations")
        print("  [6] Zitterbewegung Simulation")
        print("  [7] Molecular Information")
        print("  [8] Atomic Information")
        print("  [9] Run All Demonstrations")
        print("  [0] Exit")
        print("-" * 70)

    def get_user_choice(self, prompt: str = "  Select option: ") -> str:
        try:
            return input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "0"

    def _display_quantum_result(self, result, n_qubits: int) -> None:
        probs = result.probabilities()
        print(f"  Entropy: {result.entropy():.4f} bits")
        print(f"  Probability distribution:")
        for i, p in enumerate(probs):
            if p > 0.001:
                print(f"    |{i:0{n_qubits}b}>: {p:.4f}")

    def orbital_menu(self) -> None:
        orbitals = self.framework.list_available_orbitals()
        print("\n" + "-" * 70)
        print("  ORBITAL VISUALIZATION")
        print("-" * 70)
        print(f"  Available: {', '.join(orbitals[:10])}... ({len(orbitals)} total)")
        print("-" * 70)
        choice = self.get_user_choice("  Enter orbital name (e.g., 1s, 2p_z) or 'back': ")
        if choice in ("back", "0"):
            return
        orbital = self.framework.get_orbital(choice)
        if orbital is None:
            print(f"  Error: Orbital '{choice}' not found")
            return
        samples_input = self.get_user_choice("  Number of samples [default=100000]: ")
        num_samples = int(samples_input) if samples_input.isdigit() else 100000
        z_input = self.get_user_choice("  Nuclear charge Z [default=1]: ")
        Z = float(z_input) if z_input else 1.0
        save_input = self.get_user_choice("  Save visualization? [y/N]: ")
        save = save_input.lower() == "y"
        print(f"\n  Simulating: n={orbital.n}, l={orbital.l}, m={orbital.m}, Z={Z}")
        self.framework.visualize_orbital(orbital.name, num_samples, save, Z)
        print("  Done")

    def atom_orbital_menu(self) -> None:
        atoms = self.framework.list_available_atoms()
        print("\n" + "-" * 70)
        print("  ATOM ORBITAL VISUALIZATION")
        print("-" * 70)
        print(f"  Available atoms: {', '.join(atoms)}")
        print("-" * 70)
        choice = self.get_user_choice("  Enter atom symbol or 'back': ")
        if choice in ("back", "0"):
            return
        atom = self.framework.get_atom(choice)
        if atom is None:
            print(f"  Error: Atom '{choice}' not found")
            return
        samples_input = self.get_user_choice("  Number of samples [default=50000]: ")
        num_samples = int(samples_input) if samples_input.isdigit() else 50000
        save_input = self.get_user_choice("  Save visualizations? [y/N]: ")
        save = save_input.lower() == "y"
        self.framework.visualize_atom_orbitals(atom.symbol, num_samples, save)

    def entangled_menu(self) -> None:
        orbitals = self.framework.list_available_orbitals()
        print("\n" + "-" * 70)
        print("  ENTANGLED STATE VISUALIZATION")
        print("-" * 70)
        print(f"  Available: {', '.join(orbitals[:5])}...")
        print("-" * 70)
        choice1 = self.get_user_choice("  First orbital (e.g., 1s): ")
        if choice1 in ("back", "0"):
            return
        orb1 = self.framework.get_orbital(choice1)
        if orb1 is None:
            print(f"  Error: '{choice1}' not found")
            return
        choice2 = self.get_user_choice("  Second orbital (e.g., 2p_z): ")
        if choice2 in ("back", "0"):
            return
        orb2 = self.framework.get_orbital(choice2)
        if orb2 is None:
            print(f"  Error: '{choice2}' not found")
            return
        samples_input = self.get_user_choice("  Number of samples [default=100000]: ")
        num_samples = int(samples_input) if samples_input.isdigit() else 100000
        save_input = self.get_user_choice("  Save? [y/N]: ")
        save = save_input.lower() == "y"
        print(f"\n  Simulating: {orb1.name} + {orb2.name}")
        self.framework.visualize_entangled_state(orb1.name, orb2.name, num_samples, save)
        print("  Done")

    def quantum_circuit_menu(self) -> None:
        print("\n" + "-" * 70)
        print("  QUANTUM CIRCUIT SIMULATION")
        print("-" * 70)
        print("  [1] Bell State (2 qubits)")
        print("  [2] GHZ State (N qubits)")
        print("  [3] Custom Circuit")
        print("  [0] Back")
        print("-" * 70)
        choice = self.get_user_choice("  Select: ")
        if choice == "1":
            print("\n  Preparing Bell state...")
            result = self.framework.quantum_computer.bell_state()
            print(f"  Most probable: |{result.most_probable_bitstring()}>")
            self._display_quantum_result(result, 2)
        elif choice == "2":
            n_input = self.get_user_choice("  Number of qubits [default=3]: ")
            n_qubits = int(n_input) if n_input.isdigit() else 3
            print(f"\n  Preparing GHZ state ({n_qubits} qubits)...")
            result = self.framework.quantum_computer.ghz_state(n_qubits)
            print(f"  Most probable: |{result.most_probable_bitstring()}>")
            self._display_quantum_result(result, n_qubits)
        elif choice == "3":
            print("\n  Custom circuit mode")
            n_input = self.get_user_choice("  Number of qubits: ")
            if not n_input.isdigit():
                return
            n_qubits = int(n_input)
            circuit = self.framework.quantum_computer.create_circuit(n_qubits)
            print("  Gates: h 0 | x 0 | cnot 0 1 | ry 0 0.5 | ccx 0 1 2 | 'run' to execute")
            while True:
                cmd = self.get_user_choice("  > ")
                if cmd == "run":
                    break
                if cmd == "back":
                    return
                parts = cmd.split()
                if len(parts) < 2:
                    continue
                gate = parts[0].lower()
                targets = []
                param = None
                for p in parts[1:]:
                    try:
                        if "." in p:
                            param = float(p)
                        else:
                            targets.append(int(p))
                    except ValueError:
                        continue
                try:
                    if gate == "h" and targets:
                        circuit.h(targets[0])
                    elif gate == "x" and targets:
                        circuit.x(targets[0])
                    elif gate == "y" and targets:
                        circuit.y(targets[0])
                    elif gate == "z" and targets:
                        circuit.z(targets[0])
                    elif gate == "s" and targets:
                        circuit.s(targets[0])
                    elif gate == "t" and targets:
                        circuit.t(targets[0])
                    elif gate == "rx" and targets and param is not None:
                        circuit.rx(targets[0], param)
                    elif gate == "ry" and targets and param is not None:
                        circuit.ry(targets[0], param)
                    elif gate == "rz" and targets and param is not None:
                        circuit.rz(targets[0], param)
                    elif gate == "cnot" and len(targets) >= 2:
                        circuit.cnot(targets[0], targets[1])
                    elif gate == "cz" and len(targets) >= 2:
                        circuit.cz(targets[0], targets[1])
                    elif gate == "swap" and len(targets) >= 2:
                        circuit.swap(targets[0], targets[1])
                    elif gate == "ccx" and len(targets) >= 3:
                        circuit.ccx(targets[0], targets[1], targets[2])
                    else:
                        print(f"  Invalid: {gate} {targets} {param}")
                except Exception as e:
                    print(f"  Error: {e}")
            print("\n  Running circuit...")
            result = self.framework.run_quantum_circuit(circuit)
            print(f"  Most probable: |{result.most_probable_bitstring()}>")
            self._display_quantum_result(result, n_qubits)

    def relativistic_menu(self) -> None:
        print("\n" + "-" * 70)
        print("  RELATIVISTIC HYDROGEN CALCULATIONS")
        print("-" * 70)
        print("  [1] Fine Structure Splitting")
        print("  [2] Energy Spectrum")
        print("  [3] Compare Dirac vs Schrodinger")
        print("  [0] Back")
        print("-" * 70)
        choice = self.get_user_choice("  Select: ")
        if choice == "1":
            n_input = self.get_user_choice("  Principal n: ")
            l_input = self.get_user_choice("  Angular l: ")
            z_input = self.get_user_choice("  Nuclear charge Z [default=1]: ")
            if not n_input.isdigit() or not l_input.isdigit():
                return
            n, l, Z = int(n_input), int(l_input), float(z_input) if z_input else 1.0
            result = self.framework.compute_relativistic_energy(n, l, Z)
            print(f"\n  Fine structure for n={n}, l={l}, Z={Z}:")
            for k, v in result.items():
                print(f"    {k}: {v:.10f}")
        elif choice == "2":
            n_max_input = self.get_user_choice("  Maximum n [default=4]: ")
            z_input = self.get_user_choice("  Nuclear charge Z [default=1]: ")
            n_max = int(n_max_input) if n_max_input.isdigit() else 4
            Z = float(z_input) if z_input else 1.0
            spectrum = self.framework.compute_energy_spectrum(n_max, Z)
            print(f"\n  Energy spectrum (n<={n_max}, Z={Z}):")
            for level in spectrum:
                print(f"    {level['notation']}: E_Dirac={level['energy']:.10f} E_Schrod={level['E_schrodinger']:.10f} FS={(level['energy']-level['E_schrodinger']):.2e}")
        elif choice == "3":
            n_input = self.get_user_choice("  Principal n: ")
            if not n_input.isdigit():
                return
            n = int(n_input)
            e_schrod = -0.5 / n ** 2
            e_dirac = self.framework.dirac_atom.energy_level_dirac(n, -1)
            print(f"\n  n={n}: Schrodinger={e_schrod:.10f}, Dirac={e_dirac:.10f}")
            print(f"  Fine structure shift: {e_dirac - e_schrod:.2e}")

    def zitterbewegung_menu(self) -> None:
        print("\n" + "-" * 70)
        print("  ZITTERBEWEGUNG SIMULATION")
        print("-" * 70)
        print("  Simulates the trembling motion of relativistic electrons")
        print("-" * 70)
        dur_input = self.get_user_choice("  Duration [default=1.0]: ")
        duration = float(dur_input) if dur_input else 1.0
        dt_input = self.get_user_choice("  Time step [default=0.001]: ")
        dt = float(dt_input) if dt_input else 0.001
        sigma_input = self.get_user_choice("  Packet width sigma [default=0.1]: ")
        sigma = float(sigma_input) if sigma_input else 0.1
        print(f"\n  Running ZBW simulation (duration={duration}, dt={dt}, sigma={sigma})...")
        result = self.framework.run_zitterbewegung_simulation(duration, dt, sigma)
        print(f"\n  Results:")
        print(f"    ZBW frequency: {result['zitterbewegung_frequency']:.2f}")
        print(f"    Expected freq: {result['zitterbewegung_frequency_expected']:.2f}")
        print(f"    ZBW amplitude: {result['zitterbewegung_amplitude']:.6f}")

    def molecular_menu(self) -> None:
        molecules = self.framework.list_available_molecules()
        print("\n" + "-" * 70)
        print("  MOLECULAR INFORMATION")
        print("-" * 70)
        print(f"  Available: {', '.join(molecules)}")
        print("-" * 70)
        choice = self.get_user_choice("  Enter molecule name or 'back': ")
        if choice in ("back", "0"):
            return
        mol = self.framework.get_molecule(choice)
        if mol is None:
            print(f"  Error: '{choice}' not found")
            return
        print(f"\n  Molecule: {mol.name}")
        print(f"  Formula: {mol.formula}")
        print(f"  Description: {mol.description}")
        print(f"  Atoms: {', '.join(mol.atoms)}")
        print(f"  Electrons: {mol.n_electrons}")
        print(f"  Qubits: {mol.n_qubits}")
        print(f"  Bond length: {mol.bond_length_angstrom} A")
        print(f"  HF energy: {mol.hf_energy_hartree:.8f} Ha")
        print(f"  FCI energy: {mol.fci_energy_hartree:.8f} Ha")

    def atomic_menu(self) -> None:
        atoms = self.framework.list_available_atoms()
        print("\n" + "-" * 70)
        print("  ATOMIC INFORMATION")
        print("-" * 70)
        print(f"  Available: {', '.join(atoms)}")
        print("-" * 70)
        choice = self.get_user_choice("  Enter atom symbol or 'back': ")
        if choice in ("back", "0"):
            return
        atom = self.framework.get_atom(choice)
        if atom is None:
            print(f"  Error: '{choice}' not found")
            return
        print(f"\n  Atom: {atom.name} ({atom.symbol})")
        print(f"  Atomic number: {atom.atomic_number}")
        print(f"  Mass: {atom.mass:.6f} amu")
        print(f"  Nuclear charge: {atom.nuclear_charge}")
        print(f"  Electron config: {', '.join(atom.electron_configuration)}")

    def run(self) -> None:
        self.display_header()
        while self.running:
            self.display_main_menu()
            choice = self.get_user_choice()
            if choice == "0":
                self.running = False
                print("\n  Exiting...")
            elif choice == "1":
                self.orbital_menu()
            elif choice == "2":
                self.atom_orbital_menu()
            elif choice == "3":
                self.entangled_menu()
            elif choice == "4":
                self.quantum_circuit_menu()
            elif choice == "5":
                self.relativistic_menu()
            elif choice == "6":
                self.zitterbewegung_menu()
            elif choice == "7":
                self.molecular_menu()
            elif choice == "8":
                self.atomic_menu()
            elif choice == "9":
                self.framework.run_all_demonstrations()


def main() -> None:
    config_path = None
    for path in ["quantum_simulator_config.toml", os.path.join(os.path.dirname(__file__), "quantum_simulator_config.toml")]:
        if os.path.exists(path):
            config_path = path
            break
    framework = QuantumSimulationFramework(config_path)
    menu = InteractiveMenu(framework)
    menu.run()


if __name__ == "__main__":
    main()
