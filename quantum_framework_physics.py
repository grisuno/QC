#!/usr/bin/env python3
"""
Quantum Framework Physics Module
================================
Neural network backends for quantum physics simulations.
Provides spectral layers, Hamiltonian networks, and Dirac operators.

Author: Gris Iscomeback
License: AGPL v3
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class SpectralLayer(nn.Module):
    """
    Spectral convolution in frequency domain.
    Learns complex kernels that modulate Fourier coefficients.
    """

    def __init__(self, channels: int, grid_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.kernel_real = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )
        self.kernel_imag = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft2(x)
        freq_h, freq_w = x_fft.shape[-2], x_fft.shape[-1]
        kr = F.interpolate(
            self.kernel_real.mean(dim=0).unsqueeze(0),
            size=(freq_h, freq_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        ki = F.interpolate(
            self.kernel_imag.mean(dim=0).unsqueeze(0),
            size=(freq_h, freq_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        real_part = x_fft.real * kr - x_fft.imag * ki
        imag_part = x_fft.real * ki + x_fft.imag * kr
        return torch.fft.irfft2(
            torch.complex(real_part, imag_part),
            s=(self.grid_size, self.grid_size),
        )


class HamiltonianBackboneNet(nn.Module):
    """
    Hamiltonian backbone: single-channel field -> H|psi>.
    Shared by all physics backends as the H operator.
    """

    def __init__(self, grid_size: int, hidden_dim: int, num_spectral_layers: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralLayer(hidden_dim, grid_size) for _ in range(num_spectral_layers)]
        )
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
    """
    Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction.
    Uses spectral convolution for physics-informed evolution.
    """

    def __init__(self, grid_size: int, hidden_dim: int, expansion_dim: int, num_spectral_layers: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.input_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.expansion_proj = nn.Conv2d(hidden_dim, expansion_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralLayer(expansion_dim, grid_size) for _ in range(num_spectral_layers)]
        )
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
    """
    Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor.
    Handles 4-component spinor evolution for relativistic quantum mechanics.
    """

    def __init__(self, grid_size: int, hidden_dim: int, expansion_dim: int, num_spectral_layers: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.input_channels = 8
        self.output_channels = 8
        self.input_proj = nn.Conv2d(8, hidden_dim, kernel_size=1)
        self.expansion_proj = nn.Conv2d(hidden_dim, expansion_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralLayer(expansion_dim, grid_size) for _ in range(num_spectral_layers)]
        )
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
    """
    Dirac gamma matrices in Dirac (standard) or Weyl representation.
    gamma^0 = beta, gamma^i = beta * alpha_i
    """

    def __init__(self, representation: str = "dirac", device: str = "cpu") -> None:
        self.representation = representation
        self.device = device
        self._init_matrices()

    def _init_matrices(self) -> None:
        if self.representation == "dirac":
            self.gamma0 = torch.tensor([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1]
            ], dtype=torch.complex64, device=self.device)
            self.gamma1 = torch.tensor([
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [-1, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma2 = torch.tensor([
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
                [0, 1j, 0, 0],
                [-1j, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma3 = torch.tensor([
                [0, 0, 1, 0],
                [0, 0, 0, -1],
                [-1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=torch.complex64, device=self.device)
        elif self.representation == "weyl":
            self.gamma0 = torch.tensor([
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma1 = torch.tensor([
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [-1, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma2 = torch.tensor([
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
                [0, 1j, 0, 0],
                [-1j, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            self.gamma3 = torch.tensor([
                [0, 0, 1, 0],
                [0, 0, 0, -1],
                [-1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=torch.complex64, device=self.device)
        else:
            raise ValueError(f"Unknown gamma representation: {self.representation}")
        self.gamma5 = torch.tensor([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=torch.complex64, device=self.device)
        self.alpha_x = self.gamma0 @ self.gamma1
        self.alpha_y = self.gamma0 @ self.gamma2
        self.alpha_z = self.gamma0 @ self.gamma3
        self.beta = self.gamma0
        self.sigma_x = -1j * torch.tensor([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64, device=self.device)
        self.sigma_y = -1j * torch.tensor([
            [0, -1j, 0, 0],
            [1j, 0, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ], dtype=torch.complex64, device=self.device)
        self.sigma_z = -1j * torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=torch.complex64, device=self.device)
        self.gammas = [self.gamma0, self.gamma1, self.gamma2, self.gamma3]

    def to(self, device: str) -> "GammaMatrices":
        self.device = device
        self._init_matrices()
        return self


class PotentialGenerator:
    """Spatial potentials for eigenstate initialization."""

    def __init__(self, grid_size: int = 16, potential_depth: float = 5.0, potential_width: float = 0.3) -> None:
        self.grid_size = grid_size
        self.potential_depth = potential_depth
        self.potential_width = potential_width

    def _grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(0, 2 * math.pi, self.grid_size)
        y = torch.linspace(0, 2 * math.pi, self.grid_size)
        return torch.meshgrid(x, y, indexing="ij")

    def harmonic(self) -> torch.Tensor:
        X, Y = self._grid()
        cx, cy = math.pi, math.pi
        return 0.5 * self.potential_depth * ((X - cx) ** 2 + (Y - cy) ** 2) / (math.pi ** 2)

    def double_well(self) -> torch.Tensor:
        X, _ = self._grid()
        cx = math.pi
        w = self.potential_width * math.pi
        return self.potential_depth * ((X - cx) ** 2 / w ** 2 - 1.0) ** 2

    def coulomb(self) -> torch.Tensor:
        X, Y = self._grid()
        cx, cy = math.pi, math.pi
        r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2) + self.potential_width
        return -self.potential_depth / r

    def periodic_lattice(self) -> torch.Tensor:
        X, Y = self._grid()
        return self.potential_depth * (torch.cos(2.0 * X) + torch.cos(2.0 * Y))

    def mixed(self, seed: int) -> torch.Tensor:
        rng = np.random.RandomState(seed)
        weights = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
        parts = [self.harmonic(), self.double_well(), self.coulomb(), self.periodic_lattice()]
        result = torch.zeros(self.grid_size, self.grid_size)
        for w, v in zip(weights, parts):
            result += float(w) * v
        return result


class DiracHamiltonianOperator:
    """
    Dirac Hamiltonian operator for relativistic quantum mechanics.
    H_Dirac = c * alpha . p + beta * m * c^2 + V(r)

    In atomic units (c = 1/alpha ~ 137):
    H = c * alpha . p + beta * m * c^2 + V
    """

    def __init__(self, grid_size: int, electron_mass: float, c_light: float, device: str = "cpu") -> None:
        self.grid_size = grid_size
        self.mass = electron_mass
        self.c = c_light
        self.gamma = GammaMatrices("dirac", device)
        self.device = device
        self._precompute_operators()

    def _precompute_operators(self) -> None:
        kx = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        ky = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self.kx_grid = KX.to(self.device)
        self.ky_grid = KY.to(self.device)

    def apply_dirac_hamiltonian(self, spinor: torch.Tensor, potential: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Dirac Hamiltonian to 4-component spinor.

        Args:
            spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor
            potential: Optional scalar potential V(r)

        Returns:
            H * psi with same shape as input
        """
        if spinor.dim() == 3:
            spinor = spinor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size = spinor.shape[0]
        result = torch.zeros_like(spinor, dtype=torch.complex128)
        for c in range(4):
            psi_c = spinor[:, c, :, :]
            psi_c_fft = torch.fft.fft2(psi_c)
            px_psi_fft = self.kx_grid * psi_c_fft
            py_psi_fft = self.ky_grid * psi_c_fft
            px_psi = torch.fft.ifft2(px_psi_fft)
            py_psi = torch.fft.ifft2(py_psi_fft)
            for d in range(4):
                alpha_x_cd = self.gamma.alpha_x[c, d].item()
                alpha_y_cd = self.gamma.alpha_y[c, d].item()
                result[:, c, :, :] += self.c * (
                    alpha_x_cd * px_psi +
                    alpha_y_cd * py_psi
                )
        mass_term = self.mass * self.c ** 2
        for c in range(4):
            for d in range(4):
                beta_cd = self.gamma.beta[c, d].item()
                result[:, c, :, :] += beta_cd * mass_term * spinor[:, d, :, :]
        if potential is not None:
            for c in range(4):
                result[:, c, :, :] += potential * spinor[:, c, :, :]
        if squeeze_output:
            result = result.squeeze(0)
        return result

    def time_evolution(self, spinor: torch.Tensor, dt: float, potential: Optional[torch.Tensor] = None, normalization_eps: float = 1e-10) -> torch.Tensor:
        """
        Time evolution of Dirac spinor using first-order split-step.
        psi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi
        """
        squeeze_output = False
        if spinor.dim() == 3:
            spinor = spinor.unsqueeze(0)
            squeeze_output = True
        H_psi = self.apply_dirac_hamiltonian(spinor, potential)
        result = spinor - 1j * dt * H_psi
        norm_original = torch.norm(spinor.view(spinor.shape[0], -1), dim=1, keepdim=True)
        norm_evolved = torch.norm(result.view(result.shape[0], -1), dim=1, keepdim=True)
        norm_evolved = norm_evolved + normalization_eps
        result = result * (norm_original / norm_evolved).unsqueeze(-1).unsqueeze(-1)
        if squeeze_output:
            result = result.squeeze(0)
        return result


class LambShiftCalculator:
    """
    Calculates the Lamb shift using Bethe's formula.
    The Lamb shift is the small difference between 2s_{1/2} and 2p_{1/2} energy levels
    due to QED effects (vacuum fluctuations and self-energy).
    """

    def __init__(self, alpha_fs: float = 0.0072973525693, c_light: float = 137.035999084, electron_mass: float = 1.0) -> None:
        self.alpha = alpha_fs
        self.c = c_light
        self.mass = electron_mass

    def bethe_formula(self, n: int, l: int, Z: int = 1) -> float:
        """
        Bethe's non-relativistic formula for Lamb shift.
        Delta E_Lamb = (8*alpha^3 / 3*pi*n^3) * |psi_n(0)|^2 * ln(E_avg / E_n)
        """
        if l != 0:
            return self._higher_l_shift(n, l, Z)
        alpha = self.alpha
        bethe_log = 2.984 - 0.173 * (n - 1)
        psi_sq = Z ** 3 / (math.pi * n ** 3)
        delta_E = (8 * alpha ** 3 / (3 * math.pi * n ** 3)) * psi_sq * bethe_log
        return delta_E

    def _higher_l_shift(self, n: int, l: int, Z: int) -> float:
        alpha = self.alpha
        delta_E = alpha ** 3 * Z ** 4 / (n ** 3 * l * (l + 1)) * 0.001
        return delta_E

    def full_lamb_shift(self, n: int, l: int, j: float, Z: int = 1) -> Dict[str, float]:
        """
        Calculate full Lamb shift including radiative corrections.
        Delta E = Delta E_SE + Delta E_Uehling + Delta E_rel
        """
        alpha = self.alpha
        m = self.mass
        c = self.c
        E_SE = self.bethe_formula(n, l, Z)
        if l == 0:
            E_Uehling = (4 * alpha / (15 * math.pi)) * (Z * alpha) ** 4 * m * c ** 2 / n ** 3
        else:
            E_Uehling = 0.0
        E_total = E_SE + E_Uehling
        E_MHz = E_total * 6.57968e9
        return {
            "n": n,
            "l": l,
            "j": j,
            "Z": Z,
            "self_energy_au": E_SE,
            "vacuum_polarization_au": E_Uehling,
            "total_au": E_total,
            "total_MHz": E_MHz,
        }


class AnomalousMagneticMoment:
    """
    Calculates the electron's anomalous magnetic moment (g-2).
    The electron g-factor is slightly different from 2 due to QED effects:
    g = 2(1 + a_e) where a_e = alpha/(2*pi) + higher-order terms
    """

    def __init__(self, alpha_fs: float = 0.0072973525693) -> None:
        self.alpha = alpha_fs

    def schwinger_term(self) -> float:
        return self.alpha / (2 * math.pi)

    def second_order(self) -> float:
        C2 = 0.32847896557919378
        return (self.alpha / math.pi) ** 2 * C2

    def third_order(self) -> float:
        C3 = 1.181241456587
        return (self.alpha / math.pi) ** 3 * C3

    def fourth_order(self) -> float:
        C4 = -1.9144
        return (self.alpha / math.pi) ** 4 * C4

    def fifth_order(self) -> float:
        C5 = 7.7
        return (self.alpha / math.pi) ** 5 * C5

    def calculate_a_e(self, order: int = 5) -> Dict[str, float]:
        contributions = {
            "order_1": self.schwinger_term(),
        }
        if order >= 2:
            contributions["order_2"] = self.second_order()
        if order >= 3:
            contributions["order_3"] = self.third_order()
        if order >= 4:
            contributions["order_4"] = self.fourth_order()
        if order >= 5:
            contributions["order_5"] = self.fifth_order()
        total = sum(contributions.values())
        contributions["total"] = total
        experimental = 0.00115965218128
        contributions["experimental"] = experimental
        contributions["error"] = abs(total - experimental)
        contributions["relative_error"] = contributions["error"] / experimental
        return contributions


class DiracHydrogenAtom:
    """
    Relativistic hydrogen atom with Dirac equation.
    Computes energy levels including fine structure.
    """

    def __init__(self, c_light: float = 137.035999084, alpha_fs: float = 0.0072973525693) -> None:
        self.c = c_light
        self.alpha_fs = alpha_fs

    def energy_level_dirac(self, n: int, kappa: int) -> float:
        """
        Exact Dirac energy level for hydrogen-like atom.
        E = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)
        """
        alpha = self.alpha_fs
        kappa_abs = abs(kappa)
        sqrt_term = math.sqrt(kappa_abs ** 2 - alpha ** 2)
        denominator = n - kappa_abs + sqrt_term
        E = 1.0 / math.sqrt(1.0 + (alpha / denominator) ** 2)
        E_binding = (E - 1.0) * self.c ** 2
        return E_binding

    def fine_structure_splitting(self, n: int, l: int) -> Dict[str, float]:
        """
        Calculate fine structure splitting for given n, l.
        Returns energies for j = l+1/2 and j = l-1/2
        """
        alpha = self.alpha_fs
        if l == 0:
            E_fs = -alpha ** 2 * self.c ** 2 / (2 * n ** 4) * (4 * n / (l + 0.5) - 3)
            return {
                "j_1_2": self.energy_level_dirac(n, -1),
                "E_fine_structure": E_fs
            }
        E_j_upper = self.energy_level_dirac(n, -(l + 1))
        E_j_lower = self.energy_level_dirac(n, l)
        splitting = E_j_upper - E_j_lower
        return {
            "j_upper": l + 0.5,
            "j_lower": l - 0.5,
            "E_j_upper": E_j_upper,
            "E_j_lower": E_j_lower,
            "splitting": splitting
        }

    def energy_spectrum(self, n_max: int = 4) -> List[Dict]:
        spectrum = []
        for n in range(1, n_max + 1):
            for l in range(n):
                if l == 0:
                    E = self.energy_level_dirac(n, -1)
                    spectrum.append({
                        "n": n,
                        "l": l,
                        "j": 0.5,
                        "notation": f"{n}s_{{1/2}}",
                        "energy": E,
                        "degeneracy": 2
                    })
                else:
                    E1 = self.energy_level_dirac(n, l)
                    spectrum.append({
                        "n": n,
                        "l": l,
                        "j": l - 0.5,
                        "notation": f"{n}{['s', 'p', 'd', 'f', 'g'][l]}_{{{l - 0.5}}}",
                        "energy": E1,
                        "degeneracy": 2 * (l - 0.5) + 1
                    })
                    E2 = self.energy_level_dirac(n, -(l + 1))
                    spectrum.append({
                        "n": n,
                        "l": l,
                        "j": l + 0.5,
                        "notation": f"{n}{['s', 'p', 'd', 'f', 'g'][l]}_{{{l + 0.5}}}",
                        "energy": E2,
                        "degeneracy": 2 * (l + 0.5) + 1
                    })
        return spectrum


class ZitterbewegungSimulator:
    """
    Simulates the Zitterbewegung (trembling motion) of a relativistic electron.
    In Dirac theory, the position operator has a term oscillating with frequency
    ~ 2mc^2/hbar, which is the interference between positive and negative energy states.
    """

    def __init__(self, grid_size: int, c_light: float, electron_mass: float, device: str = "cpu") -> None:
        self.grid_size = grid_size
        self.c = c_light
        self.mass = electron_mass
        self.gamma = GammaMatrices("dirac", device)
        self.device = device

    def create_gaussian_wave_packet(self, sigma: float = 0.1, momentum: float = 0.0) -> torch.Tensor:
        x = torch.linspace(-math.pi, math.pi, self.grid_size, device=self.device)
        y = torch.linspace(-math.pi, math.pi, self.grid_size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        gaussian = torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        spinor = torch.zeros((4, self.grid_size, self.grid_size), dtype=torch.complex128, device=self.device)
        spinor[0] = gaussian
        spinor[1] = gaussian * 0.5j
        spinor[2] = gaussian * 0.01
        spinor[3] = gaussian * 0.01j
        if momentum != 0:
            phase = torch.exp(1j * momentum * X)
            for i in range(4):
                spinor[i] = spinor[i] * phase
        norm = torch.sqrt(torch.sum(torch.abs(spinor) ** 2)) + 1e-10
        spinor = spinor / norm
        return spinor

    def compute_position_expectation(self, spinor: torch.Tensor) -> Tuple[float, float]:
        x = torch.linspace(-math.pi, math.pi, self.grid_size, device=self.device)
        y = torch.linspace(-math.pi, math.pi, self.grid_size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        prob = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        for c in range(4):
            prob += torch.abs(spinor[c]) ** 2
        prob_sum = prob.sum() + 1e-10
        x_exp = (X * prob).sum() / prob_sum
        y_exp = (Y * prob).sum() / prob_sum
        return x_exp.item(), y_exp.item()

    def compute_velocity_expectation(self, spinor: torch.Tensor) -> Tuple[float, float]:
        vx_sum = 0.0
        vy_sum = 0.0
        spinor_flat = spinor.view(4, -1)
        for c in range(4):
            for d in range(4):
                ax = self.gamma.alpha_x[c, d].item()
                ay = self.gamma.alpha_y[c, d].item()
                vx_sum += ax * torch.sum(torch.conj(spinor_flat[c]) * spinor_flat[d]).real.item()
                vy_sum += ay * torch.sum(torch.conj(spinor_flat[c]) * spinor_flat[d]).real.item()
        vx = self.c * vx_sum
        vy = self.c * vy_sum
        return vx, vy
