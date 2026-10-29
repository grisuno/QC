#!/usr/bin/env python3
"""
Quantum Framework Physics Module (canonical)
============================================
Single source of truth for the neural physics stack used across the repo:

- ``SpectralLayer`` / ``HamiltonianBackboneNet`` / ``SchrodingerSpectralNet``
  / ``DiracSpectralNet``: FFT-based spectral networks (checkpoint-compatible).
- ``GammaMatrices``: Dirac gamma matrices (Dirac and Weyl representations).
- ``load_checkpoint``: safe checkpoint loading (shape validation, no silent
  partial loads, ``weights_only=True`` first with explicit trusted fallback).
- ``HamiltonianBackend`` / ``SchrodingerBackend`` / ``DiracBackend``:
  physics evolution engines.  Key invariants:

  * ``evolve_amplitude`` **preserves the input norm** (unitary evolution keeps
    ``||U psi|| = ||psi||``).  Previously every amplitude was renormalized to
    1, which destroyed superposition weights in the joint Hilbert space.
  * ``dt`` is honoured by every backend (learned propagators use a residual
    update scaled by ``dt / propagator_dt``).
  * If a checkpoint is missing or incompatible, the backend falls back to an
    **exact analytical evolution** and logs a WARNING (never a silent random
    network).  The analytical Hamiltonian path uses the exact unitary
    ``exp(-i k^2 dt / 2)`` free-particle propagator in Fourier space.

This module supersedes the duplicated copies that lived in
``quantum_computer.py``, ``quantum_simulator.py`` and
``quantum_framework_core.py``.

Author: Gris Iscomeback
License: AGPL v3
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


_LOG = _make_logger("QuantumPhysics")


# ---------------------------------------------------------------------------
# Spectral networks
# ---------------------------------------------------------------------------

class SpectralLayer(nn.Module):
    """
    Spectral convolution in frequency domain.

    Learns complex kernels of shape ``(channels, channels, G//2+1, G)`` that
    modulate the RFFT2 coefficients of the input.

    Modes:
        ``channel_mixing=False`` (default, legacy): the kernel is averaged
        over the output-channel dimension (``.mean(dim=0)``) and the same
        filter is applied to every channel.  This is the behaviour the
        pre-trained checkpoints were trained with, so it stays the default
        for backward compatibility.

        ``channel_mixing=True``: full per-channel-pair complex kernel applied
        via einsum (true channel mixing, higher expressivity).  Use this for
        newly trained models; the parameter layout is identical, so old
        checkpoints still load (they simply behave differently).
    """

    def __init__(self, channels: int, grid_size: int, channel_mixing: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.channel_mixing = channel_mixing
        self.kernel_real = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )
        self.kernel_imag = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution via RFFT2."""
        x_fft = torch.fft.rfft2(x)
        freq_h, freq_w = x_fft.shape[-2], x_fft.shape[-1]

        if self.channel_mixing:
            # Full (out_ch, in_ch) complex kernel, interpolated if the input
            # frequency grid differs from the training grid.
            kr = self.kernel_real
            ki = self.kernel_imag
            if (kr.shape[-2], kr.shape[-1]) != (freq_h, freq_w):
                kr = F.interpolate(kr, size=(freq_h, freq_w),
                                   mode="bilinear", align_corners=False)
                ki = F.interpolate(ki, size=(freq_h, freq_w),
                                   mode="bilinear", align_corners=False)
            kernel = torch.complex(kr, ki)                       # (o, i, fh, fw)
            out_fft = torch.einsum("oifw,bifw->bofw", kernel, x_fft)
        else:
            # Legacy behaviour: average over output channels, single shared
            # filter applied to every channel (what the checkpoints expect).
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
            out_fft = torch.complex(real_part, imag_part)

        return torch.fft.irfft2(out_fft, s=(self.grid_size, self.grid_size))


class HamiltonianBackboneNet(nn.Module):
    """Hamiltonian backbone: single-channel field -> H|psi>."""

    def __init__(self, grid_size: int, hidden_dim: int, num_spectral_layers: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralLayer(hidden_dim, grid_size) for _ in range(num_spectral_layers)]
        )
        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts (G,G), (1,G,G), or (B,1,G,G). Returns squeezed output."""
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.gelu(self.input_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        return self.output_proj(x).squeeze(1)


class SchrodingerSpectralNet(nn.Module):
    """Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction."""

    def __init__(self, grid_size: int, hidden_dim: int, expansion_dim: int,
                 num_spectral_layers: int) -> None:
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
        """(2,G,G) or (B,2,G,G) -> same shape."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        x = F.gelu(self.contraction_proj(x))
        return self.output_proj(x)


class DiracSpectralNet(nn.Module):
    """Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor."""

    def __init__(self, grid_size: int, hidden_dim: int, expansion_dim: int,
                 num_spectral_layers: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.input_proj = nn.Conv2d(8, hidden_dim, kernel_size=1)
        self.expansion_proj = nn.Conv2d(hidden_dim, expansion_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralLayer(expansion_dim, grid_size) for _ in range(num_spectral_layers)]
        )
        self.contraction_proj = nn.Conv2d(expansion_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(8,G,G) or (B,8,G,G) -> same shape."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        x = F.gelu(self.contraction_proj(x))
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Gamma matrices
# ---------------------------------------------------------------------------

class GammaMatrices:
    """
    Dirac gamma matrices in Dirac (standard) or Weyl representation.

    gamma^0 = beta, gamma^i = beta * alpha_i.
    Stored in complex128: entries are exact (0, ±1, ±i) and the higher
    precision avoids limiting float64 pipelines.
    """

    def __init__(self, representation: str = "dirac", device: str = "cpu") -> None:
        self.representation = representation
        self.device = device
        self._init_matrices()

    def _init_matrices(self) -> None:
        dtype = torch.complex128
        if self.representation == "dirac":
            self.gamma0 = torch.tensor(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
                dtype=dtype, device=self.device)
            self.gamma1 = torch.tensor(
                [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]],
                dtype=dtype, device=self.device)
            self.gamma2 = torch.tensor(
                [[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]],
                dtype=dtype, device=self.device)
            self.gamma3 = torch.tensor(
                [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]],
                dtype=dtype, device=self.device)
        elif self.representation == "weyl":
            self.gamma0 = torch.tensor(
                [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
                dtype=dtype, device=self.device)
            self.gamma1 = torch.tensor(
                [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]],
                dtype=dtype, device=self.device)
            self.gamma2 = torch.tensor(
                [[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]],
                dtype=dtype, device=self.device)
            self.gamma3 = torch.tensor(
                [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]],
                dtype=dtype, device=self.device)
        else:
            raise ValueError(f"Unknown gamma representation: {self.representation}")

        self.alpha_x = self.gamma0 @ self.gamma1
        self.alpha_y = self.gamma0 @ self.gamma2
        self.alpha_z = self.gamma0 @ self.gamma3
        self.beta = self.gamma0
        self.gamma5 = torch.tensor(
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=dtype, device=self.device)
        self.gammas = [self.gamma0, self.gamma1, self.gamma2, self.gamma3]

    def to(self, device: str) -> "GammaMatrices":
        """Move all matrices to device."""
        self.device = device
        self._init_matrices()
        return self


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(net: nn.Module, path: str, device: str, name: str) -> bool:
    """
    Load ``net`` weights from ``path`` with validation.

    - Tries ``weights_only=True`` first (safe); falls back to full pickle only
      with an explicit security notice (checkpoints in this repo contain
      numpy scalars saved by the training scripts).
    - Loads with ``strict=True``: silently partial networks are never used.
    - Validates that the spectral kernel shapes match the network grid size,
      preventing silent bilinear resampling of mismatched checkpoints.

    Returns True on success.  On any failure the network is left untouched
    and the caller is expected to fall back to analytical evolution.
    """
    if not os.path.exists(path):
        _LOG.warning(
            "%s: checkpoint not found at '%s' -> analytical fallback", name, path
        )
        return False

    try:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            _LOG.warning(
                "%s: '%s' requires weights_only=False (contains non-tensor "
                "pickle data). Only load checkpoints from sources you trust.",
                name, path,
            )
            ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        _LOG.warning("%s: failed to read '%s' (%s) -> analytical fallback",
                     name, path, exc)
        return False

    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        _LOG.warning("%s: '%s' is not a state dict -> analytical fallback", name, path)
        return False

    # Shape sanity check: spectral kernels must match this grid size.
    reference = net.state_dict()
    for key, value in state.items():
        if key in reference and tuple(reference[key].shape) != tuple(value.shape):
            _LOG.warning(
                "%s: shape mismatch for '%s' (checkpoint %s vs model %s) "
                "-> analytical fallback (grid/architecture mismatch)",
                name, key, tuple(value.shape), tuple(reference[key].shape),
            )
            return False

    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        _LOG.warning(
            "%s: incompatible checkpoint '%s' (%s) -> analytical fallback",
            name, path, exc,
        )
        return False

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    _LOG.info("%s: loaded %s", name, path)
    return True


# ---------------------------------------------------------------------------
# Physics backends (canonical implementations)
# ---------------------------------------------------------------------------

class _ConfigView:
    """
    Uniform read-only view over the two config dataclasses used in the repo
    (``SimulatorConfig`` in quantum_computer.py and ``FrameworkConfig`` in
    quantum_framework_core.py / quantum_simulator.py).
    """

    def __init__(self, config: Any) -> None:
        self.grid_size: int = getattr(config, "grid_size", 16)
        self.hidden_dim: int = getattr(config, "hidden_dim", 32)
        self.expansion_dim: int = getattr(config, "expansion_dim", 64)
        self.num_spectral_layers: int = getattr(config, "num_spectral_layers", 2)
        self.device: str = getattr(config, "device", "cpu")
        self.normalization_eps: float = getattr(config, "normalization_eps", 1.0e-8)
        self.dt: float = getattr(config, "dt", 0.01)
        self.gamma_representation: str = getattr(config, "gamma_representation", "dirac")
        # SimulatorConfig uses (dirac_mass, dirac_c); FrameworkConfig uses
        # (electron_mass, c_light).  Preserve each caller's semantics.
        self.dirac_mass: float = getattr(config, "dirac_mass", None) \
            if getattr(config, "dirac_mass", None) is not None \
            else getattr(config, "electron_mass", 1.0)
        self.dirac_c: float = getattr(config, "dirac_c", None) \
            if getattr(config, "dirac_c", None) is not None \
            else getattr(config, "c_light", 137.035999084)
        # Time step the learned propagators were trained for (residual scaling).
        self.propagator_dt: float = getattr(config, "propagator_dt", 0.01)


class IPhysicsBackend:
    """Interface for spatial wavefunction evolution backends."""

    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolve a single (2, G, G) wavefunction by dt. Preserves input norm."""
        raise NotImplementedError

    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        """Apply global phase e^{i*phi} to a (2, G, G) amplitude."""
        raise NotImplementedError


class HamiltonianBackend(IPhysicsBackend):
    """
    Hamiltonian physics backend.

    With a valid checkpoint the neural network acts as the Hamiltonian
    operator H (first-order Euler step, norm-preserving — an approximation,
    documented as such).  Without a checkpoint, falls back to the exact
    free-particle Hamiltonian H = -laplacian/2 evolved with the unitary
    spectral propagator exp(-i k^2 dt / 2).
    """

    def __init__(self, config: Any, checkpoint_attr: str = "hamiltonian_checkpoint") -> None:
        self.raw_config = config
        self.config = _ConfigView(config)
        self.device = self.config.device
        self.net: Optional[HamiltonianBackboneNet] = None
        self._k_squared: Optional[torch.Tensor] = None

        net = HamiltonianBackboneNet(
            self.config.grid_size, self.config.hidden_dim, self.config.num_spectral_layers
        ).to(self.device)
        path = getattr(config, checkpoint_attr, "") or ""
        if path and load_checkpoint(net, path, self.device, "HamiltonianBackend"):
            self.net = net
        self._precompute_kinetic()

    def _precompute_kinetic(self) -> None:
        """Precompute k^2 = kx^2 + ky^2 for the exact kinetic propagator."""
        G = self.config.grid_size
        kx = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        ky = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self._k_squared = (KX ** 2 + KY ** 2).to(self.device)

    def _apply_h(self, field: torch.Tensor) -> torch.Tensor:
        """Apply H to a real field (H psi for one real channel)."""
        if self.net is not None:
            with torch.no_grad():
                result = self.net(field.to(self.device))
                return result.squeeze() if result.dim() > 2 else result
        # Analytical: H = -laplacian / 2  (eigenvalue k^2/2 in Fourier space)
        fft = torch.fft.fft2(field.to(self.device))
        return torch.fft.ifft2(fft * (0.5 * self._k_squared)).real

    def _norm(self, amp: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((amp ** 2).sum())

    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Evolve by dt preserving the input norm (unitary evolution keeps
        ||psi||).  Analytical path uses the exact unitary propagator.
        """
        amp = amp.to(self.device)
        in_norm = self._norm(amp)
        if in_norm < self.config.normalization_eps:
            return amp

        if self.net is None:
            # Exact unitary evolution: psi_k -> exp(-i (k^2/2) dt) psi_k
            psi = torch.complex(amp[0], amp[1])
            phase = torch.exp(-0.5j * self._k_squared.to(psi.dtype) * dt)
            psi_t = torch.fft.ifft2(torch.fft.fft2(psi) * phase)
            out = torch.stack([psi_t.real, psi_t.imag], dim=0)
            return out  # exact propagator preserves norm by construction

        # Neural path: first-order Euler step (approximation; see module docs).
        psi_r, psi_i = amp[0], amp[1]
        h_r = self._apply_h(psi_r)
        h_i = self._apply_h(psi_i)
        new_r = psi_r + dt * h_i
        new_i = psi_i - dt * h_r
        out = torch.stack([new_r, new_i], dim=0)
        # Restore the input norm (Euler is not norm-preserving).
        out_norm = self._norm(out) + self.config.normalization_eps
        return out * (in_norm / out_norm)

    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        c = math.cos(phase_angle)
        s = math.sin(phase_angle)
        return torch.stack([c * amp[0] - s * amp[1], s * amp[0] + c * amp[1]], dim=0)


class SchrodingerBackend(IPhysicsBackend):
    """
    Schrodinger physics backend driven by the learned 2-channel propagator.

    The network is treated as a propagator trained for a fixed step
    ``propagator_dt``; arbitrary dt values are honoured through a residual
    update ``psi + (dt/propagator_dt) * (Net(psi) - psi)`` (identity at
    dt=0).  Without a checkpoint, delegates to the Hamiltonian backend
    (which itself falls back to exact unitary evolution).
    """

    def __init__(self, config: Any, hamiltonian: HamiltonianBackend) -> None:
        self.raw_config = config
        self.config = _ConfigView(config)
        self.device = self.config.device
        self.hamiltonian = hamiltonian
        self.net: Optional[SchrodingerSpectralNet] = None

        net = SchrodingerSpectralNet(
            self.config.grid_size, self.config.hidden_dim,
            self.config.expansion_dim, self.config.num_spectral_layers
        ).to(self.device)
        path = getattr(config, "schrodinger_checkpoint", "") or ""
        if path and load_checkpoint(net, path, self.device, "SchrodingerBackend"):
            self.net = net

    def evolve_amplitude(self, amp: torch.Tensor, dt: float) -> torch.Tensor:
        if self.net is None:
            return self.hamiltonian.evolve_amplitude(amp, dt)

        amp = amp.to(self.device)
        in_norm = torch.sqrt((amp ** 2).sum())
        if in_norm < self.config.normalization_eps:
            return amp
        with torch.no_grad():
            net_out = self.net(amp.unsqueeze(0)).squeeze(0)
        scale = dt / self.config.propagator_dt
        out = amp + scale * (net_out - amp)
        out_norm = torch.sqrt((out ** 2).sum()) + self.config.normalization_eps
        return out * (in_norm / out_norm)

    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        return self.hamiltonian.apply_phase(amp, phase_angle)


class DiracBackend(IPhysicsBackend):
    """
    Dirac relativistic physics backend.

    Expands each (2,G,G) amplitude to a 4-component spinor, propagates it
    (learned network with residual dt scaling, or the analytical Dirac
    Hamiltonian c*alpha.p + beta*m*c^2), then projects back.  Evolution
    preserves the input norm.
    """

    def __init__(self, config: Any, hamiltonian: HamiltonianBackend) -> None:
        self.raw_config = config
        self.config = _ConfigView(config)
        self.device = self.config.device
        self.hamiltonian = hamiltonian
        self.gamma = GammaMatrices(self.config.gamma_representation, self.device)
        self.net: Optional[DiracSpectralNet] = None

        net = DiracSpectralNet(
            self.config.grid_size, self.config.hidden_dim,
            self.config.expansion_dim, self.config.num_spectral_layers
        ).to(self.device)
        path = getattr(config, "dirac_checkpoint", "") or ""
        if path and load_checkpoint(net, path, self.device, "DiracBackend"):
            self.net = net
        self._precompute_dirac()

    def _precompute_dirac(self) -> None:
        G = self.config.grid_size
        kx = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        ky = torch.fft.fftfreq(G, d=1.0) * 2.0 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self.kx_grid = KX.to(self.device)
        self.ky_grid = KY.to(self.device)

    def _pack(self, amp: torch.Tensor) -> torch.Tensor:
        """Pack 2-channel amplitude into a 4-component spinor (heuristic)."""
        G = self.config.grid_size
        psi_c = torch.complex(amp[0].to(self.device), amp[1].to(self.device))
        s = math.sqrt(0.5)
        spinor = torch.zeros(4, G, G, dtype=torch.complex128, device=self.device)
        spinor[0] = psi_c * s
        spinor[1] = psi_c * s
        spinor[2] = psi_c.conj() * s
        spinor[3] = psi_c.conj() * s
        return spinor

    def _unpack(self, spinor: torch.Tensor) -> torch.Tensor:
        """Project 4-component spinor back to a 2-channel amplitude."""
        particle = spinor[:2].mean(dim=0)
        return torch.stack([particle.real, particle.imag], dim=0)

    def _analytical_dirac(self, spinor: torch.Tensor) -> torch.Tensor:
        """Apply analytical Dirac Hamiltonian H = c*alpha.p + beta*m*c^2."""
        m, c = self.config.dirac_mass, self.config.dirac_c
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
        amp = amp.to(self.device)
        in_norm = torch.sqrt((amp ** 2).sum())
        if in_norm < self.config.normalization_eps:
            return amp

        spinor = self._pack(amp)
        if self.net is not None:
            net_dtype = next(self.net.parameters()).dtype
            channels = torch.cat([spinor.real, spinor.imag], dim=0).unsqueeze(0).to(net_dtype)
            with torch.no_grad():
                net_out = self.net(channels).squeeze(0)
            net_spinor = torch.complex(net_out[:4], net_out[4:]).to(spinor.dtype)
            scale = dt / self.config.propagator_dt
            spinor_out = spinor + scale * (net_spinor - spinor)
        else:
            h_spinor = self._analytical_dirac(spinor)
            spinor_out = spinor - 1j * dt * h_spinor

        # Project back and restore the input norm.
        out = self._unpack(spinor_out)
        out_norm = torch.sqrt((out ** 2).sum()) + self.config.normalization_eps
        return out * (in_norm / out_norm)

    def apply_phase(self, amp: torch.Tensor, phase_angle: float) -> torch.Tensor:
        return self.hamiltonian.apply_phase(amp, phase_angle)

    def evolve_spinor(self, spinor: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolve a full 4-component spinor analytically, preserving its norm."""
        spinor = spinor.to(torch.complex128).to(self.device)
        in_norm = torch.sqrt((spinor.abs() ** 2).sum())
        h_spinor = self._analytical_dirac(spinor)
        result = spinor - 1j * dt * h_spinor
        out_norm = torch.sqrt((result.abs() ** 2).sum()) + self.config.normalization_eps
        return result * (in_norm / out_norm)


# ---------------------------------------------------------------------------
# Analytical physics utilities
# ---------------------------------------------------------------------------

class PotentialGenerator:
    """Spatial potentials for eigenstate initialization."""

    def __init__(self, grid_size: int = 16, potential_depth: float = 5.0,
                 potential_width: float = 0.3) -> None:
        self.grid_size = grid_size
        self.potential_depth = potential_depth
        self.potential_width = potential_width

    def _grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(0, 2 * math.pi, self.grid_size)
        y = torch.linspace(0, 2 * math.pi, self.grid_size)
        return torch.meshgrid(x, y, indexing="ij")

    def harmonic(self) -> torch.Tensor:
        """V = k/2 * r^2."""
        X, Y = self._grid()
        cx, cy = math.pi, math.pi
        return 0.5 * self.potential_depth * ((X - cx) ** 2 + (Y - cy) ** 2) / (math.pi ** 2)

    def double_well(self) -> torch.Tensor:
        """Double-well along x."""
        X, _ = self._grid()
        cx = math.pi
        w = self.potential_width * math.pi
        return self.potential_depth * ((X - cx) ** 2 / w ** 2 - 1.0) ** 2

    def coulomb(self) -> torch.Tensor:
        """Coulomb-like V ~ -1/r."""
        X, Y = self._grid()
        cx, cy = math.pi, math.pi
        r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2) + self.potential_width
        return -self.potential_depth / r

    def periodic_lattice(self) -> torch.Tensor:
        """Periodic cosine lattice."""
        X, Y = self._grid()
        return self.potential_depth * (torch.cos(2.0 * X) + torch.cos(2.0 * Y))

    def mixed(self, seed: int) -> torch.Tensor:
        """Dirichlet-weighted mixture of all four potentials (local RNG)."""
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

    FIXED: the kinetic term now correctly contracts alpha_{cd} with psi_d
    (previously it always used psi_c, so spinor components never mixed).
    Vectorized with einsum instead of per-element .item() loops.
    """

    def __init__(self, grid_size: int, electron_mass: float, c_light: float,
                 device: str = "cpu") -> None:
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

    def apply_dirac_hamiltonian(
        self,
        spinor: torch.Tensor,
        potential: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply the Dirac Hamiltonian to a 4-component spinor.

        Args:
            spinor: shape (4, H, W) or (batch, 4, H, W)
            potential: optional scalar potential V(r)

        Returns:
            H psi with the same shape as the input.
        """
        squeeze_output = spinor.dim() == 3
        if squeeze_output:
            spinor = spinor.unsqueeze(0)
        spinor = spinor.to(torch.complex128)

        # Momentum of each component: p psi_d for d = 0..3
        fft = torch.fft.fft2(spinor)                                  # (b, 4, H, W)
        px = torch.fft.ifft2(fft * self.kx_grid)                      # (b, 4, H, W)
        py = torch.fft.ifft2(fft * self.ky_grid)

        # (H psi)_c = sum_d [ c (alpha_x_cd p_x + alpha_y_cd p_y) + m c^2 beta_cd ] psi_d
        result = self.c * (
            torch.einsum("cd,bdhw->bchw", self.gamma.alpha_x.to(torch.complex128), px)
            + torch.einsum("cd,bdhw->bchw", self.gamma.alpha_y.to(torch.complex128), py)
        ) + self.mass * self.c ** 2 * torch.einsum(
            "cd,bdhw->bchw", self.gamma.beta.to(torch.complex128), spinor
        )

        if potential is not None:
            result = result + potential * spinor

        return result.squeeze(0) if squeeze_output else result

    def time_evolution(
        self,
        spinor: torch.Tensor,
        dt: float,
        potential: Optional[torch.Tensor] = None,
        normalization_eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        First-order time evolution psi -> (1 - i H dt) psi, preserving the
        input norm (first-order unitary approximation).
        """
        squeeze_output = spinor.dim() == 3
        if squeeze_output:
            spinor = spinor.unsqueeze(0)
        h_psi = self.apply_dirac_hamiltonian(spinor, potential)
        result = spinor - 1j * dt * h_psi
        norm_original = torch.norm(spinor.view(spinor.shape[0], -1), dim=1, keepdim=True)
        norm_evolved = torch.norm(result.view(result.shape[0], -1), dim=1, keepdim=True)
        result = result * (norm_original / (norm_evolved + normalization_eps)).unsqueeze(-1).unsqueeze(-1)
        return result.squeeze(0) if squeeze_output else result


class LambShiftCalculator:
    """
    Lamb shift of hydrogenic levels at leading order (alpha (Z alpha)^4 m c^2).

    S states:  dE = 4 alpha^3 Z^4 / (3 pi n^3) [ln(1/(Z alpha)^2) - ln k0(n,0) + 5/6]
    plus the Uehling vacuum-polarisation term -4 alpha^3 Z^4 / (15 pi n^3).
    l > 0:     dE = 4 alpha^3 Z^4 / (3 pi n^3) [-ln k0(n,l) - 3/(8 kappa (2l+1))],
    where the last term is the electron anomalous magnetic moment (kappa = -(l+1) for
    j = l+1/2 and kappa = l for j = l-1/2).  Atomic units, alpha c = 1.
    """

    BETHE_LOG = {
        (1, 0): 2.984128, (2, 0): 2.811769, (3, 0): 2.767664, (4, 0): 2.749824, (5, 0): 2.740823,
        (2, 1): -0.030017, (3, 1): -0.038191, (3, 2): -0.005232,
        (4, 1): -0.041954, (4, 2): -0.008148, (4, 3): -0.001755,
    }
    HARTREE_MHZ = 6.579683920502e9
    BETHE_LOG_S_LIMIT = 2.72

    def __init__(self, alpha_fs: float = 0.0072973525693,
                 c_light: float = 137.035999084, electron_mass: float = 1.0) -> None:
        self.alpha = alpha_fs
        self.c = c_light
        self.mass = electron_mass

    def _prefactor(self, n: int, Z: int) -> float:
        return 4 * self.alpha ** 3 * Z ** 4 / (3 * math.pi * n ** 3)

    def bethe_log(self, n: int, l: int) -> float:
        if (n, l) in self.BETHE_LOG:
            return self.BETHE_LOG[(n, l)]
        return self.BETHE_LOG_S_LIMIT if l == 0 else 0.0

    def bethe_formula(self, n: int, l: int, Z: int = 1, j: float | None = None) -> float:
        """Self-energy part of the shift in Hartree."""
        if l == 0:
            return self._prefactor(n, Z) * (math.log(1.0 / (Z * self.alpha) ** 2) - self.bethe_log(n, 0) + 5.0 / 6.0)
        j = l + 0.5 if j is None else j
        kappa = -(l + 1) if j > l else l
        return self._prefactor(n, Z) * (-self.bethe_log(n, l) - 3.0 / (8.0 * kappa * (2 * l + 1)))

    def full_lamb_shift(self, n: int, l: int, j: float, Z: int = 1) -> Dict[str, float]:
        e_se = self.bethe_formula(n, l, Z, j)
        e_uehling = -4 * self.alpha ** 3 * Z ** 4 / (15 * math.pi * n ** 3) if l == 0 else 0.0
        total = e_se + e_uehling
        return {
            "n": n, "l": l, "j": j, "Z": Z,
            "self_energy_au": e_se,
            "vacuum_polarization_au": e_uehling,
            "total_au": total,
            "total_MHz": total * self.HARTREE_MHZ,
        }


class AnomalousMagneticMoment:
    """Electron anomalous magnetic moment (g-2)/2 up to 5th order in alpha."""

    def __init__(self, alpha_fs: float = 0.0072973525693) -> None:
        self.alpha = alpha_fs

    def schwinger_term(self) -> float:
        return self.alpha / (2 * math.pi)

    def second_order(self) -> float:
        return (self.alpha / math.pi) ** 2 * 0.32847896557919378

    def third_order(self) -> float:
        return (self.alpha / math.pi) ** 3 * 1.181241456587

    def fourth_order(self) -> float:
        return (self.alpha / math.pi) ** 4 * (-1.9144)

    def fifth_order(self) -> float:
        return (self.alpha / math.pi) ** 5 * 7.7

    def calculate_a_e(self, order: int = 5) -> Dict[str, float]:
        contributions: Dict[str, float] = {"order_1": self.schwinger_term()}
        if order >= 2:
            contributions["order_2"] = self.second_order()
        if order >= 3:
            contributions["order_3"] = self.third_order()
        if order >= 4:
            contributions["order_4"] = self.fourth_order()
        if order >= 5:
            contributions["order_5"] = self.fifth_order()
        total = sum(contributions.values())
        experimental = 0.00115965218128
        contributions["total"] = total
        contributions["experimental"] = experimental
        contributions["error"] = abs(total - experimental)
        contributions["relative_error"] = contributions["error"] / experimental
        return contributions


class DiracHydrogenAtom:
    """Relativistic hydrogen energy levels from the exact Dirac spectrum."""

    def __init__(self, c_light: float = 137.035999084,
                 alpha_fs: float = 0.0072973525693) -> None:
        self.c = c_light
        self.alpha_fs = alpha_fs

    def energy_level_dirac(self, n: int, kappa: int) -> float:
        alpha = self.alpha_fs
        kappa_abs = abs(kappa)
        sqrt_term = math.sqrt(kappa_abs ** 2 - alpha ** 2)
        denominator = n - kappa_abs + sqrt_term
        E = 1.0 / math.sqrt(1.0 + (alpha / denominator) ** 2)
        return (E - 1.0) * self.c ** 2

    def fine_structure_splitting(self, n: int, l: int) -> Dict[str, float]:
        alpha = self.alpha_fs
        if l == 0:
            E_fs = -alpha ** 2 * self.c ** 2 / (2 * n ** 4) * (4 * n / (l + 0.5) - 3)
            return {"j_1_2": self.energy_level_dirac(n, -1), "E_fine_structure": E_fs}
        E_j_upper = self.energy_level_dirac(n, -(l + 1))
        E_j_lower = self.energy_level_dirac(n, l)
        return {
            "j_upper": l + 0.5,
            "j_lower": l - 0.5,
            "E_j_upper": E_j_upper,
            "E_j_lower": E_j_lower,
            "splitting": E_j_upper - E_j_lower,
        }

    def energy_spectrum(self, n_max: int = 4) -> List[Dict]:
        spectrum = []
        letters = ["s", "p", "d", "f", "g"]
        for n in range(1, n_max + 1):
            for l in range(n):
                if l == 0:
                    spectrum.append({
                        "n": n, "l": l, "j": 0.5,
                        "notation": f"{n}s_{{1/2}}",
                        "energy": self.energy_level_dirac(n, -1),
                        "degeneracy": 2,
                    })
                else:
                    spectrum.append({
                        "n": n, "l": l, "j": l - 0.5,
                        "notation": f"{n}{letters[l]}_{{{l - 0.5}}}",
                        "energy": self.energy_level_dirac(n, l),
                        "degeneracy": 2 * (l - 0.5) + 1,
                    })
                    spectrum.append({
                        "n": n, "l": l, "j": l + 0.5,
                        "notation": f"{n}{letters[l]}_{{{l + 0.5}}}",
                        "energy": self.energy_level_dirac(n, -(l + 1)),
                        "degeneracy": 2 * (l + 0.5) + 1,
                    })
        return spectrum


class ZitterbewegungSimulator:
    """
    Simulates Zitterbewegung (trembling motion) of a relativistic electron:
    interference between positive and negative energy components.
    """

    def __init__(self, grid_size: int, c_light: float, electron_mass: float,
                 device: str = "cpu") -> None:
        self.grid_size = grid_size
        self.c = c_light
        self.mass = electron_mass
        self.gamma = GammaMatrices("dirac", device)
        self.device = device

    def create_gaussian_wave_packet(self, sigma: float = 0.1,
                                   momentum: float = 0.0) -> torch.Tensor:
        x = torch.linspace(-math.pi, math.pi, self.grid_size, device=self.device)
        y = torch.linspace(-math.pi, math.pi, self.grid_size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        gaussian = torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        spinor = torch.zeros((4, self.grid_size, self.grid_size),
                             dtype=torch.complex128, device=self.device)
        spinor[0] = gaussian
        spinor[1] = gaussian * 0.5j
        spinor[2] = gaussian * 0.01
        spinor[3] = gaussian * 0.01j
        if momentum != 0:
            phase = torch.exp(1j * momentum * X)
            for i in range(4):
                spinor[i] = spinor[i] * phase
        norm = torch.sqrt(torch.sum(torch.abs(spinor) ** 2)) + 1e-10
        return spinor / norm

    def compute_position_expectation(self, spinor: torch.Tensor) -> Tuple[float, float]:
        x = torch.linspace(-math.pi, math.pi, self.grid_size, device=self.device)
        y = torch.linspace(-math.pi, math.pi, self.grid_size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        prob = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        for c in range(4):
            prob += torch.abs(spinor[c]) ** 2
        prob_sum = prob.sum() + 1e-10
        return float((X * prob).sum() / prob_sum), float((Y * prob).sum() / prob_sum)

    def compute_velocity_expectation(self, spinor: torch.Tensor) -> Tuple[float, float]:
        spinor_flat = spinor.view(4, -1).to(torch.complex128)
        alpha_x = self.gamma.alpha_x.to(torch.complex128)
        alpha_y = self.gamma.alpha_y.to(torch.complex128)
        vx = self.c * torch.einsum(
            "cd,c,d->", alpha_x, torch.conj(spinor_flat), spinor_flat
        ).real.item()
        vy = self.c * torch.einsum(
            "cd,c,d->", alpha_y, torch.conj(spinor_flat), spinor_flat
        ).real.item()
        return vx, vy
