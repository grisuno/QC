"""Fields on the 2D torus T^2 = [0, 2 pi)^2 and the exact Hamiltonian they are mapped through.

H = -(hbar^2 / 2m) laplacian, applied spectrally, so the network learns the same operator the project's analytic
backend falls back to (eigenvalue 0.5 k^2 in Fourier space, k = 2 pi fftfreq(d=1) radians per grid cell).
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset

from .config import HConfig


class TorusHamiltonian:
    """Exact kinetic operator on the periodic grid."""

    def __init__(self, config: HConfig):
        g = config.grid_size
        k = torch.fft.fftfreq(g, d=1.0) * 2 * math.pi
        kx, ky = torch.meshgrid(k, k, indexing="ij")
        self.k2 = (kx**2 + ky**2).to(config.device)
        self.symbol = config.kinetic_prefactor * config.hbar**2 / config.mass * self.k2

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(field) * self.symbol).real

    def propagate(self, psi: torch.Tensor, dt: float) -> torch.Tensor:
        """Exact unitary exp(-i H dt) on a complex field."""
        return torch.fft.ifft2(torch.fft.fft2(psi) * torch.exp(-1j * self.symbol * dt))


def random_fields(count: int, config: HConfig, generator: torch.Generator) -> torch.Tensor:
    """Band-limited random fields of random amplitude: the range of states the quantum backend feeds the network."""
    g = config.grid_size
    fields = torch.randn(count, g, g, generator=generator)
    k = torch.fft.fftfreq(g, d=1.0) * 2 * math.pi
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    radius = torch.sqrt(kx**2 + ky**2)
    cutoff = torch.empty(count, 1, 1).uniform_(config.cutoff_min, config.cutoff_max, generator=generator)
    filtered = torch.fft.ifft2(torch.fft.fft2(fields) * (radius[None] < cutoff).float()).real
    filtered = filtered / filtered.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-12)
    scale = torch.empty(count, 1, 1).uniform_(config.scale_min, config.scale_max, generator=generator)
    return filtered * scale


def line_fields(count: int, config: HConfig, generator: torch.Generator) -> torch.Tensor:
    """Fields concentrated on one grid line (a delta across, smooth along), the shape of the quantum computer's basis amplitudes.

    Random orientation and position (the torus is translation invariant), random low modes along the line, a small
    imaginary-like admixture is covered by the caller feeding re/im channels separately.
    """
    g = config.grid_size
    fields = torch.zeros(count, g, g)
    along = torch.arange(g, dtype=torch.float32)
    for i in range(count):
        profile = torch.zeros(g)
        for _ in range(int(torch.randint(1, config.line_modes + 1, (1,), generator=generator))):
            mode = int(torch.randint(0, g // 2, (1,), generator=generator))
            phase = float(torch.rand(1, generator=generator)) * 2 * math.pi
            profile = profile + float(torch.rand(1, generator=generator)) * torch.cos(2 * math.pi * mode * along / g + phase)
        line = torch.zeros(g, g)
        line[int(torch.randint(0, g, (1,), generator=generator))] = profile
        fields[i] = line if float(torch.rand(1, generator=generator)) < 0.5 else line.T
    fields = fields / fields.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-12)
    scale = torch.empty(count, 1, 1).uniform_(config.scale_min, config.scale_max, generator=generator)
    return fields * scale


def mixed_fields(count: int, config: HConfig, generator: torch.Generator) -> torch.Tensor:
    lines = int(count * config.line_fraction)
    parts = [random_fields(count - lines, config, generator)]
    if lines:
        parts.append(line_fields(lines, config, generator))
    fields = torch.cat(parts)
    return fields[torch.randperm(len(fields), generator=generator)]


class HamiltonianDataset(Dataset):
    """(psi, H psi) pairs, generated once per seed."""

    def __init__(self, config: HConfig, seed: int, samples: int | None = None):
        self.operator = TorusHamiltonian(config)
        gen = torch.Generator().manual_seed(seed)
        self.inputs = mixed_fields(samples or config.train_samples, config, gen).to(config.device)
        self.targets = self.operator.apply(self.inputs)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


def validation_set(config: HConfig) -> tuple[torch.Tensor, torch.Tensor]:
    data = HamiltonianDataset(config, config.data_seed, config.val_samples)
    return data.inputs, data.targets
