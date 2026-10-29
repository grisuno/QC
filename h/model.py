"""Hamiltonian network on the 2D torus.

Layout and forward pass are identical to quantum_framework_physics.HamiltonianBackboneNet, so a checkpoint saved from
this module loads (strict) as the project's `hamiltonian.pth`:

    Conv1x1(1 -> C)  -> GELU -> [spectral layer -> GELU] x N -> Conv1x1(C -> 1)

A spectral layer multiplies the rFFT2 of every channel by a complex kernel defined on the torus frequency grid
(G//2+1, G). The kernel is the mean over the output-channel axis of a (C, C, G//2+1, G) parameter, exactly as the
project's backend (and the trained checkpoints) expect.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HConfig


class SpectralTorusLayer(nn.Module):
    """Complex Fourier multiplier on the torus with an optional GOE/GUE imaginary ratio at initialisation."""

    def __init__(self, channels: int, grid_size: int, init_std: float, imaginary_ratio: float = 1.0):
        super().__init__()
        self.grid_size = grid_size
        shape = (channels, channels, grid_size // 2 + 1, grid_size)
        self.kernel_real = nn.Parameter(torch.randn(*shape) * init_std)
        self.kernel_imag = nn.Parameter(torch.randn(*shape) * init_std * imaginary_ratio)

    def kernel(self, freq_h: int, freq_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        kr = F.interpolate(self.kernel_real.mean(dim=0).unsqueeze(0), size=(freq_h, freq_w), mode="bilinear", align_corners=False).squeeze(0)
        ki = F.interpolate(self.kernel_imag.mean(dim=0).unsqueeze(0), size=(freq_h, freq_w), mode="bilinear", align_corners=False).squeeze(0)
        return kr, ki

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft2(x)
        kr, ki = self.kernel(x_fft.shape[-2], x_fft.shape[-1])
        real = x_fft.real * kr - x_fft.imag * ki
        imag = x_fft.real * ki + x_fft.imag * kr
        return torch.fft.irfft2(torch.complex(real, imag), s=(self.grid_size, self.grid_size))


class HamiltonianTorusNet(nn.Module):
    """Single-channel field on the torus -> H psi."""

    def __init__(self, config: HConfig | None = None, imaginary_ratio: float = 1.0):
        super().__init__()
        cfg = config or HConfig()
        self.grid_size = cfg.grid_size
        self.input_proj = nn.Conv2d(1, cfg.hidden_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralTorusLayer(cfg.hidden_dim, cfg.grid_size, cfg.init_std, imaginary_ratio) for _ in range(cfg.num_spectral_layers)]
        )
        self.output_proj = nn.Conv2d(cfg.hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.gelu(self.input_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        return self.output_proj(x).squeeze(1)
