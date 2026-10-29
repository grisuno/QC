"""Maxwell network experiment: does the checkpoint compute the Maxwell evolution it was trained for?

The network (6 real channels: Re/Im of Ex, Ey, Bz on the 16x16 torus; 1x1 projection -> expansion -> spectral layers ->
contraction -> 1x1) is reproduced from maxwell_crystal.py so a checkpoint loads strict. The target is built as in the
training set: two Euler steps of the exact TM Maxwell operator (dBz/dt = -(dEy/dx - dEx/dy), dEx/dt = dBz/dy,
dEy/dt = -dBz/dx) with the norm restored, on the real parts; imaginary channels target zero.

    python3 -m h.experiments.maxwell /path/to/maxwell/checkpoints_maxwell_phase4/latest.pth
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 16
HIDDEN = 32
EXPANSION = 64
LAYERS = 2
COMPONENTS = 3
DT = 0.01
STEPS = 2
SAMPLES = 256
ZERO_TOLERANCE = 0.05
DEFAULT_CHECKPOINT = Path.home() / "src_note/py/maxwell/checkpoints_maxwell_phase4/latest.pth"


class MaxwellSpectralLayer(nn.Module):
    def __init__(self, channels: int, grid: int):
        super().__init__()
        self.grid = grid
        shape = (channels, channels, grid // 2 + 1, grid)
        self.kernel_real = nn.Parameter(torch.zeros(*shape))
        self.kernel_imag = nn.Parameter(torch.zeros(*shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft2(x)
        h, w = x_fft.shape[-2:]
        kr = F.interpolate(self.kernel_real.mean(dim=0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        ki = F.interpolate(self.kernel_imag.mean(dim=0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        out = torch.complex(x_fft.real * kr - x_fft.imag * ki, x_fft.real * ki + x_fft.imag * kr)
        return torch.fft.irfft2(out, s=(self.grid, self.grid))


class MaxwellTorusNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = COMPONENTS * 2
        self.input_proj = nn.Conv2d(channels, HIDDEN, 1)
        self.expansion_proj = nn.Conv2d(HIDDEN, EXPANSION, 1)
        self.spectral_layers = nn.ModuleList([MaxwellSpectralLayer(EXPANSION, GRID) for _ in range(LAYERS)])
        self.contraction_proj = nn.Conv2d(EXPANSION, HIDDEN, 1)
        self.output_proj = nn.Conv2d(HIDDEN, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for layer in self.spectral_layers:
            x = F.gelu(layer(x))
        return self.output_proj(F.gelu(self.contraction_proj(x)))


class ExactMaxwell:
    """Spectral TM Maxwell operator on the periodic grid (vacuum, c = eps = mu = 1)."""

    def __init__(self):
        k = torch.fft.fftfreq(GRID, d=1.0) * 2 * math.pi
        self.kx, self.ky = torch.meshgrid(k, k, indexing="ij")

    def rate(self, f: torch.Tensor) -> torch.Tensor:
        ex, ey, bz = f[:, 0], f[:, 1], f[:, 2]
        d = lambda field, k: torch.fft.ifft2(1j * k * torch.fft.fft2(field)).real
        return torch.stack([d(bz, self.ky), -d(bz, self.kx), -(d(ey, self.kx) - d(ex, self.ky))], dim=1)

    def evolve(self, f: torch.Tensor) -> torch.Tensor:
        norm = f.flatten(1).norm(dim=1)[:, None, None, None]
        for _ in range(STEPS):
            f = f + DT * self.rate(f)
            f = f * norm / f.flatten(1).norm(dim=1)[:, None, None, None].clamp_min(1e-12)
        return f


def make_batch(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    fields = torch.randn(count, COMPONENTS, 2, GRID, GRID, generator=gen)
    fields = fields / fields.flatten(1).norm(dim=1)[:, None, None, None, None]
    real = fields[:, :, 0]
    target_real = ExactMaxwell().evolve(real * 1.0)
    target = torch.zeros_like(fields)
    target[:, :, 0] = target_real
    return fields.reshape(count, COMPONENTS * 2, GRID, GRID), target.reshape(count, COMPONENTS * 2, GRID, GRID)


@torch.no_grad()
def audit(checkpoint: str) -> Dict[str, object]:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = MaxwellTorusNet()
    net.load_state_dict(raw["model_state_dict"], strict=True)
    net.eval()
    x, y = make_batch(SAMPLES, seed=999)
    out = net(x)
    rel = (out - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)
    saved_loss = raw.get("metrics", {}).get("val_loss")
    zero_mse = 1.0 / (COMPONENTS * 2 * GRID * GRID)
    return {
        "epoch": raw.get("epoch"),
        "lambda_pressure": raw.get("lambda_pressure"),
        "relative_error": float(rel.mean()),
        "output_to_target_norm": float(out.flatten(1).norm(dim=1).mean() / y.flatten(1).norm(dim=1).mean()),
        "saved_val_loss": saved_loss,
        "mse_of_predicting_zero_for_unit_norm_target": zero_mse,
        "saved_loss_equals_zero_solution": bool(saved_loss is not None and abs(saved_loss - zero_mse) / zero_mse < ZERO_TOLERANCE),
        "computes_maxwell": bool(float(rel.mean()) < 0.1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT_CHECKPOINT))
    result = audit(parser.parse_args(argv).checkpoint)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
