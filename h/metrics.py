"""Metrics that cannot be satisfied by the trivial zero solution."""

from __future__ import annotations

import torch

from .config import HConfig
from .data import TorusHamiltonian


def relative_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample |out - target| / |target| (Frobenius). Zero output gives exactly 1."""
    return (output - target).flatten(1).norm(dim=1) / target.flatten(1).norm(dim=1).clamp_min(1e-12)


def relative_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (relative_error(output, target) ** 2).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, config: HConfig) -> dict:
    """Operator quality: relative error, norm ratio, accuracy against the target, Hermiticity defect, <H> error."""
    out = model(inputs).reshape_as(targets)
    rel = relative_error(out, targets)
    norm_ratio = (out.flatten(1).norm(dim=1) / targets.flatten(1).norm(dim=1)).mean()
    half = inputs.shape[0] // 2
    a, b = inputs[:half], inputs[half : 2 * half]
    h_a, h_b = model(a).reshape_as(a), model(b).reshape_as(b)
    cross = ((b * h_a).flatten(1).sum(1) - (a * h_b).flatten(1).sum(1)).abs()
    hermiticity = (cross / (a.flatten(1).norm(dim=1) * h_a.flatten(1).norm(dim=1)).clamp_min(1e-12)).mean()
    e_net = (inputs * out).flatten(1).sum(1) / (inputs**2).flatten(1).sum(1)
    e_ref = (inputs * targets).flatten(1).sum(1) / (inputs**2).flatten(1).sum(1)
    return {
        "rel_error": float(rel.mean()),
        "rel_error_max": float(rel.max()),
        "accuracy": float((rel < config.accuracy_target).float().mean()),
        "norm_ratio": float(norm_ratio),
        "hermiticity_defect": float(hermiticity),
        "energy_error": float(((e_net - e_ref).abs() / e_ref.abs().clamp_min(1e-12)).mean()),
    }


@torch.no_grad()
def propagation_fidelity(model: torch.nn.Module, config: HConfig, steps: int = 200, dt: float = 0.01) -> dict:
    """Euler steps driven by the network (the project's HamiltonianBackend recipe) versus the exact propagator."""
    op = TorusHamiltonian(config)
    g = config.grid_size
    axis = torch.arange(g, dtype=torch.float32) - g / 2
    xs, ys = torch.meshgrid(axis, axis, indexing="ij")
    psi0 = torch.exp(-(xs**2 + ys**2) / 4.0) * torch.exp(1j * 1.2 * xs)
    exact = psi0.clone()
    evolved = torch.stack([psi0.real, psi0.imag])
    norm0 = evolved.norm()
    for _ in range(steps):
        exact = op.propagate(exact, dt)
        h_r, h_i = model(evolved[0]).reshape(g, g), model(evolved[1]).reshape(g, g)
        evolved = torch.stack([evolved[0] + dt * h_i, evolved[1] - dt * h_r])
        evolved = evolved * (norm0 / evolved.norm())
    state = torch.complex(evolved[0], evolved[1])
    overlap = (exact.conj() * state).sum().abs() ** 2 / ((exact.abs() ** 2).sum() * (state.abs() ** 2).sum())
    idle = (exact.conj() * psi0).sum().abs() ** 2 / ((exact.abs() ** 2).sum() * (psi0.abs() ** 2).sum())
    return {"fidelity": float(overlap), "idle_fidelity": float(idle), "steps": steps, "dt": dt}
