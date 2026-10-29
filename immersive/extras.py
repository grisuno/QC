"""Exact hydrogenic extras: Born-rule sampling and the Rydberg wave packet (a classical orbit emerging from quantum states)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import sph_harm_y

from .physics import radial_function

State = Tuple[int, int, int, complex]

BORN_RADIAL_CELLS = 120
BORN_COS_CELLS = 24
BORN_PHI_CELLS = 32
RYDBERG_GRID = 80
RYDBERG_SPREAD = 1.2
RYDBERG_SPAN = 4
RYDBERG_RADIUS_FACTOR = 1.15
RYDBERG_RADIUS_SHELLS = 3


def hydrogenic_density(states: Sequence[State], coefs: Sequence[complex], r, theta, phi) -> np.ndarray:
    """|psi|^2 of a superposition of hydrogenic states at spherical coordinates (Bohr radii)."""
    psi = np.zeros(np.shape(r), dtype=complex)
    for (n, l, m, _), c in zip(states, coefs):
        psi += c * radial_function(n, l, r) * sph_harm_y(l, m, theta, phi)
    return np.abs(psi) ** 2


@dataclass
class BornSampler:
    """Draws electron positions from |psi|^2 by sampling cells of a spherical grid, then jittering inside the cell."""

    positions: np.ndarray
    mean_radius_exact: float

    @classmethod
    def build(cls, states: Sequence[State], coefs: Sequence[complex], rmax: float) -> "BornSampler":
        r_edges = np.linspace(0.0, rmax, BORN_RADIAL_CELLS + 1)
        c_edges = np.linspace(-1.0, 1.0, BORN_COS_CELLS + 1)
        p_edges = np.linspace(0.0, 2 * math.pi, BORN_PHI_CELLS + 1)
        r_mid, c_mid, p_mid = ((e[1:] + e[:-1]) / 2 for e in (r_edges, c_edges, p_edges))
        rr, cc, pp = np.meshgrid(r_mid, c_mid, p_mid, indexing="ij")
        density = hydrogenic_density(states, coefs, rr, np.arccos(cc), pp)
        weight = density * rr**2 * (r_edges[1] - r_edges[0]) * (c_edges[1] - c_edges[0]) * (p_edges[1] - p_edges[0])
        total = weight.sum()
        sampler = cls.__new__(cls)
        sampler._r_edges, sampler._c_edges, sampler._p_edges = r_edges, c_edges, p_edges
        sampler._prob = (weight / total).ravel()
        sampler._shape = weight.shape
        sampler.positions = np.zeros((0, 3))
        sampler.mean_radius_exact = float((weight * rr).sum() / total)
        sampler.norm_captured = float(total)
        return sampler

    def draw(self, count: int, rng: np.random.Generator) -> np.ndarray:
        cells = rng.choice(self._prob.size, size=count, p=self._prob)
        ir, ic, ip = np.unravel_index(cells, self._shape)
        jitter = rng.random((3, count))
        r = self._r_edges[ir] + jitter[0] * (self._r_edges[1] - self._r_edges[0])
        cos_t = self._c_edges[ic] + jitter[1] * (self._c_edges[1] - self._c_edges[0])
        phi = self._p_edges[ip] + jitter[2] * (self._p_edges[1] - self._p_edges[0])
        sin_t = np.sqrt(1 - cos_t**2)
        return np.stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], axis=1)


def circular_weights(n0: int, spread: float = RYDBERG_SPREAD, span: int = RYDBERG_SPAN) -> List[Tuple[int, float]]:
    """Gaussian weights over the shells n0-span..n0+span (n >= 2), normalised to sum(c^2) = 1."""
    shells = [n for n in range(n0 - span, n0 + span + 1) if n >= 2]
    raw = np.array([math.exp(-((n - n0) ** 2) / (4 * spread**2)) for n in shells])
    raw /= np.sqrt((raw**2).sum())
    return list(zip(shells, raw.tolist()))


def orbital_period_au(n0: int) -> float:
    """Classical Kepler period 2 pi n^3 (atomic units) of a circular orbit in shell n0."""
    return 2 * math.pi * n0**3


class RydbergPacket:
    """Coherent sum of circular states |n, l=n-1, m=n-1> with Gaussian weights around n0.

    The classical electron is a delta in time; here it is a wave packet whose centre moves at the Kepler frequency
    1/n0^3, spreads, and later revives. Energies are the exact -1/(2 n^2).
    """

    def __init__(self, n0: int = 12, grid: int = RYDBERG_GRID):
        self.n0 = n0
        self.grid = grid
        self.weights = circular_weights(n0)
        self.rmax = RYDBERG_RADIUS_FACTOR * (n0 + RYDBERG_RADIUS_SHELLS) ** 2
        axis = np.linspace(-self.rmax, self.rmax, grid, dtype=np.float32)
        xs, ys, zs = np.meshgrid(axis, axis, axis, indexing="ij")
        r = np.sqrt(xs**2 + ys**2 + zs**2)
        theta = np.arccos(np.clip(zs / np.maximum(r, 1e-6), -1, 1))
        phi = np.arctan2(ys, xs)
        self.psi = np.zeros((len(self.weights), grid, grid, grid), dtype=np.complex64)
        for k, (n, _) in enumerate(self.weights):
            condon_shortley = (-1.0) ** (n - 1)
            self.psi[k] = (condon_shortley * radial_function(n, n - 1, r) * sph_harm_y(n - 1, n - 1, theta, phi)).astype(np.complex64)
        self.energies = np.array([-0.5 / n**2 for n, _ in self.weights])
        self.coefs = np.array([c for _, c in self.weights])
        self.dv = float((2 * self.rmax / (grid - 1)) ** 3)

    @property
    def mean_radius(self) -> float:
        """<r> = n^2 + n/2 for a circular state of shell n0 (exact hydrogenic value)."""
        return self.n0**2 + 0.5 * self.n0

    @property
    def period(self) -> float:
        return orbital_period_au(self.n0)

    def amplitudes(self, t_au: float) -> np.ndarray:
        return (self.coefs * np.exp(-1j * (self.energies - self.energies[len(self.energies) // 2]) * t_au)).astype(np.complex64)

    def density(self, t_au: float) -> np.ndarray:
        field = np.tensordot(self.amplitudes(t_au), self.psi, axes=1)
        return (field.real**2 + field.imag**2).astype(np.float32)

    def centroid(self, t_au: float) -> np.ndarray:
        rho = self.density(t_au)
        axis = np.linspace(-self.rmax, self.rmax, self.grid)
        total = rho.sum()
        return np.array([
            (rho.sum(axis=(1, 2)) * axis).sum() / total,
            (rho.sum(axis=(0, 2)) * axis).sum() / total,
            (rho.sum(axis=(0, 1)) * axis).sum() / total,
        ])

    def norm(self, t_au: float) -> float:
        return float(self.density(t_au).sum() * self.dv)
