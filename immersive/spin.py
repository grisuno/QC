"""Pauli-spinor hydrogen states |n l j m_j>: exact angular coupling, fine structure from the project's Dirac backend."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import sph_harm_y

from .backend import ROOT
from .physics import HARTREE_EV, radial_function, view_radius

AU_TIME_SECONDS = 2.4188843265857e-17
HARTREE_HZ = 6.579683920502e15
TESLA_AU = 2.35051756758e5
BOHR_MAGNETON_AU = 0.5
G_ELECTRON = 2.00231930436
LATTICE = 31


def clebsch(l: int, mj2: int, sign: int) -> Tuple[float, float]:
    """Coefficients (c_up, c_down) of |l, s=1/2; j=l+sign/2, m_j> on (Y_l^{mj-1/2} up, Y_l^{mj+1/2} down)."""
    mj = 0.5 * mj2
    denom = 2 * l + 1
    if sign > 0:
        return math.sqrt(max((l + mj + 0.5) / denom, 0.0)), math.sqrt(max((l - mj + 0.5) / denom, 0.0))
    return -math.sqrt(max((l - mj + 0.5) / denom, 0.0)), math.sqrt(max((l + mj + 0.5) / denom, 0.0))


def lande_g(l: int, sign: int) -> float:
    j = l + 0.5 * sign
    if j <= 0:
        return 0.0
    s = 0.5
    return 1.0 + (j * (j + 1) + s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))


@dataclass
class SpinorComponent:
    n: int
    l: int
    sign: int
    mj2: int
    coef: complex

    @property
    def j(self) -> float:
        return self.l + 0.5 * self.sign

    @property
    def kappa(self) -> int:
        return -(self.l + 1) if self.sign > 0 else self.l

    @property
    def label(self) -> str:
        j2 = 2 * self.l + self.sign
        return f"{self.n}{'spdfg'[self.l]}{j2}/2"

    def valid(self) -> bool:
        return self.l >= 0 and (self.sign > 0 or self.l > 0) and abs(self.mj2) <= 2 * self.l + self.sign


class DiracBackendLink:
    """Fine-structure energies from relativistic_hydrogen.DiracHydrogenAtom (shimmed for SciPy >= 1.15)."""

    def __init__(self):
        self.error = ""
        self.atom = None
        try:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            import scipy.special as sp

            if not hasattr(sp, "sph_harm"):
                sp.sph_harm = lambda m, n, az, pol: sp.sph_harm_y(n, m, pol, az)
            import logging

            import relativistic_hydrogen as rh

            logging.getLogger().setLevel(logging.ERROR)
            self.atom = rh.DiracHydrogenAtom(rh.Config)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {str(exc)[:80]}"

    @property
    def ok(self) -> bool:
        return self.atom is not None

    def energy(self, comp: SpinorComponent) -> float:
        if self.atom is None:
            return -0.5 / comp.n**2
        return float(self.atom.energy_level_dirac(comp.n, comp.kappa))


@dataclass
class SpinPreset:
    key: str
    name_en: str
    name_es: str
    note_en: str
    note_es: str
    components: List[SpinorComponent]
    field_tesla: float = 0.0
    nuance: str = ""


def _c(n, l, sign, mj2, coef=1.0):
    return SpinorComponent(n, l, sign, mj2, complex(coef))


R2 = 1 / math.sqrt(2)
SPIN_PRESETS: List[SpinPreset] = [
    SpinPreset("1", "1s  spin up", "1s  spin arriba",
               "m_j=+1/2: every arrow points up, the same everywhere.", "m_j=+1/2: todas las flechas apuntan arriba, igual en todo el espacio.",
               [_c(1, 0, 1, 1)]),
    SpinPreset("2", "1s  spin (up+down)/sqrt2 in a 5 T field: Larmor precession",
               "1s  spin (arriba+abajo)/raiz2 en 5 T: precesion de Larmor",
               "A spin pointing sideways precesses around B at omega = g mu_B B / hbar.",
               "Un spin horizontal precesa alrededor de B con omega = g mu_B B / hbar.",
               [_c(1, 0, 1, 1, R2), _c(1, 0, 1, -1, R2)], 5.0),
    SpinPreset("3", "2p  spin-orbit beating: 2p1/2 + 2p3/2 (m_j=+1/2)", "2p  batido spin-orbita: 2p1/2 + 2p3/2 (m_j=+1/2)",
               "Fine structure: the spin precesses relative to the orbital motion at the splitting frequency (Dirac backend).",
               "Estructura fina: el spin precesa respecto al movimiento orbital a la frecuencia del desdoblamiento (backend de Dirac).",
               [_c(2, 1, -1, 1, R2), _c(2, 1, 1, 1, R2)]),
    SpinPreset("4", "2p3/2  m_j=+3/2: spin and orbit aligned", "2p3/2  m_j=+3/2: spin y orbita alineados",
               "A pure |m_l=1, up> ring: all arrows up while the phase circulates.", "Un anillo |m_l=1, arriba> puro: todas las flechas arriba mientras la fase circula.",
               [_c(2, 1, 1, 3)]),
    SpinPreset("5", "2p3/2  m_j=+1/2: a spin texture", "2p3/2  m_j=+1/2: una textura de spin",
               "Spin and orbit are entangled: the arrow direction changes from place to place.",
               "Spin y orbita estan entrelazados: la direccion de la flecha cambia de lugar en lugar.",
               [_c(2, 1, 1, 1)]),
    SpinPreset("6", "2p1/2  m_j=+1/2: the other fine-structure level", "2p1/2  m_j=+1/2: el otro nivel de estructura fina",
               "Same orbital shape, opposite spin-orbit coupling: j = l - 1/2.", "Misma forma orbital, acoplamiento spin-orbita opuesto: j = l - 1/2.",
               [_c(2, 1, -1, 1)]),
]


def component_energy(comp: SpinorComponent, link: DiracBackendLink, field_tesla: float) -> float:
    zeeman = lande_g(comp.l, comp.sign) * BOHR_MAGNETON_AU * (field_tesla / TESLA_AU) * 0.5 * comp.mj2
    return link.energy(comp) + zeeman


def time_scale(components: Sequence[SpinorComponent], link: DiracBackendLink, field_tesla: float, beat_seconds: float = 6.0) -> float:
    """Atomic-time units advanced per screen second, chosen so the dominant beat lasts `beat_seconds` on screen."""
    energies = [component_energy(c, link, field_tesla) for c in components]
    spread = max(energies) - min(energies)
    if spread < 1e-12:
        return 1.0
    return 2 * math.pi / beat_seconds / spread


def real_time_label(au_per_screen_second: float) -> str:
    seconds = au_per_screen_second * AU_TIME_SECONDS
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9), ("ps", 1e-12), ("fs", 1e-15), ("as", 1e-18)):
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds:.3g} s"


def evolved_coefficients(components: Sequence[SpinorComponent], energies: Sequence[float], t_au: float) -> np.ndarray:
    e0 = energies[0]
    return np.array([c.coef * np.exp(-1j * (e - e0) * t_au) for c, e in zip(components, energies)])


def spinor_field(components: Sequence[SpinorComponent], coefs: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two-component spinor (up, down) at cartesian `points` (N, 3)."""
    r = np.linalg.norm(points, axis=1)
    safe = np.where(r > 1e-9, r, 1e-9)
    theta = np.arccos(np.clip(points[:, 2] / safe, -1, 1))
    phi = np.arctan2(points[:, 1], points[:, 0])
    up = np.zeros(len(points), dtype=complex)
    dn = np.zeros(len(points), dtype=complex)
    for comp, coef in zip(components, coefs):
        radial = radial_function(comp.n, comp.l, r)
        cu, cd = clebsch(comp.l, comp.mj2, comp.sign)
        mu, md = (comp.mj2 - 1) // 2, (comp.mj2 + 1) // 2
        if abs(mu) <= comp.l:
            up += coef * cu * radial * sph_harm_y(comp.l, mu, theta, phi)
        if abs(md) <= comp.l:
            dn += coef * cd * radial * sph_harm_y(comp.l, md, theta, phi)
    return up, dn


@dataclass
class SpinMeasurement:
    sigma: np.ndarray
    norm: float
    arrows_pos: np.ndarray
    arrows_dir: np.ndarray
    arrows_weight: np.ndarray


def measure(components: Sequence[SpinorComponent], coefs: np.ndarray, reach: float, arrows: int = 40) -> SpinMeasurement:
    axis = np.linspace(-reach, reach, LATTICE)
    xs, ys, zs = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    up, dn = spinor_field(components, coefs, pts)
    rho = np.abs(up) ** 2 + np.abs(dn) ** 2
    dv = (2 * reach / (LATTICE - 1)) ** 3
    norm = float(rho.sum() * dv)
    cross = np.conj(up) * dn
    sigma = np.array([2 * np.real(cross).sum(), 2 * np.imag(cross).sum(), (np.abs(up) ** 2 - np.abs(dn) ** 2).sum()]) * dv
    sub = np.zeros((LATTICE,) * 3, dtype=bool)
    sub[1::3, 1::3, 1::3] = True
    sel = np.nonzero(sub.ravel())[0]
    top = sel[np.argsort(rho[sel])[::-1][:arrows]]
    local = np.stack([2 * np.real(cross[top]), 2 * np.imag(cross[top]), np.abs(up[top]) ** 2 - np.abs(dn[top]) ** 2], axis=1)
    local /= np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-12)
    weight = rho[top] / max(rho[top].max(), 1e-12)
    return SpinMeasurement(sigma / max(norm, 1e-12), norm, pts[top], local, weight)


def exposure(components: Sequence[SpinorComponent], energies: Sequence[float], rmax: float, percentile: float = 99.7) -> float:
    """1 / percentile of the spinor density over the visible ball, over several beat phases."""
    rng = np.random.default_rng(11)
    count = 40000
    reach = rmax * 0.72
    r = reach * rng.random(count) ** (1.0 / 3.0)
    cos_t = rng.uniform(-1, 1, count)
    sin_t = np.sqrt(1 - cos_t**2)
    phi = rng.uniform(0, 2 * np.pi, count)
    pts = np.stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], axis=1)
    spread = max(max(energies) - min(energies), 1e-9)
    samples = []
    for k in range(6):
        t = k * 2 * math.pi / 6 / spread if len(components) > 1 else 0.0
        up, dn = spinor_field(components, evolved_coefficients(components, energies, t), pts)
        samples.append(np.abs(up) ** 2 + np.abs(dn) ** 2)
    return 1.0 / max(float(np.percentile(np.concatenate(samples), percentile)), 1e-12)


_link: "DiracBackendLink | None" = None


def shared_link() -> DiracBackendLink:
    global _link
    if _link is None:
        _link = DiracBackendLink()
    return _link
