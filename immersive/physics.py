"""Physics kernels for the immersive viewer: hydrogenic tables, angular constants, qubit registers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import eval_genlaguerre, factorial, sph_harm_y

HARTREE_EV = 27.211386
RYDBERG_EV = HARTREE_EV / 2.0
MAX_STATES = 4
RADIAL_SAMPLES = 1024
ANGULAR_PROBE = (0.7, 0.4)

ANGULAR_POLY: Dict[Tuple[int, int], Callable[[float, complex], complex]] = {
    (0, 0): lambda c, w: 1.0,
    (1, 0): lambda c, w: c,
    (1, 1): lambda c, w: w,
    (2, 0): lambda c, w: 3 * c * c - 1,
    (2, 1): lambda c, w: c * w,
    (2, 2): lambda c, w: w * w,
    (3, 0): lambda c, w: c * (5 * c * c - 3),
    (3, 1): lambda c, w: (5 * c * c - 1) * w,
    (3, 2): lambda c, w: c * w * w,
    (3, 3): lambda c, w: w * w * w,
}


def unit_vector_poly_args(theta: float, phi: float) -> Tuple[float, complex]:
    """Return (cos(theta), sin(theta) e^{i phi}) used by the polynomial table."""
    return math.cos(theta), math.sin(theta) * complex(math.cos(phi), math.sin(phi))


def angular_constants() -> Dict[Tuple[int, int], float]:
    """Derive normalisation constants of Y_l^m (m >= 0) against scipy so the GLSL table cannot drift."""
    theta, phi = ANGULAR_PROBE
    c, w = unit_vector_poly_args(theta, phi)
    out = {}
    for (l, m), poly in ANGULAR_POLY.items():
        ratio = sph_harm_y(l, m, theta, phi) / poly(c, w)
        out[(l, m)] = float(ratio.real)
    return out


def ylm_from_table(l: int, m: int, theta: np.ndarray, phi: np.ndarray, consts) -> np.ndarray:
    """Evaluate Y_l^m through the same polynomial table the shader uses (for cross-checking)."""
    am = abs(m)
    c = np.cos(theta)
    w = np.sin(theta) * np.exp(1j * phi)
    y = consts[(l, am)] * ANGULAR_POLY[(l, am)](c, w)
    if m < 0:
        y = np.conj(y) * (-1.0) ** am
    return y


def radial_norm(n: int, l: int) -> float:
    return math.sqrt((2.0 / n) ** 3 * factorial(n - l - 1) / (2.0 * n * factorial(n + l)))


def radial_function(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """Hydrogenic R_nl(r) in Bohr radii."""
    rho = 2.0 * r / n
    return radial_norm(n, l) * np.exp(-rho / 2.0) * rho**l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


def radial_table(qn: Sequence[Tuple[int, int, int]], rmax: float, samples: int = RADIAL_SAMPLES) -> np.ndarray:
    r = np.linspace(0.0, rmax, samples)
    rows = [radial_function(n, l, r) for n, l, _ in qn] or [np.zeros(samples)]
    return np.asarray(rows, dtype=np.float32)


def view_radius(n_max: int) -> float:
    """Bounding-sphere radius (Bohr) that contains the visible cloud of shell n."""
    return max(6.0, 2.4 * n_max * n_max + 3.0 * n_max)


def energy_ev(n: int) -> float:
    return -RYDBERG_EV / (n * n)


def orbital_label(n: int, l: int, m: int) -> str:
    return f"{n}{'spdfg'[l]}  (m = {m:+d})"


def psi_samples(qn, coefs, r, theta, phi) -> np.ndarray:
    total = np.zeros(r.shape, dtype=complex)
    for (n, l, m), c in zip(qn, coefs):
        total += c * radial_function(n, l, r) * sph_harm_y(l, m, theta, phi)
    return total


def density_peak(qn, coefs, rmax: float, phases: int = 8, percentile: float = 99.7) -> float:
    """Exposure reference: a high percentile of |psi|^2 over the visible ball and a sweep of relative phases.

    A plain maximum is dominated by the nuclear cusp of s states and would leave extended lobes invisible.
    """
    rng = np.random.default_rng(7)
    count = 60000
    reach = rmax * 0.72
    r = reach * rng.random(count) ** (1.0 / 3.0)
    theta = np.arccos(rng.uniform(-1.0, 1.0, count))
    phi = rng.uniform(0.0, 2 * np.pi, count)
    energies = np.array([energy_ev(n) / HARTREE_EV for n, _, _ in qn])
    samples = []
    for k in range(phases):
        t = k * 2 * math.pi / phases / max(np.ptp(energies), 0.05)
        c = np.asarray(coefs) * np.exp(-1j * energies * t)
        samples.append(np.abs(psi_samples(qn, c, r, theta, phi)) ** 2)
    return max(float(np.percentile(np.concatenate(samples), percentile)), 1e-12)


def overlap_1s(bond: float) -> float:
    """Overlap integral of two 1s functions separated by `bond` Bohr."""
    return math.exp(-bond) * (1.0 + bond + bond * bond / 3.0)


@dataclass
class OrbitalPreset:
    key: str
    name_en: str
    name_es: str
    states: List[Tuple[int, int, int, complex]]
    note_en: str = ""
    note_es: str = ""


def _combo(n, l, m, sign_a=1.0, sign_b=-1.0, phase=1.0):
    k = 1 / math.sqrt(2)
    return [(n, l, -m, phase * sign_a * k), (n, l, m, phase * sign_b * k)]


PRESETS: List[OrbitalPreset] = [
    OrbitalPreset("1", "1s", "1s", [(1, 0, 0, 1)], "The ground state: a perfect sphere.", "El estado fundamental: una esfera perfecta."),
    OrbitalPreset("2", "2s", "2s", [(2, 0, 0, 1)], "A radial node: a hollow shell inside the cloud.", "Un nodo radial: una cascara hueca dentro de la nube."),
    OrbitalPreset("3", "2p z", "2p z", [(2, 1, 0, 1)], "Two lobes of opposite phase (blue / orange).", "Dos lobulos con fase opuesta (azul / naranja)."),
    OrbitalPreset("4", "2p x", "2p x", _combo(2, 1, 1), "Real orbital = superposition of m=+1 and m=-1.", "Orbital real = superposicion de m=+1 y m=-1."),
    OrbitalPreset("5", "3d z2", "3d z2", [(3, 2, 0, 1)], "Two lobes and a ring (torus).", "Dos lobulos y un anillo (toro)."),
    OrbitalPreset("6", "3d xz", "3d xz", _combo(3, 2, 1), "Four lobes between the axes.", "Cuatro lobulos entre los ejes."),
    OrbitalPreset("7", "3d x2-y2", "3d x2-y2", _combo(3, 2, 2, 1.0, 1.0), "Four lobes along the axes.", "Cuatro lobulos sobre los ejes."),
    OrbitalPreset("8", "4f z3", "4f z3", [(4, 3, 0, 1)], "Three nodal planes: the d's big sibling.", "Tres planos nodales: la hermana mayor de la d."),
    OrbitalPreset("9", "1s + 2p z (superposition)", "1s + 2p z (superposicion)",
                  [(1, 0, 0, 1 / math.sqrt(2)), (2, 1, 0, 1 / math.sqrt(2))],
                  "Beating phases: the electron sloshes up and down like a dipole.",
                  "Fases que laten: el electron oscila arriba y abajo como un dipolo."),
    OrbitalPreset("0", "2p +1 (rotating)", "2p +1 (rotante)", [(2, 1, 1, 1)],
                  "m=+1 carries angular momentum: watch the phase spin around z.",
                  "m=+1 tiene momento angular: mira la fase girar alrededor de z."),
]


class Gate:
    """A unitary exp(-i s H) with H Hermitian, so it can be animated for s in [0, 1]."""

    def __init__(self, name: str, hamiltonian: np.ndarray, qubits: Tuple[int, ...]):
        self.name = name
        self.qubits = qubits
        vals, vecs = np.linalg.eigh(hamiltonian)
        self._vals, self._vecs = vals, vecs

    def unitary(self, s: float) -> np.ndarray:
        return (self._vecs * np.exp(-1j * s * self._vals)) @ self._vecs.conj().T


I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = (PX, PY, PZ)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)
PMINUS = (I2 - PX) / 2

SINGLE_GATES = {
    "H": math.pi / 2 * (PX + PZ) / math.sqrt(2),
    "X": math.pi / 2 * PX,
    "Y": math.pi / 2 * PY,
    "Z": math.pi / 2 * PZ,
    "S": math.pi / 4 * PZ,
    "T": math.pi / 8 * PZ,
    "Rx": math.pi / 4 * PX,
    "Ry": math.pi / 4 * PY,
}


class QubitRegister:
    """Small state-vector simulator with animated gate application and reduced Bloch vectors."""

    def __init__(self, n_qubits: int, seconds_per_gate: float = 1.1):
        self.n = n_qubits
        self.duration = seconds_per_gate
        self.psi = np.zeros(2**n_qubits, dtype=complex)
        self.psi[0] = 1.0
        self._queue: List[Gate] = []
        self._active: Gate | None = None
        self._start = self.psi.copy()
        self._clock = 0.0
        self.history: List[str] = []

    def reset(self):
        self.psi[:] = 0
        self.psi[0] = 1.0
        self._queue.clear()
        self._active = None
        self.history.clear()

    def set_state(self, vec: np.ndarray):
        self.psi = np.asarray(vec, dtype=complex) / np.linalg.norm(vec)
        self._queue.clear()
        self._active = None

    def embed(self, op: np.ndarray, qubit: int) -> np.ndarray:
        mats = [I2] * self.n
        mats[qubit] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def single(self, name: str, qubit: int) -> Gate:
        g = Gate(f"{name}", self.embed(SINGLE_GATES[name], qubit), (qubit,))
        g.name = f"{name}({qubit})"
        return g

    def cnot(self, control: int, target: int) -> Gate:
        proj = self.embed(P1, control) @ self.embed(PMINUS, target)
        g = Gate(f"CNOT({control}>{target})", -math.pi * proj, (control, target))
        return g

    def queue(self, gate: Gate):
        self._queue.append(gate)

    @property
    def busy(self) -> bool:
        return self._active is not None or bool(self._queue)

    @property
    def progress(self) -> float:
        return 0.0 if self._active is None else min(1.0, self._clock / self.duration)

    @property
    def active_name(self) -> str:
        return "" if self._active is None else self._active.name

    def update(self, dt: float):
        if self._active is None and self._queue:
            self._active = self._queue.pop(0)
            self._start = self.psi.copy()
            self._clock = 0.0
        if self._active is None:
            return
        self._clock += dt
        s = min(1.0, self._clock / self.duration)
        eased = s * s * (3 - 2 * s)
        self.psi = self._active.unitary(eased) @ self._start
        if s >= 1.0:
            self.history.append(self._active.name)
            self.history = self.history[-8:]
            self._active = None

    def probabilities(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def reduced_density(self, qubit: int) -> np.ndarray:
        tensor = self.psi.reshape([2] * self.n)
        mat = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
        return mat @ mat.conj().T

    def bloch(self, qubit: int) -> np.ndarray:
        rho = self.reduced_density(qubit)
        return np.array([np.real(np.trace(rho @ p)) for p in PAULI])

    def entropy(self, qubit: int = 0) -> float:
        lam = np.clip(np.linalg.eigvalsh(self.reduced_density(qubit)), 1e-12, 1.0)
        return float(-np.sum(lam * np.log2(lam)))

    def measure(self, shots: int, rng: np.random.Generator | None = None) -> Dict[str, int]:
        rng = rng or np.random.default_rng()
        p = self.probabilities()
        p = p / p.sum()
        draws = rng.choice(len(p), size=shots, p=p)
        counts = np.bincount(draws, minlength=len(p))
        return {format(i, f"0{self.n}b"): int(c) for i, c in enumerate(counts)}
