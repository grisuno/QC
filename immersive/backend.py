"""Bridge between the viewer and the project's own simulator (quantum_computer.QuantumComputer).

Every state shown in the qubit scenes lives in a JointHilbertState (2^n amplitude fields on a GxG grid).
Gates are animated as partial unitaries applied to that tensor and committed with the project's gate registry,
so the picture is always a view of the project's computation. An independent ideal register runs in lockstep to
expose any deviation between the two.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .physics import PAULI, PMINUS, P1, SINGLE_GATES, Gate, QubitRegister

ROOT = Path(__file__).resolve().parents[1]
COMMIT_PARAMS = {"Rx": {"theta": math.pi / 2}, "Ry": {"theta": math.pi / 2}}
BACKEND_LABELS = {
    "schrodinger": "Schrodinger NN",
    "hamiltonian": "Hamiltonian NN",
    "dirac": "Dirac NN",
}


class BackendHub:
    """Loads the project's QuantumComputer once and exposes its physics backends."""

    ORDER = ("schrodinger", "hamiltonian", "dirac")
    _shared: "BackendHub | None" = None
    _lock = threading.Lock()

    @classmethod
    def shared(cls) -> "BackendHub":
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    @classmethod
    def preload(cls):
        threading.Thread(target=cls.shared, daemon=True).start()

    def __init__(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import quantum_computer as qcm

        logging.getLogger("QuantumPhysics").setLevel(logging.ERROR)

        self.qcm = qcm
        cfg = qcm.SimulatorConfig(
            device="cpu",
            hamiltonian_checkpoint=os.environ.get("Q2C_HAMILTONIAN", str(ROOT / "hamiltonian.pth")),
            schrodinger_checkpoint=str(ROOT / "checkpoint_phase3_training_epoch_18921_20260224_154739.pth"),
            dirac_checkpoint=str(ROOT / "best_dirac.pth"),
        )
        self.qc = qcm.QuantumComputer(cfg)
        self.grid = cfg.grid_size
        self.loaded: Dict[str, bool] = {n: getattr(self.qc._backends[n], "net", None) is not None for n in self.ORDER}

    def backend(self, name: str):
        return self.qc._backends[name]

    def new_state(self, n_qubits: int):
        return self.qc._factory.all_zeros(n_qubits)


def _torch():
    import torch

    return torch


def to_complex(state) -> np.ndarray:
    amps = state.amplitudes.detach().cpu().numpy()
    return amps[:, 0] + 1j * amps[:, 1]


class BackendRegister:
    """Animated n-qubit register whose state is a JointHilbertState from the project's simulator."""

    def __init__(self, n_qubits: int, backend: str = "schrodinger", seconds_per_gate: float = 1.1):
        self.hub = BackendHub.shared()
        self.n = n_qubits
        self.duration = seconds_per_gate
        self.backend_name = backend
        self.committed = self.hub.new_state(n_qubits)
        self.view = self.committed
        self.ideal = QubitRegister(n_qubits, seconds_per_gate)
        field = to_complex(self.committed)[0]
        self.reference = field / (np.linalg.norm(field) + 1e-12)
        self._queue: List[Tuple] = []
        self._active: Tuple | None = None
        self._clock = 0.0
        self.history: List[str] = []
        self.live_evolve = False
        self.error = ""
        self.evolve_steps = 0

    @property
    def be(self):
        return self.hub.backend(self.backend_name)

    def cycle_backend(self):
        order = self.hub.ORDER
        self.backend_name = order[(order.index(self.backend_name) + 1) % len(order)]
        self.error = ""

    def reset(self):
        self.committed = self.hub.new_state(self.n)
        self.view = self.committed
        self.ideal.reset()
        self._queue.clear()
        self._active = None
        self.history.clear()
        self.evolve_steps = 0

    def prepare(self, kind: str):
        self.reset()
        gates = {"0": [], "1": [("single", "X", 0)], "+": [("single", "H", 0)]}[kind]
        for spec in gates:
            self.committed = self._commit(self.committed, spec)
            self.ideal.queue(self.ideal.single(spec[1], spec[2]))
        for _ in range(int(self.duration * 100) + 5):
            self.ideal.update(0.01)
        self.view = self.committed

    def single(self, name: str, qubit: int):
        return ("single", name, qubit)

    def cnot(self, control: int, target: int):
        return ("cnot", control, target)

    def queue(self, spec: Tuple):
        self._queue.append(spec)
        if spec[0] == "single":
            self.ideal.queue(self.ideal.single(spec[1], spec[2]))
        else:
            self.ideal.queue(self.ideal.cnot(spec[1], spec[2]))

    @property
    def busy(self) -> bool:
        return self._active is not None or bool(self._queue)

    @property
    def progress(self) -> float:
        return 0.0 if self._active is None else min(1.0, self._clock / self.duration)

    @property
    def active_name(self) -> str:
        if self._active is None:
            return ""
        return f"{self._active[1]}({self._active[2]})" if self._active[0] == "single" else f"CNOT({self._active[1]}>{self._active[2]})"

    def _commit(self, state, spec: Tuple):
        gates = self.hub.qcm._GATE_REGISTRY
        if spec[0] == "single":
            return gates[spec[1]].apply(state, self.be, [spec[2]], COMMIT_PARAMS.get(spec[1]))
        return gates["CNOT"].apply(state, self.be, [spec[1], spec[2]], None)

    def _partial(self, state, spec: Tuple, s: float):
        torch = _torch()
        qcm = self.hub.qcm
        if spec[0] == "single":
            u = Gate(spec[1], SINGLE_GATES[spec[1]], (0,)).unitary(s)
            return qcm._single_qubit_unitary(state, spec[2], torch.tensor(u, dtype=torch.complex64), self.be)
        gen = -math.pi * np.kron(P1, PMINUS)
        u4 = Gate("CNOT", gen, (0, 1)).unitary(s)
        return qcm._two_qubit_unitary(state, spec[1], spec[2], torch.tensor(u4, dtype=torch.complex64))

    def update(self, dt: float):
        self.ideal.update(dt)
        if self._active is None and self._queue:
            self._active = self._queue.pop(0)
            self._clock = 0.0
        if self._active is not None:
            self._clock += dt
            s = min(1.0, self._clock / self.duration)
            eased = s * s * (3 - 2 * s)
            self.view = self._partial(self.committed, self._active, eased)
            if s >= 1.0:
                self.committed = self._commit(self.committed, self._active)
                self.view = self.committed
                self.history.append(self.active_name)
                self.history = self.history[-8:]
                self._active = None
            return
        if self.live_evolve:
            self.evolve(1)

    def evolve(self, steps: int = 1, dt: float | None = None):
        """Advance every spatial amplitude field with the selected physics backend (EvolveGate); dt=None uses the backend config dt."""
        try:
            gate = self.hub.qcm._GATE_REGISTRY["Evolve"]
            self.committed = gate.apply(self.committed, self.be, [0], {"dt": dt, "steps": steps})
            self.view = self.committed
            self.evolve_steps += steps
            self.error = ""
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}"
            self.live_evolve = False

    def reduced_density(self, qubit: int) -> np.ndarray:
        a = to_complex(self.view)
        tensor = a.reshape([2] * self.n + [-1])
        mat = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
        rho = mat @ mat.conj().T
        return rho / np.trace(rho).real

    def bloch(self, qubit: int) -> np.ndarray:
        rho = self.reduced_density(qubit)
        return np.array([np.real(np.trace(rho @ p)) for p in PAULI])

    def backend_bloch(self, qubit: int) -> np.ndarray:
        return np.array(self.view.bloch_vector(qubit))

    def entropy(self, qubit: int = 0) -> float:
        lam = np.clip(np.linalg.eigvalsh(self.reduced_density(qubit)), 1e-12, 1.0)
        return float(-np.sum(lam * np.log2(lam)))

    def probabilities(self) -> np.ndarray:
        return self.view.probabilities().detach().cpu().numpy().astype(float)

    @property
    def psi(self) -> np.ndarray:
        a = to_complex(self.view)
        return np.array([np.vdot(self.reference, a[k]) for k in range(a.shape[0])])

    def fields(self) -> np.ndarray:
        """Complex amplitude fields, shape (2^n, G, G)."""
        return to_complex(self.view)

    def deviation(self) -> float:
        """Largest gap between the project's Bloch vectors and the ideal-unitary reference."""
        return max(float(np.linalg.norm(self.bloch(q) - self.ideal.bloch(q))) for q in range(self.n))

    def hamiltonian_expectation(self) -> Dict[str, float]:
        """<H> of the register state with the project's HamiltonianBackend operator, as the backend applies it (per real channel).

        Each basis amplitude is a 2-channel field (re, im) on the grid; H acts on each channel through
        HamiltonianBackend._apply_h. The analytic free-particle -laplacian/2 on the same fields is returned for reference.
        """
        torch = _torch()
        backend = self.hub.backend("hamiltonian")
        amps = self.view.amplitudes.detach().cpu()
        probs = self.view.probabilities().detach().cpu()
        grid = self.hub.grid
        e_net = e_ref = 0.0
        for k in range(amps.shape[0]):
            psi_r, psi_i = amps[k, 0].reshape(grid, grid), amps[k, 1].reshape(grid, grid)
            norm = float((psi_r**2 + psi_i**2).sum())
            if norm < 1e-12:
                continue
            h_r, h_i = backend._apply_h(psi_r).reshape(grid, grid), backend._apply_h(psi_i).reshape(grid, grid)
            fft_r, fft_i = torch.fft.fft2(psi_r), torch.fft.fft2(psi_i)
            ref_r = torch.fft.ifft2(fft_r * 0.5 * backend._k_squared).real
            ref_i = torch.fft.ifft2(fft_i * 0.5 * backend._k_squared).real
            weight = float(probs[k])
            e_net += weight * float((psi_r * h_r + psi_i * h_i).sum()) / norm
            e_ref += weight * float((psi_r * ref_r + psi_i * ref_i).sum()) / norm
        return {"network": e_net, "reference": e_ref}

    def measure(self, shots: int, rng: np.random.Generator | None = None) -> Dict[str, int]:
        rng = rng or np.random.default_rng()
        p = self.probabilities()
        p = p / p.sum()
        draws = rng.choice(len(p), size=shots, p=p)
        counts = np.bincount(draws, minlength=len(p))
        return {format(i, f"0{self.n}b"): int(c) for i, c in enumerate(counts)}

    def describe(self) -> Dict[str, object]:
        return {
            "label": BACKEND_LABELS[self.backend_name],
            "loaded": self.hub.loaded[self.backend_name],
            "deviation": self.deviation(),
            "error": self.error,
            "grid": self.hub.grid,
            "evolve_steps": self.evolve_steps,
        }


_vqe_cache: Dict[str, object] = {}


def run_project_vqe() -> Dict[str, object]:
    """Run the project's own live H2 VQE (MPSQuantumComputer + Jordan-Wigner Hamiltonian, 2-qubit STO-3G model)."""
    if _vqe_cache:
        return dict(_vqe_cache)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from unittest import mock

    import quantum_lab
    from quantum_framework_core import FrameworkConfig, MPSQuantumComputer
    from quantum_framework_molecular import MoleculeBuilder

    two_qubit_model = mock.patch.object(MoleculeBuilder, "h2_sto3g", staticmethod(lambda bond_length=0.735: MoleculeBuilder._h2_hardcoded(bond_length)))
    with two_qubit_model:
        engine = quantum_lab.H2VQEEngine(MPSQuantumComputer(FrameworkConfig()))
    trace = list(engine.optimize(theta0=0.3))
    iterations, theta, energy = trace[-1]
    _vqe_cache.update({
        "engine": "MPSQuantumComputer", "model": "H2 STO-3G, 2-qubit JW model", "iterations": iterations, "theta": theta,
        "e_vqe": energy, "e_hf": engine.e_hf, "e_fci": engine.e_fci, "error": energy - engine.e_fci,
        "history": [e for _, _, e in trace],
    })
    return dict(_vqe_cache)


HCHECK_SIGMA = 2.0
HCHECK_STEPS = 200
HCHECK_DT = 0.01


def hamiltonian_check() -> Dict[str, float]:
    """Audit the project's Hamiltonian network against exact free-particle quantum mechanics on its own grid.

    Reports (1) the Hermiticity defect of the learned operator (a true Hamiltonian has zero),
    (2) the relative distance of H(psi) from the exact -laplacian/2 on a Gaussian probe,
    (3) the fidelity |<exact|nn>|^2 after HCHECK_STEPS Euler steps of the project's own HamiltonianBackend
    versus the exact spectral propagator, and (4) the fidelity of doing nothing, the baseline the network must beat.
    """
    torch = _torch()
    hub = BackendHub.shared()
    backend = hub.backend("hamiltonian")
    grid = hub.grid
    axis = torch.arange(grid, dtype=torch.float32) - grid / 2
    xs, ys = torch.meshgrid(axis, axis, indexing="ij")
    probe = torch.exp(-(xs**2 + ys**2) / (2 * HCHECK_SIGMA**2))
    gen = torch.Generator().manual_seed(0)
    a, b = torch.randn(grid, grid, generator=gen), torch.randn(grid, grid, generator=gen)
    h_a, h_b = backend._apply_h(a).reshape(grid, grid), backend._apply_h(b).reshape(grid, grid)
    hermiticity = float(((b * h_a).sum() - (a * h_b).sum()) / (torch.norm(a) * torch.norm(h_a) + 1e-12))
    exact = torch.fft.ifft2(torch.fft.fft2(probe) * (0.5 * backend._k_squared)).real
    h_probe = backend._apply_h(probe).reshape(grid, grid)
    kinetic_gap = float(torch.norm(h_probe - exact) / (torch.norm(exact) + 1e-12))
    amp = torch.stack([probe, torch.zeros_like(probe)])
    evolved, reference = amp.clone(), torch.complex(probe, torch.zeros_like(probe))
    phase = torch.exp(-0.5j * backend._k_squared.to(reference.dtype) * HCHECK_DT)
    for _ in range(HCHECK_STEPS):
        evolved = backend.evolve_amplitude(evolved, HCHECK_DT)
        reference = torch.fft.ifft2(torch.fft.fft2(reference) * phase)
    state = torch.complex(evolved[0], evolved[1])
    overlap = (reference.conj() * state).sum()
    fidelity = float(overlap.abs() ** 2 / ((reference.abs() ** 2).sum() * (state.abs() ** 2).sum()))
    start = torch.complex(probe, torch.zeros_like(probe))
    baseline = float((reference.conj() * start).sum().abs() ** 2 / ((reference.abs() ** 2).sum() * (start.abs() ** 2).sum()))
    return {"hermiticity": hermiticity, "kinetic_gap": kinetic_gap, "fidelity": fidelity, "idle_fidelity": baseline,
            "trained": float(backend.net is not None), "steps": float(HCHECK_STEPS)}
