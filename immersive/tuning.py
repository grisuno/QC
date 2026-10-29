"""Fine-tuning of the backend's own networks against the backend's exact operators, plus everything the 3D scene shows.

Three tasks, one per physics backend of quantum_computer.QuantumComputer. The network trained is a copy of the backend's
network (loaded from its checkpoint), the exact answer is the backend's analytical path:

    hamiltonian  net(psi) = H psi,  H = 0.5 k^2 (HamiltonianBackend._apply_h analytical branch), psi real
    schrodinger  net(psi) = psi(t + propagator_dt), exact spectral propagator exp(-i 0.5 k^2 dt), psi = (re, im)
    dirac        net(spinor) = spinor - i dt H_D spinor renormalised, H_D = DiracBackend._analytical_dirac, spinor packed
                 from the amplitude with DiracBackend._pack

A fixed training set (seen data) and held-out sets per data family (unseen data) are kept apart, so overfitting is visible.
No GL here: the scene reads the snapshots this module publishes.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import zoom

from .backend import ROOT, BackendHub

TRAIN_SIZE = 256
VAL_SIZE = 128
BATCH = 32
LEARNING_RATE = 1e-3
LR_FACTOR = 1.5
ACC_THRESHOLD = 0.01
SCALE_RANGE = (0.2, 1.0)
CUTOFF_RANGE = (1.0, math.pi)
PROJECT_TIME_MAX = 1.0
GRAD_HISTORY = 8
PUBLISH_EVERY = 10
CURVE_MAX = 600
PLATE_RES = 64
THREADS = 4
EIGENVALUE_TOL = 1e-10
ENTROPY_EPS = 1e-10
DELTA_FLOOR = 1e-8
ALPHA_FLOOR_VALUE = 20.0
FAMILIES = ("white", "smooth", "project")
DATA_MODES = FAMILIES + ("mixed",)
KINDS = ("hamiltonian", "schrodinger", "dirac")
KIND_LABEL = {"hamiltonian": "Hamiltonian", "schrodinger": "Schrodinger", "dirac": "Dirac"}
EXPORT_DIR = ROOT / "h" / "tuned"


def complex_norm(x: torch.Tensor) -> torch.Tensor:
    return x.abs().flatten(1).pow(2).sum(1).sqrt()


def make_fields(family: str, count: int, grid: int, base: torch.Tensor, exact_step: Callable, gen: torch.Generator) -> torch.Tensor:
    """Complex fields on the torus, (count, G, G), random norm in SCALE_RANGE.

    white   iid Gaussian noise, as in experiment2.py
    smooth  Gaussian noise low-passed at a random cutoff radius in CUTOFF_RANGE
    project the quantum computer's own basis amplitude, randomly rolled and phased, evolved by the exact propagator
    """
    if family == "mixed":
        parts = [make_fields(f, count // 3 + (1 if i < count % 3 else 0), grid, base, exact_step, gen) for i, f in enumerate(FAMILIES)]
        fields = torch.cat(parts)
        return fields[torch.randperm(len(fields), generator=gen)]
    if family == "project":
        out = torch.zeros(count, grid, grid, dtype=torch.complex64)
        for i in range(count):
            dx, dy = (int(v) for v in torch.randint(0, grid, (2,), generator=gen))
            phase = float(torch.rand(1, generator=gen)) * 2 * math.pi
            t = float(torch.rand(1, generator=gen)) * PROJECT_TIME_MAX
            out[i] = exact_step(torch.roll(base, (dx, dy), (0, 1)) * complex(math.cos(phase), math.sin(phase)), t)
        fields = out
    else:
        fields = torch.complex(torch.randn(count, grid, grid, generator=gen), torch.randn(count, grid, grid, generator=gen))
        if family == "smooth":
            k = torch.fft.fftfreq(grid, d=1.0) * 2 * math.pi
            kx, ky = torch.meshgrid(k, k, indexing="ij")
            cutoff = torch.empty(count, 1, 1).uniform_(*CUTOFF_RANGE, generator=gen)
            fields = torch.fft.ifft2(torch.fft.fft2(fields) * (torch.sqrt(kx**2 + ky**2)[None] < cutoff).float())
    fields = fields / complex_norm(fields).clamp_min(1e-12)[:, None, None]
    scale = torch.empty(count, 1, 1).uniform_(*SCALE_RANGE, generator=gen)
    return fields * scale


class OperatorTask:
    """Encoding, exact answer and baseline of one backend network."""

    def __init__(self, hub: BackendHub, kind: str):
        self.kind = kind
        self.hub = hub
        self.backend = hub.backend(kind)
        self.hamiltonian = hub.backend("hamiltonian")
        self.grid = hub.grid
        self.dt = float(getattr(hub.qc.config, "propagator_dt", 0.01))
        self.k2 = self.hamiltonian._k_squared.cpu()
        self.step_size = 1.0 if kind == "hamiltonian" else self.dt
        base = hub.new_state(1).amplitudes[0].detach().cpu()
        base = torch.complex(base[0], base[1])
        self.base_field = base / complex_norm(base[None])[0].clamp_min(1e-12)

    def free_step(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(psi) * torch.exp(-0.5j * self.k2 * t))

    def new_net(self) -> torch.nn.Module:
        net = self.backend.net
        if net is not None:
            copy_ = copy.deepcopy(net).cpu()
            for p in copy_.parameters():
                p.requires_grad_(True)
            return copy_
        import quantum_framework_physics as qfp

        cfg = self.hub.qc.config
        cls = {"hamiltonian": qfp.HamiltonianBackboneNet, "schrodinger": qfp.SchrodingerSpectralNet, "dirac": qfp.DiracSpectralNet}[self.kind]
        args = (cfg.grid_size, cfg.hidden_dim, cfg.num_spectral_layers) if self.kind == "hamiltonian" else (
            cfg.grid_size, cfg.hidden_dim, cfg.expansion_dim, cfg.num_spectral_layers)
        return cls(*args)

    def encode(self, psi: torch.Tensor) -> torch.Tensor:
        if self.kind == "hamiltonian":
            return psi.real.float()
        if self.kind == "schrodinger":
            return torch.stack([psi.real, psi.imag], dim=1).float()
        packed = [self.backend._pack(torch.stack([p.real, p.imag])).cpu() for p in psi]
        spinor = torch.stack(packed)
        return torch.cat([spinor.real, spinor.imag], dim=1).float()

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "hamiltonian":
            return torch.fft.ifft2(torch.fft.fft2(x) * 0.5 * self.k2).real
        if self.kind == "schrodinger":
            out = self.free_step(torch.complex(x[:, 0], x[:, 1]), self.dt)
            return torch.stack([out.real, out.imag], dim=1)
        spinor = torch.complex(x[:, :4].double(), x[:, 4:].double())
        out = torch.zeros_like(spinor)
        for i in range(len(spinor)):
            h = self.backend._analytical_dirac(spinor[i].to(self.backend.device)).cpu()
            stepped = spinor[i] - 1j * self.dt * h
            out[i] = stepped * (spinor[i].abs().pow(2).sum().sqrt() / stepped.abs().pow(2).sum().sqrt().clamp_min(1e-12))
        return torch.cat([out.real, out.imag], dim=1).float()

    def baseline(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x) if self.kind == "hamiltonian" else x

    def scalar(self, update: torch.Tensor) -> np.ndarray:
        """The (G, G) field the field plate draws from one sample's update (output - baseline)."""
        return (update if self.kind == "hamiltonian" else update[0]).detach().cpu().numpy()

    def probe(self) -> Tuple[torch.Tensor, np.ndarray]:
        """Unit-norm plane waves, one per torus wavevector (fftshift order), and the matching (m, n) grid."""
        g = self.grid
        axis = torch.arange(g, dtype=torch.float32)
        xs, ys = torch.meshgrid(axis, axis, indexing="ij")
        waves = []
        for m in range(-g // 2, g // 2):
            for n in range(-g // 2, g // 2):
                waves.append(torch.exp(1j * 2 * math.pi * (m * xs + n * ys) / g))
        psi = torch.stack(waves)
        psi = psi / complex_norm(psi).clamp_min(1e-12)[:, None, None]
        return psi, np.arange(-g // 2, g // 2)

    def rate(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        diff = outputs - self.baseline(inputs)
        return diff.flatten(1).norm(dim=1) / (self.step_size * inputs.flatten(1).norm(dim=1).clamp_min(1e-12))


def trig_interpolate(field: np.ndarray, size: int) -> np.ndarray:
    """Exact band-limited interpolation of a periodic field to size x size (FFT zero padding)."""
    n = field.shape[0]
    spectrum = np.fft.fftshift(np.fft.fft2(field))
    padded = np.zeros((size, size), dtype=complex)
    lo = (size - n) // 2
    padded[lo : lo + n, lo : lo + n] = spectrum
    return np.fft.ifft2(np.fft.ifftshift(padded)).real * (size / n) ** 2


def resample(field: np.ndarray, size: int, order: int = 3) -> np.ndarray:
    return zoom(field, (size / field.shape[0], size / field.shape[1]), order=order)


def kernels_of(net: torch.nn.Module) -> List[torch.Tensor]:
    """Channel-averaged complex spectral kernels, one (G//2+1, G) tensor per layer, as the backend applies them."""
    out = []
    for layer in net.spectral_layers:
        out.append(torch.complex(layer.kernel_real.detach().mean(dim=(0, 1)), layer.kernel_imag.detach().mean(dim=(0, 1))))
    return out


def boundary_winding(kernel: torch.Tensor) -> float:
    phase = torch.angle(kernel).numpy()
    loop = np.concatenate([phase[0, :], phase[1:-1, -1], phase[-1, ::-1], phase[-2:0:-1, 0]])
    steps = np.diff(np.append(loop, loop[0]))
    steps = (steps + math.pi) % (2 * math.pi) - math.pi
    return float(steps.sum() / (2 * math.pi))


def flat_kernels(net: torch.nn.Module) -> List[torch.Tensor]:
    return [torch.complex(l.kernel_real.detach(), l.kernel_imag.detach()).flatten() for l in net.spectral_layers]


def berry_step(prev: torch.Tensor, cur: torch.Tensor) -> float:
    a, b = prev / (prev.norm() + 1e-10), cur / (cur.norm() + 1e-10)
    overlap = (a.conj() * b).sum()
    return 0.0 if overlap.abs() < 1e-10 else float(torch.angle(overlap))


def grad_statistics(history: List[torch.Tensor]) -> Dict[str, float]:
    """kappa as in Maxwell (uncentred gram, eigenvalues > tol), T_eff = Tr(Sigma)/d (Book) and (lr/2) Var as in the code."""
    n = len(history)
    if n < 2:
        return {"kappa": float("inf"), "t_eff_book": 0.0, "t_eff_code_over_lr": 0.0}
    g = torch.stack(history)
    dim = g.shape[1]
    gram = torch.mm(g, g.t()) / max(n - 1, 1)
    eig = torch.linalg.eigvalsh(gram)
    eig = eig[eig > EIGENVALUE_TOL]
    kappa = float(eig.max() / eig.min()) if len(eig) else float("inf")
    centred = g - g.mean(dim=0, keepdim=True)
    trace = float(centred.pow(2).sum() / max(n - 1, 1))
    second = g.norm(dim=1).pow(2).mean()
    variance = float(second - g.mean(dim=0).norm() ** 2)
    return {"kappa": kappa, "t_eff_book": trace / dim, "t_eff_code_over_lr": 0.5 * variance}


@dataclass
class Snapshot:
    kind: str = ""
    ready: bool = False
    status: str = "preparing exact data"
    step: int = 0
    seed: int = 0
    lr: float = LEARNING_RATE
    loss_kind: str = "relative"
    data_mode: str = "mixed"
    running: bool = False
    applied: bool = False
    seen: Dict[str, float] = None
    unseen: Dict[str, Dict[str, float]] = None
    weights: Dict[str, float] = None
    plates: np.ndarray = None
    plate_scales: List[float] = None
    probe_family: str = ""
    probe_index: int = 0
    curves: List[Tuple[int, float, float, float]] = None
    note: str = ""
    version: int = 0


class TuneSession:
    """One fine-tuning run on one backend network. Training runs in a daemon thread and publishes Snapshots."""

    def __init__(self, kind: str, data_mode: str = "mixed", loss_kind: str = "relative", seed: Optional[int] = None):
        self.kind = kind
        self.data_mode = data_mode
        self.loss_kind = loss_kind
        self.seed = int(np.random.SeedSequence(seed).entropy % 2**32) if seed is None else int(seed)
        self.lr = LEARNING_RATE
        self.running = False
        self.alive = True
        self.probe_index = 0
        self._version = 0
        self.snapshot = Snapshot(kind=kind, data_mode=data_mode, loss_kind=loss_kind, seed=self.seed)
        self._lock = threading.Lock()
        self._commands: List[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request(self, command: str):
        with self._lock:
            self._commands.append(command)

    def read(self) -> Snapshot:
        with self._lock:
            return copy.copy(self.snapshot)

    def stop(self):
        self.alive = False

    def _publish(self, **fields):
        with self._lock:
            for key, value in fields.items():
                setattr(self.snapshot, key, value)

    def _generators(self):
        seeds = np.random.SeedSequence(self.seed).spawn(2 + len(FAMILIES))
        return [torch.Generator().manual_seed(int(s.generate_state(1)[0])) for s in seeds]

    def _build_data(self):
        gens = self._generators()
        task = self.task
        step = lambda psi, t: task.free_step(psi, t)
        train = make_fields(self.data_mode, TRAIN_SIZE, task.grid, task.base_field, step, gens[0])
        self.train_x = task.encode(train)
        self.train_y = task.exact(self.train_x)
        self.val = {}
        for family, gen in zip(FAMILIES, gens[2:]):
            x = task.encode(make_fields(family, VAL_SIZE, task.grid, task.base_field, step, gen))
            self.val[family] = (x, task.exact(x))
        self.shuffle_gen = gens[1]
        probe_psi, self.probe_axis = task.probe()
        self.probe_x = task.encode(probe_psi)
        self.probe_y = task.exact(self.probe_x)
        self.probe_exact_rate = task.rate(self.probe_x, self.probe_y).reshape(task.grid, task.grid).numpy()

    def _loss(self, out: torch.Tensor, target: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "mse":
            return ((out - target) ** 2).mean()
        num = (out - target).flatten(1).pow(2).sum(1)
        den = (target - base).flatten(1).pow(2).sum(1).clamp_min(1e-20)
        return (num / den).mean()

    @torch.no_grad()
    def _evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.model.eval()
        out = self.model(x)
        base = self.task.baseline(x)
        err = (out - y).flatten(1).norm(dim=1)
        scale = (y - base).flatten(1).norm(dim=1).clamp_min(1e-20)
        ratio = err / scale
        rel = err / y.flatten(1).norm(dim=1).clamp_min(1e-20)
        return {"err_baseline": float(ratio.mean()), "rel_error": float(rel.mean()), "accuracy": float((ratio < ACC_THRESHOLD).float().mean()),
                "mse": float(((out - y) ** 2).mean())}

    def _weight_stats(self) -> Dict[str, float]:
        params = [p.detach() for p in self.model.parameters() if p.numel()]
        delta = max(float((p - p.round()).abs().max()) for p in params)
        alpha = ALPHA_FLOOR_VALUE if delta < DELTA_FLOOR else -math.log(delta + ENTROPY_EPS)
        stats = {"delta": delta, "alpha": alpha, "berry_total": sum(self.berry), "winding_total": round(sum(self.berry) / (2 * math.pi)),
                 "step_change": self.weight_change}
        stats.update(grad_statistics(self.grads))
        for i, (b, k) in enumerate(zip(self.berry, kernels_of(self.model))):
            stats[f"berry_{i}"] = b
            stats[f"winding_{i}"] = round(b / (2 * math.pi))
            stats[f"boundary_winding_{i}"] = boundary_winding(k)
        return stats

    def _plates(self) -> Tuple[np.ndarray, List[float]]:
        task, res = self.task, PLATE_RES
        family = "project" if self.data_mode == "mixed" else self.data_mode
        x, y = self.val[family]
        index = self.probe_index % len(x)
        with torch.no_grad():
            self.model.eval()
            out = self.model(x[index : index + 1])[0]
            probe_out = self.model(self.probe_x)
        base = task.baseline(x[index : index + 1])[0]
        learned, exact = task.scalar(out - base), task.scalar(y[index] - base)
        plates = np.zeros((4, res, res, 4), np.float32)
        scales: List[float] = []

        def put(layer: int, solid: np.ndarray, ghost: np.ndarray, colour: np.ndarray):
            top = max(float(np.abs(solid).max()), float(np.abs(ghost).max()), 1e-30)
            plates[layer, ..., 0], plates[layer, ..., 1], plates[layer, ..., 2] = solid / top, ghost / top, colour
            scales.append(top)

        err_map = np.abs(learned - exact)
        put(0, trig_interpolate(learned, res), trig_interpolate(exact, res),
            np.clip(trig_interpolate(err_map, res) / max(float(np.abs(exact).max()), 1e-30), 0, 1))
        rate = task.rate(self.probe_x, probe_out).reshape(task.grid, task.grid).numpy()
        rate_s, rate_e = resample(rate, res), resample(self.probe_exact_rate, res)
        put(1, rate_s, rate_e, np.clip(np.abs(rate_s - rate_e) / max(float(np.abs(rate_e).max()), 1e-30), 0, 1))
        for layer, (cur, orig) in enumerate(zip(kernels_of(self.model), self.original_kernels)):
            put(2 + layer, resample(cur.abs().numpy(), res, 1), resample(orig.abs().numpy(), res, 1),
                resample(torch.angle(cur).numpy() / math.pi, res, 1))
        return plates, scales

    def _snapshot_all(self):
        seen = self._evaluate(self.train_x, self.train_y)
        unseen = {family: self._evaluate(*self.val[family]) for family in FAMILIES}
        plates, scales = self._plates()
        family = "project" if self.data_mode == "mixed" else self.data_mode
        self.curves.append((self.step, seen["err_baseline"], unseen[family]["err_baseline"],
                            float(np.mean([u["err_baseline"] for f, u in unseen.items() if f != family] or [0.0]))))
        if len(self.curves) > CURVE_MAX:
            self.curves = self.curves[::2]
        self._version += 1
        self._publish(version=self._version, seen=seen, unseen=unseen, weights=self._weight_stats(), plates=plates, plate_scales=scales, step=self.step,
                      lr=self.lr, loss_kind=self.loss_kind, data_mode=self.data_mode, running=self.running, probe_family=family,
                      probe_index=self.probe_index, curves=list(self.curves), applied=self.applied)

    def _track_berry(self):
        current = flat_kernels(self.model)
        for i, (p, c) in enumerate(zip(self.prev_flat, current)):
            self.berry[i] += berry_step(p, c)
        self.prev_flat = current

    def _handle(self, command: str):
        if command == "toggle":
            self.running = not self.running
        elif command == "faster":
            self.lr *= LR_FACTOR
        elif command == "slower":
            self.lr /= LR_FACTOR
        elif command == "loss":
            self.loss_kind = "mse" if self.loss_kind == "relative" else "relative"
        elif command == "probe+":
            self.probe_index += 1
        elif command == "probe-":
            self.probe_index -= 1
        elif command == "reset":
            self.model.load_state_dict(self.original_state)
            self._init_tracking()
            if self.applied:
                self.task.backend.net.load_state_dict(self.original_state)
                self.applied = False
            self.step = 0
            self.curves = []
            self.grads = []
            self._new_optimizer()
        elif command == "apply":
            if self.task.backend.net is None:
                self.task.backend.net = copy.deepcopy(self.model)
            else:
                self.task.backend.net.load_state_dict(self.model.state_dict())
            self.applied = True
        elif command == "export":
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            path = EXPORT_DIR / f"{self.kind}_tuned_step{self.step}_seed{self.seed}.pth"
            torch.save({"epoch": self.step, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": {"seen": self.snapshot.seen, "unseen": self.snapshot.unseen}, "lr": self.lr, "loss": self.loss_kind,
                        "data_mode": self.data_mode, "seed": self.seed}, path)
            self._publish(note=f"saved {path}")
        elif command.startswith("data:"):
            self.data_mode = command.split(":")[1]
            self.running = False
            self._publish(status="rebuilding exact data", ready=False)
            self._build_data()
            self.step = 0
            self.curves = []
        elif command == "newdata":
            self.seed = int(np.random.SeedSequence().entropy % 2**32)
            self.running = False
            self._publish(status="rebuilding exact data", ready=False, seed=self.seed)
            self._build_data()
            self.step = 0
            self.curves = []

    def _new_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _init_tracking(self):
        self.prev_flat = flat_kernels(self.model)
        self.berry = [0.0] * len(self.prev_flat)
        self.weight_change = 0.0

    def _run(self):
        try:
            torch.set_num_threads(THREADS)
            hub = BackendHub.shared()
            self.task = OperatorTask(hub, self.kind)
            self.model = self.task.new_net()
            self.original_state = copy.deepcopy(self.model.state_dict())
            self.original_kernels = kernels_of(self.model)
            self.applied = False
            self.curves: List[Tuple[int, float, float, float]] = []
            self.grads: List[torch.Tensor] = []
            self.step = 0
            self._init_tracking()
            self._new_optimizer()
            self._build_data()
            self._snapshot_all()
            self._publish(ready=True, status="paused")
            order = torch.randperm(TRAIN_SIZE, generator=self.shuffle_gen)
            cursor = 0
            while self.alive:
                with self._lock:
                    pending, self._commands = self._commands, []
                for command in pending:
                    self._handle(command)
                    order, cursor = torch.randperm(TRAIN_SIZE, generator=self.shuffle_gen), 0
                    self._snapshot_all()
                    self._publish(ready=True)
                if not self.running:
                    self._publish(status="paused", running=False)
                    time.sleep(0.03)
                    continue
                if cursor + BATCH > TRAIN_SIZE:
                    order, cursor = torch.randperm(TRAIN_SIZE, generator=self.shuffle_gen), 0
                idx = order[cursor : cursor + BATCH]
                cursor += BATCH
                self.model.train()
                for group in self.optimizer.param_groups:
                    group["lr"] = self.lr
                x, y = self.train_x[idx], self.train_y[idx]
                self.optimizer.zero_grad()
                loss = self._loss(self.model(x), y, self.task.baseline(x))
                loss.backward()
                self.grads.append(torch.cat([p.grad.detach().flatten() for p in self.model.parameters() if p.grad is not None]).clone())
                self.grads = self.grads[-GRAD_HISTORY:]
                before = torch.cat([p.detach().flatten() for p in self.model.parameters()])
                self.optimizer.step()
                self.weight_change = float((torch.cat([p.detach().flatten() for p in self.model.parameters()]) - before).norm())
                self.step += 1
                if self.step % PUBLISH_EVERY == 0:
                    self._track_berry()
                    self._publish(status="tuning", running=True)
                    self._snapshot_all()
        except Exception as exc:
            self._publish(status=f"error: {type(exc).__name__}: {str(exc)[:80]}", ready=False)
