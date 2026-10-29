"""Ab-initio fields for atoms and molecules (PySCF) sampled on a cubic grid for GPU volume rendering."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
HARTREE_EV = 27.211386245988
GRID = 88
MARGIN = 4.2
BASIS = "6-31g"
BOND_TOLERANCE = 1.25

COVALENT_RADII = {"H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58}
CPK = {
    "H": (0.95, 0.95, 0.98), "He": (0.75, 1.0, 1.0), "Li": (0.75, 0.50, 1.0), "Be": (0.45, 0.90, 0.30),
    "B": (1.0, 0.70, 0.70), "C": (0.40, 0.42, 0.46), "N": (0.25, 0.40, 1.0), "O": (1.0, 0.18, 0.18),
    "F": (0.55, 0.95, 0.30), "Ne": (0.55, 0.90, 0.95),
}
DISPLAY_RADIUS = {"H": 0.32, "He": 0.30, "Li": 0.55, "Be": 0.50, "B": 0.48, "C": 0.46, "N": 0.44, "O": 0.42, "F": 0.40, "Ne": 0.38}
UNPAIRED = {"H": 1, "He": 0, "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1, "Ne": 0}


@dataclass(frozen=True)
class System:
    key: str
    title: str
    atoms: Tuple[Tuple[str, Tuple[float, float, float]], ...]
    spin: int = 0
    note_en: str = ""
    note_es: str = ""


def _mol(key, title, atoms, spin=0, en="", es=""):
    return System(key, title, tuple(atoms), spin, en, es)


MOLECULES: List[System] = [
    _mol("H2", "H2  hydrogen", [("H", (0, 0, -0.3675)), ("H", (0, 0, 0.3675))], 0,
         "The simplest bond: two electrons shared in one sigma orbital.", "El enlace mas simple: dos electrones compartidos en un orbital sigma."),
    _mol("LiH", "LiH  lithium hydride", [("Li", (0, 0, -0.40)), ("H", (0, 0, 1.195))], 0,
         "Polar bond: the cloud leans toward hydrogen.", "Enlace polar: la nube se inclina hacia el hidrogeno."),
    _mol("HF", "HF  hydrogen fluoride", [("F", (0, 0, 0.0)), ("H", (0, 0, 0.917))], 0,
         "Fluorine hoards the electrons: a strong dipole.", "El fluor acapara los electrones: un dipolo fuerte."),
    _mol("H2O", "H2O  water", [("O", (0, 0, 0.1173)), ("H", (0, 0.7572, -0.4692)), ("H", (0, -0.7572, -0.4692))], 0,
         "Bent shape and two lone pairs: why water is a solvent of life.", "Forma angular y dos pares libres: por que el agua es el solvente de la vida."),
    _mol("NH3", "NH3  ammonia", [("N", (0, 0, 0.1173)), ("H", (0, 0.9377, -0.2737)), ("H", (0.8121, -0.4689, -0.2737)),
                                  ("H", (-0.8121, -0.4689, -0.2737))], 0, "A pyramid with a lone pair on top.", "Una piramide con un par libre arriba."),
    _mol("CH4", "CH4  methane", [("C", (0, 0, 0)), ("H", (0.629, 0.629, 0.629)), ("H", (-0.629, -0.629, 0.629)),
                                  ("H", (-0.629, 0.629, -0.629)), ("H", (0.629, -0.629, -0.629))], 0,
         "Perfect tetrahedron: four equivalent bonds.", "Tetraedro perfecto: cuatro enlaces equivalentes."),
    _mol("N2", "N2  nitrogen", [("N", (0, 0, -0.549)), ("N", (0, 0, 0.549))], 0,
         "A triple bond: one sigma and two pi orbitals.", "Un triple enlace: un sigma y dos pi."),
    _mol("CO", "CO  carbon monoxide", [("C", (0, 0, -0.6)), ("O", (0, 0, 0.528))], 0,
         "Isoelectronic with N2 but lopsided.", "Isoelectronico con N2 pero asimetrico."),
    _mol("O2", "O2  oxygen (triplet)", [("O", (0, 0, -0.604)), ("O", (0, 0, 0.604))], 2,
         "Two unpaired electrons: spin density explains why liquid oxygen is magnetic.",
         "Dos electrones desapareados: la densidad de spin explica por que el oxigeno liquido es magnetico."),
    _mol("C2H4", "C2H4  ethylene", [("C", (0, 0, 0.6695)), ("C", (0, 0, -0.6695)), ("H", (0, 0.9289, 1.2321)), ("H", (0, -0.9289, 1.2321)),
                                     ("H", (0, 0.9289, -1.2321)), ("H", (0, -0.9289, -1.2321))], 0,
         "The pi bond above and below the molecular plane.", "El enlace pi arriba y abajo del plano molecular."),
]
ATOM_SYSTEMS: List[System] = [
    _mol(sym, f"{sym}  atom", [(sym, (0, 0, 0))], UNPAIRED[sym],
         f"Neutral {sym} (UHF, {UNPAIRED[sym]} unpaired electron(s)).", f"{sym} neutro (UHF, {UNPAIRED[sym]} electron(es) desapareado(s)).")
    for sym in ("H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne")
]
LIBRARY: List[System] = MOLECULES + ATOM_SYSTEMS


@dataclass
class OrbitalInfo:
    index: int
    energy_ev: float
    occupation: float
    spin_label: str = ""


@dataclass
class Solved:
    system: System
    mol: object
    mf: object
    unrestricted: bool
    e_total: float
    converged: bool
    orbitals: List[OrbitalInfo]
    atoms_bohr: np.ndarray
    symbols: List[str]
    half: float
    fci_energy: Optional[float] = None
    fci_rdm: Optional[np.ndarray] = None
    homo: int = 0
    fields: Dict[str, np.ndarray] = field(default_factory=dict)


def pyscf_available() -> bool:
    try:
        import pyscf  # noqa: F401
        return True
    except ImportError:
        return False


def solve(system: System) -> Solved:
    from pyscf import fci, gto, scf

    atom_spec = "; ".join(f"{s} {x} {y} {z}" for s, (x, y, z) in system.atoms)
    mol = gto.M(atom=atom_spec, basis=BASIS, spin=system.spin, unit="Angstrom", verbose=0)
    unrestricted = system.spin > 0
    mf = (scf.UHF(mol) if unrestricted else scf.RHF(mol))
    mf.max_cycle = 200
    mf.kernel()
    if unrestricted:
        occ = mf.mo_occ
        energies = mf.mo_energy
        orbitals = [OrbitalInfo(i, float(energies[0][i]) * HARTREE_EV, float(occ[0][i]), "alpha") for i in range(mol.nao)]
        homo = int(np.max(np.nonzero(occ[0] > 0)[0]))
    else:
        orbitals = [OrbitalInfo(i, float(mf.mo_energy[i]) * HARTREE_EV, float(mf.mo_occ[i])) for i in range(mol.nao)]
        homo = int(np.max(np.nonzero(mf.mo_occ > 0)[0]))
    coords = mol.atom_coords()
    half = float(np.max(np.linalg.norm(coords, axis=1)) + MARGIN) if len(coords) > 1 else MARGIN + 1.5
    solved = Solved(system, mol, mf, unrestricted, float(mf.e_tot), bool(mf.converged), orbitals, coords,
                    [a[0] for a in system.atoms], half, homo=homo)
    if system.key == "H2":
        cis = fci.FCI(mol, mf.mo_coeff)
        e_fci, civec = cis.kernel()
        solved.fci_energy = float(e_fci)
        solved.fci_rdm = np.asarray(cis.make_rdm1(civec, mol.nao, mol.nelec))
    return solved


def _ao_values(solved: Solved, n: int = GRID):
    from pyscf.dft import numint

    half = solved.half
    axis = np.linspace(-half, half, n)
    xs, ys, zs = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    return coords, lambda chunk: numint.eval_ao(solved.mol, chunk, deriv=0)


def _evaluate(solved: Solved, quadratic) -> np.ndarray:
    """Evaluate sum_ab ao_a ao_b D_ab (density-like) chunk-wise; `quadratic` returns the AO-basis matrix."""
    coords, ao_of = _ao_values(solved)
    out = np.empty(len(coords))
    chunk = 120_000
    for start in range(0, len(coords), chunk):
        ao = ao_of(coords[start:start + chunk])
        out[start:start + chunk] = quadratic(ao)
    return out.reshape((GRID,) * 3)


def field_amplitude(solved: Solved, kind: str) -> np.ndarray:
    """Signed amplitude a(r) with density a^2: sqrt(rho), signed sqrt(spin density), or an orbital."""
    key = kind
    if key in solved.fields:
        return solved.fields[key]
    mf = solved.mf
    if kind in ("density", "spin", "fci"):
        if kind == "fci":
            dm = solved.fci_rdm
            dm_ao = mf.mo_coeff @ dm @ mf.mo_coeff.T
            rho = _evaluate(solved, lambda ao: np.einsum("pa,ab,pb->p", ao, dm_ao, ao))
            amp = np.sqrt(np.clip(rho, 0, None))
        elif solved.unrestricted:
            dm_a, dm_b = mf.make_rdm1()
            if kind == "density":
                rho = _evaluate(solved, lambda ao: np.einsum("pa,ab,pb->p", ao, dm_a + dm_b, ao))
                amp = np.sqrt(np.clip(rho, 0, None))
            else:
                rho = _evaluate(solved, lambda ao: np.einsum("pa,ab,pb->p", ao, dm_a - dm_b, ao))
                amp = np.sign(rho) * np.sqrt(np.abs(rho))
        else:
            dm = mf.make_rdm1()
            rho = _evaluate(solved, lambda ao: np.einsum("pa,ab,pb->p", ao, dm, ao))
            amp = np.sqrt(np.clip(rho, 0, None))
    else:
        index = int(kind.split(":")[1])
        coeff = mf.mo_coeff[0][:, index] if solved.unrestricted else mf.mo_coeff[:, index]
        amp = _evaluate(solved, lambda ao: ao @ coeff)
    solved.fields[key] = amp.astype(np.float32)
    return solved.fields[key]


def exposure(amp: np.ndarray, half: float) -> float:
    """1 / (99.7th percentile of a^2 inside the bounding sphere)."""
    axis = np.linspace(-half, half, amp.shape[0])
    xs, ys, zs = np.meshgrid(axis, axis, axis, indexing="ij")
    inside = (xs**2 + ys**2 + zs**2) < half * half
    values = amp[inside] ** 2
    return float(1.0 / max(np.percentile(values, 99.7), 1e-12))


def bonds(symbols: List[str], coords: np.ndarray) -> List[Tuple[int, int]]:
    out = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            limit = BOND_TOLERANCE * (COVALENT_RADII[symbols[i]] + COVALENT_RADII[symbols[j]]) * BOHR_PER_ANGSTROM
            if np.linalg.norm(coords[i] - coords[j]) < limit:
                out.append((i, j))
    return out


class AsyncSolver:
    """Runs PySCF off the render thread and caches finished systems."""

    def __init__(self):
        self.cache: Dict[str, Solved] = {}
        self.jobs: Dict[str, threading.Thread] = {}
        self.errors: Dict[str, str] = {}
        self.fields_ready: Dict[Tuple[str, str], bool] = {}

    def request(self, system: System):
        if system.key in self.cache or system.key in self.jobs:
            return

        def work():
            try:
                self.cache[system.key] = solve(system)
            except Exception as exc:
                self.errors[system.key] = f"{type(exc).__name__}: {exc}"
            finally:
                self.jobs.pop(system.key, None)

        thread = threading.Thread(target=work, daemon=True)
        self.jobs[system.key] = thread
        thread.start()

    def prefetch(self, system: System, kinds: Tuple[str, ...] = ("density",)):
        """Solve `system` and sample the given fields in one background job."""
        token = (system.key, "prefetch")
        if token in self.jobs:
            return

        def work():
            try:
                if system.key not in self.cache:
                    self.cache[system.key] = solve(system)
                solved = self.cache[system.key]
                for kind in kinds:
                    if kind == "spin" and not solved.unrestricted:
                        continue
                    field_amplitude(solved, kind)
                    self.fields_ready[(system.key, kind)] = True
            except Exception as exc:
                self.errors[system.key] = f"{type(exc).__name__}: {exc}"
            finally:
                self.jobs.pop(token, None)

        thread = threading.Thread(target=work, daemon=True)
        self.jobs[token] = thread
        thread.start()

    def request_field(self, system: System, kind: str):
        token = (system.key, kind)
        if token in self.fields_ready or token in self.jobs:
            return
        solved = self.cache.get(system.key)
        if solved is None:
            return

        def work():
            try:
                field_amplitude(solved, kind)
                self.fields_ready[token] = True
            except Exception as exc:
                self.errors[f"{system.key}/{kind}"] = f"{type(exc).__name__}: {exc}"
            finally:
                self.jobs.pop(token, None)

        thread = threading.Thread(target=work, daemon=True)
        self.jobs[token] = thread
        thread.start()
