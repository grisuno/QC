#!/usr/bin/env python3
"""
molecular_simulator.py - VERSIÓN CON OPENFERMION CORREGIDO
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

_LOG = _make_logger("MolecularSimulator")


# ---------------------------------------------------------------------------
# Molecular Data
# ---------------------------------------------------------------------------

@dataclass 
class MoleculeData:
    name: str; n_electrons: int; n_orbitals: int; n_qubits: int
    h_core: np.ndarray; eri: np.ndarray; nuclear_repulsion: float
    fci_energy: float; hf_energy: float; description: str = ""


def _h2_sto3g_pyscf():
    """H2 con datos PySCF."""
    try:
        from pyscf import gto, scf, fci as pyscf_fci, ao2mo
        
        mol = gto.M(
            atom='H 0 0 0; H 0 0 0.735',
            basis='sto-3g',
            unit='Angstrom',
            verbose=0
        )
        
        mf = scf.RHF(mol).run()
        cisolver = pyscf_fci.FCI(mol, mf.mo_coeff)
        e_fci, _ = cisolver.kernel()
        
        h_core = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri_chem = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
        
        _LOG.info("  Loaded H2 from PySCF: HF=%.8f, FCI=%.8f, E_nuc=%.6f", 
                  mf.e_tot, e_fci, mol.energy_nuc())
        
        return MoleculeData(
            "H2", 2, 2, 4,
            h_core, eri_chem, mol.energy_nuc(),
            e_fci, mf.e_tot,
            "H2 STO-3G (PySCF)"
        )
        
    except ImportError:
        _LOG.warning("  PySCF not available, using hardcoded data")
        return _h2_sto3g_hardcoded()


def _h2_sto3g_hardcoded():
    """H2 con datos hardcodeados (STO-3G @ 0.735 A)."""
    h_core = np.array([
        [-1.25246357, -0.47594872],
        [-0.47594872, -1.25246357]
    ])

    eri = np.zeros((2, 2, 2, 2))
    eri[0,0,0,0] = 0.67449314
    eri[1,1,1,1] = 0.67449314
    eri[0,0,1,1] = 0.18128754
    eri[1,1,0,0] = 0.18128754
    eri[0,1,0,1] = 0.66347211
    eri[1,0,1,0] = 0.66347211
    eri[0,1,1,0] = 0.18128754
    eri[1,0,0,1] = 0.18128754

    nuclear_repulsion = 0.71997

    # The HF/FCI labels used to be hard-coded to literature values, but
    # they did not actually match the integrals provided here (a 6 mHa
    # discrepancy that broke VQE convergence checks).  We now derive
    # them by diagonalising the JW Hamiltonian built from these
    # integrals -- honest ab initio numbers, consistent with the
    # integrals the VQE actually evaluates against.
    mol = MoleculeData(
        "H2", 2, 2, 4,
        h_core, eri, nuclear_repulsion,
        fci_energy=0.0, hf_energy=0.0,
        description="H2 STO-3G (hardcoded, energies from diagonalisation)",
    )
    n_q = 4
    dim = 1 << n_q
    H = np.zeros((dim, dim), dtype=np.float64)
    terms = _integral_jw_terms(mol)
    for k in range(dim):
        H[k, k] = nuclear_repulsion
    # Each ordered tuple (P,Q,R,S,sp,sq) in ``terms`` contributes (1/2)g·s
    # to H[k2,k] for T|k⟩ = s|k2⟩.  For unordered pairs the sum over
    # ordered tuples naturally contains both T and T† (e.g. (P,Q,R,S) and
    # (Q,P,S,R)) as separate entries, so we must NOT additionally add the
    # adjoint within the same iteration -- that would double-count the
    # diagonal and (wrongly) symmetrise off-diagonal that the other
    # ordered tuple already symmetrises.  Off-diagonal Hermitian symmetry
    # H[k,k2] = H[k2,k] is automatic once all ordered tuples are summed.
    for k in range(dim):
        for coeff, create, annihilate in terms:
            mapped = _jw_excitation_map(n_q, k, create, annihilate)
            if mapped is None:
                continue
            k2, sign = mapped
            v = coeff * sign
            H[k2, k] += v
    eigs = np.sort(np.linalg.eigvalsh(H))
    hf_energy = float(H[12, 12])  # HF state |1100> = idx 12
    fci_energy = float(eigs[0])
    mol.hf_energy = hf_energy
    mol.fci_energy = fci_energy
    return mol


class _LazyMoleculeRegistry(dict):
    """
    Lazy molecule registry: builds molecule data (which runs a PySCF
    calculation) on FIRST ACCESS instead of at import time.  Importing
    this module is now cheap and side-effect free.
    """

    def __init__(self, builders: Dict[str, object]) -> None:
        super().__init__()
        self._builders = dict(builders)
        for name in builders:
            super().__setitem__(name, None)

    def _resolve(self, name: str):
        try:
            value = super().__getitem__(name)
        except KeyError:
            raise KeyError(name) from None
        if value is None:
            value = self._builders[name]()
            super().__setitem__(name, value)
        return value

    def __getitem__(self, name: str):
        return self._resolve(name)

    def get(self, name: str, default=None):
        try:
            return self._resolve(name)
        except KeyError:
            return default


MOLECULES = _LazyMoleculeRegistry({"H2": _h2_sto3g_pyscf})


# ---------------------------------------------------------------------------
# OpenFermion Hamiltonian - CONSTRUCCIÓN CORREGIDA
# ---------------------------------------------------------------------------

def build_jw_hamiltonian_of(mol: MoleculeData):
    """
    Construye Hamiltoniano JW usando OpenFermion correctamente.
    Usa MolecularData de OpenFermion para asegurar consistencia.
    """
    try:
        from openfermion import MolecularData as OFMolecularData
        from openfermion.transforms import jordan_wigner
        from openfermion.linalg import get_sparse_operator
        from openfermionpyscf import run_pyscf
        
        # Crear molécula OpenFermion con geometría explícita
        geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.735))]
        of_mol = OFMolecularData(
            geometry=geometry,
            basis='sto-3g',
            charge=0,
            multiplicity=1,
            description='H2_0.735A'
        )
        
        # Correr PySCF a través de OpenFermion
        of_mol = run_pyscf(of_mol, run_fci=True, run_ccsd=False, run_cisd=False)
        
        _LOG.info("  OpenFermion-PySCF: HF=%.8f, FCI=%.8f", 
                  of_mol.hf_energy, of_mol.fci_energy)
        
        # Obtener Hamiltoniano molecular
        from openfermion import get_fermion_operator
        molecular_hamiltonian = of_mol.get_molecular_hamiltonian()
        fermion_op = get_fermion_operator(molecular_hamiltonian)
        
        # Transformación JW
        jw_op = jordan_wigner(fermion_op)
        
        # Convertir a nuestra representación
        paulis = []
        e_nuc = of_mol.nuclear_repulsion
        
        for term, coeff in jw_op.terms.items():
            if len(term) == 0:
                e_nuc = float(coeff.real)
            else:
                pauli_list = [(int(q), p) for q, p in sorted(term)]
                paulis.append((complex(coeff), pauli_list))
        
        # Verificación
        H_sparse = get_sparse_operator(jw_op)
        H_dense = H_sparse.toarray()
        eigs = np.linalg.eigvalsh(H_dense)
        
        _LOG.info("  OpenFermion verification:")
        _LOG.info("    Ground state: %.8f Ha (FCI target: %.8f)", eigs[0], mol.fci_energy)
        _LOG.info("    HF idx=3:  %.8f Ha", H_dense[3,3].real)
        _LOG.info("    HF idx=12: %.8f Ha", H_dense[12,12].real)
        
        # Verificar
        fci_ok = abs(eigs[0] - mol.fci_energy) < 0.01
        hf_ok = abs(H_dense[12,12].real - mol.hf_energy) < 0.01
        
        if fci_ok and hf_ok:
            _LOG.info("    ✓ OpenFermion Hamiltonian verified")
            return paulis, e_nuc, True, 12  # índice HF correcto
        else:
            _LOG.warning("    OpenFermion verification failed, using fallback")
            return None, mol.nuclear_repulsion, False, 3
            
    except Exception as e:
        _LOG.warning("  OpenFermion failed: %s", str(e))
        return None, mol.nuclear_repulsion, False, 3


# ---------------------------------------------------------------------------
# Exact JW Energy Evaluator
# ---------------------------------------------------------------------------

def _integral_jw_terms(mol: MoleculeData):
    """
    Construye los términos del Hamiltoniano molecular en codificación JW a
    partir de integrales ab initio h_core / eri (notación de químicos),
    SIN necesidad de OpenFermion.

    H = e_nuc + Σ_PQ sp h_core[P,Q] a†_{P,sp} a_{Q,sp}
              + (1/2) Σ_PQRS,sp,sq (PQ|RS) a†_{P,sp} a†_{R,sq} a_{S,sq} a_{Q,sp}

    Las integrales espaciales (PQ|RS) llevan un δ(sp_p, sp_r)δ(sp_q, sp_s)
    implícito en la integración de espín, así que (p,q) comparten espín
    ``sp`` y (r,s) comparten ``sq``.  El factor (1/2) compensa la
    simetría de intercambio (PQ|RS) = (RS|PQ) para no doble-contar.

    Cada término se devuelve como (coeficiente, create, annihilate) donde
    las listas se aplican en orden (el primer elemento actúa primero),
    listos para ``_jw_excitation_map``.
    """
    terms = []
    n_spatial = mol.n_orbitals
    h1 = np.asarray(mol.h_core, dtype=float)
    eri = np.asarray(mol.eri, dtype=float)

    # One-body: Σ_PQ sp h_core[P,Q] a†_{P,sp} a_{Q,sp}
    for P in range(n_spatial):
        for Q in range(n_spatial):
            if abs(h1[P, Q]) < 1e-14:
                continue
            for sp in range(2):
                p, q = 2 * P + sp, 2 * Q + sp
                terms.append((float(h1[P, Q]), [p], [q]))

    # Two-body: (1/2) Σ_PQRS,sp,sq (PQ|RS) a†_{P,sp} a†_{R,sq} a_{S,sq} a_{Q,sp}
    for P in range(n_spatial):
        for Q in range(n_spatial):
            for R in range(n_spatial):
                for S in range(n_spatial):
                    g = eri[P, Q, R, S]
                    if abs(g) < 1e-14:
                        continue
                    for sp in range(2):
                        for sq in range(2):
                            p, q = 2 * P + sp, 2 * Q + sp
                            r, s = 2 * R + sq, 2 * S + sq
                            if p == r or q == s:
                                # Pauli exclusion on the creation pair
                                # (a†_p a†_r = 0 if p == r) or annihilation
                                # pair (a_s a_q = 0 if q == s).
                                continue
                            terms.append((0.5 * float(g), [r, p], [q, s]))
    return terms


class ExactJWEnergy:
    """
    Evaluador exacto de energía en codificación JW.

    Usa OpenFermion cuando está disponible; si no, construye el
    Hamiltoniano desde las integrales h_core/eri de la molécula
    (ab initio, mismas matemáticas).  Solo lanza RuntimeError si no hay
    ninguna fuente de Hamiltoniano.
    """

    def __init__(self, mol: MoleculeData, n_qubits: int):
        self.mol = mol
        self.n_qubits = n_qubits
        self._integral_terms = None
        result = build_jw_hamiltonian_of(mol)
        self.paulis, self.e_nuc, self.of_ok, self.hf_idx = result
        if self.of_ok:
            _LOG.info("  Using OpenFermion JW: %d Pauli terms, E_nuc=%.6f Ha",
                      len(self.paulis), self.e_nuc)
        elif mol.h_core is not None and mol.eri is not None:
            self._integral_terms = _integral_jw_terms(mol)
            # HF state |11...10...0>: first n_electrons spin-orbitals occupied
            self.hf_idx = sum(1 << (n_qubits - 1 - i) for i in range(mol.n_electrons))
            _LOG.info("  Using integral JW Hamiltonian (no OpenFermion): "
                      "%d excitation terms, E_nuc=%.6f Ha",
                      len(self._integral_terms), self.e_nuc)
        else:
            raise RuntimeError(
                "Hamiltonian construction failed: install openfermion+openfermionpyscf "
                "or provide h_core/eri integrals in the MoleculeData"
            )
        self._verify_hf()

    @staticmethod
    def _to_scalar(amps: torch.Tensor) -> torch.Tensor:
        """
        (dim,2,G,G) → (dim,2)  ó  (dim,2) → (dim,2).
        Integra la función de onda espacial sobre la grilla manteniendo
        la estructura re/im que necesita el evaluador JW.
        """
        if amps.dim() == 2:
            return amps.double()
        return amps.sum(dim=(-2, -1)).double()   # ∫ α_k(x,y) dxdy

    def _verify_hf(self):
        dim = 2 ** self.n_qubits
        s = torch.zeros((dim, 2), dtype=torch.float64)
        s[self.hf_idx, 0] = 1.0
        e = self._evaluate(s)
        _LOG.info("  Verification: E_HF(calc)=%.8f Ha, E_HF(target)=%.8f Ha",
                  e, self.mol.hf_energy)
        if abs(e - self.mol.hf_energy) > 0.001:
            _LOG.error("  HF mismatch: %.4f Ha", abs(e - self.mol.hf_energy))

    def _apply(self, amps, pauli):
        dim = 2 ** self.n_qubits
        new = torch.zeros_like(amps)
        if not pauli:
            return amps.clone()
        for k in range(dim):
            cr, ci, kp = 1.0, 0.0, k
            for (qi, pc) in pauli:
                bp = self.n_qubits - 1 - qi
                bv = (k >> bp) & 1
                if   pc == "Z": s = 1 - 2*bv; cr *= s; ci *= s
                elif pc == "X": kp ^= (1 << bp)
                elif pc == "Y":
                    kp ^= (1 << bp)
                    if bv == 0: cr, ci = -ci,  cr
                    else:       cr, ci =  ci, -cr
            new[kp, 0] += cr * amps[k, 0] - ci * amps[k, 1]
            new[kp, 1] += cr * amps[k, 1] + ci * amps[k, 0]
        return new

    def _evaluate_integrals(self, amps: torch.Tensor) -> float:
        """Evaluate <psi|H|psi> applying the JW excitation terms directly."""
        n_q = self.n_qubits
        dim = 2 ** n_q
        ar = amps[:, 0].numpy()
        ai = amps[:, 1].numpy()
        out_np = np.zeros((dim, 2))
        for k in range(dim):
            if abs(ar[k]) < 1e-18 and abs(ai[k]) < 1e-18:
                continue
            for coeff, create, annihilate in self._integral_terms:
                mapped = _jw_excitation_map(n_q, k, create, annihilate)
                if mapped is None:
                    continue
                k2, sign = mapped
                v = coeff * sign
                # Forward term T|k> = sign|k2>: contributes to (Hψ)[k2]
                out_np[k2, 0] += v * ar[k]
                out_np[k2, 1] += v * ai[k]
                # Adjoint T†|k2> = sign|k>: contributes to (Hψ)[k]
                out_np[k, 0] += v * ar[k2]
                out_np[k, 1] += v * ai[k2]
        out = torch.from_numpy(out_np)
        ir = float((amps[:, 0] * out[:, 0] + amps[:, 1] * out[:, 1]).sum())
        return self.e_nuc + ir  # terms are real => expectation is real

    def _evaluate(self, amps: torch.Tensor) -> float:
        amps = self._to_scalar(amps)
        n = float((amps[:, 0]**2 + amps[:, 1]**2).sum())
        if n < 1e-15:
            return 0.0
        amps = amps / math.sqrt(n)
        if self._integral_terms is not None:
            return self._evaluate_integrals(amps)
        e = self.e_nuc
        for coeff, pauli in self.paulis:
            phi = self._apply(amps, pauli)
            ir = float((amps[:, 0]*phi[:, 0] + amps[:, 1]*phi[:, 1]).sum())
            ii = float((amps[:, 0]*phi[:, 1] - amps[:, 1]*phi[:, 0]).sum())
            e += coeff.real * ir - coeff.imag * ii
        return e

    def __call__(self, amps: torch.Tensor) -> float:
        return self._evaluate(amps)


# ---------------------------------------------------------------------------
# Surrogate Backend
# ---------------------------------------------------------------------------

class SurrogateEnergy:
    """Backend neuronal con calibración."""

    def __init__(self, mol, n_qubits, exact_eval, backend):
        self.mol = mol
        self.exact = exact_eval
        self.backend = backend
        self.hf_idx = exact_eval.hf_idx
        self.offset = None
        self.floor = mol.fci_energy - 0.002
        _LOG.info("  Surrogate: floor=%.6f Ha, HF_idx=%d", self.floor, self.hf_idx)

    def calibrate(self, hf_amps):
        raw = (self.backend.expectation_value(hf_amps)
               if hasattr(self.backend, 'expectation_value')
               else self.exact(hf_amps))
        exact = self.exact(hf_amps)
        self.offset = raw - exact
        _LOG.info("  Surrogate calibration:")
        _LOG.info("    Surrogate: %.8f Ha", raw)
        _LOG.info("    Exact:     %.8f Ha", exact)
        _LOG.info("    Target:    %.8f Ha", self.mol.hf_energy)
        _LOG.info("    Offset:    %.8f Ha", self.offset)
        return self.offset

    def cost_with_barrier(self, amps):
        raw = (self.backend.expectation_value(amps)
               if hasattr(self.backend, 'expectation_value')
               else self.exact(amps))
        e = raw - self.offset
        if e < self.floor:
            return self.floor + 100.0*(e - self.floor)**2, True
        return e, False


# ---------------------------------------------------------------------------
# UCCSD y VQE
# ---------------------------------------------------------------------------

def prepare_hf(mol, factory, backend):
    bits = ["0"] * mol.n_qubits
    for i in range(mol.n_electrons):
        bits[i] = "1"
    bs = "".join(bits)
    _LOG.info("  HF state: |%s> (%d e-, %d qubits)", bs, mol.n_electrons, mol.n_qubits)
    return factory.from_bitstring(bs)

def _sd_indices(n_e, n_q):
    occ = list(range(n_e))
    vir = list(range(n_e, n_q))
    singles = [(o, v) for o in occ for v in vir]
    doubles = [(o1, o2, v1, v2)
               for i, o1 in enumerate(occ) for o2 in occ[i+1:]
               for j, v1 in enumerate(vir) for v2 in vir[j+1:]]
    return singles, doubles

def _jw_excitation_map(n_qubits: int, k: int, create, annihilate):
    """
    Acción de un operador de excitación fermiónica sobre el estado base |k>
    en codificación Jordan-Wigner.  Devuelve (k', sign) o None si no aplica.

    Cada (anti)operador a_j aporta el signo de paridad JW
    (-1)^{suma de ocupaciones de qubits l < j} sobre el vector de bits
    actual.  Ver quantum_framework_molecular.UCCSDAnsatz._jw_excitation.
    """
    bits = [(k >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
    sign = 1.0
    for q in annihilate:
        if bits[q] != 1:
            return None
        if sum(bits[:q]) % 2 == 1:
            sign = -sign
        bits[q] = 0
    for q in create:
        if bits[q] != 0:
            return None
        if sum(bits[:q]) % 2 == 1:
            sign = -sign
        bits[q] = 1
    k_new = 0
    for q in range(n_qubits):
        if bits[q]:
            k_new |= 1 << (n_qubits - 1 - q)
    return k_new, sign


def _jw_givens_evolve(state, create, annihilate, theta: float):
    """
    Aplica exp(theta * (T - T†)) como rotación de Givens con signo JW
    directamente sobre las amplitudes del estado conjunto (exacto, unitario,
    conserva norma y número de partículas).
    """
    from quantum_computer import JointHilbertState

    n = state.n_qubits
    amps = state.amplitudes.clone()
    c = math.cos(theta)
    s = math.sin(theta)
    for k in range(state.dim):
        mapped = _jw_excitation_map(n, k, create, annihilate)
        if mapped is None:
            continue
        k_new, sign = mapped
        a_k = state.amplitudes[k]
        a_kp = state.amplitudes[k_new]
        amps[k] = c * a_k - sign * s * a_kp
        amps[k_new] = sign * s * a_k + c * a_kp
    return JointHilbertState(amps, n)


def uccsd(state, thetas, singles, doubles, backend, runner):
    """
    UCCSD ansatz (fermiónico, con signos Jordan-Wigner exactos).

    - Singles y doubles se aplican como rotaciones de Givens con el signo
      de paridad JW sobre las amplitudes del estado conjunto: exactas,
      unitarias, conservan número de partículas.
    - Reemplaza la implementación anterior: las singles usaban escaleras
      CNOT+Ry que no conservan número de partículas y las doubles
      ignoraban los índices de la excitación (pivote dinámico por argmax).
    - theta=0 para cualquier parámetro → identidad exacta.
    - ``backend`` y ``runner`` se conservan por compatibilidad de firma;
      la rotación directa sobre amplitudes no los necesita.
    """
    current = state
    ti = 0

    for (o, v) in singles:
        if ti >= len(thetas): break
        th = float(thetas[ti]); ti += 1
        if abs(th) < 1e-12: continue
        current = _jw_givens_evolve(current, create=[v], annihilate=[o], theta=th)

    for (o1, o2, v1, v2) in doubles:
        if ti >= len(thetas): break
        th = float(thetas[ti]); ti += 1
        if abs(th) < 1e-12: continue
        # Convención: T = a†_v1 a†_v2 a_o1 a_o2 (actúa primero el más a la derecha)
        current = _jw_givens_evolve(current, create=[v2, v1], annihilate=[o2, o1], theta=th)

    return current


@dataclass
class VQEResult:
    molecule: str; backend: str; n_qubits: int; n_parameters: int
    vqe_energy: float; hf_energy: float; fci_energy: float
    correlation_energy_captured: float; optimal_thetas: np.ndarray
    n_iterations: int; converged: bool; energy_error: float

    def __repr__(self):
        return "\n".join([
            "", "="*60, f"  VQE Result: {self.molecule}  [{self.backend}]",
            "="*60, f"  Qubits: {self.n_qubits}", f"  Parameters: {self.n_parameters}",
            "─"*60, f"  HF energy  : {self.hf_energy:+.8f} Ha",
            f"  VQE energy : {self.vqe_energy:+.8f} Ha",
            f"  FCI energy : {self.fci_energy:+.8f} Ha", "─"*60,
            f"  |VQE-FCI|  : {self.energy_error:.2e} Ha",
            f"  Correlation: {self.correlation_energy_captured*100:.1f}%",
            "="*60
        ])


class VQESolver:
    def __init__(self, qc, config):
        self.qc = qc
        self.config = config

    def _run(self, circ, be, state):
        from quantum_computer import _GATE_REGISTRY
        for inst in circ._instructions:
            gate = _GATE_REGISTRY.get(inst.gate_name)
            if gate is None: raise KeyError(f"Gate '{inst.gate_name}' not found")
            state = gate.apply(state, be, inst.targets, inst.params)
        return state

    def run(self, mol, backend="hamiltonian", max_iter=200, tol=1e-8):
        from scipy.optimize import minimize

        _LOG.info("Starting VQE for %s (%d qubits)", mol.name, mol.n_qubits)

        be = self.qc._backends[backend]
        hf_state = prepare_hf(mol, self.qc._factory, be)

        exact_eval = ExactJWEnergy(mol, mol.n_qubits)
        surrogate = SurrogateEnergy(mol, mol.n_qubits, exact_eval, be)
        surrogate.calibrate(hf_state.amplitudes)

        singles, doubles = _sd_indices(mol.n_electrons, mol.n_qubits)
        n_singles, n_doubles = len(singles), len(doubles)
        n_params = n_singles + n_doubles
        _LOG.info("  UCCSD: %d singles + %d doubles = %d parameters",
                  n_singles, n_doubles, n_params)

        # Verificación de identidad
        st0 = uccsd(hf_state.clone(), np.zeros(n_params), singles, doubles, be, self._run)
        e0 = exact_eval(st0.amplitudes)
        id_ok = abs(e0 - mol.hf_energy) < 1e-4
        _LOG.info("  [check] theta=0: E=%.8f Ha  %s",
                  e0, "✓ identity" if id_ok else f"✗ expected {mol.hf_energy:.8f}")
        if not id_ok:
            raise RuntimeError(f"Identity check failed: E={e0:.6f} != {mol.hf_energy:.6f}")

        # Scan del ángulo doble para inicialización
        _LOG.info("  Scanning double amplitude:")
        scan_min_e, scan_best_td = float('inf'), 0.0
        for td in np.linspace(-0.5, 0.5, 11):
            th = np.zeros(n_params)
            if n_doubles > 0: th[n_singles] = td
            st = uccsd(hf_state.clone(), th, singles, doubles, be, self._run)
            e_sc = exact_eval(st.amplitudes)
            marker = " ← best" if e_sc < scan_min_e else ""
            _LOG.info("    theta_d=%+.2f  E=%.8f Ha%s", td, e_sc, marker)
            if e_sc < scan_min_e:
                scan_min_e, scan_best_td = e_sc, td
        _LOG.info("  Scan: best td=%+.3f  E=%.8f  FCI=%.8f  gap=%.2e",
                  scan_best_td, scan_min_e, mol.fci_energy,
                  abs(scan_min_e - mol.fci_energy))

        theta0 = np.zeros(n_params)
        if n_doubles > 0: theta0[n_singles] = scan_best_td

        best_e, best_thetas, total_evals = float('inf'), None, 0

        def cost(thetas):
            nonlocal best_e, best_thetas, total_evals
            total_evals += 1
            state = uccsd(hf_state.clone(), thetas, singles, doubles, be, self._run)
            e = exact_eval(state.amplitudes)
            if e < best_e:
                best_e = e
                best_thetas = thetas.copy()
            if total_evals % 20 == 0 or total_evals <= 3:
                _LOG.info("  iter %3d: E=%.8f Ha  Δ_FCI=%.2e",
                          total_evals, e, abs(e - mol.fci_energy))
            return float(e)

        res = minimize(cost, theta0, method="L-BFGS-B",
                       options={"maxiter": max_iter, "ftol": tol,
                                "gtol": 1e-7, "eps": 1e-5})
        _LOG.info("  Optimizer: %s  (%d evals)", res.message, total_evals)

        final_thetas = best_thetas if best_thetas is not None else res.x
        final_state = uccsd(hf_state.clone(), final_thetas, singles, doubles, be, self._run)
        vqe_e = exact_eval(final_state.amplitudes)

        tot_corr = mol.hf_energy - mol.fci_energy
        pct = max(0.0, (mol.hf_energy - vqe_e) / tot_corr) if tot_corr > 1e-12 else 0.0
        _LOG.info("  Final: E_VQE=%.8f  E_FCI=%.8f  corr=%.1f%%",
                  vqe_e, mol.fci_energy, pct * 100)

        return VQEResult(
            mol.name, "openfermion_jw", mol.n_qubits, n_params, vqe_e,
            mol.hf_energy, mol.fci_energy, pct, final_thetas,
            total_evals, vqe_e < mol.hf_energy - 1e-6, abs(vqe_e - mol.fci_energy)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", default="H2", choices=list(MOLECULES.keys()))
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--hamiltonian-checkpoint", default="hamiltonian.pth")
    parser.add_argument("--schrodinger-checkpoint", 
                       default="checkpoint_phase3_training_epoch_18921_20260224_154739.pth")
    parser.add_argument("--dirac-checkpoint", default="best_dirac.pth")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--expansion-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from quantum_computer import QuantumComputer, SimulatorConfig
    
    cfg = SimulatorConfig(
        grid_size=args.grid_size, hidden_dim=args.hidden_dim, expansion_dim=args.expansion_dim,
        hamiltonian_checkpoint=args.hamiltonian_checkpoint,
        schrodinger_checkpoint=args.schrodinger_checkpoint,
        dirac_checkpoint=args.dirac_checkpoint,
        random_seed=args.seed,
        device="cpu",
    )
    
    qc = QuantumComputer(cfg)
    solver = VQESolver(qc, cfg)
    result = solver.run(MOLECULES[args.molecule], max_iter=args.max_iter)
    print(result)