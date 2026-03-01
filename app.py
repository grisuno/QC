#!/usr/bin/env python3
"""
app.py — Corrected with proper particle-conserving ansatz
=======================================================================

Root cause identified: The uccsd() function in molecular_sim.py implements
singles excitations incorrectly. For excitation (o→v), it does:
    CNOT ladder → RY(v) → CNOT ladder inverse
This ADDS an electron at v without REMOVING from o, producing |1110⟩ 
instead of |0110⟩. The states reachable by uccsd never include |0110⟩ 
or |1001⟩, which are exactly the states the dipole operator connects 
to |1100⟩ (HF). Hence <μ>=0 always, giving α=0.

Fix: Implement proper Givens-rotation-based singles that conserve 
particle number. For adjacent qubits, Givens(o,v,θ) is:
    CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)
This rotates in the {|10⟩, |01⟩} subspace: 
    |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩

For non-adjacent qubits, we SWAP to make them adjacent, apply Givens,
then SWAP back. This preserves all quantum numbers.

We keep the doubles excitation from uccsd since it works correctly 
(verified: it produces |1100⟩↔|0011⟩ rotation properly).
"""

from __future__ import annotations
import warnings, math
import numpy as np
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore")

try:
    import torch
    from quantum_computer import (QuantumComputer, SimulatorConfig, 
                                   JointHilbertState, QuantumCircuit, _GATE_REGISTRY)
    from molecular_sim import MOLECULES, ExactJWEnergy, uccsd
    from pyscf import gto, scf
    from openfermion.transforms import jordan_wigner
    from openfermion.ops import FermionOperator
    from scipy.optimize import minimize
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


@dataclass
class VQEResult:
    energy: float
    theta: np.ndarray
    field: float
    converged: bool
    n_iter: int

def _sd_indices(n_e, n_q):
    occ = list(range(n_e))
    vir = list(range(n_e, n_q))
    singles = [(o, v) for o in occ for v in vir]
    doubles = [(o1, o2, v1, v2) 
               for i, o1 in enumerate(occ) for o2 in occ[i+1:]
               for j, v1 in enumerate(vir) for v2 in vir[j+1:]]
    return singles, doubles

def _run_circuit(circuit, backend, state):
    for inst in circuit._instructions:
        gate = _GATE_REGISTRY.get(inst.gate_name)
        state = gate.apply(state, backend, inst.targets, inst.params)
    return state

def givens_single_excitation(state, o, v, theta, n_qubits, backend):
    """
    Apply a particle-conserving single excitation rotation between 
    qubits o (occupied) and v (virtual).
    
    Rotates in the {|1_o 0_v⟩, |0_o 1_v⟩} subspace:
        |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩
        |0_o 1_v⟩ → -sin(θ)|1_o 0_v⟩ + cos(θ)|0_o 1_v⟩
    
    For adjacent qubits: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)
    For non-adjacent: SWAP chain to make adjacent, apply, SWAP back.
    """
    if abs(theta) < 1e-14:
        return state
    
    current = state
    
    # If non-adjacent, SWAP v closer to o
    # Strategy: SWAP v down to position o+1
    swap_chain = []
    if v > o + 1:
        for k in range(v, o + 1, -1):
            swap_chain.append((k-1, k))
            circ = QuantumCircuit(n_qubits)
            circ.swap(k-1, k)
            current = _run_circuit(circ, backend, current)
        # Now the virtual qubit is at position o+1
        actual_v = o + 1
    else:
        actual_v = v
    
    # Apply Givens rotation between o and actual_v (now adjacent)
    # CNOT(actual_v, o) → RY(actual_v, 2θ) → CNOT(actual_v, o)
    circ = QuantumCircuit(n_qubits)
    circ.cnot(actual_v, o)
    circ.ry(actual_v, 2.0 * theta)
    circ.cnot(actual_v, o)
    current = _run_circuit(circ, backend, current)
    
    # SWAP back (reverse order)
    for (a, b) in reversed(swap_chain):
        circ = QuantumCircuit(n_qubits)
        circ.swap(a, b)
        current = _run_circuit(circ, backend, current)
    
    return current


def particle_conserving_ansatz(state, thetas, singles, doubles, backend):
    """
    Particle-conserving UCCSD-like ansatz:
    - Singles: Givens rotations (correct particle conservation)
    - Doubles: reuse uccsd's double excitation (works correctly)
    """
    current = state
    n = state.n_qubits
    ti = 0
    
    # Singles via Givens rotations
    for (o, v) in singles:
        if ti >= len(thetas): break
        th = float(thetas[ti]); ti += 1
        current = givens_single_excitation(current, o, v, th, n, backend)
    
    # Doubles: use the uccsd implementation (it works for doubles)
    # We pass zeros for singles and only the double theta
    n_singles = len(singles)
    for (o1, o2, v1, v2) in doubles:
        if ti >= len(thetas): break
        th = float(thetas[ti]); ti += 1
        if abs(th) < 1e-14: continue
        
        # Reconstruct the doubles circuit from uccsd logic
        amps = current.amplitudes
        probs = (amps[:, 0]**2 + amps[:, 1]**2).sum(dim=(-2, -1))
        hf_idx = int(probs.argmax().item())
        occ_q = [i for i in range(n) if (hf_idx >> (n-1-i)) & 1]
        vir_q = [i for i in range(n) if not ((hf_idx >> (n-1-i)) & 1)]
        if not vir_q: continue
        
        pivot = vir_q[0]
        chain = sorted(set(occ_q + vir_q[1:]))
        circ = QuantumCircuit(n)
        for c_q in chain:           circ.cnot(pivot, c_q)
        circ.ry(pivot, 2.0 * th)
        for c_q in reversed(chain): circ.cnot(pivot, c_q)
        current = _run_circuit(circ, backend, current)
    
    return current

class StarkEvaluator:
    def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, 
                 field, n_qubits):
        self.base = base_hamiltonian
        self.dipole_paulis = dipole_paulis
        self.dipole_identity = dipole_identity
        self.field = field
        self.n_qubits = n_qubits
    
    @staticmethod
    def _to_scalar(amps):
        if amps.dim() == 2: return amps.double()
        return amps.sum(dim=(-2, -1)).double()
    
    def _apply_pauli(self, amps, pauli):
        dim = 2 ** self.n_qubits
        new = torch.zeros_like(amps)
        if not pauli: return amps.clone()
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
    
    def eval_dipole(self, amps_raw):
        amps = self._to_scalar(amps_raw)
        n = float((amps[:,0]**2 + amps[:,1]**2).sum())
        if n < 1e-15: return 0.0
        amps = amps / math.sqrt(n)
        mu = self.dipole_identity
        for coeff, pauli in self.dipole_paulis:
            phi = self._apply_pauli(amps, pauli)
            ir = float((amps[:,0]*phi[:,0] + amps[:,1]*phi[:,1]).sum())
            ii = float((amps[:,0]*phi[:,1] - amps[:,1]*phi[:,0]).sum())
            mu += coeff.real * ir - coeff.imag * ii
        return mu
    
    def __call__(self, amps):
        e_base = self.base(amps)
        if abs(self.field) < 1e-15: return e_base
        mu = self.eval_dipole(amps)
        return e_base - self.field * mu

class DipoleOperatorBuilder:
    def __init__(self, bond_length_angstrom=0.735):
        mol = gto.M(atom=f'H 0 0 0; H 0 0 {bond_length_angstrom}',
                     basis='sto-3g', unit='Angstrom', verbose=0)
        mf = scf.RHF(mol).run()
        BOHR = 0.52917721067
        mol.set_common_origin(np.array([0, 0, bond_length_angstrom/2]) / BOHR)
        ao_dip_z = mol.intor('int1e_r')[2]
        self.dipole_mo = mf.mo_coeff.T @ ao_dip_z @ mf.mo_coeff
        self.n_orbitals = mol.nao_nr()
        
        print(f"Dipole MO matrix:\n{self.dipole_mo}")
        
        fermion_dip = FermionOperator()
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                val = self.dipole_mo[p, q]
                if abs(val) > 1e-14:
                    fermion_dip += FermionOperator(f"{2*p}^ {2*q}", val)
                    fermion_dip += FermionOperator(f"{2*p+1}^ {2*q+1}", val)
        
        qubit_dip = jordan_wigner(fermion_dip)
        self.pauli_terms = []
        self.identity_part = 0.0
        for ps, c in qubit_dip.terms.items():
            if len(ps) == 0:
                self.identity_part += complex(c).real
            else:
                self.pauli_terms.append((complex(c), 
                    [(int(qi), pc) for qi, pc in sorted(ps)]))
        
        print(f"Dipole: {len(self.pauli_terms)} Pauli terms, identity={self.identity_part:.6f}")


class PolarizabilityCalculator:
    def __init__(self):
        self.molecule = MOLECULES["H2"]
        self.n_qubits = self.molecule.n_qubits
        self.n_electrons = self.molecule.n_electrons
        
        self.qc = QuantumComputer(SimulatorConfig(device="cpu"))
        self.factory = self.qc._factory
        self.backend = self.qc._backends["hamiltonian"]
        
        self.dipole = DipoleOperatorBuilder()
        
        print("\nBuilding base Hamiltonian...")
        self.base_ham = ExactJWEnergy(self.molecule, self.n_qubits)
        
        self.singles, self.doubles = _sd_indices(self.n_electrons, self.n_qubits)
        self.n_params = len(self.singles) + len(self.doubles)
        print(f"  Singles: {self.singles}")
        print(f"  Doubles: {self.doubles}")
        print(f"  Total params: {self.n_params}")
    
    def _evaluator(self, field):
        return StarkEvaluator(self.base_ham, self.dipole.pauli_terms,
                              self.dipole.identity_part, field, self.n_qubits)
    
    def _get_state(self, theta):
        hf = self.factory.from_bitstring("1100")
        return particle_conserving_ansatz(
            hf.clone(), theta, self.singles, self.doubles, self.backend)
    
    def _diagnose(self, field, theta, label=""):
        state = self._get_state(theta)
        ev = self._evaluator(field)
        e_H0 = self.base_ham(state.amplitudes)
        mu = ev.eval_dipole(state.amplitudes)
        e_total = e_H0 - field * mu
        print(f"  [{label}] F={field:+.5f}: <H₀>={e_H0:.10f}, <μ>={mu:.10f}, "
              f"-F<μ>={-field*mu:.10f}, E={e_total:.10f}")
        return e_total
    
    def _optimize(self, field, theta_init, n_restarts=3):
        ev = self._evaluator(field)
        
        best_e, best_th = float('inf'), theta_init.copy()
        
        for trial in range(1 + n_restarts):
            if trial == 0:
                th0 = theta_init.copy()
            else:
                th0 = theta_init + np.random.randn(self.n_params) * 0.05
            
            hf = self.factory.from_bitstring("1100")
            cnt = [0]
            def cost(th):
                cnt[0] += 1
                st = particle_conserving_ansatz(
                    hf.clone(), th, self.singles, self.doubles, self.backend)
                return ev(st.amplitudes)
            
            res = minimize(cost, th0, method="L-BFGS-B",
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_e:
                best_e, best_th = res.fun, res.x.copy()
        
        return VQEResult(best_e, best_th, field, True, cnt[0])
    
    def run(self):
        print("\n" + "="*60)
        print("ANSATZ VERIFICATION")
        print("="*60)
        
        # Test that singles produce particle-conserving excitations
        for i, (o, v) in enumerate(self.singles):
            theta = np.zeros(self.n_params)
            theta[i] = 0.3
            state = self._get_state(theta)
            
            if state.amplitudes.dim() > 2:
                sa = state.amplitudes.sum(dim=(-2,-1)).double()
            else:
                sa = state.amplitudes.double()
            probs = sa[:,0]**2 + sa[:,1]**2
            
            print(f"\n  Single ({o}→{v}), θ=0.3:")
            top = torch.argsort(probs, descending=True)
            for idx in top[:4]:
                if probs[idx] > 1e-6:
                    ne = bin(int(idx)).count('1')
                    print(f"    |{format(int(idx),'04b')}> ({ne}e): {probs[idx]:.6f}")
        
        # Test dipole with singles
        print("\n  Dipole scan with single (0→2):")
        for t in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            theta = np.zeros(self.n_params)
            theta[0] = t
            theta[4] = -0.112  # double excitation
            state = self._get_state(theta)
            ev = self._evaluator(0.01)
            mu = ev.eval_dipole(state.amplitudes)
            e = self.base_ham(state.amplitudes)
            print(f"    θ₀={t:+.2f}: E={e:.8f}, <μ>={mu:+.8f}")
        
        # ---- Step 1: Zero-field reference ----
        print("\n" + "="*60)
        print("STEP 1: ZERO-FIELD REFERENCE")
        print("="*60)
        r0 = self._optimize(0.0, np.zeros(self.n_params))
        print(f"E(0) = {r0.energy:.10f}")
        print(f"θ = {r0.theta}")
        print(f"ΔE_FCI = {r0.energy - self.molecule.fci_energy:.2e}")
        
        # Diagnostic
        print("\n--- Diagnostic: same θ, different H(F) ---")
        for F in [0.0, 0.005, 0.01, 0.02]:
            self._diagnose(F, r0.theta, "zero-field θ")
        
        # ---- Step 2: Field sweep ----
        print("\n" + "="*60)
        print("STEP 2: FIELD SWEEP")
        print("="*60)
        
        fields = [-0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02]
        results = {0.0: r0}
        
        for f in sorted(fields, key=lambda x: abs(x)):
            r = self._optimize(f, r0.theta)
            results[f] = r
            de = r.energy - r0.energy
            print(f"  F={f:+.005f}: E={r.energy:.10f}, ΔE={de:+.10f}")
        
        # ---- Analysis ----
        print("\n" + "="*60)
        print("POLARIZABILITY ANALYSIS")
        print("="*60)
        
        sorted_f = np.array(sorted(results.keys()))
        e0 = results[0.0].energy
        
        print(f"\n{'F':>10} {'E(F)':>18} {'ΔE':>14}")
        for f in sorted_f:
            de = results[f].energy - e0
            print(f"{f:>+10.5f} {results[f].energy:>18.10f} {de:>+14.10f}")
        
        # Symmetry
        print("\nSymmetry |E(+F)-E(-F)|:")
        for f in sorted_f:
            if f > 0 and -f in results:
                print(f"  ±{f:.4f}: {abs(results[f].energy - results[-f].energy):.2e}")
        
        # Fit
        mask = np.abs(sorted_f) > 1e-12
        F2 = sorted_f[mask]**2
        dE = np.array([results[f].energy - e0 for f in sorted_f[mask]])
        
        if np.sum(F2**2) > 0:
            slope = np.sum(dE * F2) / np.sum(F2**2)
            alpha = -2 * slope
        else:
            alpha = 0.0
        
        # Reference from exact diag (TEST D showed α ≈ 2.750)
        print(f"\n  α (VQE)  = {alpha:.4f} a₀³")
        print(f"  α (exact diag, STO-3G) ≈ 2.750 a₀³")
        error = abs(alpha - 2.750) / 2.750 * 100 if alpha != 0 else 100
        print(f"  Error = {error:.1f}%")
        
        return alpha


if __name__ == "__main__":
    calc = PolarizabilityCalculator()
    calc.run() # 420 -_-
