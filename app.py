#!/usr/bin/env python3
"""
polarizability_debug.py
=======================
Debug completo del cálculo de polarizabilidad.
"""

from __future__ import annotations
import logging
import warnings
import numpy as np
import sys

warnings.filterwarnings("ignore")

try:
    from quantum_computer import QuantumComputer, SimulatorConfig, JointHilbertState, _GATE_REGISTRY
    from molecular_sim import MOLECULES, ExactJWEnergy, uccsd
    from pyscf import gto, scf
    import openfermion as of
    from openfermion.transforms import jordan_wigner
    from openfermion.ops import FermionOperator
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

def _sd_indices(n_e, n_q):
    occ = list(range(n_e))
    vir = list(range(n_e, n_q))
    singles = [(o, v) for o in occ for v in vir]
    doubles = [(o1, o2, v1, v2) for i, o1 in enumerate(occ) 
               for o2 in occ[i+1:] for j, v1 in enumerate(vir) for v2 in vir[j+1:]]
    return singles, doubles

def _run_circuit(circuit, backend, state):
    current_state = state
    for inst in circuit._instructions:
        gate = _GATE_REGISTRY.get(inst.gate_name)
        current_state = gate.apply(current_state, backend, inst.targets, inst.params)
    return current_state

class DipoleBuilder:
    def __init__(self):
        mol = gto.M(atom=[('H', (0, 0, 0)), ('H', (0, 0, 0.735))], basis="sto-3g", unit='Angstrom', verbose=0)
        mf = scf.RHF(mol).run()
        
        center = np.mean([[0,0,0], [0,0,0.735]], axis=0)
        mol.set_common_origin(center / 0.529177249)
        
        ao_dip = mol.intor('int1e_r')[2]
        self.dipole_mo = mf.mo_coeff.T @ ao_dip @ mf.mo_coeff
        self.dipole_mo -= np.eye(len(self.dipole_mo)) * np.mean(np.diag(self.dipole_mo))
        
        self.n_orbitals = mol.nao_nr()
        print(f"Dipole MO:\n{self.dipole_mo}")

    def get_hamiltonian(self, mol_data, field, n_qubits):
        base = ExactJWEnergy(mol_data, n_qubits)
        if field == 0:
            return base
            
        ferm_op = FermionOperator()
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                val = self.dipole_mo[p, q]
                if abs(val) > 1e-12:
                    ferm_op += FermionOperator(f"{2*p}^ {2*q}", -field * val)
                    ferm_op += FermionOperator(f"{2*p+1}^ {2*q+1}", -field * val)
        
        qubit_op = jordan_wigner(ferm_op)
        
        # Debug: verificar que hay términos no triviales
        non_trivial = [(t, c) for t, c in qubit_op.terms.items() if len(t) > 0]
        print(f"  Dipole operator has {len(non_trivial)} non-trivial terms")
        print(f"  Example: {non_trivial[0] if non_trivial else 'None'}")
        
        new_paulis = list(base.paulis)
        for term, coeff in qubit_op.terms.items():
            if len(term) > 0:
                pauli_list = [(int(q), p) for q, p in sorted(term)]
                new_paulis.append((complex(coeff), pauli_list))
        
        result = ExactJWEnergy(mol_data, n_qubits)
        result.paulis = new_paulis
        result.e_nuc = base.e_nuc
        return result

class VQE:
    def __init__(self):
        self.mol = MOLECULES.get("H2")
        self.n_qubits = self.mol.n_qubits
        self.n_electrons = self.mol.n_electrons
        
        self.qc = QuantumComputer(SimulatorConfig(device="cpu"))
        self.factory = self.qc._factory
        self.backend = self.qc._backends["hamiltonian"]
        
        self.dipole = DipoleBuilder()
        self.singles, self.doubles = _sd_indices(self.n_electrons, self.n_qubits)
        self.n_params = len(self.singles) + len(self.doubles)
        
        # Guardar evaluador base para comparación
        self.base_eval = ExactJWEnergy(self.mol, self.n_qubits)

    def test_hamiltonian(self, field):
        """Test manual: evaluar energía para varios estados theta"""
        from scipy.optimize import minimize
        
        print(f"\n{'='*50}")
        print(f"TESTING HAMILTONIAN AT F = {field:.4f}")
        print(f"{'='*50}")
        
        evaluator = self.dipole.get_hamiltonian(self.mol, field, self.n_qubits)
        hf_state = self.factory.from_bitstring("1100")
        
        # Test 1: HF state (theta=0)
        state_hf = uccsd(hf_state, np.zeros(self.n_params), self.singles, self.doubles,
                        self.backend, _run_circuit)
        e_hf = evaluator(state_hf.amplitudes)
        e_hf_base = self.base_eval(state_hf.amplitudes)
        print(f"HF state (theta=0): E={e_hf:.8f}, diff vs base={e_hf-e_hf_base:.8f}")
        
        # Test 2: Perturbado
        theta_test = np.array([0.2, 0, 0, 0, 0])  # Single excitation
        state_test = uccsd(hf_state, theta_test, self.singles, self.doubles,
                          self.backend, _run_circuit)
        e_test = evaluator(state_test.amplitudes)
        e_test_base = self.base_eval(state_test.amplitudes)
        print(f"Excited (theta=[0.2,0,0,0,0]): E={e_test:.8f}, diff vs base={e_test-e_test_base:.8f}")
        
        # Test 3: Optimizar desde HF
        def cost(t):
            s = uccsd(hf_state, t, self.singles, self.doubles, self.backend, _run_circuit)
            return evaluator(s.amplitudes)
        
        res = minimize(cost, np.zeros(self.n_params), method='L-BFGS-B', options={'maxiter': 200})
        print(f"Optimized from HF: E={res.fun:.8f}, theta={res.x}")
        
        # Test 4: Optimizar desde perturbado
        res2 = minimize(cost, theta_test, method='L-BFGS-B', options={'maxiter': 200})
        print(f"Optimized from excited: E={res2.fun:.8f}, theta={res2.x}")
        
        return res.fun

    def run(self, fields):
        results = {}
        for F in fields:
            results[F] = self.test_hamiltonian(F)
        return results

if __name__ == "__main__":
    vqe = VQE()
    fields = [0.0, 0.01, -0.01]
    results = vqe.run(fields)
    
    print(f"\n{'='*50}")
    print("RESUMEN")
    print(f"{'='*50}")
    for f, e in sorted(results.items()):
        print(f"F={f:+.3f}: E={e:.8f} Ha")
