# orphans

*Community 4 | 3 files | cohesion 0.00*

## Definition

This community groups 3 file(s) rooted at `root` with dominant language py (cohesion 0.00). Central symbols: `AnomalousMagneticMoment`, `DiracHamiltonianOperator`, `DiracHydrogenAtom`, `DiracSpectralNet`, `ExactJWEnergy`, `GammaMatrices`, `HamiltonianBackboneNet`, `LambShiftCalculator`. Core file: `quantum_framework_physics.py` (52 symbols). Documented purpose: Quantum Framework Molecular Module - FIXED VERSION.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `install.sh` | sh | utility | 0 | no |
| `quantum_framework_molecular_fixed.py` | py | utility | 29 | yes |
| `quantum_framework_physics.py` | py | utility | 52 | yes |

## Key Symbols

- `_make_logger` (function, `quantum_framework_molecular_fixed.py:50`) `def _make_logger(name)`
- `MoleculeData` (class, `quantum_framework_molecular_fixed.py:64`) `class MoleculeData`
- `MoleculeBuilder` (class, `quantum_framework_molecular_fixed.py:77`) `class MoleculeBuilder` - Build molecule data for quantum chemistry calculations.
- `h2_sto3g` (method, `quantum_framework_molecular_fixed.py:84`) `def h2_sto3g(bond_length)` - Build H2 molecule with STO-3G basis.
- `_h2_pyscf` (method, `quantum_framework_molecular_fixed.py:91`) `def _h2_pyscf(bond_length)` - Build H2 using PySCF - FIXED atom string syntax.
- `_h2_hardcoded` (method, `quantum_framework_molecular_fixed.py:126`) `def _h2_hardcoded()` - Build H2 with hardcoded values - FIXED coefficients.
- `ExactJWEnergy` (class, `quantum_framework_molecular_fixed.py:147`) `class ExactJWEnergy` - Exact Jordan-Wigner energy evaluator.
- `__init__` (method, `quantum_framework_molecular_fixed.py:155`) `def __init__(self, mol, n_qubits)`
- `_build_hamiltonian` (method, `quantum_framework_molecular_fixed.py:162`) `def _build_hamiltonian(self)` - Build the molecular Hamiltonian in JW representation.
- `_build_openfermion_hamiltonian` (method, `quantum_framework_molecular_fixed.py:169`) `def _build_openfermion_hamiltonian(self)` - Build Hamiltonian using OpenFermion - FIXED geometry.
- `_build_hardcoded_hamiltonian` (method, `quantum_framework_molecular_fixed.py:215`) `def _build_hardcoded_hamiltonian(self)` - Build hardcoded H2 Hamiltonian - FIXED coefficients.
- `_apply_pauli` (method, `quantum_framework_molecular_fixed.py:247`) `def _apply_pauli(self, state, pauli)` - Apply Pauli operator to state vector.
- `expectation_value` (method, `quantum_framework_molecular_fixed.py:281`) `def expectation_value(self, state)` - Compute ⟨ψ\|H\|ψ⟩ for the given state.
- `evaluate` (method, `quantum_framework_molecular_fixed.py:305`) `def evaluate(self, amps)` - Evaluate energy from MPS amplitudes.
- `__call__` (method, `quantum_framework_molecular_fixed.py:319`) `def __call__(self, amps)`
- `_get_sd_indices` (method, `quantum_framework_molecular_fixed.py:323`) `def _get_sd_indices(n_electrons, n_qubits)` - Get single and double excitation indices for UCCSD.
- `UCCSDAnsatz` (class, `quantum_framework_molecular_fixed.py:343`) `class UCCSDAnsatz` - Unitary Coupled Cluster Singles and Doubles ansatz.
- `__init__` (method, `quantum_framework_molecular_fixed.py:350`) `def __init__(self, n_qubits, n_electrons, backend)`
- `apply_single_excitation` (method, `quantum_framework_molecular_fixed.py:359`) `def apply_single_excitation(self, state, o, v, theta)` - Apply single excitation operator exp(theta * (a_v† a_o - a_o† a_v)).
- `apply_double_excitation` (method, `quantum_framework_molecular_fixed.py:387`) `def apply_double_excitation(self, state, o1, o2, v1, v2, theta)` - Apply double excitation operator.
- `apply` (method, `quantum_framework_molecular_fixed.py:419`) `def apply(self, state, thetas)` - Apply UCCSD ansatz to state.
- `VQEResult` (class, `quantum_framework_molecular_fixed.py:457`) `class VQEResult`
- `__repr__` (method, `quantum_framework_molecular_fixed.py:471`) `def __repr__(self)`
- `VQESolver` (class, `quantum_framework_molecular_fixed.py:491`) `class VQESolver` - Variational Quantum Eigensolver.
- `__init__` (method, `quantum_framework_molecular_fixed.py:498`) `def __init__(self, qc, config)`
- `prepare_hf_state` (method, `quantum_framework_molecular_fixed.py:502`) `def prepare_hf_state(self, mol)` - Prepare Hartree-Fock state.
- `run` (method, `quantum_framework_molecular_fixed.py:525`) `def run(self, mol, backend, max_iter, tol)` - Run VQE optimization.
- `cost` (method, `quantum_framework_molecular_fixed.py:556`) `def cost(thetas)`
- `run_vqe_demo` (method, `quantum_framework_molecular_fixed.py:614`) `def run_vqe_demo()` - Run a quick VQE demo to verify the fixes.
- `SpectralLayer` (class, `quantum_framework_physics.py:26`) `class SpectralLayer(Module)` - Spectral convolution in frequency domain.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (root: quantum_simulator) and community 4 (orphans).
- [INFERRED] shares_context community 1 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (root: quantum_framework_core) and community 4 (orphans).
- [INFERRED] shares_context community 2 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 2 (root: quantum_framework_menu) and community 4 (orphans).
- [INFERRED] shares_context community 3 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 3 (root: quantum_framework_molecular_v2) and community 4 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in orphans changed?
- Should orphans be split, given cohesion 0.00?

## Sources

- `install.sh`
- `quantum_framework_molecular_fixed.py`
- `quantum_framework_physics.py`
