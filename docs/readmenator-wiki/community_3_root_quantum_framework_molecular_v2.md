# root: quantum_framework_molecular_v2

*Community 3 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `BackendIntegrator`, `CachedHamiltonianEvaluator`, `CachedPauliOperation`, `HamiltonianBuilder`, `MPSState`, `MolecularConfig`, `MoleculeBuilder`, `MoleculeData`. Core file: `quantum_framework_molecular_v2.py` (61 symbols). Documented purpose: Quantum Framework Demo - Production Version.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `demo_molecular_vqe.py` | py | utility | 10 | yes |
| `quantum_framework_molecular_v2.py` | py | utility | 61 | yes |

## Key Symbols

- `print_header` (function, `demo_molecular_vqe.py:47`) `def print_header(title)` - Print formatted header.
- `check_dependencies` (function, `demo_molecular_vqe.py:54`) `def check_dependencies()` - Check and report available dependencies.
- `demo_h2_direct` (function, `demo_molecular_vqe.py:78`) `def demo_h2_direct()` - Demo: H2 with direct statevector (precision mode).
- `demo_h2_mps` (function, `demo_molecular_vqe.py:104`) `def demo_h2_mps()` - Demo: H2 with MPS compression.
- `demo_comparison` (function, `demo_molecular_vqe.py:131`) `def demo_comparison()` - Demo: Compare direct vs MPS.
- `demo_molecule_builder` (function, `demo_molecular_vqe.py:167`) `def demo_molecule_builder()` - Demo: OpenFermion molecule builder.
- `demo_smart_initialization` (function, `demo_molecular_vqe.py:197`) `def demo_smart_initialization()` - Demo: Smart parameter initialization.
- `demo_cached_operations` (function, `demo_molecular_vqe.py:231`) `def demo_cached_operations()` - Demo: Cached Pauli operations.
- `demo_config_from_toml` (function, `demo_molecular_vqe.py:277`) `def demo_config_from_toml()` - Demo: Configuration from TOML.
- `main` (function, `demo_molecular_vqe.py:302`) `def main()` - Run all demos.
- `_make_logger` (function, `quantum_framework_molecular_v2.py:75`) `def _make_logger(name)`
- `MolecularConfig` (class, `quantum_framework_molecular_v2.py:95`) `class MolecularConfig` - Configuration for molecular VQE simulations.
- `from_toml` (method, `quantum_framework_molecular_v2.py:131`) `def from_toml(cls, toml_path)` - Load configuration from TOML file.
- `MoleculeData` (class, `quantum_framework_molecular_v2.py:172`) `class MoleculeData` - Molecular data structure with all necessary information.
- `MoleculeBuilder` (class, `quantum_framework_molecular_v2.py:200`) `class MoleculeBuilder` - Build molecules using OpenFermion + PySCF.
- `build` (method, `quantum_framework_molecular_v2.py:208`) `def build(name, geometry, basis, charge, multiplicity, description)` - Build molecule using OpenFermion.
- `_run_pyscf_direct` (method, `quantum_framework_molecular_v2.py:289`) `def _run_pyscf_direct(geometry, basis, charge, multiplicity)` - Run PySCF directly if openfermionpyscf not available.
- `PseudoMolData` (class, `quantum_framework_molecular_v2.py:301`) `class PseudoMolData`
- `h2` (method, `quantum_framework_molecular_v2.py:324`) `def h2(bond_length, basis)` - Build H2 molecule.
- `h2o` (method, `quantum_framework_molecular_v2.py:330`) `def h2o(bond_length_oh, angle_hoh, basis)` - Build H2O molecule.
- `lih` (method, `quantum_framework_molecular_v2.py:342`) `def lih(bond_length, basis)` - Build LiH molecule.
- `HamiltonianBuilder` (class, `quantum_framework_molecular_v2.py:352`) `class HamiltonianBuilder` - Build molecular Hamiltonians using OpenFermion.
- `build_jw_hamiltonian` (method, `quantum_framework_molecular_v2.py:360`) `def build_jw_hamiltonian(mol)` - Build Jordan-Wigner transformed Hamiltonian using OpenFermion.
- `build_hamiltonian_matrix` (method, `quantum_framework_molecular_v2.py:417`) `def build_hamiltonian_matrix(mol, n_qubits)` - Build full Hamiltonian matrix for small systems.
- `_pauli_matrix` (method, `quantum_framework_molecular_v2.py:441`) `def _pauli_matrix(pauli_list, n_qubits)` - Build matrix for a Pauli string.
- `CachedPauliOperation` (class, `quantum_framework_molecular_v2.py:468`) `class CachedPauliOperation` - Precomputed Pauli operation for fast application.
- `CachedHamiltonianEvaluator` (class, `quantum_framework_molecular_v2.py:477`) `class CachedHamiltonianEvaluator` - Hamiltonian evaluator with cached Pauli operations.
- `__init__` (method, `quantum_framework_molecular_v2.py:484`) `def __init__(self, mol, config)`
- `_precompute_operations` (method, `quantum_framework_molecular_v2.py:503`) `def _precompute_operations(self)` - Precompute all Pauli operations for fast evaluation.
- `_compute_pauli_mapping` (method, `quantum_framework_molecular_v2.py:522`) `def _compute_pauli_mapping(self, pauli_list)` - Compute index mapping and phases for a Pauli string.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 2
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (root: quantum_simulator) and community 3 (root: quantum_framework_molecular_v2).
- [INFERRED] shares_context community 1 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (root: quantum_framework_core) and community 3 (root: quantum_framework_molecular_v2).
- [INFERRED] shares_context community 2 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 2 (root: quantum_framework_menu) and community 3 (root: quantum_framework_molecular_v2).
- [INFERRED] shares_context community 3 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 3 (root: quantum_framework_molecular_v2) and community 4 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in root: quantum_framework_molecular_v2 changed?
- Should root: quantum_framework_molecular_v2 be split, given cohesion 1.00?

## Sources

- `demo_molecular_vqe.py`
- `quantum_framework_molecular_v2.py`
