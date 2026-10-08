# root: quantum_framework_core

*Community 1 | 8 files | cohesion 0.53*

## Definition

This community groups 8 file(s) rooted at `root` with dominant language py (cohesion 0.53). Central symbols: `AtomData`, `BackendComparator`, `BackendComparison`, `BackendComparisonVisualizer`, `BlochSphereVisualizer`, `BrutalVizEngine`, `BrutalistConfig`, `CNOTGate`. Core file: `quantum_framework_core.py` (196 symbols). Documented purpose: Hydrogen Orbital Visualizer.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `orbital_visualizer2.py` | py | utility | 17 | yes |
| `qc_dashboard.py` | py | utility | 88 | yes |
| `qc_integration.py` | py | utility | 61 | yes |
| `quantum_dash.py` | py | presentation | 71 | yes |
| `quantum_framework_core.py` | py | utility | 196 | yes |
| `quantum_framework_visualization.py` | py | utility | 20 | yes |
| `test_qc_integration.py` | py | testing | 87 | yes |
| `test_quantum_framework.py` | py | testing | 49 | yes |

## Key Symbols

- `Config` (class, `orbital_visualizer2.py:44`) `class Config`
- `WavefunctionCalculator` (class, `orbital_visualizer2.py:83`) `class WavefunctionCalculator` - Calculates hydrogen atom wavefunctions.
- `radial_wavefunction` (method, `orbital_visualizer2.py:87`) `def radial_wavefunction(n, l, r)`
- `spherical_harmonic_real` (method, `orbital_visualizer2.py:97`) `def spherical_harmonic_real(l, m, theta, phi)`
- `psi_on_grid` (method, `orbital_visualizer2.py:107`) `def psi_on_grid(n, l, m, grid_size)`
- `HamiltonianNNProcessor` (class, `orbital_visualizer2.py:126`) `class HamiltonianNNProcessor` - Uses YOUR TRAINED MODEL for calculations.
- `__init__` (method, `orbital_visualizer2.py:129`) `def __init__(self, engine)`
- `is_model_loaded` (method, `orbital_visualizer2.py:133`) `def is_model_loaded(self)`
- `compute_expected_energy` (method, `orbital_visualizer2.py:136`) `def compute_expected_energy(self, n, l, m)`
- `MonteCarloSampler` (class, `orbital_visualizer2.py:160`) `class MonteCarloSampler` - Monte Carlo sampling for orbital visualization.
- `__init__` (method, `orbital_visualizer2.py:163`) `def __init__(self, hamiltonian_processor)`
- `find_max_probability` (method, `orbital_visualizer2.py:166`) `def find_max_probability(self, n, l, m)`
- `sample` (method, `orbital_visualizer2.py:195`) `def sample(self, n, l, m, num_samples)`
- `OrbitalVisualizer` (class, `orbital_visualizer2.py:265`) `class OrbitalVisualizer` - HIGH RESOLUTION visualization - NOT 16x16!
- `visualize` (method, `orbital_visualizer2.py:268`) `def visualize(self, data, save_path, hamiltonian_processor)`
- `_plotly` (method, `orbital_visualizer2.py:396`) `def _plotly(self, X, Y, Z, prob_norm, phases, n, l, m)`
- `main` (method, `orbital_visualizer2.py:433`) `def main()`
- `DashboardConfig` (class, `qc_dashboard.py:55`) `class DashboardConfig` - Centralised configuration for the dashboard.
- `GateItem` (class, `qc_dashboard.py:120`) `class GateItem` - A gate placed in the circuit builder.
- `SnapshotData` (class, `qc_dashboard.py:130`) `class SnapshotData` - Quantum state snapshot for visualisation.
- `H2VQESolver` (class, `qc_dashboard.py:147`) `class H2VQESolver` - Self-contained H2 VQE solver using the hardcoded STO-3G Hamiltonian.
- `__init__` (method, `qc_dashboard.py:166`) `def __init__(self, config)`
- `_pauli_operators` (method, `qc_dashboard.py:173`) `def _pauli_operators(n_qubits, qubit, op)`
- `_pauli_string_matrix` (method, `qc_dashboard.py:189`) `def _pauli_string_matrix(paulis, n_qubits)`
- `_build_hamiltonian` (method, `qc_dashboard.py:206`) `def _build_hamiltonian(self, bond_length)` - Build H2 Hamiltonian matrix for given bond length (Angstrom).
- `_ansatz_state` (method, `qc_dashboard.py:229`) `def _ansatz_state(theta)` - UCC-like ansatz for H2: \|psi(theta)> = cos(theta)*\|10> + sin(theta)*\|01>.
- `_energy` (method, `qc_dashboard.py:241`) `def _energy(self, theta, h_matrix, e_nuc)`
- `run_vqe` (method, `qc_dashboard.py:248`) `def run_vqe(self, bond_length, max_iter)`
- `energy_landscape` (method, `qc_dashboard.py:293`) `def energy_landscape(self)` - Sweep bond length and return VQE energy curve.
- `orbital_wavefunction` (method, `qc_dashboard.py:318`) `def orbital_wavefunction(bond_length, grid_points)` - Compute hydrogen 1s orbital wavefunction along the internuclear axis.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 32
- Cross-boundary resolved imports (EXTRACTED): 12

## Connections

- [EXTRACTED] depends_on community 1 <-> 0 (strength 0.9): Extracted import edge crosses communities: qc_dashboard.py imports quantum_computer.py.
- [EXTRACTED] depends_on community 1 <-> 2 (strength 0.9): Extracted import edge crosses communities: qc_dashboard.py imports quantum_3dview.py.
- [INFERRED] bridges community 0 <-> 1 (strength 0.5): Inferred cross-community bridge: quantum_simulator.py reaches test_quantum_framework.py in 5 hops.
- [INFERRED] bridges community 0 <-> 1 (strength 0.5): Inferred cross-community bridge: relativistic_hydrogen.py reaches test_quantum_framework.py in 5 hops.
- [INFERRED] shares_context community 1 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (root: quantum_framework_core) and community 3 (root: quantum_framework_molecular_v2).
- [INFERRED] shares_context community 1 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (root: quantum_framework_core) and community 4 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in root: quantum_framework_core changed?
- Should root: quantum_framework_core be split, given cohesion 0.53?

## Sources

- `orbital_visualizer2.py`
- `qc_dashboard.py`
- `qc_integration.py`
- `quantum_dash.py`
- `quantum_framework_core.py`
- `quantum_framework_visualization.py`
- `test_qc_integration.py`
- `test_quantum_framework.py`
