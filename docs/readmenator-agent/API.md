# API

## advanced_experiments.py

### _make_logger (function) `def _make_logger(name, level)`
- Defined: `advanced_experiments.py:121`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### main (method) `def main()`
- Defined: `advanced_experiments.py:1304`
- Doc: Main entry point.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### __init__ (method) `def __init__(self, n_qubits, marked_state)`
- Defined: `advanced_experiments.py:167`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### _validate (method) `def _validate(self)`
- Defined: `advanced_experiments.py:172`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### apply (method) `def apply(self, state, backend)`
- Defined: `advanced_experiments.py:176`
- Doc: Apply oracle: flip phase of marked state.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### __init__ (method) `def __init__(self, n_qubits)`
- Defined: `advanced_experiments.py:203`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### apply (method) `def apply(self, state, backend)`
- Defined: `advanced_experiments.py:206`
- Doc: Apply diffusion operator using Hadamard + Oracle on |0> + Hadamard.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `advanced_experiments.py:245`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### _calculate_entropy (method) `def _calculate_entropy(self, probs)`
- Defined: `advanced_experiments.py:266`
- Doc: Calculate Shannon entropy from probability distribution.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### _init_quantum_computer (method) `def _init_quantum_computer(self)`
- Defined: `advanced_experiments.py:278`
- Doc: Initialize quantum computer using existing quantum_computer.py infrastructure.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### run (method) `def run(self)`
- Defined: `advanced_experiments.py:300`
- Doc: Run Grover's search algorithm.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `advanced_experiments.py:440`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### bethe_formula (method) `def bethe_formula(self, n, l, Z)`
- Defined: `advanced_experiments.py:462`
- Doc: Bethe's non-relativistic formula for Lamb shift.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### _higher_l_shift (method) `def _higher_l_shift(self, n, l, Z)`
- Defined: `advanced_experiments.py:505`
- Doc: Approximate Lamb shift for l > 0.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### full_lamb_shift (method) `def full_lamb_shift(self, n, l, j, Z)`
- Defined: `advanced_experiments.py:518`
- Doc: Calculate full Lamb shift including radiative corrections.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### compare_2s_2p (method) `def compare_2s_2p(self, Z)`
- Defined: `advanced_experiments.py:558`
- Doc: Compare Lamb shift for 2s_{1/2} and 2p_{1/2} states.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `advanced_experiments.py:605`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### schwinger_term (method) `def schwinger_term(self)`
- Defined: `advanced_experiments.py:611`
- Doc: Schwinger's first-order result: a_e = α/(2π)
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### second_order (method) `def second_order(self)`
- Defined: `advanced_experiments.py:619`
- Doc: Second-order correction: (α/π)^2 * C_2
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### third_order (method) `def third_order(self)`
- Defined: `advanced_experiments.py:627`
- Doc: Third-order correction: (α/π)^3 * C_3
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### fourth_order (method) `def fourth_order(self)`
- Defined: `advanced_experiments.py:635`
- Doc: Fourth-order correction: (α/π)^4 * C_4
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### fifth_order (method) `def fifth_order(self)`
- Defined: `advanced_experiments.py:643`
- Doc: Fifth-order correction: (α/π)^5 * C_5
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### calculate_a_e (method) `def calculate_a_e(self, order)`
- Defined: `advanced_experiments.py:651`
- Doc: Calculate anomalous magnetic moment to specified order.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### full_report (method) `def full_report(self)`
- Defined: `advanced_experiments.py:686`
- Doc: Generate a full report on g-2 calculations.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `advanced_experiments.py:728`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### run_full_analysis (method) `def run_full_analysis(self)`
- Defined: `advanced_experiments.py:736`
- Doc: Run complete QED analysis.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### _calculate_energy_levels (method) `def _calculate_energy_levels(self)`
- Defined: `advanced_experiments.py:761`
- Doc: Calculate hydrogen energy levels including QED corrections.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### _dirac_energy (method) `def _dirac_energy(self, n, kappa)`
- Defined: `advanced_experiments.py:805`
- Doc: Calculate Dirac energy level.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### h2o (method) `def h2o(bond_length, angle_deg)`
- Defined: `advanced_experiments.py:866`
- Doc: Build water molecule geometry.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### nh3 (method) `def nh3(bond_length, angle_deg)`
- Defined: `advanced_experiments.py:898`
- Doc: Build ammonia molecule geometry.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### ch4 (method) `def ch4(bond_length)`
- Defined: `advanced_experiments.py:933`
- Doc: Build methane molecule geometry.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `advanced_experiments.py:976`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### run_pyscf (method) `def run_pyscf(self, molecule)`
- Defined: `advanced_experiments.py:999`
- Doc: Run PySCF calculation for the molecule.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### _hardcoded_values (method) `def _hardcoded_values(self, molecule)`
- Defined: `advanced_experiments.py:1066`
- Doc: Return hardcoded reference values for common molecules.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `advanced_experiments.py:1110`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### run_analysis (method) `def run_analysis(self, molecule_name)`
- Defined: `advanced_experiments.py:1123`
- Doc: Run complete analysis for a molecule.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### run_all (method) `def run_all(self)`
- Defined: `advanced_experiments.py:1164`
- Doc: Run analysis for all molecules.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### scan_bond_length (method) `def scan_bond_length(self, molecule_name, r_min, r_max, n_points)`
- Defined: `advanced_experiments.py:1175`
- Doc: Scan potential energy surface by varying bond length.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### __init__ (method) `def __init__(self)`
- Defined: `advanced_experiments.py:1228`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### run_grover (method) `def run_grover(self, n_qubits, marked_state)`
- Defined: `advanced_experiments.py:1235`
- Doc: Run Grover's algorithm experiment.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### run_qed (method) `def run_qed(self)`
- Defined: `advanced_experiments.py:1252`
- Doc: Run QED effects experiment.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### run_polyatomic (method) `def run_polyatomic(self, molecule)`
- Defined: `advanced_experiments.py:1266`
- Doc: Run polyatomic molecule experiment.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

### run_all (method) `def run_all(self)`
- Defined: `advanced_experiments.py:1278`
- Doc: Run all experiments.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
- Imported by: `quantum_visualizer.py`

## app.py

### _sd_indices (method) `def _sd_indices(n_e, n_q)`
- Defined: `app.py:46`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _run_circuit (method) `def _run_circuit(circuit, backend, state)`
- Defined: `app.py:55`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### givens_single_excitation (method) `def givens_single_excitation(state, o, v, theta, n_qubits, backend)`
- Defined: `app.py:61`
- Doc: Apply a particle-conserving single excitation rotation between 
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### particle_conserving_ansatz (method) `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)`
- Defined: `app.py:109`
- Doc: Particle-conserving UCCSD-like ansatz:
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- Defined: `app.py:152`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _to_scalar (method) `def _to_scalar(amps)`
- Defined: `app.py:161`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _apply_pauli (method) `def _apply_pauli(self, amps, pauli)`
- Defined: `app.py:165`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### eval_dipole (method) `def eval_dipole(self, amps_raw)`
- Defined: `app.py:184`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __call__ (method) `def __call__(self, amps)`
- Defined: `app.py:197`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, bond_length_angstrom)`
- Defined: `app.py:204`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __init__ (method) `def __init__(self)`
- Defined: `app.py:238`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _evaluator (method) `def _evaluator(self, field)`
- Defined: `app.py:258`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _get_state (method) `def _get_state(self, theta)`
- Defined: `app.py:262`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _diagnose (method) `def _diagnose(self, field, theta, label)`
- Defined: `app.py:267`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _optimize (method) `def _optimize(self, field, theta_init, n_restarts)`
- Defined: `app.py:277`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### run (method) `def run(self)`
- Defined: `app.py:303`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### cost (method) `def cost(th)`
- Defined: `app.py:290`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

## demo_molecular_vqe.py

### print_header (function) `def print_header(title)`
- Defined: `demo_molecular_vqe.py:47`
- Doc: Print formatted header.
- Depends on: `quantum_framework_molecular_v2.py`

### check_dependencies (function) `def check_dependencies()`
- Defined: `demo_molecular_vqe.py:54`
- Doc: Check and report available dependencies.
- Depends on: `quantum_framework_molecular_v2.py`

### demo_h2_direct (function) `def demo_h2_direct()`
- Defined: `demo_molecular_vqe.py:78`
- Doc: Demo: H2 with direct statevector (precision mode).
- Depends on: `quantum_framework_molecular_v2.py`

### demo_h2_mps (function) `def demo_h2_mps()`
- Defined: `demo_molecular_vqe.py:104`
- Doc: Demo: H2 with MPS compression.
- Depends on: `quantum_framework_molecular_v2.py`

### demo_comparison (function) `def demo_comparison()`
- Defined: `demo_molecular_vqe.py:131`
- Doc: Demo: Compare direct vs MPS.
- Depends on: `quantum_framework_molecular_v2.py`

### demo_molecule_builder (function) `def demo_molecule_builder()`
- Defined: `demo_molecular_vqe.py:167`
- Doc: Demo: OpenFermion molecule builder.
- Depends on: `quantum_framework_molecular_v2.py`

### demo_smart_initialization (function) `def demo_smart_initialization()`
- Defined: `demo_molecular_vqe.py:197`
- Doc: Demo: Smart parameter initialization.
- Depends on: `quantum_framework_molecular_v2.py`

### demo_cached_operations (function) `def demo_cached_operations()`
- Defined: `demo_molecular_vqe.py:231`
- Doc: Demo: Cached Pauli operations.
- Depends on: `quantum_framework_molecular_v2.py`

### demo_config_from_toml (function) `def demo_config_from_toml()`
- Defined: `demo_molecular_vqe.py:277`
- Doc: Demo: Configuration from TOML.
- Depends on: `quantum_framework_molecular_v2.py`

### main (function) `def main()`
- Defined: `demo_molecular_vqe.py:302`
- Doc: Run all demos.
- Depends on: `quantum_framework_molecular_v2.py`

## entangled_hydrogen.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `entangled_hydrogen.py:44`
- Doc: Create a module-level logger with consistent formatter.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### main (method) `def main()`
- Defined: `entangled_hydrogen.py:893`
- Doc: Main entry point for entangled hydrogen visualization.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### name (method) `def name(self)`
- Defined: `entangled_hydrogen.py:119`
- Doc: Return the name of the entangled state.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### prepare (method) `def prepare(self, n_qubits)`
- Defined: `entangled_hydrogen.py:123`
- Doc: Prepare the entangled state on n qubits.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### get_theoretical_entropy (method) `def get_theoretical_entropy(self)`
- Defined: `entangled_hydrogen.py:127`
- Doc: Return the theoretical Shannon entropy in bits.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### name (method) `def name(self)`
- Defined: `entangled_hydrogen.py:135`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### prepare (method) `def prepare(self, qc, backend)`
- Defined: `entangled_hydrogen.py:138`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### get_theoretical_entropy (method) `def get_theoretical_entropy(self)`
- Defined: `entangled_hydrogen.py:141`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### __init__ (method) `def __init__(self, n_qubits)`
- Defined: `entangled_hydrogen.py:148`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### name (method) `def name(self)`
- Defined: `entangled_hydrogen.py:152`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### prepare (method) `def prepare(self, qc, backend)`
- Defined: `entangled_hydrogen.py:155`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### get_theoretical_entropy (method) `def get_theoretical_entropy(self)`
- Defined: `entangled_hydrogen.py:158`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### __init__ (method) `def __init__(self, n_qubits)`
- Defined: `entangled_hydrogen.py:165`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### name (method) `def name(self)`
- Defined: `entangled_hydrogen.py:169`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### prepare (method) `def prepare(self, qc, backend, factory)`
- Defined: `entangled_hydrogen.py:172`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### get_theoretical_entropy (method) `def get_theoretical_entropy(self)`
- Defined: `entangled_hydrogen.py:190`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `entangled_hydrogen.py:200`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### radial_wavefunction (method) `def radial_wavefunction(n, l, r)`
- Defined: `entangled_hydrogen.py:204`
- Doc: Calculate non-relativistic radial wavefunction R_nl(r).
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### spherical_harmonic_real (method) `def spherical_harmonic_real(l, m, theta, phi)`
- Defined: `entangled_hydrogen.py:215`
- Doc: Calculate real spherical harmonics Y_lm(theta, phi).
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### psi_on_grid (method) `def psi_on_grid(self, n, l, m)`
- Defined: `entangled_hydrogen.py:225`
- Doc: Calculate wavefunction on 2D grid for quantum computer processing.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### psi_3d (method) `def psi_3d(self, n, l, m, r, theta, phi)`
- Defined: `entangled_hydrogen.py:245`
- Doc: Calculate full 3D wavefunction psi_nlm(r, theta, phi).
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### __init__ (method) `def __init__(self, config, wavefunction_calc)`
- Defined: `entangled_hydrogen.py:259`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### find_max_probability (method) `def find_max_probability(self, n, l, m)`
- Defined: `entangled_hydrogen.py:263`
- Doc: Find maximum probability for rejection sampling.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### sample_orbital (method) `def sample_orbital(self, n, l, m, num_samples)`
- Defined: `entangled_hydrogen.py:295`
- Doc: Sample points from a single hydrogen orbital using Monte Carlo.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### sample_entangled_state (method) `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples, entanglement_weight)`
- Defined: `entangled_hydrogen.py:362`
- Doc: Sample from entangled hydrogen state.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `entangled_hydrogen.py:414`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### visualize (method) `def visualize(self, data, quantum_result, save_path)`
- Defined: `entangled_hydrogen.py:417`
- Doc: Create visualization of entangled hydrogen state.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `entangled_hydrogen.py:577`
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### _initialize_quantum_computer (method) `def _initialize_quantum_computer(self)`
- Defined: `entangled_hydrogen.py:591`
- Doc: Initialize the quantum computer using the existing quantum_computer.py module.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### run_bell_entangled_hydrogen (method) `def run_bell_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples, suffix)`
- Defined: `entangled_hydrogen.py:630`
- Doc: Run Bell state entangled hydrogen visualization.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### run_ghz_entangled_hydrogen (method) `def run_ghz_entangled_hydrogen(self, orbitals, backend, num_samples)`
- Defined: `entangled_hydrogen.py:672`
- Doc: Run GHZ state entangled hydrogen visualization.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### run_entangled_h_with_molecular_energy (method) `def run_entangled_h_with_molecular_energy(self, n1, l1, m1, n2, l2, m2, backend, num_samples)`
- Defined: `entangled_hydrogen.py:745`
- Doc: Run entangled hydrogen with molecular energy evaluation using molecular_sim.py.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### run_relativistic_entangled_hydrogen (method) `def run_relativistic_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples)`
- Defined: `entangled_hydrogen.py:794`
- Doc: Run entangled hydrogen with relativistic Dirac calculations.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

### run_all_demonstrations (method) `def run_all_demonstrations(self, num_samples)`
- Defined: `entangled_hydrogen.py:852`
- Doc: Run all available entangled hydrogen demonstrations.
- Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`

## higgs_four_lepton_analysis.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `higgs_four_lepton_analysis.py:57`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### main (method) `def main()`
- Defined: `higgs_four_lepton_analysis.py:747`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __post_init__ (method) `def __post_init__(self)`
- Defined: `higgs_four_lepton_analysis.py:136`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### from_energy_momentum (method) `def from_energy_momentum(cls, E, px, py, pz)`
- Defined: `higgs_four_lepton_analysis.py:151`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __add__ (method) `def __add__(self, other)`
- Defined: `higgs_four_lepton_analysis.py:154`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### pt (method) `def pt(self)`
- Defined: `higgs_four_lepton_analysis.py:171`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### eta (method) `def eta(self)`
- Defined: `higgs_four_lepton_analysis.py:173`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### phi (method) `def phi(self)`
- Defined: `higgs_four_lepton_analysis.py:175`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### energy (method) `def energy(self)`
- Defined: `higgs_four_lepton_analysis.py:177`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### mass (method) `def mass(self)`
- Defined: `higgs_four_lepton_analysis.py:179`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __post_init__ (method) `def __post_init__(self)`
- Defined: `higgs_four_lepton_analysis.py:193`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### check_higgs (method) `def check_higgs(self, cfg)`
- Defined: `higgs_four_lepton_analysis.py:200`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `higgs_four_lepton_analysis.py:213`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _precompute_momentum_grids (method) `def _precompute_momentum_grids(self)`
- Defined: `higgs_four_lepton_analysis.py:247`
- Doc: Precompute k-space grids for Dirac equation.
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### momentum_to_spinor_wavefunction (method) `def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)`
- Defined: `higgs_four_lepton_analysis.py:255`
- Doc: Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### evolve_with_dirac_backend (method) `def evolve_with_dirac_backend(self, psi, steps)`
- Defined: `higgs_four_lepton_analysis.py:296`
- Doc: Evolve a wavefunction using the DiracBackend neural network.
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### evolve_with_schrodinger_backend (method) `def evolve_with_schrodinger_backend(self, psi, steps)`
- Defined: `higgs_four_lepton_analysis.py:308`
- Doc: Evolve using the SchrodingerBackend neural network.
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### evolve_with_hamiltonian_backend (method) `def evolve_with_hamiltonian_backend(self, psi, steps)`
- Defined: `higgs_four_lepton_analysis.py:315`
- Doc: Evolve using the HamiltonianBackend neural network.
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### compute_dirac_current (method) `def compute_dirac_current(self, px, py, pz, energy, mass, charge)`
- Defined: `higgs_four_lepton_analysis.py:322`
- Doc: Compute the Dirac current using the actual neural network backends.
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### compute_spinor_amplitude (method) `def compute_spinor_amplitude(self, psi)`
- Defined: `higgs_four_lepton_analysis.py:370`
- Doc: Compute complex amplitude from wavefunction for helicity analysis.
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `higgs_four_lepton_analysis.py:380`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### parse_file (method) `def parse_file(self, filepath, event_type)`
- Defined: `higgs_four_lepton_analysis.py:383`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _parse_row (method) `def _parse_row(self, row, event_type)`
- Defined: `higgs_four_lepton_analysis.py:396`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config, quantum_processor)`
- Defined: `higgs_four_lepton_analysis.py:439`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### compute_quantum_helix (method) `def compute_quantum_helix(self, px, py, pz, charge, mass, energy)`
- Defined: `higgs_four_lepton_analysis.py:443`
- Doc: Compute helical trajectory using actual DiracBackend evolution.
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### create_visualization (method) `def create_visualization(self, events, output_path)`
- Defined: `higgs_four_lepton_analysis.py:474`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _create_detector (method) `def _create_detector(self)`
- Defined: `higgs_four_lepton_analysis.py:566`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### _create_explosion (method) `def _create_explosion(self, vx, vy, vz, energy)`
- Defined: `higgs_four_lepton_analysis.py:581`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `higgs_four_lepton_analysis.py:617`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### fetch_data (method) `def fetch_data(self)`
- Defined: `higgs_four_lepton_analysis.py:630`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### load_events (method) `def load_events(self)`
- Defined: `higgs_four_lepton_analysis.py:645`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### analyze_with_quantum_backends (method) `def analyze_with_quantum_backends(self)`
- Defined: `higgs_four_lepton_analysis.py:672`
- Doc: Run analysis using actual quantum backends.
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### generate_visualization (method) `def generate_visualization(self)`
- Defined: `higgs_four_lepton_analysis.py:728`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

### run (method) `def run(self)`
- Defined: `higgs_four_lepton_analysis.py:734`
- Depends on: `quantum_computer.py`
- Imported by: `quantum_framework_menu.py`

## higgs_quantum_analysis.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `higgs_quantum_analysis.py:57`
- Depends on: `quantum_computer.py`

### main (method) `def main()`
- Defined: `higgs_quantum_analysis.py:747`
- Depends on: `quantum_computer.py`

### __post_init__ (method) `def __post_init__(self)`
- Defined: `higgs_quantum_analysis.py:136`
- Depends on: `quantum_computer.py`

### from_energy_momentum (method) `def from_energy_momentum(cls, E, px, py, pz)`
- Defined: `higgs_quantum_analysis.py:151`
- Depends on: `quantum_computer.py`

### __add__ (method) `def __add__(self, other)`
- Defined: `higgs_quantum_analysis.py:154`
- Depends on: `quantum_computer.py`

### pt (method) `def pt(self)`
- Defined: `higgs_quantum_analysis.py:171`
- Depends on: `quantum_computer.py`

### eta (method) `def eta(self)`
- Defined: `higgs_quantum_analysis.py:173`
- Depends on: `quantum_computer.py`

### phi (method) `def phi(self)`
- Defined: `higgs_quantum_analysis.py:175`
- Depends on: `quantum_computer.py`

### energy (method) `def energy(self)`
- Defined: `higgs_quantum_analysis.py:177`
- Depends on: `quantum_computer.py`

### mass (method) `def mass(self)`
- Defined: `higgs_quantum_analysis.py:179`
- Depends on: `quantum_computer.py`

### __post_init__ (method) `def __post_init__(self)`
- Defined: `higgs_quantum_analysis.py:193`
- Depends on: `quantum_computer.py`

### check_higgs (method) `def check_higgs(self, cfg)`
- Defined: `higgs_quantum_analysis.py:200`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `higgs_quantum_analysis.py:213`
- Depends on: `quantum_computer.py`

### _precompute_momentum_grids (method) `def _precompute_momentum_grids(self)`
- Defined: `higgs_quantum_analysis.py:247`
- Doc: Precompute k-space grids for Dirac equation.
- Depends on: `quantum_computer.py`

### momentum_to_spinor_wavefunction (method) `def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)`
- Defined: `higgs_quantum_analysis.py:255`
- Doc: Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)
- Depends on: `quantum_computer.py`

### evolve_with_dirac_backend (method) `def evolve_with_dirac_backend(self, psi, steps)`
- Defined: `higgs_quantum_analysis.py:296`
- Doc: Evolve a wavefunction using the DiracBackend neural network.
- Depends on: `quantum_computer.py`

### evolve_with_schrodinger_backend (method) `def evolve_with_schrodinger_backend(self, psi, steps)`
- Defined: `higgs_quantum_analysis.py:308`
- Doc: Evolve using the SchrodingerBackend neural network.
- Depends on: `quantum_computer.py`

### evolve_with_hamiltonian_backend (method) `def evolve_with_hamiltonian_backend(self, psi, steps)`
- Defined: `higgs_quantum_analysis.py:315`
- Doc: Evolve using the HamiltonianBackend neural network.
- Depends on: `quantum_computer.py`

### compute_dirac_current (method) `def compute_dirac_current(self, px, py, pz, energy, mass, charge)`
- Defined: `higgs_quantum_analysis.py:322`
- Doc: Compute the Dirac current using the actual neural network backends.
- Depends on: `quantum_computer.py`

### compute_spinor_amplitude (method) `def compute_spinor_amplitude(self, psi)`
- Defined: `higgs_quantum_analysis.py:370`
- Doc: Compute complex amplitude from wavefunction for helicity analysis.
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `higgs_quantum_analysis.py:380`
- Depends on: `quantum_computer.py`

### parse_file (method) `def parse_file(self, filepath, event_type)`
- Defined: `higgs_quantum_analysis.py:383`
- Depends on: `quantum_computer.py`

### _parse_row (method) `def _parse_row(self, row, event_type)`
- Defined: `higgs_quantum_analysis.py:396`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, config, quantum_processor)`
- Defined: `higgs_quantum_analysis.py:439`
- Depends on: `quantum_computer.py`

### compute_quantum_helix (method) `def compute_quantum_helix(self, px, py, pz, charge, mass, energy)`
- Defined: `higgs_quantum_analysis.py:443`
- Doc: Compute helical trajectory using actual DiracBackend evolution.
- Depends on: `quantum_computer.py`

### create_visualization (method) `def create_visualization(self, events, output_path)`
- Defined: `higgs_quantum_analysis.py:474`
- Depends on: `quantum_computer.py`

### _create_detector (method) `def _create_detector(self)`
- Defined: `higgs_quantum_analysis.py:566`
- Depends on: `quantum_computer.py`

### _create_explosion (method) `def _create_explosion(self, vx, vy, vz, energy)`
- Defined: `higgs_quantum_analysis.py:581`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `higgs_quantum_analysis.py:617`
- Depends on: `quantum_computer.py`

### fetch_data (method) `def fetch_data(self)`
- Defined: `higgs_quantum_analysis.py:630`
- Depends on: `quantum_computer.py`

### load_events (method) `def load_events(self)`
- Defined: `higgs_quantum_analysis.py:645`
- Depends on: `quantum_computer.py`

### analyze_with_quantum_backends (method) `def analyze_with_quantum_backends(self)`
- Defined: `higgs_quantum_analysis.py:672`
- Doc: Run analysis using actual quantum backends.
- Depends on: `quantum_computer.py`

### generate_visualization (method) `def generate_visualization(self)`
- Defined: `higgs_quantum_analysis.py:728`
- Depends on: `quantum_computer.py`

### run (method) `def run(self)`
- Defined: `higgs_quantum_analysis.py:734`
- Depends on: `quantum_computer.py`

## molecular_sim.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `molecular_sim.py:18`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### _h2_sto3g_pyscf (method) `def _h2_sto3g_pyscf()`
- Defined: `molecular_sim.py:41`
- Doc: H2 con datos PySCF.
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### _h2_sto3g_hardcoded (method) `def _h2_sto3g_hardcoded()`
- Defined: `molecular_sim.py:75`
- Doc: H2 con datos hardcodeados.
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### build_jw_hamiltonian_of (method) `def build_jw_hamiltonian_of(mol)`
- Defined: `molecular_sim.py:111`
- Doc: Construye Hamiltoniano JW usando OpenFermion correctamente.
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### prepare_hf (method) `def prepare_hf(mol, factory, backend)`
- Defined: `molecular_sim.py:304`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### _sd_indices (method) `def _sd_indices(n_e, n_q)`
- Defined: `molecular_sim.py:312`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### uccsd (method) `def uccsd(state, thetas, singles, doubles, backend, runner)`
- Defined: `molecular_sim.py:321`
- Doc: UCCSD ansatz.
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### __init__ (method) `def __init__(self, mol, n_qubits)`
- Defined: `molecular_sim.py:190`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### _to_scalar (method) `def _to_scalar(amps)`
- Defined: `molecular_sim.py:203`
- Doc: (dim,2,G,G) → (dim,2)  ó  (dim,2) → (dim,2).
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### _verify_hf (method) `def _verify_hf(self)`
- Defined: `molecular_sim.py:213`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### _apply (method) `def _apply(self, amps, pauli)`
- Defined: `molecular_sim.py:223`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### _evaluate (method) `def _evaluate(self, amps)`
- Defined: `molecular_sim.py:243`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### __call__ (method) `def __call__(self, amps)`
- Defined: `molecular_sim.py:257`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### __init__ (method) `def __init__(self, mol, n_qubits, exact_eval, backend)`
- Defined: `molecular_sim.py:268`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### calibrate (method) `def calibrate(self, hf_amps)`
- Defined: `molecular_sim.py:277`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### cost_with_barrier (method) `def cost_with_barrier(self, amps)`
- Defined: `molecular_sim.py:290`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### __repr__ (method) `def __repr__(self)`
- Defined: `molecular_sim.py:377`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### __init__ (method) `def __init__(self, qc, config)`
- Defined: `molecular_sim.py:391`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### _run (method) `def _run(self, circ, be, state)`
- Defined: `molecular_sim.py:395`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### run (method) `def run(self, mol, backend, max_iter, tol)`
- Defined: `molecular_sim.py:403`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

### cost (method) `def cost(thetas)`
- Defined: `molecular_sim.py:451`
- Depends on: `quantum_computer.py`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`

## orbital_visualizer2.py

### main (method) `def main()`
- Defined: `orbital_visualizer2.py:433`
- Imported by: `qc_dashboard.py`

### radial_wavefunction (method) `def radial_wavefunction(n, l, r)`
- Defined: `orbital_visualizer2.py:87`
- Imported by: `qc_dashboard.py`

### spherical_harmonic_real (method) `def spherical_harmonic_real(l, m, theta, phi)`
- Defined: `orbital_visualizer2.py:97`
- Imported by: `qc_dashboard.py`

### psi_on_grid (method) `def psi_on_grid(n, l, m, grid_size)`
- Defined: `orbital_visualizer2.py:107`
- Imported by: `qc_dashboard.py`

### __init__ (method) `def __init__(self, engine)`
- Defined: `orbital_visualizer2.py:129`
- Imported by: `qc_dashboard.py`

### is_model_loaded (method) `def is_model_loaded(self)`
- Defined: `orbital_visualizer2.py:133`
- Imported by: `qc_dashboard.py`

### compute_expected_energy (method) `def compute_expected_energy(self, n, l, m)`
- Defined: `orbital_visualizer2.py:136`
- Imported by: `qc_dashboard.py`

### __init__ (method) `def __init__(self, hamiltonian_processor)`
- Defined: `orbital_visualizer2.py:163`
- Imported by: `qc_dashboard.py`

### find_max_probability (method) `def find_max_probability(self, n, l, m)`
- Defined: `orbital_visualizer2.py:166`
- Imported by: `qc_dashboard.py`

### sample (method) `def sample(self, n, l, m, num_samples)`
- Defined: `orbital_visualizer2.py:195`
- Imported by: `qc_dashboard.py`

### visualize (method) `def visualize(self, data, save_path, hamiltonian_processor)`
- Defined: `orbital_visualizer2.py:268`
- Imported by: `qc_dashboard.py`

### _plotly (method) `def _plotly(self, X, Y, Z, prob_norm, phases, n, l, m)`
- Defined: `orbital_visualizer2.py:396`
- Imported by: `qc_dashboard.py`

## polarizability_v3.py

### _sd_indices (method) `def _sd_indices(n_e, n_q)`
- Defined: `polarizability_v3.py:58`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### _run_circuit (method) `def _run_circuit(circuit, backend, state)`
- Defined: `polarizability_v3.py:67`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### givens_single_excitation (method) `def givens_single_excitation(state, o, v, theta, n_qubits, backend)`
- Defined: `polarizability_v3.py:73`
- Doc: Apply a particle-conserving single excitation rotation between 
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### particle_conserving_ansatz (method) `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)`
- Defined: `polarizability_v3.py:121`
- Doc: Particle-conserving UCCSD-like ansatz:
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### __init__ (method) `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- Defined: `polarizability_v3.py:164`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### _to_scalar (method) `def _to_scalar(amps)`
- Defined: `polarizability_v3.py:173`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### _apply_pauli (method) `def _apply_pauli(self, amps, pauli)`
- Defined: `polarizability_v3.py:177`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### eval_dipole (method) `def eval_dipole(self, amps_raw)`
- Defined: `polarizability_v3.py:196`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### __call__ (method) `def __call__(self, amps)`
- Defined: `polarizability_v3.py:209`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### __init__ (method) `def __init__(self, bond_length_angstrom)`
- Defined: `polarizability_v3.py:216`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### __init__ (method) `def __init__(self)`
- Defined: `polarizability_v3.py:250`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### _evaluator (method) `def _evaluator(self, field)`
- Defined: `polarizability_v3.py:270`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### _get_state (method) `def _get_state(self, theta)`
- Defined: `polarizability_v3.py:274`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### _diagnose (method) `def _diagnose(self, field, theta, label)`
- Defined: `polarizability_v3.py:279`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### _optimize (method) `def _optimize(self, field, theta_init, n_restarts)`
- Defined: `polarizability_v3.py:289`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### run (method) `def run(self)`
- Defined: `polarizability_v3.py:315`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

### cost (method) `def cost(th)`
- Defined: `polarizability_v3.py:302`
- Depends on: `molecular_sim.py`, `quantum_computer.py`

## qc_dashboard.py

### _capture_mpl_fig (method) `def _capture_mpl_fig(func)`
- Defined: `qc_dashboard.py:1133`
- Doc: Run a function that creates a matplotlib figure and capture it as PNG bytes.
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### main (method) `def main()`
- Defined: `qc_dashboard.py:1994`
- Doc: Launch the Streamlit dashboard.
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_dashboard.py:166`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _pauli_operators (method) `def _pauli_operators(n_qubits, qubit, op)`
- Defined: `qc_dashboard.py:173`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _pauli_string_matrix (method) `def _pauli_string_matrix(paulis, n_qubits)`
- Defined: `qc_dashboard.py:189`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _build_hamiltonian (method) `def _build_hamiltonian(self, bond_length)`
- Defined: `qc_dashboard.py:206`
- Doc: Build H2 Hamiltonian matrix for given bond length (Angstrom).
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _ansatz_state (method) `def _ansatz_state(theta)`
- Defined: `qc_dashboard.py:229`
- Doc: UCC-like ansatz for H2: |psi(theta)> = cos(theta)*|10> + sin(theta)*|01>.
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _energy (method) `def _energy(self, theta, h_matrix, e_nuc)`
- Defined: `qc_dashboard.py:241`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### run_vqe (method) `def run_vqe(self, bond_length, max_iter)`
- Defined: `qc_dashboard.py:248`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### energy_landscape (method) `def energy_landscape(self)`
- Defined: `qc_dashboard.py:293`
- Doc: Sweep bond length and return VQE energy curve.
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### orbital_wavefunction (method) `def orbital_wavefunction(bond_length, grid_points)`
- Defined: `qc_dashboard.py:318`
- Doc: Compute hydrogen 1s orbital wavefunction along the internuclear axis.
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_dashboard.py:343`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _init_plotting (method) `def _init_plotting(self)`
- Defined: `qc_dashboard.py:349`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### available (method) `def available(self)`
- Defined: `qc_dashboard.py:362`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_full_dashboard (method) `def render_full_dashboard(self, snapshots, current)`
- Defined: `qc_dashboard.py:365`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_entropy_chart (method) `def render_entropy_chart(self, snapshots)`
- Defined: `qc_dashboard.py:403`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_entanglement_profile (method) `def render_entanglement_profile(self, snapshots)`
- Defined: `qc_dashboard.py:429`
- Doc: Entropy vs cut position for the latest snapshot.
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_vqe_convergence (method) `def render_vqe_convergence(self, convergence, e_hf, e_fci)`
- Defined: `qc_dashboard.py:474`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_energy_landscape (method) `def render_energy_landscape(self, bond_lengths, vqe_energies, hf_energies, fci_energies)`
- Defined: `qc_dashboard.py:501`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_orbital_plot (method) `def render_orbital_plot(self, orbital_data)`
- Defined: `qc_dashboard.py:533`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_entropy_scaling (method) `def render_entropy_scaling(self, data)`
- Defined: `qc_dashboard.py:562`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_probabilities (method) `def _render_probabilities(self, snap, ax)`
- Defined: `qc_dashboard.py:589`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_bloch_sphere (method) `def _render_bloch_sphere(self, snap, ax)`
- Defined: `qc_dashboard.py:606`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_phase_space (method) `def _render_phase_space(self, snap, ax)`
- Defined: `qc_dashboard.py:635`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_orbital_2d_projections (method) `def render_orbital_2d_projections(self, data)`
- Defined: `qc_dashboard.py:670`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _hex_to_rgb (method) `def _hex_to_rgb(h)`
- Defined: `qc_dashboard.py:715`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_dashboard.py:728`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _init_framework (method) `def _init_framework(self)`
- Defined: `qc_dashboard.py:737`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _get_mps_gate_registry (method) `def _get_mps_gate_registry()`
- Defined: `qc_dashboard.py:760`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _get_sv_gate_registry (method) `def _get_sv_gate_registry()`
- Defined: `qc_dashboard.py:768`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### execute_circuit (method) `def execute_circuit(self, gates, n_qubits)`
- Defined: `qc_dashboard.py:775`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _mps_execute (method) `def _mps_execute(self, gates, n_qubits)`
- Defined: `qc_dashboard.py:788`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _sv_execute (method) `def _sv_execute(self, gates, n_qubits)`
- Defined: `qc_dashboard.py:839`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _snapshot_from_mps (method) `def _snapshot_from_mps(self, state, step, gate_name)`
- Defined: `qc_dashboard.py:886`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _snapshot_from_sv (method) `def _snapshot_from_sv(self, state, step, gate_name)`
- Defined: `qc_dashboard.py:898`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _compute_bloch_mps (method) `def _compute_bloch_mps(state, n_qubits)`
- Defined: `qc_dashboard.py:917`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _synthetic_execute (method) `def _synthetic_execute(self, gates, n_qubits)`
- Defined: `qc_dashboard.py:933`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_dashboard.py:982`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _init_plotly (method) `def _init_plotly(self)`
- Defined: `qc_dashboard.py:988`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### available (method) `def available(self)`
- Defined: `qc_dashboard.py:998`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_bloch_3d (method) `def render_bloch_3d(self, bloch_vectors)`
- Defined: `qc_dashboard.py:1001`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_probability_3d (method) `def render_probability_3d(self, probabilities, n_qubits)`
- Defined: `qc_dashboard.py:1049`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_state_3d (method) `def render_state_3d(self, probabilities, phases)`
- Defined: `qc_dashboard.py:1080`
- Doc: 3D scatter plot: X=real, Y=imaginary, Z=probability.
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self)`
- Defined: `qc_dashboard.py:1154`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _init_real (method) `def _init_real(self)`
- Defined: `qc_dashboard.py:1163`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### available (method) `def available(self)`
- Defined: `qc_dashboard.py:1210`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### entangled_available (method) `def entangled_available(self)`
- Defined: `qc_dashboard.py:1214`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### sample (method) `def sample(self, n, l, m, num_samples)`
- Defined: `qc_dashboard.py:1217`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_to_bytes (method) `def render_to_bytes(self, data)`
- Defined: `qc_dashboard.py:1222`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### sample_entangled (method) `def sample_entangled(self, n1, l1, m1, n2, l2, m2, num_samples)`
- Defined: `qc_dashboard.py:1240`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_entangled_to_bytes (method) `def render_entangled_to_bytes(self, data)`
- Defined: `qc_dashboard.py:1250`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self)`
- Defined: `qc_dashboard.py:1272`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _init_real (method) `def _init_real(self)`
- Defined: `qc_dashboard.py:1279`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### dash_available (method) `def dash_available(self)`
- Defined: `qc_dashboard.py:1297`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### hologram_available (method) `def hologram_available(self)`
- Defined: `qc_dashboard.py:1301`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### run_brutal_viz (method) `def run_brutal_viz(self, circuit_name)`
- Defined: `qc_dashboard.py:1304`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### render_hologram (method) `def render_hologram(self, snapshots, backend_comp)`
- Defined: `qc_dashboard.py:1327`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_dashboard.py:1347`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### run_comparison (method) `def run_comparison(self, gates, n_qubits)`
- Defined: `qc_dashboard.py:1351`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_dashboard.py:1374`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### run (method) `def run(self)`
- Defined: `qc_dashboard.py:1384`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _ensure_streamlit (method) `def _ensure_streamlit()`
- Defined: `qc_dashboard.py:1399`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _init_session (method) `def _init_session(st)`
- Defined: `qc_dashboard.py:1408`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_ui (method) `def _render_ui(self, st)`
- Defined: `qc_dashboard.py:1432`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_sidebar_controls (method) `def _render_sidebar_controls(self, st)`
- Defined: `qc_dashboard.py:1447`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_main_panel (method) `def _render_main_panel(self, st)`
- Defined: `qc_dashboard.py:1535`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_playground_tab (method) `def _render_playground_tab(self, st)`
- Defined: `qc_dashboard.py:1563`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_qasm_tab (method) `def _render_qasm_tab(self, st)`
- Defined: `qc_dashboard.py:1606`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_entropy_tab (method) `def _render_entropy_tab(self, st)`
- Defined: `qc_dashboard.py:1669`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_entanglement_tab (method) `def _render_entanglement_tab(self, st)`
- Defined: `qc_dashboard.py:1688`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_molecule_tab (method) `def _render_molecule_tab(self, st)`
- Defined: `qc_dashboard.py:1732`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_3d_tab (method) `def _render_3d_tab(self, st)`
- Defined: `qc_dashboard.py:1803`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _render_orbital_tab (method) `def _render_orbital_tab(self, st)`
- Defined: `qc_dashboard.py:1855`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _auto_run (method) `def _auto_run(self, st)`
- Defined: `qc_dashboard.py:1958`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _build_qasm_from_gates (method) `def _build_qasm_from_gates(self, gates)`
- Defined: `qc_dashboard.py:1969`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

### _parse_orb (method) `def _parse_orb(k)`
- Defined: `qc_dashboard.py:1936`
- Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
- Imported by: `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_qc_integration.py`

## qc_integration.py

### main (method) `def main()`
- Defined: `qc_integration.py:851`
- Doc: Command-line interface for the integration bridge.
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### __post_init__ (method) `def __post_init__(self)`
- Defined: `qc_integration.py:115`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### __post_init__ (method) `def __post_init__(self)`
- Defined: `qc_integration.py:137`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### num_qubits (method) `def num_qubits(self)`
- Defined: `qc_integration.py:141`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### append (method) `def append(self, gate)`
- Defined: `qc_integration.py:156`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### __len__ (method) `def __len__(self)`
- Defined: `qc_integration.py:164`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### __repr__ (method) `def __repr__(self)`
- Defined: `qc_integration.py:167`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### export (method) `def export(self, circuit)`
- Defined: `qc_integration.py:185`
- Doc: Export a CircuitIR to the target format.
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### import_ (method) `def import_(self, data)`
- Defined: `qc_integration.py:189`
- Doc: Import from the target format into a CircuitIR.
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_integration.py:238`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### export (method) `def export(self, circuit)`
- Defined: `qc_integration.py:243`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### import_ (method) `def import_(self, data)`
- Defined: `qc_integration.py:277`
- Doc: Parse an OpenQASM 2.0 string into a CircuitIR.
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _param_names_for_gate (method) `def _param_names_for_gate(name)`
- Defined: `qc_integration.py:339`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_integration.py:361`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### export (method) `def export(self, circuit)`
- Defined: `qc_integration.py:364`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### import_ (method) `def import_(self, data)`
- Defined: `qc_integration.py:379`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _ensure_qiskit (method) `def _ensure_qiskit()`
- Defined: `qc_integration.py:403`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _build_qiskit_method_map (method) `def _build_qiskit_method_map(qc)`
- Defined: `qc_integration.py:414`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _build_reverse_gate_map (method) `def _build_reverse_gate_map()`
- Defined: `qc_integration.py:433`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_integration.py:458`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### export (method) `def export(self, circuit)`
- Defined: `qc_integration.py:461`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### import_ (method) `def import_(self, data)`
- Defined: `qc_integration.py:483`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _ensure_pennylane (method) `def _ensure_pennylane()`
- Defined: `qc_integration.py:506`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _build_gate_ops (method) `def _build_gate_ops(pl)`
- Defined: `qc_integration.py:517`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _build_reverse_ops (method) `def _build_reverse_ops(pl)`
- Defined: `qc_integration.py:535`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_integration.py:565`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### to_circuit_ir (method) `def to_circuit_ir(self, circuit, framework_type)`
- Defined: `qc_integration.py:568`
- Doc: Extract a CircuitIR from a framework circuit object.
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### from_circuit_ir (method) `def from_circuit_ir(self, cir, framework_type)`
- Defined: `qc_integration.py:584`
- Doc: Build a framework circuit object from a CircuitIR.
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _detect_framework (method) `def _detect_framework(circuit)`
- Defined: `qc_integration.py:598`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _from_mps_circuit (method) `def _from_mps_circuit(circuit)`
- Defined: `qc_integration.py:607`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _to_mps_circuit (method) `def _to_mps_circuit(cir)`
- Defined: `qc_integration.py:623`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _from_sv_circuit (method) `def _from_sv_circuit(circuit)`
- Defined: `qc_integration.py:631`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _to_sv_circuit (method) `def _to_sv_circuit(cir)`
- Defined: `qc_integration.py:647`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### bell_state (method) `def bell_state()`
- Defined: `qc_integration.py:663`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### ghz_state (method) `def ghz_state(n_qubits)`
- Defined: `qc_integration.py:670`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### qft (method) `def qft(n_qubits)`
- Defined: `qc_integration.py:678`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### w_state (method) `def w_state(n_qubits)`
- Defined: `qc_integration.py:693`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### grover (method) `def grover(n_qubits, marked, iterations)`
- Defined: `qc_integration.py:701`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `qc_integration.py:753`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### _init_optional_adapters (method) `def _init_optional_adapters(self)`
- Defined: `qc_integration.py:763`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### export_qasm (method) `def export_qasm(self, circuit)`
- Defined: `qc_integration.py:780`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### import_qasm (method) `def import_qasm(self, qasm_str)`
- Defined: `qc_integration.py:783`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### to_qiskit (method) `def to_qiskit(self, circuit)`
- Defined: `qc_integration.py:788`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### from_qiskit (method) `def from_qiskit(self, qiskit_circuit)`
- Defined: `qc_integration.py:793`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### to_pennylane (method) `def to_pennylane(self, circuit)`
- Defined: `qc_integration.py:800`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### from_pennylane (method) `def from_pennylane(self, pennylane_data)`
- Defined: `qc_integration.py:805`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### to_circuit_ir (method) `def to_circuit_ir(self, circuit, framework_type)`
- Defined: `qc_integration.py:812`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### from_circuit_ir (method) `def from_circuit_ir(self, cir, framework_type)`
- Defined: `qc_integration.py:819`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### export_qasm_from_framework (method) `def export_qasm_from_framework(self, circuit, framework_type)`
- Defined: `qc_integration.py:828`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### import_qasm_to_framework (method) `def import_qasm_to_framework(self, qasm_str, framework_type)`
- Defined: `qc_integration.py:837`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

### circuit_fn (method) `def circuit_fn()`
- Defined: `qc_integration.py:467`
- Depends on: `quantum_computer.py`, `quantum_framework_core.py`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `test_qc_integration.py`

## quantum_3dview.py

### demo_brutal (method) `def demo_brutal()`
- Defined: `quantum_3dview.py:614`
- Doc: Demostración de visualización brutal
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### create_synthetic_snapshots (method) `def create_synthetic_snapshots()`
- Defined: `quantum_3dview.py:653`
- Doc: Crea datos sintéticos para demostración
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### colors (method) `def colors(self)`
- Defined: `quantum_3dview.py:52`
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_3dview.py:87`
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### create_amplitude_hologram (method) `def create_amplitude_hologram(self, snapshots, backend_comparison)`
- Defined: `quantum_3dview.py:92`
- Doc: Crea visualización holográfica de amplitudes en 3D con efecto de partículas
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _add_holographic_field (method) `def _add_holographic_field(self, fig, snapshots, row, col)`
- Defined: `quantum_3dview.py:148`
- Doc: Añade campo de amplitudes 3D con efecto de partículas flotantes
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _add_entropy_trails (method) `def _add_entropy_trails(self, fig, snapshots, row, col)`
- Defined: `quantum_3dview.py:245`
- Doc: Añade trazas de entropía con efecto de cola luminosa
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _add_bloch_sphere_holographic (method) `def _add_bloch_sphere_holographic(self, fig, snapshot, backend_name, row, col)`
- Defined: `quantum_3dview.py:292`
- Doc: Esfera de Bloch con efectos de holograma y partículas orbitales
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _interpolate_color (method) `def _interpolate_color(self, color1, color2, factor)`
- Defined: `quantum_3dview.py:389`
- Doc: Interpola entre dos colores hex
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, model)`
- Defined: `quantum_3dview.py:406`
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### create_topology_map (method) `def create_topology_map(self)`
- Defined: `quantum_3dview.py:410`
- Doc: Crea mapa 3D de la arquitectura neuronal con activaciones en tiempo real
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _extract_layers (method) `def _extract_layers(self, model)`
- Defined: `quantum_3dview.py:477`
- Doc: Extrae información de capas del modelo PyTorch
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _generate_quantum_topology (method) `def _generate_quantum_topology(self)`
- Defined: `quantum_3dview.py:490`
- Doc: Genera topología representativa de backend cuántico
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, sample_rate)`
- Defined: `quantum_3dview.py:503`
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### state_to_audio (method) `def state_to_audio(self, snapshot, duration)`
- Defined: `quantum_3dview.py:506`
- Doc: Convierte un estado cuántico en onda de audio
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _adsr_envelope (method) `def _adsr_envelope(self, length, intensity)`
- Defined: `quantum_3dview.py:538`
- Doc: Genera envolvente ADSR proporcional a la intensidad del estado
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_3dview.py:563`
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### generate_full_report (method) `def generate_full_report(self, snapshots, backend_comparison)`
- Defined: `quantum_3dview.py:570`
- Doc: Genera reporte completo con múltiples visualizaciones
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _save_audio (method) `def _save_audio(self, audio, path)`
- Defined: `quantum_3dview.py:605`
- Doc: Guarda audio como WAV
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### hex_to_rgb (method) `def hex_to_rgb(hex_color)`
- Defined: `quantum_3dview.py:391`
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### rgb_to_hex (method) `def rgb_to_hex(rgb)`
- Defined: `quantum_3dview.py:395`
- Depends on: `quantum_computer.py`, `quantum_visualizer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

## quantum_computer.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `quantum_computer.py:58`
- Doc: Create a module-level logger with a consistent formatter.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _solve_eigenstate (method) `def _solve_eigenstate(config, potential, n)`
- Defined: `quantum_computer.py:440`
- Doc: Solve the 1D marginal Hamiltonian and return the n-th eigenstate
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _build_basis_amplitude (method) `def _build_basis_amplitude(config, basis_idx)`
- Defined: `quantum_computer.py:461`
- Doc: Build the (2, G, G) spatial wavefunction for amplitude at basis index basis_idx.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _single_qubit_unitary (method) `def _single_qubit_unitary(state, qubit, u, backend)`
- Defined: `quantum_computer.py:739`
- Doc: Apply a 2x2 unitary u to qubit j in the joint Hilbert space.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _two_qubit_unitary (method) `def _two_qubit_unitary(state, ctrl, tgt, u4)`
- Defined: `quantum_computer.py:789`
- Doc: Apply a 4x4 unitary in the {|00>,|01>,|10>,|11>} subspace of (ctrl, tgt).
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### register_gate (method) `def register_gate(name, gate)`
- Defined: `quantum_computer.py:1179`
- Doc: Register a custom gate without modifying existing code (Open/Closed Principle).
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _check (method) `def _check(label, condition)`
- Defined: `quantum_computer.py:1602`
- Doc: Print PASS/FAIL for a single assertion. Returns True if passed.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### run_phase_tests (method) `def run_phase_tests(config)`
- Defined: `quantum_computer.py:1609`
- Doc: Property-based test suite for quantum phase coherence and unitarity.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _demo (method) `def _demo(config)`
- Defined: `quantum_computer.py:1836`
- Doc: Run demo suite validating entanglement on all backends.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, channels, grid_size)`
- Defined: `quantum_computer.py:113`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_computer.py:124`
- Doc: Apply spectral convolution via RFFT2.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `quantum_computer.py:150`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_computer.py:159`
- Doc: Accepts (G,G), (1,G,G), or (B,1,G,G). Returns squeezed output.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- Defined: `quantum_computer.py:176`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_computer.py:188`
- Doc: (2,G,G) or (B,2,G,G) -> same shape.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- Defined: `quantum_computer.py:205`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_computer.py:217`
- Doc: (8,G,G) or (B,8,G,G) -> same shape.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, representation, device)`
- Defined: `quantum_computer.py:236`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _init_matrices (method) `def _init_matrices(self)`
- Defined: `quantum_computer.py:241`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### to (method) `def to(self, device)`
- Defined: `quantum_computer.py:272`
- Doc: Move all matrices to device.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, amplitudes, n_qubits)`
- Defined: `quantum_computer.py:305`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### normalize_ (method) `def normalize_(self)`
- Defined: `quantum_computer.py:317`
- Doc: In-place normalization: sum_k P(k) = 1.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### probabilities (method) `def probabilities(self)`
- Defined: `quantum_computer.py:323`
- Doc: Return (2^n,) tensor of Born probabilities P(k) for each basis state.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### marginal_probability_one (method) `def marginal_probability_one(self, qubit)`
- Defined: `quantum_computer.py:328`
- Doc: Marginal Born probability P(qubit_j = |1>).
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### most_probable_basis_state (method) `def most_probable_basis_state(self)`
- Defined: `quantum_computer.py:343`
- Doc: Return the index k with the highest probability.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### bloch_vector (method) `def bloch_vector(self, qubit)`
- Defined: `quantum_computer.py:347`
- Doc: Compute the reduced Bloch vector for qubit j by partial trace.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### clone (method) `def clone(self)`
- Defined: `quantum_computer.py:385`
- Doc: Return a deep copy.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_computer.py:393`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _grid (method) `def _grid(self)`
- Defined: `quantum_computer.py:397`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### harmonic (method) `def harmonic(self)`
- Defined: `quantum_computer.py:402`
- Doc: V = k/2 * r^2.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### double_well (method) `def double_well(self)`
- Defined: `quantum_computer.py:410`
- Doc: Double-well along x.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### coulomb (method) `def coulomb(self)`
- Defined: `quantum_computer.py:417`
- Doc: Coulomb-like V ~ -1/r.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### periodic_lattice (method) `def periodic_lattice(self)`
- Defined: `quantum_computer.py:424`
- Doc: Periodic cosine lattice.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### mixed (method) `def mixed(self, seed)`
- Defined: `quantum_computer.py:429`
- Doc: Dirichlet-weighted mixture of all four potentials.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_computer.py:477`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _empty (method) `def _empty(self, n_qubits)`
- Defined: `quantum_computer.py:480`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### all_zeros (method) `def all_zeros(self, n_qubits)`
- Defined: `quantum_computer.py:484`
- Doc: Initialize register in |00...0>.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### basis_state (method) `def basis_state(self, n_qubits, k)`
- Defined: `quantum_computer.py:492`
- Doc: Initialize register in computational basis state |k>.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### from_bitstring (method) `def from_bitstring(self, bitstring)`
- Defined: `quantum_computer.py:502`
- Doc: Initialize in the basis state given by binary string.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_computer.py:515`
- Doc: Evolve a single (2, G, G) wavefunction by dt under H.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_computer.py:519`
- Doc: Apply global phase e^{i*phi} to a (2, G, G) amplitude.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_computer.py:531`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _load (method) `def _load(self)`
- Defined: `quantum_computer.py:539`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _precompute_laplacian (method) `def _precompute_laplacian(self)`
- Defined: `quantum_computer.py:558`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _apply_h (method) `def _apply_h(self, field)`
- Defined: `quantum_computer.py:565`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_computer.py:573`
- Doc: dpsi/dt = -i H psi  =>  psi' = psi + dt * (-i H psi) = psi + dt*(H_i*r - H_r*i).
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_computer.py:585`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, config, hamiltonian)`
- Defined: `quantum_computer.py:599`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _load (method) `def _load(self)`
- Defined: `quantum_computer.py:606`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_computer.py:628`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_computer.py:636`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, config, hamiltonian)`
- Defined: `quantum_computer.py:648`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _load (method) `def _load(self)`
- Defined: `quantum_computer.py:657`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _precompute_dirac (method) `def _precompute_dirac(self)`
- Defined: `quantum_computer.py:679`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _pack (method) `def _pack(self, amp)`
- Defined: `quantum_computer.py:690`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _unpack (method) `def _unpack(self, spinor)`
- Defined: `quantum_computer.py:701`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _analytical_dirac (method) `def _analytical_dirac(self, spinor)`
- Defined: `quantum_computer.py:707`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_computer.py:722`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_computer.py:735`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:849`
- Doc: Gate identifier.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:853`
- Doc: Apply gate to joint state, return new joint state.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:867`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:870`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:882`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:885`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:896`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:899`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:910`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:913`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:924`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:927`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:938`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:941`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:953`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:956`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:969`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:972`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:985`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:988`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:1007`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:1010`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:1030`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:1033`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:1053`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:1056`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:1076`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:1079`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:1115`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:1118`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### name (method) `def name(self)`
- Defined: `quantum_computer.py:1139`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_computer.py:1142`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, n_qubits)`
- Defined: `quantum_computer.py:1200`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### h (method) `def h(self, q)`
- Defined: `quantum_computer.py:1206`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### x (method) `def x(self, q)`
- Defined: `quantum_computer.py:1209`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### y (method) `def y(self, q)`
- Defined: `quantum_computer.py:1212`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### z (method) `def z(self, q)`
- Defined: `quantum_computer.py:1215`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### s (method) `def s(self, q)`
- Defined: `quantum_computer.py:1218`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### t (method) `def t(self, q)`
- Defined: `quantum_computer.py:1221`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### rx (method) `def rx(self, q, theta)`
- Defined: `quantum_computer.py:1224`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### ry (method) `def ry(self, q, theta)`
- Defined: `quantum_computer.py:1227`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### rz (method) `def rz(self, q, theta)`
- Defined: `quantum_computer.py:1230`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### cnot (method) `def cnot(self, ctrl, tgt)`
- Defined: `quantum_computer.py:1233`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### cx (method) `def cx(self, ctrl, tgt)`
- Defined: `quantum_computer.py:1236`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### cz (method) `def cz(self, ctrl, tgt)`
- Defined: `quantum_computer.py:1239`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### swap (method) `def swap(self, a, b)`
- Defined: `quantum_computer.py:1242`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### toffoli (method) `def toffoli(self, c0, c1, tgt)`
- Defined: `quantum_computer.py:1245`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### ccx (method) `def ccx(self, c0, c1, tgt)`
- Defined: `quantum_computer.py:1248`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### evolve (method) `def evolve(self, qubits, dt, steps)`
- Defined: `quantum_computer.py:1251`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### barrier (method) `def barrier(self)`
- Defined: `quantum_computer.py:1254`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _append (method) `def _append(self, gate_name, targets, params)`
- Defined: `quantum_computer.py:1257`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### depth (method) `def depth(self)`
- Defined: `quantum_computer.py:1265`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __len__ (method) `def __len__(self)`
- Defined: `quantum_computer.py:1268`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __repr__ (method) `def __repr__(self)`
- Defined: `quantum_computer.py:1271`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### probabilities (method) `def probabilities(self)`
- Defined: `quantum_computer.py:1293`
- Doc: Alias: marginal P(|1>) per qubit index.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### most_probable_bitstring (method) `def most_probable_bitstring(self)`
- Defined: `quantum_computer.py:1297`
- Doc: Return the bitstring with the highest probability.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### expectation_z (method) `def expectation_z(self, qubit)`
- Defined: `quantum_computer.py:1301`
- Doc: <Z>_j = P(0) - P(1) in [-1, +1].
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### entropy (method) `def entropy(self)`
- Defined: `quantum_computer.py:1305`
- Doc: Shannon entropy of the full probability distribution in bits.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __repr__ (method) `def __repr__(self)`
- Defined: `quantum_computer.py:1313`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_computer.py:1361`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _select_backend (method) `def _select_backend(self, name)`
- Defined: `quantum_computer.py:1375`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### _state_to_result (method) `def _state_to_result(self, state)`
- Defined: `quantum_computer.py:1380`
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### run (method) `def run(self, circuit, backend, initial_states)`
- Defined: `quantum_computer.py:1388`
- Doc: Execute a quantum circuit on the joint Hilbert space.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### run_with_state_snapshots (method) `def run_with_state_snapshots(self, circuit, backend, snapshot_after)`
- Defined: `quantum_computer.py:1420`
- Doc: Execute circuit with non-destructive probability snapshots.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### bell_state (method) `def bell_state(self, backend)`
- Defined: `quantum_computer.py:1449`
- Doc: |Phi+> = (|00> + |11>) / sqrt(2).
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### ghz_state (method) `def ghz_state(self, n_qubits, backend)`
- Defined: `quantum_computer.py:1459`
- Doc: (|00...0> + |11...1>) / sqrt(2).
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### quantum_fourier_transform (method) `def quantum_fourier_transform(self, n_qubits, backend)`
- Defined: `quantum_computer.py:1471`
- Doc: QFT on |00...0>. Standard H + controlled-Rz decomposition.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### grover_oracle_search (method) `def grover_oracle_search(self, n_qubits, target_bitstring, backend, n_iterations)`
- Defined: `quantum_computer.py:1481`
- Doc: Grover's search algorithm with correct phase oracle and diffusion operator.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### variational_ansatz (method) `def variational_ansatz(self, n_qubits, n_layers, thetas, backend)`
- Defined: `quantum_computer.py:1559`
- Doc: Hardware-efficient ansatz: Ry layers + CNOT chain. len(thetas)=n_qubits*n_layers.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### teleportation (method) `def teleportation(self, backend)`
- Defined: `quantum_computer.py:1572`
- Doc: 3-qubit teleportation protocol.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

### deutsch_jozsa (method) `def deutsch_jozsa(self, n_input_qubits, is_constant, backend)`
- Defined: `quantum_computer.py:1585`
- Doc: Deutsch-Jozsa: constant -> all inputs |0>, balanced -> at least one |1>.
- Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `molecular_sim.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`, `topological_hilbert_compression2.py`

## quantum_dash.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `quantum_dash.py:91`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### main (method) `def main()`
- Defined: `quantum_dash.py:1199`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### colors (method) `def colors(self)`
- Defined: `quantum_dash.py:168`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### plotly_template (method) `def plotly_template(self)`
- Defined: `quantum_dash.py:234`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### render (method) `def render(self, data, axes, config)`
- Defined: `quantum_dash.py:277`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### render (method) `def render(self, snapshot, axes, config)`
- Defined: `quantum_dash.py:282`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _render_empty (method) `def _render_empty(self, axes, config)`
- Defined: `quantum_dash.py:318`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _generate_colors (method) `def _generate_colors(self, probs, config)`
- Defined: `quantum_dash.py:326`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### render (method) `def render(self, snapshot, axes, config)`
- Defined: `quantum_dash.py:341`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _render_empty_sphere (method) `def _render_empty_sphere(self, axes, config)`
- Defined: `quantum_dash.py:375`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _draw_sphere_wireframe (method) `def _draw_sphere_wireframe(self, axes, config)`
- Defined: `quantum_dash.py:381`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _draw_axes (method) `def _draw_axes(self, axes, config)`
- Defined: `quantum_dash.py:402`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _draw_bloch_vector (method) `def _draw_bloch_vector(self, axes, bx, by, bz, color, qubit_idx, config)`
- Defined: `quantum_dash.py:417`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _draw_uncertainty_ring (method) `def _draw_uncertainty_ring(self, axes, bx, by, bz, color, config)`
- Defined: `quantum_dash.py:431`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### render (method) `def render(self, snapshot, axes, config)`
- Defined: `quantum_dash.py:448`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _render_empty (method) `def _render_empty(self, axes, config)`
- Defined: `quantum_dash.py:495`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### render (method) `def render(self, snapshots, axes, config)`
- Defined: `quantum_dash.py:510`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _render_empty (method) `def _render_empty(self, axes, config)`
- Defined: `quantum_dash.py:556`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _interpolate_colors (method) `def _interpolate_colors(self, color1, color2, n)`
- Defined: `quantum_dash.py:564`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _hex_to_rgb (method) `def _hex_to_rgb(self, hex_color)`
- Defined: `quantum_dash.py:578`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### render (method) `def render(self, comparisons, axes, config)`
- Defined: `quantum_dash.py:584`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _render_empty (method) `def _render_empty(self, axes, config)`
- Defined: `quantum_dash.py:624`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### render (method) `def render(self, comparisons, axes, config)`
- Defined: `quantum_dash.py:634`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _render_empty (method) `def _render_empty(self, axes, config)`
- Defined: `quantum_dash.py:665`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_dash.py:675`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### compute_probabilities (method) `def compute_probabilities(self, state)`
- Defined: `quantum_dash.py:678`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### compute_phases (method) `def compute_phases(self, state)`
- Defined: `quantum_dash.py:684`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### compute_entropy (method) `def compute_entropy(self, probs)`
- Defined: `quantum_dash.py:696`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### compute_bloch_vectors (method) `def compute_bloch_vectors(self, state)`
- Defined: `quantum_dash.py:704`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### create_snapshot (method) `def create_snapshot(self, state, step, gate_name, backend_name)`
- Defined: `quantum_dash.py:713`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### bell_state (method) `def bell_state()`
- Defined: `quantum_dash.py:756`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### ghz_state (method) `def ghz_state(n_qubits)`
- Defined: `quantum_dash.py:760`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### qft (method) `def qft(n_qubits)`
- Defined: `quantum_dash.py:767`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### grover_oracle (method) `def grover_oracle(n_qubits, marked)`
- Defined: `quantum_dash.py:782`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### grover_diffusion (method) `def grover_diffusion(n_qubits)`
- Defined: `quantum_dash.py:794`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, qc, config)`
- Defined: `quantum_dash.py:807`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### execute_sequence (method) `def execute_sequence(self, gates, n_qubits, backend_name)`
- Defined: `quantum_dash.py:812`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### compare_backends (method) `def compare_backends(self, gates, n_qubits)`
- Defined: `quantum_dash.py:835`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_dash.py:874`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### build_full_figure (method) `def build_full_figure(self, snapshots, comparisons)`
- Defined: `quantum_dash.py:883`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### build_summary_figure (method) `def build_summary_figure(self, snapshots, comparisons)`
- Defined: `quantum_dash.py:923`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_dash.py:959`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _initialize (method) `def _initialize(self)`
- Defined: `quantum_dash.py:966`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _init_quantum_computer (method) `def _init_quantum_computer(self)`
- Defined: `quantum_dash.py:976`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### visualize_bell_state (method) `def visualize_bell_state(self)`
- Defined: `quantum_dash.py:1010`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### visualize_ghz_state (method) `def visualize_ghz_state(self, n_qubits)`
- Defined: `quantum_dash.py:1016`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### visualize_qft (method) `def visualize_qft(self, n_qubits)`
- Defined: `quantum_dash.py:1022`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### visualize_grover (method) `def visualize_grover(self, n_qubits, marked_state)`
- Defined: `quantum_dash.py:1028`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _execute_and_visualize (method) `def _execute_and_visualize(self, gates, n_qubits, name)`
- Defined: `quantum_dash.py:1043`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _create_synthetic_snapshots (method) `def _create_synthetic_snapshots(self, n_qubits, gates)`
- Defined: `quantum_dash.py:1086`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _create_synthetic_comparisons (method) `def _create_synthetic_comparisons(self, n_qubits, gates)`
- Defined: `quantum_dash.py:1129`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _save_data (method) `def _save_data(self, path, snapshots, comparisons)`
- Defined: `quantum_dash.py:1153`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### run_all (method) `def run_all(self)`
- Defined: `quantum_dash.py:1174`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

### _print_summary (method) `def _print_summary(self, results)`
- Defined: `quantum_dash.py:1189`
- Depends on: `molecular_sim.py`, `quantum_computer.py`
- Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`

## quantum_framework_core.py

### _make_logger (function) `def _make_logger(name, level)`
- Defined: `quantum_framework_core.py:46`
- Doc: Create a configured logger instance.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### run_scaling_benchmark (method) `def run_scaling_benchmark(config, max_qubits)`
- Defined: `quantum_framework_core.py:2045`
- Doc: Run scaling benchmark to demonstrate MPS memory efficiency.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### run_grover_search (method) `def run_grover_search(qc, n_qubits, marked_states)`
- Defined: `quantum_framework_core.py:2098`
- Doc: Run Grover's search algorithm and return results.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __post_init__ (method) `def __post_init__(self)`
- Defined: `quantum_framework_core.py:139`
- Doc: Initialize random seeds after configuration.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### from_toml (method) `def from_toml(cls, toml_path)`
- Defined: `quantum_framework_core.py:147`
- Doc: Load configuration from TOML file.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, config_path)`
- Defined: `quantum_framework_core.py:265`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _find_config (method) `def _find_config(self)`
- Defined: `quantum_framework_core.py:274`
- Doc: Find configuration file in standard locations.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _load (method) `def _load(self)`
- Defined: `quantum_framework_core.py:286`
- Doc: Load configuration from TOML file.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _load_defaults (method) `def _load_defaults(self)`
- Defined: `quantum_framework_core.py:300`
- Doc: Load default configuration values.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _parse_atoms (method) `def _parse_atoms(self)`
- Defined: `quantum_framework_core.py:318`
- Doc: Parse atoms from configuration data.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _parse_molecules (method) `def _parse_molecules(self)`
- Defined: `quantum_framework_core.py:332`
- Doc: Parse molecules from configuration data.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _parse_orbitals (method) `def _parse_orbitals(self)`
- Defined: `quantum_framework_core.py:352`
- Doc: Parse orbitals from configuration data.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _parse_experiments (method) `def _parse_experiments(self)`
- Defined: `quantum_framework_core.py:364`
- Doc: Parse experiments from configuration data.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### get_atom (method) `def get_atom(self, symbol)`
- Defined: `quantum_framework_core.py:375`
- Doc: Get atom data by symbol (case-insensitive).
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### get_molecule (method) `def get_molecule(self, name)`
- Defined: `quantum_framework_core.py:384`
- Doc: Get molecule data by name (case-insensitive).
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### get_orbital (method) `def get_orbital(self, name)`
- Defined: `quantum_framework_core.py:393`
- Doc: Get orbital data by name (case-insensitive).
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### get_experiment (method) `def get_experiment(self, name)`
- Defined: `quantum_framework_core.py:402`
- Doc: Get experiment data by name.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### atoms (method) `def atoms(self)`
- Defined: `quantum_framework_core.py:407`
- Doc: Return all atoms.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### molecules (method) `def molecules(self)`
- Defined: `quantum_framework_core.py:412`
- Doc: Return all molecules.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### orbitals (method) `def orbitals(self)`
- Defined: `quantum_framework_core.py:417`
- Doc: Return all orbitals.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### experiments (method) `def experiments(self)`
- Defined: `quantum_framework_core.py:422`
- Doc: Return all experiments.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### get_molecules_by_qubits (method) `def get_molecules_by_qubits(self, max_qubits)`
- Defined: `quantum_framework_core.py:426`
- Doc: Get molecules that fit within qubit budget.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### get_atoms_by_qubits (method) `def get_atoms_by_qubits(self, max_qubits)`
- Defined: `quantum_framework_core.py:430`
- Doc: Get atoms that fit within qubit budget.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### n_qubits (method) `def n_qubits(self)`
- Defined: `quantum_framework_core.py:440`
- Doc: Return number of qubits.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### amplitude (method) `def amplitude(self, basis_index)`
- Defined: `quantum_framework_core.py:445`
- Doc: Compute amplitude for a computational basis state.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply_single_qubit_gate (method) `def apply_single_qubit_gate(self, qubit, gate)`
- Defined: `quantum_framework_core.py:450`
- Doc: Apply single-qubit gate in-place.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply_two_qubit_gate (method) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)`
- Defined: `quantum_framework_core.py:455`
- Doc: Apply two-qubit gate in-place.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### norm (method) `def norm(self)`
- Defined: `quantum_framework_core.py:460`
- Doc: Compute state norm.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### probabilities (method) `def probabilities(self)`
- Defined: `quantum_framework_core.py:465`
- Doc: Compute measurement probabilities.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### entropy (method) `def entropy(self)`
- Defined: `quantum_framework_core.py:470`
- Doc: Compute von Neumann entropy.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### memory_bytes (method) `def memory_bytes(self)`
- Defined: `quantum_framework_core.py:475`
- Doc: Return memory usage in bytes.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, chi_left, chi_right, d, device, dtype)`
- Defined: `quantum_framework_core.py:488`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _initialize (method) `def _initialize(self)`
- Defined: `quantum_framework_core.py:504`
- Doc: Initialize core tensor for |0> product state (exact).
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### tensor (method) `def tensor(self)`
- Defined: `quantum_framework_core.py:517`
- Doc: Return the core tensor.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### tensor (method) `def tensor(self, value)`
- Defined: `quantum_framework_core.py:524`
- Doc: Set the core tensor, preserving complex dtype when needed.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### left_canonicalize (method) `def left_canonicalize(self)`
- Defined: `quantum_framework_core.py:533`
- Doc: Bring core to left-canonical form, return singular values.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### right_canonicalize (method) `def right_canonicalize(self)`
- Defined: `quantum_framework_core.py:543`
- Doc: Bring core to right-canonical form, return singular values.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, n_qubits, config)`
- Defined: `quantum_framework_core.py:567`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _initialize (method) `def _initialize(self)`
- Defined: `quantum_framework_core.py:576`
- Doc: Initialize MPS with product state |00...0>.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### n_qubits (method) `def n_qubits(self)`
- Defined: `quantum_framework_core.py:600`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _bond_dimension (method) `def _bond_dimension(self, site)`
- Defined: `quantum_framework_core.py:603`
- Doc: Compute bond dimension at given site.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### amplitude (method) `def amplitude(self, basis_index)`
- Defined: `quantum_framework_core.py:610`
- Doc: Compute amplitude for computational basis state.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply_single_qubit_gate (method) `def apply_single_qubit_gate(self, qubit, gate)`
- Defined: `quantum_framework_core.py:631`
- Doc: Apply single-qubit gate in-place.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply_two_qubit_gate (method) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)`
- Defined: `quantum_framework_core.py:655`
- Doc: Apply two-qubit gate in-place.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _swap_qubits_in_gate (method) `def _swap_qubits_in_gate(self, gate)`
- Defined: `quantum_framework_core.py:674`
- Doc: Swap qubit ordering in two-qubit gate.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _apply_adjacent_gate (method) `def _apply_adjacent_gate(self, qubit, gate)`
- Defined: `quantum_framework_core.py:684`
- Doc: Apply gate to adjacent qubit pair.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _apply_nonadjacent_gate (method) `def _apply_nonadjacent_gate(self, qubit_a, qubit_b, gate)`
- Defined: `quantum_framework_core.py:747`
- Doc: Apply gate to non-adjacent qubit pair using SWAP network.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### norm (method) `def norm(self)`
- Defined: `quantum_framework_core.py:762`
- Doc: Compute state norm.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _canonicalize (method) `def _canonicalize(self)`
- Defined: `quantum_framework_core.py:770`
- Doc: Bring MPS to canonical form.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### probabilities (method) `def probabilities(self)`
- Defined: `quantum_framework_core.py:778`
- Doc: Compute measurement probabilities.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### entropy (method) `def entropy(self)`
- Defined: `quantum_framework_core.py:795`
- Doc: Compute maximum entanglement entropy across all cuts.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### memory_bytes (method) `def memory_bytes(self)`
- Defined: `quantum_framework_core.py:822`
- Doc: Return memory usage in bytes.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### entanglement_entropy (method) `def entanglement_entropy(self, cut)`
- Defined: `quantum_framework_core.py:829`
- Doc: Compute entanglement entropy at given cut between qubits cut-1 and cut.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### to_statevector (method) `def to_statevector(self)`
- Defined: `quantum_framework_core.py:874`
- Doc: Convert MPS to full statevector (only for small systems).
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### most_probable_bitstring (method) `def most_probable_bitstring(self)`
- Defined: `quantum_framework_core.py:890`
- Doc: Return most probable basis state as bitstring.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### clone (method) `def clone(self)`
- Defined: `quantum_framework_core.py:896`
- Doc: Return a deep copy.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, n_qubits, config)`
- Defined: `quantum_framework_core.py:924`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _initialize (method) `def _initialize(self)`
- Defined: `quantum_framework_core.py:935`
- Doc: Initialize vacuum core with ground state.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _compute_berry_phases (method) `def _compute_berry_phases(self)`
- Defined: `quantum_framework_core.py:941`
- Doc: Compute Berry phases between active states.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### add_active_state (method) `def add_active_state(self, basis_index, winding_number)`
- Defined: `quantum_framework_core.py:948`
- Doc: Add a basis state to the active subspace.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _compute_winding_number (method) `def _compute_winding_number(self, basis_index)`
- Defined: `quantum_framework_core.py:962`
- Doc: Compute winding number for a basis state.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### is_topologically_protected (method) `def is_topologically_protected(self, basis_index)`
- Defined: `quantum_framework_core.py:968`
- Doc: Check if state is topologically protected.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### sparsity (method) `def sparsity(self)`
- Defined: `quantum_framework_core.py:973`
- Doc: Compute vacuum sparsity.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### project_to_active (method) `def project_to_active(self, state)`
- Defined: `quantum_framework_core.py:979`
- Doc: Project state onto active subspace.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_framework_core.py:998`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### compute_winding_number (method) `def compute_winding_number(self, state, qubit)`
- Defined: `quantum_framework_core.py:1003`
- Doc: Compute winding number for a qubit.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### compute_berry_phase (method) `def compute_berry_phase(self, state, qubit_a, qubit_b)`
- Defined: `quantum_framework_core.py:1015`
- Doc: Compute Berry phase between two qubits.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### is_protected (method) `def is_protected(self, state, vacuum_core)`
- Defined: `quantum_framework_core.py:1031`
- Doc: Check if state is topologically protected.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, channels, grid_size)`
- Defined: `quantum_framework_core.py:1043`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_framework_core.py:1053`
- Doc: Apply spectral convolution via RFFT2.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `quantum_framework_core.py:1079`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_framework_core.py:1088`
- Doc: Apply Hamiltonian backbone network.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- Defined: `quantum_framework_core.py:1105`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_framework_core.py:1119`
- Doc: Apply Schrodinger evolution network.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- Defined: `quantum_framework_core.py:1136`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_framework_core.py:1150`
- Doc: Apply Dirac evolution network.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, representation, device)`
- Defined: `quantum_framework_core.py:1167`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _init_matrices (method) `def _init_matrices(self)`
- Defined: `quantum_framework_core.py:1172`
- Doc: Initialize gamma matrices.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### to (method) `def to(self, device)`
- Defined: `quantum_framework_core.py:1208`
- Doc: Move all matrices to device.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_framework_core.py:1219`
- Doc: Evolve a single amplitude by time dt.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_framework_core.py:1224`
- Doc: Apply global phase to amplitude.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_framework_core.py:1237`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _load (method) `def _load(self)`
- Defined: `quantum_framework_core.py:1245`
- Doc: Load model from checkpoint.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _precompute_laplacian (method) `def _precompute_laplacian(self)`
- Defined: `quantum_framework_core.py:1266`
- Doc: Precompute Laplacian kernel for kinetic energy.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _apply_h (method) `def _apply_h(self, field)`
- Defined: `quantum_framework_core.py:1274`
- Doc: Apply Hamiltonian operator to field.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_framework_core.py:1284`
- Doc: Evolve amplitude by time dt.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_framework_core.py:1298`
- Doc: Apply global phase.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, config, hamiltonian)`
- Defined: `quantum_framework_core.py:1312`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _load (method) `def _load(self)`
- Defined: `quantum_framework_core.py:1319`
- Doc: Load model from checkpoint.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_framework_core.py:1342`
- Doc: Evolve amplitude by time dt.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_framework_core.py:1353`
- Doc: Apply global phase.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, config, hamiltonian)`
- Defined: `quantum_framework_core.py:1366`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _load (method) `def _load(self)`
- Defined: `quantum_framework_core.py:1375`
- Doc: Load model from checkpoint.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _precompute_dirac (method) `def _precompute_dirac(self)`
- Defined: `quantum_framework_core.py:1398`
- Doc: Precompute momentum grids for Dirac operator.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _pack (method) `def _pack(self, amp)`
- Defined: `quantum_framework_core.py:1407`
- Doc: Pack 2-channel amplitude to 4-component spinor.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _unpack (method) `def _unpack(self, spinor)`
- Defined: `quantum_framework_core.py:1421`
- Doc: Unpack 4-component spinor to 2-channel amplitude.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _analytical_dirac (method) `def _analytical_dirac(self, spinor)`
- Defined: `quantum_framework_core.py:1428`
- Doc: Apply analytical Dirac Hamiltonian to spinor.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_framework_core.py:1447`
- Doc: Evolve amplitude by time dt using Dirac equation.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_framework_core.py:1463`
- Doc: Apply global phase.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### evolve_spinor (method) `def evolve_spinor(self, spinor, dt)`
- Defined: `quantum_framework_core.py:1467`
- Doc: Evolve full 4-component spinor by time dt.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1481`
- Doc: Return gate name.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1486`
- Doc: Apply gate to state and return new state.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1500`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1503`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1515`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1518`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1529`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1532`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1543`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1546`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1557`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1560`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1571`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1574`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1586`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1589`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1602`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1605`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1618`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1621`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1635`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1638`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1660`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1663`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1680`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1683`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### name (method) `def name(self)`
- Defined: `quantum_framework_core.py:1700`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### apply (method) `def apply(self, state, targets, params)`
- Defined: `quantum_framework_core.py:1703`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, n_qubits)`
- Defined: `quantum_framework_core.py:1744`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _append (method) `def _append(self, gate_name, targets, params)`
- Defined: `quantum_framework_core.py:1748`
- Doc: Append an instruction to the circuit.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### h (method) `def h(self, qubit)`
- Defined: `quantum_framework_core.py:1759`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### x (method) `def x(self, qubit)`
- Defined: `quantum_framework_core.py:1762`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### y (method) `def y(self, qubit)`
- Defined: `quantum_framework_core.py:1765`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### z (method) `def z(self, qubit)`
- Defined: `quantum_framework_core.py:1768`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### s (method) `def s(self, qubit)`
- Defined: `quantum_framework_core.py:1771`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### t (method) `def t(self, qubit)`
- Defined: `quantum_framework_core.py:1774`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### rx (method) `def rx(self, qubit, theta)`
- Defined: `quantum_framework_core.py:1777`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### ry (method) `def ry(self, qubit, theta)`
- Defined: `quantum_framework_core.py:1780`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### rz (method) `def rz(self, qubit, theta)`
- Defined: `quantum_framework_core.py:1783`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### crz (method) `def crz(self, control, target, theta)`
- Defined: `quantum_framework_core.py:1786`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### cnot (method) `def cnot(self, control, target)`
- Defined: `quantum_framework_core.py:1789`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### cz (method) `def cz(self, control, target)`
- Defined: `quantum_framework_core.py:1792`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### swap (method) `def swap(self, qubit1, qubit2)`
- Defined: `quantum_framework_core.py:1795`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __len__ (method) `def __len__(self)`
- Defined: `quantum_framework_core.py:1798`
- Doc: Return number of instructions.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __bool__ (method) `def __bool__(self)`
- Defined: `quantum_framework_core.py:1802`
- Doc: True if circuit has instructions.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### run (method) `def run(self, state)`
- Defined: `quantum_framework_core.py:1806`
- Doc: Execute circuit on state.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_framework_core.py:1827`
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### create_circuit (method) `def create_circuit(self, n_qubits)`
- Defined: `quantum_framework_core.py:1843`
- Doc: Create a new quantum circuit.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### create_state (method) `def create_state(self, n_qubits)`
- Defined: `quantum_framework_core.py:1851`
- Doc: Create initial state |00...0>.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### bell_state (method) `def bell_state(self, n_qubits)`
- Defined: `quantum_framework_core.py:1865`
- Doc: Prepare Bell state |Phi+> = (|00> + |11>) / sqrt(2).
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### ghz_state (method) `def ghz_state(self, n_qubits)`
- Defined: `quantum_framework_core.py:1873`
- Doc: Prepare GHZ state (|00...0> + |11...1>) / sqrt(2).
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### w_state (method) `def w_state(self, n_qubits)`
- Defined: `quantum_framework_core.py:1882`
- Doc: Prepare W state: |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _build_w_state_direct (method) `def _build_w_state_direct(self, n_qubits, max_bond)`
- Defined: `quantum_framework_core.py:1905`
- Doc: Build W state using direct statevector-to-MPS conversion via successive SVD.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### run_circuit (method) `def run_circuit(self, circuit, initial_state)`
- Defined: `quantum_framework_core.py:1985`
- Doc: Execute circuit on state.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### get_backend (method) `def get_backend(self, name)`
- Defined: `quantum_framework_core.py:1995`
- Doc: Get physics backend by name.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### memory_usage (method) `def memory_usage(self, state)`
- Defined: `quantum_framework_core.py:1999`
- Doc: Compute memory usage for a state.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### compression_ratio (method) `def compression_ratio(self, state)`
- Defined: `quantum_framework_core.py:2012`
- Doc: Compute compression ratio vs full statevector.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### detect_phase (method) `def detect_phase(self, state)`
- Defined: `quantum_framework_core.py:2019`
- Doc: Detect Hilbert space phase from state properties.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

### _compute_average_bond_dimension (method) `def _compute_average_bond_dimension(self, state)`
- Defined: `quantum_framework_core.py:2037`
- Doc: Compute average bond dimension across MPS cores.
- Imported by: `qc_dashboard.py`, `qc_dashboard.py`, `qc_dashboard.py`, `qc_integration.py`, `qc_integration.py`, `quantum_framework_main.py`, `quantum_framework_menu.py`, `quantum_lab.py`, `test_qc_integration.py`, `test_qc_integration.py`, `test_quantum_framework.py`, `test_quantum_framework.py`, `test_quantum_framework.py`

## quantum_framework_main.py

### setup_logging (function) `def setup_logging(verbose)`
- Defined: `quantum_framework_main.py:52`
- Doc: Configure logging level based on verbosity.
- Depends on: `quantum_framework_core.py`, `quantum_framework_menu.py`, `quantum_lab.py`

### run_benchmark (function) `def run_benchmark(args, config)`
- Defined: `quantum_framework_main.py:61`
- Doc: Run scaling benchmark.
- Depends on: `quantum_framework_core.py`, `quantum_framework_menu.py`, `quantum_lab.py`

### run_experiment (function) `def run_experiment(args, config, config_loader)`
- Defined: `quantum_framework_main.py:101`
- Doc: Run a specific experiment by name.
- Depends on: `quantum_framework_core.py`, `quantum_framework_menu.py`, `quantum_lab.py`

### run_molecular_simulation (function) `def run_molecular_simulation(args, config, config_loader)`
- Defined: `quantum_framework_main.py:173`
- Doc: Run molecular simulation.
- Depends on: `quantum_framework_core.py`, `quantum_framework_menu.py`, `quantum_lab.py`

### run_orbital_visualization (function) `def run_orbital_visualization(args, config, config_loader)`
- Defined: `quantum_framework_main.py:210`
- Doc: Run orbital visualization.
- Depends on: `quantum_framework_core.py`, `quantum_framework_menu.py`, `quantum_lab.py`

### print_info (function) `def print_info(config, config_loader)`
- Defined: `quantum_framework_main.py:242`
- Doc: Print framework information.
- Depends on: `quantum_framework_core.py`, `quantum_framework_menu.py`, `quantum_lab.py`

### main (function) `def main()`
- Defined: `quantum_framework_main.py:292`
- Doc: Main entry point.
- Depends on: `quantum_framework_core.py`, `quantum_framework_menu.py`, `quantum_lab.py`

## quantum_framework_menu.py

### run_interactive_menu (method) `def run_interactive_menu(config, config_loader)`
- Defined: `quantum_framework_menu.py:2357`
- Doc: Run the interactive menu system.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### run_all_experiments (method) `def run_all_experiments(config, config_loader)`
- Defined: `quantum_framework_menu.py:2363`
- Doc: Run ALL experiments automatically for debugging.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### __init__ (method) `def __init__(self, config, config_loader)`
- Defined: `quantum_framework_menu.py:72`
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### clear_screen (method) `def clear_screen(self)`
- Defined: `quantum_framework_menu.py:79`
- Doc: Clear the terminal screen.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### print_header (method) `def print_header(self, title)`
- Defined: `quantum_framework_menu.py:83`
- Doc: Print formatted header.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### print_menu (method) `def print_menu(self, title, options)`
- Defined: `quantum_framework_menu.py:90`
- Doc: Print formatted menu with options.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### get_input (method) `def get_input(self, prompt)`
- Defined: `quantum_framework_menu.py:97`
- Doc: Get user input with history tracking.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### pause (method) `def pause(self, message)`
- Defined: `quantum_framework_menu.py:106`
- Doc: Wait for user to press Enter.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### run (method) `def run(self)`
- Defined: `quantum_framework_menu.py:113`
- Doc: Run the main menu loop.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_main_menu (method) `def _show_main_menu(self)`
- Defined: `quantum_framework_menu.py:118`
- Doc: Display main menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_circuit_menu (method) `def _show_circuit_menu(self)`
- Defined: `quantum_framework_menu.py:163`
- Doc: Display quantum circuits menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _custom_circuit (method) `def _custom_circuit(self)`
- Defined: `quantum_framework_menu.py:197`
- Doc: Create and run a custom circuit.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _bell_state_demo (method) `def _bell_state_demo(self)`
- Defined: `quantum_framework_menu.py:279`
- Doc: Demonstrate Bell state preparation.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _ghz_state_demo (method) `def _ghz_state_demo(self)`
- Defined: `quantum_framework_menu.py:299`
- Doc: Demonstrate GHZ state preparation.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _w_state_demo (method) `def _w_state_demo(self)`
- Defined: `quantum_framework_menu.py:328`
- Doc: Demonstrate W state preparation.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _single_qubit_gates_demo (method) `def _single_qubit_gates_demo(self)`
- Defined: `quantum_framework_menu.py:356`
- Doc: Demonstrate single qubit gates.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _two_qubit_gates_demo (method) `def _two_qubit_gates_demo(self)`
- Defined: `quantum_framework_menu.py:397`
- Doc: Demonstrate two qubit gates.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_entanglement_menu (method) `def _show_entanglement_menu(self)`
- Defined: `quantum_framework_menu.py:431`
- Doc: Display entanglement experiments menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _bell_entropy_experiment (method) `def _bell_entropy_experiment(self)`
- Defined: `quantum_framework_menu.py:459`
- Doc: Measure entropy of Bell states.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _ghz_scaling_experiment (method) `def _ghz_scaling_experiment(self)`
- Defined: `quantum_framework_menu.py:472`
- Doc: Study GHZ entanglement scaling with qubit count.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _entropy_by_cut_experiment (method) `def _entropy_by_cut_experiment(self)`
- Defined: `quantum_framework_menu.py:497`
- Doc: Measure entanglement entropy at different cuts.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _entropy_heatmap_experiment (method) `def _entropy_heatmap_experiment(self)`
- Defined: `quantum_framework_menu.py:522`
- Doc: Generate entropy heatmap for different states.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_molecular_menu (method) `def _show_molecular_menu(self)`
- Defined: `quantum_framework_menu.py:583`
- Doc: Display molecular simulations menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _list_molecules (method) `def _list_molecules(self)`
- Defined: `quantum_framework_menu.py:617`
- Doc: List all available molecules.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _molecule_info (method) `def _molecule_info(self)`
- Defined: `quantum_framework_menu.py:636`
- Doc: Show detailed molecule information.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _vqe_ground_state (method) `def _vqe_ground_state(self)`
- Defined: `quantum_framework_menu.py:667`
- Doc: Run VQE for molecular ground state.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _energy_landscape (method) `def _energy_landscape(self)`
- Defined: `quantum_framework_menu.py:751`
- Doc: Plot molecular energy landscape.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _bond_dissociation (method) `def _bond_dissociation(self)`
- Defined: `quantum_framework_menu.py:808`
- Doc: Simulate bond dissociation curve.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_orbital_menu (method) `def _show_orbital_menu(self)`
- Defined: `quantum_framework_menu.py:868`
- Doc: Display orbital visualization menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _list_orbitals (method) `def _list_orbitals(self)`
- Defined: `quantum_framework_menu.py:899`
- Doc: List all available orbitals.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _visualize_orbital (method) `def _visualize_orbital(self)`
- Defined: `quantum_framework_menu.py:915`
- Doc: Visualize a single orbital.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _generate_orbital_plot (method) `def _generate_orbital_plot(self, orb, num_samples)`
- Defined: `quantum_framework_menu.py:947`
- Doc: Generate orbital visualization plot.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _compare_orbitals (method) `def _compare_orbitals(self)`
- Defined: `quantum_framework_menu.py:1064`
- Doc: Compare multiple orbitals.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _radial_wavefunction (method) `def _radial_wavefunction(self)`
- Defined: `quantum_framework_menu.py:1146`
- Doc: Plot radial wavefunction.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _angular_wavefunction (method) `def _angular_wavefunction(self)`
- Defined: `quantum_framework_menu.py:1202`
- Doc: Plot angular wavefunction.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_relativistic_menu (method) `def _show_relativistic_menu(self)`
- Defined: `quantum_framework_menu.py:1264`
- Doc: Display relativistic physics menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _dirac_energy_levels (method) `def _dirac_energy_levels(self)`
- Defined: `quantum_framework_menu.py:1292`
- Doc: Calculate Dirac energy levels.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _dirac_energy (method) `def _dirac_energy(self, n, kappa, alpha, c)`
- Defined: `quantum_framework_menu.py:1321`
- Doc: Calculate Dirac energy level.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _fine_structure (method) `def _fine_structure(self)`
- Defined: `quantum_framework_menu.py:1329`
- Doc: Calculate fine structure corrections.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _zitterbewegung (method) `def _zitterbewegung(self)`
- Defined: `quantum_framework_menu.py:1347`
- Doc: Simulate Zitterbewegung.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _spin_orbit (method) `def _spin_orbit(self)`
- Defined: `quantum_framework_menu.py:1363`
- Doc: Calculate spin-orbit coupling.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_qed_menu (method) `def _show_qed_menu(self)`
- Defined: `quantum_framework_menu.py:1378`
- Doc: Display QED effects menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _lamb_shift (method) `def _lamb_shift(self)`
- Defined: `quantum_framework_menu.py:1406`
- Doc: Calculate Lamb shift.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _anomalous_moment (method) `def _anomalous_moment(self)`
- Defined: `quantum_framework_menu.py:1426`
- Doc: Calculate anomalous magnetic moment.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _vacuum_polarization (method) `def _vacuum_polarization(self)`
- Defined: `quantum_framework_menu.py:1453`
- Doc: Calculate vacuum polarization effects.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _full_qed (method) `def _full_qed(self)`
- Defined: `quantum_framework_menu.py:1468`
- Doc: Show full QED corrections.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_algorithms_menu (method) `def _show_algorithms_menu(self)`
- Defined: `quantum_framework_menu.py:1482`
- Doc: Display quantum algorithms menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _grover_search (method) `def _grover_search(self)`
- Defined: `quantum_framework_menu.py:1510`
- Doc: Demonstrate Grover's search algorithm.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _apply_grover_iteration (method) `def _apply_grover_iteration(self, state, marked, n_qubits)`
- Defined: `quantum_framework_menu.py:1558`
- Doc: Apply one Grover iteration: oracle then diffusion.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _qft_demo (method) `def _qft_demo(self)`
- Defined: `quantum_framework_menu.py:1607`
- Doc: Demonstrate Quantum Fourier Transform.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _phase_estimation (method) `def _phase_estimation(self)`
- Defined: `quantum_framework_menu.py:1669`
- Doc: Demonstrate phase estimation.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _vqe_demo (method) `def _vqe_demo(self)`
- Defined: `quantum_framework_menu.py:1746`
- Doc: Demonstrate VQE.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_benchmark_menu (method) `def _show_benchmark_menu(self)`
- Defined: `quantum_framework_menu.py:1816`
- Doc: Display benchmarks menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _mps_scaling_benchmark (method) `def _mps_scaling_benchmark(self)`
- Defined: `quantum_framework_menu.py:1844`
- Doc: Run MPS scaling benchmark.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _gate_performance (method) `def _gate_performance(self)`
- Defined: `quantum_framework_menu.py:1872`
- Doc: Benchmark gate performance.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _memory_comparison (method) `def _memory_comparison(self)`
- Defined: `quantum_framework_menu.py:1907`
- Doc: Compare memory usage.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _entanglement_scaling (method) `def _entanglement_scaling(self)`
- Defined: `quantum_framework_menu.py:1927`
- Doc: Study entanglement scaling.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_config_menu (method) `def _show_config_menu(self)`
- Defined: `quantum_framework_menu.py:1953`
- Doc: Display configuration menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _view_config (method) `def _view_config(self)`
- Defined: `quantum_framework_menu.py:1984`
- Doc: View current configuration.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _list_atoms (method) `def _list_atoms(self)`
- Defined: `quantum_framework_menu.py:1999`
- Doc: List available atoms.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _list_molecules_config (method) `def _list_molecules_config(self)`
- Defined: `quantum_framework_menu.py:2016`
- Doc: List available molecules.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _list_experiments (method) `def _list_experiments(self)`
- Defined: `quantum_framework_menu.py:2020`
- Doc: List available experiments.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _system_info (method) `def _system_info(self)`
- Defined: `quantum_framework_menu.py:2036`
- Doc: Show system information.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_particle_physics_menu (method) `def _show_particle_physics_menu(self)`
- Defined: `quantum_framework_menu.py:2055`
- Doc: Display particle physics menu (Higgs analysis).
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _run_higgs_analysis (method) `def _run_higgs_analysis(self)`
- Defined: `quantum_framework_menu.py:2077`
- Doc: Run the Higgs boson 4-lepton quantum analysis.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _higgs_about (method) `def _higgs_about(self)`
- Defined: `quantum_framework_menu.py:2101`
- Doc: Show information about the Higgs analysis.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_visualization_menu (method) `def _show_visualization_menu(self)`
- Defined: `quantum_framework_menu.py:2129`
- Doc: Display quantum visualization menu.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _run_quantum_dash (method) `def _run_quantum_dash(self)`
- Defined: `quantum_framework_menu.py:2154`
- Doc: Run brutalist quantum state visualizer.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _run_quantum_3dview (method) `def _run_quantum_3dview(self)`
- Defined: `quantum_framework_menu.py:2201`
- Doc: Run 3D holographic quantum dashboard.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _run_quantum_visualizer (method) `def _run_quantum_visualizer(self)`
- Defined: `quantum_framework_menu.py:2247`
- Doc: Run the standard quantum state visualizer.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _run_polarizability_vqe (method) `def _run_polarizability_vqe(self)`
- Defined: `quantum_framework_menu.py:2285`
- Doc: Run H2 polarizability / Stark effect VQE from app.py.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _show_help (method) `def _show_help(self)`
- Defined: `quantum_framework_menu.py:2316`
- Doc: Show help information.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### _quit (method) `def _quit(self)`
- Defined: `quantum_framework_menu.py:2350`
- Doc: Exit the menu system.
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### test_header (method) `def test_header(name)`
- Defined: `quantum_framework_menu.py:2379`
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### radial_wf (method) `def radial_wf(n, l, r)`
- Defined: `quantum_framework_menu.py:951`
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### spherical_harm_real (method) `def spherical_harm_real(l, m, theta, phi)`
- Defined: `quantum_framework_menu.py:959`
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

### compute_energy (method) `def compute_energy(state, n_qubits)`
- Defined: `quantum_framework_menu.py:1764`
- Depends on: `app.py`, `higgs_four_lepton_analysis.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_visualizer.py`
- Imported by: `quantum_framework_main.py`

## quantum_framework_molecular.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `quantum_framework_molecular.py:58`
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### run_vqe_h2 (method) `def run_vqe_h2(max_iter)`
- Defined: `quantum_framework_molecular.py:630`
- Doc: Run VQE for H2 molecule - convenience function.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### h2_sto3g (method) `def h2_sto3g(bond_length)`
- Defined: `quantum_framework_molecular.py:89`
- Doc: Build H2 molecule with STO-3G basis.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### _h2_pyscf (method) `def _h2_pyscf(bond_length)`
- Defined: `quantum_framework_molecular.py:96`
- Doc: Build H2 using PySCF.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### _h2_hardcoded (method) `def _h2_hardcoded(bond_length)`
- Defined: `quantum_framework_molecular.py:128`
- Doc: Build H2 with hardcoded values - CORRECTED for 2-qubit active space.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### __init__ (method) `def __init__(self, mol, n_qubits)`
- Defined: `quantum_framework_molecular.py:163`
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### _build_hamiltonian (method) `def _build_hamiltonian(self)`
- Defined: `quantum_framework_molecular.py:170`
- Doc: Build the molecular Hamiltonian in JW representation.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### _build_openfermion_hamiltonian (method) `def _build_openfermion_hamiltonian(self)`
- Defined: `quantum_framework_molecular.py:178`
- Doc: Build Hamiltonian using OpenFermion - FIXED API.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### _build_hardcoded_hamiltonian (method) `def _build_hardcoded_hamiltonian(self)`
- Defined: `quantum_framework_molecular.py:231`
- Doc: Build hardcoded H2 Hamiltonian - CORRECTED COEFFICIENTS.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### _apply_pauli (method) `def _apply_pauli(self, state, pauli)`
- Defined: `quantum_framework_molecular.py:276`
- Doc: Apply Pauli operator to state vector - CORRECTED.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### expectation_value (method) `def expectation_value(self, state)`
- Defined: `quantum_framework_molecular.py:302`
- Doc: Compute ⟨ψ|H|ψ⟩ for the given state.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### evaluate (method) `def evaluate(self, amps)`
- Defined: `quantum_framework_molecular.py:318`
- Doc: Evaluate energy from amplitudes (supports both numpy and torch).
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### __call__ (method) `def __call__(self, amps)`
- Defined: `quantum_framework_molecular.py:330`
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### __init__ (method) `def __init__(self, n_qubits, n_electrons)`
- Defined: `quantum_framework_molecular.py:345`
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### apply_double_excitation_2q (method) `def apply_double_excitation_2q(self, state, theta)`
- Defined: `quantum_framework_molecular.py:373`
- Doc: Apply double excitation for 2-qubit H2 model.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### apply_single_excitation (method) `def apply_single_excitation(self, state, o, v, theta)`
- Defined: `quantum_framework_molecular.py:395`
- Doc: Apply single excitation as Givens rotation.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### apply_double_excitation (method) `def apply_double_excitation(self, state, o1, o2, v1, v2, theta)`
- Defined: `quantum_framework_molecular.py:416`
- Doc: Apply double excitation for 4+ qubit systems.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### apply (method) `def apply(self, state, thetas)`
- Defined: `quantum_framework_molecular.py:443`
- Doc: Apply UCCSD ansatz to state.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### __repr__ (method) `def __repr__(self)`
- Defined: `quantum_framework_molecular.py:490`
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### __init__ (method) `def __init__(self, mol)`
- Defined: `quantum_framework_molecular.py:517`
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### prepare_hf_state (method) `def prepare_hf_state(self)`
- Defined: `quantum_framework_molecular.py:528`
- Doc: Prepare Hartree-Fock state.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### run (method) `def run(self, max_iter, tol)`
- Defined: `quantum_framework_molecular.py:546`
- Doc: Run VQE optimization.
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

### cost (method) `def cost(thetas)`
- Defined: `quantum_framework_molecular.py:566`
- Imported by: `quantum_framework_menu.py`, `quantum_lab.py`

## quantum_framework_molecular_fixed.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `quantum_framework_molecular_fixed.py:50`

### _get_sd_indices (method) `def _get_sd_indices(n_electrons, n_qubits)`
- Defined: `quantum_framework_molecular_fixed.py:323`
- Doc: Get single and double excitation indices for UCCSD.

### run_vqe_demo (method) `def run_vqe_demo()`
- Defined: `quantum_framework_molecular_fixed.py:614`
- Doc: Run a quick VQE demo to verify the fixes.

### h2_sto3g (method) `def h2_sto3g(bond_length)`
- Defined: `quantum_framework_molecular_fixed.py:84`
- Doc: Build H2 molecule with STO-3G basis.

### _h2_pyscf (method) `def _h2_pyscf(bond_length)`
- Defined: `quantum_framework_molecular_fixed.py:91`
- Doc: Build H2 using PySCF - FIXED atom string syntax.

### _h2_hardcoded (method) `def _h2_hardcoded()`
- Defined: `quantum_framework_molecular_fixed.py:126`
- Doc: Build H2 with hardcoded values - FIXED coefficients.

### __init__ (method) `def __init__(self, mol, n_qubits)`
- Defined: `quantum_framework_molecular_fixed.py:155`

### _build_hamiltonian (method) `def _build_hamiltonian(self)`
- Defined: `quantum_framework_molecular_fixed.py:162`
- Doc: Build the molecular Hamiltonian in JW representation.

### _build_openfermion_hamiltonian (method) `def _build_openfermion_hamiltonian(self)`
- Defined: `quantum_framework_molecular_fixed.py:169`
- Doc: Build Hamiltonian using OpenFermion - FIXED geometry.

### _build_hardcoded_hamiltonian (method) `def _build_hardcoded_hamiltonian(self)`
- Defined: `quantum_framework_molecular_fixed.py:215`
- Doc: Build hardcoded H2 Hamiltonian - FIXED coefficients.

### _apply_pauli (method) `def _apply_pauli(self, state, pauli)`
- Defined: `quantum_framework_molecular_fixed.py:247`
- Doc: Apply Pauli operator to state vector.

### expectation_value (method) `def expectation_value(self, state)`
- Defined: `quantum_framework_molecular_fixed.py:281`
- Doc: Compute ⟨ψ|H|ψ⟩ for the given state.

### evaluate (method) `def evaluate(self, amps)`
- Defined: `quantum_framework_molecular_fixed.py:305`
- Doc: Evaluate energy from MPS amplitudes.

### __call__ (method) `def __call__(self, amps)`
- Defined: `quantum_framework_molecular_fixed.py:319`

### __init__ (method) `def __init__(self, n_qubits, n_electrons, backend)`
- Defined: `quantum_framework_molecular_fixed.py:350`

### apply_single_excitation (method) `def apply_single_excitation(self, state, o, v, theta)`
- Defined: `quantum_framework_molecular_fixed.py:359`
- Doc: Apply single excitation operator exp(theta * (a_v† a_o - a_o† a_v)).

### apply_double_excitation (method) `def apply_double_excitation(self, state, o1, o2, v1, v2, theta)`
- Defined: `quantum_framework_molecular_fixed.py:387`
- Doc: Apply double excitation operator.

### apply (method) `def apply(self, state, thetas)`
- Defined: `quantum_framework_molecular_fixed.py:419`
- Doc: Apply UCCSD ansatz to state.

### __repr__ (method) `def __repr__(self)`
- Defined: `quantum_framework_molecular_fixed.py:471`

### __init__ (method) `def __init__(self, qc, config)`
- Defined: `quantum_framework_molecular_fixed.py:498`

### prepare_hf_state (method) `def prepare_hf_state(self, mol)`
- Defined: `quantum_framework_molecular_fixed.py:502`
- Doc: Prepare Hartree-Fock state.

### run (method) `def run(self, mol, backend, max_iter, tol)`
- Defined: `quantum_framework_molecular_fixed.py:525`
- Doc: Run VQE optimization.

### cost (method) `def cost(thetas)`
- Defined: `quantum_framework_molecular_fixed.py:556`

## quantum_framework_molecular_v2.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `quantum_framework_molecular_v2.py:75`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### run_vqe (method) `def run_vqe(molecule, precision_mode, config_path)`
- Defined: `quantum_framework_molecular_v2.py:1332`
- Doc: Convenience function to run VQE.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### from_toml (method) `def from_toml(cls, toml_path)`
- Defined: `quantum_framework_molecular_v2.py:131`
- Doc: Load configuration from TOML file.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### build (method) `def build(name, geometry, basis, charge, multiplicity, description)`
- Defined: `quantum_framework_molecular_v2.py:208`
- Doc: Build molecule using OpenFermion.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _run_pyscf_direct (method) `def _run_pyscf_direct(geometry, basis, charge, multiplicity)`
- Defined: `quantum_framework_molecular_v2.py:289`
- Doc: Run PySCF directly if openfermionpyscf not available.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### h2 (method) `def h2(bond_length, basis)`
- Defined: `quantum_framework_molecular_v2.py:324`
- Doc: Build H2 molecule.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### h2o (method) `def h2o(bond_length_oh, angle_hoh, basis)`
- Defined: `quantum_framework_molecular_v2.py:330`
- Doc: Build H2O molecule.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### lih (method) `def lih(bond_length, basis)`
- Defined: `quantum_framework_molecular_v2.py:342`
- Doc: Build LiH molecule.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### build_jw_hamiltonian (method) `def build_jw_hamiltonian(mol)`
- Defined: `quantum_framework_molecular_v2.py:360`
- Doc: Build Jordan-Wigner transformed Hamiltonian using OpenFermion.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### build_hamiltonian_matrix (method) `def build_hamiltonian_matrix(mol, n_qubits)`
- Defined: `quantum_framework_molecular_v2.py:417`
- Doc: Build full Hamiltonian matrix for small systems.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _pauli_matrix (method) `def _pauli_matrix(pauli_list, n_qubits)`
- Defined: `quantum_framework_molecular_v2.py:441`
- Doc: Build matrix for a Pauli string.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### __init__ (method) `def __init__(self, mol, config)`
- Defined: `quantum_framework_molecular_v2.py:484`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _precompute_operations (method) `def _precompute_operations(self)`
- Defined: `quantum_framework_molecular_v2.py:503`
- Doc: Precompute all Pauli operations for fast evaluation.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _compute_pauli_mapping (method) `def _compute_pauli_mapping(self, pauli_list)`
- Defined: `quantum_framework_molecular_v2.py:522`
- Doc: Compute index mapping and phases for a Pauli string.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### apply_pauli_fast (method) `def apply_pauli_fast(self, state, op)`
- Defined: `quantum_framework_molecular_v2.py:561`
- Doc: Apply cached Pauli operation.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### expectation_value (method) `def expectation_value(self, state)`
- Defined: `quantum_framework_molecular_v2.py:568`
- Doc: Compute energy expectation value.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### batch_expectation (method) `def batch_expectation(self, states)`
- Defined: `quantum_framework_molecular_v2.py:587`
- Doc: Compute expectation for batch of states.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### __init__ (method) `def __init__(self, mol, ansatz, evaluator, config)`
- Defined: `quantum_framework_molecular_v2.py:624`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### estimate_mp2_amplitude (method) `def estimate_mp2_amplitude(self)`
- Defined: `quantum_framework_molecular_v2.py:630`
- Doc: Estimate doubles amplitude from MP2 theory.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### scan_parameter_space (method) `def scan_parameter_space(self, hf_state, n_samples, param_range)`
- Defined: `quantum_framework_molecular_v2.py:657`
- Doc: Systematic scan of parameter space.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _parabolic_refinement (method) `def _parabolic_refinement(self, x_vals, y_vals, best_idx, n_params)`
- Defined: `quantum_framework_molecular_v2.py:712`
- Doc: Refine minimum using parabolic interpolation.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### initialize (method) `def initialize(self, hf_state)`
- Defined: `quantum_framework_molecular_v2.py:735`
- Doc: Complete initialization with all techniques.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### __init__ (method) `def __init__(self, n_qubits, n_electrons, config)`
- Defined: `quantum_framework_molecular_v2.py:756`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _generate_excitations (method) `def _generate_excitations(self)`
- Defined: `quantum_framework_molecular_v2.py:764`
- Doc: Generate all single and double excitations.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### apply (method) `def apply(self, state, thetas)`
- Defined: `quantum_framework_molecular_v2.py:785`
- Doc: Apply UCCSD ansatz to state.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _apply_single (method) `def _apply_single(self, state, i, a, theta)`
- Defined: `quantum_framework_molecular_v2.py:817`
- Doc: Apply single excitation as Givens rotation.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _apply_double (method) `def _apply_double(self, state, i, j, a, b, theta)`
- Defined: `quantum_framework_molecular_v2.py:838`
- Doc: Apply double excitation.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### verify_identity (method) `def verify_identity(self, hf_state, evaluator, hf_energy)`
- Defined: `quantum_framework_molecular_v2.py:862`
- Doc: Verify that θ=0 gives HF state.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### __init__ (method) `def __init__(self, n_qubits, config)`
- Defined: `quantum_framework_molecular_v2.py:889`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### to_statevector (method) `def to_statevector(self)`
- Defined: `quantum_framework_molecular_v2.py:910`
- Doc: Convert MPS to full statevector.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### from_statevector (method) `def from_statevector(cls, state, n_qubits, config)`
- Defined: `quantum_framework_molecular_v2.py:918`
- Doc: Create MPS from statevector.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### compute_entanglement (method) `def compute_entanglement(self, bond_idx)`
- Defined: `quantum_framework_molecular_v2.py:953`
- Doc: Compute entanglement entropy at bond.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### __init__ (method) `def __init__(self, n_qubits, n_particles)`
- Defined: `quantum_framework_molecular_v2.py:989`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _generate_fock_states (method) `def _generate_fock_states(self)`
- Defined: `quantum_framework_molecular_v2.py:1000`
- Doc: Generate all states with fixed particle number.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### hf_state (method) `def hf_state(self)`
- Defined: `quantum_framework_molecular_v2.py:1013`
- Doc: Create HF state in subspace.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### apply_excitation (method) `def apply_excitation(self, state, occ, vir, theta)`
- Defined: `quantum_framework_molecular_v2.py:1027`
- Doc: Apply excitation preserving particle number.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### to_full_statevector (method) `def to_full_statevector(self, state)`
- Defined: `quantum_framework_molecular_v2.py:1051`
- Doc: Convert subspace state to full statevector.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### __repr__ (method) `def __repr__(self)`
- Defined: `quantum_framework_molecular_v2.py:1081`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### __init__ (method) `def __init__(self, mol, config)`
- Defined: `quantum_framework_molecular_v2.py:1117`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### prepare_hf_state (method) `def prepare_hf_state(self)`
- Defined: `quantum_framework_molecular_v2.py:1137`
- Doc: Prepare Hartree-Fock state.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### evaluate (method) `def evaluate(self, state)`
- Defined: `quantum_framework_molecular_v2.py:1152`
- Doc: Evaluate energy.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### apply_ansatz (method) `def apply_ansatz(self, state, thetas)`
- Defined: `quantum_framework_molecular_v2.py:1159`
- Doc: Apply UCCSD ansatz.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### run (method) `def run(self)`
- Defined: `quantum_framework_molecular_v2.py:1184`
- Doc: Run VQE optimization.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_framework_molecular_v2.py:1294`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### _load_models (method) `def _load_models(self)`
- Defined: `quantum_framework_molecular_v2.py:1301`
- Doc: Load pre-trained backend models.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### get_backend_energy (method) `def get_backend_energy(self, state, backend_name)`
- Defined: `quantum_framework_molecular_v2.py:1318`
- Doc: Get energy estimate from backend model.
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

### cost (method) `def cost(thetas)`
- Defined: `quantum_framework_molecular_v2.py:1212`
- Imported by: `demo_molecular_vqe.py`, `demo_molecular_vqe.py`

## quantum_framework_physics.py

### __init__ (method) `def __init__(self, channels, grid_size)`
- Defined: `quantum_framework_physics.py:32`

### forward (method) `def forward(self, x)`
- Defined: `quantum_framework_physics.py:43`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `quantum_framework_physics.py:72`

### forward (method) `def forward(self, x)`
- Defined: `quantum_framework_physics.py:81`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- Defined: `quantum_framework_physics.py:98`

### forward (method) `def forward(self, x)`
- Defined: `quantum_framework_physics.py:109`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- Defined: `quantum_framework_physics.py:126`

### forward (method) `def forward(self, x)`
- Defined: `quantum_framework_physics.py:139`

### __init__ (method) `def __init__(self, representation, device)`
- Defined: `quantum_framework_physics.py:156`

### _init_matrices (method) `def _init_matrices(self)`
- Defined: `quantum_framework_physics.py:161`

### to (method) `def to(self, device)`
- Defined: `quantum_framework_physics.py:244`

### __init__ (method) `def __init__(self, grid_size, potential_depth, potential_width)`
- Defined: `quantum_framework_physics.py:253`

### _grid (method) `def _grid(self)`
- Defined: `quantum_framework_physics.py:258`

### harmonic (method) `def harmonic(self)`
- Defined: `quantum_framework_physics.py:263`

### double_well (method) `def double_well(self)`
- Defined: `quantum_framework_physics.py:268`

### coulomb (method) `def coulomb(self)`
- Defined: `quantum_framework_physics.py:274`

### periodic_lattice (method) `def periodic_lattice(self)`
- Defined: `quantum_framework_physics.py:280`

### mixed (method) `def mixed(self, seed)`
- Defined: `quantum_framework_physics.py:284`

### __init__ (method) `def __init__(self, grid_size, electron_mass, c_light, device)`
- Defined: `quantum_framework_physics.py:303`

### _precompute_operators (method) `def _precompute_operators(self)`
- Defined: `quantum_framework_physics.py:311`

### apply_dirac_hamiltonian (method) `def apply_dirac_hamiltonian(self, spinor, potential)`
- Defined: `quantum_framework_physics.py:318`
- Doc: Apply Dirac Hamiltonian to 4-component spinor.

### time_evolution (method) `def time_evolution(self, spinor, dt, potential, normalization_eps)`
- Defined: `quantum_framework_physics.py:362`
- Doc: Time evolution of Dirac spinor using first-order split-step.

### __init__ (method) `def __init__(self, alpha_fs, c_light, electron_mass)`
- Defined: `quantum_framework_physics.py:389`

### bethe_formula (method) `def bethe_formula(self, n, l, Z)`
- Defined: `quantum_framework_physics.py:394`
- Doc: Bethe's non-relativistic formula for Lamb shift.

### _higher_l_shift (method) `def _higher_l_shift(self, n, l, Z)`
- Defined: `quantum_framework_physics.py:407`

### full_lamb_shift (method) `def full_lamb_shift(self, n, l, j, Z)`
- Defined: `quantum_framework_physics.py:412`
- Doc: Calculate full Lamb shift including radiative corrections.

### __init__ (method) `def __init__(self, alpha_fs)`
- Defined: `quantum_framework_physics.py:446`

### schwinger_term (method) `def schwinger_term(self)`
- Defined: `quantum_framework_physics.py:449`

### second_order (method) `def second_order(self)`
- Defined: `quantum_framework_physics.py:452`

### third_order (method) `def third_order(self)`
- Defined: `quantum_framework_physics.py:456`

### fourth_order (method) `def fourth_order(self)`
- Defined: `quantum_framework_physics.py:460`

### fifth_order (method) `def fifth_order(self)`
- Defined: `quantum_framework_physics.py:464`

### calculate_a_e (method) `def calculate_a_e(self, order)`
- Defined: `quantum_framework_physics.py:468`

### __init__ (method) `def __init__(self, c_light, alpha_fs)`
- Defined: `quantum_framework_physics.py:495`

### energy_level_dirac (method) `def energy_level_dirac(self, n, kappa)`
- Defined: `quantum_framework_physics.py:499`
- Doc: Exact Dirac energy level for hydrogen-like atom.

### fine_structure_splitting (method) `def fine_structure_splitting(self, n, l)`
- Defined: `quantum_framework_physics.py:512`
- Doc: Calculate fine structure splitting for given n, l.

### energy_spectrum (method) `def energy_spectrum(self, n_max)`
- Defined: `quantum_framework_physics.py:535`

### __init__ (method) `def __init__(self, grid_size, c_light, electron_mass, device)`
- Defined: `quantum_framework_physics.py:578`

### create_gaussian_wave_packet (method) `def create_gaussian_wave_packet(self, sigma, momentum)`
- Defined: `quantum_framework_physics.py:585`

### compute_position_expectation (method) `def compute_position_expectation(self, spinor)`
- Defined: `quantum_framework_physics.py:603`

### compute_velocity_expectation (method) `def compute_velocity_expectation(self, spinor)`
- Defined: `quantum_framework_physics.py:615`

## quantum_framework_visualization.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `quantum_framework_visualization.py:46`
- Imported by: `qc_dashboard.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_framework_visualization.py:65`
- Imported by: `qc_dashboard.py`

### radial_wavefunction (method) `def radial_wavefunction(n, l, r)`
- Defined: `quantum_framework_visualization.py:69`
- Imported by: `qc_dashboard.py`

### spherical_harmonic_real (method) `def spherical_harmonic_real(l, m, theta, phi)`
- Defined: `quantum_framework_visualization.py:81`
- Imported by: `qc_dashboard.py`

### psi_3d (method) `def psi_3d(self, n, l, m, r, theta, phi)`
- Defined: `quantum_framework_visualization.py:92`
- Imported by: `qc_dashboard.py`

### psi_on_grid (method) `def psi_on_grid(self, n, l, m)`
- Defined: `quantum_framework_visualization.py:97`
- Imported by: `qc_dashboard.py`

### __init__ (method) `def __init__(self, config, wavefunction_calc)`
- Defined: `quantum_framework_visualization.py:121`
- Imported by: `qc_dashboard.py`

### find_max_probability (method) `def find_max_probability(self, n, l, m)`
- Defined: `quantum_framework_visualization.py:125`
- Imported by: `qc_dashboard.py`

### sample (method) `def sample(self, n, l, m, num_samples)`
- Defined: `quantum_framework_visualization.py:149`
- Imported by: `qc_dashboard.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_framework_visualization.py:208`
- Imported by: `qc_dashboard.py`

### visualize (method) `def visualize(self, data, save_path)`
- Defined: `quantum_framework_visualization.py:211`
- Imported by: `qc_dashboard.py`

### __init__ (method) `def __init__(self, config, wavefunction_calc)`
- Defined: `quantum_framework_visualization.py:296`
- Imported by: `qc_dashboard.py`

### sample_entangled_state (method) `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples, entanglement_weight)`
- Defined: `quantum_framework_visualization.py:301`
- Imported by: `qc_dashboard.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_framework_visualization.py:331`
- Imported by: `qc_dashboard.py`

### visualize (method) `def visualize(self, data, quantum_result, save_path)`
- Defined: `quantum_framework_visualization.py:334`
- Imported by: `qc_dashboard.py`

## quantum_lab.py

### probability_bars (function) `def probability_bars(probs, n_qubits, lang, max_rows)`
- Defined: `quantum_lab.py:325`
- Doc: Build a table of probability bars for a state's distribution.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### counts_bars (function) `def counts_bars(counts, total, lang)`
- Defined: `quantum_lab.py:353`
- Doc: Build a table of measurement-count bars.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### draw_circuit (function) `def draw_circuit(n_qubits, instructions)`
- Defined: `quantum_lab.py:369`
- Doc: Render an ASCII timeline of the circuit, one line per qubit.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### parse_angle (function) `def parse_angle(token)`
- Defined: `quantum_lab.py:394`
- Doc: Parse an angle like '1.57', 'pi', '-pi/2' or '3*pi/4' into radians.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### sample_measurements (function) `def sample_measurements(probs, n_qubits, n_samples, rng)`
- Defined: `quantum_lab.py:414`
- Doc: Draw measurement outcomes from a probability distribution.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _radial (method) `def _radial(n, l, r)`
- Defined: `quantum_lab.py:440`
- Doc: Hydrogen radial wavefunction R_nl(r) in atomic units.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _ang_s (method) `def _ang_s(x, y, z, r)`
- Defined: `quantum_lab.py:452`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _ang_pz (method) `def _ang_pz(x, y, z, r)`
- Defined: `quantum_lab.py:456`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _ang_px (method) `def _ang_px(x, y, z, r)`
- Defined: `quantum_lab.py:460`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _ang_dz2 (method) `def _ang_dz2(x, y, z, r)`
- Defined: `quantum_lab.py:464`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _ang_dxz (method) `def _ang_dxz(x, y, z, r)`
- Defined: `quantum_lab.py:469`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _ang_dxy (method) `def _ang_dxy(x, y, z, r)`
- Defined: `quantum_lab.py:473`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### field_to_text (method) `def field_to_text(psi)`
- Defined: `quantum_lab.py:490`
- Doc: Render a real scalar field as colored ASCII: brightness = |psi|^2, color = sign.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### render_orbital (method) `def render_orbital(spec, rows, cols)`
- Defined: `quantum_lab.py:516`
- Doc: Render |psi|^2 of a hydrogen orbital on a plane slice as colored ASCII.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### render_h2_molecular_orbital (method) `def render_h2_molecular_orbital(kind, rows, cols)`
- Defined: `quantum_lab.py:536`
- Doc: Render the bonding or antibonding LCAO molecular orbital of H2.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### landscape_plot (method) `def landscape_plot(energies, e_hf, e_fci, marker, rows)`
- Defined: `quantum_lab.py:556`
- Doc: Draw an ASCII plot of E(theta) over one full period with HF and FCI
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### launch_quantum_lab (method) `def launch_quantum_lab(config, loader, lang, lesson)`
- Defined: `quantum_lab.py:1474`
- Doc: Entry point used both standalone and from quantum_framework_main.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### main (method) `def main()`
- Defined: `quantum_lab.py:1500`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### row_of (method) `def row_of(e)`
- Defined: `quantum_lab.py:567`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### __init__ (method) `def __init__(self, qc)`
- Defined: `quantum_lab.py:613`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### ansatz_instructions (method) `def ansatz_instructions(self, theta)`
- Defined: `quantum_lab.py:621`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### energy (method) `def energy(self, theta)`
- Defined: `quantum_lab.py:624`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### correlation_pct (method) `def correlation_pct(self, e)`
- Defined: `quantum_lab.py:633`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### landscape (method) `def landscape(self, cols)`
- Defined: `quantum_lab.py:636`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### optimize (method) `def optimize(self, theta0, lr, max_iters, tol)`
- Defined: `quantum_lab.py:640`
- Doc: Gradient descent; yields (iteration, theta, energy) live.
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### __init__ (method) `def __init__(self, config, loader, lang)`
- Defined: `quantum_lab.py:675`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### t (method) `def t(self, key)`
- Defined: `quantum_lab.py:683`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### pause (method) `def pause(self)`
- Defined: `quantum_lab.py:690`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### panel (method) `def panel(self, body, title, style)`
- Defined: `quantum_lab.py:697`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### show_state (method) `def show_state(self, probs, n_qubits, title)`
- Defined: `quantum_lab.py:701`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### run_quiz (method) `def run_quiz(self, quiz)`
- Defined: `quantum_lab.py:707`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### lessons (method) `def lessons(self)`
- Defined: `quantum_lab.py:728`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_superposition (method) `def _demo_superposition(self)`
- Defined: `quantum_lab.py:733`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_rotation (method) `def _demo_rotation(self)`
- Defined: `quantum_lab.py:744`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_measurement (method) `def _demo_measurement(self)`
- Defined: `quantum_lab.py:766`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_bell (method) `def _demo_bell(self)`
- Defined: `quantum_lab.py:779`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_ghz_w (method) `def _demo_ghz_w(self)`
- Defined: `quantum_lab.py:792`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_grover (method) `def _demo_grover(self)`
- Defined: `quantum_lab.py:799`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_molecule (method) `def _demo_molecule(self)`
- Defined: `quantum_lab.py:815`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _vqe (method) `def _vqe(self)`
- Defined: `quantum_lab.py:821`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_h2_clouds (method) `def _demo_h2_clouds(self)`
- Defined: `quantum_lab.py:826`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_vqe_ansatz (method) `def _demo_vqe_ansatz(self)`
- Defined: `quantum_lab.py:835`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_vqe_landscape (method) `def _demo_vqe_landscape(self)`
- Defined: `quantum_lab.py:840`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _demo_vqe_live (method) `def _demo_vqe_live(self)`
- Defined: `quantum_lab.py:847`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _lessons_en (method) `def _lessons_en(self)`
- Defined: `quantum_lab.py:875`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _lessons_es (method) `def _lessons_es(self)`
- Defined: `quantum_lab.py:1015`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### run_lesson (method) `def run_lesson(self, lesson)`
- Defined: `quantum_lab.py:1159`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### lessons_menu (method) `def lessons_menu(self)`
- Defined: `quantum_lab.py:1175`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _rebuild_state (method) `def _rebuild_state(self, n_qubits, instructions)`
- Defined: `quantum_lab.py:1198`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _playground_dashboard (method) `def _playground_dashboard(self, n_qubits, state, instructions)`
- Defined: `quantum_lab.py:1216`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _show_amplitudes (method) `def _show_amplitudes(self, state, n_qubits)`
- Defined: `quantum_lab.py:1228`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### playground (method) `def playground(self)`
- Defined: `quantum_lab.py:1249`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### _molecule_card (method) `def _molecule_card(self, mol)`
- Defined: `quantum_lab.py:1346`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### molecule_explorer (method) `def molecule_explorer(self)`
- Defined: `quantum_lab.py:1365`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### orbital_viewer (method) `def orbital_viewer(self)`
- Defined: `quantum_lab.py:1383`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### chemistry_menu (method) `def chemistry_menu(self)`
- Defined: `quantum_lab.py:1403`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### glossary (method) `def glossary(self)`
- Defined: `quantum_lab.py:1422`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### banner (method) `def banner(self)`
- Defined: `quantum_lab.py:1437`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

### main_menu (method) `def main_menu(self)`
- Defined: `quantum_lab.py:1443`
- Depends on: `quantum_framework_core.py`, `quantum_framework_molecular.py`
- Imported by: `quantum_framework_main.py`

## quantum_simulator.py

### _make_logger (function) `def _make_logger(name, level)`
- Defined: `quantum_simulator.py:54`
- Imported by: `advanced_experiments.py`

### _single_qubit_unitary (method) `def _single_qubit_unitary(state, qubit, u, backend)`
- Defined: `quantum_simulator.py:587`
- Imported by: `advanced_experiments.py`

### _two_qubit_unitary (method) `def _two_qubit_unitary(state, ctrl, tgt, u4)`
- Defined: `quantum_simulator.py:611`
- Imported by: `advanced_experiments.py`

### _solve_eigenstate (method) `def _solve_eigenstate(config, potential, n)`
- Defined: `quantum_simulator.py:949`
- Imported by: `advanced_experiments.py`

### _build_basis_amplitude (method) `def _build_basis_amplitude(config, basis_idx)`
- Defined: `quantum_simulator.py:965`
- Imported by: `advanced_experiments.py`

### main (method) `def main()`
- Defined: `quantum_simulator.py:1868`
- Imported by: `advanced_experiments.py`

### from_toml (method) `def from_toml(cls, toml_path)`
- Defined: `quantum_simulator.py:112`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config_path)`
- Defined: `quantum_simulator.py:201`
- Imported by: `advanced_experiments.py`

### _find_config (method) `def _find_config(self)`
- Defined: `quantum_simulator.py:209`
- Imported by: `advanced_experiments.py`

### _load (method) `def _load(self)`
- Defined: `quantum_simulator.py:219`
- Imported by: `advanced_experiments.py`

### _load_defaults (method) `def _load_defaults(self)`
- Defined: `quantum_simulator.py:229`
- Imported by: `advanced_experiments.py`

### _parse_atoms (method) `def _parse_atoms(self)`
- Defined: `quantum_simulator.py:236`
- Imported by: `advanced_experiments.py`

### _parse_molecules (method) `def _parse_molecules(self)`
- Defined: `quantum_simulator.py:241`
- Imported by: `advanced_experiments.py`

### _parse_orbitals (method) `def _parse_orbitals(self)`
- Defined: `quantum_simulator.py:246`
- Imported by: `advanced_experiments.py`

### get_atom (method) `def get_atom(self, symbol)`
- Defined: `quantum_simulator.py:251`
- Imported by: `advanced_experiments.py`

### get_molecule (method) `def get_molecule(self, name)`
- Defined: `quantum_simulator.py:259`
- Imported by: `advanced_experiments.py`

### get_orbital (method) `def get_orbital(self, name)`
- Defined: `quantum_simulator.py:267`
- Imported by: `advanced_experiments.py`

### atoms (method) `def atoms(self)`
- Defined: `quantum_simulator.py:276`
- Imported by: `advanced_experiments.py`

### molecules (method) `def molecules(self)`
- Defined: `quantum_simulator.py:280`
- Imported by: `advanced_experiments.py`

### orbitals (method) `def orbitals(self)`
- Defined: `quantum_simulator.py:284`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, channels, grid_size)`
- Defined: `quantum_simulator.py:289`
- Imported by: `advanced_experiments.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_simulator.py:295`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `quantum_simulator.py:304`
- Imported by: `advanced_experiments.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_simulator.py:310`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- Defined: `quantum_simulator.py:322`
- Imported by: `advanced_experiments.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_simulator.py:330`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- Defined: `quantum_simulator.py:342`
- Imported by: `advanced_experiments.py`

### forward (method) `def forward(self, x)`
- Defined: `quantum_simulator.py:350`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, representation, device)`
- Defined: `quantum_simulator.py:362`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, amplitudes, n_qubits)`
- Defined: `quantum_simulator.py:381`
- Imported by: `advanced_experiments.py`

### normalize_ (method) `def normalize_(self)`
- Defined: `quantum_simulator.py:388`
- Imported by: `advanced_experiments.py`

### probabilities (method) `def probabilities(self)`
- Defined: `quantum_simulator.py:393`
- Imported by: `advanced_experiments.py`

### entropy (method) `def entropy(self)`
- Defined: `quantum_simulator.py:397`
- Imported by: `advanced_experiments.py`

### most_probable_bitstring (method) `def most_probable_bitstring(self)`
- Defined: `quantum_simulator.py:402`
- Imported by: `advanced_experiments.py`

### clone (method) `def clone(self)`
- Defined: `quantum_simulator.py:406`
- Imported by: `advanced_experiments.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_simulator.py:412`
- Imported by: `advanced_experiments.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_simulator.py:416`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_simulator.py:421`
- Imported by: `advanced_experiments.py`

### _load (method) `def _load(self)`
- Defined: `quantum_simulator.py:429`
- Imported by: `advanced_experiments.py`

### _precompute_laplacian (method) `def _precompute_laplacian(self)`
- Defined: `quantum_simulator.py:443`
- Imported by: `advanced_experiments.py`

### _apply_h (method) `def _apply_h(self, field)`
- Defined: `quantum_simulator.py:450`
- Imported by: `advanced_experiments.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_simulator.py:458`
- Imported by: `advanced_experiments.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_simulator.py:465`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config, hamiltonian)`
- Defined: `quantum_simulator.py:471`
- Imported by: `advanced_experiments.py`

### _load (method) `def _load(self)`
- Defined: `quantum_simulator.py:478`
- Imported by: `advanced_experiments.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_simulator.py:493`
- Imported by: `advanced_experiments.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_simulator.py:501`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config, hamiltonian)`
- Defined: `quantum_simulator.py:506`
- Imported by: `advanced_experiments.py`

### _load (method) `def _load(self)`
- Defined: `quantum_simulator.py:515`
- Imported by: `advanced_experiments.py`

### _precompute_dirac (method) `def _precompute_dirac(self)`
- Defined: `quantum_simulator.py:530`
- Imported by: `advanced_experiments.py`

### _pack (method) `def _pack(self, amp)`
- Defined: `quantum_simulator.py:537`
- Imported by: `advanced_experiments.py`

### _unpack (method) `def _unpack(self, spinor)`
- Defined: `quantum_simulator.py:547`
- Imported by: `advanced_experiments.py`

### _analytical_dirac (method) `def _analytical_dirac(self, spinor)`
- Defined: `quantum_simulator.py:553`
- Imported by: `advanced_experiments.py`

### evolve_amplitude (method) `def evolve_amplitude(self, amp, dt)`
- Defined: `quantum_simulator.py:564`
- Imported by: `advanced_experiments.py`

### apply_phase (method) `def apply_phase(self, amp, phase_angle)`
- Defined: `quantum_simulator.py:577`
- Imported by: `advanced_experiments.py`

### evolve_spinor (method) `def evolve_spinor(self, spinor, dt)`
- Defined: `quantum_simulator.py:580`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:641`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:645`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:651`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:654`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:664`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:667`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:676`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:679`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:688`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:691`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:700`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:703`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:712`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:715`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:725`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:728`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:739`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:742`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:753`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:756`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:768`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:771`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:780`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:783`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:792`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:795`
- Imported by: `advanced_experiments.py`

### name (method) `def name(self)`
- Defined: `quantum_simulator.py:804`
- Imported by: `advanced_experiments.py`

### apply (method) `def apply(self, state, backend, targets, params)`
- Defined: `quantum_simulator.py:807`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, n_qubits)`
- Defined: `quantum_simulator.py:839`
- Imported by: `advanced_experiments.py`

### _append (method) `def _append(self, gate_name, targets, params)`
- Defined: `quantum_simulator.py:843`
- Imported by: `advanced_experiments.py`

### h (method) `def h(self, qubit)`
- Defined: `quantum_simulator.py:846`
- Imported by: `advanced_experiments.py`

### x (method) `def x(self, qubit)`
- Defined: `quantum_simulator.py:849`
- Imported by: `advanced_experiments.py`

### y (method) `def y(self, qubit)`
- Defined: `quantum_simulator.py:852`
- Imported by: `advanced_experiments.py`

### z (method) `def z(self, qubit)`
- Defined: `quantum_simulator.py:855`
- Imported by: `advanced_experiments.py`

### s (method) `def s(self, qubit)`
- Defined: `quantum_simulator.py:858`
- Imported by: `advanced_experiments.py`

### t (method) `def t(self, qubit)`
- Defined: `quantum_simulator.py:861`
- Imported by: `advanced_experiments.py`

### rx (method) `def rx(self, qubit, theta)`
- Defined: `quantum_simulator.py:864`
- Imported by: `advanced_experiments.py`

### ry (method) `def ry(self, qubit, theta)`
- Defined: `quantum_simulator.py:867`
- Imported by: `advanced_experiments.py`

### rz (method) `def rz(self, qubit, theta)`
- Defined: `quantum_simulator.py:870`
- Imported by: `advanced_experiments.py`

### cnot (method) `def cnot(self, control, target)`
- Defined: `quantum_simulator.py:873`
- Imported by: `advanced_experiments.py`

### cz (method) `def cz(self, control, target)`
- Defined: `quantum_simulator.py:876`
- Imported by: `advanced_experiments.py`

### swap (method) `def swap(self, qubit1, qubit2)`
- Defined: `quantum_simulator.py:879`
- Imported by: `advanced_experiments.py`

### ccx (method) `def ccx(self, ctrl0, ctrl1, target)`
- Defined: `quantum_simulator.py:882`
- Imported by: `advanced_experiments.py`

### run (method) `def run(self, state, backend)`
- Defined: `quantum_simulator.py:885`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, state)`
- Defined: `quantum_simulator.py:895`
- Imported by: `advanced_experiments.py`

### entropy (method) `def entropy(self)`
- Defined: `quantum_simulator.py:898`
- Imported by: `advanced_experiments.py`

### most_probable_bitstring (method) `def most_probable_bitstring(self)`
- Defined: `quantum_simulator.py:901`
- Imported by: `advanced_experiments.py`

### probabilities (method) `def probabilities(self)`
- Defined: `quantum_simulator.py:904`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_simulator.py:909`
- Imported by: `advanced_experiments.py`

### _grid (method) `def _grid(self)`
- Defined: `quantum_simulator.py:913`
- Imported by: `advanced_experiments.py`

### harmonic (method) `def harmonic(self)`
- Defined: `quantum_simulator.py:918`
- Imported by: `advanced_experiments.py`

### double_well (method) `def double_well(self)`
- Defined: `quantum_simulator.py:923`
- Imported by: `advanced_experiments.py`

### coulomb (method) `def coulomb(self)`
- Defined: `quantum_simulator.py:929`
- Imported by: `advanced_experiments.py`

### periodic_lattice (method) `def periodic_lattice(self)`
- Defined: `quantum_simulator.py:935`
- Imported by: `advanced_experiments.py`

### mixed (method) `def mixed(self, seed)`
- Defined: `quantum_simulator.py:939`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_simulator.py:973`
- Imported by: `advanced_experiments.py`

### _empty (method) `def _empty(self, n_qubits)`
- Defined: `quantum_simulator.py:976`
- Imported by: `advanced_experiments.py`

### all_zeros (method) `def all_zeros(self, n_qubits)`
- Defined: `quantum_simulator.py:979`
- Imported by: `advanced_experiments.py`

### basis_state (method) `def basis_state(self, n_qubits, k)`
- Defined: `quantum_simulator.py:986`
- Imported by: `advanced_experiments.py`

### from_bitstring (method) `def from_bitstring(self, bitstring)`
- Defined: `quantum_simulator.py:995`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_simulator.py:1000`
- Imported by: `advanced_experiments.py`

### create_circuit (method) `def create_circuit(self, n_qubits)`
- Defined: `quantum_simulator.py:1011`
- Imported by: `advanced_experiments.py`

### run_circuit (method) `def run_circuit(self, circuit, initial_state, backend)`
- Defined: `quantum_simulator.py:1014`
- Imported by: `advanced_experiments.py`

### bell_state (method) `def bell_state(self, backend)`
- Defined: `quantum_simulator.py:1021`
- Imported by: `advanced_experiments.py`

### ghz_state (method) `def ghz_state(self, n_qubits, backend)`
- Defined: `quantum_simulator.py:1027`
- Imported by: `advanced_experiments.py`

### factory (method) `def factory(self)`
- Defined: `quantum_simulator.py:1035`
- Imported by: `advanced_experiments.py`

### backends (method) `def backends(self)`
- Defined: `quantum_simulator.py:1039`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_simulator.py:1044`
- Imported by: `advanced_experiments.py`

### radial_wavefunction (method) `def radial_wavefunction(n, l, r)`
- Defined: `quantum_simulator.py:1048`
- Imported by: `advanced_experiments.py`

### spherical_harmonic_real (method) `def spherical_harmonic_real(l, m, theta, phi)`
- Defined: `quantum_simulator.py:1060`
- Imported by: `advanced_experiments.py`

### psi_3d (method) `def psi_3d(self, n, l, m, r, theta, phi)`
- Defined: `quantum_simulator.py:1071`
- Imported by: `advanced_experiments.py`

### energy_analytical (method) `def energy_analytical(self, n)`
- Defined: `quantum_simulator.py:1076`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config, wavefunction_calc)`
- Defined: `quantum_simulator.py:1081`
- Imported by: `advanced_experiments.py`

### find_max_probability (method) `def find_max_probability(self, n, l, m)`
- Defined: `quantum_simulator.py:1085`
- Imported by: `advanced_experiments.py`

### sample_orbital (method) `def sample_orbital(self, n, l, m, num_samples, Z)`
- Defined: `quantum_simulator.py:1109`
- Imported by: `advanced_experiments.py`

### sample_entangled_state (method) `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples)`
- Defined: `quantum_simulator.py:1146`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_simulator.py:1160`
- Imported by: `advanced_experiments.py`

### energy_level_dirac (method) `def energy_level_dirac(self, n, kappa, Z)`
- Defined: `quantum_simulator.py:1165`
- Imported by: `advanced_experiments.py`

### energy_schrodinger (method) `def energy_schrodinger(self, n, Z)`
- Defined: `quantum_simulator.py:1173`
- Imported by: `advanced_experiments.py`

### fine_structure_splitting (method) `def fine_structure_splitting(self, n, l, Z)`
- Defined: `quantum_simulator.py:1176`
- Imported by: `advanced_experiments.py`

### energy_spectrum (method) `def energy_spectrum(self, n_max, Z)`
- Defined: `quantum_simulator.py:1184`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config, dirac_backend)`
- Defined: `quantum_simulator.py:1201`
- Imported by: `advanced_experiments.py`

### create_gaussian_wave_packet (method) `def create_gaussian_wave_packet(self, sigma, momentum)`
- Defined: `quantum_simulator.py:1206`
- Imported by: `advanced_experiments.py`

### compute_position_expectation (method) `def compute_position_expectation(self, spinor)`
- Defined: `quantum_simulator.py:1224`
- Imported by: `advanced_experiments.py`

### compute_velocity_expectation (method) `def compute_velocity_expectation(self, spinor)`
- Defined: `quantum_simulator.py:1235`
- Imported by: `advanced_experiments.py`

### simulate (method) `def simulate(self, duration, dt, sigma)`
- Defined: `quantum_simulator.py:1246`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_simulator.py:1283`
- Imported by: `advanced_experiments.py`

### visualize (method) `def visualize(self, data, save_path, title_suffix)`
- Defined: `quantum_simulator.py:1286`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_simulator.py:1344`
- Imported by: `advanced_experiments.py`

### visualize (method) `def visualize(self, data, quantum_result, save_path)`
- Defined: `quantum_simulator.py:1347`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, config_path)`
- Defined: `quantum_simulator.py:1410`
- Imported by: `advanced_experiments.py`

### _ensure_output_dir (method) `def _ensure_output_dir(self)`
- Defined: `quantum_simulator.py:1422`
- Imported by: `advanced_experiments.py`

### list_available_atoms (method) `def list_available_atoms(self)`
- Defined: `quantum_simulator.py:1425`
- Imported by: `advanced_experiments.py`

### list_available_molecules (method) `def list_available_molecules(self)`
- Defined: `quantum_simulator.py:1428`
- Imported by: `advanced_experiments.py`

### list_available_orbitals (method) `def list_available_orbitals(self)`
- Defined: `quantum_simulator.py:1431`
- Imported by: `advanced_experiments.py`

### get_atom (method) `def get_atom(self, symbol)`
- Defined: `quantum_simulator.py:1434`
- Imported by: `advanced_experiments.py`

### get_molecule (method) `def get_molecule(self, name)`
- Defined: `quantum_simulator.py:1437`
- Imported by: `advanced_experiments.py`

### get_orbital (method) `def get_orbital(self, name)`
- Defined: `quantum_simulator.py:1440`
- Imported by: `advanced_experiments.py`

### run_quantum_circuit (method) `def run_quantum_circuit(self, circuit, backend)`
- Defined: `quantum_simulator.py:1443`
- Imported by: `advanced_experiments.py`

### visualize_orbital (method) `def visualize_orbital(self, orbital_name, num_samples, save, Z, title_suffix)`
- Defined: `quantum_simulator.py:1446`
- Imported by: `advanced_experiments.py`

### visualize_atom_orbitals (method) `def visualize_atom_orbitals(self, atom_symbol, num_samples, save)`
- Defined: `quantum_simulator.py:1458`
- Imported by: `advanced_experiments.py`

### visualize_entangled_state (method) `def visualize_entangled_state(self, orbital1, orbital2, num_samples, save)`
- Defined: `quantum_simulator.py:1482`
- Imported by: `advanced_experiments.py`

### compute_relativistic_energy (method) `def compute_relativistic_energy(self, n, l, Z)`
- Defined: `quantum_simulator.py:1496`
- Imported by: `advanced_experiments.py`

### compute_energy_spectrum (method) `def compute_energy_spectrum(self, n_max, Z)`
- Defined: `quantum_simulator.py:1499`
- Imported by: `advanced_experiments.py`

### run_zitterbewegung_simulation (method) `def run_zitterbewegung_simulation(self, duration, dt, sigma)`
- Defined: `quantum_simulator.py:1502`
- Imported by: `advanced_experiments.py`

### run_all_demonstrations (method) `def run_all_demonstrations(self, num_samples)`
- Defined: `quantum_simulator.py:1505`
- Imported by: `advanced_experiments.py`

### __init__ (method) `def __init__(self, framework)`
- Defined: `quantum_simulator.py:1536`
- Imported by: `advanced_experiments.py`

### display_header (method) `def display_header(self)`
- Defined: `quantum_simulator.py:1540`
- Imported by: `advanced_experiments.py`

### display_main_menu (method) `def display_main_menu(self)`
- Defined: `quantum_simulator.py:1548`
- Imported by: `advanced_experiments.py`

### get_user_choice (method) `def get_user_choice(self, prompt)`
- Defined: `quantum_simulator.py:1564`
- Imported by: `advanced_experiments.py`

### _display_quantum_result (method) `def _display_quantum_result(self, result, n_qubits)`
- Defined: `quantum_simulator.py:1570`
- Imported by: `advanced_experiments.py`

### orbital_menu (method) `def orbital_menu(self)`
- Defined: `quantum_simulator.py:1578`
- Imported by: `advanced_experiments.py`

### atom_orbital_menu (method) `def atom_orbital_menu(self)`
- Defined: `quantum_simulator.py:1602`
- Imported by: `advanced_experiments.py`

### entangled_menu (method) `def entangled_menu(self)`
- Defined: `quantum_simulator.py:1622`
- Imported by: `advanced_experiments.py`

### quantum_circuit_menu (method) `def quantum_circuit_menu(self)`
- Defined: `quantum_simulator.py:1651`
- Imported by: `advanced_experiments.py`

### relativistic_menu (method) `def relativistic_menu(self)`
- Defined: `quantum_simulator.py:1737`
- Imported by: `advanced_experiments.py`

### zitterbewegung_menu (method) `def zitterbewegung_menu(self)`
- Defined: `quantum_simulator.py:1777`
- Imported by: `advanced_experiments.py`

### molecular_menu (method) `def molecular_menu(self)`
- Defined: `quantum_simulator.py:1796`
- Imported by: `advanced_experiments.py`

### atomic_menu (method) `def atomic_menu(self)`
- Defined: `quantum_simulator.py:1820`
- Imported by: `advanced_experiments.py`

### run (method) `def run(self)`
- Defined: `quantum_simulator.py:1840`
- Imported by: `advanced_experiments.py`

## quantum_visualizer.py

### _make_logger (function) `def _make_logger(name)`
- Defined: `quantum_visualizer.py:93`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### main (method) `def main()`
- Defined: `quantum_visualizer.py:856`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### render (method) `def render(self, data, axes, config)`
- Defined: `quantum_visualizer.py:192`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### render (method) `def render(self, data, axes, config)`
- Defined: `quantum_visualizer.py:197`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### _get_colors (method) `def _get_colors(self, probs, config)`
- Defined: `quantum_visualizer.py:218`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### render (method) `def render(self, data, axes, config)`
- Defined: `quantum_visualizer.py:228`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### render (method) `def render(self, data, axes, config)`
- Defined: `quantum_visualizer.py:259`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### render (method) `def render(self, snapshots, axes, config)`
- Defined: `quantum_visualizer.py:292`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### render (method) `def render(self, results, axes, config)`
- Defined: `quantum_visualizer.py:314`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_visualizer.py:342`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### compute_probabilities (method) `def compute_probabilities(self, state)`
- Defined: `quantum_visualizer.py:345`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### compute_phases (method) `def compute_phases(self, state)`
- Defined: `quantum_visualizer.py:349`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### compute_entropy (method) `def compute_entropy(self, probs)`
- Defined: `quantum_visualizer.py:360`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### compute_bloch_vectors (method) `def compute_bloch_vectors(self, state)`
- Defined: `quantum_visualizer.py:367`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### create_snapshot (method) `def create_snapshot(self, state, step, gate_name)`
- Defined: `quantum_visualizer.py:374`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, qc, config)`
- Defined: `quantum_visualizer.py:398`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### execute_sequence (method) `def execute_sequence(self, gates, n_qubits, backend_name)`
- Defined: `quantum_visualizer.py:403`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### compare_backends (method) `def compare_backends(self, gates, n_qubits, reference_backend)`
- Defined: `quantum_visualizer.py:423`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### bell_state (method) `def bell_state()`
- Defined: `quantum_visualizer.py:459`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### ghz_state (method) `def ghz_state(n_qubits)`
- Defined: `quantum_visualizer.py:466`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### qft (method) `def qft(n_qubits)`
- Defined: `quantum_visualizer.py:473`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### grover_oracle (method) `def grover_oracle(n_qubits, marked)`
- Defined: `quantum_visualizer.py:488`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### grover_diffusion (method) `def grover_diffusion(n_qubits)`
- Defined: `quantum_visualizer.py:501`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### custom_sequence (method) `def custom_sequence(sequence)`
- Defined: `quantum_visualizer.py:514`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_visualizer.py:529`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### build_evolution_figure (method) `def build_evolution_figure(self, snapshots, backend_results)`
- Defined: `quantum_visualizer.py:537`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### build_summary_figure (method) `def build_summary_figure(self, snapshots, backend_results)`
- Defined: `quantum_visualizer.py:563`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### _render_backend_fidelity (method) `def _render_backend_fidelity(self, results, axes)`
- Defined: `quantum_visualizer.py:594`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `quantum_visualizer.py:613`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### _initialize (method) `def _initialize(self)`
- Defined: `quantum_visualizer.py:620`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### _init_quantum_computer (method) `def _init_quantum_computer(self)`
- Defined: `quantum_visualizer.py:629`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### visualize_bell_state (method) `def visualize_bell_state(self)`
- Defined: `quantum_visualizer.py:654`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### visualize_ghz_state (method) `def visualize_ghz_state(self, n_qubits)`
- Defined: `quantum_visualizer.py:683`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### visualize_qft (method) `def visualize_qft(self, n_qubits)`
- Defined: `quantum_visualizer.py:711`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### visualize_grover (method) `def visualize_grover(self, n_qubits, marked_state)`
- Defined: `quantum_visualizer.py:739`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### visualize_custom_circuit (method) `def visualize_custom_circuit(self, gates, n_qubits, name)`
- Defined: `quantum_visualizer.py:778`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### run_all_visualizations (method) `def run_all_visualizations(self)`
- Defined: `quantum_visualizer.py:810`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### _save_figure (method) `def _save_figure(self, fig, name)`
- Defined: `quantum_visualizer.py:825`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

### _print_summary (method) `def _print_summary(self, results)`
- Defined: `quantum_visualizer.py:842`
- Depends on: `advanced_experiments.py`, `molecular_sim.py`, `quantum_computer.py`
- Imported by: `quantum_3dview.py`, `quantum_framework_menu.py`

## relativistic_hydrogen.py

### main (method) `def main()`
- Defined: `relativistic_hydrogen.py:1613`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### create_logger (method) `def create_logger(name, level)`
- Defined: `relativistic_hydrogen.py:97`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, device)`
- Defined: `relativistic_hydrogen.py:118`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### _init_matrices (method) `def _init_matrices(self)`
- Defined: `relativistic_hydrogen.py:122`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:207`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### _precompute_operators (method) `def _precompute_operators(self)`
- Defined: `relativistic_hydrogen.py:215`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### apply_dirac_hamiltonian (method) `def apply_dirac_hamiltonian(self, spinor, potential)`
- Defined: `relativistic_hydrogen.py:223`
- Doc: Apply Dirac Hamiltonian to 4-component spinor.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### time_evolution (method) `def time_evolution(self, spinor, dt, potential)`
- Defined: `relativistic_hydrogen.py:282`
- Doc: Time evolution of Dirac spinor using first-order split-step.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, channels, grid_size)`
- Defined: `relativistic_hydrogen.py:311`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### forward (method) `def forward(self, x)`
- Defined: `relativistic_hydrogen.py:322`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, spinor_components)`
- Defined: `relativistic_hydrogen.py:356`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### forward (method) `def forward(self, x)`
- Defined: `relativistic_hydrogen.py:379`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:397`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### _find_best_checkpoint (method) `def _find_best_checkpoint(self)`
- Defined: `relativistic_hydrogen.py:406`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### _load_model (method) `def _load_model(self)`
- Defined: `relativistic_hydrogen.py:452`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### apply_hamiltonian (method) `def apply_hamiltonian(self, spinor, potential)`
- Defined: `relativistic_hydrogen.py:498`
- Doc: Apply Hamiltonian using analytical operator.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### evolve_spinor (method) `def evolve_spinor(self, spinor, dt, potential)`
- Defined: `relativistic_hydrogen.py:506`
- Doc: Evolve spinor in time using the analytical Dirac operator.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:521`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### energy_level_dirac (method) `def energy_level_dirac(self, n, kappa)`
- Defined: `relativistic_hydrogen.py:526`
- Doc: Exact Dirac energy level for hydrogen-like atom.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### fine_structure_splitting (method) `def fine_structure_splitting(self, n, l)`
- Defined: `relativistic_hydrogen.py:556`
- Doc: Calculate fine structure splitting for given n, l.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### energy_spectrum (method) `def energy_spectrum(self, n_max)`
- Defined: `relativistic_hydrogen.py:597`
- Doc: Generate relativistic energy spectrum up to n_max.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, config, model_wrapper)`
- Defined: `relativistic_hydrogen.py:650`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### create_gaussian_wave_packet (method) `def create_gaussian_wave_packet(self, sigma, momentum)`
- Defined: `relativistic_hydrogen.py:657`
- Doc: Create a Gaussian wave packet for a free particle.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### compute_position_expectation (method) `def compute_position_expectation(self, spinor)`
- Defined: `relativistic_hydrogen.py:702`
- Doc: Compute expectation value of position operator.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### compute_velocity_expectation (method) `def compute_velocity_expectation(self, spinor)`
- Defined: `relativistic_hydrogen.py:724`
- Doc: Compute expectation value of velocity operator.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### simulate (method) `def simulate(self, duration, dt, sigma)`
- Defined: `relativistic_hydrogen.py:750`
- Doc: Run Zitterbewegung simulation.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:837`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### radial_wavefunction_schrodinger (method) `def radial_wavefunction_schrodinger(n, l, r)`
- Defined: `relativistic_hydrogen.py:843`
- Doc: Non-relativistic radial wavefunction for comparison.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### radial_wavefunction_dirac (method) `def radial_wavefunction_dirac(self, n, kappa, r, Z)`
- Defined: `relativistic_hydrogen.py:853`
- Doc: Relativistic radial wavefunctions for hydrogen.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### spherical_harmonic_real (method) `def spherical_harmonic_real(self, l, m, theta, phi)`
- Defined: `relativistic_hydrogen.py:900`
- Doc: Real spherical harmonics.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### spin_angular_function (method) `def spin_angular_function(self, kappa, m_j, theta, phi)`
- Defined: `relativistic_hydrogen.py:910`
- Doc: Spin-angular functions Omega_{kappa,m_j}(theta, phi).
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, config, model_wrapper)`
- Defined: `relativistic_hydrogen.py:960`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### sample_orbital (method) `def sample_orbital(self, n, l, j, num_samples)`
- Defined: `relativistic_hydrogen.py:966`
- Doc: Sample points from a relativistic hydrogen orbital.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:1079`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### visualize_orbital (method) `def visualize_orbital(self, data, save_path)`
- Defined: `relativistic_hydrogen.py:1082`
- Doc: Visualize relativistic orbital.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### visualize_energy_spectrum (method) `def visualize_energy_spectrum(self, spectrum, save_path)`
- Defined: `relativistic_hydrogen.py:1213`
- Doc: Visualize relativistic energy spectrum with fine structure.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### visualize_zitterbewegung (method) `def visualize_zitterbewegung(self, zbw_data, save_path)`
- Defined: `relativistic_hydrogen.py:1296`
- Doc: Visualize Zitterbewegung oscillation.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:1374`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### print_header (method) `def print_header(self)`
- Defined: `relativistic_hydrogen.py:1398`
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### validate_fine_structure (method) `def validate_fine_structure(self)`
- Defined: `relativistic_hydrogen.py:1419`
- Doc: Validate fine structure energy corrections.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### validate_zitterbewegung (method) `def validate_zitterbewegung(self)`
- Defined: `relativistic_hydrogen.py:1477`
- Doc: Validate Zitterbewegung simulation.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### validate_energy_spectrum (method) `def validate_energy_spectrum(self)`
- Defined: `relativistic_hydrogen.py:1509`
- Doc: Validate complete energy spectrum.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### validate_orbital (method) `def validate_orbital(self, orbital_name, num_samples)`
- Defined: `relativistic_hydrogen.py:1524`
- Doc: Validate single orbital visualization.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### run_full_validation (method) `def run_full_validation(self)`
- Defined: `relativistic_hydrogen.py:1541`
- Doc: Run complete validation suite.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

### interactive_mode (method) `def interactive_mode(self)`
- Defined: `relativistic_hydrogen.py:1575`
- Doc: Run in interactive mode.
- Imported by: `advanced_experiments.py`, `entangled_hydrogen.py`

## test_qc_integration.py

### config (function) `def config()`
- Defined: `test_qc_integration.py:42`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### bridge (function) `def bridge(config)`
- Defined: `test_qc_integration.py:47`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### qasm_adapter (function) `def qasm_adapter(config)`
- Defined: `test_qc_integration.py:52`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### bell_circuit (function) `def bell_circuit()`
- Defined: `test_qc_integration.py:57`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### ghz_circuit (function) `def ghz_circuit()`
- Defined: `test_qc_integration.py:62`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_default_config_has_supported_gates (method) `def test_default_config_has_supported_gates(self, config)`
- Defined: `test_qc_integration.py:73`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_gate_name_map_is_complete (method) `def test_gate_name_map_is_complete(self, config)`
- Defined: `test_qc_integration.py:79`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_reverse_gate_name_map_is_consistent (method) `def test_reverse_gate_name_map_is_consistent(self, config)`
- Defined: `test_qc_integration.py:83`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_qasm_version_default (method) `def test_qasm_version_default(self, config)`
- Defined: `test_qc_integration.py:87`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_max_qubits_defaults_are_positive (method) `def test_max_qubits_defaults_are_positive(self, config)`
- Defined: `test_qc_integration.py:90`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_create_single_qubit_gate (method) `def test_create_single_qubit_gate(self)`
- Defined: `test_qc_integration.py:103`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_create_two_qubit_gate (method) `def test_create_two_qubit_gate(self)`
- Defined: `test_qc_integration.py:109`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_create_gate_with_params (method) `def test_create_gate_with_params(self)`
- Defined: `test_qc_integration.py:114`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_targets_are_immutable (method) `def test_targets_are_immutable(self)`
- Defined: `test_qc_integration.py:118`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_create_empty_circuit (method) `def test_create_empty_circuit(self)`
- Defined: `test_qc_integration.py:131`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_append_gate (method) `def test_append_gate(self)`
- Defined: `test_qc_integration.py:136`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_append_gate_out_of_range_raises (method) `def test_append_gate_out_of_range_raises(self)`
- Defined: `test_qc_integration.py:141`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_multiple_gates (method) `def test_multiple_gates(self)`
- Defined: `test_qc_integration.py:146`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_repr_includes_qubits_and_gates (method) `def test_repr_includes_qubits_and_gates(self)`
- Defined: `test_qc_integration.py:152`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_bell_state_contains_header (method) `def test_export_bell_state_contains_header(self, qasm_adapter, bell_circuit)`
- Defined: `test_qc_integration.py:168`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_bell_state_has_qreg_and_creg (method) `def test_export_bell_state_has_qreg_and_creg(self, qasm_adapter, bell_circuit)`
- Defined: `test_qc_integration.py:173`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_bell_state_has_gates (method) `def test_export_bell_state_has_gates(self, qasm_adapter, bell_circuit)`
- Defined: `test_qc_integration.py:178`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_ghz_state (method) `def test_export_ghz_state(self, qasm_adapter, ghz_circuit)`
- Defined: `test_qc_integration.py:183`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_qft_has_swap (method) `def test_export_qft_has_swap(self, qasm_adapter, config)`
- Defined: `test_qc_integration.py:189`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_parametric_gate (method) `def test_export_parametric_gate(self, qasm_adapter)`
- Defined: `test_qc_integration.py:194`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_exceeds_max_qubits_raises (method) `def test_export_exceeds_max_qubits_raises(self, qasm_adapter)`
- Defined: `test_qc_integration.py:201`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_import_bell_state_roundtrip (method) `def test_import_bell_state_roundtrip(self, qasm_adapter, bell_circuit)`
- Defined: `test_qc_integration.py:206`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_import_ghz_roundtrip (method) `def test_import_ghz_roundtrip(self, qasm_adapter, ghz_circuit)`
- Defined: `test_qc_integration.py:214`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_import_from_standard_qasm_string (method) `def test_import_from_standard_qasm_string(self, qasm_adapter)`
- Defined: `test_qc_integration.py:220`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_import_with_parametric_gates (method) `def test_import_with_parametric_gates(self, qasm_adapter)`
- Defined: `test_qc_integration.py:234`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_import_empty_qasm_returns_zero_qubit_circuit (method) `def test_import_empty_qasm_returns_zero_qubit_circuit(self, qasm_adapter)`
- Defined: `test_qc_integration.py:247`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_bell_state_has_two_gates (method) `def test_bell_state_has_two_gates(self)`
- Defined: `test_qc_integration.py:263`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_bell_state_has_two_qubits (method) `def test_bell_state_has_two_qubits(self)`
- Defined: `test_qc_integration.py:269`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_ghz_state (method) `def test_ghz_state(self)`
- Defined: `test_qc_integration.py:273`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_qft_three_qubits (method) `def test_qft_three_qubits(self)`
- Defined: `test_qc_integration.py:278`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_grover_iterations (method) `def test_grover_iterations(self)`
- Defined: `test_qc_integration.py:283`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_qasm_returns_string (method) `def test_export_qasm_returns_string(self, bridge, bell_circuit)`
- Defined: `test_qc_integration.py:295`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_import_qasm_roundtrip (method) `def test_import_qasm_roundtrip(self, bridge, bell_circuit)`
- Defined: `test_qc_integration.py:300`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_full_openqasm_roundtrip_bell (method) `def test_full_openqasm_roundtrip_bell(self, bridge)`
- Defined: `test_qc_integration.py:306`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_full_openqasm_roundtrip_ghz (method) `def test_full_openqasm_roundtrip_ghz(self, bridge)`
- Defined: `test_qc_integration.py:313`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_full_openqasm_roundtrip_qft (method) `def test_full_openqasm_roundtrip_qft(self, bridge)`
- Defined: `test_qc_integration.py:320`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_qiskit_not_available_by_default (method) `def test_qiskit_not_available_by_default(self, bridge)`
- Defined: `test_qc_integration.py:327`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_pennylane_not_available_by_default (method) `def test_pennylane_not_available_by_default(self, bridge)`
- Defined: `test_qc_integration.py:332`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_qasm_custom_qreg_name (method) `def test_export_qasm_custom_qreg_name(self, bridge, bell_circuit)`
- Defined: `test_qc_integration.py:337`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_create_single_qubit_gate (method) `def test_create_single_qubit_gate(self)`
- Defined: `test_qc_integration.py:349`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_create_two_qubit_gate (method) `def test_create_two_qubit_gate(self)`
- Defined: `test_qc_integration.py:357`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_create_parametric_gate (method) `def test_create_parametric_gate(self)`
- Defined: `test_qc_integration.py:362`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_default_values (method) `def test_default_values(self)`
- Defined: `test_qc_integration.py:375`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_gate_list_includes_standard_gates (method) `def test_gate_list_includes_standard_gates(self)`
- Defined: `test_qc_integration.py:382`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_qasm_initial_contains_header (method) `def test_qasm_initial_contains_header(self)`
- Defined: `test_qc_integration.py:389`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_engine_available_with_matplotlib (method) `def test_engine_available_with_matplotlib(self)`
- Defined: `test_qc_integration.py:403`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_render_full_dashboard_returns_bytes (method) `def test_render_full_dashboard_returns_bytes(self)`
- Defined: `test_qc_integration.py:415`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_synthetic_execute_returns_snapshots (method) `def test_synthetic_execute_returns_snapshots(self)`
- Defined: `test_qc_integration.py:444`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_empty_circuit_returns_init_snapshot (method) `def test_empty_circuit_returns_init_snapshot(self)`
- Defined: `test_qc_integration.py:457`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_create_snapshot (method) `def test_create_snapshot(self)`
- Defined: `test_qc_integration.py:476`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_entropy_updates (method) `def test_entropy_updates(self)`
- Defined: `test_qc_integration.py:489`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_probabilities_normalized (method) `def test_probabilities_normalized(self)`
- Defined: `test_qc_integration.py:500`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_gate_instruction_empty_targets (method) `def test_gate_instruction_empty_targets(self)`
- Defined: `test_qc_integration.py:520`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_circuit_ir_append_negative_qubit_raises (method) `def test_circuit_ir_append_negative_qubit_raises(self)`
- Defined: `test_qc_integration.py:524`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_openqasm_import_empty_string (method) `def test_openqasm_import_empty_string(self, qasm_adapter)`
- Defined: `test_qc_integration.py:529`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_openqasm_import_garbage_string (method) `def test_openqasm_import_garbage_string(self, qasm_adapter)`
- Defined: `test_qc_integration.py:534`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_openqasm_export_zero_qubit_circuit (method) `def test_openqasm_export_zero_qubit_circuit(self, qasm_adapter)`
- Defined: `test_qc_integration.py:538`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_standard_circuit_factory_qft_one_qubit (method) `def test_standard_circuit_factory_qft_one_qubit(self)`
- Defined: `test_qc_integration.py:543`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_standard_circuit_factory_grover_minimal (method) `def test_standard_circuit_factory_grover_minimal(self)`
- Defined: `test_qc_integration.py:548`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_circuit_ir_repr_no_gates (method) `def test_circuit_ir_repr_no_gates(self)`
- Defined: `test_qc_integration.py:552`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_synthetic_snapshot_probabilities_sum_to_one (method) `def test_synthetic_snapshot_probabilities_sum_to_one(self)`
- Defined: `test_qc_integration.py:557`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_visualisation_engine_handles_no_snapshots (method) `def test_visualisation_engine_handles_no_snapshots(self)`
- Defined: `test_qc_integration.py:570`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_build_export_import_qasm_roundtrip (method) `def test_build_export_import_qasm_roundtrip(self, bridge)`
- Defined: `test_qc_integration.py:588`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_qasm_to_circuitir_to_framework_mps (method) `def test_qasm_to_circuitir_to_framework_mps(self, bridge)`
- Defined: `test_qc_integration.py:598`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_ghz_export_qasm_and_reimport_matches (method) `def test_ghz_export_qasm_and_reimport_matches(self, bridge)`
- Defined: `test_qc_integration.py:609`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_qft_circuit_qasm_roundtrip (method) `def test_qft_circuit_qasm_roundtrip(self, bridge)`
- Defined: `test_qc_integration.py:616`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_export_qasm_with_custom_names (method) `def test_export_qasm_with_custom_names(self, bridge)`
- Defined: `test_qc_integration.py:622`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_import_qasm_preserves_gate_order (method) `def test_import_qasm_preserves_gate_order(self, bridge)`
- Defined: `test_qc_integration.py:628`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

### test_full_pipeline_build_export_import_to_mps (method) `def test_full_pipeline_build_export_import_to_mps(self, bridge)`
- Defined: `test_qc_integration.py:644`
- Depends on: `qc_dashboard.py`, `qc_integration.py`, `quantum_framework_core.py`

## test_quantum_framework.py

### config (function) `def config()`
- Defined: `test_quantum_framework.py:43`
- Doc: Create default framework configuration.
- Depends on: `quantum_framework_core.py`

### qc (function) `def qc(config)`
- Defined: `test_quantum_framework.py:53`
- Doc: Create quantum computer instance.
- Depends on: `quantum_framework_core.py`

### config_precision (function) `def config_precision()`
- Defined: `test_quantum_framework.py:59`
- Doc: Create precision mode configuration.
- Depends on: `quantum_framework_core.py`

### test_default_config (method) `def test_default_config(self)`
- Defined: `test_quantum_framework.py:75`
- Doc: Test default configuration values.
- Depends on: `quantum_framework_core.py`

### test_custom_config (method) `def test_custom_config(self)`
- Defined: `test_quantum_framework.py:83`
- Doc: Test custom configuration values.
- Depends on: `quantum_framework_core.py`

### test_create_state (method) `def test_create_state(self, config)`
- Defined: `test_quantum_framework.py:98`
- Doc: Test state creation.
- Depends on: `quantum_framework_core.py`

### test_initial_state (method) `def test_initial_state(self, qc)`
- Defined: `test_quantum_framework.py:104`
- Doc: Test initial |00...0> state.
- Depends on: `quantum_framework_core.py`

### test_clone_state (method) `def test_clone_state(self, qc)`
- Defined: `test_quantum_framework.py:113`
- Doc: Test state cloning.
- Depends on: `quantum_framework_core.py`

### test_bell_entropy (method) `def test_bell_entropy(self, qc)`
- Defined: `test_quantum_framework.py:133`
- Doc: Test Bell state entropy.
- Depends on: `quantum_framework_core.py`

### test_bell_probabilities (method) `def test_bell_probabilities(self, qc)`
- Defined: `test_quantum_framework.py:141`
- Doc: Test Bell state probabilities.
- Depends on: `quantum_framework_core.py`

### test_bell_entanglement (method) `def test_bell_entanglement(self, qc)`
- Defined: `test_quantum_framework.py:154`
- Doc: Test Bell state entanglement.
- Depends on: `quantum_framework_core.py`

### test_ghz_entropy (method) `def test_ghz_entropy(self, qc)`
- Defined: `test_quantum_framework.py:170`
- Doc: Test GHZ state entropy.
- Depends on: `quantum_framework_core.py`

### test_ghz_probabilities (method) `def test_ghz_probabilities(self, qc)`
- Defined: `test_quantum_framework.py:179`
- Doc: Test GHZ state probabilities.
- Depends on: `quantum_framework_core.py`

### test_ghz_scaling (method) `def test_ghz_scaling(self, qc)`
- Defined: `test_quantum_framework.py:192`
- Doc: Test GHZ state memory scaling.
- Depends on: `quantum_framework_core.py`

### test_w_state_probabilities (method) `def test_w_state_probabilities(self, qc)`
- Defined: `test_quantum_framework.py:212`
- Doc: Test W state probabilities.
- Depends on: `quantum_framework_core.py`

### test_w_state_entropy (method) `def test_w_state_entropy(self, qc)`
- Defined: `test_quantum_framework.py:227`
- Doc: Test W state entropy.
- Depends on: `quantum_framework_core.py`

### test_w_state_no_zero_probabilities (method) `def test_w_state_no_zero_probabilities(self, qc)`
- Defined: `test_quantum_framework.py:247`
- Doc: Test that W state has non-zero probabilities for single-excitation states.
- Depends on: `quantum_framework_core.py`

### test_hadamard_gate (method) `def test_hadamard_gate(self, qc)`
- Defined: `test_quantum_framework.py:264`
- Doc: Test Hadamard gate.
- Depends on: `quantum_framework_core.py`

### test_pauli_x_gate (method) `def test_pauli_x_gate(self, qc)`
- Defined: `test_quantum_framework.py:275`
- Doc: Test Pauli-X gate.
- Depends on: `quantum_framework_core.py`

### test_pauli_z_gate (method) `def test_pauli_z_gate(self, qc)`
- Defined: `test_quantum_framework.py:285`
- Doc: Test Pauli-Z gate on |+> state.
- Depends on: `quantum_framework_core.py`

### test_cnot_gate (method) `def test_cnot_gate(self, qc)`
- Defined: `test_quantum_framework.py:297`
- Doc: Test CNOT gate.
- Depends on: `quantum_framework_core.py`

### test_swap_gate (method) `def test_swap_gate(self, qc)`
- Defined: `test_quantum_framework.py:309`
- Doc: Test SWAP gate.
- Depends on: `quantum_framework_core.py`

### test_rotation_gates (method) `def test_rotation_gates(self, qc)`
- Defined: `test_quantum_framework.py:321`
- Doc: Test rotation gates.
- Depends on: `quantum_framework_core.py`

### test_hzh_equals_x (method) `def test_hzh_equals_x(self, qc)`
- Defined: `test_quantum_framework.py:341`
- Doc: Test HZH = X identity.
- Depends on: `quantum_framework_core.py`

### test_xx_equals_identity (method) `def test_xx_equals_identity(self, qc)`
- Defined: `test_quantum_framework.py:354`
- Doc: Test XX = I identity.
- Depends on: `quantum_framework_core.py`

### test_cnot_cnot_equals_identity (method) `def test_cnot_cnot_equals_identity(self, qc)`
- Defined: `test_quantum_framework.py:366`
- Doc: Test CNOT CNOT = I identity.
- Depends on: `quantum_framework_core.py`

### test_norm_preservation (method) `def test_norm_preservation(self, qc)`
- Defined: `test_quantum_framework.py:381`
- Doc: Test that norm is preserved after gates.
- Depends on: `quantum_framework_core.py`

### test_grover_3_qubits (method) `def test_grover_3_qubits(self, qc)`
- Defined: `test_quantum_framework.py:406`
- Doc: Test Grover search on 3 qubits.
- Depends on: `quantum_framework_core.py`

### test_grover_speedup (method) `def test_grover_speedup(self, qc)`
- Defined: `test_quantum_framework.py:417`
- Doc: Test Grover speedup.
- Depends on: `quantum_framework_core.py`

### test_qft_entropy (method) `def test_qft_entropy(self, qc)`
- Defined: `test_quantum_framework.py:434`
- Doc: Test QFT entropy.
- Depends on: `quantum_framework_core.py`

### test_memory_linear_scaling (method) `def test_memory_linear_scaling(self)`
- Defined: `test_quantum_framework.py:461`
- Doc: Test that memory scales sub-exponentially with qubits (MPS property).
- Depends on: `quantum_framework_core.py`

### test_compression_ratio (method) `def test_compression_ratio(self)`
- Defined: `test_quantum_framework.py:478`
- Doc: Test MPS compression ratio vs full statevector for large n.
- Depends on: `quantum_framework_core.py`

### test_precision_mode_config (method) `def test_precision_mode_config(self)`
- Defined: `test_quantum_framework.py:504`
- Doc: Test precision mode configuration.
- Depends on: `quantum_framework_core.py`

### test_precision_mode_bell_state (method) `def test_precision_mode_bell_state(self, config_precision)`
- Defined: `test_quantum_framework.py:510`
- Doc: Test Bell state in precision mode.
- Depends on: `quantum_framework_core.py`

### test_single_qubit (method) `def test_single_qubit(self, qc)`
- Defined: `test_quantum_framework.py:528`
- Doc: Test single qubit operations.
- Depends on: `quantum_framework_core.py`

### test_large_bond_dimension (method) `def test_large_bond_dimension(self)`
- Defined: `test_quantum_framework.py:539`
- Doc: Test with large bond dimension.
- Depends on: `quantum_framework_core.py`

### test_empty_circuit (method) `def test_empty_circuit(self, qc)`
- Defined: `test_quantum_framework.py:549`
- Doc: Test empty circuit.
- Depends on: `quantum_framework_core.py`

## topological_hilbert_compression2.py

### main (method) `def main()`
- Defined: `topological_hilbert_compression2.py:956`
- Depends on: `quantum_computer.py`

### __post_init__ (method) `def __post_init__(self)`
- Defined: `topological_hilbert_compression2.py:80`
- Depends on: `quantum_computer.py`

### amplitude (method) `def amplitude(self, basis_index)`
- Defined: `topological_hilbert_compression2.py:87`
- Depends on: `quantum_computer.py`

### apply_single_qubit_gate (method) `def apply_single_qubit_gate(self, qubit, gate)`
- Defined: `topological_hilbert_compression2.py:91`
- Depends on: `quantum_computer.py`

### apply_two_qubit_gate (method) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)`
- Defined: `topological_hilbert_compression2.py:95`
- Depends on: `quantum_computer.py`

### norm (method) `def norm(self)`
- Defined: `topological_hilbert_compression2.py:99`
- Depends on: `quantum_computer.py`

### probabilities (method) `def probabilities(self)`
- Defined: `topological_hilbert_compression2.py:103`
- Depends on: `quantum_computer.py`

### entropy (method) `def entropy(self)`
- Defined: `topological_hilbert_compression2.py:107`
- Depends on: `quantum_computer.py`

### memory_bytes (method) `def memory_bytes(self)`
- Defined: `topological_hilbert_compression2.py:111`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, chi_left, chi_right, d, device, dtype)`
- Defined: `topological_hilbert_compression2.py:121`
- Depends on: `quantum_computer.py`

### _initialize (method) `def _initialize(self)`
- Defined: `topological_hilbert_compression2.py:130`
- Depends on: `quantum_computer.py`

### tensor (method) `def tensor(self)`
- Defined: `topological_hilbert_compression2.py:136`
- Depends on: `quantum_computer.py`

### tensor (method) `def tensor(self, value)`
- Defined: `topological_hilbert_compression2.py:142`
- Depends on: `quantum_computer.py`

### left_canonicalize (method) `def left_canonicalize(self)`
- Defined: `topological_hilbert_compression2.py:147`
- Depends on: `quantum_computer.py`

### right_canonicalize (method) `def right_canonicalize(self)`
- Defined: `topological_hilbert_compression2.py:154`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, n_qubits, config)`
- Defined: `topological_hilbert_compression2.py:172`
- Depends on: `quantum_computer.py`

### _initialize (method) `def _initialize(self)`
- Defined: `topological_hilbert_compression2.py:181`
- Depends on: `quantum_computer.py`

### _bond_dimension (method) `def _bond_dimension(self, site)`
- Defined: `topological_hilbert_compression2.py:195`
- Depends on: `quantum_computer.py`

### amplitude (method) `def amplitude(self, basis_index)`
- Defined: `topological_hilbert_compression2.py:198`
- Depends on: `quantum_computer.py`

### apply_single_qubit_gate (method) `def apply_single_qubit_gate(self, qubit, gate)`
- Defined: `topological_hilbert_compression2.py:209`
- Depends on: `quantum_computer.py`

### apply_two_qubit_gate (method) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)`
- Defined: `topological_hilbert_compression2.py:219`
- Depends on: `quantum_computer.py`

### _swap_qubits_in_gate (method) `def _swap_qubits_in_gate(self, gate)`
- Defined: `topological_hilbert_compression2.py:236`
- Depends on: `quantum_computer.py`

### _apply_adjacent_gate (method) `def _apply_adjacent_gate(self, qubit, gate)`
- Defined: `topological_hilbert_compression2.py:245`
- Depends on: `quantum_computer.py`

### _apply_nonadjacent_gate (method) `def _apply_nonadjacent_gate(self, qubit_a, qubit_b, gate)`
- Defined: `topological_hilbert_compression2.py:285`
- Depends on: `quantum_computer.py`

### norm (method) `def norm(self)`
- Defined: `topological_hilbert_compression2.py:294`
- Depends on: `quantum_computer.py`

### _canonicalize (method) `def _canonicalize(self)`
- Defined: `topological_hilbert_compression2.py:301`
- Depends on: `quantum_computer.py`

### probabilities (method) `def probabilities(self)`
- Defined: `topological_hilbert_compression2.py:308`
- Depends on: `quantum_computer.py`

### entropy (method) `def entropy(self)`
- Defined: `topological_hilbert_compression2.py:318`
- Depends on: `quantum_computer.py`

### memory_bytes (method) `def memory_bytes(self)`
- Defined: `topological_hilbert_compression2.py:329`
- Depends on: `quantum_computer.py`

### entanglement_entropy (method) `def entanglement_entropy(self, cut)`
- Defined: `topological_hilbert_compression2.py:335`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, n_qubits, config)`
- Defined: `topological_hilbert_compression2.py:361`
- Depends on: `quantum_computer.py`

### _initialize (method) `def _initialize(self)`
- Defined: `topological_hilbert_compression2.py:370`
- Depends on: `quantum_computer.py`

### _compute_berry_phases (method) `def _compute_berry_phases(self)`
- Defined: `topological_hilbert_compression2.py:375`
- Depends on: `quantum_computer.py`

### add_active_state (method) `def add_active_state(self, basis_index, winding_number)`
- Defined: `topological_hilbert_compression2.py:381`
- Depends on: `quantum_computer.py`

### _compute_winding_number (method) `def _compute_winding_number(self, basis_index)`
- Defined: `topological_hilbert_compression2.py:389`
- Depends on: `quantum_computer.py`

### is_topologically_protected (method) `def is_topologically_protected(self, basis_index)`
- Defined: `topological_hilbert_compression2.py:394`
- Depends on: `quantum_computer.py`

### sparsity (method) `def sparsity(self)`
- Defined: `topological_hilbert_compression2.py:398`
- Depends on: `quantum_computer.py`

### project_to_active (method) `def project_to_active(self, state)`
- Defined: `topological_hilbert_compression2.py:403`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `topological_hilbert_compression2.py:419`
- Depends on: `quantum_computer.py`

### compute_winding_number (method) `def compute_winding_number(self, state, qubit)`
- Defined: `topological_hilbert_compression2.py:424`
- Depends on: `quantum_computer.py`

### compute_berry_phase (method) `def compute_berry_phase(self, state, qubit_a, qubit_b)`
- Defined: `topological_hilbert_compression2.py:433`
- Depends on: `quantum_computer.py`

### is_protected (method) `def is_protected(self, state, vacuum_core)`
- Defined: `topological_hilbert_compression2.py:444`
- Depends on: `quantum_computer.py`

### can_handle (method) `def can_handle(self, n_qubits)`
- Defined: `topological_hilbert_compression2.py:454`
- Depends on: `quantum_computer.py`

### create_state (method) `def create_state(self, n_qubits)`
- Defined: `topological_hilbert_compression2.py:458`
- Depends on: `quantum_computer.py`

### apply_gate (method) `def apply_gate(self, state, gate_name, targets, params)`
- Defined: `topological_hilbert_compression2.py:462`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `topological_hilbert_compression2.py:472`
- Depends on: `quantum_computer.py`

### _load_quantum_computer (method) `def _load_quantum_computer(self)`
- Defined: `topological_hilbert_compression2.py:479`
- Depends on: `quantum_computer.py`

### can_handle (method) `def can_handle(self, n_qubits)`
- Defined: `topological_hilbert_compression2.py:497`
- Depends on: `quantum_computer.py`

### create_state (method) `def create_state(self, n_qubits)`
- Defined: `topological_hilbert_compression2.py:500`
- Depends on: `quantum_computer.py`

### apply_gate (method) `def apply_gate(self, state, gate_name, targets, params)`
- Defined: `topological_hilbert_compression2.py:507`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `topological_hilbert_compression2.py:521`
- Depends on: `quantum_computer.py`

### _initialize_gate_cache (method) `def _initialize_gate_cache(self)`
- Defined: `topological_hilbert_compression2.py:526`
- Depends on: `quantum_computer.py`

### can_handle (method) `def can_handle(self, n_qubits)`
- Defined: `topological_hilbert_compression2.py:537`
- Depends on: `quantum_computer.py`

### create_state (method) `def create_state(self, n_qubits)`
- Defined: `topological_hilbert_compression2.py:540`
- Depends on: `quantum_computer.py`

### apply_gate (method) `def apply_gate(self, state, gate_name, targets, params)`
- Defined: `topological_hilbert_compression2.py:543`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `topological_hilbert_compression2.py:581`
- Depends on: `quantum_computer.py`

### _select_backend (method) `def _select_backend(self, n_qubits, force_mps)`
- Defined: `topological_hilbert_compression2.py:590`
- Depends on: `quantum_computer.py`

### create_circuit (method) `def create_circuit(self, n_qubits, force_mps)`
- Defined: `topological_hilbert_compression2.py:602`
- Depends on: `quantum_computer.py`

### h (method) `def h(self, qubit)`
- Defined: `topological_hilbert_compression2.py:610`
- Depends on: `quantum_computer.py`

### x (method) `def x(self, qubit)`
- Defined: `topological_hilbert_compression2.py:613`
- Depends on: `quantum_computer.py`

### y (method) `def y(self, qubit)`
- Defined: `topological_hilbert_compression2.py:616`
- Depends on: `quantum_computer.py`

### z (method) `def z(self, qubit)`
- Defined: `topological_hilbert_compression2.py:619`
- Depends on: `quantum_computer.py`

### rx (method) `def rx(self, qubit, theta)`
- Defined: `topological_hilbert_compression2.py:622`
- Depends on: `quantum_computer.py`

### ry (method) `def ry(self, qubit, theta)`
- Defined: `topological_hilbert_compression2.py:625`
- Depends on: `quantum_computer.py`

### rz (method) `def rz(self, qubit, theta)`
- Defined: `topological_hilbert_compression2.py:628`
- Depends on: `quantum_computer.py`

### cnot (method) `def cnot(self, control, target)`
- Defined: `topological_hilbert_compression2.py:631`
- Depends on: `quantum_computer.py`

### cz (method) `def cz(self, control, target)`
- Defined: `topological_hilbert_compression2.py:634`
- Depends on: `quantum_computer.py`

### swap (method) `def swap(self, qubit_a, qubit_b)`
- Defined: `topological_hilbert_compression2.py:637`
- Depends on: `quantum_computer.py`

### run (method) `def run(self)`
- Defined: `topological_hilbert_compression2.py:640`
- Depends on: `quantum_computer.py`

### probabilities (method) `def probabilities(self)`
- Defined: `topological_hilbert_compression2.py:650`
- Depends on: `quantum_computer.py`

### entropy (method) `def entropy(self)`
- Defined: `topological_hilbert_compression2.py:655`
- Depends on: `quantum_computer.py`

### memory_usage (method) `def memory_usage(self)`
- Defined: `topological_hilbert_compression2.py:666`
- Depends on: `quantum_computer.py`

### compression_ratio (method) `def compression_ratio(self)`
- Defined: `topological_hilbert_compression2.py:682`
- Depends on: `quantum_computer.py`

### detect_phase (method) `def detect_phase(self)`
- Defined: `topological_hilbert_compression2.py:691`
- Depends on: `quantum_computer.py`

### _compute_average_bond_dimension (method) `def _compute_average_bond_dimension(self)`
- Defined: `topological_hilbert_compression2.py:714`
- Depends on: `quantum_computer.py`

### __init__ (method) `def __init__(self, config)`
- Defined: `topological_hilbert_compression2.py:734`
- Depends on: `quantum_computer.py`

### run_bell_state (method) `def run_bell_state(self, n_qubits, use_mps)`
- Defined: `topological_hilbert_compression2.py:739`
- Depends on: `quantum_computer.py`

### run_ghz_state (method) `def run_ghz_state(self, n_qubits, use_mps)`
- Defined: `topological_hilbert_compression2.py:758`
- Depends on: `quantum_computer.py`

### run_w_state (method) `def run_w_state(self, n_qubits, use_mps)`
- Defined: `topological_hilbert_compression2.py:777`
- Depends on: `quantum_computer.py`

### prepare_ghz_state (method) `def prepare_ghz_state(self, n_qubits, force_mps)`
- Defined: `topological_hilbert_compression2.py:798`
- Doc: Prepare GHZ state directly without SWAP overhead.
- Depends on: `quantum_computer.py`

### run_scaling_benchmark (method) `def run_scaling_benchmark(self, max_qubits, use_mps)`
- Defined: `topological_hilbert_compression2.py:844`
- Depends on: `quantum_computer.py`

### run_all (method) `def run_all(self)`
- Defined: `topological_hilbert_compression2.py:897`
- Depends on: `quantum_computer.py`

### _print_summary (method) `def _print_summary(self)`
- Defined: `topological_hilbert_compression2.py:909`
- Depends on: `quantum_computer.py`
