# API (page 1 of 3)
Pages: [API.md](API.md), [API_p2.md](API_p2.md), [API_p3.md](API_p3.md)

## advanced_experiments.py
Depends on: `molecular_sim.py`, `quantum_computer.py`, `quantum_simulator.py`, `relativistic_hydrogen.py`
Imported by: `quantum_visualizer.py`
- `GroverOracle.__init__` (method) `advanced_experiments.py:167` `def __init__(self, n_qubits, marked_state)`
- `GroverOracle.apply` (method) `advanced_experiments.py:176` `def apply(self, state, backend)` -- Apply oracle: flip phase of marked state. |x> -> (-1)^{f(x)} |x> where f(x)=1 only for marked state.
- `GroverDiffusionOperator.__init__` (method) `advanced_experiments.py:203` `def __init__(self, n_qubits)`
- `GroverDiffusionOperator.apply` (method) `advanced_experiments.py:206` `def apply(self, state, backend)` -- Apply diffusion operator using Hadamard + Oracle on |0> + Hadamard.
- `GroverSearch.__init__` (method) `advanced_experiments.py:245` `def __init__(self, config)`
- `GroverSearch.run` (method) `advanced_experiments.py:300` `def run(self)` -- Run Grover's search algorithm.
- `LambShiftCalculator.__init__` (method) `advanced_experiments.py:440` `def __init__(self, config)`
- `LambShiftCalculator.bethe_formula` (method) `advanced_experiments.py:462` `def bethe_formula(self, n, l, Z)` -- Bethe's non-relativistic formula for Lamb shift.
- `LambShiftCalculator.full_lamb_shift` (method) `advanced_experiments.py:518` `def full_lamb_shift(self, n, l, j, Z)` -- Calculate full Lamb shift including radiative corrections.
- `LambShiftCalculator.compare_2s_2p` (method) `advanced_experiments.py:558` `def compare_2s_2p(self, Z)` -- Compare Lamb shift for 2s_{1/2} and 2p_{1/2} states.
- `AnomalousMagneticMoment.__init__` (method) `advanced_experiments.py:605` `def __init__(self, config)`
- `AnomalousMagneticMoment.schwinger_term` (method) `advanced_experiments.py:611` `def schwinger_term(self)` -- Schwinger's first-order result: a_e = α/(2π)
- `AnomalousMagneticMoment.second_order` (method) `advanced_experiments.py:619` `def second_order(self)` -- Second-order correction: (α/π)^2 * C_2
- `AnomalousMagneticMoment.third_order` (method) `advanced_experiments.py:627` `def third_order(self)` -- Third-order correction: (α/π)^3 * C_3
- `AnomalousMagneticMoment.fourth_order` (method) `advanced_experiments.py:635` `def fourth_order(self)` -- Fourth-order correction: (α/π)^4 * C_4
- `AnomalousMagneticMoment.fifth_order` (method) `advanced_experiments.py:643` `def fifth_order(self)` -- Fifth-order correction: (α/π)^5 * C_5
- `AnomalousMagneticMoment.calculate_a_e` (method) `advanced_experiments.py:651` `def calculate_a_e(self, order)` -- Calculate anomalous magnetic moment to specified order.
- `AnomalousMagneticMoment.full_report` (method) `advanced_experiments.py:686` `def full_report(self)` -- Generate a full report on g-2 calculations.
- `QEDEffectsExperiment.__init__` (method) `advanced_experiments.py:728` `def __init__(self, config)`
- `QEDEffectsExperiment.run_full_analysis` (method) `advanced_experiments.py:736` `def run_full_analysis(self)` -- Run complete QED analysis.
- `MoleculeBuilder.h2o` (method) `advanced_experiments.py:866` `def h2o(bond_length, angle_deg)` -- Build water molecule geometry.
- `MoleculeBuilder.nh3` (method) `advanced_experiments.py:898` `def nh3(bond_length, angle_deg)` -- Build ammonia molecule geometry.
- `MoleculeBuilder.ch4` (method) `advanced_experiments.py:933` `def ch4(bond_length)` -- Build methane molecule geometry.
- `PolyatomicVQE.__init__` (method) `advanced_experiments.py:976` `def __init__(self, config)`
- `PolyatomicVQE.run_pyscf` (method) `advanced_experiments.py:999` `def run_pyscf(self, molecule)` -- Run PySCF calculation for the molecule.
- `PolyatomicExperiment.__init__` (method) `advanced_experiments.py:1110` `def __init__(self, config)`
- `PolyatomicExperiment.run_analysis` (method) `advanced_experiments.py:1123` `def run_analysis(self, molecule_name)` -- Run complete analysis for a molecule.
- `PolyatomicExperiment.run_all` (method) `advanced_experiments.py:1164` `def run_all(self)` -- Run analysis for all molecules.
- `PolyatomicExperiment.scan_bond_length` (method) `advanced_experiments.py:1175` `def scan_bond_length(self, molecule_name, r_min, r_max, n_points)` -- Scan potential energy surface by varying bond length.
- `AdvancedExperimentRunner.__init__` (method) `advanced_experiments.py:1228` `def __init__(self)`
- `AdvancedExperimentRunner.run_grover` (method) `advanced_experiments.py:1235` `def run_grover(self, n_qubits, marked_state)` -- Run Grover's algorithm experiment.
- `AdvancedExperimentRunner.run_qed` (method) `advanced_experiments.py:1252` `def run_qed(self)` -- Run QED effects experiment.
- `AdvancedExperimentRunner.run_polyatomic` (method) `advanced_experiments.py:1266` `def run_polyatomic(self, molecule)` -- Run polyatomic molecule experiment.
- `AdvancedExperimentRunner.run_all` (method) `advanced_experiments.py:1278` `def run_all(self)` -- Run all experiments.
- `AdvancedExperimentRunner.main` (method) `advanced_experiments.py:1304` `def main()` -- Main entry point.

## app.py
Depends on: `molecular_sim.py`, `quantum_computer.py`
Imported by: `quantum_framework_menu.py`
- `VQEResult.givens_single_excitation` (method) `app.py:61` `def givens_single_excitation(state, o, v, theta, n_qubits, backend)` -- Apply a particle-conserving single excitation rotation between qubits o (occupied) and v (virtual).
- `VQEResult.particle_conserving_ansatz` (method) `app.py:109` `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)` -- Particle-conserving UCCSD-like ansatz: - Singles: Givens rotations (correct particle conservation) - Doubles: reuse...
- `StarkEvaluator.__init__` (method) `app.py:152` `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- `StarkEvaluator.eval_dipole` (method) `app.py:184` `def eval_dipole(self, amps_raw)`
- `StarkEvaluator.__call__` (method) `app.py:197` `def __call__(self, amps)`
- `DipoleOperatorBuilder.__init__` (method) `app.py:204` `def __init__(self, bond_length_angstrom)`
- `PolarizabilityCalculator.__init__` (method) `app.py:238` `def __init__(self)`
- `PolarizabilityCalculator.cost` (method) `app.py:290` `def cost(th)`
- `PolarizabilityCalculator.run` (method) `app.py:303` `def run(self)`

## demo_molecular_vqe.py
Depends on: `quantum_framework_molecular_v2.py`
- `print_header` (function) `demo_molecular_vqe.py:47` `def print_header(title)` -- Print formatted header.
- `check_dependencies` (function) `demo_molecular_vqe.py:54` `def check_dependencies()` -- Check and report available dependencies.
- `demo_h2_direct` (function) `demo_molecular_vqe.py:78` `def demo_h2_direct()` -- Demo: H2 with direct statevector (precision mode).
- `demo_h2_mps` (function) `demo_molecular_vqe.py:104` `def demo_h2_mps()` -- Demo: H2 with MPS compression.
- `demo_comparison` (function) `demo_molecular_vqe.py:131` `def demo_comparison()` -- Demo: Compare direct vs MPS.
- `demo_molecule_builder` (function) `demo_molecular_vqe.py:167` `def demo_molecule_builder()` -- Demo: OpenFermion molecule builder.
- `demo_smart_initialization` (function) `demo_molecular_vqe.py:197` `def demo_smart_initialization()` -- Demo: Smart parameter initialization.
- `demo_cached_operations` (function) `demo_molecular_vqe.py:231` `def demo_cached_operations()` -- Demo: Cached Pauli operations.
- `demo_config_from_toml` (function) `demo_molecular_vqe.py:277` `def demo_config_from_toml()` -- Demo: Configuration from TOML.
- `main` (function) `demo_molecular_vqe.py:302` `def main()` -- Run all demos.

## entangled_hydrogen.py
Depends on: `molecular_sim.py`, `quantum_computer.py`, `relativistic_hydrogen.py`
- `IEntangledState.name` (method) `entangled_hydrogen.py:119` `def name(self)` -- Return the name of the entangled state.
- `IEntangledState.prepare` (method) `entangled_hydrogen.py:123` `def prepare(self, n_qubits)` -- Prepare the entangled state on n qubits.
- `IEntangledState.get_theoretical_entropy` (method) `entangled_hydrogen.py:127` `def get_theoretical_entropy(self)` -- Return the theoretical Shannon entropy in bits.
- `BellState.name` (method) `entangled_hydrogen.py:135` `def name(self)`
- `BellState.prepare` (method) `entangled_hydrogen.py:138` `def prepare(self, qc, backend)`
- `BellState.get_theoretical_entropy` (method) `entangled_hydrogen.py:141` `def get_theoretical_entropy(self)`
- `GHZState.__init__` (method) `entangled_hydrogen.py:148` `def __init__(self, n_qubits)`
- `GHZState.name` (method) `entangled_hydrogen.py:152` `def name(self)`
- `GHZState.prepare` (method) `entangled_hydrogen.py:155` `def prepare(self, qc, backend)`
- `GHZState.get_theoretical_entropy` (method) `entangled_hydrogen.py:158` `def get_theoretical_entropy(self)`
- `WState.__init__` (method) `entangled_hydrogen.py:165` `def __init__(self, n_qubits)`
- `WState.name` (method) `entangled_hydrogen.py:169` `def name(self)`
- `WState.prepare` (method) `entangled_hydrogen.py:172` `def prepare(self, qc, backend, factory)`
- `WState.get_theoretical_entropy` (method) `entangled_hydrogen.py:190` `def get_theoretical_entropy(self)`
- `WavefunctionCalculator.__init__` (method) `entangled_hydrogen.py:200` `def __init__(self, config)`
- `WavefunctionCalculator.radial_wavefunction` (method) `entangled_hydrogen.py:204` `def radial_wavefunction(n, l, r)` -- Calculate non-relativistic radial wavefunction R_nl(r).
- `WavefunctionCalculator.spherical_harmonic_real` (method) `entangled_hydrogen.py:215` `def spherical_harmonic_real(l, m, theta, phi)` -- Calculate real spherical harmonics Y_lm(theta, phi).
- `WavefunctionCalculator.psi_on_grid` (method) `entangled_hydrogen.py:225` `def psi_on_grid(self, n, l, m)` -- Calculate wavefunction on 2D grid for quantum computer processing.
- `WavefunctionCalculator.psi_3d` (method) `entangled_hydrogen.py:245` `def psi_3d(self, n, l, m, r, theta, phi)` -- Calculate full 3D wavefunction psi_nlm(r, theta, phi).
- `EntangledHydrogenSampler.__init__` (method) `entangled_hydrogen.py:259` `def __init__(self, config, wavefunction_calc)`
- `EntangledHydrogenSampler.find_max_probability` (method) `entangled_hydrogen.py:263` `def find_max_probability(self, n, l, m)` -- Find maximum probability for rejection sampling.
- `EntangledHydrogenSampler.sample_orbital` (method) `entangled_hydrogen.py:295` `def sample_orbital(self, n, l, m, num_samples)` -- Sample points from a single hydrogen orbital using Monte Carlo.
- `EntangledHydrogenSampler.sample_entangled_state` (method) `entangled_hydrogen.py:362` `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples, entanglement_weight)` -- Sample from entangled hydrogen state.
- `EntangledHydrogenVisualizer.__init__` (method) `entangled_hydrogen.py:414` `def __init__(self, config)`
- `EntangledHydrogenVisualizer.visualize` (method) `entangled_hydrogen.py:417` `def visualize(self, data, quantum_result, save_path)` -- Create visualization of entangled hydrogen state.
- `EntangledHydrogenExperiment.__init__` (method) `entangled_hydrogen.py:577` `def __init__(self, config)`
- `EntangledHydrogenExperiment.run_bell_entangled_hydrogen` (method) `entangled_hydrogen.py:630` `def run_bell_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples, suffix)` -- Run Bell state entangled hydrogen visualization.
- `EntangledHydrogenExperiment.run_ghz_entangled_hydrogen` (method) `entangled_hydrogen.py:672` `def run_ghz_entangled_hydrogen(self, orbitals, backend, num_samples)` -- Run GHZ state entangled hydrogen visualization.
- `EntangledHydrogenExperiment.run_entangled_h_with_molecular_energy` (method) `entangled_hydrogen.py:745` `def run_entangled_h_with_molecular_energy(self, n1, l1, m1, n2, l2, m2, backend, num_samples)` -- Run entangled hydrogen with molecular energy evaluation using molecular_sim.py.
- `EntangledHydrogenExperiment.run_relativistic_entangled_hydrogen` (method) `entangled_hydrogen.py:794` `def run_relativistic_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples)` -- Run entangled hydrogen with relativistic Dirac calculations.
- `EntangledHydrogenExperiment.run_all_demonstrations` (method) `entangled_hydrogen.py:852` `def run_all_demonstrations(self, num_samples)` -- Run all available entangled hydrogen demonstrations.
- `EntangledHydrogenExperiment.main` (method) `entangled_hydrogen.py:893` `def main()` -- Main entry point for entangled hydrogen visualization.

## higgs_four_lepton_analysis.py
Depends on: `quantum_computer.py`
Imported by: `quantum_framework_menu.py`
- `FourMomentum.from_energy_momentum` (method) `higgs_four_lepton_analysis.py:151` `def from_energy_momentum(cls, E, px, py, pz)`
- `Lepton.pt` (method) `higgs_four_lepton_analysis.py:171` `def pt(self)`
- `Lepton.eta` (method) `higgs_four_lepton_analysis.py:173` `def eta(self)`
- `Lepton.phi` (method) `higgs_four_lepton_analysis.py:175` `def phi(self)`
- `Lepton.energy` (method) `higgs_four_lepton_analysis.py:177` `def energy(self)`
- `Lepton.mass` (method) `higgs_four_lepton_analysis.py:179` `def mass(self)`
- `Event.check_higgs` (method) `higgs_four_lepton_analysis.py:200` `def check_higgs(self, cfg)`
- `QuantumSpinorProcessor.__init__` (method) `higgs_four_lepton_analysis.py:213` `def __init__(self, config)`
- `QuantumSpinorProcessor.momentum_to_spinor_wavefunction` (method) `higgs_four_lepton_analysis.py:255` `def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)` -- Convert particle momentum to a 2-channel spatial wavefunction (2, G, G) suitable for processing by the DiracBackend.
- `QuantumSpinorProcessor.evolve_with_dirac_backend` (method) `higgs_four_lepton_analysis.py:296` `def evolve_with_dirac_backend(self, psi, steps)` -- Evolve a wavefunction using the DiracBackend neural network.
- `QuantumSpinorProcessor.evolve_with_schrodinger_backend` (method) `higgs_four_lepton_analysis.py:308` `def evolve_with_schrodinger_backend(self, psi, steps)` -- Evolve using the SchrodingerBackend neural network.
- `QuantumSpinorProcessor.evolve_with_hamiltonian_backend` (method) `higgs_four_lepton_analysis.py:315` `def evolve_with_hamiltonian_backend(self, psi, steps)` -- Evolve using the HamiltonianBackend neural network.
- `QuantumSpinorProcessor.compute_dirac_current` (method) `higgs_four_lepton_analysis.py:322` `def compute_dirac_current(self, px, py, pz, energy, mass, charge)` -- Compute the Dirac current using the actual neural network backends.
- `QuantumSpinorProcessor.compute_spinor_amplitude` (method) `higgs_four_lepton_analysis.py:370` `def compute_spinor_amplitude(self, psi)` -- Compute complex amplitude from wavefunction for helicity analysis.
- `EventParser.__init__` (method) `higgs_four_lepton_analysis.py:380` `def __init__(self, config)`
- `EventParser.parse_file` (method) `higgs_four_lepton_analysis.py:383` `def parse_file(self, filepath, event_type)`
- `Visualizer.__init__` (method) `higgs_four_lepton_analysis.py:439` `def __init__(self, config, quantum_processor)`
- `Visualizer.compute_quantum_helix` (method) `higgs_four_lepton_analysis.py:443` `def compute_quantum_helix(self, px, py, pz, charge, mass, energy)` -- Compute helical trajectory using actual DiracBackend evolution.
- `Visualizer.create_visualization` (method) `higgs_four_lepton_analysis.py:474` `def create_visualization(self, events, output_path)`
- `HiggsQuantumAnalysis.__init__` (method) `higgs_four_lepton_analysis.py:617` `def __init__(self, config)`
- `HiggsQuantumAnalysis.fetch_data` (method) `higgs_four_lepton_analysis.py:630` `def fetch_data(self)`
- `HiggsQuantumAnalysis.load_events` (method) `higgs_four_lepton_analysis.py:645` `def load_events(self)`
- `HiggsQuantumAnalysis.analyze_with_quantum_backends` (method) `higgs_four_lepton_analysis.py:672` `def analyze_with_quantum_backends(self)` -- Run analysis using actual quantum backends.
- `HiggsQuantumAnalysis.generate_visualization` (method) `higgs_four_lepton_analysis.py:728` `def generate_visualization(self)`
- `HiggsQuantumAnalysis.run` (method) `higgs_four_lepton_analysis.py:734` `def run(self)`
- `HiggsQuantumAnalysis.main` (method) `higgs_four_lepton_analysis.py:747` `def main()`

## higgs_quantum_analysis.py
Depends on: `quantum_computer.py`
- `FourMomentum.from_energy_momentum` (method) `higgs_quantum_analysis.py:151` `def from_energy_momentum(cls, E, px, py, pz)`
- `Lepton.pt` (method) `higgs_quantum_analysis.py:171` `def pt(self)`
- `Lepton.eta` (method) `higgs_quantum_analysis.py:173` `def eta(self)`
- `Lepton.phi` (method) `higgs_quantum_analysis.py:175` `def phi(self)`
- `Lepton.energy` (method) `higgs_quantum_analysis.py:177` `def energy(self)`
- `Lepton.mass` (method) `higgs_quantum_analysis.py:179` `def mass(self)`
- `Event.check_higgs` (method) `higgs_quantum_analysis.py:200` `def check_higgs(self, cfg)`
- `QuantumSpinorProcessor.__init__` (method) `higgs_quantum_analysis.py:213` `def __init__(self, config)`
- `QuantumSpinorProcessor.momentum_to_spinor_wavefunction` (method) `higgs_quantum_analysis.py:255` `def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)` -- Convert particle momentum to a 2-channel spatial wavefunction (2, G, G) suitable for processing by the DiracBackend.
- `QuantumSpinorProcessor.evolve_with_dirac_backend` (method) `higgs_quantum_analysis.py:296` `def evolve_with_dirac_backend(self, psi, steps)` -- Evolve a wavefunction using the DiracBackend neural network.
- `QuantumSpinorProcessor.evolve_with_schrodinger_backend` (method) `higgs_quantum_analysis.py:308` `def evolve_with_schrodinger_backend(self, psi, steps)` -- Evolve using the SchrodingerBackend neural network.
- `QuantumSpinorProcessor.evolve_with_hamiltonian_backend` (method) `higgs_quantum_analysis.py:315` `def evolve_with_hamiltonian_backend(self, psi, steps)` -- Evolve using the HamiltonianBackend neural network.
- `QuantumSpinorProcessor.compute_dirac_current` (method) `higgs_quantum_analysis.py:322` `def compute_dirac_current(self, px, py, pz, energy, mass, charge)` -- Compute the Dirac current using the actual neural network backends.
- `QuantumSpinorProcessor.compute_spinor_amplitude` (method) `higgs_quantum_analysis.py:370` `def compute_spinor_amplitude(self, psi)` -- Compute complex amplitude from wavefunction for helicity analysis.
- `EventParser.__init__` (method) `higgs_quantum_analysis.py:380` `def __init__(self, config)`
- `EventParser.parse_file` (method) `higgs_quantum_analysis.py:383` `def parse_file(self, filepath, event_type)`
- `Visualizer.__init__` (method) `higgs_quantum_analysis.py:439` `def __init__(self, config, quantum_processor)`
- `Visualizer.compute_quantum_helix` (method) `higgs_quantum_analysis.py:443` `def compute_quantum_helix(self, px, py, pz, charge, mass, energy)` -- Compute helical trajectory using actual DiracBackend evolution.
- `Visualizer.create_visualization` (method) `higgs_quantum_analysis.py:474` `def create_visualization(self, events, output_path)`
- `HiggsQuantumAnalysis.__init__` (method) `higgs_quantum_analysis.py:617` `def __init__(self, config)`
- `HiggsQuantumAnalysis.fetch_data` (method) `higgs_quantum_analysis.py:630` `def fetch_data(self)`
- `HiggsQuantumAnalysis.load_events` (method) `higgs_quantum_analysis.py:645` `def load_events(self)`
- `HiggsQuantumAnalysis.analyze_with_quantum_backends` (method) `higgs_quantum_analysis.py:672` `def analyze_with_quantum_backends(self)` -- Run analysis using actual quantum backends.
- `HiggsQuantumAnalysis.generate_visualization` (method) `higgs_quantum_analysis.py:728` `def generate_visualization(self)`
- `HiggsQuantumAnalysis.run` (method) `higgs_quantum_analysis.py:734` `def run(self)`
- `HiggsQuantumAnalysis.main` (method) `higgs_quantum_analysis.py:747` `def main()`

## molecular_sim.py
Depends on: `quantum_computer.py`
Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`
- `MoleculeData.build_jw_hamiltonian_of` (method) `molecular_sim.py:111` `def build_jw_hamiltonian_of(mol)` -- Construye Hamiltoniano JW usando OpenFermion correctamente.
- `ExactJWEnergy.__init__` (method) `molecular_sim.py:190` `def __init__(self, mol, n_qubits)`
- `ExactJWEnergy.__call__` (method) `molecular_sim.py:257` `def __call__(self, amps)`
- `SurrogateEnergy.__init__` (method) `molecular_sim.py:268` `def __init__(self, mol, n_qubits, exact_eval, backend)`
- `SurrogateEnergy.calibrate` (method) `molecular_sim.py:277` `def calibrate(self, hf_amps)`
- `SurrogateEnergy.cost_with_barrier` (method) `molecular_sim.py:290` `def cost_with_barrier(self, amps)`
- `SurrogateEnergy.prepare_hf` (method) `molecular_sim.py:304` `def prepare_hf(mol, factory, backend)`
- `SurrogateEnergy.uccsd` (method) `molecular_sim.py:321` `def uccsd(state, thetas, singles, doubles, backend, runner)` -- UCCSD ansatz.
- `VQESolver.__init__` (method) `molecular_sim.py:391` `def __init__(self, qc, config)`
- `VQESolver.run` (method) `molecular_sim.py:403` `def run(self, mol, backend, max_iter, tol)`
- `VQESolver.cost` (method) `molecular_sim.py:451` `def cost(thetas)`

## orbital_visualizer2.py
Imported by: `qc_dashboard.py`
- `WavefunctionCalculator.radial_wavefunction` (method) `orbital_visualizer2.py:87` `def radial_wavefunction(n, l, r)`
- `WavefunctionCalculator.spherical_harmonic_real` (method) `orbital_visualizer2.py:97` `def spherical_harmonic_real(l, m, theta, phi)`
- `WavefunctionCalculator.psi_on_grid` (method) `orbital_visualizer2.py:107` `def psi_on_grid(n, l, m, grid_size)`
- `HamiltonianNNProcessor.__init__` (method) `orbital_visualizer2.py:129` `def __init__(self, engine)`
- `HamiltonianNNProcessor.is_model_loaded` (method) `orbital_visualizer2.py:133` `def is_model_loaded(self)`
- `HamiltonianNNProcessor.compute_expected_energy` (method) `orbital_visualizer2.py:136` `def compute_expected_energy(self, n, l, m)`
- `MonteCarloSampler.__init__` (method) `orbital_visualizer2.py:163` `def __init__(self, hamiltonian_processor)`
- `MonteCarloSampler.find_max_probability` (method) `orbital_visualizer2.py:166` `def find_max_probability(self, n, l, m)`
- `MonteCarloSampler.sample` (method) `orbital_visualizer2.py:195` `def sample(self, n, l, m, num_samples)`
- `OrbitalVisualizer.visualize` (method) `orbital_visualizer2.py:268` `def visualize(self, data, save_path, hamiltonian_processor)`
- `OrbitalVisualizer.main` (method) `orbital_visualizer2.py:433` `def main()`

## polarizability_v3.py
Depends on: `molecular_sim.py`, `quantum_computer.py`
- `VQEResult.givens_single_excitation` (method) `polarizability_v3.py:73` `def givens_single_excitation(state, o, v, theta, n_qubits, backend)` -- Apply a particle-conserving single excitation rotation between qubits o (occupied) and v (virtual).
- `VQEResult.particle_conserving_ansatz` (method) `polarizability_v3.py:121` `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)` -- Particle-conserving UCCSD-like ansatz: - Singles: Givens rotations (correct particle conservation) - Doubles: reuse...
- `StarkEvaluator.__init__` (method) `polarizability_v3.py:164` `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- `StarkEvaluator.eval_dipole` (method) `polarizability_v3.py:196` `def eval_dipole(self, amps_raw)`
- `StarkEvaluator.__call__` (method) `polarizability_v3.py:209` `def __call__(self, amps)`
- `DipoleOperatorBuilder.__init__` (method) `polarizability_v3.py:216` `def __init__(self, bond_length_angstrom)`
- `PolarizabilityCalculator.__init__` (method) `polarizability_v3.py:250` `def __init__(self)`
- `PolarizabilityCalculator.cost` (method) `polarizability_v3.py:302` `def cost(th)`
- `PolarizabilityCalculator.run` (method) `polarizability_v3.py:315` `def run(self)`

## qc_dashboard.py
Depends on: `orbital_visualizer2.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_computer.py`, `quantum_dash.py`, `quantum_framework_core.py`, `quantum_framework_visualization.py`
Imported by: `test_qc_integration.py`
- `H2VQESolver.__init__` (method) `qc_dashboard.py:166` `def __init__(self, config)`
- `H2VQESolver.run_vqe` (method) `qc_dashboard.py:248` `def run_vqe(self, bond_length, max_iter)`
- `H2VQESolver.energy_landscape` (method) `qc_dashboard.py:293` `def energy_landscape(self)` -- Sweep bond length and return VQE energy curve.
- `H2VQESolver.orbital_wavefunction` (method) `qc_dashboard.py:318` `def orbital_wavefunction(bond_length, grid_points)` -- Compute hydrogen 1s orbital wavefunction along the internuclear axis.
- `VisualisationEngine.__init__` (method) `qc_dashboard.py:343` `def __init__(self, config)`
- `VisualisationEngine.available` (method) `qc_dashboard.py:362` `def available(self)`
- `VisualisationEngine.render_full_dashboard` (method) `qc_dashboard.py:365` `def render_full_dashboard(self, snapshots, current)`
- `VisualisationEngine.render_entropy_chart` (method) `qc_dashboard.py:403` `def render_entropy_chart(self, snapshots)`
- `VisualisationEngine.render_entanglement_profile` (method) `qc_dashboard.py:429` `def render_entanglement_profile(self, snapshots)` -- Entropy vs cut position for the latest snapshot.
- `VisualisationEngine.render_vqe_convergence` (method) `qc_dashboard.py:474` `def render_vqe_convergence(self, convergence, e_hf, e_fci)`
- `VisualisationEngine.render_energy_landscape` (method) `qc_dashboard.py:501` `def render_energy_landscape(self, bond_lengths, vqe_energies, hf_energies, fci_energies)`
- `VisualisationEngine.render_orbital_plot` (method) `qc_dashboard.py:533` `def render_orbital_plot(self, orbital_data)`
- `VisualisationEngine.render_entropy_scaling` (method) `qc_dashboard.py:562` `def render_entropy_scaling(self, data)`
- `VisualisationEngine.render_orbital_2d_projections` (method) `qc_dashboard.py:670` `def render_orbital_2d_projections(self, data)`
- `SimulatorBackend.__init__` (method) `qc_dashboard.py:728` `def __init__(self, config)`
- `SimulatorBackend.execute_circuit` (method) `qc_dashboard.py:775` `def execute_circuit(self, gates, n_qubits)`
- `Plotly3DEngine.__init__` (method) `qc_dashboard.py:982` `def __init__(self, config)`
- `Plotly3DEngine.available` (method) `qc_dashboard.py:998` `def available(self)`
- `Plotly3DEngine.render_bloch_3d` (method) `qc_dashboard.py:1001` `def render_bloch_3d(self, bloch_vectors)`
- `Plotly3DEngine.render_probability_3d` (method) `qc_dashboard.py:1049` `def render_probability_3d(self, probabilities, n_qubits)`
- `Plotly3DEngine.render_state_3d` (method) `qc_dashboard.py:1080` `def render_state_3d(self, probabilities, phases)` -- 3D scatter plot: X=real, Y=imaginary, Z=probability.
- `RealOrbitalEngine.__init__` (method) `qc_dashboard.py:1154` `def __init__(self)`
- `_FakeConfig.available` (method) `qc_dashboard.py:1210` `def available(self)`
- `_FakeConfig.entangled_available` (method) `qc_dashboard.py:1214` `def entangled_available(self)`
- `_FakeConfig.sample` (method) `qc_dashboard.py:1217` `def sample(self, n, l, m, num_samples)`
- `_FakeConfig.render_to_bytes` (method) `qc_dashboard.py:1222` `def render_to_bytes(self, data)`
- `_FakeConfig.sample_entangled` (method) `qc_dashboard.py:1240` `def sample_entangled(self, n1, l1, m1, n2, l2, m2, num_samples)`
- `_FakeConfig.render_entangled_to_bytes` (method) `qc_dashboard.py:1250` `def render_entangled_to_bytes(self, data)`
- `BrutalVizEngine.__init__` (method) `qc_dashboard.py:1272` `def __init__(self)`
- `BrutalVizEngine.dash_available` (method) `qc_dashboard.py:1297` `def dash_available(self)`
- `BrutalVizEngine.hologram_available` (method) `qc_dashboard.py:1301` `def hologram_available(self)`
- `BrutalVizEngine.run_brutal_viz` (method) `qc_dashboard.py:1304` `def run_brutal_viz(self, circuit_name)`
- `BrutalVizEngine.render_hologram` (method) `qc_dashboard.py:1327` `def render_hologram(self, snapshots, backend_comp)`
- `BackendComparator.__init__` (method) `qc_dashboard.py:1347` `def __init__(self, config)`
- `BackendComparator.run_comparison` (method) `qc_dashboard.py:1351` `def run_comparison(self, gates, n_qubits)`
- `DashboardApp.__init__` (method) `qc_dashboard.py:1374` `def __init__(self, config)`
- `DashboardApp.run` (method) `qc_dashboard.py:1384` `def run(self)`
- `DashboardApp.main` (method) `qc_dashboard.py:1994` `def main()` -- Launch the Streamlit dashboard.

## qc_integration.py
Depends on: `quantum_computer.py`, `quantum_framework_core.py`
Imported by: `qc_dashboard.py`, `test_qc_integration.py`
- `GateInstruction.num_qubits` (method) `qc_integration.py:141` `def num_qubits(self)`
- `CircuitIR.append` (method) `qc_integration.py:156` `def append(self, gate)`
- `IQCAdapter.export` (method) `qc_integration.py:185` `def export(self, circuit)` -- Export a CircuitIR to the target format.
- `IQCAdapter.import_` (method) `qc_integration.py:189` `def import_(self, data)` -- Import from the target format into a CircuitIR.
- `OpenQasmAdapter.__init__` (method) `qc_integration.py:238` `def __init__(self, config)`
- `OpenQasmAdapter.export` (method) `qc_integration.py:243` `def export(self, circuit)`
- `OpenQasmAdapter.import_` (method) `qc_integration.py:277` `def import_(self, data)` -- Parse an OpenQASM 2.0 string into a CircuitIR.
- `QiskitAdapter.__init__` (method) `qc_integration.py:361` `def __init__(self, config)`
- `QiskitAdapter.export` (method) `qc_integration.py:364` `def export(self, circuit)`
- `QiskitAdapter.import_` (method) `qc_integration.py:379` `def import_(self, data)`
- `PennyLaneAdapter.__init__` (method) `qc_integration.py:458` `def __init__(self, config)`
- `PennyLaneAdapter.export` (method) `qc_integration.py:461` `def export(self, circuit)`
- `PennyLaneAdapter.circuit_fn` (method) `qc_integration.py:467` `def circuit_fn()`
- `PennyLaneAdapter.import_` (method) `qc_integration.py:483` `def import_(self, data)`
- `FrameworkAdapter.__init__` (method) `qc_integration.py:565` `def __init__(self, config)`
- `FrameworkAdapter.to_circuit_ir` (method) `qc_integration.py:568` `def to_circuit_ir(self, circuit, framework_type)` -- Extract a CircuitIR from a framework circuit object.
- `FrameworkAdapter.from_circuit_ir` (method) `qc_integration.py:584` `def from_circuit_ir(self, cir, framework_type)` -- Build a framework circuit object from a CircuitIR.
- `StandardCircuitFactory.bell_state` (method) `qc_integration.py:663` `def bell_state()`
- `StandardCircuitFactory.ghz_state` (method) `qc_integration.py:670` `def ghz_state(n_qubits)`
- `StandardCircuitFactory.qft` (method) `qc_integration.py:678` `def qft(n_qubits)`
- `StandardCircuitFactory.w_state` (method) `qc_integration.py:693` `def w_state(n_qubits)`
- `StandardCircuitFactory.grover` (method) `qc_integration.py:701` `def grover(n_qubits, marked, iterations)`
- `IntegrationBridge.__init__` (method) `qc_integration.py:753` `def __init__(self, config)`
- `IntegrationBridge.export_qasm` (method) `qc_integration.py:780` `def export_qasm(self, circuit)`
- `IntegrationBridge.import_qasm` (method) `qc_integration.py:783` `def import_qasm(self, qasm_str)`
- `IntegrationBridge.to_qiskit` (method) `qc_integration.py:788` `def to_qiskit(self, circuit)`
- `IntegrationBridge.from_qiskit` (method) `qc_integration.py:793` `def from_qiskit(self, qiskit_circuit)`
- `IntegrationBridge.to_pennylane` (method) `qc_integration.py:800` `def to_pennylane(self, circuit)`
- `IntegrationBridge.from_pennylane` (method) `qc_integration.py:805` `def from_pennylane(self, pennylane_data)`
- `IntegrationBridge.to_circuit_ir` (method) `qc_integration.py:812` `def to_circuit_ir(self, circuit, framework_type)`
- `IntegrationBridge.from_circuit_ir` (method) `qc_integration.py:819` `def from_circuit_ir(self, cir, framework_type)`
- `IntegrationBridge.export_qasm_from_framework` (method) `qc_integration.py:828` `def export_qasm_from_framework(self, circuit, framework_type)`
- `IntegrationBridge.import_qasm_to_framework` (method) `qc_integration.py:837` `def import_qasm_to_framework(self, qasm_str, framework_type)`
- `IntegrationBridge.main` (method) `qc_integration.py:851` `def main()` -- Command-line interface for the integration bridge.

## quantum_3dview.py
Depends on: `quantum_computer.py`, `quantum_visualizer.py`
Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`
- `BrutalConfig.colors` (method) `quantum_3dview.py:52` `def colors(self)`
- `QuantumHologram.__init__` (method) `quantum_3dview.py:87` `def __init__(self, config)`
- `QuantumHologram.create_amplitude_hologram` (method) `quantum_3dview.py:92` `def create_amplitude_hologram(self, snapshots, backend_comparison)` -- Crea visualización holográfica de amplitudes en 3D con efecto de partículas
- `QuantumHologram.hex_to_rgb` (method) `quantum_3dview.py:391` `def hex_to_rgb(hex_color)`
- `QuantumHologram.rgb_to_hex` (method) `quantum_3dview.py:395` `def rgb_to_hex(rgb)`
- `QuantumNeuralTopology.__init__` (method) `quantum_3dview.py:406` `def __init__(self, model)`
- `QuantumNeuralTopology.create_topology_map` (method) `quantum_3dview.py:410` `def create_topology_map(self)` -- Crea mapa 3D de la arquitectura neuronal con activaciones en tiempo real
- `QuantumSonification.__init__` (method) `quantum_3dview.py:503` `def __init__(self, sample_rate)`
- `QuantumSonification.state_to_audio` (method) `quantum_3dview.py:506` `def state_to_audio(self, snapshot, duration)` -- Convierte un estado cuántico en onda de audio - Amplitudes controlan volumen - Fases controlan paneo estéreo...
- `BrutalDashboard.__init__` (method) `quantum_3dview.py:563` `def __init__(self, config)`
- `BrutalDashboard.generate_full_report` (method) `quantum_3dview.py:570` `def generate_full_report(self, snapshots, backend_comparison)` -- Genera reporte completo con múltiples visualizaciones
- `BrutalDashboard.demo_brutal` (method) `quantum_3dview.py:614` `def demo_brutal()` -- Demostración de visualización brutal
- `BrutalDashboard.create_synthetic_snapshots` (method) `quantum_3dview.py:653` `def create_synthetic_snapshots()` -- Crea datos sintéticos para demostración

## quantum_computer.py
Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `higgs_four_lepton_analysis.py`, `higgs_quantum_analysis.py`, `molecular_sim.py`, `polarizability_v3.py`, `qc_dashboard.py`, `qc_integration.py`, `quantum_3dview.py`, `quantum_dash.py`, `quantum_visualizer.py`, `topological_hilbert_compression2.py`
- `SpectralLayer.__init__` (method) `quantum_computer.py:113` `def __init__(self, channels, grid_size)`
- `SpectralLayer.forward` (method) `quantum_computer.py:124` `def forward(self, x)` -- Apply spectral convolution via RFFT2.
- `HamiltonianBackboneNet.__init__` (method) `quantum_computer.py:150` `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `HamiltonianBackboneNet.forward` (method) `quantum_computer.py:159` `def forward(self, x)` -- Accepts (G,G), (1,G,G), or (B,1,G,G).
- `SchrodingerSpectralNet.__init__` (method) `quantum_computer.py:176` `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `SchrodingerSpectralNet.forward` (method) `quantum_computer.py:188` `def forward(self, x)` -- (2,G,G) or (B,2,G,G) -> same shape.
- `DiracSpectralNet.__init__` (method) `quantum_computer.py:205` `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `DiracSpectralNet.forward` (method) `quantum_computer.py:217` `def forward(self, x)` -- (8,G,G) or (B,8,G,G) -> same shape.
- `GammaMatrices.__init__` (method) `quantum_computer.py:236` `def __init__(self, representation, device)`
- `GammaMatrices.to` (method) `quantum_computer.py:272` `def to(self, device)` -- Move all matrices to device.
- `JointHilbertState.__init__` (method) `quantum_computer.py:305` `def __init__(self, amplitudes, n_qubits)`
- `JointHilbertState.normalize_` (method) `quantum_computer.py:317` `def normalize_(self)` -- In-place normalization: sum_k P(k) = 1.
- `JointHilbertState.probabilities` (method) `quantum_computer.py:323` `def probabilities(self)` -- Return (2^n,) tensor of Born probabilities P(k) for each basis state.
- `JointHilbertState.marginal_probability_one` (method) `quantum_computer.py:328` `def marginal_probability_one(self, qubit)` -- Marginal Born probability P(qubit_j = |1>).
- `JointHilbertState.most_probable_basis_state` (method) `quantum_computer.py:343` `def most_probable_basis_state(self)` -- Return the index k with the highest probability.
- `JointHilbertState.bloch_vector` (method) `quantum_computer.py:347` `def bloch_vector(self, qubit)` -- Compute the reduced Bloch vector for qubit j by partial trace.
- `JointHilbertState.clone` (method) `quantum_computer.py:385` `def clone(self)` -- Return a deep copy.
- `PotentialGenerator.__init__` (method) `quantum_computer.py:393` `def __init__(self, config)`
- `PotentialGenerator.harmonic` (method) `quantum_computer.py:402` `def harmonic(self)`
- `PotentialGenerator.double_well` (method) `quantum_computer.py:410` `def double_well(self)` -- Double-well along x.
- `PotentialGenerator.coulomb` (method) `quantum_computer.py:417` `def coulomb(self)` -- Coulomb-like V ~ -1/r.
- `PotentialGenerator.periodic_lattice` (method) `quantum_computer.py:424` `def periodic_lattice(self)` -- Periodic cosine lattice.
- `PotentialGenerator.mixed` (method) `quantum_computer.py:429` `def mixed(self, seed)` -- Dirichlet-weighted mixture of all four potentials.
- `JointStateFactory.__init__` (method) `quantum_computer.py:477` `def __init__(self, config)`
- `JointStateFactory.all_zeros` (method) `quantum_computer.py:484` `def all_zeros(self, n_qubits)` -- Initialize register in |00...0>.
- `JointStateFactory.basis_state` (method) `quantum_computer.py:492` `def basis_state(self, n_qubits, k)` -- Initialize register in computational basis state |k>.
- `JointStateFactory.from_bitstring` (method) `quantum_computer.py:502` `def from_bitstring(self, bitstring)` -- Initialize in the basis state given by binary string.
- `IPhysicsBackend.evolve_amplitude` (method) `quantum_computer.py:515` `def evolve_amplitude(self, amp, dt)` -- Evolve a single (2, G, G) wavefunction by dt under H.
- `IPhysicsBackend.apply_phase` (method) `quantum_computer.py:519` `def apply_phase(self, amp, phase_angle)` -- Apply global phase e^{i*phi} to a (2, G, G) amplitude.
- `HamiltonianBackend.__init__` (method) `quantum_computer.py:531` `def __init__(self, config)`
- `HamiltonianBackend.evolve_amplitude` (method) `quantum_computer.py:573` `def evolve_amplitude(self, amp, dt)` -- dpsi/dt = -i H psi  =>  psi' = psi + dt * (-i H psi) = psi + dt*(H_i*r - H_r*i).
- `HamiltonianBackend.apply_phase` (method) `quantum_computer.py:585` `def apply_phase(self, amp, phase_angle)`
- `SchrodingerBackend.__init__` (method) `quantum_computer.py:599` `def __init__(self, config, hamiltonian)`
- `SchrodingerBackend.evolve_amplitude` (method) `quantum_computer.py:628` `def evolve_amplitude(self, amp, dt)`
- `SchrodingerBackend.apply_phase` (method) `quantum_computer.py:636` `def apply_phase(self, amp, phase_angle)`
- `DiracBackend.__init__` (method) `quantum_computer.py:648` `def __init__(self, config, hamiltonian)`
- `DiracBackend.evolve_amplitude` (method) `quantum_computer.py:722` `def evolve_amplitude(self, amp, dt)`
- `DiracBackend.apply_phase` (method) `quantum_computer.py:735` `def apply_phase(self, amp, phase_angle)`
- `IQuantumGate.name` (method) `quantum_computer.py:849` `def name(self)` -- Gate identifier.
- `IQuantumGate.apply` (method) `quantum_computer.py:853` `def apply(self, state, backend, targets, params)` -- Apply gate to joint state, return new joint state.
- `HadamardGate.name` (method) `quantum_computer.py:867` `def name(self)`
- `HadamardGate.apply` (method) `quantum_computer.py:870` `def apply(self, state, backend, targets, params)`
- `PauliXGate.name` (method) `quantum_computer.py:882` `def name(self)`
- `PauliXGate.apply` (method) `quantum_computer.py:885` `def apply(self, state, backend, targets, params)`
- `PauliYGate.name` (method) `quantum_computer.py:896` `def name(self)`
- `PauliYGate.apply` (method) `quantum_computer.py:899` `def apply(self, state, backend, targets, params)`
- `PauliZGate.name` (method) `quantum_computer.py:910` `def name(self)`
- `PauliZGate.apply` (method) `quantum_computer.py:913` `def apply(self, state, backend, targets, params)`
- `SGate.name` (method) `quantum_computer.py:924` `def name(self)`
- `SGate.apply` (method) `quantum_computer.py:927` `def apply(self, state, backend, targets, params)`
- `TGate.name` (method) `quantum_computer.py:938` `def name(self)`
- `TGate.apply` (method) `quantum_computer.py:941` `def apply(self, state, backend, targets, params)`
- `RxGate.name` (method) `quantum_computer.py:953` `def name(self)`
- `RxGate.apply` (method) `quantum_computer.py:956` `def apply(self, state, backend, targets, params)`
- `RyGate.name` (method) `quantum_computer.py:969` `def name(self)`
- `RyGate.apply` (method) `quantum_computer.py:972` `def apply(self, state, backend, targets, params)`
- `RzGate.name` (method) `quantum_computer.py:985` `def name(self)`
- `RzGate.apply` (method) `quantum_computer.py:988` `def apply(self, state, backend, targets, params)`
- `CNOTGate.name` (method) `quantum_computer.py:1007` `def name(self)`
- `CNOTGate.apply` (method) `quantum_computer.py:1010` `def apply(self, state, backend, targets, params)`
- `CZGate.name` (method) `quantum_computer.py:1030` `def name(self)`
- `CZGate.apply` (method) `quantum_computer.py:1033` `def apply(self, state, backend, targets, params)`
- `SWAPGate.name` (method) `quantum_computer.py:1053` `def name(self)`
- `SWAPGate.apply` (method) `quantum_computer.py:1056` `def apply(self, state, backend, targets, params)`
- `ToffoliGate.name` (method) `quantum_computer.py:1076` `def name(self)`
- `ToffoliGate.apply` (method) `quantum_computer.py:1079` `def apply(self, state, backend, targets, params)`
- `MCZGate.name` (method) `quantum_computer.py:1115` `def name(self)`
- `MCZGate.apply` (method) `quantum_computer.py:1118` `def apply(self, state, backend, targets, params)`
- `EvolveGate.name` (method) `quantum_computer.py:1139` `def name(self)`
- `EvolveGate.apply` (method) `quantum_computer.py:1142` `def apply(self, state, backend, targets, params)`
- `EvolveGate.register_gate` (method) `quantum_computer.py:1179` `def register_gate(name, gate)` -- Register a custom gate without modifying existing code (Open/Closed Principle).
- `QuantumCircuit.__init__` (method) `quantum_computer.py:1200` `def __init__(self, n_qubits)`
- `QuantumCircuit.h` (method) `quantum_computer.py:1206` `def h(self, q)`
- `QuantumCircuit.x` (method) `quantum_computer.py:1209` `def x(self, q)`
- `QuantumCircuit.y` (method) `quantum_computer.py:1212` `def y(self, q)`
- `QuantumCircuit.z` (method) `quantum_computer.py:1215` `def z(self, q)`
- `QuantumCircuit.s` (method) `quantum_computer.py:1218` `def s(self, q)`
- `QuantumCircuit.t` (method) `quantum_computer.py:1221` `def t(self, q)`
- `QuantumCircuit.rx` (method) `quantum_computer.py:1224` `def rx(self, q, theta)`
- `QuantumCircuit.ry` (method) `quantum_computer.py:1227` `def ry(self, q, theta)`
- `QuantumCircuit.rz` (method) `quantum_computer.py:1230` `def rz(self, q, theta)`
- `QuantumCircuit.cnot` (method) `quantum_computer.py:1233` `def cnot(self, ctrl, tgt)`
- `QuantumCircuit.cx` (method) `quantum_computer.py:1236` `def cx(self, ctrl, tgt)`
- `QuantumCircuit.cz` (method) `quantum_computer.py:1239` `def cz(self, ctrl, tgt)`
- `QuantumCircuit.swap` (method) `quantum_computer.py:1242` `def swap(self, a, b)`
- `QuantumCircuit.toffoli` (method) `quantum_computer.py:1245` `def toffoli(self, c0, c1, tgt)`
- `QuantumCircuit.ccx` (method) `quantum_computer.py:1248` `def ccx(self, c0, c1, tgt)`
- `QuantumCircuit.evolve` (method) `quantum_computer.py:1251` `def evolve(self, qubits, dt, steps)`
- `QuantumCircuit.barrier` (method) `quantum_computer.py:1254` `def barrier(self)`
- `QuantumCircuit.depth` (method) `quantum_computer.py:1265` `def depth(self)`
- `MeasurementResult.probabilities` (method) `quantum_computer.py:1293` `def probabilities(self)` -- Alias: marginal P(|1>) per qubit index.
- `MeasurementResult.most_probable_bitstring` (method) `quantum_computer.py:1297` `def most_probable_bitstring(self)` -- Return the bitstring with the highest probability.
- `MeasurementResult.expectation_z` (method) `quantum_computer.py:1301` `def expectation_z(self, qubit)`
- `MeasurementResult.entropy` (method) `quantum_computer.py:1305` `def entropy(self)` -- Shannon entropy of the full probability distribution in bits.
- `QuantumComputer.__init__` (method) `quantum_computer.py:1361` `def __init__(self, config)`
- `QuantumComputer.run` (method) `quantum_computer.py:1388` `def run(self, circuit, backend, initial_states)` -- Execute a quantum circuit on the joint Hilbert space.
- `QuantumComputer.run_with_state_snapshots` (method) `quantum_computer.py:1420` `def run_with_state_snapshots(self, circuit, backend, snapshot_after)` -- Execute circuit with non-destructive probability snapshots.
- `QuantumComputer.bell_state` (method) `quantum_computer.py:1449` `def bell_state(self, backend)` -- |Phi+> = (|00> + |11>) / sqrt(2).
- `QuantumComputer.ghz_state` (method) `quantum_computer.py:1459` `def ghz_state(self, n_qubits, backend)` -- (|00...0> + |11...1>) / sqrt(2).
- `QuantumComputer.quantum_fourier_transform` (method) `quantum_computer.py:1471` `def quantum_fourier_transform(self, n_qubits, backend)` -- QFT on |00...0>.
- `QuantumComputer.grover_oracle_search` (method) `quantum_computer.py:1481` `def grover_oracle_search(self, n_qubits, target_bitstring, backend, n_iterations)` -- Grover's search algorithm with correct phase oracle and diffusion operator.
- `QuantumComputer.variational_ansatz` (method) `quantum_computer.py:1559` `def variational_ansatz(self, n_qubits, n_layers, thetas, backend)` -- Hardware-efficient ansatz: Ry layers + CNOT chain. len(thetas)=n_qubits*n_layers.
- `QuantumComputer.teleportation` (method) `quantum_computer.py:1572` `def teleportation(self, backend)` -- 3-qubit teleportation protocol.
- `QuantumComputer.deutsch_jozsa` (method) `quantum_computer.py:1585` `def deutsch_jozsa(self, n_input_qubits, is_constant, backend)` -- Deutsch-Jozsa: constant -> all inputs |0>, balanced -> at least one |1>.
- `QuantumComputer.run_phase_tests` (method) `quantum_computer.py:1609` `def run_phase_tests(config)` -- Property-based test suite for quantum phase coherence and unitarity.

## quantum_dash.py
Depends on: `molecular_sim.py`, `quantum_computer.py`
Imported by: `qc_dashboard.py`, `quantum_framework_menu.py`
- `BrutalistConfig.colors` (method) `quantum_dash.py:168` `def colors(self)`
- `BrutalistConfig.plotly_template` (method) `quantum_dash.py:234` `def plotly_template(self)`
- `IVisualComponent.render` (method) `quantum_dash.py:277` `def render(self, data, axes, config)`
- `ProbabilityVisualizer.render` (method) `quantum_dash.py:282` `def render(self, snapshot, axes, config)`
- `BlochSphereVisualizer.render` (method) `quantum_dash.py:341` `def render(self, snapshot, axes, config)`
- `PhaseSpaceVisualizer.render` (method) `quantum_dash.py:448` `def render(self, snapshot, axes, config)`
- `EntropyVisualizer.render` (method) `quantum_dash.py:510` `def render(self, snapshots, axes, config)`
- `BackendComparisonVisualizer.render` (method) `quantum_dash.py:584` `def render(self, comparisons, axes, config)`
- `FidelityVisualizer.render` (method) `quantum_dash.py:634` `def render(self, comparisons, axes, config)`
- `QuantumStateAnalyzer.__init__` (method) `quantum_dash.py:675` `def __init__(self, config)`
- `QuantumStateAnalyzer.compute_probabilities` (method) `quantum_dash.py:678` `def compute_probabilities(self, state)`
- `QuantumStateAnalyzer.compute_phases` (method) `quantum_dash.py:684` `def compute_phases(self, state)`
- `QuantumStateAnalyzer.compute_entropy` (method) `quantum_dash.py:696` `def compute_entropy(self, probs)`
- `QuantumStateAnalyzer.compute_bloch_vectors` (method) `quantum_dash.py:704` `def compute_bloch_vectors(self, state)`
- `QuantumStateAnalyzer.create_snapshot` (method) `quantum_dash.py:713` `def create_snapshot(self, state, step, gate_name, backend_name)`
- `StandardCircuits.bell_state` (method) `quantum_dash.py:756` `def bell_state()`
- `StandardCircuits.ghz_state` (method) `quantum_dash.py:760` `def ghz_state(n_qubits)`
- `StandardCircuits.qft` (method) `quantum_dash.py:767` `def qft(n_qubits)`
- `StandardCircuits.grover_oracle` (method) `quantum_dash.py:782` `def grover_oracle(n_qubits, marked)`
- `StandardCircuits.grover_diffusion` (method) `quantum_dash.py:794` `def grover_diffusion(n_qubits)`
- `CircuitExecutor.__init__` (method) `quantum_dash.py:807` `def __init__(self, qc, config)`
- `CircuitExecutor.execute_sequence` (method) `quantum_dash.py:812` `def execute_sequence(self, gates, n_qubits, backend_name)`
- `CircuitExecutor.compare_backends` (method) `quantum_dash.py:835` `def compare_backends(self, gates, n_qubits)`
- `FigureBuilder.__init__` (method) `quantum_dash.py:874` `def __init__(self, config)`
- `FigureBuilder.build_full_figure` (method) `quantum_dash.py:883` `def build_full_figure(self, snapshots, comparisons)`
- `FigureBuilder.build_summary_figure` (method) `quantum_dash.py:923` `def build_summary_figure(self, snapshots, comparisons)`
- `QuantumVisualizer.__init__` (method) `quantum_dash.py:959` `def __init__(self, config)`
- `QuantumVisualizer.visualize_bell_state` (method) `quantum_dash.py:1010` `def visualize_bell_state(self)`
- `QuantumVisualizer.visualize_ghz_state` (method) `quantum_dash.py:1016` `def visualize_ghz_state(self, n_qubits)`
- `QuantumVisualizer.visualize_qft` (method) `quantum_dash.py:1022` `def visualize_qft(self, n_qubits)`
- `QuantumVisualizer.visualize_grover` (method) `quantum_dash.py:1028` `def visualize_grover(self, n_qubits, marked_state)`
- `QuantumVisualizer.run_all` (method) `quantum_dash.py:1174` `def run_all(self)`
- `QuantumVisualizer.main` (method) `quantum_dash.py:1199` `def main()`


Next: [API_p2.md](API_p2.md)
