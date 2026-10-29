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
- `main` (function) `advanced_experiments.py:1304` `def main()` -- Main entry point.

## app.py
Depends on: `molecular_sim.py`, `quantum_computer.py`
Imported by: `quantum_framework_menu.py`
- `givens_single_excitation` (function) `app.py:61` `def givens_single_excitation(state, o, v, theta, n_qubits, backend)` -- Apply a particle-conserving single excitation rotation between qubits o (occupied) and v (virtual).
- `particle_conserving_ansatz` (function) `app.py:109` `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)` -- Particle-conserving UCCSD-like ansatz: - Singles: Givens rotations (correct particle conservation) - Doubles: reuse...
- `StarkEvaluator.__init__` (method) `app.py:152` `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- `StarkEvaluator.eval_dipole` (method) `app.py:184` `def eval_dipole(self, amps_raw)`
- `StarkEvaluator.__call__` (method) `app.py:197` `def __call__(self, amps)`
- `DipoleOperatorBuilder.__init__` (method) `app.py:204` `def __init__(self, bond_length_angstrom)`
- `PolarizabilityCalculator.__init__` (method) `app.py:238` `def __init__(self)`
- `cost` (function) `app.py:290` `def cost(th)`
- `PolarizabilityCalculator.run` (method) `app.py:303` `def run(self)`

## demo_molecular_vqe.py
Depends on: `quantum_framework_molecular_v2.py`
- `print_header` (function) `demo_molecular_vqe.py:47` `def print_header(title)` -- Print formatted header.
- `check_dependencies` (function) `demo_molecular_vqe.py:54` `def check_dependencies()` -- Check and report available dependencies.
- `demo_h2_direct` (function) `demo_molecular_vqe.py:78` `def demo_h2_direct()` -- Demo: H2 with direct statevector (precision mode).
- `demo_h2_mps` (function) `demo_molecular_vqe.py:104` `def demo_h2_mps()` -- Demo: H2 with precision_mode=False (NOTE: solver is statevector; see module docs).
- `demo_comparison` (function) `demo_molecular_vqe.py:132` `def demo_comparison()` -- Demo: Compare direct vs MPS.
- `demo_molecule_builder` (function) `demo_molecular_vqe.py:168` `def demo_molecule_builder()` -- Demo: OpenFermion molecule builder.
- `demo_smart_initialization` (function) `demo_molecular_vqe.py:198` `def demo_smart_initialization()` -- Demo: Smart parameter initialization.
- `demo_cached_operations` (function) `demo_molecular_vqe.py:232` `def demo_cached_operations()` -- Demo: Cached Pauli operations.
- `demo_config_from_toml` (function) `demo_molecular_vqe.py:278` `def demo_config_from_toml()` -- Demo: Configuration from TOML.
- `main` (function) `demo_molecular_vqe.py:303` `def main()` -- Run all demos.

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
- `main` (function) `entangled_hydrogen.py:893` `def main()` -- Main entry point for entangled hydrogen visualization.

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
- `main` (function) `higgs_four_lepton_analysis.py:747` `def main()`

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
- `main` (function) `higgs_quantum_analysis.py:747` `def main()`

## immersive/app.py
Depends on: `immersive/backend.py`, `immersive/glkit.py`, `immersive/hud.py`, `immersive/postfx.py`, `immersive/scenes.py`, `immersive/shaders.py`
Imported by: `qc_immersive.py`, `test_immersive.py`
- `build_tour` (function) `immersive/app.py:71` `def build_tour()`
- `cmd` (function) `immersive/app.py:72` `def cmd()`
- `spin` (function) `immersive/app.py:75` `def spin(speed)`
- `App.__init__` (method) `immersive/app.py:93` `def __init__(self, args)`
- `App.go` (method) `immersive/app.py:147` `def go(self, name, instant)`
- `App.run_command` (method) `immersive/app.py:157` `def run_command(self, cmd)`
- `App.start_tour` (method) `immersive/app.py:169` `def start_tour(self)`
- `App.stop_tour` (method) `immersive/app.py:176` `def stop_tour(self)`
- `App.advance_tour` (method) `immersive/app.py:182` `def advance_tour(self)`
- `App.update_tour` (method) `immersive/app.py:197` `def update_tour(self, dt)`
- `App.screenshot` (method) `immersive/app.py:214` `def screenshot(self, path)`
- `App.handle_event` (method) `immersive/app.py:224` `def handle_event(self, ev)`
- `App.update` (method) `immersive/app.py:264` `def update(self, dt)`
- `App.draw_chrome` (method) `immersive/app.py:286` `def draw_chrome(self)`
- `App.frame` (method) `immersive/app.py:329` `def frame(self, dt)`
- `App.run` (method) `immersive/app.py:344` `def run(self)`
- `parse_args` (function) `immersive/app.py:369` `def parse_args(argv)`
- `main` (function) `immersive/app.py:382` `def main(argv)`

## immersive/backend.py
Depends on: `immersive/physics.py`, `quantum_computer.py`, `quantum_framework_core.py`, `quantum_framework_molecular.py`, `quantum_lab.py`
Imported by: `immersive/app.py`, `immersive/scenes.py`, `immersive/spin.py`, `test_immersive.py`
- `BackendHub.shared` (method) `immersive/backend.py:39` `def shared(cls)`
- `BackendHub.preload` (method) `immersive/backend.py:46` `def preload(cls)`
- `BackendHub.__init__` (method) `immersive/backend.py:49` `def __init__(self)`
- `BackendHub.backend` (method) `immersive/backend.py:67` `def backend(self, name)`
- `BackendHub.new_state` (method) `immersive/backend.py:70` `def new_state(self, n_qubits)`
- `to_complex` (function) `immersive/backend.py:80` `def to_complex(state)`
- `BackendRegister.__init__` (method) `immersive/backend.py:88` `def __init__(self, n_qubits, backend, seconds_per_gate)`
- `BackendRegister.be` (method) `immersive/backend.py:107` `def be(self)`
- `BackendRegister.cycle_backend` (method) `immersive/backend.py:110` `def cycle_backend(self)`
- `BackendRegister.reset` (method) `immersive/backend.py:115` `def reset(self)`
- `BackendRegister.prepare` (method) `immersive/backend.py:124` `def prepare(self, kind)`
- `BackendRegister.single` (method) `immersive/backend.py:134` `def single(self, name, qubit)`
- `BackendRegister.cnot` (method) `immersive/backend.py:137` `def cnot(self, control, target)`
- `BackendRegister.queue` (method) `immersive/backend.py:140` `def queue(self, spec)`
- `BackendRegister.busy` (method) `immersive/backend.py:148` `def busy(self)`
- `BackendRegister.progress` (method) `immersive/backend.py:152` `def progress(self)`
- `BackendRegister.active_name` (method) `immersive/backend.py:156` `def active_name(self)`
- `BackendRegister.update` (method) `immersive/backend.py:177` `def update(self, dt)`
- `BackendRegister.evolve` (method) `immersive/backend.py:197` `def evolve(self, steps, dt)` -- Advance every spatial amplitude field with the selected physics backend (EvolveGate).
- `BackendRegister.reduced_density` (method) `immersive/backend.py:209` `def reduced_density(self, qubit)`
- `BackendRegister.bloch` (method) `immersive/backend.py:216` `def bloch(self, qubit)`
- `BackendRegister.backend_bloch` (method) `immersive/backend.py:220` `def backend_bloch(self, qubit)`
- `BackendRegister.entropy` (method) `immersive/backend.py:223` `def entropy(self, qubit)`
- `BackendRegister.probabilities` (method) `immersive/backend.py:227` `def probabilities(self)`
- `BackendRegister.psi` (method) `immersive/backend.py:231` `def psi(self)`
- `BackendRegister.fields` (method) `immersive/backend.py:235` `def fields(self)` -- Complex amplitude fields, shape (2^n, G, G).
- `BackendRegister.deviation` (method) `immersive/backend.py:239` `def deviation(self)` -- Largest gap between the project's Bloch vectors and the ideal-unitary reference.
- `BackendRegister.measure` (method) `immersive/backend.py:243` `def measure(self, shots, rng)`
- `BackendRegister.describe` (method) `immersive/backend.py:251` `def describe(self)`
- `run_project_vqe` (function) `immersive/backend.py:265` `def run_project_vqe()` -- Run the project's own live H2 VQE (MPSQuantumComputer + Jordan-Wigner Hamiltonian, 2-qubit STO-3G model).

## immersive/glkit.py
Imported by: `immersive/app.py`, `immersive/hud.py`, `immersive/postfx.py`, `immersive/scenes.py`
- `Program.__init__` (method) `immersive/glkit.py:35` `def __init__(self, fragment_source, label, vertex_source)`
- `Program.use` (method) `immersive/glkit.py:49` `def use(self)`
- `Program.loc` (method) `immersive/glkit.py:52` `def loc(self, name)`
- `Program.f1` (method) `immersive/glkit.py:57` `def f1(self, name, v)`
- `Program.i1` (method) `immersive/glkit.py:60` `def i1(self, name, v)`
- `Program.v2` (method) `immersive/glkit.py:63` `def v2(self, name, v)`
- `Program.v3` (method) `immersive/glkit.py:66` `def v3(self, name, v)`
- `Program.v2a` (method) `immersive/glkit.py:69` `def v2a(self, name, arr)`
- `Program.v3a` (method) `immersive/glkit.py:73` `def v3a(self, name, arr)`
- `Program.v4ia` (method) `immersive/glkit.py:77` `def v4ia(self, name, arr)`
- `Program.v3ia` (method) `immersive/glkit.py:81` `def v3ia(self, name, arr)`
- `FullscreenPass.__init__` (method) `immersive/glkit.py:89` `def __init__(self)`
- `FullscreenPass.draw` (method) `immersive/glkit.py:92` `def draw(self)`
- `make_texture` (function) `immersive/glkit.py:97` `def make_texture(width, height, internal, fmt, dtype, data, linear)`
- `upload_radial` (function) `immersive/glkit.py:110` `def upload_radial(tex, table)`
- `upload_volume` (function) `immersive/glkit.py:117` `def upload_volume(tex, volume)` -- Upload a (x, y, z) float volume as a linear R32F 3D texture (texture axes s,t,r = x,y,z).
- `OrbitCamera.__init__` (method) `immersive/glkit.py:138` `def __init__(self, dist, yaw, pitch)`
- `OrbitCamera.drag` (method) `immersive/glkit.py:151` `def drag(self, dx, dy, dt)`
- `OrbitCamera.zoom` (method) `immersive/glkit.py:158` `def zoom(self, steps)`
- `OrbitCamera.fly_to` (method) `immersive/glkit.py:162` `def fly_to(self, dist)`
- `OrbitCamera.update` (method) `immersive/glkit.py:165` `def update(self, dt)`
- `OrbitCamera.eye` (method) `immersive/glkit.py:175` `def eye(self)`
- `OrbitCamera.basis` (method) `immersive/glkit.py:179` `def basis(self)`
- `OrbitCamera.set_uniforms` (method) `immersive/glkit.py:185` `def set_uniforms(self, prog, width, height)`
- `OrbitCamera.project` (method) `immersive/glkit.py:194` `def project(self, world, width, height)` -- Return (x, y, in_front) in pixel coordinates (origin top-left).

## immersive/hud.py
Depends on: `immersive/glkit.py`, `immersive/shaders.py`
Imported by: `immersive/app.py`, `immersive/scenes.py`
- `Lang.__init__` (method) `immersive/hud.py:27` `def __init__(self, table, code)`
- `Lang.toggle` (method) `immersive/hud.py:31` `def toggle(self)`
- `Lang.__call__` (method) `immersive/hud.py:34` `def __call__(self, key)`
- `Hud.__init__` (method) `immersive/hud.py:41` `def __init__(self, width, height)`
- `Hud.resize` (method) `immersive/hud.py:51` `def resize(self, width, height)`
- `Hud.font` (method) `immersive/hud.py:57` `def font(self, size, bold)`
- `Hud.begin` (method) `immersive/hud.py:65` `def begin(self)`
- `Hud.s` (method) `immersive/hud.py:69` `def s(self, v)`
- `Hud.text` (method) `immersive/hud.py:72` `def text(self, msg, x, y, size, color, bold, anchor, alpha, shadow)`
- `Hud.panel` (method) `immersive/hud.py:99` `def panel(self, rect, radius)`
- `Hud.bar` (method) `immersive/hud.py:105` `def bar(self, rect, frac, color, back)`
- `Hud.button` (method) `immersive/hud.py:113` `def button(self, rect, label, action, hot)`
- `Hud.hit` (method) `immersive/hud.py:122` `def hit(self, pos)`
- `Hud.heatmap` (method) `immersive/hud.py:128` `def heatmap(self, values, rect, peak)`
- `Hud.hue_color` (method) `immersive/hud.py:138` `def hue_color(self, phase)`
- `Hud.wrap` (method) `immersive/hud.py:142` `def wrap(self, msg, width_px, size, bold)`
- `Hud.end` (method) `immersive/hud.py:156` `def end(self, fade)`

## immersive/matter.py
- `pyscf_available` (function) `immersive/matter.py:102` `def pyscf_available()`
- `solve` (function) `immersive/matter.py:110` `def solve(system)`
- `field_amplitude` (function) `immersive/matter.py:160` `def field_amplitude(solved, kind)` -- Signed amplitude a(r) with density a^2: sqrt(rho), signed sqrt(spin density), or an orbital.
- `exposure` (function) `immersive/matter.py:192` `def exposure(amp, half)` -- 1 / (99.7th percentile of a^2 inside the bounding sphere).
- `bonds` (function) `immersive/matter.py:201` `def bonds(symbols, coords)`
- `AsyncSolver.__init__` (method) `immersive/matter.py:214` `def __init__(self)`
- `AsyncSolver.request` (method) `immersive/matter.py:220` `def request(self, system)`
- `work` (function) `immersive/matter.py:224` `def work()`
- `AsyncSolver.prefetch` (method) `immersive/matter.py:236` `def prefetch(self, system, kinds)` -- Solve `system` and sample the given fields in one background job.
- `work` (function) `immersive/matter.py:242` `def work()`
- `AsyncSolver.request_field` (method) `immersive/matter.py:261` `def request_field(self, system, kind)`
- `work` (function) `immersive/matter.py:269` `def work()`

## immersive/physics.py
Imported by: `immersive/backend.py`, `immersive/scenes.py`, `immersive/shaders.py`, `immersive/spin.py`, `test_immersive.py`
- `unit_vector_poly_args` (function) `immersive/physics.py:32` `def unit_vector_poly_args(theta, phi)` -- Return (cos(theta), sin(theta) e^{i phi}) used by the polynomial table.
- `angular_constants` (function) `immersive/physics.py:37` `def angular_constants()` -- Derive normalisation constants of Y_l^m (m >= 0) against scipy so the GLSL table cannot drift.
- `ylm_from_table` (function) `immersive/physics.py:48` `def ylm_from_table(l, m, theta, phi, consts)` -- Evaluate Y_l^m through the same polynomial table the shader uses (for cross-checking).
- `radial_norm` (function) `immersive/physics.py:59` `def radial_norm(n, l)`
- `radial_function` (function) `immersive/physics.py:63` `def radial_function(n, l, r)` -- Hydrogenic R_nl(r) in Bohr radii.
- `radial_table` (function) `immersive/physics.py:69` `def radial_table(qn, rmax, samples)`
- `view_radius` (function) `immersive/physics.py:75` `def view_radius(n_max)` -- Bounding-sphere radius (Bohr) that contains the visible cloud of shell n.
- `energy_ev` (function) `immersive/physics.py:80` `def energy_ev(n)`
- `orbital_label` (function) `immersive/physics.py:84` `def orbital_label(n, l, m)`
- `psi_samples` (function) `immersive/physics.py:88` `def psi_samples(qn, coefs, r, theta, phi)`
- `density_peak` (function) `immersive/physics.py:95` `def density_peak(qn, coefs, rmax, phases, percentile)` -- Exposure reference: a high percentile of |psi|^2 over the visible ball and a sweep of relative phases.
- `overlap_1s` (function) `immersive/physics.py:115` `def overlap_1s(bond)` -- Overlap integral of two 1s functions separated by `bond` Bohr.
- `Gate.__init__` (method) `immersive/physics.py:157` `def __init__(self, name, hamiltonian, qubits)`
- `Gate.unitary` (method) `immersive/physics.py:163` `def unitary(self, s)`
- `QubitRegister.__init__` (method) `immersive/physics.py:190` `def __init__(self, n_qubits, seconds_per_gate)`
- `QubitRegister.reset` (method) `immersive/physics.py:201` `def reset(self)`
- `QubitRegister.set_state` (method) `immersive/physics.py:208` `def set_state(self, vec)`
- `QubitRegister.embed` (method) `immersive/physics.py:213` `def embed(self, op, qubit)`
- `QubitRegister.single` (method) `immersive/physics.py:221` `def single(self, name, qubit)`
- `QubitRegister.cnot` (method) `immersive/physics.py:226` `def cnot(self, control, target)`
- `QubitRegister.queue` (method) `immersive/physics.py:231` `def queue(self, gate)`
- `QubitRegister.busy` (method) `immersive/physics.py:235` `def busy(self)`
- `QubitRegister.progress` (method) `immersive/physics.py:239` `def progress(self)`
- `QubitRegister.active_name` (method) `immersive/physics.py:243` `def active_name(self)`
- `QubitRegister.update` (method) `immersive/physics.py:246` `def update(self, dt)`
- `QubitRegister.probabilities` (method) `immersive/physics.py:262` `def probabilities(self)`
- `QubitRegister.reduced_density` (method) `immersive/physics.py:265` `def reduced_density(self, qubit)`
- `QubitRegister.bloch` (method) `immersive/physics.py:270` `def bloch(self, qubit)`
- `QubitRegister.entropy` (method) `immersive/physics.py:274` `def entropy(self, qubit)`
- `QubitRegister.measure` (method) `immersive/physics.py:278` `def measure(self, shots, rng)`

## immersive/postfx.py
Depends on: `immersive/glkit.py`
Imported by: `immersive/app.py`
- `Target.__init__` (method) `immersive/postfx.py:90` `def __init__(self, width, height)`
- `Target.bind` (method) `immersive/postfx.py:107` `def bind(self)`
- `Target.destroy` (method) `immersive/postfx.py:111` `def destroy(self)`
- `PostProcess.__init__` (method) `immersive/postfx.py:119` `def __init__(self, width, height)`
- `PostProcess.resize` (method) `immersive/postfx.py:133` `def resize(self, width, height)`
- `PostProcess.begin` (method) `immersive/postfx.py:142` `def begin(self)`
- `PostProcess.present` (method) `immersive/postfx.py:154` `def present(self, time, width, height)`
- `source_size` (function) `immersive/postfx.py:193` `def source_size(post, index)` -- Texel grid of the texture each downsample level reads from.

## immersive/scenes.py
Depends on: `immersive/backend.py`, `immersive/glkit.py`, `immersive/hud.py`, `immersive/physics.py`
Imported by: `immersive/app.py`
- `Scene.__init__` (method) `immersive/scenes.py:139` `def __init__(self, app)`
- `Scene.enter` (method) `immersive/scenes.py:144` `def enter(self)`
- `Scene.loading` (method) `immersive/scenes.py:147` `def loading(self)`
- `Scene.update` (method) `immersive/scenes.py:150` `def update(self, dt)`
- `Scene.render` (method) `immersive/scenes.py:153` `def render(self)`
- `Scene.draw_hud` (method) `immersive/scenes.py:156` `def draw_hud(self, hud, tr)`
- `Scene.command` (method) `immersive/scenes.py:159` `def command(self, cmd)`
- `Scene.on_key` (method) `immersive/scenes.py:162` `def on_key(self, ev)`
- `Scene.panel_rect` (method) `immersive/scenes.py:169` `def panel_rect(self, hud, x, y, w, h)`
- `VolumeScene.__init__` (method) `immersive/scenes.py:178` `def __init__(self, app)`
- `VolumeScene.cloud_radius` (method) `immersive/scenes.py:201` `def cloud_radius(self)`
- `VolumeScene.cut_normal` (method) `immersive/scenes.py:204` `def cut_normal(self)`
- `VolumeScene.command` (method) `immersive/scenes.py:213` `def command(self, cmd)`
- `VolumeScene.update` (method) `immersive/scenes.py:236` `def update(self, dt)`
- `VolumeScene.set_volume_uniforms` (method) `immersive/scenes.py:243` `def set_volume_uniforms(self)`
- `VolumeScene.phase_legend` (method) `immersive/scenes.py:263` `def phase_legend(self, hud, tr, x, y)`
- `VolumeScene.cut_hud` (method) `immersive/scenes.py:273` `def cut_hud(self, hud, tr, x, y)`
- `AtomScene.__init__` (method) `immersive/scenes.py:287` `def __init__(self, app)`
- `AtomScene.enter` (method) `immersive/scenes.py:304` `def enter(self)`
- `AtomScene.load_preset` (method) `immersive/scenes.py:307` `def load_preset(self, preset)`
- `AtomScene.set_states` (method) `immersive/scenes.py:314` `def set_states(self, states, label)`
- `AtomScene.command` (method) `immersive/scenes.py:328` `def command(self, cmd)`
- `AtomScene.coefficients` (method) `immersive/scenes.py:350` `def coefficients(self)`
- `AtomScene.update` (method) `immersive/scenes.py:364` `def update(self, dt)`
- `AtomScene.render` (method) `immersive/scenes.py:368` `def render(self)`
- `AtomScene.draw_hud` (method) `immersive/scenes.py:379` `def draw_hud(self, hud, tr)`
- `AtomScene.draw_energy_panel` (method) `immersive/scenes.py:416` `def draw_energy_panel(self, hud, tr, top)`
- `SpinScene.__init__` (method) `immersive/scenes.py:458` `def __init__(self, app)`
- `SpinScene.enter` (method) `immersive/scenes.py:476` `def enter(self)`
- `SpinScene.load` (method) `immersive/scenes.py:479` `def load(self, preset)`
- `SpinScene.refresh` (method) `immersive/scenes.py:486` `def refresh(self)`
- `SpinScene.command` (method) `immersive/scenes.py:495` `def command(self, cmd)`
- `SpinScene.coefficients` (method) `immersive/scenes.py:508` `def coefficients(self)`
- `SpinScene.update` (method) `immersive/scenes.py:511` `def update(self, dt)`
- `SpinScene.render` (method) `immersive/scenes.py:519` `def render(self)`
- `SpinScene.draw_hud` (method) `immersive/scenes.py:549` `def draw_hud(self, hud, tr)`
- `SpinScene.spin_inset` (method) `immersive/scenes.py:597` `def spin_inset(self, hud, cx, cy, radius, sigma)`
- `MatterScene.__init__` (method) `immersive/scenes.py:618` `def __init__(self, app)`
- `MatterScene.enter` (method) `immersive/scenes.py:639` `def enter(self)`
- `MatterScene.warm_up` (method) `immersive/scenes.py:643` `def warm_up(self, keys)`
- `MatterScene.system` (method) `immersive/scenes.py:651` `def system(self)`
- `MatterScene.loading` (method) `immersive/scenes.py:654` `def loading(self)`
- `MatterScene.command` (method) `immersive/scenes.py:658` `def command(self, cmd)`
- `MatterScene.wanted_kind` (method) `immersive/scenes.py:688` `def wanted_kind(self)`
- `MatterScene.update` (method) `immersive/scenes.py:699` `def update(self, dt)`
- `MatterScene.apply` (method) `immersive/scenes.py:719` `def apply(self, solved, kind, token)`
- `MatterScene.render` (method) `immersive/scenes.py:744` `def render(self)`
- `MatterScene.draw_hud` (method) `immersive/scenes.py:769` `def draw_hud(self, hud, tr)`
- `MatterScene.draw_vqe` (method) `immersive/scenes.py:831` `def draw_vqe(self, hud, tr)`
- `fmt_c` (function) `immersive/scenes.py:856` `def fmt_c(z)`
- `backend_command` (function) `immersive/scenes.py:860` `def backend_command(reg, cmd)`
- `draw_backend_panel` (function) `immersive/scenes.py:867` `def draw_backend_panel(hud, tr, reg, x, y)` -- Show which engine produced the picture, plus the backend's spatial wavefunction fields.
- `BlochScene.__init__` (method) `immersive/scenes.py:910` `def __init__(self, app)`
- `BlochScene.enter` (method) `immersive/scenes.py:926` `def enter(self)`
- `BlochScene.command` (method) `immersive/scenes.py:930` `def command(self, cmd)`
- `BlochScene.update` (method) `immersive/scenes.py:945` `def update(self, dt)`
- `BlochScene.render` (method) `immersive/scenes.py:959` `def render(self)`
- `BlochScene.sphere_labels` (method) `immersive/scenes.py:981` `def sphere_labels(self, hud, centre, tag)`
- `BlochScene.draw_hud` (method) `immersive/scenes.py:989` `def draw_hud(self, hud, tr)`
- `EntangleScene.__init__` (method) `immersive/scenes.py:1046` `def __init__(self, app)`
- `EntangleScene.centres` (method) `immersive/scenes.py:1062` `def centres(self)`
- `EntangleScene.enter` (method) `immersive/scenes.py:1065` `def enter(self)`
- `EntangleScene.command` (method) `immersive/scenes.py:1069` `def command(self, cmd)`
- `EntangleScene.update` (method) `immersive/scenes.py:1099` `def update(self, dt)`
- `EntangleScene.render` (method) `immersive/scenes.py:1122` `def render(self)`
- `EntangleScene.draw_hud` (method) `immersive/scenes.py:1147` `def draw_hud(self, hud, tr)`

## immersive/shaders.py
Depends on: `immersive/physics.py`
Imported by: `immersive/app.py`, `immersive/hud.py`
- `orbital_fragment` (function) `immersive/shaders.py:578` `def orbital_fragment()`
- `bloch_fragment` (function) `immersive/shaders.py:585` `def bloch_fragment()`

## immersive/spin.py
Depends on: `immersive/backend.py`, `immersive/physics.py`, `relativistic_hydrogen.py`
- `clebsch` (function) `immersive/spin.py:24` `def clebsch(l, mj2, sign)` -- Coefficients (c_up, c_down) of |l, s=1/2; j=l+sign/2, m_j> on (Y_l^{mj-1/2} up, Y_l^{mj+1/2} down).
- `lande_g` (function) `immersive/spin.py:33` `def lande_g(l, sign)`
- `SpinorComponent.j` (method) `immersive/spin.py:50` `def j(self)`
- `SpinorComponent.kappa` (method) `immersive/spin.py:54` `def kappa(self)`
- `SpinorComponent.label` (method) `immersive/spin.py:58` `def label(self)`
- `SpinorComponent.valid` (method) `immersive/spin.py:62` `def valid(self)`
- `DiracBackendLink.__init__` (method) `immersive/spin.py:69` `def __init__(self)`
- `DiracBackendLink.ok` (method) `immersive/spin.py:89` `def ok(self)`
- `DiracBackendLink.energy` (method) `immersive/spin.py:92` `def energy(self, comp)`
- `component_energy` (function) `immersive/spin.py:141` `def component_energy(comp, link, field_tesla)`
- `time_scale` (function) `immersive/spin.py:146` `def time_scale(components, link, field_tesla, beat_seconds)` -- Atomic-time units advanced per screen second, chosen so the dominant beat lasts `beat_seconds` on screen.
- `real_time_label` (function) `immersive/spin.py:155` `def real_time_label(au_per_screen_second)`
- `evolved_coefficients` (function) `immersive/spin.py:163` `def evolved_coefficients(components, energies, t_au)`
- `spinor_field` (function) `immersive/spin.py:168` `def spinor_field(components, coefs, points)` -- Two-component spinor (up, down) at cartesian `points` (N, 3).
- `measure` (function) `immersive/spin.py:196` `def measure(components, coefs, reach, arrows)`
- `exposure` (function) `immersive/spin.py:216` `def exposure(components, energies, rmax, percentile)` -- 1 / percentile of the spinor density over the visible ball, over several beat phases.
- `shared_link` (function) `immersive/spin.py:238` `def shared_link()`

## molecular_sim.py
Depends on: `quantum_computer.py`
Imported by: `advanced_experiments.py`, `app.py`, `entangled_hydrogen.py`, `polarizability_v3.py`, `quantum_dash.py`, `quantum_visualizer.py`
- `_LazyMoleculeRegistry.__init__` (method) `molecular_sim.py:143` `def __init__(self, builders)`
- `_LazyMoleculeRegistry.get` (method) `molecular_sim.py:162` `def get(self, name, default)`
- `build_jw_hamiltonian_of` (function) `molecular_sim.py:176` `def build_jw_hamiltonian_of(mol)` -- Construye Hamiltoniano JW usando OpenFermion correctamente.
- `ExactJWEnergy.__init__` (method) `molecular_sim.py:315` `def __init__(self, mol, n_qubits)`
- `ExactJWEnergy.__call__` (method) `molecular_sim.py:421` `def __call__(self, amps)`
- `SurrogateEnergy.__init__` (method) `molecular_sim.py:432` `def __init__(self, mol, n_qubits, exact_eval, backend)`
- `SurrogateEnergy.calibrate` (method) `molecular_sim.py:441` `def calibrate(self, hf_amps)`
- `SurrogateEnergy.cost_with_barrier` (method) `molecular_sim.py:454` `def cost_with_barrier(self, amps)`
- `prepare_hf` (function) `molecular_sim.py:468` `def prepare_hf(mol, factory, backend)`
- `uccsd` (function) `molecular_sim.py:539` `def uccsd(state, thetas, singles, doubles, backend, runner)` -- UCCSD ansatz (fermiónico, con signos Jordan-Wigner exactos).
- `VQESolver.__init__` (method) `molecular_sim.py:593` `def __init__(self, qc, config)`
- `VQESolver.run` (method) `molecular_sim.py:605` `def run(self, mol, backend, max_iter, tol)`
- `cost` (function) `molecular_sim.py:653` `def cost(thetas)`

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
- `main` (function) `orbital_visualizer2.py:433` `def main()`

## polarizability_v3.py
Depends on: `molecular_sim.py`, `quantum_computer.py`
- `givens_single_excitation` (function) `polarizability_v3.py:73` `def givens_single_excitation(state, o, v, theta, n_qubits, backend)` -- Apply a particle-conserving single excitation rotation between qubits o (occupied) and v (virtual).
- `particle_conserving_ansatz` (function) `polarizability_v3.py:121` `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)` -- Particle-conserving UCCSD-like ansatz: - Singles: Givens rotations (correct particle conservation) - Doubles: reuse...
- `StarkEvaluator.__init__` (method) `polarizability_v3.py:164` `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- `StarkEvaluator.eval_dipole` (method) `polarizability_v3.py:196` `def eval_dipole(self, amps_raw)`
- `StarkEvaluator.__call__` (method) `polarizability_v3.py:209` `def __call__(self, amps)`
- `DipoleOperatorBuilder.__init__` (method) `polarizability_v3.py:216` `def __init__(self, bond_length_angstrom)`
- `PolarizabilityCalculator.__init__` (method) `polarizability_v3.py:250` `def __init__(self)`
- `cost` (function) `polarizability_v3.py:302` `def cost(th)`
- `PolarizabilityCalculator.run` (method) `polarizability_v3.py:315` `def run(self)`


Next: [API_p2.md](API_p2.md)
