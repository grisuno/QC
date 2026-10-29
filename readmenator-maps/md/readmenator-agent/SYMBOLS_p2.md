# Symbols (page 2 of 5)
Previous: [SYMBOLS.md](SYMBOLS.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `__init__` | method | `molecular_sim.py:432` | `def __init__(self, mol, n_qubits, exact_eval, backend)` |
| `__init__` | method | `molecular_sim.py:593` | `def __init__(self, qc, config)` |
| `__repr__` | method | `molecular_sim.py:579` | `def __repr__(self)` |
| `_apply` | method | `molecular_sim.py:359` | `def _apply(self, amps, pauli)` |
| `_evaluate` | method | `molecular_sim.py:405` | `def _evaluate(self, amps)` |
| `_evaluate_integrals` | method | `molecular_sim.py:379` | `def _evaluate_integrals(self, amps)` |
| `_h2_sto3g_hardcoded` | function | `molecular_sim.py:75` | `def _h2_sto3g_hardcoded()` |
| `_h2_sto3g_pyscf` | function | `molecular_sim.py:41` | `def _h2_sto3g_pyscf()` |
| `_integral_jw_terms` | function | `molecular_sim.py:252` | `def _integral_jw_terms(mol)` |
| `_jw_excitation_map` | function | `molecular_sim.py:485` | `def _jw_excitation_map(n_qubits, k, create, annihilate)` |
| `_jw_givens_evolve` | function | `molecular_sim.py:515` | `def _jw_givens_evolve(state, create, annihilate, theta)` |
| `_make_logger` | function | `molecular_sim.py:18` | `def _make_logger(name)` |
| `_resolve` | method | `molecular_sim.py:149` | `def _resolve(self, name)` |
| `_run` | method | `molecular_sim.py:597` | `def _run(self, circ, be, state)` |
| `_sd_indices` | function | `molecular_sim.py:476` | `def _sd_indices(n_e, n_q)` |
| `_to_scalar` | method | `molecular_sim.py:339` | `def _to_scalar(amps)` |
| `_verify_hf` | method | `molecular_sim.py:349` | `def _verify_hf(self)` |
| `build_jw_hamiltonian_of` | function | `molecular_sim.py:176` | `def build_jw_hamiltonian_of(mol)` |
| `calibrate` | method | `molecular_sim.py:441` | `def calibrate(self, hf_amps)` |
| `cost` | function | `molecular_sim.py:653` | `def cost(thetas)` |
| `cost_with_barrier` | method | `molecular_sim.py:454` | `def cost_with_barrier(self, amps)` |
| `get` | method | `molecular_sim.py:162` | `def get(self, name, default)` |
| `prepare_hf` | function | `molecular_sim.py:468` | `def prepare_hf(mol, factory, backend)` |
| `run` | method | `molecular_sim.py:605` | `def run(self, mol, backend, max_iter, tol)` |
| `uccsd` | function | `molecular_sim.py:539` | `def uccsd(state, thetas, singles, doubles, backend, runner)` |
| `Config` | class | `orbital_visualizer2.py:44` | `class Config` |
| `HamiltonianNNProcessor` | class | `orbital_visualizer2.py:126` | `class HamiltonianNNProcessor` |
| `MonteCarloSampler` | class | `orbital_visualizer2.py:160` | `class MonteCarloSampler` |
| `OrbitalVisualizer` | class | `orbital_visualizer2.py:265` | `class OrbitalVisualizer` |
| `WavefunctionCalculator` | class | `orbital_visualizer2.py:83` | `class WavefunctionCalculator` |
| `__init__` | method | `orbital_visualizer2.py:129` | `def __init__(self, engine)` |
| `__init__` | method | `orbital_visualizer2.py:163` | `def __init__(self, hamiltonian_processor)` |
| `_plotly` | method | `orbital_visualizer2.py:396` | `def _plotly(self, X, Y, Z, prob_norm, phases, n, l, m)` |
| `compute_expected_energy` | method | `orbital_visualizer2.py:136` | `def compute_expected_energy(self, n, l, m)` |
| `find_max_probability` | method | `orbital_visualizer2.py:166` | `def find_max_probability(self, n, l, m)` |
| `is_model_loaded` | method | `orbital_visualizer2.py:133` | `def is_model_loaded(self)` |
| `main` | function | `orbital_visualizer2.py:433` | `def main()` |
| `psi_on_grid` | method | `orbital_visualizer2.py:107` | `def psi_on_grid(n, l, m, grid_size)` |
| `radial_wavefunction` | method | `orbital_visualizer2.py:87` | `def radial_wavefunction(n, l, r)` |
| `sample` | method | `orbital_visualizer2.py:195` | `def sample(self, n, l, m, num_samples)` |
| `spherical_harmonic_real` | method | `orbital_visualizer2.py:97` | `def spherical_harmonic_real(l, m, theta, phi)` |
| `visualize` | method | `orbital_visualizer2.py:268` | `def visualize(self, data, save_path, hamiltonian_processor)` |
| `DipoleOperatorBuilder` | class | `polarizability_v3.py:215` | `class DipoleOperatorBuilder` |
| `PolarizabilityCalculator` | class | `polarizability_v3.py:249` | `class PolarizabilityCalculator` |
| `StarkEvaluator` | class | `polarizability_v3.py:163` | `class StarkEvaluator` |
| `VQEResult` | class | `polarizability_v3.py:51` | `class VQEResult` |
| `__call__` | method | `polarizability_v3.py:209` | `def __call__(self, amps)` |
| `__init__` | method | `polarizability_v3.py:164` | `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)` |
| `__init__` | method | `polarizability_v3.py:216` | `def __init__(self, bond_length_angstrom)` |
| `__init__` | method | `polarizability_v3.py:250` | `def __init__(self)` |
| `_apply_pauli` | method | `polarizability_v3.py:177` | `def _apply_pauli(self, amps, pauli)` |
| `_diagnose` | method | `polarizability_v3.py:279` | `def _diagnose(self, field, theta, label)` |
| `_evaluator` | method | `polarizability_v3.py:270` | `def _evaluator(self, field)` |
| `_get_state` | method | `polarizability_v3.py:274` | `def _get_state(self, theta)` |
| `_optimize` | method | `polarizability_v3.py:289` | `def _optimize(self, field, theta_init, n_restarts)` |
| `_run_circuit` | function | `polarizability_v3.py:67` | `def _run_circuit(circuit, backend, state)` |
| `_sd_indices` | function | `polarizability_v3.py:58` | `def _sd_indices(n_e, n_q)` |
| `_to_scalar` | method | `polarizability_v3.py:173` | `def _to_scalar(amps)` |
| `cost` | function | `polarizability_v3.py:302` | `def cost(th)` |
| `eval_dipole` | method | `polarizability_v3.py:196` | `def eval_dipole(self, amps_raw)` |
| `givens_single_excitation` | function | `polarizability_v3.py:73` | `def givens_single_excitation(state, o, v, theta, n_qubits, backend)` |
| `particle_conserving_ansatz` | function | `polarizability_v3.py:121` | `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)` |
| `run` | method | `polarizability_v3.py:315` | `def run(self)` |
| `BackendComparator` | class | `qc_dashboard.py:1344` | `class BackendComparator` |
| `BrutalVizEngine` | class | `qc_dashboard.py:1269` | `class BrutalVizEngine` |
| `DashboardApp` | class | `qc_dashboard.py:1371` | `class DashboardApp` |
| `DashboardConfig` | class | `qc_dashboard.py:55` | `class DashboardConfig` |
| `GateItem` | class | `qc_dashboard.py:120` | `class GateItem` |
| `H2VQESolver` | class | `qc_dashboard.py:147` | `class H2VQESolver` |
| `Plotly3DEngine` | class | `qc_dashboard.py:979` | `class Plotly3DEngine` |
| `RealOrbitalEngine` | class | `qc_dashboard.py:1151` | `class RealOrbitalEngine` |
| `SimulatorBackend` | class | `qc_dashboard.py:725` | `class SimulatorBackend` |
| `SnapshotData` | class | `qc_dashboard.py:130` | `class SnapshotData` |
| `VisualisationEngine` | class | `qc_dashboard.py:340` | `class VisualisationEngine` |
| `_FakeConfig` | class | `qc_dashboard.py:1176` | `class _FakeConfig` |
| `__init__` | method | `qc_dashboard.py:166` | `def __init__(self, config)` |
| `__init__` | method | `qc_dashboard.py:343` | `def __init__(self, config)` |
| `__init__` | method | `qc_dashboard.py:728` | `def __init__(self, config)` |
| `__init__` | method | `qc_dashboard.py:982` | `def __init__(self, config)` |
| `__init__` | method | `qc_dashboard.py:1154` | `def __init__(self)` |
| `__init__` | method | `qc_dashboard.py:1272` | `def __init__(self)` |
| `__init__` | method | `qc_dashboard.py:1347` | `def __init__(self, config)` |
| `__init__` | method | `qc_dashboard.py:1374` | `def __init__(self, config)` |
| `_ansatz_state` | method | `qc_dashboard.py:229` | `def _ansatz_state(theta)` |
| `_auto_run` | method | `qc_dashboard.py:1958` | `def _auto_run(self, st)` |
| `_build_hamiltonian` | method | `qc_dashboard.py:206` | `def _build_hamiltonian(self, bond_length)` |
| `_build_qasm_from_gates` | method | `qc_dashboard.py:1969` | `def _build_qasm_from_gates(self, gates)` |
| `_capture_mpl_fig` | function | `qc_dashboard.py:1133` | `def _capture_mpl_fig(func)` |
| `_compute_bloch_mps` | method | `qc_dashboard.py:917` | `def _compute_bloch_mps(state, n_qubits)` |
| `_energy` | method | `qc_dashboard.py:241` | `def _energy(self, theta, h_matrix, e_nuc)` |
| `_ensure_streamlit` | method | `qc_dashboard.py:1399` | `def _ensure_streamlit()` |
| `_get_mps_gate_registry` | method | `qc_dashboard.py:760` | `def _get_mps_gate_registry()` |
| `_get_sv_gate_registry` | method | `qc_dashboard.py:768` | `def _get_sv_gate_registry()` |
| `_hex_to_rgb` | method | `qc_dashboard.py:715` | `def _hex_to_rgb(h)` |
| `_init_framework` | method | `qc_dashboard.py:737` | `def _init_framework(self)` |
| `_init_plotly` | method | `qc_dashboard.py:988` | `def _init_plotly(self)` |
| `_init_plotting` | method | `qc_dashboard.py:349` | `def _init_plotting(self)` |
| `_init_real` | method | `qc_dashboard.py:1163` | `def _init_real(self)` |
| `_init_real` | method | `qc_dashboard.py:1279` | `def _init_real(self)` |
| `_init_session` | method | `qc_dashboard.py:1408` | `def _init_session(st)` |
| `_mps_execute` | method | `qc_dashboard.py:788` | `def _mps_execute(self, gates, n_qubits)` |
| `_parse_orb` | function | `qc_dashboard.py:1936` | `def _parse_orb(k)` |
| `_pauli_operators` | method | `qc_dashboard.py:173` | `def _pauli_operators(n_qubits, qubit, op)` |
| `_pauli_string_matrix` | method | `qc_dashboard.py:189` | `def _pauli_string_matrix(paulis, n_qubits)` |
| `_render_3d_tab` | method | `qc_dashboard.py:1803` | `def _render_3d_tab(self, st)` |
| `_render_bloch_sphere` | method | `qc_dashboard.py:606` | `def _render_bloch_sphere(self, snap, ax)` |
| `_render_entanglement_tab` | method | `qc_dashboard.py:1688` | `def _render_entanglement_tab(self, st)` |
| `_render_entropy_tab` | method | `qc_dashboard.py:1669` | `def _render_entropy_tab(self, st)` |
| `_render_main_panel` | method | `qc_dashboard.py:1535` | `def _render_main_panel(self, st)` |
| `_render_molecule_tab` | method | `qc_dashboard.py:1732` | `def _render_molecule_tab(self, st)` |
| `_render_orbital_tab` | method | `qc_dashboard.py:1855` | `def _render_orbital_tab(self, st)` |
| `_render_phase_space` | method | `qc_dashboard.py:635` | `def _render_phase_space(self, snap, ax)` |
| `_render_playground_tab` | method | `qc_dashboard.py:1563` | `def _render_playground_tab(self, st)` |
| `_render_probabilities` | method | `qc_dashboard.py:589` | `def _render_probabilities(self, snap, ax)` |
| `_render_qasm_tab` | method | `qc_dashboard.py:1606` | `def _render_qasm_tab(self, st)` |
| `_render_sidebar_controls` | method | `qc_dashboard.py:1447` | `def _render_sidebar_controls(self, st)` |
| `_render_ui` | method | `qc_dashboard.py:1432` | `def _render_ui(self, st)` |
| `_snapshot_from_mps` | method | `qc_dashboard.py:886` | `def _snapshot_from_mps(self, state, step, gate_name)` |
| `_snapshot_from_sv` | method | `qc_dashboard.py:898` | `def _snapshot_from_sv(self, state, step, gate_name)` |
| `_sv_execute` | method | `qc_dashboard.py:839` | `def _sv_execute(self, gates, n_qubits)` |
| `_synthetic_execute` | method | `qc_dashboard.py:933` | `def _synthetic_execute(self, gates, n_qubits)` |
| `available` | method | `qc_dashboard.py:362` | `def available(self)` |
| `available` | method | `qc_dashboard.py:998` | `def available(self)` |
| `available` | method | `qc_dashboard.py:1210` | `def available(self)` |
| `dash_available` | method | `qc_dashboard.py:1297` | `def dash_available(self)` |
| `energy_landscape` | method | `qc_dashboard.py:293` | `def energy_landscape(self)` |
| `entangled_available` | method | `qc_dashboard.py:1214` | `def entangled_available(self)` |
| `execute_circuit` | method | `qc_dashboard.py:775` | `def execute_circuit(self, gates, n_qubits)` |
| `hologram_available` | method | `qc_dashboard.py:1301` | `def hologram_available(self)` |
| `main` | function | `qc_dashboard.py:1994` | `def main()` |
| `orbital_wavefunction` | method | `qc_dashboard.py:318` | `def orbital_wavefunction(bond_length, grid_points)` |
| `render_bloch_3d` | method | `qc_dashboard.py:1001` | `def render_bloch_3d(self, bloch_vectors)` |
| `render_energy_landscape` | method | `qc_dashboard.py:501` | `def render_energy_landscape(self, bond_lengths, vqe_energies, hf_energies, fci_energies)` |
| `render_entangled_to_bytes` | method | `qc_dashboard.py:1250` | `def render_entangled_to_bytes(self, data)` |
| `render_entanglement_profile` | method | `qc_dashboard.py:429` | `def render_entanglement_profile(self, snapshots)` |
| `render_entropy_chart` | method | `qc_dashboard.py:403` | `def render_entropy_chart(self, snapshots)` |
| `render_entropy_scaling` | method | `qc_dashboard.py:562` | `def render_entropy_scaling(self, data)` |
| `render_full_dashboard` | method | `qc_dashboard.py:365` | `def render_full_dashboard(self, snapshots, current)` |
| `render_hologram` | method | `qc_dashboard.py:1327` | `def render_hologram(self, snapshots, backend_comp)` |
| `render_orbital_2d_projections` | method | `qc_dashboard.py:670` | `def render_orbital_2d_projections(self, data)` |
| `render_orbital_plot` | method | `qc_dashboard.py:533` | `def render_orbital_plot(self, orbital_data)` |
| `render_probability_3d` | method | `qc_dashboard.py:1049` | `def render_probability_3d(self, probabilities, n_qubits)` |
| `render_state_3d` | method | `qc_dashboard.py:1080` | `def render_state_3d(self, probabilities, phases)` |
| `render_to_bytes` | method | `qc_dashboard.py:1222` | `def render_to_bytes(self, data)` |
| `render_vqe_convergence` | method | `qc_dashboard.py:474` | `def render_vqe_convergence(self, convergence, e_hf, e_fci)` |
| `run` | method | `qc_dashboard.py:1384` | `def run(self)` |
| `run_brutal_viz` | method | `qc_dashboard.py:1304` | `def run_brutal_viz(self, circuit_name)` |
| `run_comparison` | method | `qc_dashboard.py:1351` | `def run_comparison(self, gates, n_qubits)` |
| `run_vqe` | method | `qc_dashboard.py:248` | `def run_vqe(self, bond_length, max_iter)` |
| `sample` | method | `qc_dashboard.py:1217` | `def sample(self, n, l, m, num_samples)` |
| `sample_entangled` | method | `qc_dashboard.py:1240` | `def sample_entangled(self, n1, l1, m1, n2, l2, m2, num_samples)` |
| `CircuitIR` | class | `qc_integration.py:146` | `class CircuitIR` |
| `FrameworkAdapter` | class | `qc_integration.py:557` | `class FrameworkAdapter` |
| `GateInstruction` | class | `qc_integration.py:130` | `class GateInstruction` |
| `IQCAdapter` | class | `qc_integration.py:181` | `class IQCAdapter(ABC)` |
| `IntegrationBridge` | class | `qc_integration.py:727` | `class IntegrationBridge` |
| `IntegrationConfig` | class | `qc_integration.py:77` | `class IntegrationConfig` |
| `OpenQasmAdapter` | class | `qc_integration.py:226` | `class OpenQasmAdapter(IQCAdapter)` |
| `PennyLaneAdapter` | class | `qc_integration.py:455` | `class PennyLaneAdapter(IQCAdapter)` |
| `QiskitAdapter` | class | `qc_integration.py:358` | `class QiskitAdapter(IQCAdapter)` |
| `StandardCircuitFactory` | class | `qc_integration.py:659` | `class StandardCircuitFactory` |
| `__init__` | method | `qc_integration.py:238` | `def __init__(self, config)` |
| `__init__` | method | `qc_integration.py:361` | `def __init__(self, config)` |
| `__init__` | method | `qc_integration.py:458` | `def __init__(self, config)` |
| `__init__` | method | `qc_integration.py:565` | `def __init__(self, config)` |
| `__init__` | method | `qc_integration.py:753` | `def __init__(self, config)` |
| `__len__` | method | `qc_integration.py:164` | `def __len__(self)` |
| `__post_init__` | method | `qc_integration.py:115` | `def __post_init__(self)` |
| `__post_init__` | method | `qc_integration.py:137` | `def __post_init__(self)` |
| `__repr__` | method | `qc_integration.py:167` | `def __repr__(self)` |
| `_build_gate_ops` | method | `qc_integration.py:517` | `def _build_gate_ops(pl)` |
| `_build_qiskit_method_map` | method | `qc_integration.py:414` | `def _build_qiskit_method_map(qc)` |
| `_build_reverse_gate_map` | method | `qc_integration.py:433` | `def _build_reverse_gate_map()` |
| `_build_reverse_ops` | method | `qc_integration.py:535` | `def _build_reverse_ops(pl)` |
| `_detect_framework` | method | `qc_integration.py:598` | `def _detect_framework(circuit)` |
| `_ensure_pennylane` | method | `qc_integration.py:506` | `def _ensure_pennylane()` |
| `_ensure_qiskit` | method | `qc_integration.py:403` | `def _ensure_qiskit()` |
| `_from_mps_circuit` | method | `qc_integration.py:607` | `def _from_mps_circuit(circuit)` |
| `_from_sv_circuit` | method | `qc_integration.py:631` | `def _from_sv_circuit(circuit)` |
| `_init_optional_adapters` | method | `qc_integration.py:763` | `def _init_optional_adapters(self)` |
| `_param_names_for_gate` | method | `qc_integration.py:339` | `def _param_names_for_gate(name)` |
| `_to_mps_circuit` | method | `qc_integration.py:623` | `def _to_mps_circuit(cir)` |
| `_to_sv_circuit` | method | `qc_integration.py:647` | `def _to_sv_circuit(cir)` |
| `append` | method | `qc_integration.py:156` | `def append(self, gate)` |
| `bell_state` | method | `qc_integration.py:663` | `def bell_state()` |
| `circuit_fn` | function | `qc_integration.py:467` | `def circuit_fn()` |
| `export` | method | `qc_integration.py:185` | `def export(self, circuit)` |
| `export` | method | `qc_integration.py:243` | `def export(self, circuit)` |
| `export` | method | `qc_integration.py:364` | `def export(self, circuit)` |
| `export` | method | `qc_integration.py:461` | `def export(self, circuit)` |
| `export_qasm` | method | `qc_integration.py:780` | `def export_qasm(self, circuit)` |
| `export_qasm_from_framework` | method | `qc_integration.py:828` | `def export_qasm_from_framework(self, circuit, framework_type)` |
| `from_circuit_ir` | method | `qc_integration.py:584` | `def from_circuit_ir(self, cir, framework_type)` |
| `from_circuit_ir` | method | `qc_integration.py:819` | `def from_circuit_ir(self, cir, framework_type)` |
| `from_pennylane` | method | `qc_integration.py:805` | `def from_pennylane(self, pennylane_data)` |
| `from_qiskit` | method | `qc_integration.py:793` | `def from_qiskit(self, qiskit_circuit)` |
| `ghz_state` | method | `qc_integration.py:670` | `def ghz_state(n_qubits)` |
| `grover` | method | `qc_integration.py:701` | `def grover(n_qubits, marked, iterations)` |
| `import_` | method | `qc_integration.py:189` | `def import_(self, data)` |
| `import_` | method | `qc_integration.py:277` | `def import_(self, data)` |
| `import_` | method | `qc_integration.py:379` | `def import_(self, data)` |
| `import_` | method | `qc_integration.py:483` | `def import_(self, data)` |
| `import_qasm` | method | `qc_integration.py:783` | `def import_qasm(self, qasm_str)` |
| `import_qasm_to_framework` | method | `qc_integration.py:837` | `def import_qasm_to_framework(self, qasm_str, framework_type)` |
| `main` | function | `qc_integration.py:851` | `def main()` |
| `num_qubits` | method | `qc_integration.py:141` | `def num_qubits(self)` |
| `qft` | method | `qc_integration.py:678` | `def qft(n_qubits)` |
| `to_circuit_ir` | method | `qc_integration.py:568` | `def to_circuit_ir(self, circuit, framework_type)` |
| `to_circuit_ir` | method | `qc_integration.py:812` | `def to_circuit_ir(self, circuit, framework_type)` |
| `to_pennylane` | method | `qc_integration.py:800` | `def to_pennylane(self, circuit)` |
| `to_qiskit` | method | `qc_integration.py:788` | `def to_qiskit(self, circuit)` |
| `w_state` | method | `qc_integration.py:693` | `def w_state(n_qubits)` |
| `Experience` | class | `qc_launcher.py:38` | `class Experience` |
| `Launcher` | class | `qc_launcher.py:87` | `class Launcher(App)` |
| `__init__` | method | `qc_launcher.py:105` | `def __init__(self)` |
| `_options` | method | `qc_launcher.py:128` | `def _options(self)` |
| `action_launch` | method | `qc_launcher.py:167` | `def action_launch(self)` |
| `action_toggle_lang` | method | `qc_launcher.py:151` | `def action_toggle_lang(self)` |
| `build_command` | method | `qc_launcher.py:159` | `def build_command(self, exp)` |
| `compose` | method | `qc_launcher.py:114` | `def compose(self)` |
| `on_mount` | method | `qc_launcher.py:131` | `def on_mount(self)` |
| `on_option_list_option_highlighted` | method | `qc_launcher.py:143` | `def on_option_list_option_highlighted(self, event)` |
| `on_option_list_option_selected` | method | `qc_launcher.py:147` | `def on_option_list_option_selected(self, event)` |
| `refresh_detail` | method | `qc_launcher.py:136` | `def refresh_detail(self)` |
| `t` | method | `qc_launcher.py:111` | `def t(self, key)` |
| `BrutalConfig` | class | `quantum_3dview.py:39` | `class BrutalConfig` |
| `BrutalDashboard` | class | `quantum_3dview.py:560` | `class BrutalDashboard` |
| `BrutalTheme` | class | `quantum_3dview.py:32` | `class BrutalTheme(Enum)` |
| `QuantumHologram` | class | `quantum_3dview.py:84` | `class QuantumHologram` |
| `QuantumNeuralTopology` | class | `quantum_3dview.py:403` | `class QuantumNeuralTopology` |
| `QuantumSonification` | class | `quantum_3dview.py:500` | `class QuantumSonification` |
| `__init__` | method | `quantum_3dview.py:87` | `def __init__(self, config)` |
| `__init__` | method | `quantum_3dview.py:406` | `def __init__(self, model)` |
| `__init__` | method | `quantum_3dview.py:503` | `def __init__(self, sample_rate)` |
| `__init__` | method | `quantum_3dview.py:563` | `def __init__(self, config)` |
| `_add_bloch_sphere_holographic` | method | `quantum_3dview.py:292` | `def _add_bloch_sphere_holographic(self, fig, snapshot, backend_name, row, col)` |
| `_add_entropy_trails` | method | `quantum_3dview.py:245` | `def _add_entropy_trails(self, fig, snapshots, row, col)` |
| `_add_holographic_field` | method | `quantum_3dview.py:148` | `def _add_holographic_field(self, fig, snapshots, row, col)` |
| `_adsr_envelope` | method | `quantum_3dview.py:538` | `def _adsr_envelope(self, length, intensity)` |
| `_extract_layers` | method | `quantum_3dview.py:477` | `def _extract_layers(self, model)` |
| `_generate_quantum_topology` | method | `quantum_3dview.py:490` | `def _generate_quantum_topology(self)` |
| `_interpolate_color` | method | `quantum_3dview.py:389` | `def _interpolate_color(self, color1, color2, factor)` |
| `_save_audio` | method | `quantum_3dview.py:605` | `def _save_audio(self, audio, path)` |
| `colors` | method | `quantum_3dview.py:52` | `def colors(self)` |
| `create_amplitude_hologram` | method | `quantum_3dview.py:92` | `def create_amplitude_hologram(self, snapshots, backend_comparison)` |
| `create_synthetic_snapshots` | function | `quantum_3dview.py:653` | `def create_synthetic_snapshots()` |
| `create_topology_map` | method | `quantum_3dview.py:410` | `def create_topology_map(self)` |
| `demo_brutal` | function | `quantum_3dview.py:614` | `def demo_brutal()` |
| `generate_full_report` | method | `quantum_3dview.py:570` | `def generate_full_report(self, snapshots, backend_comparison)` |
| `hex_to_rgb` | function | `quantum_3dview.py:391` | `def hex_to_rgb(hex_color)` |
| `rgb_to_hex` | function | `quantum_3dview.py:395` | `def rgb_to_hex(rgb)` |
| `state_to_audio` | method | `quantum_3dview.py:506` | `def state_to_audio(self, snapshot, duration)` |
| `CNOTGate` | class | `quantum_computer.py:620` | `class CNOTGate(IQuantumGate)` |
| `CZGate` | class | `quantum_computer.py:644` | `class CZGate(IQuantumGate)` |
| `CircuitInstruction` | class | `quantum_computer.py:818` | `class CircuitInstruction` |
| `EvolveGate` | class | `quantum_computer.py:751` | `class EvolveGate(IQuantumGate)` |
| `HadamardGate` | class | `quantum_computer.py:485` | `class HadamardGate(IQuantumGate)` |
| `IQuantumGate` | class | `quantum_computer.py:466` | `class IQuantumGate(ABC)` |
| `JointHilbertState` | class | `quantum_computer.py:119` | `class JointHilbertState` |
| `JointStateFactory` | class | `quantum_computer.py:326` | `class JointStateFactory` |
| `MCZGate` | class | `quantum_computer.py:719` | `class MCZGate(IQuantumGate)` |
| `MeasurementResult` | class | `quantum_computer.py:911` | `class MeasurementResult` |
| `PauliXGate` | class | `quantum_computer.py:500` | `class PauliXGate(IQuantumGate)` |
| `PauliYGate` | class | `quantum_computer.py:514` | `class PauliYGate(IQuantumGate)` |
| `PauliZGate` | class | `quantum_computer.py:528` | `class PauliZGate(IQuantumGate)` |
| `PotentialGenerator` | class | `quantum_computer.py:230` | `class PotentialGenerator` |
| `QuantumCircuit` | class | `quantum_computer.py:825` | `class QuantumCircuit` |
| `QuantumComputer` | class | `quantum_computer.py:965` | `class QuantumComputer` |
| `RxGate` | class | `quantum_computer.py:571` | `class RxGate(IQuantumGate)` |
| `RyGate` | class | `quantum_computer.py:587` | `class RyGate(IQuantumGate)` |
| `RzGate` | class | `quantum_computer.py:603` | `class RzGate(IQuantumGate)` |
| `SGate` | class | `quantum_computer.py:542` | `class SGate(IQuantumGate)` |
| `SWAPGate` | class | `quantum_computer.py:667` | `class SWAPGate(IQuantumGate)` |
| `SimulatorConfig` | class | `quantum_computer.py:87` | `class SimulatorConfig` |
| `TGate` | class | `quantum_computer.py:556` | `class TGate(IQuantumGate)` |
| `ToffoliGate` | class | `quantum_computer.py:690` | `class ToffoliGate(IQuantumGate)` |
| `__init__` | method | `quantum_computer.py:145` | `def __init__(self, amplitudes, n_qubits)` |
| `__init__` | method | `quantum_computer.py:233` | `def __init__(self, config)` |
| `__init__` | method | `quantum_computer.py:329` | `def __init__(self, config)` |
| `__init__` | method | `quantum_computer.py:832` | `def __init__(self, n_qubits)` |
| `__init__` | method | `quantum_computer.py:993` | `def __init__(self, config)` |
| `__len__` | method | `quantum_computer.py:900` | `def __len__(self)` |
| `__repr__` | method | `quantum_computer.py:903` | `def __repr__(self)` |
| `__repr__` | method | `quantum_computer.py:945` | `def __repr__(self)` |
| `_append` | method | `quantum_computer.py:889` | `def _append(self, gate_name, targets, params)` |
| `_build_basis_amplitude` | function | `quantum_computer.py:309` | `def _build_basis_amplitude(config, basis_idx, generator)` |
| `_check` | function | `quantum_computer.py:1237` | `def _check(label, condition)` |
| `_demo` | function | `quantum_computer.py:1474` | `def _demo(config)` |
| `_empty` | method | `quantum_computer.py:334` | `def _empty(self, n_qubits)` |
| `_grid` | method | `quantum_computer.py:237` | `def _grid(self)` |
| `_make_logger` | function | `quantum_computer.py:71` | `def _make_logger(name)` |
| `_select_backend` | method | `quantum_computer.py:1007` | `def _select_backend(self, name)` |
| `_single_qubit_unitary` | function | `quantum_computer.py:361` | `def _single_qubit_unitary(state, qubit, u, backend)` |
| `_solve_eigenstate` | function | `quantum_computer.py:280` | `def _solve_eigenstate(config, potential, n, generator)` |
| `_state_to_result` | method | `quantum_computer.py:1012` | `def _state_to_result(self, state)` |
| `_two_qubit_unitary` | function | `quantum_computer.py:411` | `def _two_qubit_unitary(state, ctrl, tgt, u4)` |
| `all_zeros` | method | `quantum_computer.py:338` | `def all_zeros(self, n_qubits)` |
| `apply` | method | `quantum_computer.py:475` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:492` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:507` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:521` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:535` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:549` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:563` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:578` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:594` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:610` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:632` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:655` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:678` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:701` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:739` | `def apply(self, state, backend, targets, params)` |
| `apply` | method | `quantum_computer.py:773` | `def apply(self, state, backend, targets, params)` |
| `barrier` | method | `quantum_computer.py:886` | `def barrier(self)` |
| `basis_state` | method | `quantum_computer.py:346` | `def basis_state(self, n_qubits, k)` |
| `bell_state` | method | `quantum_computer.py:1081` | `def bell_state(self, backend)` |
| `bloch_vector` | method | `quantum_computer.py:187` | `def bloch_vector(self, qubit)` |
| `ccx` | method | `quantum_computer.py:880` | `def ccx(self, c0, c1, tgt)` |
| `clone` | method | `quantum_computer.py:225` | `def clone(self)` |
| `cnot` | method | `quantum_computer.py:865` | `def cnot(self, ctrl, tgt)` |
| `coulomb` | method | `quantum_computer.py:257` | `def coulomb(self)` |
| `cx` | method | `quantum_computer.py:868` | `def cx(self, ctrl, tgt)` |
| `cz` | method | `quantum_computer.py:871` | `def cz(self, ctrl, tgt)` |
| `depth` | method | `quantum_computer.py:897` | `def depth(self)` |
| `deutsch_jozsa` | method | `quantum_computer.py:1217` | `def deutsch_jozsa(self, n_input_qubits, is_constant, backend)` |
| `double_well` | method | `quantum_computer.py:250` | `def double_well(self)` |
| `entropy` | method | `quantum_computer.py:937` | `def entropy(self)` |
| `evolve` | method | `quantum_computer.py:883` | `def evolve(self, qubits, dt, steps)` |
| `expectation_z` | method | `quantum_computer.py:933` | `def expectation_z(self, qubit)` |
| `from_bitstring` | method | `quantum_computer.py:356` | `def from_bitstring(self, bitstring)` |
| `ghz_state` | method | `quantum_computer.py:1091` | `def ghz_state(self, n_qubits, backend)` |
| `grover_oracle_search` | method | `quantum_computer.py:1113` | `def grover_oracle_search(self, n_qubits, target_bitstring, backend, n_iterations)` |
| `h` | method | `quantum_computer.py:838` | `def h(self, q)` |
| `harmonic` | method | `quantum_computer.py:242` | `def harmonic(self)` |
| `marginal_probability_one` | method | `quantum_computer.py:168` | `def marginal_probability_one(self, qubit)` |
| `mixed` | method | `quantum_computer.py:269` | `def mixed(self, seed)` |
| `most_probable_basis_state` | method | `quantum_computer.py:183` | `def most_probable_basis_state(self)` |
| `most_probable_bitstring` | method | `quantum_computer.py:929` | `def most_probable_bitstring(self)` |
| `name` | method | `quantum_computer.py:471` | `def name(self)` |
| `name` | method | `quantum_computer.py:489` | `def name(self)` |
| `name` | method | `quantum_computer.py:504` | `def name(self)` |
| `name` | method | `quantum_computer.py:518` | `def name(self)` |
| `name` | method | `quantum_computer.py:532` | `def name(self)` |
| `name` | method | `quantum_computer.py:546` | `def name(self)` |
| `name` | method | `quantum_computer.py:560` | `def name(self)` |
| `name` | method | `quantum_computer.py:575` | `def name(self)` |
| `name` | method | `quantum_computer.py:591` | `def name(self)` |
| `name` | method | `quantum_computer.py:607` | `def name(self)` |
| `name` | method | `quantum_computer.py:629` | `def name(self)` |
| `name` | method | `quantum_computer.py:652` | `def name(self)` |
| `name` | method | `quantum_computer.py:675` | `def name(self)` |
| `name` | method | `quantum_computer.py:698` | `def name(self)` |
| `name` | method | `quantum_computer.py:736` | `def name(self)` |
| `name` | method | `quantum_computer.py:770` | `def name(self)` |
| `normalize_` | method | `quantum_computer.py:157` | `def normalize_(self)` |
| `periodic_lattice` | method | `quantum_computer.py:264` | `def periodic_lattice(self)` |
| `probabilities` | method | `quantum_computer.py:163` | `def probabilities(self)` |
| `probabilities` | method | `quantum_computer.py:925` | `def probabilities(self)` |
| `quantum_fourier_transform` | method | `quantum_computer.py:1103` | `def quantum_fourier_transform(self, n_qubits, backend)` |
| `register_gate` | function | `quantum_computer.py:811` | `def register_gate(name, gate)` |
| `run` | method | `quantum_computer.py:1020` | `def run(self, circuit, backend, initial_states)` |
| `run_phase_tests` | function | `quantum_computer.py:1245` | `def run_phase_tests(config)` |
| `run_with_state_snapshots` | method | `quantum_computer.py:1052` | `def run_with_state_snapshots(self, circuit, backend, snapshot_after)` |
| `rx` | method | `quantum_computer.py:856` | `def rx(self, q, theta)` |
| `ry` | method | `quantum_computer.py:859` | `def ry(self, q, theta)` |
| `rz` | method | `quantum_computer.py:862` | `def rz(self, q, theta)` |
| `s` | method | `quantum_computer.py:850` | `def s(self, q)` |
| `swap` | method | `quantum_computer.py:874` | `def swap(self, a, b)` |
| `t` | method | `quantum_computer.py:853` | `def t(self, q)` |
| `teleportation` | method | `quantum_computer.py:1204` | `def teleportation(self, backend)` |
| `toffoli` | method | `quantum_computer.py:877` | `def toffoli(self, c0, c1, tgt)` |
| `variational_ansatz` | method | `quantum_computer.py:1191` | `def variational_ansatz(self, n_qubits, n_layers, thetas, backend)` |
| `x` | method | `quantum_computer.py:841` | `def x(self, q)` |
| `y` | method | `quantum_computer.py:844` | `def y(self, q)` |
| `z` | method | `quantum_computer.py:847` | `def z(self, q)` |
| `BackendComparison` | class | `quantum_dash.py:255` | `class BackendComparison` |
| `BackendComparisonVisualizer` | class | `quantum_dash.py:583` | `class BackendComparisonVisualizer(IVisualComponent)` |
| `BlochSphereVisualizer` | class | `quantum_dash.py:340` | `class BlochSphereVisualizer(IVisualComponent)` |
| `BrutalistConfig` | class | `quantum_dash.py:115` | `class BrutalistConfig` |
| `CircuitExecutor` | class | `quantum_dash.py:806` | `class CircuitExecutor` |
| `ColorScheme` | class | `quantum_dash.py:106` | `class ColorScheme(Enum)` |
| `EntropyVisualizer` | class | `quantum_dash.py:509` | `class EntropyVisualizer(IVisualComponent)` |
| `FidelityVisualizer` | class | `quantum_dash.py:633` | `class FidelityVisualizer(IVisualComponent)` |
| `FigureBuilder` | class | `quantum_dash.py:873` | `class FigureBuilder` |
| `IVisualComponent` | class | `quantum_dash.py:275` | `class IVisualComponent(ABC)` |
| `PhaseSpaceVisualizer` | class | `quantum_dash.py:447` | `class PhaseSpaceVisualizer(IVisualComponent)` |
| `ProbabilityVisualizer` | class | `quantum_dash.py:281` | `class ProbabilityVisualizer(IVisualComponent)` |
| `QuantumSnapshot` | class | `quantum_dash.py:239` | `class QuantumSnapshot` |
| `QuantumStateAnalyzer` | class | `quantum_dash.py:674` | `class QuantumStateAnalyzer` |
| `QuantumVisualizer` | class | `quantum_dash.py:958` | `class QuantumVisualizer` |
| `StandardCircuits` | class | `quantum_dash.py:754` | `class StandardCircuits` |
| `VisualizationOutput` | class | `quantum_dash.py:268` | `class VisualizationOutput` |
| `__init__` | method | `quantum_dash.py:675` | `def __init__(self, config)` |
| `__init__` | method | `quantum_dash.py:807` | `def __init__(self, qc, config)` |
| `__init__` | method | `quantum_dash.py:874` | `def __init__(self, config)` |
| `__init__` | method | `quantum_dash.py:959` | `def __init__(self, config)` |
| `_create_synthetic_comparisons` | method | `quantum_dash.py:1129` | `def _create_synthetic_comparisons(self, n_qubits, gates)` |
| `_create_synthetic_snapshots` | method | `quantum_dash.py:1086` | `def _create_synthetic_snapshots(self, n_qubits, gates)` |
| `_draw_axes` | method | `quantum_dash.py:402` | `def _draw_axes(self, axes, config)` |
| `_draw_bloch_vector` | method | `quantum_dash.py:417` | `def _draw_bloch_vector(self, axes, bx, by, bz, color, qubit_idx, config)` |
| `_draw_sphere_wireframe` | method | `quantum_dash.py:381` | `def _draw_sphere_wireframe(self, axes, config)` |
| `_draw_uncertainty_ring` | method | `quantum_dash.py:431` | `def _draw_uncertainty_ring(self, axes, bx, by, bz, color, config)` |
| `_execute_and_visualize` | method | `quantum_dash.py:1043` | `def _execute_and_visualize(self, gates, n_qubits, name)` |
| `_generate_colors` | method | `quantum_dash.py:326` | `def _generate_colors(self, probs, config)` |
| `_hex_to_rgb` | method | `quantum_dash.py:578` | `def _hex_to_rgb(self, hex_color)` |
| `_init_quantum_computer` | method | `quantum_dash.py:976` | `def _init_quantum_computer(self)` |
| `_initialize` | method | `quantum_dash.py:966` | `def _initialize(self)` |
| `_interpolate_colors` | method | `quantum_dash.py:564` | `def _interpolate_colors(self, color1, color2, n)` |
| `_make_logger` | function | `quantum_dash.py:91` | `def _make_logger(name)` |
| `_print_summary` | method | `quantum_dash.py:1189` | `def _print_summary(self, results)` |
| `_render_empty` | method | `quantum_dash.py:318` | `def _render_empty(self, axes, config)` |
| `_render_empty` | method | `quantum_dash.py:495` | `def _render_empty(self, axes, config)` |
| `_render_empty` | method | `quantum_dash.py:556` | `def _render_empty(self, axes, config)` |
| `_render_empty` | method | `quantum_dash.py:624` | `def _render_empty(self, axes, config)` |
| `_render_empty` | method | `quantum_dash.py:665` | `def _render_empty(self, axes, config)` |
| `_render_empty_sphere` | method | `quantum_dash.py:375` | `def _render_empty_sphere(self, axes, config)` |
| `_save_data` | method | `quantum_dash.py:1153` | `def _save_data(self, path, snapshots, comparisons)` |
| `bell_state` | method | `quantum_dash.py:756` | `def bell_state()` |
| `build_full_figure` | method | `quantum_dash.py:883` | `def build_full_figure(self, snapshots, comparisons)` |
| `build_summary_figure` | method | `quantum_dash.py:923` | `def build_summary_figure(self, snapshots, comparisons)` |
| `colors` | method | `quantum_dash.py:168` | `def colors(self)` |
| `compare_backends` | method | `quantum_dash.py:835` | `def compare_backends(self, gates, n_qubits)` |
| `compute_bloch_vectors` | method | `quantum_dash.py:704` | `def compute_bloch_vectors(self, state)` |
| `compute_entropy` | method | `quantum_dash.py:696` | `def compute_entropy(self, probs)` |
| `compute_phases` | method | `quantum_dash.py:684` | `def compute_phases(self, state)` |
| `compute_probabilities` | method | `quantum_dash.py:678` | `def compute_probabilities(self, state)` |
| `create_snapshot` | method | `quantum_dash.py:713` | `def create_snapshot(self, state, step, gate_name, backend_name)` |
| `execute_sequence` | method | `quantum_dash.py:812` | `def execute_sequence(self, gates, n_qubits, backend_name)` |
| `ghz_state` | method | `quantum_dash.py:760` | `def ghz_state(n_qubits)` |
| `grover_diffusion` | method | `quantum_dash.py:794` | `def grover_diffusion(n_qubits)` |
| `grover_oracle` | method | `quantum_dash.py:782` | `def grover_oracle(n_qubits, marked)` |
| `main` | function | `quantum_dash.py:1199` | `def main()` |
| `plotly_template` | method | `quantum_dash.py:234` | `def plotly_template(self)` |
| `qft` | method | `quantum_dash.py:767` | `def qft(n_qubits)` |
| `render` | method | `quantum_dash.py:277` | `def render(self, data, axes, config)` |
| `render` | method | `quantum_dash.py:282` | `def render(self, snapshot, axes, config)` |
| `render` | method | `quantum_dash.py:341` | `def render(self, snapshot, axes, config)` |
| `render` | method | `quantum_dash.py:448` | `def render(self, snapshot, axes, config)` |
| `render` | method | `quantum_dash.py:510` | `def render(self, snapshots, axes, config)` |
| `render` | method | `quantum_dash.py:584` | `def render(self, comparisons, axes, config)` |
| `render` | method | `quantum_dash.py:634` | `def render(self, comparisons, axes, config)` |
| `run_all` | method | `quantum_dash.py:1174` | `def run_all(self)` |
| `visualize_bell_state` | method | `quantum_dash.py:1010` | `def visualize_bell_state(self)` |
| `visualize_ghz_state` | method | `quantum_dash.py:1016` | `def visualize_ghz_state(self, n_qubits)` |
| `visualize_grover` | method | `quantum_dash.py:1028` | `def visualize_grover(self, n_qubits, marked_state)` |
| `visualize_qft` | method | `quantum_dash.py:1022` | `def visualize_qft(self, n_qubits)` |
| `AtomData` | class | `quantum_framework_core.py:233` | `class AtomData` |
| `CNOTGate` | class | `quantum_framework_core.py:1299` | `class CNOTGate(IQuantumGate)` |
| `CRzGate` | class | `quantum_framework_core.py:1274` | `class CRzGate(IQuantumGate)` |
| `CZGate` | class | `quantum_framework_core.py:1319` | `class CZGate(IQuantumGate)` |
| `CircuitInstruction` | class | `quantum_framework_core.py:1377` | `class CircuitInstruction` |
| `ConfigLoader` | class | `quantum_framework_core.py:274` | `class ConfigLoader` |
| `FrameworkConfig` | class | `quantum_framework_core.py:85` | `class FrameworkConfig` |
| `HadamardGate` | class | `quantum_framework_core.py:1139` | `class HadamardGate(IQuantumGate)` |
| `HilbertPhase` | class | `quantum_framework_core.py:75` | `class HilbertPhase(Enum)` |
| `IQuantumGate` | class | `quantum_framework_core.py:1119` | `class IQuantumGate(ABC)` |
| `ITensorNetwork` | class | `quantum_framework_core.py:450` | `class ITensorNetwork(ABC)` |
| `MPSCore` | class | `quantum_framework_core.py:495` | `class MPSCore` |
| `MPSQuantumComputer` | class | `quantum_framework_core.py:1459` | `class MPSQuantumComputer` |
| `MPSState` | class | `quantum_framework_core.py:577` | `class MPSState(ITensorNetwork)` |
| `MoleculeData` | class | `quantum_framework_core.py:245` | `class MoleculeData` |
| `OrbitalData` | class | `quantum_framework_core.py:265` | `class OrbitalData` |
| `PauliXGate` | class | `quantum_framework_core.py:1154` | `class PauliXGate(IQuantumGate)` |
| `PauliYGate` | class | `quantum_framework_core.py:1168` | `class PauliYGate(IQuantumGate)` |
| `PauliZGate` | class | `quantum_framework_core.py:1182` | `class PauliZGate(IQuantumGate)` |
| `QuantumCircuit` | class | `quantum_framework_core.py:1384` | `class QuantumCircuit` |
| `RxGate` | class | `quantum_framework_core.py:1225` | `class RxGate(IQuantumGate)` |
| `RyGate` | class | `quantum_framework_core.py:1241` | `class RyGate(IQuantumGate)` |
| `RzGate` | class | `quantum_framework_core.py:1257` | `class RzGate(IQuantumGate)` |
| `SGate` | class | `quantum_framework_core.py:1196` | `class SGate(IQuantumGate)` |
| `SWAPGate` | class | `quantum_framework_core.py:1339` | `class SWAPGate(IQuantumGate)` |
| `TGate` | class | `quantum_framework_core.py:1210` | `class TGate(IQuantumGate)` |
| `TopologicalProtector` | class | `quantum_framework_core.py:1067` | `class TopologicalProtector` |
| `VacuumCore` | class | `quantum_framework_core.py:995` | `class VacuumCore` |
| `__bool__` | method | `quantum_framework_core.py:1445` | `def __bool__(self)` |
| `__init__` | method | `quantum_framework_core.py:280` | `def __init__(self, config_path)` |
| `__init__` | method | `quantum_framework_core.py:503` | `def __init__(self, chi_left, chi_right, d, device, dtype)` |
| `__init__` | method | `quantum_framework_core.py:590` | `def __init__(self, n_qubits, config)` |
| `__init__` | method | `quantum_framework_core.py:1003` | `def __init__(self, n_qubits, config)` |
| `__init__` | method | `quantum_framework_core.py:1077` | `def __init__(self, config)` |
| `__init__` | method | `quantum_framework_core.py:1387` | `def __init__(self, n_qubits)` |
| `__init__` | method | `quantum_framework_core.py:1470` | `def __init__(self, config)` |
| `__len__` | method | `quantum_framework_core.py:1441` | `def __len__(self)` |
| `__post_init__` | method | `quantum_framework_core.py:153` | `def __post_init__(self)` |
| `_append` | method | `quantum_framework_core.py:1391` | `def _append(self, gate_name, targets, params)` |
| `_apply_adjacent_gate` | method | `quantum_framework_core.py:703` | `def _apply_adjacent_gate(self, qubit, gate)` |
| `_apply_nonadjacent_gate` | method | `quantum_framework_core.py:787` | `def _apply_nonadjacent_gate(self, qubit_a, qubit_b, gate)` |
| `_build_w_state_direct` | method | `quantum_framework_core.py:1548` | `def _build_w_state_direct(self, n_qubits, max_bond)` |
| `_canonicalize` | method | `quantum_framework_core.py:818` | `def _canonicalize(self)` |
| `_canonicalize_to` | method | `quantum_framework_core.py:822` | `def _canonicalize_to(self, center)` |
| `_compute_average_bond_dimension` | method | `quantum_framework_core.py:1683` | `def _compute_average_bond_dimension(self, state)` |
| `_compute_berry_phases` | method | `quantum_framework_core.py:1020` | `def _compute_berry_phases(self)` |
| `_compute_winding_number` | method | `quantum_framework_core.py:1041` | `def _compute_winding_number(self, basis_index)` |

Next: [SYMBOLS_p3.md](SYMBOLS_p3.md)
