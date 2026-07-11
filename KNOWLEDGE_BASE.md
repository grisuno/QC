# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 30 | **Total Symbols Extracted:** 1790 | **Total Imports:** 512

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
    qc_dashboard_py["qc_dashboard.py (py)"]
    class qc_dashboard_py mod;
    qc_dashboard_py_DashboardConfig["DashboardConfig"]
    class qc_dashboard_py_DashboardConfig cls;
    qc_dashboard_py --> qc_dashboard_py_DashboardConfig
    qc_dashboard_py_GateItem["GateItem"]
    class qc_dashboard_py_GateItem cls;
    qc_dashboard_py --> qc_dashboard_py_GateItem
    qc_dashboard_py_SnapshotData["SnapshotData"]
    class qc_dashboard_py_SnapshotData cls;
    qc_dashboard_py --> qc_dashboard_py_SnapshotData
    qc_dashboard_py_H2VQESolver["H2VQESolver"]
    class qc_dashboard_py_H2VQESolver cls;
    qc_dashboard_py --> qc_dashboard_py_H2VQESolver
    qc_dashboard_py_VisualisationEngine["VisualisationEngine"]
    class qc_dashboard_py_VisualisationEngine cls;
    qc_dashboard_py --> qc_dashboard_py_VisualisationEngine
    quantum_framework_menu_py["quantum_framework_menu.py (py)"]
    class quantum_framework_menu_py mod;
    quantum_framework_menu_py_MenuSystem["MenuSystem"]
    class quantum_framework_menu_py_MenuSystem cls;
    quantum_framework_menu_py --> quantum_framework_menu_py_MenuSystem
    quantum_framework_menu_py_run_interactive_menu["run_interactive_menu"]
    class quantum_framework_menu_py_run_interactive_menu fn;
    quantum_framework_menu_py --> quantum_framework_menu_py_run_interactive_menu
    quantum_framework_menu_py_run_all_experiments["run_all_experiments"]
    class quantum_framework_menu_py_run_all_experiments fn;
    quantum_framework_menu_py --> quantum_framework_menu_py_run_all_experiments
    quantum_framework_menu_py___init__["__init__"]
    class quantum_framework_menu_py___init__ fn;
    quantum_framework_menu_py --> quantum_framework_menu_py___init__
    quantum_framework_menu_py_clear_screen["clear_screen"]
    class quantum_framework_menu_py_clear_screen fn;
    quantum_framework_menu_py --> quantum_framework_menu_py_clear_screen
    quantum_dash_py["quantum_dash.py (py)"]
    class quantum_dash_py mod;
    quantum_dash_py__make_logger["_make_logger"]
    class quantum_dash_py__make_logger fn;
    quantum_dash_py --> quantum_dash_py__make_logger
    quantum_dash_py_ColorScheme["ColorScheme"]
    class quantum_dash_py_ColorScheme cls;
    quantum_dash_py --> quantum_dash_py_ColorScheme
    quantum_dash_py_BrutalistConfig["BrutalistConfig"]
    class quantum_dash_py_BrutalistConfig cls;
    quantum_dash_py --> quantum_dash_py_BrutalistConfig
    quantum_dash_py_QuantumSnapshot["QuantumSnapshot"]
    class quantum_dash_py_QuantumSnapshot cls;
    quantum_dash_py --> quantum_dash_py_QuantumSnapshot
    quantum_dash_py_BackendComparison["BackendComparison"]
    class quantum_dash_py_BackendComparison cls;
    quantum_dash_py --> quantum_dash_py_BackendComparison
    test_qc_integration_py["test_qc_integration.py (py)"]
    class test_qc_integration_py mod;
    test_qc_integration_py_config["config"]
    class test_qc_integration_py_config fn;
    test_qc_integration_py --> test_qc_integration_py_config
    test_qc_integration_py_bridge["bridge"]
    class test_qc_integration_py_bridge fn;
    test_qc_integration_py --> test_qc_integration_py_bridge
    test_qc_integration_py_qasm_adapter["qasm_adapter"]
    class test_qc_integration_py_qasm_adapter fn;
    test_qc_integration_py --> test_qc_integration_py_qasm_adapter
    test_qc_integration_py_bell_circuit["bell_circuit"]
    class test_qc_integration_py_bell_circuit fn;
    test_qc_integration_py --> test_qc_integration_py_bell_circuit
    test_qc_integration_py_ghz_circuit["ghz_circuit"]
    class test_qc_integration_py_ghz_circuit fn;
    test_qc_integration_py --> test_qc_integration_py_ghz_circuit
    quantum_lab_py["quantum_lab.py (py)"]
    class quantum_lab_py mod;
    quantum_lab_py_probability_bars["probability_bars"]
    class quantum_lab_py_probability_bars fn;
    quantum_lab_py --> quantum_lab_py_probability_bars
    quantum_lab_py_counts_bars["counts_bars"]
    class quantum_lab_py_counts_bars fn;
    quantum_lab_py --> quantum_lab_py_counts_bars
    quantum_lab_py_draw_circuit["draw_circuit"]
    class quantum_lab_py_draw_circuit fn;
    quantum_lab_py --> quantum_lab_py_draw_circuit
    quantum_lab_py_parse_angle["parse_angle"]
    class quantum_lab_py_parse_angle fn;
    quantum_lab_py --> quantum_lab_py_parse_angle
    quantum_lab_py_sample_measurements["sample_measurements"]
    class quantum_lab_py_sample_measurements fn;
    quantum_lab_py --> quantum_lab_py_sample_measurements
    quantum_visualizer_py["quantum_visualizer.py (py)"]
    class quantum_visualizer_py mod;
    quantum_visualizer_py__make_logger["_make_logger"]
    class quantum_visualizer_py__make_logger fn;
    quantum_visualizer_py --> quantum_visualizer_py__make_logger
    quantum_visualizer_py_VisualizerConfig["VisualizerConfig"]
    class quantum_visualizer_py_VisualizerConfig cls;
    quantum_visualizer_py --> quantum_visualizer_py_VisualizerConfig
    quantum_visualizer_py_QuantumStateSnapshot["QuantumStateSnapshot"]
    class quantum_visualizer_py_QuantumStateSnapshot cls;
    quantum_visualizer_py --> quantum_visualizer_py_QuantumStateSnapshot
    quantum_visualizer_py_BackendComparisonResult["BackendComparisonResult"]
    class quantum_visualizer_py_BackendComparisonResult cls;
    quantum_visualizer_py --> quantum_visualizer_py_BackendComparisonResult
    quantum_visualizer_py_VisualizationResult["VisualizationResult"]
    class quantum_visualizer_py_VisualizationResult cls;
    quantum_visualizer_py --> quantum_visualizer_py_VisualizationResult
    quantum_framework_molecular_v2_py["quantum_framework_molecular_v2.py (py)"]
    class quantum_framework_molecular_v2_py mod;
    quantum_framework_molecular_v2_py__make_logger["_make_logger"]
    class quantum_framework_molecular_v2_py__make_logger fn;
    quantum_framework_molecular_v2_py --> quantum_framework_molecular_v2_py__make_logger
    quantum_framework_molecular_v2_py_MolecularConfig["MolecularConfig"]
    class quantum_framework_molecular_v2_py_MolecularConfig cls;
    quantum_framework_molecular_v2_py --> quantum_framework_molecular_v2_py_MolecularConfig
    quantum_framework_molecular_v2_py_MoleculeData["MoleculeData"]
    class quantum_framework_molecular_v2_py_MoleculeData cls;
    quantum_framework_molecular_v2_py --> quantum_framework_molecular_v2_py_MoleculeData
    quantum_framework_molecular_v2_py_MoleculeBuilder["MoleculeBuilder"]
    class quantum_framework_molecular_v2_py_MoleculeBuilder cls;
    quantum_framework_molecular_v2_py --> quantum_framework_molecular_v2_py_MoleculeBuilder
    quantum_framework_molecular_v2_py_HamiltonianBuilder["HamiltonianBuilder"]
    class quantum_framework_molecular_v2_py_HamiltonianBuilder cls;
    quantum_framework_molecular_v2_py --> quantum_framework_molecular_v2_py_HamiltonianBuilder
    entangled_hydrogen_py["entangled_hydrogen.py (py)"]
    class entangled_hydrogen_py mod;
    entangled_hydrogen_py__make_logger["_make_logger"]
    class entangled_hydrogen_py__make_logger fn;
    entangled_hydrogen_py --> entangled_hydrogen_py__make_logger
    entangled_hydrogen_py_EntangledHydrogenConfig["EntangledHydrogenConfig"]
    class entangled_hydrogen_py_EntangledHydrogenConfig cls;
    entangled_hydrogen_py --> entangled_hydrogen_py_EntangledHydrogenConfig
    entangled_hydrogen_py_IEntangledState["IEntangledState"]
    class entangled_hydrogen_py_IEntangledState cls;
    entangled_hydrogen_py --> entangled_hydrogen_py_IEntangledState
    entangled_hydrogen_py_BellState["BellState"]
    class entangled_hydrogen_py_BellState cls;
    entangled_hydrogen_py --> entangled_hydrogen_py_BellState
    entangled_hydrogen_py_GHZState["GHZState"]
    class entangled_hydrogen_py_GHZState cls;
    entangled_hydrogen_py --> entangled_hydrogen_py_GHZState
    relativistic_hydrogen_py["relativistic_hydrogen.py (py)"]
    class relativistic_hydrogen_py mod;
    relativistic_hydrogen_py_Config["Config"]
    class relativistic_hydrogen_py_Config cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_Config
    relativistic_hydrogen_py_LoggerFactory["LoggerFactory"]
    class relativistic_hydrogen_py_LoggerFactory cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_LoggerFactory
    relativistic_hydrogen_py_GammaMatrices["GammaMatrices"]
    class relativistic_hydrogen_py_GammaMatrices cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_GammaMatrices
    relativistic_hydrogen_py_DiracHamiltonianOperator["DiracHamiltonianOperator"]
    class relativistic_hydrogen_py_DiracHamiltonianOperator cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_DiracHamiltonianOperator
    relativistic_hydrogen_py_SpectralLayer["SpectralLayer"]
    class relativistic_hydrogen_py_SpectralLayer cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_SpectralLayer
    molecular_sim_py["molecular_sim.py (py)"]
    class molecular_sim_py mod;
    molecular_sim_py__make_logger["_make_logger"]
    class molecular_sim_py__make_logger fn;
    molecular_sim_py --> molecular_sim_py__make_logger
    molecular_sim_py_MoleculeData["MoleculeData"]
    class molecular_sim_py_MoleculeData cls;
    molecular_sim_py --> molecular_sim_py_MoleculeData
    molecular_sim_py__h2_sto3g_pyscf["_h2_sto3g_pyscf"]
    class molecular_sim_py__h2_sto3g_pyscf fn;
    molecular_sim_py --> molecular_sim_py__h2_sto3g_pyscf
    molecular_sim_py__h2_sto3g_hardcoded["_h2_sto3g_hardcoded"]
    class molecular_sim_py__h2_sto3g_hardcoded fn;
    molecular_sim_py --> molecular_sim_py__h2_sto3g_hardcoded
    molecular_sim_py_build_jw_hamiltonian_of["build_jw_hamiltonian_of"]
    class molecular_sim_py_build_jw_hamiltonian_of fn;
    molecular_sim_py --> molecular_sim_py_build_jw_hamiltonian_of
    advanced_experiments_py["advanced_experiments.py (py)"]
    class advanced_experiments_py mod;
    advanced_experiments_py__make_logger["_make_logger"]
    class advanced_experiments_py__make_logger fn;
    advanced_experiments_py --> advanced_experiments_py__make_logger
    advanced_experiments_py_GroverConfig["GroverConfig"]
    class advanced_experiments_py_GroverConfig cls;
    advanced_experiments_py --> advanced_experiments_py_GroverConfig
    advanced_experiments_py_GroverOracle["GroverOracle"]
    class advanced_experiments_py_GroverOracle cls;
    advanced_experiments_py --> advanced_experiments_py_GroverOracle
    advanced_experiments_py_GroverDiffusionOperator["GroverDiffusionOperator"]
    class advanced_experiments_py_GroverDiffusionOperator cls;
    advanced_experiments_py --> advanced_experiments_py_GroverDiffusionOperator
    advanced_experiments_py_GroverSearch["GroverSearch"]
    class advanced_experiments_py_GroverSearch cls;
    advanced_experiments_py --> advanced_experiments_py_GroverSearch
    topological_hilbert_compression2_py["topological_hilbert_compression2.py (py)"]
    class topological_hilbert_compression2_py mod;
    topological_hilbert_compression2_py_HilbertPhase["HilbertPhase"]
    class topological_hilbert_compression2_py_HilbertPhase cls;
    topological_hilbert_compression2_py --> topological_hilbert_compression2_py_HilbertPhase
    topological_hilbert_compression2_py_TopologicalCompressionConfig["TopologicalCompressionConfig"]
    class topological_hilbert_compression2_py_TopologicalCompressionConfig cls;
    topological_hilbert_compression2_py --> topological_hilbert_compression2_py_TopologicalCompressionConfig
    topological_hilbert_compression2_py_ITensorNetwork["ITensorNetwork"]
    class topological_hilbert_compression2_py_ITensorNetwork cls;
    topological_hilbert_compression2_py --> topological_hilbert_compression2_py_ITensorNetwork
    topological_hilbert_compression2_py_MPSCore["MPSCore"]
    class topological_hilbert_compression2_py_MPSCore cls;
    topological_hilbert_compression2_py --> topological_hilbert_compression2_py_MPSCore
    topological_hilbert_compression2_py_MPSState["MPSState"]
    class topological_hilbert_compression2_py_MPSState cls;
    topological_hilbert_compression2_py --> topological_hilbert_compression2_py_MPSState
    higgs_four_lepton_analysis_py["higgs_four_lepton_analysis.py (py)"]
    class higgs_four_lepton_analysis_py mod;
    higgs_four_lepton_analysis_py__make_logger["_make_logger"]
    class higgs_four_lepton_analysis_py__make_logger fn;
    higgs_four_lepton_analysis_py --> higgs_four_lepton_analysis_py__make_logger
    higgs_four_lepton_analysis_py_LeptonType["LeptonType"]
    class higgs_four_lepton_analysis_py_LeptonType cls;
    higgs_four_lepton_analysis_py --> higgs_four_lepton_analysis_py_LeptonType
    higgs_four_lepton_analysis_py_EventType["EventType"]
    class higgs_four_lepton_analysis_py_EventType cls;
    higgs_four_lepton_analysis_py --> higgs_four_lepton_analysis_py_EventType
    higgs_four_lepton_analysis_py_Config["Config"]
    class higgs_four_lepton_analysis_py_Config cls;
    higgs_four_lepton_analysis_py --> higgs_four_lepton_analysis_py_Config
    higgs_four_lepton_analysis_py_FourMomentum["FourMomentum"]
    class higgs_four_lepton_analysis_py_FourMomentum cls;
    higgs_four_lepton_analysis_py --> higgs_four_lepton_analysis_py_FourMomentum
    higgs_quantum_analysis_py["higgs_quantum_analysis.py (py)"]
    class higgs_quantum_analysis_py mod;
    higgs_quantum_analysis_py__make_logger["_make_logger"]
    class higgs_quantum_analysis_py__make_logger fn;
    higgs_quantum_analysis_py --> higgs_quantum_analysis_py__make_logger
    higgs_quantum_analysis_py_LeptonType["LeptonType"]
    class higgs_quantum_analysis_py_LeptonType cls;
    higgs_quantum_analysis_py --> higgs_quantum_analysis_py_LeptonType
    higgs_quantum_analysis_py_EventType["EventType"]
    class higgs_quantum_analysis_py_EventType cls;
    higgs_quantum_analysis_py --> higgs_quantum_analysis_py_EventType
    higgs_quantum_analysis_py_Config["Config"]
    class higgs_quantum_analysis_py_Config cls;
    higgs_quantum_analysis_py --> higgs_quantum_analysis_py_Config
    higgs_quantum_analysis_py_FourMomentum["FourMomentum"]
    class higgs_quantum_analysis_py_FourMomentum cls;
    higgs_quantum_analysis_py --> higgs_quantum_analysis_py_FourMomentum
    quantum_simulator_py["quantum_simulator.py (py)"]
    class quantum_simulator_py mod;
    quantum_simulator_py__make_logger["_make_logger"]
    class quantum_simulator_py__make_logger fn;
    quantum_simulator_py --> quantum_simulator_py__make_logger
    quantum_simulator_py_FrameworkConfig["FrameworkConfig"]
    class quantum_simulator_py_FrameworkConfig cls;
    quantum_simulator_py --> quantum_simulator_py_FrameworkConfig
    quantum_simulator_py_AtomData["AtomData"]
    class quantum_simulator_py_AtomData cls;
    quantum_simulator_py --> quantum_simulator_py_AtomData
    quantum_simulator_py_MoleculeData["MoleculeData"]
    class quantum_simulator_py_MoleculeData cls;
    quantum_simulator_py --> quantum_simulator_py_MoleculeData
    quantum_simulator_py_OrbitalData["OrbitalData"]
    class quantum_simulator_py_OrbitalData cls;
    quantum_simulator_py --> quantum_simulator_py_OrbitalData
    quantum_framework_core_py["quantum_framework_core.py (py)"]
    class quantum_framework_core_py mod;
    quantum_framework_core_py__make_logger["_make_logger"]
    class quantum_framework_core_py__make_logger fn;
    quantum_framework_core_py --> quantum_framework_core_py__make_logger
    quantum_framework_core_py_HilbertPhase["HilbertPhase"]
    class quantum_framework_core_py_HilbertPhase cls;
    quantum_framework_core_py --> quantum_framework_core_py_HilbertPhase
    quantum_framework_core_py_FrameworkConfig["FrameworkConfig"]
    class quantum_framework_core_py_FrameworkConfig cls;
    quantum_framework_core_py --> quantum_framework_core_py_FrameworkConfig
    quantum_framework_core_py_AtomData["AtomData"]
    class quantum_framework_core_py_AtomData cls;
    quantum_framework_core_py --> quantum_framework_core_py_AtomData
    quantum_framework_core_py_MoleculeData["MoleculeData"]
    class quantum_framework_core_py_MoleculeData cls;
    quantum_framework_core_py --> quantum_framework_core_py_MoleculeData
    quantum_framework_molecular_fixed_py["quantum_framework_molecular_fixed.py (py)"]
    class quantum_framework_molecular_fixed_py mod;
    quantum_framework_molecular_fixed_py__make_logger["_make_logger"]
    class quantum_framework_molecular_fixed_py__make_logger fn;
    quantum_framework_molecular_fixed_py --> quantum_framework_molecular_fixed_py__make_logger
    quantum_framework_molecular_fixed_py_MoleculeData["MoleculeData"]
    class quantum_framework_molecular_fixed_py_MoleculeData cls;
    quantum_framework_molecular_fixed_py --> quantum_framework_molecular_fixed_py_MoleculeData
    quantum_framework_molecular_fixed_py_MoleculeBuilder["MoleculeBuilder"]
    class quantum_framework_molecular_fixed_py_MoleculeBuilder cls;
    quantum_framework_molecular_fixed_py --> quantum_framework_molecular_fixed_py_MoleculeBuilder
    quantum_framework_molecular_fixed_py_ExactJWEnergy["ExactJWEnergy"]
    class quantum_framework_molecular_fixed_py_ExactJWEnergy cls;
    quantum_framework_molecular_fixed_py --> quantum_framework_molecular_fixed_py_ExactJWEnergy
    quantum_framework_molecular_fixed_py__get_sd_indices["_get_sd_indices"]
    class quantum_framework_molecular_fixed_py__get_sd_indices fn;
    quantum_framework_molecular_fixed_py --> quantum_framework_molecular_fixed_py__get_sd_indices
    quantum_3dview_py["quantum_3dview.py (py)"]
    class quantum_3dview_py mod;
    quantum_3dview_py_BrutalTheme["BrutalTheme"]
    class quantum_3dview_py_BrutalTheme cls;
    quantum_3dview_py --> quantum_3dview_py_BrutalTheme
    quantum_3dview_py_BrutalConfig["BrutalConfig"]
    class quantum_3dview_py_BrutalConfig cls;
    quantum_3dview_py --> quantum_3dview_py_BrutalConfig
    quantum_3dview_py_QuantumHologram["QuantumHologram"]
    class quantum_3dview_py_QuantumHologram cls;
    quantum_3dview_py --> quantum_3dview_py_QuantumHologram
    quantum_3dview_py_QuantumNeuralTopology["QuantumNeuralTopology"]
    class quantum_3dview_py_QuantumNeuralTopology cls;
    quantum_3dview_py --> quantum_3dview_py_QuantumNeuralTopology
    quantum_3dview_py_QuantumSonification["QuantumSonification"]
    class quantum_3dview_py_QuantumSonification cls;
    quantum_3dview_py --> quantum_3dview_py_QuantumSonification
    quantum_framework_molecular_py["quantum_framework_molecular.py (py)"]
    class quantum_framework_molecular_py mod;
    quantum_framework_molecular_py__make_logger["_make_logger"]
    class quantum_framework_molecular_py__make_logger fn;
    quantum_framework_molecular_py --> quantum_framework_molecular_py__make_logger
    quantum_framework_molecular_py_MoleculeData["MoleculeData"]
    class quantum_framework_molecular_py_MoleculeData cls;
    quantum_framework_molecular_py --> quantum_framework_molecular_py_MoleculeData
    quantum_framework_molecular_py_MoleculeBuilder["MoleculeBuilder"]
    class quantum_framework_molecular_py_MoleculeBuilder cls;
    quantum_framework_molecular_py --> quantum_framework_molecular_py_MoleculeBuilder
    quantum_framework_molecular_py_ExactJWEnergy["ExactJWEnergy"]
    class quantum_framework_molecular_py_ExactJWEnergy cls;
    quantum_framework_molecular_py --> quantum_framework_molecular_py_ExactJWEnergy
    quantum_framework_molecular_py_UCCSDAnsatz["UCCSDAnsatz"]
    class quantum_framework_molecular_py_UCCSDAnsatz cls;
    quantum_framework_molecular_py --> quantum_framework_molecular_py_UCCSDAnsatz
    qc_integration_py["qc_integration.py (py)"]
    class qc_integration_py mod;
    qc_integration_py_IntegrationConfig["IntegrationConfig"]
    class qc_integration_py_IntegrationConfig cls;
    qc_integration_py --> qc_integration_py_IntegrationConfig
    qc_integration_py_GateInstruction["GateInstruction"]
    class qc_integration_py_GateInstruction cls;
    qc_integration_py --> qc_integration_py_GateInstruction
    qc_integration_py_CircuitIR["CircuitIR"]
    class qc_integration_py_CircuitIR cls;
    qc_integration_py --> qc_integration_py_CircuitIR
    qc_integration_py_IQCAdapter["IQCAdapter"]
    class qc_integration_py_IQCAdapter cls;
    qc_integration_py --> qc_integration_py_IQCAdapter
    qc_integration_py_OpenQasmAdapter["OpenQasmAdapter"]
    class qc_integration_py_OpenQasmAdapter cls;
    qc_integration_py --> qc_integration_py_OpenQasmAdapter
    app_py["app.py (py)"]
    class app_py mod;
    app_py_VQEResult["VQEResult"]
    class app_py_VQEResult cls;
    app_py --> app_py_VQEResult
    app_py__sd_indices["_sd_indices"]
    class app_py__sd_indices fn;
    app_py --> app_py__sd_indices
    app_py__run_circuit["_run_circuit"]
    class app_py__run_circuit fn;
    app_py --> app_py__run_circuit
    app_py_givens_single_excitation["givens_single_excitation"]
    class app_py_givens_single_excitation fn;
    app_py --> app_py_givens_single_excitation
    app_py_particle_conserving_ansatz["particle_conserving_ansatz"]
    class app_py_particle_conserving_ansatz fn;
    app_py --> app_py_particle_conserving_ansatz
    polarizability_v3_py["polarizability_v3.py (py)"]
    class polarizability_v3_py mod;
    polarizability_v3_py_VQEResult["VQEResult"]
    class polarizability_v3_py_VQEResult cls;
    polarizability_v3_py --> polarizability_v3_py_VQEResult
    polarizability_v3_py__sd_indices["_sd_indices"]
    class polarizability_v3_py__sd_indices fn;
    polarizability_v3_py --> polarizability_v3_py__sd_indices
    polarizability_v3_py__run_circuit["_run_circuit"]
    class polarizability_v3_py__run_circuit fn;
    polarizability_v3_py --> polarizability_v3_py__run_circuit
    polarizability_v3_py_givens_single_excitation["givens_single_excitation"]
    class polarizability_v3_py_givens_single_excitation fn;
    polarizability_v3_py --> polarizability_v3_py_givens_single_excitation
    polarizability_v3_py_particle_conserving_ansatz["particle_conserving_ansatz"]
    class polarizability_v3_py_particle_conserving_ansatz fn;
    polarizability_v3_py --> polarizability_v3_py_particle_conserving_ansatz
    quantum_framework_visualization_py["quantum_framework_visualization.py (py)"]
    class quantum_framework_visualization_py mod;
    quantum_framework_visualization_py__make_logger["_make_logger"]
    class quantum_framework_visualization_py__make_logger fn;
    quantum_framework_visualization_py --> quantum_framework_visualization_py__make_logger
    quantum_framework_visualization_py_WavefunctionCalculator["WavefunctionCalculator"]
    class quantum_framework_visualization_py_WavefunctionCalculator cls;
    quantum_framework_visualization_py --> quantum_framework_visualization_py_WavefunctionCalculator
    quantum_framework_visualization_py_MonteCarloSampler["MonteCarloSampler"]
    class quantum_framework_visualization_py_MonteCarloSampler cls;
    quantum_framework_visualization_py --> quantum_framework_visualization_py_MonteCarloSampler
    quantum_framework_visualization_py_OrbitalVisualizer["OrbitalVisualizer"]
    class quantum_framework_visualization_py_OrbitalVisualizer cls;
    quantum_framework_visualization_py --> quantum_framework_visualization_py_OrbitalVisualizer
    quantum_framework_visualization_py_EntangledHydrogenSampler["EntangledHydrogenSampler"]
    class quantum_framework_visualization_py_EntangledHydrogenSampler cls;
    quantum_framework_visualization_py --> quantum_framework_visualization_py_EntangledHydrogenSampler
    quantum_computer_py["quantum_computer.py (py)"]
    class quantum_computer_py mod;
    quantum_computer_py__make_logger["_make_logger"]
    class quantum_computer_py__make_logger fn;
    quantum_computer_py --> quantum_computer_py__make_logger
    quantum_computer_py_SimulatorConfig["SimulatorConfig"]
    class quantum_computer_py_SimulatorConfig cls;
    quantum_computer_py --> quantum_computer_py_SimulatorConfig
    quantum_computer_py_SpectralLayer["SpectralLayer"]
    class quantum_computer_py_SpectralLayer cls;
    quantum_computer_py --> quantum_computer_py_SpectralLayer
    quantum_computer_py_HamiltonianBackboneNet["HamiltonianBackboneNet"]
    class quantum_computer_py_HamiltonianBackboneNet cls;
    quantum_computer_py --> quantum_computer_py_HamiltonianBackboneNet
    quantum_computer_py_SchrodingerSpectralNet["SchrodingerSpectralNet"]
    class quantum_computer_py_SchrodingerSpectralNet cls;
    quantum_computer_py --> quantum_computer_py_SchrodingerSpectralNet
    orbital_visualizer2_py["orbital_visualizer2.py (py)"]
    class orbital_visualizer2_py mod;
    orbital_visualizer2_py_Config["Config"]
    class orbital_visualizer2_py_Config cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_Config
    orbital_visualizer2_py_WavefunctionCalculator["WavefunctionCalculator"]
    class orbital_visualizer2_py_WavefunctionCalculator cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_WavefunctionCalculator
    orbital_visualizer2_py_HamiltonianNNProcessor["HamiltonianNNProcessor"]
    class orbital_visualizer2_py_HamiltonianNNProcessor cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_HamiltonianNNProcessor
    orbital_visualizer2_py_MonteCarloSampler["MonteCarloSampler"]
    class orbital_visualizer2_py_MonteCarloSampler cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_MonteCarloSampler
    orbital_visualizer2_py_OrbitalVisualizer["OrbitalVisualizer"]
    class orbital_visualizer2_py_OrbitalVisualizer cls;
    orbital_visualizer2_py --> orbital_visualizer2_py_OrbitalVisualizer
    test_quantum_framework_py["test_quantum_framework.py (py)"]
    class test_quantum_framework_py mod;
    test_quantum_framework_py_config["config"]
    class test_quantum_framework_py_config fn;
    test_quantum_framework_py --> test_quantum_framework_py_config
    test_quantum_framework_py_qc["qc"]
    class test_quantum_framework_py_qc fn;
    test_quantum_framework_py --> test_quantum_framework_py_qc
    test_quantum_framework_py_config_precision["config_precision"]
    class test_quantum_framework_py_config_precision fn;
    test_quantum_framework_py --> test_quantum_framework_py_config_precision
    test_quantum_framework_py_TestFrameworkConfig["TestFrameworkConfig"]
    class test_quantum_framework_py_TestFrameworkConfig cls;
    test_quantum_framework_py --> test_quantum_framework_py_TestFrameworkConfig
    test_quantum_framework_py_TestMPSState["TestMPSState"]
    class test_quantum_framework_py_TestMPSState cls;
    test_quantum_framework_py --> test_quantum_framework_py_TestMPSState
    quantum_framework_main_py["quantum_framework_main.py (py)"]
    class quantum_framework_main_py mod;
    quantum_framework_main_py_setup_logging["setup_logging"]
    class quantum_framework_main_py_setup_logging fn;
    quantum_framework_main_py --> quantum_framework_main_py_setup_logging
    quantum_framework_main_py_run_benchmark["run_benchmark"]
    class quantum_framework_main_py_run_benchmark fn;
    quantum_framework_main_py --> quantum_framework_main_py_run_benchmark
    quantum_framework_main_py_run_experiment["run_experiment"]
    class quantum_framework_main_py_run_experiment fn;
    quantum_framework_main_py --> quantum_framework_main_py_run_experiment
    quantum_framework_main_py_run_molecular_simulation["run_molecular_simulation"]
    class quantum_framework_main_py_run_molecular_simulation fn;
    quantum_framework_main_py --> quantum_framework_main_py_run_molecular_simulation
    quantum_framework_main_py_run_orbital_visualization["run_orbital_visualization"]
    class quantum_framework_main_py_run_orbital_visualization fn;
    quantum_framework_main_py --> quantum_framework_main_py_run_orbital_visualization
    demo_molecular_vqe_py["demo_molecular_vqe.py (py)"]
    class demo_molecular_vqe_py mod;
    demo_molecular_vqe_py_print_header["print_header"]
    class demo_molecular_vqe_py_print_header fn;
    demo_molecular_vqe_py --> demo_molecular_vqe_py_print_header
    demo_molecular_vqe_py_check_dependencies["check_dependencies"]
    class demo_molecular_vqe_py_check_dependencies fn;
    demo_molecular_vqe_py --> demo_molecular_vqe_py_check_dependencies
    demo_molecular_vqe_py_demo_h2_direct["demo_h2_direct"]
    class demo_molecular_vqe_py_demo_h2_direct fn;
    demo_molecular_vqe_py --> demo_molecular_vqe_py_demo_h2_direct
    demo_molecular_vqe_py_demo_h2_mps["demo_h2_mps"]
    class demo_molecular_vqe_py_demo_h2_mps fn;
    demo_molecular_vqe_py --> demo_molecular_vqe_py_demo_h2_mps
    demo_molecular_vqe_py_demo_comparison["demo_comparison"]
    class demo_molecular_vqe_py_demo_comparison fn;
    demo_molecular_vqe_py --> demo_molecular_vqe_py_demo_comparison
    quantum_framework_physics_py["quantum_framework_physics.py (py)"]
    class quantum_framework_physics_py mod;
    quantum_framework_physics_py_SpectralLayer["SpectralLayer"]
    class quantum_framework_physics_py_SpectralLayer cls;
    quantum_framework_physics_py --> quantum_framework_physics_py_SpectralLayer
    quantum_framework_physics_py_HamiltonianBackboneNet["HamiltonianBackboneNet"]
    class quantum_framework_physics_py_HamiltonianBackboneNet cls;
    quantum_framework_physics_py --> quantum_framework_physics_py_HamiltonianBackboneNet
    quantum_framework_physics_py_SchrodingerSpectralNet["SchrodingerSpectralNet"]
    class quantum_framework_physics_py_SchrodingerSpectralNet cls;
    quantum_framework_physics_py --> quantum_framework_physics_py_SchrodingerSpectralNet
    quantum_framework_physics_py_DiracSpectralNet["DiracSpectralNet"]
    class quantum_framework_physics_py_DiracSpectralNet cls;
    quantum_framework_physics_py --> quantum_framework_physics_py_DiracSpectralNet
    quantum_framework_physics_py_GammaMatrices["GammaMatrices"]
    class quantum_framework_physics_py_GammaMatrices cls;
    quantum_framework_physics_py --> quantum_framework_physics_py_GammaMatrices
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext___future__["__future__"]
    class ext___future__ ext;
    advanced_experiments_py -.->|imports| ext___future__
    ext_logging["logging"]
    class ext_logging ext;
    advanced_experiments_py -.->|imports| ext_logging
    ext_math["math"]
    class ext_math ext;
    advanced_experiments_py -.->|imports| ext_math
    ext_os["os"]
    class ext_os ext;
    advanced_experiments_py -.->|imports| ext_os
    ext_sys["sys"]
    class ext_sys ext;
    advanced_experiments_py -.->|imports| ext_sys
    ext_warnings["warnings"]
    class ext_warnings ext;
    advanced_experiments_py -.->|imports| ext_warnings
    ext_dataclasses["dataclasses"]
    class ext_dataclasses ext;
    advanced_experiments_py -.->|imports| ext_dataclasses
    ext_typing["typing"]
    class ext_typing ext;
    advanced_experiments_py -.->|imports| ext_typing
    ext_abc["abc"]
    class ext_abc ext;
    advanced_experiments_py -.->|imports| ext_abc
    ext_numpy["numpy"]
    class ext_numpy ext;
    advanced_experiments_py -.->|imports| ext_numpy
    ext_torch["torch"]
    class ext_torch ext;
    advanced_experiments_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    advanced_experiments_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    advanced_experiments_py -.->|imports| ext_torch_nn_functional
    ext_quantum_computer["quantum_computer"]
    class ext_quantum_computer ext;
    advanced_experiments_py -.->|imports| ext_quantum_computer
    ext_quantum_simulator["quantum_simulator"]
    class ext_quantum_simulator ext;
    advanced_experiments_py -.->|imports| ext_quantum_simulator
    ext_relativistic_hydrogen["relativistic_hydrogen"]
    class ext_relativistic_hydrogen ext;
    advanced_experiments_py -.->|imports| ext_relativistic_hydrogen
    ext_molecular_sim["molecular_sim"]
    class ext_molecular_sim ext;
    advanced_experiments_py -.->|imports| ext_molecular_sim
    ext_argparse["argparse"]
    class ext_argparse ext;
    advanced_experiments_py -.->|imports| ext_argparse
    ext_pyscf["pyscf"]
    class ext_pyscf ext;
    advanced_experiments_py -.->|imports| ext_pyscf
    app_py -.->|imports| ext___future__
    app_py -.->|imports| ext_warnings
    app_py -.->|imports| ext_math
    app_py -.->|imports| ext_numpy
    app_py -.->|imports| ext_sys
    app_py -.->|imports| ext_dataclasses
    app_py -.->|imports| ext_typing
    app_py -.->|imports| ext_torch
    app_py -.->|imports| ext_quantum_computer
    app_py -.->|imports| ext_molecular_sim
    app_py -.->|imports| ext_pyscf
    ext_openfermion_transforms["openfermion.transforms"]
    class ext_openfermion_transforms ext;
    app_py -.->|imports| ext_openfermion_transforms
    ext_openfermion_ops["openfermion.ops"]
    class ext_openfermion_ops ext;
    app_py -.->|imports| ext_openfermion_ops
    ext_scipy_optimize["scipy.optimize"]
    class ext_scipy_optimize ext;
    app_py -.->|imports| ext_scipy_optimize
    demo_molecular_vqe_py -.->|imports| ext_logging
    demo_molecular_vqe_py -.->|imports| ext_sys
    ext_time["time"]
    class ext_time ext;
    demo_molecular_vqe_py -.->|imports| ext_time
    demo_molecular_vqe_py -.->|imports| ext_typing
    ext_quantum_framework_molecular_v2["quantum_framework_molecular_v2"]
    class ext_quantum_framework_molecular_v2 ext;
    demo_molecular_vqe_py -.->|imports| ext_quantum_framework_molecular_v2
    demo_molecular_vqe_py -.->|imports| ext_quantum_framework_molecular_v2
    demo_molecular_vqe_py -.->|imports| ext_numpy
    demo_molecular_vqe_py -.->|imports| ext_os
    ext_traceback["traceback"]
    class ext_traceback ext;
    demo_molecular_vqe_py -.->|imports| ext_traceback
    entangled_hydrogen_py -.->|imports| ext___future__
    entangled_hydrogen_py -.->|imports| ext_logging
    entangled_hydrogen_py -.->|imports| ext_math
    entangled_hydrogen_py -.->|imports| ext_os
    entangled_hydrogen_py -.->|imports| ext_sys
    entangled_hydrogen_py -.->|imports| ext_warnings
    entangled_hydrogen_py -.->|imports| ext_abc
    entangled_hydrogen_py -.->|imports| ext_dataclasses
    entangled_hydrogen_py -.->|imports| ext_typing
    entangled_hydrogen_py -.->|imports| ext_numpy
    entangled_hydrogen_py -.->|imports| ext_torch
    entangled_hydrogen_py -.->|imports| ext_torch_nn
    entangled_hydrogen_py -.->|imports| ext_torch_nn_functional
    ext_scipy_special["scipy.special"]
    class ext_scipy_special ext;
    entangled_hydrogen_py -.->|imports| ext_scipy_special
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    entangled_hydrogen_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib["matplotlib"]
    class ext_matplotlib ext;
    entangled_hydrogen_py -.->|imports| ext_matplotlib
    ext_matplotlib_colors["matplotlib.colors"]
    class ext_matplotlib_colors ext;
    entangled_hydrogen_py -.->|imports| ext_matplotlib_colors
    entangled_hydrogen_py -.->|imports| ext_argparse
    entangled_hydrogen_py -.->|imports| ext_quantum_computer
    entangled_hydrogen_py -.->|imports| ext_quantum_computer
    entangled_hydrogen_py -.->|imports| ext_molecular_sim
    entangled_hydrogen_py -.->|imports| ext_relativistic_hydrogen
    higgs_four_lepton_analysis_py -.->|imports| ext___future__
    ext_csv["csv"]
    class ext_csv ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_csv
    higgs_four_lepton_analysis_py -.->|imports| ext_logging
    higgs_four_lepton_analysis_py -.->|imports| ext_math
    higgs_four_lepton_analysis_py -.->|imports| ext_os
    higgs_four_lepton_analysis_py -.->|imports| ext_sys
    ext_urllib_request["urllib.request"]
    class ext_urllib_request ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_urllib_request
    ext_urllib_error["urllib.error"]
    class ext_urllib_error ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_urllib_error
    higgs_four_lepton_analysis_py -.->|imports| ext_dataclasses
    ext_enum["enum"]
    class ext_enum ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_enum
    ext_pathlib["pathlib"]
    class ext_pathlib ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_pathlib
    higgs_four_lepton_analysis_py -.->|imports| ext_typing
    higgs_four_lepton_analysis_py -.->|imports| ext_numpy
    higgs_four_lepton_analysis_py -.->|imports| ext_quantum_computer
    higgs_four_lepton_analysis_py -.->|imports| ext_torch
    higgs_four_lepton_analysis_py -.->|imports| ext_torch_nn_functional
    ext_plotly_graph_objects["plotly.graph_objects"]
    class ext_plotly_graph_objects ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_plotly_graph_objects
    ext_plotly_subplots["plotly.subplots"]
    class ext_plotly_subplots ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_plotly_subplots
    higgs_quantum_analysis_py -.->|imports| ext___future__
    higgs_quantum_analysis_py -.->|imports| ext_csv
    higgs_quantum_analysis_py -.->|imports| ext_logging
    higgs_quantum_analysis_py -.->|imports| ext_math
    higgs_quantum_analysis_py -.->|imports| ext_os
    higgs_quantum_analysis_py -.->|imports| ext_sys
    higgs_quantum_analysis_py -.->|imports| ext_urllib_request
    higgs_quantum_analysis_py -.->|imports| ext_urllib_error
    higgs_quantum_analysis_py -.->|imports| ext_dataclasses
    higgs_quantum_analysis_py -.->|imports| ext_enum
    higgs_quantum_analysis_py -.->|imports| ext_pathlib
    higgs_quantum_analysis_py -.->|imports| ext_typing
    higgs_quantum_analysis_py -.->|imports| ext_numpy
    higgs_quantum_analysis_py -.->|imports| ext_quantum_computer
    higgs_quantum_analysis_py -.->|imports| ext_torch
    higgs_quantum_analysis_py -.->|imports| ext_torch_nn_functional
    higgs_quantum_analysis_py -.->|imports| ext_plotly_graph_objects
    higgs_quantum_analysis_py -.->|imports| ext_plotly_subplots
    molecular_sim_py -.->|imports| ext___future__
    molecular_sim_py -.->|imports| ext_logging
    molecular_sim_py -.->|imports| ext_math
    molecular_sim_py -.->|imports| ext_os
    molecular_sim_py -.->|imports| ext_sys
    molecular_sim_py -.->|imports| ext_dataclasses
    molecular_sim_py -.->|imports| ext_typing
    molecular_sim_py -.->|imports| ext_numpy
    molecular_sim_py -.->|imports| ext_torch
    molecular_sim_py -.->|imports| ext_quantum_computer
    molecular_sim_py -.->|imports| ext_argparse
    molecular_sim_py -.->|imports| ext_quantum_computer
    molecular_sim_py -.->|imports| ext_pyscf
    ext_openfermion["openfermion"]
    class ext_openfermion ext;
    molecular_sim_py -.->|imports| ext_openfermion
    molecular_sim_py -.->|imports| ext_openfermion_transforms
    ext_openfermion_linalg["openfermion.linalg"]
    class ext_openfermion_linalg ext;
    molecular_sim_py -.->|imports| ext_openfermion_linalg
    ext_openfermionpyscf["openfermionpyscf"]
    class ext_openfermionpyscf ext;
    molecular_sim_py -.->|imports| ext_openfermionpyscf
    molecular_sim_py -.->|imports| ext_openfermion
    molecular_sim_py -.->|imports| ext_quantum_computer
    molecular_sim_py -.->|imports| ext_scipy_optimize
    orbital_visualizer2_py -.->|imports| ext_numpy
    orbital_visualizer2_py -.->|imports| ext_scipy_special
    orbital_visualizer2_py -.->|imports| ext_matplotlib_pyplot
    orbital_visualizer2_py -.->|imports| ext_matplotlib
    orbital_visualizer2_py -.->|imports| ext_torch
    orbital_visualizer2_py -.->|imports| ext_os
    orbital_visualizer2_py -.->|imports| ext_sys
    orbital_visualizer2_py -.->|imports| ext_warnings
    orbital_visualizer2_py -.->|imports| ext_typing
    ext_schrodinger_crystal_fixed2["schrodinger_crystal_fixed2"]
    class ext_schrodinger_crystal_fixed2 ext;
    orbital_visualizer2_py -.->|imports| ext_schrodinger_crystal_fixed2
    orbital_visualizer2_py -.->|imports| ext_plotly_graph_objects
    orbital_visualizer2_py -.->|imports| ext_traceback
    polarizability_v3_py -.->|imports| ext___future__
    polarizability_v3_py -.->|imports| ext_warnings
    polarizability_v3_py -.->|imports| ext_math
    polarizability_v3_py -.->|imports| ext_numpy
    polarizability_v3_py -.->|imports| ext_sys
    polarizability_v3_py -.->|imports| ext_dataclasses
    polarizability_v3_py -.->|imports| ext_typing
    polarizability_v3_py -.->|imports| ext_torch
    polarizability_v3_py -.->|imports| ext_quantum_computer
    polarizability_v3_py -.->|imports| ext_molecular_sim
    polarizability_v3_py -.->|imports| ext_pyscf
    polarizability_v3_py -.->|imports| ext_openfermion_transforms
    polarizability_v3_py -.->|imports| ext_openfermion_ops
    polarizability_v3_py -.->|imports| ext_scipy_optimize
    qc_dashboard_py -.->|imports| ext___future__
    ext_io["io"]
    class ext_io ext;
    qc_dashboard_py -.->|imports| ext_io
    qc_dashboard_py -.->|imports| ext_logging
    qc_dashboard_py -.->|imports| ext_math
    qc_dashboard_py -.->|imports| ext_os
    qc_dashboard_py -.->|imports| ext_sys
    ext_tempfile["tempfile"]
    class ext_tempfile ext;
    qc_dashboard_py -.->|imports| ext_tempfile
    qc_dashboard_py -.->|imports| ext_dataclasses
    qc_dashboard_py -.->|imports| ext_pathlib
    qc_dashboard_py -.->|imports| ext_typing
    qc_dashboard_py -.->|imports| ext_numpy
    qc_dashboard_py -.->|imports| ext_matplotlib
    qc_dashboard_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib_patches["matplotlib.patches"]
    class ext_matplotlib_patches ext;
    qc_dashboard_py -.->|imports| ext_matplotlib_patches
    ext_quantum_framework_core["quantum_framework_core"]
    class ext_quantum_framework_core ext;
    qc_dashboard_py -.->|imports| ext_quantum_framework_core
    qc_dashboard_py -.->|imports| ext_quantum_computer
    qc_dashboard_py -.->|imports| ext_tempfile
    qc_dashboard_py -.->|imports| ext_tempfile
    qc_dashboard_py -.->|imports| ext_tempfile
    ext_streamlit["streamlit"]
    class ext_streamlit ext;
    qc_dashboard_py -.->|imports| ext_streamlit
    qc_dashboard_py -.->|imports| ext_matplotlib
    qc_dashboard_py -.->|imports| ext_matplotlib_pyplot
    ext_mpl_toolkits_mplot3d["mpl_toolkits.mplot3d"]
    class ext_mpl_toolkits_mplot3d ext;
    qc_dashboard_py -.->|imports| ext_mpl_toolkits_mplot3d
    qc_dashboard_py -.->|imports| ext_quantum_framework_core
    qc_dashboard_py -.->|imports| ext_quantum_framework_core
    qc_dashboard_py -.->|imports| ext_quantum_computer
    qc_dashboard_py -.->|imports| ext_plotly_graph_objects
    qc_dashboard_py -.->|imports| ext_plotly_subplots
    ext_orbital_visualizer2["orbital_visualizer2"]
    class ext_orbital_visualizer2 ext;
    qc_dashboard_py -.->|imports| ext_orbital_visualizer2
    qc_dashboard_py -.->|imports| ext_numpy
    ext_quantum_framework_visualization["quantum_framework_visualization"]
    class ext_quantum_framework_visualization ext;
    qc_dashboard_py -.->|imports| ext_quantum_framework_visualization
    ext_quantum_dash["quantum_dash"]
    class ext_quantum_dash ext;
    qc_dashboard_py -.->|imports| ext_quantum_dash
    ext_quantum_3dview["quantum_3dview"]
    class ext_quantum_3dview ext;
    qc_dashboard_py -.->|imports| ext_quantum_3dview
    qc_dashboard_py -.->|imports| ext_streamlit
    ext_qc_integration["qc_integration"]
    class ext_qc_integration ext;
    qc_dashboard_py -.->|imports| ext_qc_integration
    qc_dashboard_py -.->|imports| ext_quantum_computer
    qc_dashboard_py -.->|imports| ext_qc_integration
    qc_dashboard_py -.->|imports| ext_qc_integration
    qc_integration_py -.->|imports| ext___future__
    qc_integration_py -.->|imports| ext_logging
    qc_integration_py -.->|imports| ext_math
    ext_re["re"]
    class ext_re ext;
    qc_integration_py -.->|imports| ext_re
    qc_integration_py -.->|imports| ext_abc
    qc_integration_py -.->|imports| ext_dataclasses
    qc_integration_py -.->|imports| ext_typing
    qc_integration_py -.->|imports| ext_argparse
    qc_integration_py -.->|imports| ext_quantum_framework_core
    qc_integration_py -.->|imports| ext_quantum_framework_core
    qc_integration_py -.->|imports| ext_quantum_computer
    qc_integration_py -.->|imports| ext_quantum_computer
    ext_qiskit["qiskit"]
    class ext_qiskit ext;
    qc_integration_py -.->|imports| ext_qiskit
    ext_pennylane["pennylane"]
    class ext_pennylane ext;
    qc_integration_py -.->|imports| ext_pennylane
    quantum_3dview_py -.->|imports| ext_numpy
    quantum_3dview_py -.->|imports| ext_torch
    quantum_3dview_py -.->|imports| ext_plotly_graph_objects
    quantum_3dview_py -.->|imports| ext_plotly_subplots
    ext_plotly_express["plotly.express"]
    class ext_plotly_express ext;
    quantum_3dview_py -.->|imports| ext_plotly_express
    quantum_3dview_py -.->|imports| ext_dataclasses
    quantum_3dview_py -.->|imports| ext_typing
    quantum_3dview_py -.->|imports| ext_enum
    ext_colorsys["colorsys"]
    class ext_colorsys ext;
    quantum_3dview_py -.->|imports| ext_colorsys
    ext_json["json"]
    class ext_json ext;
    quantum_3dview_py -.->|imports| ext_json
    quantum_3dview_py -.->|imports| ext_pathlib
    quantum_3dview_py -.->|imports| ext_warnings
    quantum_3dview_py -.->|imports| ext_quantum_computer
    ext_quantum_visualizer["quantum_visualizer"]
    class ext_quantum_visualizer ext;
    quantum_3dview_py -.->|imports| ext_quantum_visualizer
    ext_IPython_display["IPython.display"]
    class ext_IPython_display ext;
    quantum_3dview_py -.->|imports| ext_IPython_display
    ext_scipy_io["scipy.io"]
    class ext_scipy_io ext;
    quantum_3dview_py -.->|imports| ext_scipy_io
    quantum_computer_py -.->|imports| ext___future__
    quantum_computer_py -.->|imports| ext_logging
    quantum_computer_py -.->|imports| ext_math
    quantum_computer_py -.->|imports| ext_os
    quantum_computer_py -.->|imports| ext_warnings
    quantum_computer_py -.->|imports| ext_abc
    quantum_computer_py -.->|imports| ext_dataclasses
    quantum_computer_py -.->|imports| ext_typing
    quantum_computer_py -.->|imports| ext_numpy
    quantum_computer_py -.->|imports| ext_torch
    quantum_computer_py -.->|imports| ext_torch_nn
    quantum_computer_py -.->|imports| ext_torch_nn_functional
    quantum_computer_py -.->|imports| ext_argparse
    quantum_dash_py -.->|imports| ext___future__
    quantum_dash_py -.->|imports| ext_logging
    quantum_dash_py -.->|imports| ext_math
    quantum_dash_py -.->|imports| ext_os
    quantum_dash_py -.->|imports| ext_sys
    quantum_dash_py -.->|imports| ext_warnings
    quantum_dash_py -.->|imports| ext_abc
    quantum_dash_py -.->|imports| ext_dataclasses
    quantum_dash_py -.->|imports| ext_enum
    quantum_dash_py -.->|imports| ext_pathlib
    quantum_dash_py -.->|imports| ext_typing
    quantum_dash_py -.->|imports| ext_numpy
    quantum_dash_py -.->|imports| ext_torch
    quantum_dash_py -.->|imports| ext_plotly_graph_objects
    quantum_dash_py -.->|imports| ext_plotly_subplots
    quantum_dash_py -.->|imports| ext_plotly_express
    quantum_dash_py -.->|imports| ext_matplotlib
    quantum_dash_py -.->|imports| ext_matplotlib_pyplot
    quantum_dash_py -.->|imports| ext_matplotlib_patches
    quantum_dash_py -.->|imports| ext_mpl_toolkits_mplot3d
    quantum_dash_py -.->|imports| ext_mpl_toolkits_mplot3d
    quantum_dash_py -.->|imports| ext_argparse
    quantum_dash_py -.->|imports| ext_quantum_computer
    quantum_dash_py -.->|imports| ext_molecular_sim
    quantum_framework_core_py -.->|imports| ext___future__
    quantum_framework_core_py -.->|imports| ext_logging
    quantum_framework_core_py -.->|imports| ext_math
    quantum_framework_core_py -.->|imports| ext_os
    quantum_framework_core_py -.->|imports| ext_warnings
    quantum_framework_core_py -.->|imports| ext_abc
    quantum_framework_core_py -.->|imports| ext_dataclasses
    quantum_framework_core_py -.->|imports| ext_enum
    quantum_framework_core_py -.->|imports| ext_typing
    quantum_framework_core_py -.->|imports| ext_numpy
    quantum_framework_core_py -.->|imports| ext_torch
    quantum_framework_core_py -.->|imports| ext_torch_nn
    quantum_framework_core_py -.->|imports| ext_torch_nn_functional
    ext_tomllib["tomllib"]
    class ext_tomllib ext;
    quantum_framework_core_py -.->|imports| ext_tomllib
    quantum_framework_core_py -.->|imports| ext_time
    quantum_framework_core_py -.->|imports| ext_argparse
    ext_tomli["tomli"]
    class ext_tomli ext;
    quantum_framework_core_py -.->|imports| ext_tomli
    quantum_framework_main_py -.->|imports| ext___future__
    quantum_framework_main_py -.->|imports| ext_argparse
    quantum_framework_main_py -.->|imports| ext_logging
    quantum_framework_main_py -.->|imports| ext_os
    quantum_framework_main_py -.->|imports| ext_sys
    quantum_framework_main_py -.->|imports| ext_typing
    quantum_framework_main_py -.->|imports| ext_torch
    quantum_framework_main_py -.->|imports| ext_quantum_framework_core
    ext_quantum_framework_menu["quantum_framework_menu"]
    class ext_quantum_framework_menu ext;
    quantum_framework_main_py -.->|imports| ext_quantum_framework_menu
    ext_quantum_lab["quantum_lab"]
    class ext_quantum_lab ext;
    quantum_framework_main_py -.->|imports| ext_quantum_lab
    quantum_framework_menu_py -.->|imports| ext___future__
    quantum_framework_menu_py -.->|imports| ext_logging
    quantum_framework_menu_py -.->|imports| ext_math
    quantum_framework_menu_py -.->|imports| ext_os
    quantum_framework_menu_py -.->|imports| ext_sys
    quantum_framework_menu_py -.->|imports| ext_time
    quantum_framework_menu_py -.->|imports| ext_dataclasses
    quantum_framework_menu_py -.->|imports| ext_typing
    quantum_framework_menu_py -.->|imports| ext_numpy
    quantum_framework_menu_py -.->|imports| ext_torch
    quantum_framework_menu_py -.->|imports| ext_quantum_framework_core
    ext_quantum_framework_molecular["quantum_framework_molecular"]
    class ext_quantum_framework_molecular ext;
    quantum_framework_menu_py -.->|imports| ext_quantum_framework_molecular
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    quantum_framework_menu_py -.->|imports| ext_matplotlib_pyplot
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    ext_higgs_four_lepton_analysis["higgs_four_lepton_analysis"]
    class ext_higgs_four_lepton_analysis ext;
    quantum_framework_menu_py -.->|imports| ext_higgs_four_lepton_analysis
    quantum_framework_menu_py -.->|imports| ext_quantum_dash
    quantum_framework_menu_py -.->|imports| ext_quantum_3dview
    quantum_framework_menu_py -.->|imports| ext_json
    quantum_framework_menu_py -.->|imports| ext_quantum_visualizer
    ext_app["app"]
    class ext_app ext;
    quantum_framework_menu_py -.->|imports| ext_app
    quantum_framework_menu_py -.->|imports| ext_traceback
    quantum_framework_molecular_py -.->|imports| ext___future__
    quantum_framework_molecular_py -.->|imports| ext_logging
    quantum_framework_molecular_py -.->|imports| ext_math
    quantum_framework_molecular_py -.->|imports| ext_os
    quantum_framework_molecular_py -.->|imports| ext_warnings
    quantum_framework_molecular_py -.->|imports| ext_dataclasses
    quantum_framework_molecular_py -.->|imports| ext_typing
    quantum_framework_molecular_py -.->|imports| ext_numpy
    quantum_framework_molecular_py -.->|imports| ext_torch
    quantum_framework_molecular_py -.->|imports| ext_pyscf
    quantum_framework_molecular_py -.->|imports| ext_openfermion
    quantum_framework_molecular_py -.->|imports| ext_openfermion_transforms
    quantum_framework_molecular_py -.->|imports| ext_openfermion_ops
    quantum_framework_molecular_py -.->|imports| ext_openfermionpyscf
    quantum_framework_molecular_py -.->|imports| ext_scipy_optimize
    quantum_framework_molecular_fixed_py -.->|imports| ext___future__
    quantum_framework_molecular_fixed_py -.->|imports| ext_logging
    quantum_framework_molecular_fixed_py -.->|imports| ext_math
    quantum_framework_molecular_fixed_py -.->|imports| ext_os
    quantum_framework_molecular_fixed_py -.->|imports| ext_warnings
    quantum_framework_molecular_fixed_py -.->|imports| ext_dataclasses
    quantum_framework_molecular_fixed_py -.->|imports| ext_typing
    quantum_framework_molecular_fixed_py -.->|imports| ext_numpy
    quantum_framework_molecular_fixed_py -.->|imports| ext_torch
    quantum_framework_molecular_fixed_py -.->|imports| ext_pyscf
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermion
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermion_transforms
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermion_linalg
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermion_ops
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermionpyscf
    quantum_framework_molecular_fixed_py -.->|imports| ext_scipy_optimize
    quantum_framework_molecular_v2_py -.->|imports| ext___future__
    quantum_framework_molecular_v2_py -.->|imports| ext_logging
    quantum_framework_molecular_v2_py -.->|imports| ext_math
    quantum_framework_molecular_v2_py -.->|imports| ext_os
    quantum_framework_molecular_v2_py -.->|imports| ext_warnings
    quantum_framework_molecular_v2_py -.->|imports| ext_dataclasses
    quantum_framework_molecular_v2_py -.->|imports| ext_typing
    quantum_framework_molecular_v2_py -.->|imports| ext_numpy
    quantum_framework_molecular_v2_py -.->|imports| ext_torch
    quantum_framework_molecular_v2_py -.->|imports| ext_pyscf
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermion
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermion_transforms
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermion_ops
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermion_linalg
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermionpyscf
    quantum_framework_molecular_v2_py -.->|imports| ext_scipy_optimize
    quantum_framework_molecular_v2_py -.->|imports| ext_tomllib
    quantum_framework_molecular_v2_py -.->|imports| ext_argparse
    quantum_framework_molecular_v2_py -.->|imports| ext_tomli
    quantum_framework_molecular_v2_py -.->|imports| ext_math
    ext_itertools["itertools"]
    class ext_itertools ext;
    quantum_framework_molecular_v2_py -.->|imports| ext_itertools
    quantum_framework_molecular_v2_py -.->|imports| ext_time
    quantum_framework_physics_py -.->|imports| ext___future__
    quantum_framework_physics_py -.->|imports| ext_math
    quantum_framework_physics_py -.->|imports| ext_warnings
    quantum_framework_physics_py -.->|imports| ext_typing
    quantum_framework_physics_py -.->|imports| ext_numpy
    quantum_framework_physics_py -.->|imports| ext_torch
    quantum_framework_physics_py -.->|imports| ext_torch_nn
    quantum_framework_physics_py -.->|imports| ext_torch_nn_functional
    quantum_framework_visualization_py -.->|imports| ext___future__
    quantum_framework_visualization_py -.->|imports| ext_logging
    quantum_framework_visualization_py -.->|imports| ext_math
    quantum_framework_visualization_py -.->|imports| ext_os
    quantum_framework_visualization_py -.->|imports| ext_warnings
    quantum_framework_visualization_py -.->|imports| ext_dataclasses
    quantum_framework_visualization_py -.->|imports| ext_typing
    quantum_framework_visualization_py -.->|imports| ext_numpy
    quantum_framework_visualization_py -.->|imports| ext_torch
    quantum_framework_visualization_py -.->|imports| ext_scipy_special
    quantum_framework_visualization_py -.->|imports| ext_matplotlib_pyplot
    quantum_framework_visualization_py -.->|imports| ext_matplotlib
    quantum_framework_visualization_py -.->|imports| ext_matplotlib_colors
    quantum_framework_visualization_py -.->|imports| ext_plotly_graph_objects
    quantum_lab_py -.->|imports| ext___future__
    quantum_lab_py -.->|imports| ext_argparse
    quantum_lab_py -.->|imports| ext_math
    quantum_lab_py -.->|imports| ext_os
    quantum_lab_py -.->|imports| ext_sys
    quantum_lab_py -.->|imports| ext_time
    quantum_lab_py -.->|imports| ext_dataclasses
    quantum_lab_py -.->|imports| ext_typing
    quantum_lab_py -.->|imports| ext_numpy
    quantum_lab_py -.->|imports| ext_quantum_framework_core
    quantum_lab_py -.->|imports| ext_torch
    ext_rich_console["rich.console"]
    class ext_rich_console ext;
    quantum_lab_py -.->|imports| ext_rich_console
    ext_rich_panel["rich.panel"]
    class ext_rich_panel ext;
    quantum_lab_py -.->|imports| ext_rich_panel
    ext_rich_table["rich.table"]
    class ext_rich_table ext;
    quantum_lab_py -.->|imports| ext_rich_table
    ext_rich_text["rich.text"]
    class ext_rich_text ext;
    quantum_lab_py -.->|imports| ext_rich_text
    ext_rich_rule["rich.rule"]
    class ext_rich_rule ext;
    quantum_lab_py -.->|imports| ext_rich_rule
    ext_rich_prompt["rich.prompt"]
    class ext_rich_prompt ext;
    quantum_lab_py -.->|imports| ext_rich_prompt
    ext_rich_align["rich.align"]
    class ext_rich_align ext;
    quantum_lab_py -.->|imports| ext_rich_align
    ext_rich_columns["rich.columns"]
    class ext_rich_columns ext;
    quantum_lab_py -.->|imports| ext_rich_columns
    ext_rich_live["rich.live"]
    class ext_rich_live ext;
    quantum_lab_py -.->|imports| ext_rich_live
    quantum_lab_py -.->|imports| ext_scipy_special
    quantum_lab_py -.->|imports| ext_math
    quantum_lab_py -.->|imports| ext_quantum_framework_molecular
    quantum_simulator_py -.->|imports| ext___future__
    quantum_simulator_py -.->|imports| ext_logging
    quantum_simulator_py -.->|imports| ext_math
    quantum_simulator_py -.->|imports| ext_os
    quantum_simulator_py -.->|imports| ext_warnings
    quantum_simulator_py -.->|imports| ext_abc
    quantum_simulator_py -.->|imports| ext_dataclasses
    quantum_simulator_py -.->|imports| ext_enum
    quantum_simulator_py -.->|imports| ext_typing
    quantum_simulator_py -.->|imports| ext_numpy
    quantum_simulator_py -.->|imports| ext_torch
    quantum_simulator_py -.->|imports| ext_torch_nn
    quantum_simulator_py -.->|imports| ext_torch_nn_functional
    quantum_simulator_py -.->|imports| ext_tomllib
    quantum_simulator_py -.->|imports| ext_scipy_special
    quantum_simulator_py -.->|imports| ext_matplotlib_pyplot
    quantum_simulator_py -.->|imports| ext_tomli
    quantum_visualizer_py -.->|imports| ext___future__
    quantum_visualizer_py -.->|imports| ext_logging
    quantum_visualizer_py -.->|imports| ext_math
    quantum_visualizer_py -.->|imports| ext_os
    quantum_visualizer_py -.->|imports| ext_sys
    quantum_visualizer_py -.->|imports| ext_warnings
    quantum_visualizer_py -.->|imports| ext_abc
    quantum_visualizer_py -.->|imports| ext_dataclasses
    quantum_visualizer_py -.->|imports| ext_typing
    quantum_visualizer_py -.->|imports| ext_numpy
    quantum_visualizer_py -.->|imports| ext_torch
    quantum_visualizer_py -.->|imports| ext_quantum_computer
    quantum_visualizer_py -.->|imports| ext_molecular_sim
    ext_advanced_experiments["advanced_experiments"]
    class ext_advanced_experiments ext;
    quantum_visualizer_py -.->|imports| ext_advanced_experiments
    quantum_visualizer_py -.->|imports| ext_argparse
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    quantum_visualizer_py -.->|imports| ext_matplotlib_patches
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    quantum_visualizer_py -.->|imports| ext_mpl_toolkits_mplot3d
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    quantum_visualizer_py -.->|imports| ext_mpl_toolkits_mplot3d
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    relativistic_hydrogen_py -.->|imports| ext_numpy
    relativistic_hydrogen_py -.->|imports| ext_scipy_special
    ext_scipy["scipy"]
    class ext_scipy ext;
    relativistic_hydrogen_py -.->|imports| ext_scipy
    relativistic_hydrogen_py -.->|imports| ext_matplotlib_pyplot
    relativistic_hydrogen_py -.->|imports| ext_matplotlib
    relativistic_hydrogen_py -.->|imports| ext_matplotlib_colors
    relativistic_hydrogen_py -.->|imports| ext_torch
    relativistic_hydrogen_py -.->|imports| ext_torch_nn
    relativistic_hydrogen_py -.->|imports| ext_torch_nn_functional
    relativistic_hydrogen_py -.->|imports| ext_os
    relativistic_hydrogen_py -.->|imports| ext_sys
    relativistic_hydrogen_py -.->|imports| ext_warnings
    relativistic_hydrogen_py -.->|imports| ext_json
    relativistic_hydrogen_py -.->|imports| ext_typing
    relativistic_hydrogen_py -.->|imports| ext_dataclasses
    relativistic_hydrogen_py -.->|imports| ext_abc
    relativistic_hydrogen_py -.->|imports| ext_logging
    relativistic_hydrogen_py -.->|imports| ext_math
    ext_glob["glob"]
    class ext_glob ext;
    relativistic_hydrogen_py -.->|imports| ext_glob
    relativistic_hydrogen_py -.->|imports| ext_traceback
    test_qc_integration_py -.->|imports| ext___future__
    test_qc_integration_py -.->|imports| ext_math
    test_qc_integration_py -.->|imports| ext_typing
    ext_pytest["pytest"]
    class ext_pytest ext;
    test_qc_integration_py -.->|imports| ext_pytest
    test_qc_integration_py -.->|imports| ext_numpy
    test_qc_integration_py -.->|imports| ext_qc_integration
    ext_qc_dashboard["qc_dashboard"]
    class ext_qc_dashboard ext;
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_quantum_framework_core
    test_qc_integration_py -.->|imports| ext_quantum_framework_core
    test_quantum_framework_py -.->|imports| ext_math
    test_quantum_framework_py -.->|imports| ext_sys
    test_quantum_framework_py -.->|imports| ext_os
    test_quantum_framework_py -.->|imports| ext_typing
    test_quantum_framework_py -.->|imports| ext_pytest
    test_quantum_framework_py -.->|imports| ext_numpy
    test_quantum_framework_py -.->|imports| ext_torch
    test_quantum_framework_py -.->|imports| ext_quantum_framework_core
    test_quantum_framework_py -.->|imports| ext_quantum_framework_core
    test_quantum_framework_py -.->|imports| ext_quantum_framework_core
    topological_hilbert_compression2_py -.->|imports| ext___future__
    topological_hilbert_compression2_py -.->|imports| ext_logging
    topological_hilbert_compression2_py -.->|imports| ext_math
    topological_hilbert_compression2_py -.->|imports| ext_os
    topological_hilbert_compression2_py -.->|imports| ext_sys
    topological_hilbert_compression2_py -.->|imports| ext_warnings
    topological_hilbert_compression2_py -.->|imports| ext_abc
    topological_hilbert_compression2_py -.->|imports| ext_dataclasses
    topological_hilbert_compression2_py -.->|imports| ext_enum
    topological_hilbert_compression2_py -.->|imports| ext_typing
    topological_hilbert_compression2_py -.->|imports| ext_numpy
    topological_hilbert_compression2_py -.->|imports| ext_torch
    topological_hilbert_compression2_py -.->|imports| ext_torch_nn
    topological_hilbert_compression2_py -.->|imports| ext_torch_nn_functional
    topological_hilbert_compression2_py -.->|imports| ext_argparse
    topological_hilbert_compression2_py -.->|imports| ext_quantum_computer
    topological_hilbert_compression2_py -.->|imports| ext_time
    topological_hilbert_compression2_py -.->|imports| ext_quantum_computer
```

---

## Architecture Reference

### PY (29 files)

#### `advanced_experiments.py`
**Path:** `advanced_experiments.py`

**Classs:**
- `GroverConfig` (line 141) - *Configuration for Grover's algorithm experiments.*
- `GroverOracle` (line 159) - *Oracle for Grover's algorithm.
Marks the target state by applying a phase flip.

Uses the existing MCZGate from quantum_computer.py for multi-controlled Z.*
- `GroverDiffusionOperator` (line 194) - *Diffusion operator (Grover diffusion / inversion about mean).

D = 2|s><s| - I where |s> = H^⊗n |0>

Implemented using the existing gate infrastructure.*
- `GroverSearch` (line 240) - *Complete Grover's algorithm implementation using existing quantum_computer.py infrastructure.*
- `QEDConfig` (line 404) - *Configuration for QED effects calculations.*
- `LambShiftCalculator` (line 430) - *Calculates the Lamb shift using Bethe's formula and more accurate methods.

The Lamb shift is the small difference between 2s_{1/2} and 2p_{1/2} energy levels
due to QED effects (vacuum fluctuations and self-energy).

Uses the existing Dirac infrastructure from relativistic_hydrogen.py*
- `AnomalousMagneticMoment` (line 595) - *Calculates the electron's anomalous magnetic moment (g-2).

The electron g-factor is slightly different from 2 due to QED effects:
g = 2(1 + a_e) where a_e = α/(2π) + higher-order terms

Uses the existing Dirac infrastructure for baseline calculations.*
- `QEDEffectsExperiment` (line 722) - *Complete QED effects experiment combining Lamb shift and g-2.
Uses existing relativistic_hydrogen.py infrastructure.*
- `PolyatomicMoleculeData` (line 831) - *Data for polyatomic molecules.*
- `MoleculeBuilder` (line 849) - *Build molecule data for VQE calculations.
Uses existing molecular_sim.py infrastructure.*
- `PolyatomicVQE` (line 970) - *VQE solver for polyatomic molecules.
Uses existing molecular_sim.py infrastructure.*
- `PolyatomicExperiment` (line 1105) - *Complete polyatomic molecule experiment.*
- `AdvancedExperimentRunner` (line 1223) - *Runs all three advanced experiments.*

**Functions:**
- `_make_logger` (line 121)
- `main` (line 1304) - *Main entry point.*
- `__init__` (line 167)
- `_validate` (line 172)
- `apply` (line 176) - *Apply oracle: flip phase of marked state.
|x> -> (-1)^{f(x)} |x> where f(x)=1 only for marked state.

Uses amplitude-level phase manipulation for exact implementation.*
- `__init__` (line 203)
- `apply` (line 206) - *Apply diffusion operator using Hadamard + Oracle on |0> + Hadamard.

D = H^⊗n (2|0><0| - I) H^⊗n*
- `__init__` (line 245)
- `_calculate_entropy` (line 266) - *Calculate Shannon entropy from probability distribution.
H = -sum(p * log2(p))*
- `_init_quantum_computer` (line 278) - *Initialize quantum computer using existing quantum_computer.py infrastructure.*
- `run` (line 300) - *Run Grover's search algorithm.

Returns:
    Dictionary with results including success probability and evolution history.*
- `__init__` (line 440)
- `bethe_formula` (line 462) - *Bethe's non-relativistic formula for Lamb shift.

ΔE_Lamb = (8α^3 / 3πn^3) * |ψ_n(0)|^2 * ln(E_avg / E_n)

For s-states: |ψ_n(0)|^2 = Z^3 / (π n^3 a0^3)

Args:
    n: Principal quantum number
    l: Angular momentum quantum number
    Z: Nuclear charge (default 1 for hydrogen)

Returns:
    Lamb shift in atomic units*
- `_higher_l_shift` (line 505) - *Approximate Lamb shift for l > 0.
Much smaller than for s-states.*
- `full_lamb_shift` (line 518) - *Calculate full Lamb shift including radiative corrections.

ΔE = ΔE_SE + ΔE_Uehling + ΔE_rel

Where:
- ΔE_SE: Self-energy (main contribution)
- ΔE_Uehling: Vacuum polarization (Uehling potential)
- ΔE_rel: Relativistic corrections*
- `compare_2s_2p` (line 558) - *Compare Lamb shift for 2s_{1/2} and 2p_{1/2} states.

This is the classic Lamb shift measurement: the 2s_{1/2} - 2p_{1/2} splitting.
Experimentally: ~1057.8 MHz*
- `__init__` (line 605)
- `schwinger_term` (line 611) - *Schwinger's first-order result: a_e = α/(2π)

This is the leading QED correction.*
- `second_order` (line 619) - *Second-order correction: (α/π)^2 * C_2
C_2 ≈ 0.328478965...*
- `third_order` (line 627) - *Third-order correction: (α/π)^3 * C_3
C_3 ≈ 1.181241456...*
- `fourth_order` (line 635) - *Fourth-order correction: (α/π)^4 * C_4
C_4 ≈ -1.9144(35)*
- `fifth_order` (line 643) - *Fifth-order correction: (α/π)^5 * C_5
C_5 ≈ 7.7(1.1)*
- `calculate_a_e` (line 651) - *Calculate anomalous magnetic moment to specified order.

Args:
    order: Maximum order to include (1-5)

Returns:
    Dictionary with contributions at each order*
- `full_report` (line 686) - *Generate a full report on g-2 calculations.*
- `__init__` (line 728)
- `run_full_analysis` (line 736) - *Run complete QED analysis.*
- `_calculate_energy_levels` (line 761) - *Calculate hydrogen energy levels including QED corrections.*
- `_dirac_energy` (line 805) - *Calculate Dirac energy level.
Uses existing relativistic_hydrogen.py if available.*
- `h2o` (line 866) - *Build water molecule geometry.

H2O geometry:
    H1 at (0, 0, 0)
    O  at (r_OH, 0, 0)
    H2 at (r_OH + r_OH*cos(θ), r_OH*sin(θ), 0)*
- `nh3` (line 898) - *Build ammonia molecule geometry.

NH3 has trigonal pyramidal geometry.*
- `ch4` (line 933) - *Build methane molecule geometry.

CH4 has tetrahedral geometry.*
- `__init__` (line 976)
- `run_pyscf` (line 999) - *Run PySCF calculation for the molecule.*
- `_hardcoded_values` (line 1066) - *Return hardcoded reference values for common molecules.*
- `__init__` (line 1110)
- `run_analysis` (line 1123) - *Run complete analysis for a molecule.*
- `run_all` (line 1164) - *Run analysis for all molecules.*
- `scan_bond_length` (line 1175) - *Scan potential energy surface by varying bond length.*
- `__init__` (line 1228)
- `run_grover` (line 1235) - *Run Grover's algorithm experiment.*
- `run_qed` (line 1252) - *Run QED effects experiment.*
- `run_polyatomic` (line 1266) - *Run polyatomic molecule experiment.*
- `run_all` (line 1278) - *Run all experiments.*

#### `app.py`
**Path:** `app.py`

**Classs:**
- `VQEResult` (line 39)
- `StarkEvaluator` (line 151)
- `DipoleOperatorBuilder` (line 203)
- `PolarizabilityCalculator` (line 237)

**Functions:**
- `_sd_indices` (line 46)
- `_run_circuit` (line 55)
- `givens_single_excitation` (line 61) - *Apply a particle-conserving single excitation rotation between 
qubits o (occupied) and v (virtual).

Rotates in the {|1_o 0_v⟩, |0_o 1_v⟩} subspace:
    |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩
    |0_o 1_v⟩ → -sin(θ)|1_o 0_v⟩ + cos(θ)|0_o 1_v⟩

For adjacent qubits: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)
For non-adjacent: SWAP chain to make adjacent, apply, SWAP back.*
- `particle_conserving_ansatz` (line 109) - *Particle-conserving UCCSD-like ansatz:
- Singles: Givens rotations (correct particle conservation)
- Doubles: reuse uccsd's double excitation (works correctly)*
- `__init__` (line 152)
- `_to_scalar` (line 161)
- `_apply_pauli` (line 165)
- `eval_dipole` (line 184)
- `__call__` (line 197)
- `__init__` (line 204)
- `__init__` (line 238)
- `_evaluator` (line 258)
- `_get_state` (line 262)
- `_diagnose` (line 267)
- `_optimize` (line 277)
- `run` (line 303)
- `cost` (line 290)

#### `demo_molecular_vqe.py`
**Path:** `demo_molecular_vqe.py`

**Functions:**
- `print_header` (line 47) - *Print formatted header.*
- `check_dependencies` (line 54) - *Check and report available dependencies.*
- `demo_h2_direct` (line 78) - *Demo: H2 with direct statevector (precision mode).*
- `demo_h2_mps` (line 104) - *Demo: H2 with MPS compression.*
- `demo_comparison` (line 131) - *Demo: Compare direct vs MPS.*
- `demo_molecule_builder` (line 167) - *Demo: OpenFermion molecule builder.*
- `demo_smart_initialization` (line 197) - *Demo: Smart parameter initialization.*
- `demo_cached_operations` (line 231) - *Demo: Cached Pauli operations.*
- `demo_config_from_toml` (line 277) - *Demo: Configuration from TOML.*
- `main` (line 302) - *Run all demos.*

#### `entangled_hydrogen.py`
**Path:** `entangled_hydrogen.py`

**Classs:**
- `EntangledHydrogenConfig` (line 61) - *Configuration for entangled hydrogen visualization system.

All parameters are parametric and configurable from this class.*
- `IEntangledState` (line 114) - *Abstract interface for entangled quantum states.*
- `BellState` (line 131) - *Bell state: |Phi+> = (|00> + |11>) / sqrt(2).*
- `GHZState` (line 145) - *GHZ state: (|00...0> + |11...1>) / sqrt(2).*
- `WState` (line 162) - *W state: (|001> + |010> + |100>) / sqrt(3).*
- `WavefunctionCalculator` (line 194) - *Calculates hydrogen atom wavefunctions for entangled state visualization.
Uses the same implementation as orbital_visualizer2.py.*
- `EntangledHydrogenSampler` (line 253) - *Monte Carlo sampler for entangled hydrogen states.
Samples from the joint probability distribution of entangled orbitals.*
- `EntangledHydrogenVisualizer` (line 408) - *Visualizer for entangled hydrogen states.
Creates high-resolution visualizations similar to orbital_visualizer2.py.*
- `EntangledHydrogenExperiment` (line 569) - *Main experiment class for entangled hydrogen visualization.

Uses the existing quantum_computer.py, molecular_sim.py, and
visualization components from orbital_visualizer2.py and relativistic_hydrogen.py.*

**Functions:**
- `_make_logger` (line 44) - *Create a module-level logger with consistent formatter.*
- `main` (line 893) - *Main entry point for entangled hydrogen visualization.*
- `name` (line 119) - *Return the name of the entangled state.*
- `prepare` (line 123) - *Prepare the entangled state on n qubits.*
- `get_theoretical_entropy` (line 127) - *Return the theoretical Shannon entropy in bits.*
- `name` (line 135)
- `prepare` (line 138)
- `get_theoretical_entropy` (line 141)
- `__init__` (line 148)
- `name` (line 152)
- `prepare` (line 155)
- `get_theoretical_entropy` (line 158)
- `__init__` (line 165)
- `name` (line 169)
- `prepare` (line 172)
- `get_theoretical_entropy` (line 190)
- `__init__` (line 200)
- `radial_wavefunction` (line 204) - *Calculate non-relativistic radial wavefunction R_nl(r).*
- `spherical_harmonic_real` (line 215) - *Calculate real spherical harmonics Y_lm(theta, phi).*
- `psi_on_grid` (line 225) - *Calculate wavefunction on 2D grid for quantum computer processing.*
- `psi_3d` (line 245) - *Calculate full 3D wavefunction psi_nlm(r, theta, phi).*
- `__init__` (line 259)
- `find_max_probability` (line 263) - *Find maximum probability for rejection sampling.*
- `sample_orbital` (line 295) - *Sample points from a single hydrogen orbital using Monte Carlo.*
- `sample_entangled_state` (line 362) - *Sample from entangled hydrogen state.

Creates a superposition of two orbitals with entanglement_weight:
|psi> = sqrt(1-w) * |n1,l1,m1> + sqrt(w) * |n2,l2,m2>

For true entanglement visualization, we create a joint state:
|Psi> = (|n1,l1,m1>|n2,l2,m2> + |n2,l2,m2>|n1,l1,m1>) / sqrt(2)*
- `__init__` (line 414)
- `visualize` (line 417) - *Create visualization of entangled hydrogen state.*
- `__init__` (line 577)
- `_initialize_quantum_computer` (line 591) - *Initialize the quantum computer using the existing quantum_computer.py module.*
- `run_bell_entangled_hydrogen` (line 630) - *Run Bell state entangled hydrogen visualization.

Creates a Bell state and correlates it with hydrogen orbitals.*
- `run_ghz_entangled_hydrogen` (line 672) - *Run GHZ state entangled hydrogen visualization.

Creates a GHZ state with n qubits and correlates with n orbitals.*
- `run_entangled_h_with_molecular_energy` (line 745) - *Run entangled hydrogen with molecular energy evaluation using molecular_sim.py.*
- `run_relativistic_entangled_hydrogen` (line 794) - *Run entangled hydrogen with relativistic Dirac calculations.
Uses components from relativistic_hydrogen.py.*
- `run_all_demonstrations` (line 852) - *Run all available entangled hydrogen demonstrations.*

#### `higgs_four_lepton_analysis.py`
**Path:** `higgs_four_lepton_analysis.py`

**Classs:**
- `LeptonType` (line 72)
- `EventType` (line 78)
- `Config` (line 86)
- `FourMomentum` (line 125)
- `Lepton` (line 164)
- `Event` (line 183)
- `QuantumSpinorProcessor` (line 205) - *Uses the ACTUAL DiracBackend from quantum_computer.py for spinor calculations.

This is NOT a fake implementation - it uses the neural network
spectral layers trained (or randomly initialized) for Dirac spinor evolution.*
- `EventParser` (line 377) - *Parse CMS CSV files with actual column names.*
- `Visualizer` (line 436) - *3D visualization using Plotly with quantum-processed trajectories.*
- `HiggsQuantumAnalysis` (line 612) - *Main analysis class using quantum backends from quantum_computer.py*

**Functions:**
- `_make_logger` (line 57)
- `main` (line 747)
- `__post_init__` (line 136)
- `from_energy_momentum` (line 151)
- `__add__` (line 154)
- `pt` (line 171)
- `eta` (line 173)
- `phi` (line 175)
- `energy` (line 177)
- `mass` (line 179)
- `__post_init__` (line 193)
- `check_higgs` (line 200)
- `__init__` (line 213)
- `_precompute_momentum_grids` (line 247) - *Precompute k-space grids for Dirac equation.*
- `momentum_to_spinor_wavefunction` (line 255) - *Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)
suitable for processing by the DiracBackend.

The wavefunction encodes the momentum as a plane wave with the correct
relativistic dispersion relation.*
- `evolve_with_dirac_backend` (line 296) - *Evolve a wavefunction using the DiracBackend neural network.

This applies the actual spectral layers from the trained (or random)
Dirac network.*
- `evolve_with_schrodinger_backend` (line 308) - *Evolve using the SchrodingerBackend neural network.*
- `evolve_with_hamiltonian_backend` (line 315) - *Evolve using the HamiltonianBackend neural network.*
- `compute_dirac_current` (line 322) - *Compute the Dirac current using the actual neural network backends.

Returns (jx, jy, jz, info_dict) where info contains quantum observables.*
- `compute_spinor_amplitude` (line 370) - *Compute complex amplitude from wavefunction for helicity analysis.*
- `__init__` (line 380)
- `parse_file` (line 383)
- `_parse_row` (line 396)
- `__init__` (line 439)
- `compute_quantum_helix` (line 443) - *Compute helical trajectory using actual DiracBackend evolution.*
- `create_visualization` (line 474)
- `_create_detector` (line 566)
- `_create_explosion` (line 581)
- `__init__` (line 617)
- `fetch_data` (line 630)
- `load_events` (line 645)
- `analyze_with_quantum_backends` (line 672) - *Run analysis using actual quantum backends.*
- `generate_visualization` (line 728)
- `run` (line 734)

#### `higgs_quantum_analysis.py`
**Path:** `higgs_quantum_analysis.py`

**Classs:**
- `LeptonType` (line 72)
- `EventType` (line 78)
- `Config` (line 86)
- `FourMomentum` (line 125)
- `Lepton` (line 164)
- `Event` (line 183)
- `QuantumSpinorProcessor` (line 205) - *Uses the ACTUAL DiracBackend from quantum_computer.py for spinor calculations.

This is NOT a fake implementation - it uses the neural network
spectral layers trained (or randomly initialized) for Dirac spinor evolution.*
- `EventParser` (line 377) - *Parse CMS CSV files with actual column names.*
- `Visualizer` (line 436) - *3D visualization using Plotly with quantum-processed trajectories.*
- `HiggsQuantumAnalysis` (line 612) - *Main analysis class using quantum backends from quantum_computer.py*

**Functions:**
- `_make_logger` (line 57)
- `main` (line 747)
- `__post_init__` (line 136)
- `from_energy_momentum` (line 151)
- `__add__` (line 154)
- `pt` (line 171)
- `eta` (line 173)
- `phi` (line 175)
- `energy` (line 177)
- `mass` (line 179)
- `__post_init__` (line 193)
- `check_higgs` (line 200)
- `__init__` (line 213)
- `_precompute_momentum_grids` (line 247) - *Precompute k-space grids for Dirac equation.*
- `momentum_to_spinor_wavefunction` (line 255) - *Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)
suitable for processing by the DiracBackend.

The wavefunction encodes the momentum as a plane wave with the correct
relativistic dispersion relation.*
- `evolve_with_dirac_backend` (line 296) - *Evolve a wavefunction using the DiracBackend neural network.

This applies the actual spectral layers from the trained (or random)
Dirac network.*
- `evolve_with_schrodinger_backend` (line 308) - *Evolve using the SchrodingerBackend neural network.*
- `evolve_with_hamiltonian_backend` (line 315) - *Evolve using the HamiltonianBackend neural network.*
- `compute_dirac_current` (line 322) - *Compute the Dirac current using the actual neural network backends.

Returns (jx, jy, jz, info_dict) where info contains quantum observables.*
- `compute_spinor_amplitude` (line 370) - *Compute complex amplitude from wavefunction for helicity analysis.*
- `__init__` (line 380)
- `parse_file` (line 383)
- `_parse_row` (line 396)
- `__init__` (line 439)
- `compute_quantum_helix` (line 443) - *Compute helical trajectory using actual DiracBackend evolution.*
- `create_visualization` (line 474)
- `_create_detector` (line 566)
- `_create_explosion` (line 581)
- `__init__` (line 617)
- `fetch_data` (line 630)
- `load_events` (line 645)
- `analyze_with_quantum_backends` (line 672) - *Run analysis using actual quantum backends.*
- `generate_visualization` (line 728)
- `run` (line 734)

#### `molecular_sim.py`
**Path:** `molecular_sim.py`

**Classs:**
- `MoleculeData` (line 35)
- `ExactJWEnergy` (line 187) - *Evaluador exacto usando OpenFermion JW.*
- `SurrogateEnergy` (line 265) - *Backend neuronal con calibración.*
- `VQEResult` (line 371)
- `VQESolver` (line 390)

**Functions:**
- `_make_logger` (line 18)
- `_h2_sto3g_pyscf` (line 41) - *H2 con datos PySCF.*
- `_h2_sto3g_hardcoded` (line 75) - *H2 con datos hardcodeados.*
- `build_jw_hamiltonian_of` (line 111) - *Construye Hamiltoniano JW usando OpenFermion correctamente.
Usa MolecularData de OpenFermion para asegurar consistencia.*
- `prepare_hf` (line 304)
- `_sd_indices` (line 312)
- `uccsd` (line 321) - *UCCSD ansatz.

- Singles: cadena JW estándar, conserva número de partículas.
- Doubles: rotación Givens en el subespacio {|HF⟩, |exc⟩} determinado
  dinámicamente desde las amplitudes (robusto ante cambio de convención).
- theta=0 para cualquier parámetro → circuito vacío → identidad exacta.*
- `__init__` (line 190)
- `_to_scalar` (line 203) - *(dim,2,G,G) → (dim,2)  ó  (dim,2) → (dim,2).
Integra la función de onda espacial sobre la grilla manteniendo
la estructura re/im que necesita el evaluador JW.*
- `_verify_hf` (line 213)
- `_apply` (line 223)
- `_evaluate` (line 243)
- `__call__` (line 257)
- `__init__` (line 268)
- `calibrate` (line 277)
- `cost_with_barrier` (line 290)
- `__repr__` (line 377)
- `__init__` (line 391)
- `_run` (line 395)
- `run` (line 403)
- `cost` (line 451)

#### `orbital_visualizer2.py`
**Path:** `orbital_visualizer2.py`

**Classs:**
- `Config` (line 44)
- `WavefunctionCalculator` (line 83) - *Calculates hydrogen atom wavefunctions.*
- `HamiltonianNNProcessor` (line 126) - *Uses YOUR TRAINED MODEL for calculations.*
- `MonteCarloSampler` (line 160) - *Monte Carlo sampling for orbital visualization.*
- `OrbitalVisualizer` (line 265) - *HIGH RESOLUTION visualization - NOT 16x16!*

**Functions:**
- `main` (line 433)
- `radial_wavefunction` (line 87)
- `spherical_harmonic_real` (line 97)
- `psi_on_grid` (line 107)
- `__init__` (line 129)
- `is_model_loaded` (line 133)
- `compute_expected_energy` (line 136)
- `__init__` (line 163)
- `find_max_probability` (line 166)
- `sample` (line 195)
- `visualize` (line 268)
- `_plotly` (line 396)

#### `polarizability_v3.py`
**Path:** `polarizability_v3.py`

**Classs:**
- `VQEResult` (line 51)
- `StarkEvaluator` (line 163)
- `DipoleOperatorBuilder` (line 215)
- `PolarizabilityCalculator` (line 249)

**Functions:**
- `_sd_indices` (line 58)
- `_run_circuit` (line 67)
- `givens_single_excitation` (line 73) - *Apply a particle-conserving single excitation rotation between 
qubits o (occupied) and v (virtual).

Rotates in the {|1_o 0_v⟩, |0_o 1_v⟩} subspace:
    |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩
    |0_o 1_v⟩ → -sin(θ)|1_o 0_v⟩ + cos(θ)|0_o 1_v⟩

For adjacent qubits: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)
For non-adjacent: SWAP chain to make adjacent, apply, SWAP back.*
- `particle_conserving_ansatz` (line 121) - *Particle-conserving UCCSD-like ansatz:
- Singles: Givens rotations (correct particle conservation)
- Doubles: reuse uccsd's double excitation (works correctly)*
- `__init__` (line 164)
- `_to_scalar` (line 173)
- `_apply_pauli` (line 177)
- `eval_dipole` (line 196)
- `__call__` (line 209)
- `__init__` (line 216)
- `__init__` (line 250)
- `_evaluator` (line 270)
- `_get_state` (line 274)
- `_diagnose` (line 279)
- `_optimize` (line 289)
- `run` (line 315)
- `cost` (line 302)

#### `qc_dashboard.py`
**Path:** `qc_dashboard.py`

**Classs:**
- `DashboardConfig` (line 55) - *Centralised configuration for the dashboard.*
- `GateItem` (line 120) - *A gate placed in the circuit builder.*
- `SnapshotData` (line 130) - *Quantum state snapshot for visualisation.*
- `H2VQESolver` (line 147) - *Self-contained H2 VQE solver using the hardcoded STO-3G Hamiltonian.

The Hamiltonian in the Jordan-Wigner 2-qubit active space:
    H = E_nuc + hZZ * Z0Z1 + hXX * X0X1 + hYY * Y0Y1

Reference energies:
    E_HF = -1.11675928 Ha,  E_FCI = -1.13728383 Ha,  E_nuc = 0.71996899 Ha

Usage
-----
    solver = H2VQESolver()
    result = solver.run_vqe()
    print(result["energy"])  # converged VQE energy

    landscape = solver.energy_landscape()
    print(landscape["bond_lengths"], landscape["energies"])*
- `VisualisationEngine` (line 340) - *Renders figures from quantum state snapshots using matplotlib.*
- `SimulatorBackend` (line 725) - *Thin wrapper around the QC framework simulator for dashboard use.*
- `Plotly3DEngine` (line 979) - *Optional Plotly-based 3D visualisations.*
- `RealOrbitalEngine` (line 1151) - *Wrapper around the repo's real orbital_visualizer2.py scripts.*
- `BrutalVizEngine` (line 1269) - *Wrapper around the repo's real quantum_dash.py and quantum_3dview.py.*
- `BackendComparator` (line 1344) - *Compare circuit execution across all available backends.*
- `DashboardApp` (line 1371) - *Streamlit-based interactive quantum playground.*
- `_FakeConfig` (line 1176)

**Functions:**
- `_capture_mpl_fig` (line 1133) - *Run a function that creates a matplotlib figure and capture it as PNG bytes.*
- `main` (line 1994) - *Launch the Streamlit dashboard.*
- `__init__` (line 166)
- `_pauli_operators` (line 173)
- `_pauli_string_matrix` (line 189)
- `_build_hamiltonian` (line 206) - *Build H2 Hamiltonian matrix for given bond length (Angstrom).

The coefficients scale with bond length to reproduce the Morse-like well.*
- `_ansatz_state` (line 229) - *UCC-like ansatz for H2: |psi(theta)> = cos(theta)*|10> + sin(theta)*|01>.*
- `_energy` (line 241)
- `run_vqe` (line 248)
- `energy_landscape` (line 293) - *Sweep bond length and return VQE energy curve.*
- `orbital_wavefunction` (line 318) - *Compute hydrogen 1s orbital wavefunction along the internuclear axis.*
- `__init__` (line 343)
- `_init_plotting` (line 349)
- `available` (line 362)
- `render_full_dashboard` (line 365)
- `render_entropy_chart` (line 403)
- `render_entanglement_profile` (line 429) - *Entropy vs cut position for the latest snapshot.*
- `render_vqe_convergence` (line 474)
- `render_energy_landscape` (line 501)
- `render_orbital_plot` (line 533)
- `render_entropy_scaling` (line 562)
- `_render_probabilities` (line 589)
- `_render_bloch_sphere` (line 606)
- `_render_phase_space` (line 635)
- `render_orbital_2d_projections` (line 670)
- `_hex_to_rgb` (line 715)
- `__init__` (line 728)
- `_init_framework` (line 737)
- `_get_mps_gate_registry` (line 760)
- `_get_sv_gate_registry` (line 768)
- `execute_circuit` (line 775)
- `_mps_execute` (line 788)
- `_sv_execute` (line 839)
- `_snapshot_from_mps` (line 886)
- `_snapshot_from_sv` (line 898)
- `_compute_bloch_mps` (line 917)
- `_synthetic_execute` (line 933)
- `__init__` (line 982)
- `_init_plotly` (line 988)
- `available` (line 998)
- `render_bloch_3d` (line 1001)
- `render_probability_3d` (line 1049)
- `render_state_3d` (line 1080) - *3D scatter plot: X=real, Y=imaginary, Z=probability.*
- `__init__` (line 1154)
- `_init_real` (line 1163)
- `available` (line 1210)
- `entangled_available` (line 1214)
- `sample` (line 1217)
- `render_to_bytes` (line 1222)
- `sample_entangled` (line 1240)
- `render_entangled_to_bytes` (line 1250)
- `__init__` (line 1272)
- `_init_real` (line 1279)
- `dash_available` (line 1297)
- `hologram_available` (line 1301)
- `run_brutal_viz` (line 1304)
- `render_hologram` (line 1327)
- `__init__` (line 1347)
- `run_comparison` (line 1351)
- `__init__` (line 1374)
- `run` (line 1384)
- `_ensure_streamlit` (line 1399)
- `_init_session` (line 1408)
- `_render_ui` (line 1432)
- `_render_sidebar_controls` (line 1447)
- `_render_main_panel` (line 1535)
- `_render_playground_tab` (line 1563)
- `_render_qasm_tab` (line 1606)
- `_render_entropy_tab` (line 1669)
- `_render_entanglement_tab` (line 1688)
- `_render_molecule_tab` (line 1732)
- `_render_3d_tab` (line 1803)
- `_render_orbital_tab` (line 1855)
- `_auto_run` (line 1958)
- `_build_qasm_from_gates` (line 1969)
- `_parse_orb` (line 1936)

#### `qc_integration.py`
**Path:** `qc_integration.py`

**Classs:**
- `IntegrationConfig` (line 77) - *Centralised configuration for the integration bridge.

All tunable parameters live here -- no hardcoded values in logic.*
- `GateInstruction` (line 130) - *A single quantum gate instruction.*
- `CircuitIR` (line 146) - *Intermediate representation of a quantum circuit.

This is the lingua franca between the QC framework and external formats.*
- `IQCAdapter` (line 181) - *Interface for a format-specific adapter (Interface Segregation).*
- `OpenQasmAdapter` (line 226) - *Adapter for OpenQASM 2.0 format.*
- `QiskitAdapter` (line 358) - *Adapter for Qiskit QuantumCircuit format.*
- `PennyLaneAdapter` (line 455) - *Adapter for PennyLane format.*
- `FrameworkAdapter` (line 557) - *Converts between CircuitIR and the actual QC framework circuit types.

Supports both:
  - quantum_framework_core.MPSQuantumComputer / QuantumCircuit (MPS)
  - quantum_computer.QuantumComputer / QuantumCircuit (statevector)*
- `StandardCircuitFactory` (line 659) - *Build CircuitIR instances for common quantum algorithms.*
- `IntegrationBridge` (line 727) - *Facade that exposes all format conversions through a single API.

Usage
-----
    bridge = IntegrationBridge()
    cir = StandardCircuitFactory.bell_state()

    # OpenQASM
    qasm_str = bridge.export_qasm(cir)
    assert isinstance(qasm_str, str)

    cir_restored = bridge.import_qasm(qasm_str)
    assert cir_restored.n_qubits == cir.n_qubits

    # Qiskit (if installed)
    if bridge._qiskit_available:
        qc_qiskit = bridge.to_qiskit(cir)
        cir_back = bridge.from_qiskit(qc_qiskit)

    # PennyLane (if installed)
    if bridge._pennylane_available:
        qnode = bridge.to_pennylane(cir)
        result = qnode()*

**Functions:**
- `main` (line 851) - *Command-line interface for the integration bridge.*
- `__post_init__` (line 115)
- `__post_init__` (line 137)
- `num_qubits` (line 141)
- `append` (line 156)
- `__len__` (line 164)
- `__repr__` (line 167)
- `export` (line 185) - *Export a CircuitIR to the target format.*
- `import_` (line 189) - *Import from the target format into a CircuitIR.*
- `__init__` (line 238)
- `export` (line 243)
- `import_` (line 277) - *Parse an OpenQASM 2.0 string into a CircuitIR.*
- `_param_names_for_gate` (line 339)
- `__init__` (line 361)
- `export` (line 364)
- `import_` (line 379)
- `_ensure_qiskit` (line 403)
- `_build_qiskit_method_map` (line 414)
- `_build_reverse_gate_map` (line 433)
- `__init__` (line 458)
- `export` (line 461)
- `import_` (line 483)
- `_ensure_pennylane` (line 506)
- `_build_gate_ops` (line 517)
- `_build_reverse_ops` (line 535)
- `__init__` (line 565)
- `to_circuit_ir` (line 568) - *Extract a CircuitIR from a framework circuit object.*
- `from_circuit_ir` (line 584) - *Build a framework circuit object from a CircuitIR.*
- `_detect_framework` (line 598)
- `_from_mps_circuit` (line 607)
- `_to_mps_circuit` (line 623)
- `_from_sv_circuit` (line 631)
- `_to_sv_circuit` (line 647)
- `bell_state` (line 663)
- `ghz_state` (line 670)
- `qft` (line 678)
- `w_state` (line 693)
- `grover` (line 701)
- `__init__` (line 753)
- `_init_optional_adapters` (line 763)
- `export_qasm` (line 780)
- `import_qasm` (line 783)
- `to_qiskit` (line 788)
- `from_qiskit` (line 793)
- `to_pennylane` (line 800)
- `from_pennylane` (line 805)
- `to_circuit_ir` (line 812)
- `from_circuit_ir` (line 819)
- `export_qasm_from_framework` (line 828)
- `import_qasm_to_framework` (line 837)
- `circuit_fn` (line 467)

#### `quantum_3dview.py`
**Path:** `quantum_3dview.py`

**Classs:**
- `BrutalTheme` (line 32)
- `BrutalConfig` (line 39)
- `QuantumHologram` (line 84) - *Visualizador holográfico 3D de estados cuánticos*
- `QuantumNeuralTopology` (line 403) - *Visualiza la topología interna de las redes neuronales cuánticas*
- `QuantumSonification` (line 500) - *Convierte estados cuánticos en audio para percepción alternativa*
- `BrutalDashboard` (line 560) - *Dashboard interactivo completo con todas las visualizaciones brutales*

**Functions:**
- `demo_brutal` (line 614) - *Demostración de visualización brutal*
- `create_synthetic_snapshots` (line 653) - *Crea datos sintéticos para demostración*
- `colors` (line 52)
- `__init__` (line 87)
- `create_amplitude_hologram` (line 92) - *Crea visualización holográfica de amplitudes en 3D con efecto de partículas*
- `_add_holographic_field` (line 148) - *Añade campo de amplitudes 3D con efecto de partículas flotantes*
- `_add_entropy_trails` (line 245) - *Añade trazas de entropía con efecto de cola luminosa*
- `_add_bloch_sphere_holographic` (line 292) - *Esfera de Bloch con efectos de holograma y partículas orbitales*
- `_interpolate_color` (line 389) - *Interpola entre dos colores hex*
- `__init__` (line 406)
- `create_topology_map` (line 410) - *Crea mapa 3D de la arquitectura neuronal con activaciones en tiempo real*
- `_extract_layers` (line 477) - *Extrae información de capas del modelo PyTorch*
- `_generate_quantum_topology` (line 490) - *Genera topología representativa de backend cuántico*
- `__init__` (line 503)
- `state_to_audio` (line 506) - *Convierte un estado cuántico en onda de audio
- Amplitudes controlan volumen
- Fases controlan paneo estéreo
- Probabilidades controlan frecuencia*
- `_adsr_envelope` (line 538) - *Genera envolvente ADSR proporcional a la intensidad del estado*
- `__init__` (line 563)
- `generate_full_report` (line 570) - *Genera reporte completo con múltiples visualizaciones*
- `_save_audio` (line 605) - *Guarda audio como WAV*
- `hex_to_rgb` (line 391)
- `rgb_to_hex` (line 395)

#### `quantum_computer.py`
**Path:** `quantum_computer.py`

**Classs:**
- `SimulatorConfig` (line 74) - *Global configuration for the quantum computer simulator.

grid_size, hidden_dim, expansion_dim, num_spectral_layers must match
the values used when training the checkpoint files.*
- `SpectralLayer` (line 105) - *Spectral convolution in frequency domain.

Learns complex kernels that modulate Fourier coefficients.
Architecture is identical to the training scripts.*
- `HamiltonianBackboneNet` (line 144) - *Hamiltonian backbone: single-channel field -> H|psi>.
Shared by all physics backends as the H operator.*
- `SchrodingerSpectralNet` (line 171) - *Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction.*
- `DiracSpectralNet` (line 200) - *Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor.*
- `GammaMatrices` (line 233) - *Dirac gamma matrices in Dirac or Weyl representation.*
- `JointHilbertState` (line 279) - *Joint quantum state of n qubits in the full 2^n dimensional Hilbert space.

The state is stored as a tensor of shape (2^n, 2, G, G):
    - dim 0: computational basis index k in {0, ..., 2^n - 1}
             bit j of k is the state of qubit j (MSB = qubit 0)
    - dim 1: channel 0 = real part, channel 1 = imaginary part
    - dim 2: spatial x (G grid points)
    - dim 3: spatial y (G grid points)

Each amplitude alpha_k is a spatial wavefunction. The overall quantum
amplitude for basis state |k> is the complex field alpha_k(x,y).
The Born probability of measuring |k> is:

    P(k) = integral |alpha_k(x,y)|^2 dx dy
         = sum_{x,y} (alpha_k_real^2 + alpha_k_imag^2)

normalized so that sum_k P(k) = 1.

This representation correctly supports:
    - Superposition: multiple k indices have non-zero amplitude
    - Entanglement: amplitudes do not factorize across qubits
    - Coherent multi-qubit gates: exact permutation and mixing of amplitudes*
- `PotentialGenerator` (line 390) - *Spatial potentials for eigenstate initialization.*
- `JointStateFactory` (line 474) - *Builds JointHilbertState tensors for common initial conditions.*
- `IPhysicsBackend` (line 511) - *Abstract physics backend for spatial wavefunction evolution.*
- `HamiltonianBackend` (line 523) - *Physics backend driven by the Hamiltonian neural network.

Performs first-order Schrodinger time evolution:
    psi(t+dt) = psi(t) - i*dt*H*psi(t)*
- `SchrodingerBackend` (line 591) - *Physics backend driven by the Schrodinger network.

Uses the learned 2-channel spectral network for wavefunction propagation.
Falls back to HamiltonianBackend if checkpoint is unavailable.*
- `DiracBackend` (line 640) - *Physics backend driven by the Dirac network.

Expands each (2,G,G) amplitude to a 4-component spinor, propagates
via the Dirac network, then projects back to (2,G,G).*
- `IQuantumGate` (line 844) - *Abstract quantum gate operating on the joint Hilbert space.*
- `HadamardGate` (line 863) - *H = [[1,1],[1,-1]] / sqrt(2).*
- `PauliXGate` (line 878) - *X = [[0,1],[1,0]].*
- `PauliYGate` (line 892) - *Y = [[0,-i],[i,0]].*
- `PauliZGate` (line 906) - *Z = [[1,0],[0,-1]].*
- `SGate` (line 920) - *S = [[1,0],[0,i]].*
- `TGate` (line 934) - *T = [[1,0],[0,e^{i*pi/4}]].*
- `RxGate` (line 949) - *Rx(theta) = exp(-i*theta/2 * X).*
- `RyGate` (line 965) - *Ry(theta) = exp(-i*theta/2 * Y).*
- `RzGate` (line 981) - *Rz(theta) = exp(-i*theta/2 * Z).*
- `CNOTGate` (line 998) - *CNOT: |ctrl tgt> -> |ctrl, ctrl XOR tgt>.

4x4 matrix (|00>,|01>,|10>,|11>):
    |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>*
- `CZGate` (line 1022) - *CZ: applies phase -1 to |11>.

4x4 matrix: diag(1, 1, 1, -1).*
- `SWAPGate` (line 1045) - *SWAP: exchanges two qubits.

4x4 matrix: |01>->|10>, |10>->|01>, others unchanged.*
- `ToffoliGate` (line 1068) - *Toffoli (CCX): flips target iff both controls are |1>.

Exact amplitude permutation in the 8-element 3-qubit subspace.*
- `MCZGate` (line 1098) - *Multi-Controlled Z gate: applies phase -1 to the single basis state
where ALL qubits in targets are |1>.

This is the exact oracle primitive needed by Grover's algorithm.
For n target qubits it marks the state |11...1> with a global phase of -1
and leaves all other basis states unchanged.

Implementation: iterate over all basis states k; if every bit
corresponding to a qubit in targets is set to 1, negate that amplitude
(multiply real and imaginary parts by -1).

targets: list of qubit indices that must all be |1> for the phase flip.*
- `EvolveGate` (line 1130) - *Free Hamiltonian evolution applied to every amplitude in the joint state.

Uses the active physics backend.
params: {"dt": float, "steps": int}*
- `CircuitInstruction` (line 1186) - *Single gate instruction.*
- `QuantumCircuit` (line 1193) - *Ordered sequence of quantum gate instructions.

Pure data structure: stores instructions, does not execute them.*
- `MeasurementResult` (line 1279) - *Non-destructive Born-rule measurement of the full register.

Contains the complete probability distribution over all 2^n basis states,
per-qubit marginals, and Bloch vectors. The state is never modified.*
- `QuantumComputer` (line 1333) - *Collapse-free quantum computer simulator with joint Hilbert space.

The n-qubit state is stored as a (2^n, 2, G, G) tensor. All gates
operate via exact unitary transformations on amplitude pairs. Measurement
is non-destructive Born-rule readout — the state is never collapsed.

Backends:
    "hamiltonian" : Hamiltonian NN spectral operator
    "schrodinger" : Schrodinger evolution network
    "dirac"       : Dirac relativistic spinor network

Usage:
    config = SimulatorConfig(
        hamiltonian_checkpoint="weights/latest.pth",
        schrodinger_checkpoint="weights/schrodinger_crystal_final.pth",
        dirac_checkpoint="weights/dirac_phase5_latest.pth",
    )
    qc = QuantumComputer(config)
    circuit = QuantumCircuit(2)
    circuit.h(0).cnot(0, 1)
    result = qc.run(circuit, backend="schrodinger")
    print(result)
    # Most probable: |00> P=0.5  or  |11> P=0.5
    # Shannon entropy: 1.0000 bits  <- true entanglement*

**Functions:**
- `_make_logger` (line 58) - *Create a module-level logger with a consistent formatter.*
- `_solve_eigenstate` (line 440) - *Solve the 1D marginal Hamiltonian and return the n-th eigenstate
as a normalized 2-channel (2, G, G) real tensor.*
- `_build_basis_amplitude` (line 461) - *Build the (2, G, G) spatial wavefunction for amplitude at basis index basis_idx.

Each computational basis state gets its own spatial eigenstate profile.
The excitation level is proportional to the popcount of the basis index.*
- `_single_qubit_unitary` (line 739) - *Apply a 2x2 unitary u to qubit j in the joint Hilbert space.

For each pair of basis states (k0, k1) that differ only in bit j:
    alpha_{k0}' = u[0,0]*alpha_{k0} + u[0,1]*alpha_{k1}
    alpha_{k1}' = u[1,0]*alpha_{k0} + u[1,1]*alpha_{k1}

Complex scalar * (2,G,G) amplitude:
    (a+ib)(psi_r + i*psi_i) = (a*psi_r - b*psi_i) + i*(a*psi_i + b*psi_r)

This is exact, preserves unitarity, and correctly creates superpositions.*
- `_two_qubit_unitary` (line 789) - *Apply a 4x4 unitary in the {|00>,|01>,|10>,|11>} subspace of (ctrl, tgt).

For each group of 4 basis states sharing all bits except ctrl and tgt,
apply the 4x4 unitary to the amplitude quadruplet.

Ordering within the 4x4 block: |00>=0, |01>=1, |10>=2, |11>=3
(first bit = ctrl, second bit = tgt).

This correctly implements CNOT, CZ, SWAP and any 2-qubit gate.*
- `register_gate` (line 1179) - *Register a custom gate without modifying existing code (Open/Closed Principle).*
- `_check` (line 1602) - *Print PASS/FAIL for a single assertion. Returns True if passed.*
- `run_phase_tests` (line 1609) - *Property-based test suite for quantum phase coherence and unitarity.

Each test has a known exact analytic answer derived from the unitary
algebra. Tests are designed to be sensitive to phase errors — circuits
where wrong relative phases produce DIFFERENT probabilities, not just
different phases that cancel out at measurement.

Returns the number of failed tests.*
- `_demo` (line 1836) - *Run demo suite validating entanglement on all backends.*
- `__init__` (line 113)
- `forward` (line 124) - *Apply spectral convolution via RFFT2.*
- `__init__` (line 150)
- `forward` (line 159) - *Accepts (G,G), (1,G,G), or (B,1,G,G). Returns squeezed output.*
- `__init__` (line 176)
- `forward` (line 188) - *(2,G,G) or (B,2,G,G) -> same shape.*
- `__init__` (line 205)
- `forward` (line 217) - *(8,G,G) or (B,8,G,G) -> same shape.*
- `__init__` (line 236)
- `_init_matrices` (line 241)
- `to` (line 272) - *Move all matrices to device.*
- `__init__` (line 305)
- `normalize_` (line 317) - *In-place normalization: sum_k P(k) = 1.*
- `probabilities` (line 323) - *Return (2^n,) tensor of Born probabilities P(k) for each basis state.*
- `marginal_probability_one` (line 328) - *Marginal Born probability P(qubit_j = |1>).

Sums P(k) over all basis states k where bit j == 1.
Bit ordering: qubit 0 is the MSB of k.*
- `most_probable_basis_state` (line 343) - *Return the index k with the highest probability.*
- `bloch_vector` (line 347) - *Compute the reduced Bloch vector for qubit j by partial trace.

rho_j[0,0] = P(qubit=0), rho_j[1,1] = P(qubit=1)
rho_j[0,1] = sum_{pairs} alpha_{k0}^* alpha_{k1} (off-diagonal coherence)
bx = 2 Re(rho_j[0,1]), by = -2 Im(rho_j[0,1]), bz = P(0) - P(1)*
- `clone` (line 385) - *Return a deep copy.*
- `__init__` (line 393)
- `_grid` (line 397)
- `harmonic` (line 402) - *V = k/2 * r^2.*
- `double_well` (line 410) - *Double-well along x.*
- `coulomb` (line 417) - *Coulomb-like V ~ -1/r.*
- `periodic_lattice` (line 424) - *Periodic cosine lattice.*
- `mixed` (line 429) - *Dirichlet-weighted mixture of all four potentials.*
- `__init__` (line 477)
- `_empty` (line 480)
- `all_zeros` (line 484) - *Initialize register in |00...0>.*
- `basis_state` (line 492) - *Initialize register in computational basis state |k>.*
- `from_bitstring` (line 502) - *Initialize in the basis state given by binary string.*
- `evolve_amplitude` (line 515) - *Evolve a single (2, G, G) wavefunction by dt under H.*
- `apply_phase` (line 519) - *Apply global phase e^{i*phi} to a (2, G, G) amplitude.*
- `__init__` (line 531)
- `_load` (line 539)
- `_precompute_laplacian` (line 558)
- `_apply_h` (line 565)
- `evolve_amplitude` (line 573) - *dpsi/dt = -i H psi  =>  psi' = psi + dt * (-i H psi) = psi + dt*(H_i*r - H_r*i).*
- `apply_phase` (line 585)
- `__init__` (line 599)
- `_load` (line 606)
- `evolve_amplitude` (line 628)
- `apply_phase` (line 636)
- `__init__` (line 648)
- `_load` (line 657)
- `_precompute_dirac` (line 679)
- `_pack` (line 690)
- `_unpack` (line 701)
- `_analytical_dirac` (line 707)
- `evolve_amplitude` (line 722)
- `apply_phase` (line 735)
- `name` (line 849) - *Gate identifier.*
- `apply` (line 853) - *Apply gate to joint state, return new joint state.*
- `name` (line 867)
- `apply` (line 870)
- `name` (line 882)
- `apply` (line 885)
- `name` (line 896)
- `apply` (line 899)
- `name` (line 910)
- `apply` (line 913)
- `name` (line 924)
- `apply` (line 927)
- `name` (line 938)
- `apply` (line 941)
- `name` (line 953)
- `apply` (line 956)
- `name` (line 969)
- `apply` (line 972)
- `name` (line 985)
- `apply` (line 988)
- `name` (line 1007)
- `apply` (line 1010)
- `name` (line 1030)
- `apply` (line 1033)
- `name` (line 1053)
- `apply` (line 1056)
- `name` (line 1076)
- `apply` (line 1079)
- `name` (line 1115)
- `apply` (line 1118)
- `name` (line 1139)
- `apply` (line 1142)
- `__init__` (line 1200)
- `h` (line 1206)
- `x` (line 1209)
- `y` (line 1212)
- `z` (line 1215)
- `s` (line 1218)
- `t` (line 1221)
- `rx` (line 1224)
- `ry` (line 1227)
- `rz` (line 1230)
- `cnot` (line 1233)
- `cx` (line 1236)
- `cz` (line 1239)
- `swap` (line 1242)
- `toffoli` (line 1245)
- `ccx` (line 1248)
- `evolve` (line 1251)
- `barrier` (line 1254)
- `_append` (line 1257)
- `depth` (line 1265)
- `__len__` (line 1268)
- `__repr__` (line 1271)
- `probabilities` (line 1293) - *Alias: marginal P(|1>) per qubit index.*
- `most_probable_bitstring` (line 1297) - *Return the bitstring with the highest probability.*
- `expectation_z` (line 1301) - *<Z>_j = P(0) - P(1) in [-1, +1].*
- `entropy` (line 1305) - *Shannon entropy of the full probability distribution in bits.*
- `__repr__` (line 1313)
- `__init__` (line 1361)
- `_select_backend` (line 1375)
- `_state_to_result` (line 1380)
- `run` (line 1388) - *Execute a quantum circuit on the joint Hilbert space.

Args:
    circuit:        The QuantumCircuit to execute.
    backend:        Physics backend name.
    initial_states: Optional {qubit_idx: "0" or "1"}.

Returns:
    Non-destructive MeasurementResult with full distribution.*
- `run_with_state_snapshots` (line 1420) - *Execute circuit with non-destructive probability snapshots.

The state is never collapsed between snapshots.*
- `bell_state` (line 1449) - *|Phi+> = (|00> + |11>) / sqrt(2).

Expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit.*
- `ghz_state` (line 1459) - *(|00...0> + |11...1>) / sqrt(2).

Expected: P(|00...0>)=P(|11...1>)=0.5, all others 0.*
- `quantum_fourier_transform` (line 1471) - *QFT on |00...0>. Standard H + controlled-Rz decomposition.*
- `grover_oracle_search` (line 1481) - *Grover's search algorithm with correct phase oracle and diffusion operator.

The optimal number of iterations is floor(pi/4 * sqrt(2^n)) which gives
the highest probability of measuring the target state.

Oracle construction for target |t_0 t_1 ... t_{n-1}>:
    1. Apply X to every qubit i where t_i == '0'.
       This maps the target bitstring to |11...1>.
    2. Apply MCZ on all n qubits.
       MCZ flips the phase of |11...1> -> exactly the target state
       (after the X conjugation) gets phase -1.
    3. Undo the X gates from step 1.

Diffusion operator (inversion about the mean):
    H^n  X^n  MCZ  X^n  H^n

Both steps use MCZGate which applies phase -1 to the unique basis state
where all specified qubits are |1>. This is exact for any n.

Args:
    n_qubits:        Number of qubits.
    target_bitstring: Binary string of length n_qubits.
    backend:         Physics backend name.
    n_iterations:    Number of Grover iterations. Defaults to
                     max(1, round(pi/4 * sqrt(2^n_qubits))).

Returns:
    MeasurementResult. The target bitstring should have the highest
    probability after the optimal number of iterations.*
- `variational_ansatz` (line 1559) - *Hardware-efficient ansatz: Ry layers + CNOT chain. len(thetas)=n_qubits*n_layers.*
- `teleportation` (line 1572) - *3-qubit teleportation protocol.

q0 prepared in Ry(pi/3). q2 should match q0's state after corrections.*
- `deutsch_jozsa` (line 1585) - *Deutsch-Jozsa: constant -> all inputs |0>, balanced -> at least one |1>.*

#### `quantum_dash.py`
**Path:** `quantum_dash.py`

**Classs:**
- `ColorScheme` (line 106)
- `BrutalistConfig` (line 115)
- `QuantumSnapshot` (line 239)
- `BackendComparison` (line 255)
- `VisualizationOutput` (line 268)
- `IVisualComponent` (line 275)
- `ProbabilityVisualizer` (line 281)
- `BlochSphereVisualizer` (line 340)
- `PhaseSpaceVisualizer` (line 447)
- `EntropyVisualizer` (line 509)
- `BackendComparisonVisualizer` (line 583)
- `FidelityVisualizer` (line 633)
- `QuantumStateAnalyzer` (line 674)
- `StandardCircuits` (line 754)
- `CircuitExecutor` (line 806)
- `FigureBuilder` (line 873)
- `QuantumVisualizer` (line 958)

**Functions:**
- `_make_logger` (line 91)
- `main` (line 1199)
- `colors` (line 168)
- `plotly_template` (line 234)
- `render` (line 277)
- `render` (line 282)
- `_render_empty` (line 318)
- `_generate_colors` (line 326)
- `render` (line 341)
- `_render_empty_sphere` (line 375)
- `_draw_sphere_wireframe` (line 381)
- `_draw_axes` (line 402)
- `_draw_bloch_vector` (line 417)
- `_draw_uncertainty_ring` (line 431)
- `render` (line 448)
- `_render_empty` (line 495)
- `render` (line 510)
- `_render_empty` (line 556)
- `_interpolate_colors` (line 564)
- `_hex_to_rgb` (line 578)
- `render` (line 584)
- `_render_empty` (line 624)
- `render` (line 634)
- `_render_empty` (line 665)
- `__init__` (line 675)
- `compute_probabilities` (line 678)
- `compute_phases` (line 684)
- `compute_entropy` (line 696)
- `compute_bloch_vectors` (line 704)
- `create_snapshot` (line 713)
- `bell_state` (line 756)
- `ghz_state` (line 760)
- `qft` (line 767)
- `grover_oracle` (line 782)
- `grover_diffusion` (line 794)
- `__init__` (line 807)
- `execute_sequence` (line 812)
- `compare_backends` (line 835)
- `__init__` (line 874)
- `build_full_figure` (line 883)
- `build_summary_figure` (line 923)
- `__init__` (line 959)
- `_initialize` (line 966)
- `_init_quantum_computer` (line 976)
- `visualize_bell_state` (line 1010)
- `visualize_ghz_state` (line 1016)
- `visualize_qft` (line 1022)
- `visualize_grover` (line 1028)
- `_execute_and_visualize` (line 1043)
- `_create_synthetic_snapshots` (line 1086)
- `_create_synthetic_comparisons` (line 1129)
- `_save_data` (line 1153)
- `run_all` (line 1174)
- `_print_summary` (line 1189)

#### `quantum_framework_core.py`
**Path:** `quantum_framework_core.py`

**Classs:**
- `HilbertPhase` (line 62) - *Phase classification for Hilbert space compression.*
- `FrameworkConfig` (line 72) - *Unified configuration for the quantum simulation framework.
Loads from TOML file with fallback to sensible defaults.*
- `AtomData` (line 218) - *Atomic data structure.*
- `MoleculeData` (line 230) - *Molecular data structure.*
- `OrbitalData` (line 250) - *Atomic orbital data structure.*
- `ConfigLoader` (line 259) - *Configuration loader that parses TOML files and provides
access to atoms, molecules, and orbitals data.*
- `ITensorNetwork` (line 435) - *Abstract interface for tensor network quantum states.*
- `MPSCore` (line 480) - *Matrix Product State core tensor A^{[k]}_{i_k} with bond indices.

Shape: (chi_left, d, chi_right) where d=2 for qubits.
Memory per core: O(chi^2 * d) = O(chi^2)*
- `MPSState` (line 554) - *Matrix Product State representation of n-qubit quantum state.

|psi> = sum_{i_1...i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} ... A^{[n]}_{i_n} |i_1...i_n>

Memory: O(n * chi^2 * d) vs O(d^n) for full statevector.

Example scaling:
    n=30, chi=16: ~30KB vs 8GB for statevector
    n=33, chi=16: ~33KB vs 64GB for statevector*
- `VacuumCore` (line 916) - *Vacuum Core architecture for topological protection.

Projects irrelevant Hilbert subspace to zero, achieving
high sparsity (target 99.99%) while preserving quantum information.*
- `TopologicalProtector` (line 988) - *Provides topological protection for quantum states.

Monitors:
    - Winding numbers
    - Berry phases
    - Edge state preservation*
- `SpectralLayer` (line 1040) - *Spectral convolution layer in frequency domain.*
- `HamiltonianBackboneNet` (line 1076) - *Hamiltonian backbone network for spectral operations.*
- `SchrodingerSpectralNet` (line 1102) - *Schrodinger network for wavefunction evolution.*
- `DiracSpectralNet` (line 1133) - *Dirac network for relativistic spinor evolution.*
- `GammaMatrices` (line 1164) - *Dirac gamma matrices in standard or Weyl representation.*
- `IPhysicsBackend` (line 1215) - *Abstract interface for physics backends.*
- `HamiltonianBackend` (line 1229) - *Hamiltonian backend using neural network for spectral operations.

Performs first-order Schrodinger time evolution:
    psi(t+dt) = psi(t) - i*dt*H*psi(t)*
- `SchrodingerBackend` (line 1305) - *Schrodinger backend using learned 2-channel spectral network.

Falls back to HamiltonianBackend if checkpoint unavailable.*
- `DiracBackend` (line 1358) - *Dirac backend for relativistic spinor evolution.

Expands (2,G,G) amplitude to 4-component spinor, propagates,
then projects back.*
- `IQuantumGate` (line 1476) - *Abstract interface for quantum gates.*
- `HadamardGate` (line 1496) - *Hadamard gate: H = [[1,1],[1,-1]] / sqrt(2).*
- `PauliXGate` (line 1511) - *Pauli-X gate: X = [[0,1],[1,0]].*
- `PauliYGate` (line 1525) - *Pauli-Y gate: Y = [[0,-i],[i,0]].*
- `PauliZGate` (line 1539) - *Pauli-Z gate: Z = [[1,0],[0,-1]].*
- `SGate` (line 1553) - *S gate: S = [[1,0],[0,i]].*
- `TGate` (line 1567) - *T gate: T = [[1,0],[0,e^{i*pi/4}]].*
- `RxGate` (line 1582) - *Rotation-X gate: Rx(theta) = exp(-i*theta/2 * X).*
- `RyGate` (line 1598) - *Rotation-Y gate: Ry(theta) = exp(-i*theta/2 * Y).*
- `RzGate` (line 1614) - *Rotation-Z gate: Rz(theta) = exp(-i*theta/2 * Z).*
- `CRzGate` (line 1631) - *Controlled-Rz gate: applies Rz to target if control is |1>.*
- `CNOTGate` (line 1656) - *CNOT gate: flips target if control is |1>.*
- `CZGate` (line 1676) - *Controlled-Z gate: applies phase -1 to |11>.*
- `SWAPGate` (line 1696) - *SWAP gate: exchanges two qubits.*
- `CircuitInstruction` (line 1734) - *Single instruction in a quantum circuit.*
- `QuantumCircuit` (line 1741) - *Quantum circuit builder for MPS states.*
- `MPSQuantumComputer` (line 1816) - *Main quantum computer using MPS representation.

Provides:
    - State preparation
    - Circuit execution
    - Backend selection
    - Memory-efficient simulation for up to 33+ qubits*

**Functions:**
- `_make_logger` (line 46) - *Create a configured logger instance.*
- `run_scaling_benchmark` (line 2045) - *Run scaling benchmark to demonstrate MPS memory efficiency.

Returns:
    Dictionary with qubit counts, memory usage, compression ratios.*
- `run_grover_search` (line 2098) - *Run Grover's search algorithm and return results.

Implements the algorithm at the statevector level for correctness
(exact oracle via direct phase flip), then reads off probabilities.

Args:
    qc: MPSQuantumComputer instance (unused for computation, kept for API compatibility)
    n_qubits: number of qubits
    marked_states: list of integers representing marked computational basis states

Returns:
    dict with keys: probability, marked_states (as bit strings), speedup, iterations*
- `__post_init__` (line 139) - *Initialize random seeds after configuration.*
- `from_toml` (line 147) - *Load configuration from TOML file.*
- `__init__` (line 265)
- `_find_config` (line 274) - *Find configuration file in standard locations.*
- `_load` (line 286) - *Load configuration from TOML file.*
- `_load_defaults` (line 300) - *Load default configuration values.*
- `_parse_atoms` (line 318) - *Parse atoms from configuration data.*
- `_parse_molecules` (line 332) - *Parse molecules from configuration data.*
- `_parse_orbitals` (line 352) - *Parse orbitals from configuration data.*
- `_parse_experiments` (line 364) - *Parse experiments from configuration data.*
- `get_atom` (line 375) - *Get atom data by symbol (case-insensitive).*
- `get_molecule` (line 384) - *Get molecule data by name (case-insensitive).*
- `get_orbital` (line 393) - *Get orbital data by name (case-insensitive).*
- `get_experiment` (line 402) - *Get experiment data by name.*
- `atoms` (line 407) - *Return all atoms.*
- `molecules` (line 412) - *Return all molecules.*
- `orbitals` (line 417) - *Return all orbitals.*
- `experiments` (line 422) - *Return all experiments.*
- `get_molecules_by_qubits` (line 426) - *Get molecules that fit within qubit budget.*
- `get_atoms_by_qubits` (line 430) - *Get atoms that fit within qubit budget.*
- `n_qubits` (line 440) - *Return number of qubits.*
- `amplitude` (line 445) - *Compute amplitude for a computational basis state.*
- `apply_single_qubit_gate` (line 450) - *Apply single-qubit gate in-place.*
- `apply_two_qubit_gate` (line 455) - *Apply two-qubit gate in-place.*
- `norm` (line 460) - *Compute state norm.*
- `probabilities` (line 465) - *Compute measurement probabilities.*
- `entropy` (line 470) - *Compute von Neumann entropy.*
- `memory_bytes` (line 475) - *Return memory usage in bytes.*
- `__init__` (line 488)
- `_initialize` (line 504) - *Initialize core tensor for |0> product state (exact).*
- `tensor` (line 517) - *Return the core tensor.*
- `tensor` (line 524) - *Set the core tensor, preserving complex dtype when needed.*
- `left_canonicalize` (line 533) - *Bring core to left-canonical form, return singular values.*
- `right_canonicalize` (line 543) - *Bring core to right-canonical form, return singular values.*
- `__init__` (line 567)
- `_initialize` (line 576) - *Initialize MPS with product state |00...0>.*
- `n_qubits` (line 600)
- `_bond_dimension` (line 603) - *Compute bond dimension at given site.*
- `amplitude` (line 610) - *Compute amplitude for computational basis state.*
- `apply_single_qubit_gate` (line 631) - *Apply single-qubit gate in-place.

Works in complex128 so that gates with imaginary entries (Y, S, T, Rz…)
are handled correctly.  The core tensor is promoted to complex128 when
the result has a non-negligible imaginary component; otherwise it is
kept as the config dtype (typically float64).*
- `apply_two_qubit_gate` (line 655) - *Apply two-qubit gate in-place.*
- `_swap_qubits_in_gate` (line 674) - *Swap qubit ordering in two-qubit gate.*
- `_apply_adjacent_gate` (line 684) - *Apply gate to adjacent qubit pair.*
- `_apply_nonadjacent_gate` (line 747) - *Apply gate to non-adjacent qubit pair using SWAP network.*
- `norm` (line 762) - *Compute state norm.*
- `_canonicalize` (line 770) - *Bring MPS to canonical form.*
- `probabilities` (line 778) - *Compute measurement probabilities.*
- `entropy` (line 795) - *Compute maximum entanglement entropy across all cuts.

For n=1 qubits, returns Shannon entropy of the probability distribution.
For n>1, returns the maximum entanglement entropy across all bipartite cuts.*
- `memory_bytes` (line 822) - *Return memory usage in bytes.*
- `entanglement_entropy` (line 829) - *Compute entanglement entropy at given cut between qubits cut-1 and cut.
Uses Schmidt decomposition from the MPS bond.*
- `to_statevector` (line 874) - *Convert MPS to full statevector (only for small systems).*
- `most_probable_bitstring` (line 890) - *Return most probable basis state as bitstring.*
- `clone` (line 896) - *Return a deep copy.*
- `__init__` (line 924)
- `_initialize` (line 935) - *Initialize vacuum core with ground state.*
- `_compute_berry_phases` (line 941) - *Compute Berry phases between active states.*
- `add_active_state` (line 948) - *Add a basis state to the active subspace.*
- `_compute_winding_number` (line 962) - *Compute winding number for a basis state.*
- `is_topologically_protected` (line 968) - *Check if state is topologically protected.*
- `sparsity` (line 973) - *Compute vacuum sparsity.*
- `project_to_active` (line 979) - *Project state onto active subspace.*
- `__init__` (line 998)
- `compute_winding_number` (line 1003) - *Compute winding number for a qubit.*
- `compute_berry_phase` (line 1015) - *Compute Berry phase between two qubits.*
- `is_protected` (line 1031) - *Check if state is topologically protected.*
- `__init__` (line 1043)
- `forward` (line 1053) - *Apply spectral convolution via RFFT2.*
- `__init__` (line 1079)
- `forward` (line 1088) - *Apply Hamiltonian backbone network.*
- `__init__` (line 1105)
- `forward` (line 1119) - *Apply Schrodinger evolution network.*
- `__init__` (line 1136)
- `forward` (line 1150) - *Apply Dirac evolution network.*
- `__init__` (line 1167)
- `_init_matrices` (line 1172) - *Initialize gamma matrices.*
- `to` (line 1208) - *Move all matrices to device.*
- `evolve_amplitude` (line 1219) - *Evolve a single amplitude by time dt.*
- `apply_phase` (line 1224) - *Apply global phase to amplitude.*
- `__init__` (line 1237)
- `_load` (line 1245) - *Load model from checkpoint.*
- `_precompute_laplacian` (line 1266) - *Precompute Laplacian kernel for kinetic energy.*
- `_apply_h` (line 1274) - *Apply Hamiltonian operator to field.*
- `evolve_amplitude` (line 1284) - *Evolve amplitude by time dt.*
- `apply_phase` (line 1298) - *Apply global phase.*
- `__init__` (line 1312)
- `_load` (line 1319) - *Load model from checkpoint.*
- `evolve_amplitude` (line 1342) - *Evolve amplitude by time dt.*
- `apply_phase` (line 1353) - *Apply global phase.*
- `__init__` (line 1366)
- `_load` (line 1375) - *Load model from checkpoint.*
- `_precompute_dirac` (line 1398) - *Precompute momentum grids for Dirac operator.*
- `_pack` (line 1407) - *Pack 2-channel amplitude to 4-component spinor.*
- `_unpack` (line 1421) - *Unpack 4-component spinor to 2-channel amplitude.*
- `_analytical_dirac` (line 1428) - *Apply analytical Dirac Hamiltonian to spinor.*
- `evolve_amplitude` (line 1447) - *Evolve amplitude by time dt using Dirac equation.*
- `apply_phase` (line 1463) - *Apply global phase.*
- `evolve_spinor` (line 1467) - *Evolve full 4-component spinor by time dt.*
- `name` (line 1481) - *Return gate name.*
- `apply` (line 1486) - *Apply gate to state and return new state.*
- `name` (line 1500)
- `apply` (line 1503)
- `name` (line 1515)
- `apply` (line 1518)
- `name` (line 1529)
- `apply` (line 1532)
- `name` (line 1543)
- `apply` (line 1546)
- `name` (line 1557)
- `apply` (line 1560)
- `name` (line 1571)
- `apply` (line 1574)
- `name` (line 1586)
- `apply` (line 1589)
- `name` (line 1602)
- `apply` (line 1605)
- `name` (line 1618)
- `apply` (line 1621)
- `name` (line 1635)
- `apply` (line 1638)
- `name` (line 1660)
- `apply` (line 1663)
- `name` (line 1680)
- `apply` (line 1683)
- `name` (line 1700)
- `apply` (line 1703)
- `__init__` (line 1744)
- `_append` (line 1748) - *Append an instruction to the circuit.*
- `h` (line 1759)
- `x` (line 1762)
- `y` (line 1765)
- `z` (line 1768)
- `s` (line 1771)
- `t` (line 1774)
- `rx` (line 1777)
- `ry` (line 1780)
- `rz` (line 1783)
- `crz` (line 1786)
- `cnot` (line 1789)
- `cz` (line 1792)
- `swap` (line 1795)
- `__len__` (line 1798) - *Return number of instructions.*
- `__bool__` (line 1802) - *True if circuit has instructions.*
- `run` (line 1806) - *Execute circuit on state.*
- `__init__` (line 1827)
- `create_circuit` (line 1843) - *Create a new quantum circuit.*
- `create_state` (line 1851) - *Create initial state |00...0>.*
- `bell_state` (line 1865) - *Prepare Bell state |Phi+> = (|00> + |11>) / sqrt(2).*
- `ghz_state` (line 1873) - *Prepare GHZ state (|00...0> + |11...1>) / sqrt(2).*
- `w_state` (line 1882) - *Prepare W state: |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n

Uses direct statevector-to-MPS conversion via successive SVD.
This guarantees exact representation (up to numerical precision).

The W state has a unique entanglement structure:
- The entanglement entropy for a cut separating k qubits from (n-k) is:
  S(k) = H({k/n, (n-k)/n}) where H is the binary entropy function
- Maximum entropy is 1 bit when n is even and k = n/2
- For W₃: S_max ≈ 0.9183 bits (H({1/3, 2/3}))
- This is DIFFERENT from GHZ which has entropy = 1 bit for any cut

Returns:
    MPSState: The W state in MPS representation*
- `_build_w_state_direct` (line 1905) - *Build W state using direct statevector-to-MPS conversion via successive SVD.

This guarantees exact representation (up to numerical precision).*
- `run_circuit` (line 1985) - *Execute circuit on state.*
- `get_backend` (line 1995) - *Get physics backend by name.*
- `memory_usage` (line 1999) - *Compute memory usage for a state.*
- `compression_ratio` (line 2012) - *Compute compression ratio vs full statevector.*
- `detect_phase` (line 2019) - *Detect Hilbert space phase from state properties.*
- `_compute_average_bond_dimension` (line 2037) - *Compute average bond dimension across MPS cores.*

#### `quantum_framework_main.py`
**Path:** `quantum_framework_main.py`

**Functions:**
- `setup_logging` (line 52) - *Configure logging level based on verbosity.*
- `run_benchmark` (line 61) - *Run scaling benchmark.*
- `run_experiment` (line 101) - *Run a specific experiment by name.*
- `run_molecular_simulation` (line 173) - *Run molecular simulation.*
- `run_orbital_visualization` (line 210) - *Run orbital visualization.*
- `print_info` (line 242) - *Print framework information.*
- `main` (line 292) - *Main entry point.*

#### `quantum_framework_menu.py`
**Path:** `quantum_framework_menu.py`

**Classs:**
- `MenuSystem` (line 64) - *Interactive menu system for the quantum simulation framework.

Provides structured access to all framework capabilities through
a hierarchical menu system with real-time feedback.*

**Functions:**
- `run_interactive_menu` (line 2357) - *Run the interactive menu system.*
- `run_all_experiments` (line 2363) - *Run ALL experiments automatically for debugging.
This function executes all available experiments without user interaction.
Uses CORRECT physics formulas verified against experimental data.*
- `__init__` (line 72)
- `clear_screen` (line 79) - *Clear the terminal screen.*
- `print_header` (line 83) - *Print formatted header.*
- `print_menu` (line 90) - *Print formatted menu with options.*
- `get_input` (line 97) - *Get user input with history tracking.*
- `pause` (line 106) - *Wait for user to press Enter.*
- `run` (line 113) - *Run the main menu loop.*
- `_show_main_menu` (line 118) - *Display main menu.*
- `_show_circuit_menu` (line 163) - *Display quantum circuits menu.*
- `_custom_circuit` (line 197) - *Create and run a custom circuit.*
- `_bell_state_demo` (line 279) - *Demonstrate Bell state preparation.*
- `_ghz_state_demo` (line 299) - *Demonstrate GHZ state preparation.*
- `_w_state_demo` (line 328) - *Demonstrate W state preparation.*
- `_single_qubit_gates_demo` (line 356) - *Demonstrate single qubit gates.*
- `_two_qubit_gates_demo` (line 397) - *Demonstrate two qubit gates.*
- `_show_entanglement_menu` (line 431) - *Display entanglement experiments menu.*
- `_bell_entropy_experiment` (line 459) - *Measure entropy of Bell states.*
- `_ghz_scaling_experiment` (line 472) - *Study GHZ entanglement scaling with qubit count.*
- `_entropy_by_cut_experiment` (line 497) - *Measure entanglement entropy at different cuts.*
- `_entropy_heatmap_experiment` (line 522) - *Generate entropy heatmap for different states.*
- `_show_molecular_menu` (line 583) - *Display molecular simulations menu.*
- `_list_molecules` (line 617) - *List all available molecules.*
- `_molecule_info` (line 636) - *Show detailed molecule information.*
- `_vqe_ground_state` (line 667) - *Run VQE for molecular ground state.*
- `_energy_landscape` (line 751) - *Plot molecular energy landscape.*
- `_bond_dissociation` (line 808) - *Simulate bond dissociation curve.*
- `_show_orbital_menu` (line 868) - *Display orbital visualization menu.*
- `_list_orbitals` (line 899) - *List all available orbitals.*
- `_visualize_orbital` (line 915) - *Visualize a single orbital.*
- `_generate_orbital_plot` (line 947) - *Generate orbital visualization plot.*
- `_compare_orbitals` (line 1064) - *Compare multiple orbitals.*
- `_radial_wavefunction` (line 1146) - *Plot radial wavefunction.*
- `_angular_wavefunction` (line 1202) - *Plot angular wavefunction.*
- `_show_relativistic_menu` (line 1264) - *Display relativistic physics menu.*
- `_dirac_energy_levels` (line 1292) - *Calculate Dirac energy levels.*
- `_dirac_energy` (line 1321) - *Calculate Dirac energy level.*
- `_fine_structure` (line 1329) - *Calculate fine structure corrections.*
- `_zitterbewegung` (line 1347) - *Simulate Zitterbewegung.*
- `_spin_orbit` (line 1363) - *Calculate spin-orbit coupling.*
- `_show_qed_menu` (line 1378) - *Display QED effects menu.*
- `_lamb_shift` (line 1406) - *Calculate Lamb shift.*
- `_anomalous_moment` (line 1426) - *Calculate anomalous magnetic moment.*
- `_vacuum_polarization` (line 1453) - *Calculate vacuum polarization effects.*
- `_full_qed` (line 1468) - *Show full QED corrections.*
- `_show_algorithms_menu` (line 1482) - *Display quantum algorithms menu.*
- `_grover_search` (line 1510) - *Demonstrate Grover's search algorithm.*
- `_apply_grover_iteration` (line 1558) - *Apply one Grover iteration: oracle then diffusion.*
- `_qft_demo` (line 1607) - *Demonstrate Quantum Fourier Transform.*
- `_phase_estimation` (line 1669) - *Demonstrate phase estimation.*
- `_vqe_demo` (line 1746) - *Demonstrate VQE.*
- `_show_benchmark_menu` (line 1816) - *Display benchmarks menu.*
- `_mps_scaling_benchmark` (line 1844) - *Run MPS scaling benchmark.*
- `_gate_performance` (line 1872) - *Benchmark gate performance.*
- `_memory_comparison` (line 1907) - *Compare memory usage.*
- `_entanglement_scaling` (line 1927) - *Study entanglement scaling.*
- `_show_config_menu` (line 1953) - *Display configuration menu.*
- `_view_config` (line 1984) - *View current configuration.*
- `_list_atoms` (line 1999) - *List available atoms.*
- `_list_molecules_config` (line 2016) - *List available molecules.*
- `_list_experiments` (line 2020) - *List available experiments.*
- `_system_info` (line 2036) - *Show system information.*
- `_show_particle_physics_menu` (line 2055) - *Display particle physics menu (Higgs analysis).*
- `_run_higgs_analysis` (line 2077) - *Run the Higgs boson 4-lepton quantum analysis.*
- `_higgs_about` (line 2101) - *Show information about the Higgs analysis.*
- `_show_visualization_menu` (line 2129) - *Display quantum visualization menu.*
- `_run_quantum_dash` (line 2154) - *Run brutalist quantum state visualizer.*
- `_run_quantum_3dview` (line 2201) - *Run 3D holographic quantum dashboard.*
- `_run_quantum_visualizer` (line 2247) - *Run the standard quantum state visualizer.*
- `_run_polarizability_vqe` (line 2285) - *Run H2 polarizability / Stark effect VQE from app.py.*
- `_show_help` (line 2316) - *Show help information.*
- `_quit` (line 2350) - *Exit the menu system.*
- `test_header` (line 2379)
- `radial_wf` (line 951)
- `spherical_harm_real` (line 959)
- `compute_energy` (line 1764)

#### `quantum_framework_molecular.py`
**Path:** `quantum_framework_molecular.py`

**Classs:**
- `MoleculeData` (line 72)
- `MoleculeBuilder` (line 85) - *Build molecule data for quantum chemistry calculations.*
- `ExactJWEnergy` (line 156) - *Exact Jordan-Wigner energy evaluator.

CORRECTED: Uses verified Hamiltonian coefficients from standard references.*
- `UCCSDAnsatz` (line 334) - *Unitary Coupled Cluster Singles and Doubles ansatz.

For H2 in the 2-qubit active space model:
- Qubit 0 represents the bonding orbital occupation
- Qubit 1 represents the anti-bonding orbital occupation
- HF state: |10> (bonding occupied, anti-bonding empty)
- The double excitation |10> <-> |01> is mediated by X0X1 + Y0Y1 terms*
- `VQEResult` (line 476)
- `VQESolver` (line 510) - *Variational Quantum Eigensolver.

CORRECTED: Uses proper UCCSD ansatz and Hamiltonian evaluation.*

**Functions:**
- `_make_logger` (line 58)
- `run_vqe_h2` (line 630) - *Run VQE for H2 molecule - convenience function.*
- `h2_sto3g` (line 89) - *Build H2 molecule with STO-3G basis.*
- `_h2_pyscf` (line 96) - *Build H2 using PySCF.*
- `_h2_hardcoded` (line 128) - *Build H2 with hardcoded values - CORRECTED for 2-qubit active space.*
- `__init__` (line 163)
- `_build_hamiltonian` (line 170) - *Build the molecular Hamiltonian in JW representation.*
- `_build_openfermion_hamiltonian` (line 178) - *Build Hamiltonian using OpenFermion - FIXED API.*
- `_build_hardcoded_hamiltonian` (line 231) - *Build hardcoded H2 Hamiltonian - CORRECTED COEFFICIENTS.

The H2/STO-3G Hamiltonian in the minimal active space (2 qubits)
with Jordan-Wigner transformation.

Derived from reference energies:
- E_HF = -1.11675928 Ha
- E_FCI = -1.13728383 Ha
- E_nuc = 0.71996899 Ha

The Hamiltonian has the form:
H = E_nuc + h1*Z0 + h2*Z1 + h3*Z0*Z1 + h4*X0*X1 + h5*Y0*Y1

For the 2-qubit active space model where:
- |10> is the HF state (bonding orbital occupied)
- The ground state is a superposition of |10> and |01>*
- `_apply_pauli` (line 276) - *Apply Pauli operator to state vector - CORRECTED.*
- `expectation_value` (line 302) - *Compute ⟨ψ|H|ψ⟩ for the given state.*
- `evaluate` (line 318) - *Evaluate energy from amplitudes (supports both numpy and torch).*
- `__call__` (line 330)
- `__init__` (line 345)
- `apply_double_excitation_2q` (line 373) - *Apply double excitation for 2-qubit H2 model.

This rotates between |10> and |01>:
|10> -> cos(theta)*|10> - sin(theta)*|01>
|01> -> sin(theta)*|10> + cos(theta)*|01>*
- `apply_single_excitation` (line 395) - *Apply single excitation as Givens rotation.*
- `apply_double_excitation` (line 416) - *Apply double excitation for 4+ qubit systems.*
- `apply` (line 443) - *Apply UCCSD ansatz to state.*
- `__repr__` (line 490)
- `__init__` (line 517)
- `prepare_hf_state` (line 528) - *Prepare Hartree-Fock state.

For H2 with 2 electrons in 4 spin-orbitals:
|1100⟩ means electrons in orbitals 0 and 1.*
- `run` (line 546) - *Run VQE optimization.*
- `cost` (line 566)

#### `quantum_framework_molecular_fixed.py`
**Path:** `quantum_framework_molecular_fixed.py`

**Classs:**
- `MoleculeData` (line 64)
- `MoleculeBuilder` (line 77) - *Build molecule data for quantum chemistry calculations.
Uses PySCF when available, falls back to hardcoded values.*
- `ExactJWEnergy` (line 147) - *Exact Jordan-Wigner energy evaluator.
Computes molecular energy using JW transformation.

FIXED: Proper Hamiltonian construction and expectation values.*
- `UCCSDAnsatz` (line 343) - *Unitary Coupled Cluster Singles and Doubles ansatz.

FIXED: Correct excitation operator implementation.*
- `VQEResult` (line 457)
- `VQESolver` (line 491) - *Variational Quantum Eigensolver.

FIXED: Correct HF state preparation and energy evaluation.*

**Functions:**
- `_make_logger` (line 50)
- `_get_sd_indices` (line 323) - *Get single and double excitation indices for UCCSD.*
- `run_vqe_demo` (line 614) - *Run a quick VQE demo to verify the fixes.*
- `h2_sto3g` (line 84) - *Build H2 molecule with STO-3G basis.*
- `_h2_pyscf` (line 91) - *Build H2 using PySCF - FIXED atom string syntax.*
- `_h2_hardcoded` (line 126) - *Build H2 with hardcoded values - FIXED coefficients.*
- `__init__` (line 155)
- `_build_hamiltonian` (line 162) - *Build the molecular Hamiltonian in JW representation.*
- `_build_openfermion_hamiltonian` (line 169) - *Build Hamiltonian using OpenFermion - FIXED geometry.*
- `_build_hardcoded_hamiltonian` (line 215) - *Build hardcoded H2 Hamiltonian - FIXED coefficients.

The H2/STO-3G Hamiltonian in JW form (standard reference):
H = g0*I + g1*Z0 + g2*Z1 + g3*Z0*Z1 + g4*X0*X1 + g5*Y0*Y1

With coefficients for bond length 0.735 Å:*
- `_apply_pauli` (line 247) - *Apply Pauli operator to state vector.

FIXED: Correct amplitude indexing and phase handling.*
- `expectation_value` (line 281) - *Compute ⟨ψ|H|ψ⟩ for the given state.

FIXED: Correct inner product calculation.*
- `evaluate` (line 305) - *Evaluate energy from MPS amplitudes.*
- `__call__` (line 319)
- `__init__` (line 350)
- `apply_single_excitation` (line 359) - *Apply single excitation operator exp(theta * (a_v† a_o - a_o† a_v)).

In JW, this is a Givens rotation between orbitals o and v.*
- `apply_double_excitation` (line 387) - *Apply double excitation operator.

Simplified: applies pairwise excitation with rotation.*
- `apply` (line 419) - *Apply UCCSD ansatz to state.*
- `__repr__` (line 471)
- `__init__` (line 498)
- `prepare_hf_state` (line 502) - *Prepare Hartree-Fock state.

FIXED: Correct bitstring with 0s and 1s.

For H2 with 2 electrons in 4 spin-orbitals:
|1100⟩ means electrons in orbitals 0 and 1 (occupied spin-orbitals)*
- `run` (line 525) - *Run VQE optimization.*
- `cost` (line 556)

#### `quantum_framework_molecular_v2.py`
**Path:** `quantum_framework_molecular_v2.py`

**Classs:**
- `MolecularConfig` (line 95) - *Configuration for molecular VQE simulations.*
- `MoleculeData` (line 172) - *Molecular data structure with all necessary information.*
- `MoleculeBuilder` (line 200) - *Build molecules using OpenFermion + PySCF.

NO hardcoded values - everything comes from quantum chemistry calculations.*
- `HamiltonianBuilder` (line 352) - *Build molecular Hamiltonians using OpenFermion.

NO hardcoded coefficients - everything from first principles.*
- `CachedPauliOperation` (line 468) - *Precomputed Pauli operation for fast application.*
- `CachedHamiltonianEvaluator` (line 477) - *Hamiltonian evaluator with cached Pauli operations.

Improvement: x10-100 speedup for repeated evaluations.*
- `SmartInitializer` (line 613) - *Smart parameter initialization for VQE.

Improvements:
- MP2 amplitude estimation
- Systematic parameter scan
- Parabolic refinement
- Sensitivity analysis*
- `UCCSDAnsatz` (line 746) - *Unitary Coupled Cluster Singles and Doubles ansatz.

Features:
- Particle-conserving excitations
- Works with both direct and MPS representations
- Identity check (θ=0 → HF state)*
- `MPSState` (line 879) - *Matrix Product State for scalable quantum simulation.

Features:
- Adaptive bond dimension
- Efficient gate application
- Entanglement tracking*
- `ParticleConservingState` (line 982) - *State representation that preserves particle number symmetry.

Improvement: x4 reduction in Hilbert space, better convergence.*
- `VQEResult` (line 1064) - *VQE result container.*
- `VQESolver` (line 1104) - *Production VQE solver with all improvements.

Features:
- OpenFermion Hamiltonians (no hardcoded values)
- Precision mode flag (direct vs MPS)
- Smart initialization with MP2 + scan
- Cached Pauli operations
- Particle conservation option
- Backend integration*
- `BackendIntegrator` (line 1287) - *Integration with existing backends (Hamiltonian, Schrodinger, Dirac).

Uses pre-trained models from QC repository.*
- `PseudoMolData` (line 301)

**Functions:**
- `_make_logger` (line 75)
- `run_vqe` (line 1332) - *Convenience function to run VQE.

Args:
    molecule: Molecule name ("H2", "H2O", "LiH")
    precision_mode: Use direct statevector (True) or MPS (False)
    config_path: Path to TOML config file
    **kwargs: Additional config overrides

Returns:
    VQEResult*
- `from_toml` (line 131) - *Load configuration from TOML file.*
- `build` (line 208) - *Build molecule using OpenFermion.

Args:
    name: Molecule name (e.g., "H2", "H2O")
    geometry: List of (atom_symbol, (x, y, z)) in Angstrom
    basis: Basis set (e.g., "sto-3g", "6-31g")
    charge: Molecular charge
    multiplicity: Spin multiplicity
    description: Optional description

Returns:
    MoleculeData with all properties computed*
- `_run_pyscf_direct` (line 289) - *Run PySCF directly if openfermionpyscf not available.*
- `h2` (line 324) - *Build H2 molecule.*
- `h2o` (line 330) - *Build H2O molecule.*
- `lih` (line 342) - *Build LiH molecule.*
- `build_jw_hamiltonian` (line 360) - *Build Jordan-Wigner transformed Hamiltonian using OpenFermion.

Args:
    mol: MoleculeData with geometry, basis, etc.

Returns:
    (pauli_terms, nuclear_repulsion)*
- `build_hamiltonian_matrix` (line 417) - *Build full Hamiltonian matrix for small systems.

Args:
    mol: MoleculeData
    n_qubits: Number of qubits

Returns:
    Hamiltonian matrix (2^n_qubits, 2^n_qubits)*
- `_pauli_matrix` (line 441) - *Build matrix for a Pauli string.*
- `__init__` (line 484)
- `_precompute_operations` (line 503) - *Precompute all Pauli operations for fast evaluation.*
- `_compute_pauli_mapping` (line 522) - *Compute index mapping and phases for a Pauli string.*
- `apply_pauli_fast` (line 561) - *Apply cached Pauli operation.*
- `expectation_value` (line 568) - *Compute energy expectation value.*
- `batch_expectation` (line 587) - *Compute expectation for batch of states.*
- `__init__` (line 624)
- `estimate_mp2_amplitude` (line 630) - *Estimate doubles amplitude from MP2 theory.

θ_MP2 ≈ t_2^(1) / 2
where t_2^(1) = <ij||ab> / (ε_i + ε_j - ε_a - ε_b)*
- `scan_parameter_space` (line 657) - *Systematic scan of parameter space.

Returns:
    (best_thetas, best_energy)*
- `_parabolic_refinement` (line 712) - *Refine minimum using parabolic interpolation.*
- `initialize` (line 735) - *Complete initialization with all techniques.*
- `__init__` (line 756)
- `_generate_excitations` (line 764) - *Generate all single and double excitations.*
- `apply` (line 785) - *Apply UCCSD ansatz to state.

Args:
    state: State vector (2^n_qubits,)
    thetas: Parameters (n_params,)

Returns:
    Transformed state vector*
- `_apply_single` (line 817) - *Apply single excitation as Givens rotation.*
- `_apply_double` (line 838) - *Apply double excitation.*
- `verify_identity` (line 862) - *Verify that θ=0 gives HF state.*
- `__init__` (line 889)
- `to_statevector` (line 910) - *Convert MPS to full statevector.*
- `from_statevector` (line 918) - *Create MPS from statevector.*
- `compute_entanglement` (line 953) - *Compute entanglement entropy at bond.*
- `__init__` (line 989)
- `_generate_fock_states` (line 1000) - *Generate all states with fixed particle number.*
- `hf_state` (line 1013) - *Create HF state in subspace.*
- `apply_excitation` (line 1027) - *Apply excitation preserving particle number.*
- `to_full_statevector` (line 1051) - *Convert subspace state to full statevector.*
- `__repr__` (line 1081)
- `__init__` (line 1117)
- `prepare_hf_state` (line 1137) - *Prepare Hartree-Fock state.*
- `evaluate` (line 1152) - *Evaluate energy.*
- `apply_ansatz` (line 1159) - *Apply UCCSD ansatz.*
- `run` (line 1184) - *Run VQE optimization.*
- `__init__` (line 1294)
- `_load_models` (line 1301) - *Load pre-trained backend models.*
- `get_backend_energy` (line 1318) - *Get energy estimate from backend model.*
- `cost` (line 1212)

#### `quantum_framework_physics.py`
**Path:** `quantum_framework_physics.py`

**Classs:**
- `SpectralLayer` (line 26) - *Spectral convolution in frequency domain.
Learns complex kernels that modulate Fourier coefficients.*
- `HamiltonianBackboneNet` (line 66) - *Hamiltonian backbone: single-channel field -> H|psi>.
Shared by all physics backends as the H operator.*
- `SchrodingerSpectralNet` (line 92) - *Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction.
Uses spectral convolution for physics-informed evolution.*
- `DiracSpectralNet` (line 120) - *Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor.
Handles 4-component spinor evolution for relativistic quantum mechanics.*
- `GammaMatrices` (line 150) - *Dirac gamma matrices in Dirac (standard) or Weyl representation.
gamma^0 = beta, gamma^i = beta * alpha_i*
- `PotentialGenerator` (line 250) - *Spatial potentials for eigenstate initialization.*
- `DiracHamiltonianOperator` (line 294) - *Dirac Hamiltonian operator for relativistic quantum mechanics.
H_Dirac = c * alpha . p + beta * m * c^2 + V(r)

In atomic units (c = 1/alpha ~ 137):
H = c * alpha . p + beta * m * c^2 + V*
- `LambShiftCalculator` (line 382) - *Calculates the Lamb shift using Bethe's formula.
The Lamb shift is the small difference between 2s_{1/2} and 2p_{1/2} energy levels
due to QED effects (vacuum fluctuations and self-energy).*
- `AnomalousMagneticMoment` (line 439) - *Calculates the electron's anomalous magnetic moment (g-2).
The electron g-factor is slightly different from 2 due to QED effects:
g = 2(1 + a_e) where a_e = alpha/(2*pi) + higher-order terms*
- `DiracHydrogenAtom` (line 489) - *Relativistic hydrogen atom with Dirac equation.
Computes energy levels including fine structure.*
- `ZitterbewegungSimulator` (line 571) - *Simulates the Zitterbewegung (trembling motion) of a relativistic electron.
In Dirac theory, the position operator has a term oscillating with frequency
~ 2mc^2/hbar, which is the interference between positive and negative energy states.*

**Functions:**
- `__init__` (line 32)
- `forward` (line 43)
- `__init__` (line 72)
- `forward` (line 81)
- `__init__` (line 98)
- `forward` (line 109)
- `__init__` (line 126)
- `forward` (line 139)
- `__init__` (line 156)
- `_init_matrices` (line 161)
- `to` (line 244)
- `__init__` (line 253)
- `_grid` (line 258)
- `harmonic` (line 263)
- `double_well` (line 268)
- `coulomb` (line 274)
- `periodic_lattice` (line 280)
- `mixed` (line 284)
- `__init__` (line 303)
- `_precompute_operators` (line 311)
- `apply_dirac_hamiltonian` (line 318) - *Apply Dirac Hamiltonian to 4-component spinor.

Args:
    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor
    potential: Optional scalar potential V(r)

Returns:
    H * psi with same shape as input*
- `time_evolution` (line 362) - *Time evolution of Dirac spinor using first-order split-step.
psi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi*
- `__init__` (line 389)
- `bethe_formula` (line 394) - *Bethe's non-relativistic formula for Lamb shift.
Delta E_Lamb = (8*alpha^3 / 3*pi*n^3) * |psi_n(0)|^2 * ln(E_avg / E_n)*
- `_higher_l_shift` (line 407)
- `full_lamb_shift` (line 412) - *Calculate full Lamb shift including radiative corrections.
Delta E = Delta E_SE + Delta E_Uehling + Delta E_rel*
- `__init__` (line 446)
- `schwinger_term` (line 449)
- `second_order` (line 452)
- `third_order` (line 456)
- `fourth_order` (line 460)
- `fifth_order` (line 464)
- `calculate_a_e` (line 468)
- `__init__` (line 495)
- `energy_level_dirac` (line 499) - *Exact Dirac energy level for hydrogen-like atom.
E = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)*
- `fine_structure_splitting` (line 512) - *Calculate fine structure splitting for given n, l.
Returns energies for j = l+1/2 and j = l-1/2*
- `energy_spectrum` (line 535)
- `__init__` (line 578)
- `create_gaussian_wave_packet` (line 585)
- `compute_position_expectation` (line 603)
- `compute_velocity_expectation` (line 615)

#### `quantum_framework_visualization.py`
**Path:** `quantum_framework_visualization.py`

**Classs:**
- `WavefunctionCalculator` (line 59) - *Calculates hydrogen atom wavefunctions for visualization.
Uses analytical formulas for radial and angular parts.*
- `MonteCarloSampler` (line 115) - *Monte Carlo sampling for orbital visualization.
Uses rejection sampling to generate 3D point clouds.*
- `OrbitalVisualizer` (line 202) - *High-resolution visualization of hydrogen orbitals.
Creates 2D projections and 3D scatter plots.*
- `EntangledHydrogenSampler` (line 290) - *Monte Carlo sampler for entangled hydrogen states.
Samples from joint probability distribution of entangled orbitals.*
- `EntangledHydrogenVisualizer` (line 325) - *Visualizer for entangled hydrogen states.
Creates high-resolution visualizations with multiple orbitals.*

**Functions:**
- `_make_logger` (line 46)
- `__init__` (line 65)
- `radial_wavefunction` (line 69)
- `spherical_harmonic_real` (line 81)
- `psi_3d` (line 92)
- `psi_on_grid` (line 97)
- `__init__` (line 121)
- `find_max_probability` (line 125)
- `sample` (line 149)
- `__init__` (line 208)
- `visualize` (line 211)
- `__init__` (line 296)
- `sample_entangled_state` (line 301)
- `__init__` (line 331)
- `visualize` (line 334)

#### `quantum_lab.py`
**Path:** `quantum_lab.py`

**Classs:**
- `OrbitalSpec` (line 431) - *Definition of a real hydrogen orbital for the ASCII viewer.*
- `H2VQEEngine` (line 604) - *Minimal live VQE for H2 in the 2-qubit active space.

The trial state cos(theta/2)|01> + sin(theta/2)|10> is prepared with a
real circuit (Ry, CNOT, X) on the MPS engine, and the energy is read
from the Jordan-Wigner H2 Hamiltonian of quantum_framework_molecular.*
- `Quiz` (line 659)
- `Lesson` (line 667)
- `QuantumLab` (line 672) - *Interactive educational TUI driven by the real Q2C engine.*

**Functions:**
- `probability_bars` (line 325) - *Build a table of probability bars for a state's distribution.*
- `counts_bars` (line 353) - *Build a table of measurement-count bars.*
- `draw_circuit` (line 369) - *Render an ASCII timeline of the circuit, one line per qubit.*
- `parse_angle` (line 394) - *Parse an angle like '1.57', 'pi', '-pi/2' or '3*pi/4' into radians.*
- `sample_measurements` (line 414) - *Draw measurement outcomes from a probability distribution.*
- `_radial` (line 440) - *Hydrogen radial wavefunction R_nl(r) in atomic units.*
- `_ang_s` (line 452)
- `_ang_pz` (line 456)
- `_ang_px` (line 460)
- `_ang_dz2` (line 464)
- `_ang_dxz` (line 469)
- `_ang_dxy` (line 473)
- `field_to_text` (line 490) - *Render a real scalar field as colored ASCII: brightness = |psi|^2, color = sign.*
- `render_orbital` (line 516) - *Render |psi|^2 of a hydrogen orbital on a plane slice as colored ASCII.*
- `render_h2_molecular_orbital` (line 536) - *Render the bonding or antibonding LCAO molecular orbital of H2.*
- `landscape_plot` (line 556) - *Draw an ASCII plot of E(theta) over one full period with HF and FCI
reference lines and an optional optimizer marker.*
- `launch_quantum_lab` (line 1474) - *Entry point used both standalone and from quantum_framework_main.*
- `main` (line 1500)
- `row_of` (line 567)
- `__init__` (line 613)
- `ansatz_instructions` (line 621)
- `energy` (line 624)
- `correlation_pct` (line 633)
- `landscape` (line 636)
- `optimize` (line 640) - *Gradient descent; yields (iteration, theta, energy) live.*
- `__init__` (line 675)
- `t` (line 683)
- `pause` (line 690)
- `panel` (line 697)
- `show_state` (line 701)
- `run_quiz` (line 707)
- `lessons` (line 728)
- `_demo_superposition` (line 733)
- `_demo_rotation` (line 744)
- `_demo_measurement` (line 766)
- `_demo_bell` (line 779)
- `_demo_ghz_w` (line 792)
- `_demo_grover` (line 799)
- `_demo_molecule` (line 815)
- `_vqe` (line 821)
- `_demo_h2_clouds` (line 826)
- `_demo_vqe_ansatz` (line 835)
- `_demo_vqe_landscape` (line 840)
- `_demo_vqe_live` (line 847)
- `_lessons_en` (line 875)
- `_lessons_es` (line 1015)
- `run_lesson` (line 1159)
- `lessons_menu` (line 1175)
- `_rebuild_state` (line 1198)
- `_playground_dashboard` (line 1216)
- `_show_amplitudes` (line 1228)
- `playground` (line 1249)
- `_molecule_card` (line 1346)
- `molecule_explorer` (line 1365)
- `orbital_viewer` (line 1383)
- `chemistry_menu` (line 1403)
- `glossary` (line 1422)
- `banner` (line 1437)
- `main_menu` (line 1443)

#### `quantum_simulator.py`
**Path:** `quantum_simulator.py`

**Classs:**
- `FrameworkConfig` (line 68)
- `AtomData` (line 164)
- `MoleculeData` (line 174)
- `OrbitalData` (line 192)
- `ConfigLoader` (line 200)
- `SpectralLayer` (line 288)
- `HamiltonianBackboneNet` (line 303)
- `SchrodingerSpectralNet` (line 321)
- `DiracSpectralNet` (line 341)
- `GammaMatrices` (line 361)
- `JointHilbertState` (line 380)
- `IPhysicsBackend` (line 410)
- `HamiltonianBackend` (line 420)
- `SchrodingerBackend` (line 470)
- `DiracBackend` (line 505)
- `IQuantumGate` (line 638)
- `HadamardGate` (line 649)
- `PauliXGate` (line 662)
- `PauliYGate` (line 674)
- `PauliZGate` (line 686)
- `SGate` (line 698)
- `TGate` (line 710)
- `RxGate` (line 723)
- `RyGate` (line 737)
- `RzGate` (line 751)
- `CNOTGate` (line 766)
- `CZGate` (line 778)
- `SWAPGate` (line 790)
- `ToffoliGate` (line 802)
- `CircuitInstruction` (line 832)
- `QuantumCircuit` (line 838)
- `QuantumResult` (line 894)
- `PotentialGenerator` (line 908)
- `JointStateFactory` (line 972)
- `QuantumComputer` (line 999)
- `WavefunctionCalculator` (line 1043)
- `MonteCarloSampler` (line 1080)
- `DiracHydrogenAtom` (line 1159)
- `ZitterbewegungSimulator` (line 1200)
- `OrbitalVisualizer` (line 1282)
- `EntangledVisualizer` (line 1343)
- `QuantumSimulationFramework` (line 1409)
- `InteractiveMenu` (line 1535)

**Functions:**
- `_make_logger` (line 54)
- `_single_qubit_unitary` (line 587)
- `_two_qubit_unitary` (line 611)
- `_solve_eigenstate` (line 949)
- `_build_basis_amplitude` (line 965)
- `main` (line 1868)
- `from_toml` (line 112)
- `__init__` (line 201)
- `_find_config` (line 209)
- `_load` (line 219)
- `_load_defaults` (line 229)
- `_parse_atoms` (line 236)
- `_parse_molecules` (line 241)
- `_parse_orbitals` (line 246)
- `get_atom` (line 251)
- `get_molecule` (line 259)
- `get_orbital` (line 267)
- `atoms` (line 276)
- `molecules` (line 280)
- `orbitals` (line 284)
- `__init__` (line 289)
- `forward` (line 295)
- `__init__` (line 304)
- `forward` (line 310)
- `__init__` (line 322)
- `forward` (line 330)
- `__init__` (line 342)
- `forward` (line 350)
- `__init__` (line 362)
- `__init__` (line 381)
- `normalize_` (line 388)
- `probabilities` (line 393)
- `entropy` (line 397)
- `most_probable_bitstring` (line 402)
- `clone` (line 406)
- `evolve_amplitude` (line 412)
- `apply_phase` (line 416)
- `__init__` (line 421)
- `_load` (line 429)
- `_precompute_laplacian` (line 443)
- `_apply_h` (line 450)
- `evolve_amplitude` (line 458)
- `apply_phase` (line 465)
- `__init__` (line 471)
- `_load` (line 478)
- `evolve_amplitude` (line 493)
- `apply_phase` (line 501)
- `__init__` (line 506)
- `_load` (line 515)
- `_precompute_dirac` (line 530)
- `_pack` (line 537)
- `_unpack` (line 547)
- `_analytical_dirac` (line 553)
- `evolve_amplitude` (line 564)
- `apply_phase` (line 577)
- `evolve_spinor` (line 580)
- `name` (line 641)
- `apply` (line 645)
- `name` (line 651)
- `apply` (line 654)
- `name` (line 664)
- `apply` (line 667)
- `name` (line 676)
- `apply` (line 679)
- `name` (line 688)
- `apply` (line 691)
- `name` (line 700)
- `apply` (line 703)
- `name` (line 712)
- `apply` (line 715)
- `name` (line 725)
- `apply` (line 728)
- `name` (line 739)
- `apply` (line 742)
- `name` (line 753)
- `apply` (line 756)
- `name` (line 768)
- `apply` (line 771)
- `name` (line 780)
- `apply` (line 783)
- `name` (line 792)
- `apply` (line 795)
- `name` (line 804)
- `apply` (line 807)
- `__init__` (line 839)
- `_append` (line 843)
- `h` (line 846)
- `x` (line 849)
- `y` (line 852)
- `z` (line 855)
- `s` (line 858)
- `t` (line 861)
- `rx` (line 864)
- `ry` (line 867)
- `rz` (line 870)
- `cnot` (line 873)
- `cz` (line 876)
- `swap` (line 879)
- `ccx` (line 882)
- `run` (line 885)
- `__init__` (line 895)
- `entropy` (line 898)
- `most_probable_bitstring` (line 901)
- `probabilities` (line 904)
- `__init__` (line 909)
- `_grid` (line 913)
- `harmonic` (line 918)
- `double_well` (line 923)
- `coulomb` (line 929)
- `periodic_lattice` (line 935)
- `mixed` (line 939)
- `__init__` (line 973)
- `_empty` (line 976)
- `all_zeros` (line 979)
- `basis_state` (line 986)
- `from_bitstring` (line 995)
- `__init__` (line 1000)
- `create_circuit` (line 1011)
- `run_circuit` (line 1014)
- `bell_state` (line 1021)
- `ghz_state` (line 1027)
- `factory` (line 1035)
- `backends` (line 1039)
- `__init__` (line 1044)
- `radial_wavefunction` (line 1048)
- `spherical_harmonic_real` (line 1060)
- `psi_3d` (line 1071)
- `energy_analytical` (line 1076)
- `__init__` (line 1081)
- `find_max_probability` (line 1085)
- `sample_orbital` (line 1109)
- `sample_entangled_state` (line 1146)
- `__init__` (line 1160)
- `energy_level_dirac` (line 1165)
- `energy_schrodinger` (line 1173)
- `fine_structure_splitting` (line 1176)
- `energy_spectrum` (line 1184)
- `__init__` (line 1201)
- `create_gaussian_wave_packet` (line 1206)
- `compute_position_expectation` (line 1224)
- `compute_velocity_expectation` (line 1235)
- `simulate` (line 1246)
- `__init__` (line 1283)
- `visualize` (line 1286)
- `__init__` (line 1344)
- `visualize` (line 1347)
- `__init__` (line 1410)
- `_ensure_output_dir` (line 1422)
- `list_available_atoms` (line 1425)
- `list_available_molecules` (line 1428)
- `list_available_orbitals` (line 1431)
- `get_atom` (line 1434)
- `get_molecule` (line 1437)
- `get_orbital` (line 1440)
- `run_quantum_circuit` (line 1443)
- `visualize_orbital` (line 1446)
- `visualize_atom_orbitals` (line 1458)
- `visualize_entangled_state` (line 1482)
- `compute_relativistic_energy` (line 1496)
- `compute_energy_spectrum` (line 1499)
- `run_zitterbewegung_simulation` (line 1502)
- `run_all_demonstrations` (line 1505)
- `__init__` (line 1536)
- `display_header` (line 1540)
- `display_main_menu` (line 1548)
- `get_user_choice` (line 1564)
- `_display_quantum_result` (line 1570)
- `orbital_menu` (line 1578)
- `atom_orbital_menu` (line 1602)
- `entangled_menu` (line 1622)
- `quantum_circuit_menu` (line 1651)
- `relativistic_menu` (line 1737)
- `zitterbewegung_menu` (line 1777)
- `molecular_menu` (line 1796)
- `atomic_menu` (line 1820)
- `run` (line 1840)

#### `quantum_visualizer.py`
**Path:** `quantum_visualizer.py`

**Classs:**
- `VisualizerConfig` (line 109)
- `QuantumStateSnapshot` (line 157)
- `BackendComparisonResult` (line 170)
- `VisualizationResult` (line 181)
- `IVisualizationComponent` (line 190)
- `ProbabilityBarRenderer` (line 196)
- `BlochSphereRenderer` (line 227)
- `PhasePlotRenderer` (line 258)
- `EntropyPlotRenderer` (line 291)
- `BackendComparisonRenderer` (line 313)
- `QuantumStateAnalyzer` (line 341)
- `CircuitExecutor` (line 397)
- `StandardCircuits` (line 457)
- `FigureBuilder` (line 528)
- `QuantumVisualizer` (line 612)

**Functions:**
- `_make_logger` (line 93)
- `main` (line 856)
- `render` (line 192)
- `render` (line 197)
- `_get_colors` (line 218)
- `render` (line 228)
- `render` (line 259)
- `render` (line 292)
- `render` (line 314)
- `__init__` (line 342)
- `compute_probabilities` (line 345)
- `compute_phases` (line 349)
- `compute_entropy` (line 360)
- `compute_bloch_vectors` (line 367)
- `create_snapshot` (line 374)
- `__init__` (line 398)
- `execute_sequence` (line 403)
- `compare_backends` (line 423)
- `bell_state` (line 459)
- `ghz_state` (line 466)
- `qft` (line 473)
- `grover_oracle` (line 488)
- `grover_diffusion` (line 501)
- `custom_sequence` (line 514)
- `__init__` (line 529)
- `build_evolution_figure` (line 537)
- `build_summary_figure` (line 563)
- `_render_backend_fidelity` (line 594)
- `__init__` (line 613)
- `_initialize` (line 620)
- `_init_quantum_computer` (line 629)
- `visualize_bell_state` (line 654)
- `visualize_ghz_state` (line 683)
- `visualize_qft` (line 711)
- `visualize_grover` (line 739)
- `visualize_custom_circuit` (line 778)
- `run_all_visualizations` (line 810)
- `_save_figure` (line 825)
- `_print_summary` (line 842)

#### `relativistic_hydrogen.py`
**Path:** `relativistic_hydrogen.py`

**Classs:**
- `Config` (line 41)
- `LoggerFactory` (line 95)
- `GammaMatrices` (line 113) - *Dirac gamma matrices in Dirac (standard) representation.
gamma^0 = beta, gamma^i = beta * alpha_i*
- `DiracHamiltonianOperator` (line 199) - *Dirac Hamiltonian operator for relativistic quantum mechanics.
H_Dirac = c * alpha . p + beta * m * c^2 + V(r)

In atomic units (c = 1/alpha ~ 137):
H = c * alpha . p + beta * m * c^2 + V*
- `SpectralLayer` (line 310)
- `DiracSpectralNetwork` (line 351) - *Neural network for learning Dirac equation dynamics.
Handles 4-component spinors with real and imaginary parts (8 channels total).*
- `DiracModelWrapper` (line 393) - *Wrapper to load and use the trained Dirac model.*
- `DiracHydrogenAtom` (line 516) - *Relativistic hydrogen atom with Dirac equation.
Computes energy levels including fine structure.*
- `ZitterbewegungSimulator` (line 640) - *Simulates the Zitterbewegung (trembling motion) of a relativistic electron.

In Dirac theory, the position operator has a term oscillating with frequency
~ 2mc^2/hbar, which is the interference between positive and negative energy states.

<x(t)> = <x(0)> + (p/m) * t + oscillating term
The oscillating term has amplitude ~ hbar/(2mc) ~ 10^-12 m*
- `DiracWavefunctionCalculator` (line 833) - *Calculate relativistic hydrogen wavefunctions.*
- `DiracMonteCarloSampler` (line 956) - *Monte Carlo sampling for relativistic orbital visualization.*
- `DiracVisualizer` (line 1075) - *Visualization suite for Dirac equation results.*
- `DiracValidationSuite` (line 1370) - *Complete validation suite for Dirac equation grokking.*

**Functions:**
- `main` (line 1613)
- `create_logger` (line 97)
- `__init__` (line 118)
- `_init_matrices` (line 122)
- `__init__` (line 207)
- `_precompute_operators` (line 215)
- `apply_dirac_hamiltonian` (line 223) - *Apply Dirac Hamiltonian to 4-component spinor.

Args:
    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor
    potential: Optional scalar potential V(r)

Returns:
    H * psi with same shape as input*
- `time_evolution` (line 282) - *Time evolution of Dirac spinor using first-order split-step.
psi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi*
- `__init__` (line 311)
- `forward` (line 322)
- `__init__` (line 356)
- `forward` (line 379)
- `__init__` (line 397)
- `_find_best_checkpoint` (line 406)
- `_load_model` (line 452)
- `apply_hamiltonian` (line 498) - *Apply Hamiltonian using analytical operator.
The NN model learns spinor evolution, but the Hamiltonian operator
is applied analytically for physical validation.*
- `evolve_spinor` (line 506) - *Evolve spinor in time using the analytical Dirac operator.*
- `__init__` (line 521)
- `energy_level_dirac` (line 526) - *Exact Dirac energy level for hydrogen-like atom.

E = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)

For hydrogen (Z=1):
E = mc^2 * [1 + (alpha^2 / (n - |kappa| + sqrt(kappa^2 - alpha^2)))^2]^(-1/2)

Args:
    n: Principal quantum number
    kappa: Relativistic quantum number (kappa = -(l+1) for j=l+1/2, kappa = l for j=l-1/2)

Returns:
    Energy in atomic units (relative to m*c^2)*
- `fine_structure_splitting` (line 556) - *Calculate fine structure splitting for given n, l.

Fine structure includes:
1. Relativistic correction to kinetic energy
2. Spin-orbit coupling
3. Darwin term (for l=0)

Returns energies for j = l+1/2 and j = l-1/2*
- `energy_spectrum` (line 597) - *Generate relativistic energy spectrum up to n_max.*
- `__init__` (line 650)
- `create_gaussian_wave_packet` (line 657) - *Create a Gaussian wave packet for a free particle.

For Dirac, we need a 4-component spinor that's a superposition
of positive energy states.*
- `compute_position_expectation` (line 702) - *Compute expectation value of position operator.
<x> = <psi| x |psi>*
- `compute_velocity_expectation` (line 724) - *Compute expectation value of velocity operator.
In Dirac theory, v = c * alpha

<v_x> = c * <psi| alpha_x |psi>*
- `simulate` (line 750) - *Run Zitterbewegung simulation.

Returns time evolution of position and velocity showing the
oscillatory ZBW term.*
- `__init__` (line 837)
- `radial_wavefunction_schrodinger` (line 843) - *Non-relativistic radial wavefunction for comparison.*
- `radial_wavefunction_dirac` (line 853) - *Relativistic radial wavefunctions for hydrogen.

Returns (f, g) - small and large components.
For bound states, the Dirac radial functions are:
f(r) = sqrt((E+mc^2)/(2E)) * G(r)
g(r) = sqrt((E-mc^2)/(2E)) * F(r)

Simplified version using Sommerfeld fine-structure formula.*
- `spherical_harmonic_real` (line 900) - *Real spherical harmonics.*
- `spin_angular_function` (line 910) - *Spin-angular functions Omega_{kappa,m_j}(theta, phi).

These couple the orbital and spin degrees of freedom.*
- `__init__` (line 960)
- `sample_orbital` (line 966) - *Sample points from a relativistic hydrogen orbital.*
- `__init__` (line 1079)
- `visualize_orbital` (line 1082) - *Visualize relativistic orbital.*
- `visualize_energy_spectrum` (line 1213) - *Visualize relativistic energy spectrum with fine structure.*
- `visualize_zitterbewegung` (line 1296) - *Visualize Zitterbewegung oscillation.*
- `__init__` (line 1374)
- `print_header` (line 1398)
- `validate_fine_structure` (line 1419) - *Validate fine structure energy corrections.*
- `validate_zitterbewegung` (line 1477) - *Validate Zitterbewegung simulation.*
- `validate_energy_spectrum` (line 1509) - *Validate complete energy spectrum.*
- `validate_orbital` (line 1524) - *Validate single orbital visualization.*
- `run_full_validation` (line 1541) - *Run complete validation suite.*
- `interactive_mode` (line 1575) - *Run in interactive mode.*

#### `test_qc_integration.py`
**Path:** `test_qc_integration.py`

**Classs:**
- `TestIntegrationConfig` (line 70) - *IntegrationConfig: centralised configuration with no hardcoded values.*
- `TestGateInstruction` (line 100) - *GateInstruction: lightweight quantum gate descriptor.*
- `TestCircuitIR` (line 128) - *CircuitIR: intermediate representation of a quantum circuit.*
- `TestOpenQasmAdapter` (line 165) - *OpenQasmAdapter: OpenQASM 2.0 string <-> CircuitIR conversion.*
- `TestStandardCircuitFactory` (line 260) - *StandardCircuitFactory: builds CircuitIR for common algorithms.*
- `TestIntegrationBridge` (line 292) - *IntegrationBridge: facade for all format conversions.*
- `TestGateItem` (line 346) - *GateItem: circuit builder gate representation.*
- `TestDashboardConfig` (line 372) - *DashboardConfig: centralised configuration for dashboard.*
- `TestVisualisationEngine` (line 400) - *VisualisationEngine: renders figures from quantum state snapshots.*
- `TestSimulatorBackend` (line 441) - *SimulatorBackend: lightweight wrapper around QC framework.*
- `TestSnapshotData` (line 473) - *SnapshotData: quantum state snapshot for dashboard visualisation.*
- `TestSadPaths` (line 517) - *Edge cases and error conditions across all modules.*
- `TestEndToEnd` (line 585) - *End-to-end scenarios combining multiple modules.*

**Functions:**
- `config` (line 42)
- `bridge` (line 47)
- `qasm_adapter` (line 52)
- `bell_circuit` (line 57)
- `ghz_circuit` (line 62)
- `test_default_config_has_supported_gates` (line 73)
- `test_gate_name_map_is_complete` (line 79)
- `test_reverse_gate_name_map_is_consistent` (line 83)
- `test_qasm_version_default` (line 87)
- `test_max_qubits_defaults_are_positive` (line 90)
- `test_create_single_qubit_gate` (line 103)
- `test_create_two_qubit_gate` (line 109)
- `test_create_gate_with_params` (line 114)
- `test_targets_are_immutable` (line 118)
- `test_create_empty_circuit` (line 131)
- `test_append_gate` (line 136)
- `test_append_gate_out_of_range_raises` (line 141)
- `test_multiple_gates` (line 146)
- `test_repr_includes_qubits_and_gates` (line 152)
- `test_export_bell_state_contains_header` (line 168)
- `test_export_bell_state_has_qreg_and_creg` (line 173)
- `test_export_bell_state_has_gates` (line 178)
- `test_export_ghz_state` (line 183)
- `test_export_qft_has_swap` (line 189)
- `test_export_parametric_gate` (line 194)
- `test_export_exceeds_max_qubits_raises` (line 201)
- `test_import_bell_state_roundtrip` (line 206)
- `test_import_ghz_roundtrip` (line 214)
- `test_import_from_standard_qasm_string` (line 220)
- `test_import_with_parametric_gates` (line 234)
- `test_import_empty_qasm_returns_zero_qubit_circuit` (line 247)
- `test_bell_state_has_two_gates` (line 263)
- `test_bell_state_has_two_qubits` (line 269)
- `test_ghz_state` (line 273)
- `test_qft_three_qubits` (line 278)
- `test_grover_iterations` (line 283)
- `test_export_qasm_returns_string` (line 295)
- `test_import_qasm_roundtrip` (line 300)
- `test_full_openqasm_roundtrip_bell` (line 306)
- `test_full_openqasm_roundtrip_ghz` (line 313)
- `test_full_openqasm_roundtrip_qft` (line 320)
- `test_qiskit_not_available_by_default` (line 327)
- `test_pennylane_not_available_by_default` (line 332)
- `test_export_qasm_custom_qreg_name` (line 337)
- `test_create_single_qubit_gate` (line 349)
- `test_create_two_qubit_gate` (line 357)
- `test_create_parametric_gate` (line 362)
- `test_default_values` (line 375)
- `test_gate_list_includes_standard_gates` (line 382)
- `test_qasm_initial_contains_header` (line 389)
- `test_engine_available_with_matplotlib` (line 403)
- `test_render_full_dashboard_returns_bytes` (line 415)
- `test_synthetic_execute_returns_snapshots` (line 444)
- `test_empty_circuit_returns_init_snapshot` (line 457)
- `test_create_snapshot` (line 476)
- `test_entropy_updates` (line 489)
- `test_probabilities_normalized` (line 500)
- `test_gate_instruction_empty_targets` (line 520)
- `test_circuit_ir_append_negative_qubit_raises` (line 524)
- `test_openqasm_import_empty_string` (line 529)
- `test_openqasm_import_garbage_string` (line 534)
- `test_openqasm_export_zero_qubit_circuit` (line 538)
- `test_standard_circuit_factory_qft_one_qubit` (line 543)
- `test_standard_circuit_factory_grover_minimal` (line 548)
- `test_circuit_ir_repr_no_gates` (line 552)
- `test_synthetic_snapshot_probabilities_sum_to_one` (line 557)
- `test_visualisation_engine_handles_no_snapshots` (line 570)
- `test_build_export_import_qasm_roundtrip` (line 588)
- `test_qasm_to_circuitir_to_framework_mps` (line 598)
- `test_ghz_export_qasm_and_reimport_matches` (line 609)
- `test_qft_circuit_qasm_roundtrip` (line 616)
- `test_export_qasm_with_custom_names` (line 622)
- `test_import_qasm_preserves_gate_order` (line 628)
- `test_full_pipeline_build_export_import_to_mps` (line 644)

#### `test_quantum_framework.py`
**Path:** `test_quantum_framework.py`

**Classs:**
- `TestFrameworkConfig` (line 72) - *Tests for FrameworkConfig.*
- `TestMPSState` (line 95) - *Tests for MPSState.*
- `TestBellState` (line 130) - *Tests for Bell state preparation.*
- `TestGHZState` (line 167) - *Tests for GHZ state preparation.*
- `TestWState` (line 209) - *Tests for W state preparation.*
- `TestQuantumGates` (line 261) - *Tests for quantum gates.*
- `TestPhaseCoherence` (line 338) - *Tests for phase coherence and unitarity.*
- `TestGroverAlgorithm` (line 403) - *Tests for Grover search algorithm.*
- `TestQFT` (line 431) - *Tests for Quantum Fourier Transform.*
- `TestMemoryScaling` (line 458) - *Tests for memory scaling.*
- `TestPrecisionMode` (line 501) - *Tests for precision mode.*
- `TestEdgeCases` (line 525) - *Tests for edge cases.*

**Functions:**
- `config` (line 43) - *Create default framework configuration.*
- `qc` (line 53) - *Create quantum computer instance.*
- `config_precision` (line 59) - *Create precision mode configuration.*
- `test_default_config` (line 75) - *Test default configuration values.*
- `test_custom_config` (line 83) - *Test custom configuration values.*
- `test_create_state` (line 98) - *Test state creation.*
- `test_initial_state` (line 104) - *Test initial |00...0> state.*
- `test_clone_state` (line 113) - *Test state cloning.*
- `test_bell_entropy` (line 133) - *Test Bell state entropy.*
- `test_bell_probabilities` (line 141) - *Test Bell state probabilities.*
- `test_bell_entanglement` (line 154) - *Test Bell state entanglement.*
- `test_ghz_entropy` (line 170) - *Test GHZ state entropy.*
- `test_ghz_probabilities` (line 179) - *Test GHZ state probabilities.*
- `test_ghz_scaling` (line 192) - *Test GHZ state memory scaling.*
- `test_w_state_probabilities` (line 212) - *Test W state probabilities.*
- `test_w_state_entropy` (line 227) - *Test W state entropy.*
- `test_w_state_no_zero_probabilities` (line 247) - *Test that W state has non-zero probabilities for single-excitation states.*
- `test_hadamard_gate` (line 264) - *Test Hadamard gate.*
- `test_pauli_x_gate` (line 275) - *Test Pauli-X gate.*
- `test_pauli_z_gate` (line 285) - *Test Pauli-Z gate on |+> state.*
- `test_cnot_gate` (line 297) - *Test CNOT gate.*
- `test_swap_gate` (line 309) - *Test SWAP gate.*
- `test_rotation_gates` (line 321) - *Test rotation gates.*
- `test_hzh_equals_x` (line 341) - *Test HZH = X identity.*
- `test_xx_equals_identity` (line 354) - *Test XX = I identity.*
- `test_cnot_cnot_equals_identity` (line 366) - *Test CNOT CNOT = I identity.*
- `test_norm_preservation` (line 381) - *Test that norm is preserved after gates.*
- `test_grover_3_qubits` (line 406) - *Test Grover search on 3 qubits.*
- `test_grover_speedup` (line 417) - *Test Grover speedup.*
- `test_qft_entropy` (line 434) - *Test QFT entropy.*
- `test_memory_linear_scaling` (line 461) - *Test that memory scales sub-exponentially with qubits (MPS property).*
- `test_compression_ratio` (line 478) - *Test MPS compression ratio vs full statevector for large n.*
- `test_precision_mode_config` (line 504) - *Test precision mode configuration.*
- `test_precision_mode_bell_state` (line 510) - *Test Bell state in precision mode.*
- `test_single_qubit` (line 528) - *Test single qubit operations.*
- `test_large_bond_dimension` (line 539) - *Test with large bond dimension.*
- `test_empty_circuit` (line 549) - *Test empty circuit.*

#### `topological_hilbert_compression2.py`
**Path:** `topological_hilbert_compression2.py`

**Classs:**
- `HilbertPhase` (line 47)
- `TopologicalCompressionConfig` (line 56)
- `ITensorNetwork` (line 85)
- `MPSCore` (line 115) - *Matrix Product State core tensor A^{[k]}_{i_k} with left and right bond indices.
Shape: (chi_left, d, chi_right) where d=2 for qubits.*
- `MPSState` (line 162) - *Matrix Product State representation of n-qubit quantum state.

|psi> = sum_{i_1...i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} ... A^{[n]}_{i_n} |i_1...i_n>

Memory: O(n * chi^2 * d) vs O(d^n) for full statevector.
For n=30, chi=16: ~30KB vs 8GB for statevector.*
- `VacuumCore` (line 351) - *Vacuum Core architecture that projects irrelevant Hilbert subspace to zero.

Inspired by the HPU-Core achieving 99.996% sparsity:
- Active core: small subspace carrying quantum information
- Vacuum: 99%+ of Hilbert space forced to zero by regularization
- Protection: topological invariants prevent core destruction*
- `TopologicalProtector` (line 411) - *Provides topological protection for quantum states via:
- Winding number monitoring
- Berry phase calculation
- Edge state preservation*
- `HybridBackend` (line 452)
- `DirectBackend` (line 466) - *Direct tensor backend using JointHilbertState representation.
Limited to config.max_qubits_direct qubits due to exponential memory.*
- `MPSBackend` (line 515) - *Matrix Product State backend for scalable quantum simulation.
Handles up to config.max_qubits_mps qubits with sub-exponential memory.*
- `TopologicalHilbertSimulator` (line 572) - *Main simulator implementing hybrid architecture for scalable quantum simulation.

Architecture:
    - n <= max_qubits_direct: HPU-Core direct tensor (exact wavefunction evolution)
    - n > max_qubits_direct: MPS with vacuum core compression*
- `Schrodinger20Experiment` (line 723) - *Validates topological compression on 20-qubit molecular ground state.

Target: H2O (10 electrons, ~20 qubits)
Metrics:
    - Energy error < 1 mHa vs Qiskit VQE
    - Inference time < 100ms
    - Memory < 50MB (vs 500MB for direct tensor)*

**Functions:**
- `main` (line 956)
- `__post_init__` (line 80)
- `amplitude` (line 87)
- `apply_single_qubit_gate` (line 91)
- `apply_two_qubit_gate` (line 95)
- `norm` (line 99)
- `probabilities` (line 103)
- `entropy` (line 107)
- `memory_bytes` (line 111)
- `__init__` (line 121)
- `_initialize` (line 130)
- `tensor` (line 136)
- `tensor` (line 142)
- `left_canonicalize` (line 147)
- `right_canonicalize` (line 154)
- `__init__` (line 172)
- `_initialize` (line 181)
- `_bond_dimension` (line 195)
- `amplitude` (line 198)
- `apply_single_qubit_gate` (line 209)
- `apply_two_qubit_gate` (line 219)
- `_swap_qubits_in_gate` (line 236)
- `_apply_adjacent_gate` (line 245)
- `_apply_nonadjacent_gate` (line 285)
- `norm` (line 294)
- `_canonicalize` (line 301)
- `probabilities` (line 308)
- `entropy` (line 318)
- `memory_bytes` (line 329)
- `entanglement_entropy` (line 335)
- `__init__` (line 361)
- `_initialize` (line 370)
- `_compute_berry_phases` (line 375)
- `add_active_state` (line 381)
- `_compute_winding_number` (line 389)
- `is_topologically_protected` (line 394)
- `sparsity` (line 398)
- `project_to_active` (line 403)
- `__init__` (line 419)
- `compute_winding_number` (line 424)
- `compute_berry_phase` (line 433)
- `is_protected` (line 444)
- `can_handle` (line 454)
- `create_state` (line 458)
- `apply_gate` (line 462)
- `__init__` (line 472)
- `_load_quantum_computer` (line 479)
- `can_handle` (line 497)
- `create_state` (line 500)
- `apply_gate` (line 507)
- `__init__` (line 521)
- `_initialize_gate_cache` (line 526)
- `can_handle` (line 537)
- `create_state` (line 540)
- `apply_gate` (line 543)
- `__init__` (line 581)
- `_select_backend` (line 590)
- `create_circuit` (line 602)
- `h` (line 610)
- `x` (line 613)
- `y` (line 616)
- `z` (line 619)
- `rx` (line 622)
- `ry` (line 625)
- `rz` (line 628)
- `cnot` (line 631)
- `cz` (line 634)
- `swap` (line 637)
- `run` (line 640)
- `probabilities` (line 650)
- `entropy` (line 655)
- `memory_usage` (line 666)
- `compression_ratio` (line 682)
- `detect_phase` (line 691)
- `_compute_average_bond_dimension` (line 714)
- `__init__` (line 734)
- `run_bell_state` (line 739)
- `run_ghz_state` (line 758)
- `run_w_state` (line 777)
- `prepare_ghz_state` (line 798) - *Prepare GHZ state directly without SWAP overhead.

GHZ: |00...0> + |11...1> normalized.
MPS representation:
- A^{[0]}_{i_0,α_1}: A[0,0,0]=1/√2, A[0,1,1]=1/√2, shape (1,2,2)
- A^{[k]}_{α_k,i_k,α_{k+1}}: A[0,0,0]=A[1,1,1]=1, shape (2,2,2)
- A^{[n-1]}_{α_{n-1},i_{n-1},0}: A[0,0,0]=A[1,1,0]=1, shape (2,2,1)*
- `run_scaling_benchmark` (line 844)
- `run_all` (line 897)
- `_print_summary` (line 909)

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
