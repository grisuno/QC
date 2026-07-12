# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 30 | **Total Symbols Extracted:** 1790 | **Total Imports:** 512

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
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

**Classes:**
- `GroverConfig` (line 141) `class GroverConfig` - *Configuration for Grover's algorithm experiments.*
- `GroverOracle` (line 159) `class GroverOracle` - *Oracle for Grover's algorithm.
Marks the target state by applying a phase flip.

Uses the existing MCZGate from quantum_computer.py for multi-controlled Z.*
- `GroverDiffusionOperator` (line 194) `class GroverDiffusionOperator` - *Diffusion operator (Grover diffusion / inversion about mean).

D = 2|s><s| - I where |s> = H^⊗n |0>

Implemented using the existing gate infrastructure.*
- `GroverSearch` (line 240) `class GroverSearch` - *Complete Grover's algorithm implementation using existing quantum_computer.py infrastructure.*
- `QEDConfig` (line 404) `class QEDConfig` - *Configuration for QED effects calculations.*
- `LambShiftCalculator` (line 430) `class LambShiftCalculator` - *Calculates the Lamb shift using Bethe's formula and more accurate methods.

The Lamb shift is the small difference between 2s_{1/2} and 2p_{1/2} energy levels
due to QED effects (vacuum fluctuations and self-energy).

Uses the existing Dirac infrastructure from relativistic_hydrogen.py*
- `AnomalousMagneticMoment` (line 595) `class AnomalousMagneticMoment` - *Calculates the electron's anomalous magnetic moment (g-2).

The electron g-factor is slightly different from 2 due to QED effects:
g = 2(1 + a_e) where a_e = α/(2π) + higher-order terms

Uses the existing Dirac infrastructure for baseline calculations.*
- `QEDEffectsExperiment` (line 722) `class QEDEffectsExperiment` - *Complete QED effects experiment combining Lamb shift and g-2.
Uses existing relativistic_hydrogen.py infrastructure.*
- `PolyatomicMoleculeData` (line 831) `class PolyatomicMoleculeData` - *Data for polyatomic molecules.*
- `MoleculeBuilder` (line 849) `class MoleculeBuilder` - *Build molecule data for VQE calculations.
Uses existing molecular_sim.py infrastructure.*
- `PolyatomicVQE` (line 970) `class PolyatomicVQE` - *VQE solver for polyatomic molecules.
Uses existing molecular_sim.py infrastructure.*
- `PolyatomicExperiment` (line 1105) `class PolyatomicExperiment` - *Complete polyatomic molecule experiment.*
- `AdvancedExperimentRunner` (line 1223) `class AdvancedExperimentRunner` - *Runs all three advanced experiments.*

**Functions:**
- `_make_logger` (line 121) `def _make_logger(name, level)`
- `main` (line 1304) `def main()` - *Main entry point.*
- `__init__` (line 167) `def __init__(self, n_qubits, marked_state)`
- `_validate` (line 172) `def _validate(self)`
- `apply` (line 176) `def apply(self, state, backend)` - *Apply oracle: flip phase of marked state.
|x> -> (-1)^{f(x)} |x> where f(x)=1 only for marked state.

Uses amplitude-level phase manipulation for exact implementation.*
- `__init__` (line 203) `def __init__(self, n_qubits)`
- `apply` (line 206) `def apply(self, state, backend)` - *Apply diffusion operator using Hadamard + Oracle on |0> + Hadamard.

D = H^⊗n (2|0><0| - I) H^⊗n*
- `__init__` (line 245) `def __init__(self, config)`
- `_calculate_entropy` (line 266) `def _calculate_entropy(self, probs)` - *Calculate Shannon entropy from probability distribution.
H = -sum(p * log2(p))*
- `_init_quantum_computer` (line 278) `def _init_quantum_computer(self)` - *Initialize quantum computer using existing quantum_computer.py infrastructure.*
- `run` (line 300) `def run(self)` - *Run Grover's search algorithm.

Returns:
    Dictionary with results including success probability and evolution history.*
- `__init__` (line 440) `def __init__(self, config)`
- `bethe_formula` (line 462) `def bethe_formula(self, n, l, Z)` - *Bethe's non-relativistic formula for Lamb shift.

ΔE_Lamb = (8α^3 / 3πn^3) * |ψ_n(0)|^2 * ln(E_avg / E_n)

For s-states: |ψ_n(0)|^2 = Z^3 / (π n^3 a0^3)

Args:
    n: Principal quantum number
    l: Angular momentum quantum number
    Z: Nuclear charge (default 1 for hydrogen)

Returns:
    Lamb shift in atomic units*
- `_higher_l_shift` (line 505) `def _higher_l_shift(self, n, l, Z)` - *Approximate Lamb shift for l > 0.
Much smaller than for s-states.*
- `full_lamb_shift` (line 518) `def full_lamb_shift(self, n, l, j, Z)` - *Calculate full Lamb shift including radiative corrections.

ΔE = ΔE_SE + ΔE_Uehling + ΔE_rel

Where:
- ΔE_SE: Self-energy (main contribution)
- ΔE_Uehling: Vacuum polarization (Uehling potential)
- ΔE_rel: Relativistic corrections*
- `compare_2s_2p` (line 558) `def compare_2s_2p(self, Z)` - *Compare Lamb shift for 2s_{1/2} and 2p_{1/2} states.

This is the classic Lamb shift measurement: the 2s_{1/2} - 2p_{1/2} splitting.
Experimentally: ~1057.8 MHz*
- `__init__` (line 605) `def __init__(self, config)`
- `schwinger_term` (line 611) `def schwinger_term(self)` - *Schwinger's first-order result: a_e = α/(2π)

This is the leading QED correction.*
- `second_order` (line 619) `def second_order(self)` - *Second-order correction: (α/π)^2 * C_2
C_2 ≈ 0.328478965...*
- `third_order` (line 627) `def third_order(self)` - *Third-order correction: (α/π)^3 * C_3
C_3 ≈ 1.181241456...*
- `fourth_order` (line 635) `def fourth_order(self)` - *Fourth-order correction: (α/π)^4 * C_4
C_4 ≈ -1.9144(35)*
- `fifth_order` (line 643) `def fifth_order(self)` - *Fifth-order correction: (α/π)^5 * C_5
C_5 ≈ 7.7(1.1)*
- `calculate_a_e` (line 651) `def calculate_a_e(self, order)` - *Calculate anomalous magnetic moment to specified order.

Args:
    order: Maximum order to include (1-5)

Returns:
    Dictionary with contributions at each order*
- `full_report` (line 686) `def full_report(self)` - *Generate a full report on g-2 calculations.*
- `__init__` (line 728) `def __init__(self, config)`
- `run_full_analysis` (line 736) `def run_full_analysis(self)` - *Run complete QED analysis.*
- `_calculate_energy_levels` (line 761) `def _calculate_energy_levels(self)` - *Calculate hydrogen energy levels including QED corrections.*
- `_dirac_energy` (line 805) `def _dirac_energy(self, n, kappa)` - *Calculate Dirac energy level.
Uses existing relativistic_hydrogen.py if available.*
- `h2o` (line 866) `def h2o(bond_length, angle_deg)` - *Build water molecule geometry.

H2O geometry:
    H1 at (0, 0, 0)
    O  at (r_OH, 0, 0)
    H2 at (r_OH + r_OH*cos(θ), r_OH*sin(θ), 0)*
- `nh3` (line 898) `def nh3(bond_length, angle_deg)` - *Build ammonia molecule geometry.

NH3 has trigonal pyramidal geometry.*
- `ch4` (line 933) `def ch4(bond_length)` - *Build methane molecule geometry.

CH4 has tetrahedral geometry.*
- `__init__` (line 976) `def __init__(self, config)`
- `run_pyscf` (line 999) `def run_pyscf(self, molecule)` - *Run PySCF calculation for the molecule.*
- `_hardcoded_values` (line 1066) `def _hardcoded_values(self, molecule)` - *Return hardcoded reference values for common molecules.*
- `__init__` (line 1110) `def __init__(self, config)`
- `run_analysis` (line 1123) `def run_analysis(self, molecule_name)` - *Run complete analysis for a molecule.*
- `run_all` (line 1164) `def run_all(self)` - *Run analysis for all molecules.*
- `scan_bond_length` (line 1175) `def scan_bond_length(self, molecule_name, r_min, r_max, n_points)` - *Scan potential energy surface by varying bond length.*
- `__init__` (line 1228) `def __init__(self)`
- `run_grover` (line 1235) `def run_grover(self, n_qubits, marked_state)` - *Run Grover's algorithm experiment.*
- `run_qed` (line 1252) `def run_qed(self)` - *Run QED effects experiment.*
- `run_polyatomic` (line 1266) `def run_polyatomic(self, molecule)` - *Run polyatomic molecule experiment.*
- `run_all` (line 1278) `def run_all(self)` - *Run all experiments.*

#### `app.py`
**Path:** `app.py`

**Classes:**
- `VQEResult` (line 39) `class VQEResult`
- `StarkEvaluator` (line 151) `class StarkEvaluator`
- `DipoleOperatorBuilder` (line 203) `class DipoleOperatorBuilder`
- `PolarizabilityCalculator` (line 237) `class PolarizabilityCalculator`

**Functions:**
- `_sd_indices` (line 46) `def _sd_indices(n_e, n_q)`
- `_run_circuit` (line 55) `def _run_circuit(circuit, backend, state)`
- `givens_single_excitation` (line 61) `def givens_single_excitation(state, o, v, theta, n_qubits, backend)` - *Apply a particle-conserving single excitation rotation between 
qubits o (occupied) and v (virtual).

Rotates in the {|1_o 0_v⟩, |0_o 1_v⟩} subspace:
    |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩
    |0_o 1_v⟩ → -sin(θ)|1_o 0_v⟩ + cos(θ)|0_o 1_v⟩

For adjacent qubits: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)
For non-adjacent: SWAP chain to make adjacent, apply, SWAP back.*
- `particle_conserving_ansatz` (line 109) `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)` - *Particle-conserving UCCSD-like ansatz:
- Singles: Givens rotations (correct particle conservation)
- Doubles: reuse uccsd's double excitation (works correctly)*
- `__init__` (line 152) `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- `_to_scalar` (line 161) `def _to_scalar(amps)`
- `_apply_pauli` (line 165) `def _apply_pauli(self, amps, pauli)`
- `eval_dipole` (line 184) `def eval_dipole(self, amps_raw)`
- `__call__` (line 197) `def __call__(self, amps)`
- `__init__` (line 204) `def __init__(self, bond_length_angstrom)`
- `__init__` (line 238) `def __init__(self)`
- `_evaluator` (line 258) `def _evaluator(self, field)`
- `_get_state` (line 262) `def _get_state(self, theta)`
- `_diagnose` (line 267) `def _diagnose(self, field, theta, label)`
- `_optimize` (line 277) `def _optimize(self, field, theta_init, n_restarts)`
- `run` (line 303) `def run(self)`
- `cost` (line 290) `def cost(th)`

#### `demo_molecular_vqe.py`
**Path:** `demo_molecular_vqe.py`

**Functions:**
- `print_header` (line 47) `def print_header(title)` - *Print formatted header.*
- `check_dependencies` (line 54) `def check_dependencies()` - *Check and report available dependencies.*
- `demo_h2_direct` (line 78) `def demo_h2_direct()` - *Demo: H2 with direct statevector (precision mode).*
- `demo_h2_mps` (line 104) `def demo_h2_mps()` - *Demo: H2 with MPS compression.*
- `demo_comparison` (line 131) `def demo_comparison()` - *Demo: Compare direct vs MPS.*
- `demo_molecule_builder` (line 167) `def demo_molecule_builder()` - *Demo: OpenFermion molecule builder.*
- `demo_smart_initialization` (line 197) `def demo_smart_initialization()` - *Demo: Smart parameter initialization.*
- `demo_cached_operations` (line 231) `def demo_cached_operations()` - *Demo: Cached Pauli operations.*
- `demo_config_from_toml` (line 277) `def demo_config_from_toml()` - *Demo: Configuration from TOML.*
- `main` (line 302) `def main()` - *Run all demos.*

#### `entangled_hydrogen.py`
**Path:** `entangled_hydrogen.py`

**Classes:**
- `EntangledHydrogenConfig` (line 61) `class EntangledHydrogenConfig` - *Configuration for entangled hydrogen visualization system.

All parameters are parametric and configurable from this class.*
- `IEntangledState` (line 114) `class IEntangledState(ABC)` - *Abstract interface for entangled quantum states.*
- `BellState` (line 131) `class BellState(IEntangledState)` - *Bell state: |Phi+> = (|00> + |11>) / sqrt(2).*
- `GHZState` (line 145) `class GHZState(IEntangledState)` - *GHZ state: (|00...0> + |11...1>) / sqrt(2).*
- `WState` (line 162) `class WState(IEntangledState)` - *W state: (|001> + |010> + |100>) / sqrt(3).*
- `WavefunctionCalculator` (line 194) `class WavefunctionCalculator` - *Calculates hydrogen atom wavefunctions for entangled state visualization.
Uses the same implementation as orbital_visualizer2.py.*
- `EntangledHydrogenSampler` (line 253) `class EntangledHydrogenSampler` - *Monte Carlo sampler for entangled hydrogen states.
Samples from the joint probability distribution of entangled orbitals.*
- `EntangledHydrogenVisualizer` (line 408) `class EntangledHydrogenVisualizer` - *Visualizer for entangled hydrogen states.
Creates high-resolution visualizations similar to orbital_visualizer2.py.*
- `EntangledHydrogenExperiment` (line 569) `class EntangledHydrogenExperiment` - *Main experiment class for entangled hydrogen visualization.

Uses the existing quantum_computer.py, molecular_sim.py, and
visualization components from orbital_visualizer2.py and relativistic_hydrogen.py.*

**Functions:**
- `_make_logger` (line 44) `def _make_logger(name)` - *Create a module-level logger with consistent formatter.*
- `main` (line 893) `def main()` - *Main entry point for entangled hydrogen visualization.*
- `name` (line 119) `def name(self)` - *Return the name of the entangled state.*
- `prepare` (line 123) `def prepare(self, n_qubits)` - *Prepare the entangled state on n qubits.*
- `get_theoretical_entropy` (line 127) `def get_theoretical_entropy(self)` - *Return the theoretical Shannon entropy in bits.*
- `name` (line 135) `def name(self)`
- `prepare` (line 138) `def prepare(self, qc, backend)`
- `get_theoretical_entropy` (line 141) `def get_theoretical_entropy(self)`
- `__init__` (line 148) `def __init__(self, n_qubits)`
- `name` (line 152) `def name(self)`
- `prepare` (line 155) `def prepare(self, qc, backend)`
- `get_theoretical_entropy` (line 158) `def get_theoretical_entropy(self)`
- `__init__` (line 165) `def __init__(self, n_qubits)`
- `name` (line 169) `def name(self)`
- `prepare` (line 172) `def prepare(self, qc, backend, factory)`
- `get_theoretical_entropy` (line 190) `def get_theoretical_entropy(self)`
- `__init__` (line 200) `def __init__(self, config)`
- `radial_wavefunction` (line 204) `def radial_wavefunction(n, l, r)` - *Calculate non-relativistic radial wavefunction R_nl(r).*
- `spherical_harmonic_real` (line 215) `def spherical_harmonic_real(l, m, theta, phi)` - *Calculate real spherical harmonics Y_lm(theta, phi).*
- `psi_on_grid` (line 225) `def psi_on_grid(self, n, l, m)` - *Calculate wavefunction on 2D grid for quantum computer processing.*
- `psi_3d` (line 245) `def psi_3d(self, n, l, m, r, theta, phi)` - *Calculate full 3D wavefunction psi_nlm(r, theta, phi).*
- `__init__` (line 259) `def __init__(self, config, wavefunction_calc)`
- `find_max_probability` (line 263) `def find_max_probability(self, n, l, m)` - *Find maximum probability for rejection sampling.*
- `sample_orbital` (line 295) `def sample_orbital(self, n, l, m, num_samples)` - *Sample points from a single hydrogen orbital using Monte Carlo.*
- `sample_entangled_state` (line 362) `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples, entanglement_weight)` - *Sample from entangled hydrogen state.

Creates a superposition of two orbitals with entanglement_weight:
|psi> = sqrt(1-w) * |n1,l1,m1> + sqrt(w) * |n2,l2,m2>

For true entanglement visualization, we create a joint state:
|Psi> = (|n1,l1,m1>|n2,l2,m2> + |n2,l2,m2>|n1,l1,m1>) / sqrt(2)*
- `__init__` (line 414) `def __init__(self, config)`
- `visualize` (line 417) `def visualize(self, data, quantum_result, save_path)` - *Create visualization of entangled hydrogen state.*
- `__init__` (line 577) `def __init__(self, config)`
- `_initialize_quantum_computer` (line 591) `def _initialize_quantum_computer(self)` - *Initialize the quantum computer using the existing quantum_computer.py module.*
- `run_bell_entangled_hydrogen` (line 630) `def run_bell_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples, suffix)` - *Run Bell state entangled hydrogen visualization.

Creates a Bell state and correlates it with hydrogen orbitals.*
- `run_ghz_entangled_hydrogen` (line 672) `def run_ghz_entangled_hydrogen(self, orbitals, backend, num_samples)` - *Run GHZ state entangled hydrogen visualization.

Creates a GHZ state with n qubits and correlates with n orbitals.*
- `run_entangled_h_with_molecular_energy` (line 745) `def run_entangled_h_with_molecular_energy(self, n1, l1, m1, n2, l2, m2, backend, num_samples)` - *Run entangled hydrogen with molecular energy evaluation using molecular_sim.py.*
- `run_relativistic_entangled_hydrogen` (line 794) `def run_relativistic_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples)` - *Run entangled hydrogen with relativistic Dirac calculations.
Uses components from relativistic_hydrogen.py.*
- `run_all_demonstrations` (line 852) `def run_all_demonstrations(self, num_samples)` - *Run all available entangled hydrogen demonstrations.*

#### `higgs_four_lepton_analysis.py`
**Path:** `higgs_four_lepton_analysis.py`

**Classes:**
- `LeptonType` (line 72) `class LeptonType(Enum)`
- `EventType` (line 78) `class EventType(Enum)`
- `Config` (line 86) `class Config`
- `FourMomentum` (line 125) `class FourMomentum`
- `Lepton` (line 164) `class Lepton`
- `Event` (line 183) `class Event`
- `QuantumSpinorProcessor` (line 205) `class QuantumSpinorProcessor` - *Uses the ACTUAL DiracBackend from quantum_computer.py for spinor calculations.

This is NOT a fake implementation - it uses the neural network
spectral layers trained (or randomly initialized) for Dirac spinor evolution.*
- `EventParser` (line 377) `class EventParser` - *Parse CMS CSV files with actual column names.*
- `Visualizer` (line 436) `class Visualizer` - *3D visualization using Plotly with quantum-processed trajectories.*
- `HiggsQuantumAnalysis` (line 612) `class HiggsQuantumAnalysis` - *Main analysis class using quantum backends from quantum_computer.py*

**Functions:**
- `_make_logger` (line 57) `def _make_logger(name)`
- `main` (line 747) `def main()`
- `__post_init__` (line 136) `def __post_init__(self)`
- `from_energy_momentum` (line 151) `def from_energy_momentum(cls, E, px, py, pz)`
- `__add__` (line 154) `def __add__(self, other)`
- `pt` (line 171) `def pt(self)`
- `eta` (line 173) `def eta(self)`
- `phi` (line 175) `def phi(self)`
- `energy` (line 177) `def energy(self)`
- `mass` (line 179) `def mass(self)`
- `__post_init__` (line 193) `def __post_init__(self)`
- `check_higgs` (line 200) `def check_higgs(self, cfg)`
- `__init__` (line 213) `def __init__(self, config)`
- `_precompute_momentum_grids` (line 247) `def _precompute_momentum_grids(self)` - *Precompute k-space grids for Dirac equation.*
- `momentum_to_spinor_wavefunction` (line 255) `def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)` - *Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)
suitable for processing by the DiracBackend.

The wavefunction encodes the momentum as a plane wave with the correct
relativistic dispersion relation.*
- `evolve_with_dirac_backend` (line 296) `def evolve_with_dirac_backend(self, psi, steps)` - *Evolve a wavefunction using the DiracBackend neural network.

This applies the actual spectral layers from the trained (or random)
Dirac network.*
- `evolve_with_schrodinger_backend` (line 308) `def evolve_with_schrodinger_backend(self, psi, steps)` - *Evolve using the SchrodingerBackend neural network.*
- `evolve_with_hamiltonian_backend` (line 315) `def evolve_with_hamiltonian_backend(self, psi, steps)` - *Evolve using the HamiltonianBackend neural network.*
- `compute_dirac_current` (line 322) `def compute_dirac_current(self, px, py, pz, energy, mass, charge)` - *Compute the Dirac current using the actual neural network backends.

Returns (jx, jy, jz, info_dict) where info contains quantum observables.*
- `compute_spinor_amplitude` (line 370) `def compute_spinor_amplitude(self, psi)` - *Compute complex amplitude from wavefunction for helicity analysis.*
- `__init__` (line 380) `def __init__(self, config)`
- `parse_file` (line 383) `def parse_file(self, filepath, event_type)`
- `_parse_row` (line 396) `def _parse_row(self, row, event_type)`
- `__init__` (line 439) `def __init__(self, config, quantum_processor)`
- `compute_quantum_helix` (line 443) `def compute_quantum_helix(self, px, py, pz, charge, mass, energy)` - *Compute helical trajectory using actual DiracBackend evolution.*
- `create_visualization` (line 474) `def create_visualization(self, events, output_path)`
- `_create_detector` (line 566) `def _create_detector(self)`
- `_create_explosion` (line 581) `def _create_explosion(self, vx, vy, vz, energy)`
- `__init__` (line 617) `def __init__(self, config)`
- `fetch_data` (line 630) `def fetch_data(self)`
- `load_events` (line 645) `def load_events(self)`
- `analyze_with_quantum_backends` (line 672) `def analyze_with_quantum_backends(self)` - *Run analysis using actual quantum backends.*
- `generate_visualization` (line 728) `def generate_visualization(self)`
- `run` (line 734) `def run(self)`

#### `higgs_quantum_analysis.py`
**Path:** `higgs_quantum_analysis.py`

**Classes:**
- `LeptonType` (line 72) `class LeptonType(Enum)`
- `EventType` (line 78) `class EventType(Enum)`
- `Config` (line 86) `class Config`
- `FourMomentum` (line 125) `class FourMomentum`
- `Lepton` (line 164) `class Lepton`
- `Event` (line 183) `class Event`
- `QuantumSpinorProcessor` (line 205) `class QuantumSpinorProcessor` - *Uses the ACTUAL DiracBackend from quantum_computer.py for spinor calculations.

This is NOT a fake implementation - it uses the neural network
spectral layers trained (or randomly initialized) for Dirac spinor evolution.*
- `EventParser` (line 377) `class EventParser` - *Parse CMS CSV files with actual column names.*
- `Visualizer` (line 436) `class Visualizer` - *3D visualization using Plotly with quantum-processed trajectories.*
- `HiggsQuantumAnalysis` (line 612) `class HiggsQuantumAnalysis` - *Main analysis class using quantum backends from quantum_computer.py*

**Functions:**
- `_make_logger` (line 57) `def _make_logger(name)`
- `main` (line 747) `def main()`
- `__post_init__` (line 136) `def __post_init__(self)`
- `from_energy_momentum` (line 151) `def from_energy_momentum(cls, E, px, py, pz)`
- `__add__` (line 154) `def __add__(self, other)`
- `pt` (line 171) `def pt(self)`
- `eta` (line 173) `def eta(self)`
- `phi` (line 175) `def phi(self)`
- `energy` (line 177) `def energy(self)`
- `mass` (line 179) `def mass(self)`
- `__post_init__` (line 193) `def __post_init__(self)`
- `check_higgs` (line 200) `def check_higgs(self, cfg)`
- `__init__` (line 213) `def __init__(self, config)`
- `_precompute_momentum_grids` (line 247) `def _precompute_momentum_grids(self)` - *Precompute k-space grids for Dirac equation.*
- `momentum_to_spinor_wavefunction` (line 255) `def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)` - *Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)
suitable for processing by the DiracBackend.

The wavefunction encodes the momentum as a plane wave with the correct
relativistic dispersion relation.*
- `evolve_with_dirac_backend` (line 296) `def evolve_with_dirac_backend(self, psi, steps)` - *Evolve a wavefunction using the DiracBackend neural network.

This applies the actual spectral layers from the trained (or random)
Dirac network.*
- `evolve_with_schrodinger_backend` (line 308) `def evolve_with_schrodinger_backend(self, psi, steps)` - *Evolve using the SchrodingerBackend neural network.*
- `evolve_with_hamiltonian_backend` (line 315) `def evolve_with_hamiltonian_backend(self, psi, steps)` - *Evolve using the HamiltonianBackend neural network.*
- `compute_dirac_current` (line 322) `def compute_dirac_current(self, px, py, pz, energy, mass, charge)` - *Compute the Dirac current using the actual neural network backends.

Returns (jx, jy, jz, info_dict) where info contains quantum observables.*
- `compute_spinor_amplitude` (line 370) `def compute_spinor_amplitude(self, psi)` - *Compute complex amplitude from wavefunction for helicity analysis.*
- `__init__` (line 380) `def __init__(self, config)`
- `parse_file` (line 383) `def parse_file(self, filepath, event_type)`
- `_parse_row` (line 396) `def _parse_row(self, row, event_type)`
- `__init__` (line 439) `def __init__(self, config, quantum_processor)`
- `compute_quantum_helix` (line 443) `def compute_quantum_helix(self, px, py, pz, charge, mass, energy)` - *Compute helical trajectory using actual DiracBackend evolution.*
- `create_visualization` (line 474) `def create_visualization(self, events, output_path)`
- `_create_detector` (line 566) `def _create_detector(self)`
- `_create_explosion` (line 581) `def _create_explosion(self, vx, vy, vz, energy)`
- `__init__` (line 617) `def __init__(self, config)`
- `fetch_data` (line 630) `def fetch_data(self)`
- `load_events` (line 645) `def load_events(self)`
- `analyze_with_quantum_backends` (line 672) `def analyze_with_quantum_backends(self)` - *Run analysis using actual quantum backends.*
- `generate_visualization` (line 728) `def generate_visualization(self)`
- `run` (line 734) `def run(self)`

#### `molecular_sim.py`
**Path:** `molecular_sim.py`

**Classes:**
- `MoleculeData` (line 35) `class MoleculeData`
- `ExactJWEnergy` (line 187) `class ExactJWEnergy` - *Evaluador exacto usando OpenFermion JW.*
- `SurrogateEnergy` (line 265) `class SurrogateEnergy` - *Backend neuronal con calibración.*
- `VQEResult` (line 371) `class VQEResult`
- `VQESolver` (line 390) `class VQESolver`

**Functions:**
- `_make_logger` (line 18) `def _make_logger(name)`
- `_h2_sto3g_pyscf` (line 41) `def _h2_sto3g_pyscf()` - *H2 con datos PySCF.*
- `_h2_sto3g_hardcoded` (line 75) `def _h2_sto3g_hardcoded()` - *H2 con datos hardcodeados.*
- `build_jw_hamiltonian_of` (line 111) `def build_jw_hamiltonian_of(mol)` - *Construye Hamiltoniano JW usando OpenFermion correctamente.
Usa MolecularData de OpenFermion para asegurar consistencia.*
- `prepare_hf` (line 304) `def prepare_hf(mol, factory, backend)`
- `_sd_indices` (line 312) `def _sd_indices(n_e, n_q)`
- `uccsd` (line 321) `def uccsd(state, thetas, singles, doubles, backend, runner)` - *UCCSD ansatz.

- Singles: cadena JW estándar, conserva número de partículas.
- Doubles: rotación Givens en el subespacio {|HF⟩, |exc⟩} determinado
  dinámicamente desde las amplitudes (robusto ante cambio de convención).
- theta=0 para cualquier parámetro → circuito vacío → identidad exacta.*
- `__init__` (line 190) `def __init__(self, mol, n_qubits)`
- `_to_scalar` (line 203) `def _to_scalar(amps)` - *(dim,2,G,G) → (dim,2)  ó  (dim,2) → (dim,2).
Integra la función de onda espacial sobre la grilla manteniendo
la estructura re/im que necesita el evaluador JW.*
- `_verify_hf` (line 213) `def _verify_hf(self)`
- `_apply` (line 223) `def _apply(self, amps, pauli)`
- `_evaluate` (line 243) `def _evaluate(self, amps)`
- `__call__` (line 257) `def __call__(self, amps)`
- `__init__` (line 268) `def __init__(self, mol, n_qubits, exact_eval, backend)`
- `calibrate` (line 277) `def calibrate(self, hf_amps)`
- `cost_with_barrier` (line 290) `def cost_with_barrier(self, amps)`
- `__repr__` (line 377) `def __repr__(self)`
- `__init__` (line 391) `def __init__(self, qc, config)`
- `_run` (line 395) `def _run(self, circ, be, state)`
- `run` (line 403) `def run(self, mol, backend, max_iter, tol)`
- `cost` (line 451) `def cost(thetas)`

#### `orbital_visualizer2.py`
**Path:** `orbital_visualizer2.py`

**Classes:**
- `Config` (line 44) `class Config`
- `WavefunctionCalculator` (line 83) `class WavefunctionCalculator` - *Calculates hydrogen atom wavefunctions.*
- `HamiltonianNNProcessor` (line 126) `class HamiltonianNNProcessor` - *Uses YOUR TRAINED MODEL for calculations.*
- `MonteCarloSampler` (line 160) `class MonteCarloSampler` - *Monte Carlo sampling for orbital visualization.*
- `OrbitalVisualizer` (line 265) `class OrbitalVisualizer` - *HIGH RESOLUTION visualization - NOT 16x16!*

**Functions:**
- `main` (line 433) `def main()`
- `radial_wavefunction` (line 87) `def radial_wavefunction(n, l, r)`
- `spherical_harmonic_real` (line 97) `def spherical_harmonic_real(l, m, theta, phi)`
- `psi_on_grid` (line 107) `def psi_on_grid(n, l, m, grid_size)`
- `__init__` (line 129) `def __init__(self, engine)`
- `is_model_loaded` (line 133) `def is_model_loaded(self)`
- `compute_expected_energy` (line 136) `def compute_expected_energy(self, n, l, m)`
- `__init__` (line 163) `def __init__(self, hamiltonian_processor)`
- `find_max_probability` (line 166) `def find_max_probability(self, n, l, m)`
- `sample` (line 195) `def sample(self, n, l, m, num_samples)`
- `visualize` (line 268) `def visualize(self, data, save_path, hamiltonian_processor)`
- `_plotly` (line 396) `def _plotly(self, X, Y, Z, prob_norm, phases, n, l, m)`

#### `polarizability_v3.py`
**Path:** `polarizability_v3.py`

**Classes:**
- `VQEResult` (line 51) `class VQEResult`
- `StarkEvaluator` (line 163) `class StarkEvaluator`
- `DipoleOperatorBuilder` (line 215) `class DipoleOperatorBuilder`
- `PolarizabilityCalculator` (line 249) `class PolarizabilityCalculator`

**Functions:**
- `_sd_indices` (line 58) `def _sd_indices(n_e, n_q)`
- `_run_circuit` (line 67) `def _run_circuit(circuit, backend, state)`
- `givens_single_excitation` (line 73) `def givens_single_excitation(state, o, v, theta, n_qubits, backend)` - *Apply a particle-conserving single excitation rotation between 
qubits o (occupied) and v (virtual).

Rotates in the {|1_o 0_v⟩, |0_o 1_v⟩} subspace:
    |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩
    |0_o 1_v⟩ → -sin(θ)|1_o 0_v⟩ + cos(θ)|0_o 1_v⟩

For adjacent qubits: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)
For non-adjacent: SWAP chain to make adjacent, apply, SWAP back.*
- `particle_conserving_ansatz` (line 121) `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)` - *Particle-conserving UCCSD-like ansatz:
- Singles: Givens rotations (correct particle conservation)
- Doubles: reuse uccsd's double excitation (works correctly)*
- `__init__` (line 164) `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- `_to_scalar` (line 173) `def _to_scalar(amps)`
- `_apply_pauli` (line 177) `def _apply_pauli(self, amps, pauli)`
- `eval_dipole` (line 196) `def eval_dipole(self, amps_raw)`
- `__call__` (line 209) `def __call__(self, amps)`
- `__init__` (line 216) `def __init__(self, bond_length_angstrom)`
- `__init__` (line 250) `def __init__(self)`
- `_evaluator` (line 270) `def _evaluator(self, field)`
- `_get_state` (line 274) `def _get_state(self, theta)`
- `_diagnose` (line 279) `def _diagnose(self, field, theta, label)`
- `_optimize` (line 289) `def _optimize(self, field, theta_init, n_restarts)`
- `run` (line 315) `def run(self)`
- `cost` (line 302) `def cost(th)`

#### `qc_dashboard.py`
**Path:** `qc_dashboard.py`

**Classes:**
- `DashboardConfig` (line 55) `class DashboardConfig` - *Centralised configuration for the dashboard.*
- `GateItem` (line 120) `class GateItem` - *A gate placed in the circuit builder.*
- `SnapshotData` (line 130) `class SnapshotData` - *Quantum state snapshot for visualisation.*
- `H2VQESolver` (line 147) `class H2VQESolver` - *Self-contained H2 VQE solver using the hardcoded STO-3G Hamiltonian.

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
- `VisualisationEngine` (line 340) `class VisualisationEngine` - *Renders figures from quantum state snapshots using matplotlib.*
- `SimulatorBackend` (line 725) `class SimulatorBackend` - *Thin wrapper around the QC framework simulator for dashboard use.*
- `Plotly3DEngine` (line 979) `class Plotly3DEngine` - *Optional Plotly-based 3D visualisations.*
- `RealOrbitalEngine` (line 1151) `class RealOrbitalEngine` - *Wrapper around the repo's real orbital_visualizer2.py scripts.*
- `BrutalVizEngine` (line 1269) `class BrutalVizEngine` - *Wrapper around the repo's real quantum_dash.py and quantum_3dview.py.*
- `BackendComparator` (line 1344) `class BackendComparator` - *Compare circuit execution across all available backends.*
- `DashboardApp` (line 1371) `class DashboardApp` - *Streamlit-based interactive quantum playground.*
- `_FakeConfig` (line 1176) `class _FakeConfig`

**Functions:**
- `_capture_mpl_fig` (line 1133) `def _capture_mpl_fig(func)` - *Run a function that creates a matplotlib figure and capture it as PNG bytes.*
- `main` (line 1994) `def main()` - *Launch the Streamlit dashboard.*
- `__init__` (line 166) `def __init__(self, config)`
- `_pauli_operators` (line 173) `def _pauli_operators(n_qubits, qubit, op)`
- `_pauli_string_matrix` (line 189) `def _pauli_string_matrix(paulis, n_qubits)`
- `_build_hamiltonian` (line 206) `def _build_hamiltonian(self, bond_length)` - *Build H2 Hamiltonian matrix for given bond length (Angstrom).

The coefficients scale with bond length to reproduce the Morse-like well.*
- `_ansatz_state` (line 229) `def _ansatz_state(theta)` - *UCC-like ansatz for H2: |psi(theta)> = cos(theta)*|10> + sin(theta)*|01>.*
- `_energy` (line 241) `def _energy(self, theta, h_matrix, e_nuc)`
- `run_vqe` (line 248) `def run_vqe(self, bond_length, max_iter)`
- `energy_landscape` (line 293) `def energy_landscape(self)` - *Sweep bond length and return VQE energy curve.*
- `orbital_wavefunction` (line 318) `def orbital_wavefunction(bond_length, grid_points)` - *Compute hydrogen 1s orbital wavefunction along the internuclear axis.*
- `__init__` (line 343) `def __init__(self, config)`
- `_init_plotting` (line 349) `def _init_plotting(self)`
- `available` (line 362) `def available(self)`
- `render_full_dashboard` (line 365) `def render_full_dashboard(self, snapshots, current)`
- `render_entropy_chart` (line 403) `def render_entropy_chart(self, snapshots)`
- `render_entanglement_profile` (line 429) `def render_entanglement_profile(self, snapshots)` - *Entropy vs cut position for the latest snapshot.*
- `render_vqe_convergence` (line 474) `def render_vqe_convergence(self, convergence, e_hf, e_fci)`
- `render_energy_landscape` (line 501) `def render_energy_landscape(self, bond_lengths, vqe_energies, hf_energies, fci_energies)`
- `render_orbital_plot` (line 533) `def render_orbital_plot(self, orbital_data)`
- `render_entropy_scaling` (line 562) `def render_entropy_scaling(self, data)`
- `_render_probabilities` (line 589) `def _render_probabilities(self, snap, ax)`
- `_render_bloch_sphere` (line 606) `def _render_bloch_sphere(self, snap, ax)`
- `_render_phase_space` (line 635) `def _render_phase_space(self, snap, ax)`
- `render_orbital_2d_projections` (line 670) `def render_orbital_2d_projections(self, data)`
- `_hex_to_rgb` (line 715) `def _hex_to_rgb(h)`
- `__init__` (line 728) `def __init__(self, config)`
- `_init_framework` (line 737) `def _init_framework(self)`
- `_get_mps_gate_registry` (line 760) `def _get_mps_gate_registry()`
- `_get_sv_gate_registry` (line 768) `def _get_sv_gate_registry()`
- `execute_circuit` (line 775) `def execute_circuit(self, gates, n_qubits)`
- `_mps_execute` (line 788) `def _mps_execute(self, gates, n_qubits)`
- `_sv_execute` (line 839) `def _sv_execute(self, gates, n_qubits)`
- `_snapshot_from_mps` (line 886) `def _snapshot_from_mps(self, state, step, gate_name)`
- `_snapshot_from_sv` (line 898) `def _snapshot_from_sv(self, state, step, gate_name)`
- `_compute_bloch_mps` (line 917) `def _compute_bloch_mps(state, n_qubits)`
- `_synthetic_execute` (line 933) `def _synthetic_execute(self, gates, n_qubits)`
- `__init__` (line 982) `def __init__(self, config)`
- `_init_plotly` (line 988) `def _init_plotly(self)`
- `available` (line 998) `def available(self)`
- `render_bloch_3d` (line 1001) `def render_bloch_3d(self, bloch_vectors)`
- `render_probability_3d` (line 1049) `def render_probability_3d(self, probabilities, n_qubits)`
- `render_state_3d` (line 1080) `def render_state_3d(self, probabilities, phases)` - *3D scatter plot: X=real, Y=imaginary, Z=probability.*
- `__init__` (line 1154) `def __init__(self)`
- `_init_real` (line 1163) `def _init_real(self)`
- `available` (line 1210) `def available(self)`
- `entangled_available` (line 1214) `def entangled_available(self)`
- `sample` (line 1217) `def sample(self, n, l, m, num_samples)`
- `render_to_bytes` (line 1222) `def render_to_bytes(self, data)`
- `sample_entangled` (line 1240) `def sample_entangled(self, n1, l1, m1, n2, l2, m2, num_samples)`
- `render_entangled_to_bytes` (line 1250) `def render_entangled_to_bytes(self, data)`
- `__init__` (line 1272) `def __init__(self)`
- `_init_real` (line 1279) `def _init_real(self)`
- `dash_available` (line 1297) `def dash_available(self)`
- `hologram_available` (line 1301) `def hologram_available(self)`
- `run_brutal_viz` (line 1304) `def run_brutal_viz(self, circuit_name)`
- `render_hologram` (line 1327) `def render_hologram(self, snapshots, backend_comp)`
- `__init__` (line 1347) `def __init__(self, config)`
- `run_comparison` (line 1351) `def run_comparison(self, gates, n_qubits)`
- `__init__` (line 1374) `def __init__(self, config)`
- `run` (line 1384) `def run(self)`
- `_ensure_streamlit` (line 1399) `def _ensure_streamlit()`
- `_init_session` (line 1408) `def _init_session(st)`
- `_render_ui` (line 1432) `def _render_ui(self, st)`
- `_render_sidebar_controls` (line 1447) `def _render_sidebar_controls(self, st)`
- `_render_main_panel` (line 1535) `def _render_main_panel(self, st)`
- `_render_playground_tab` (line 1563) `def _render_playground_tab(self, st)`
- `_render_qasm_tab` (line 1606) `def _render_qasm_tab(self, st)`
- `_render_entropy_tab` (line 1669) `def _render_entropy_tab(self, st)`
- `_render_entanglement_tab` (line 1688) `def _render_entanglement_tab(self, st)`
- `_render_molecule_tab` (line 1732) `def _render_molecule_tab(self, st)`
- `_render_3d_tab` (line 1803) `def _render_3d_tab(self, st)`
- `_render_orbital_tab` (line 1855) `def _render_orbital_tab(self, st)`
- `_auto_run` (line 1958) `def _auto_run(self, st)`
- `_build_qasm_from_gates` (line 1969) `def _build_qasm_from_gates(self, gates)`
- `_parse_orb` (line 1936) `def _parse_orb(k)`

#### `qc_integration.py`
**Path:** `qc_integration.py`

**Classes:**
- `IntegrationConfig` (line 77) `class IntegrationConfig` - *Centralised configuration for the integration bridge.

All tunable parameters live here -- no hardcoded values in logic.*
- `GateInstruction` (line 130) `class GateInstruction` - *A single quantum gate instruction.*
- `CircuitIR` (line 146) `class CircuitIR` - *Intermediate representation of a quantum circuit.

This is the lingua franca between the QC framework and external formats.*
- `IQCAdapter` (line 181) `class IQCAdapter(ABC)` - *Interface for a format-specific adapter (Interface Segregation).*
- `OpenQasmAdapter` (line 226) `class OpenQasmAdapter(IQCAdapter)` - *Adapter for OpenQASM 2.0 format.*
- `QiskitAdapter` (line 358) `class QiskitAdapter(IQCAdapter)` - *Adapter for Qiskit QuantumCircuit format.*
- `PennyLaneAdapter` (line 455) `class PennyLaneAdapter(IQCAdapter)` - *Adapter for PennyLane format.*
- `FrameworkAdapter` (line 557) `class FrameworkAdapter` - *Converts between CircuitIR and the actual QC framework circuit types.

Supports both:
  - quantum_framework_core.MPSQuantumComputer / QuantumCircuit (MPS)
  - quantum_computer.QuantumComputer / QuantumCircuit (statevector)*
- `StandardCircuitFactory` (line 659) `class StandardCircuitFactory` - *Build CircuitIR instances for common quantum algorithms.*
- `IntegrationBridge` (line 727) `class IntegrationBridge` - *Facade that exposes all format conversions through a single API.

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
- `main` (line 851) `def main()` - *Command-line interface for the integration bridge.*
- `__post_init__` (line 115) `def __post_init__(self)`
- `__post_init__` (line 137) `def __post_init__(self)`
- `num_qubits` (line 141) `def num_qubits(self)`
- `append` (line 156) `def append(self, gate)`
- `__len__` (line 164) `def __len__(self)`
- `__repr__` (line 167) `def __repr__(self)`
- `export` (line 185) `def export(self, circuit)` - *Export a CircuitIR to the target format.*
- `import_` (line 189) `def import_(self, data)` - *Import from the target format into a CircuitIR.*
- `__init__` (line 238) `def __init__(self, config)`
- `export` (line 243) `def export(self, circuit)`
- `import_` (line 277) `def import_(self, data)` - *Parse an OpenQASM 2.0 string into a CircuitIR.*
- `_param_names_for_gate` (line 339) `def _param_names_for_gate(name)`
- `__init__` (line 361) `def __init__(self, config)`
- `export` (line 364) `def export(self, circuit)`
- `import_` (line 379) `def import_(self, data)`
- `_ensure_qiskit` (line 403) `def _ensure_qiskit()`
- `_build_qiskit_method_map` (line 414) `def _build_qiskit_method_map(qc)`
- `_build_reverse_gate_map` (line 433) `def _build_reverse_gate_map()`
- `__init__` (line 458) `def __init__(self, config)`
- `export` (line 461) `def export(self, circuit)`
- `import_` (line 483) `def import_(self, data)`
- `_ensure_pennylane` (line 506) `def _ensure_pennylane()`
- `_build_gate_ops` (line 517) `def _build_gate_ops(pl)`
- `_build_reverse_ops` (line 535) `def _build_reverse_ops(pl)`
- `__init__` (line 565) `def __init__(self, config)`
- `to_circuit_ir` (line 568) `def to_circuit_ir(self, circuit, framework_type)` - *Extract a CircuitIR from a framework circuit object.*
- `from_circuit_ir` (line 584) `def from_circuit_ir(self, cir, framework_type)` - *Build a framework circuit object from a CircuitIR.*
- `_detect_framework` (line 598) `def _detect_framework(circuit)`
- `_from_mps_circuit` (line 607) `def _from_mps_circuit(circuit)`
- `_to_mps_circuit` (line 623) `def _to_mps_circuit(cir)`
- `_from_sv_circuit` (line 631) `def _from_sv_circuit(circuit)`
- `_to_sv_circuit` (line 647) `def _to_sv_circuit(cir)`
- `bell_state` (line 663) `def bell_state()`
- `ghz_state` (line 670) `def ghz_state(n_qubits)`
- `qft` (line 678) `def qft(n_qubits)`
- `w_state` (line 693) `def w_state(n_qubits)`
- `grover` (line 701) `def grover(n_qubits, marked, iterations)`
- `__init__` (line 753) `def __init__(self, config)`
- `_init_optional_adapters` (line 763) `def _init_optional_adapters(self)`
- `export_qasm` (line 780) `def export_qasm(self, circuit)`
- `import_qasm` (line 783) `def import_qasm(self, qasm_str)`
- `to_qiskit` (line 788) `def to_qiskit(self, circuit)`
- `from_qiskit` (line 793) `def from_qiskit(self, qiskit_circuit)`
- `to_pennylane` (line 800) `def to_pennylane(self, circuit)`
- `from_pennylane` (line 805) `def from_pennylane(self, pennylane_data)`
- `to_circuit_ir` (line 812) `def to_circuit_ir(self, circuit, framework_type)`
- `from_circuit_ir` (line 819) `def from_circuit_ir(self, cir, framework_type)`
- `export_qasm_from_framework` (line 828) `def export_qasm_from_framework(self, circuit, framework_type)`
- `import_qasm_to_framework` (line 837) `def import_qasm_to_framework(self, qasm_str, framework_type)`
- `circuit_fn` (line 467) `def circuit_fn()`

#### `quantum_3dview.py`
**Path:** `quantum_3dview.py`

**Classes:**
- `BrutalTheme` (line 32) `class BrutalTheme(Enum)`
- `BrutalConfig` (line 39) `class BrutalConfig`
- `QuantumHologram` (line 84) `class QuantumHologram` - *Visualizador holográfico 3D de estados cuánticos*
- `QuantumNeuralTopology` (line 403) `class QuantumNeuralTopology` - *Visualiza la topología interna de las redes neuronales cuánticas*
- `QuantumSonification` (line 500) `class QuantumSonification` - *Convierte estados cuánticos en audio para percepción alternativa*
- `BrutalDashboard` (line 560) `class BrutalDashboard` - *Dashboard interactivo completo con todas las visualizaciones brutales*

**Functions:**
- `demo_brutal` (line 614) `def demo_brutal()` - *Demostración de visualización brutal*
- `create_synthetic_snapshots` (line 653) `def create_synthetic_snapshots()` - *Crea datos sintéticos para demostración*
- `colors` (line 52) `def colors(self)`
- `__init__` (line 87) `def __init__(self, config)`
- `create_amplitude_hologram` (line 92) `def create_amplitude_hologram(self, snapshots, backend_comparison)` - *Crea visualización holográfica de amplitudes en 3D con efecto de partículas*
- `_add_holographic_field` (line 148) `def _add_holographic_field(self, fig, snapshots, row, col)` - *Añade campo de amplitudes 3D con efecto de partículas flotantes*
- `_add_entropy_trails` (line 245) `def _add_entropy_trails(self, fig, snapshots, row, col)` - *Añade trazas de entropía con efecto de cola luminosa*
- `_add_bloch_sphere_holographic` (line 292) `def _add_bloch_sphere_holographic(self, fig, snapshot, backend_name, row, col)` - *Esfera de Bloch con efectos de holograma y partículas orbitales*
- `_interpolate_color` (line 389) `def _interpolate_color(self, color1, color2, factor)` - *Interpola entre dos colores hex*
- `__init__` (line 406) `def __init__(self, model)`
- `create_topology_map` (line 410) `def create_topology_map(self)` - *Crea mapa 3D de la arquitectura neuronal con activaciones en tiempo real*
- `_extract_layers` (line 477) `def _extract_layers(self, model)` - *Extrae información de capas del modelo PyTorch*
- `_generate_quantum_topology` (line 490) `def _generate_quantum_topology(self)` - *Genera topología representativa de backend cuántico*
- `__init__` (line 503) `def __init__(self, sample_rate)`
- `state_to_audio` (line 506) `def state_to_audio(self, snapshot, duration)` - *Convierte un estado cuántico en onda de audio
- Amplitudes controlan volumen
- Fases controlan paneo estéreo
- Probabilidades controlan frecuencia*
- `_adsr_envelope` (line 538) `def _adsr_envelope(self, length, intensity)` - *Genera envolvente ADSR proporcional a la intensidad del estado*
- `__init__` (line 563) `def __init__(self, config)`
- `generate_full_report` (line 570) `def generate_full_report(self, snapshots, backend_comparison)` - *Genera reporte completo con múltiples visualizaciones*
- `_save_audio` (line 605) `def _save_audio(self, audio, path)` - *Guarda audio como WAV*
- `hex_to_rgb` (line 391) `def hex_to_rgb(hex_color)`
- `rgb_to_hex` (line 395) `def rgb_to_hex(rgb)`

#### `quantum_computer.py`
**Path:** `quantum_computer.py`

**Classes:**
- `SimulatorConfig` (line 74) `class SimulatorConfig` - *Global configuration for the quantum computer simulator.

grid_size, hidden_dim, expansion_dim, num_spectral_layers must match
the values used when training the checkpoint files.*
- `SpectralLayer` (line 105) `class SpectralLayer` - *Spectral convolution in frequency domain.

Learns complex kernels that modulate Fourier coefficients.
Architecture is identical to the training scripts.*
- `HamiltonianBackboneNet` (line 144) `class HamiltonianBackboneNet` - *Hamiltonian backbone: single-channel field -> H|psi>.
Shared by all physics backends as the H operator.*
- `SchrodingerSpectralNet` (line 171) `class SchrodingerSpectralNet` - *Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction.*
- `DiracSpectralNet` (line 200) `class DiracSpectralNet` - *Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor.*
- `GammaMatrices` (line 233) `class GammaMatrices` - *Dirac gamma matrices in Dirac or Weyl representation.*
- `JointHilbertState` (line 279) `class JointHilbertState` - *Joint quantum state of n qubits in the full 2^n dimensional Hilbert space.

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
- `PotentialGenerator` (line 390) `class PotentialGenerator` - *Spatial potentials for eigenstate initialization.*
- `JointStateFactory` (line 474) `class JointStateFactory` - *Builds JointHilbertState tensors for common initial conditions.*
- `IPhysicsBackend` (line 511) `class IPhysicsBackend(ABC)` - *Abstract physics backend for spatial wavefunction evolution.*
- `HamiltonianBackend` (line 523) `class HamiltonianBackend(IPhysicsBackend)` - *Physics backend driven by the Hamiltonian neural network.

Performs first-order Schrodinger time evolution:
    psi(t+dt) = psi(t) - i*dt*H*psi(t)*
- `SchrodingerBackend` (line 591) `class SchrodingerBackend(IPhysicsBackend)` - *Physics backend driven by the Schrodinger network.

Uses the learned 2-channel spectral network for wavefunction propagation.
Falls back to HamiltonianBackend if checkpoint is unavailable.*
- `DiracBackend` (line 640) `class DiracBackend(IPhysicsBackend)` - *Physics backend driven by the Dirac network.

Expands each (2,G,G) amplitude to a 4-component spinor, propagates
via the Dirac network, then projects back to (2,G,G).*
- `IQuantumGate` (line 844) `class IQuantumGate(ABC)` - *Abstract quantum gate operating on the joint Hilbert space.*
- `HadamardGate` (line 863) `class HadamardGate(IQuantumGate)` - *H = [[1,1],[1,-1]] / sqrt(2).*
- `PauliXGate` (line 878) `class PauliXGate(IQuantumGate)` - *X = [[0,1],[1,0]].*
- `PauliYGate` (line 892) `class PauliYGate(IQuantumGate)` - *Y = [[0,-i],[i,0]].*
- `PauliZGate` (line 906) `class PauliZGate(IQuantumGate)` - *Z = [[1,0],[0,-1]].*
- `SGate` (line 920) `class SGate(IQuantumGate)` - *S = [[1,0],[0,i]].*
- `TGate` (line 934) `class TGate(IQuantumGate)` - *T = [[1,0],[0,e^{i*pi/4}]].*
- `RxGate` (line 949) `class RxGate(IQuantumGate)` - *Rx(theta) = exp(-i*theta/2 * X).*
- `RyGate` (line 965) `class RyGate(IQuantumGate)` - *Ry(theta) = exp(-i*theta/2 * Y).*
- `RzGate` (line 981) `class RzGate(IQuantumGate)` - *Rz(theta) = exp(-i*theta/2 * Z).*
- `CNOTGate` (line 998) `class CNOTGate(IQuantumGate)` - *CNOT: |ctrl tgt> -> |ctrl, ctrl XOR tgt>.

4x4 matrix (|00>,|01>,|10>,|11>):
    |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>*
- `CZGate` (line 1022) `class CZGate(IQuantumGate)` - *CZ: applies phase -1 to |11>.

4x4 matrix: diag(1, 1, 1, -1).*
- `SWAPGate` (line 1045) `class SWAPGate(IQuantumGate)` - *SWAP: exchanges two qubits.

4x4 matrix: |01>->|10>, |10>->|01>, others unchanged.*
- `ToffoliGate` (line 1068) `class ToffoliGate(IQuantumGate)` - *Toffoli (CCX): flips target iff both controls are |1>.

Exact amplitude permutation in the 8-element 3-qubit subspace.*
- `MCZGate` (line 1098) `class MCZGate(IQuantumGate)` - *Multi-Controlled Z gate: applies phase -1 to the single basis state
where ALL qubits in targets are |1>.

This is the exact oracle primitive needed by Grover's algorithm.
For n target qubits it marks the state |11...1> with a global phase of -1
and leaves all other basis states unchanged.

Implementation: iterate over all basis states k; if every bit
corresponding to a qubit in targets is set to 1, negate that amplitude
(multiply real and imaginary parts by -1).

targets: list of qubit indices that must all be |1> for the phase flip.*
- `EvolveGate` (line 1130) `class EvolveGate(IQuantumGate)` - *Free Hamiltonian evolution applied to every amplitude in the joint state.

Uses the active physics backend.
params: {"dt": float, "steps": int}*
- `CircuitInstruction` (line 1186) `class CircuitInstruction` - *Single gate instruction.*
- `QuantumCircuit` (line 1193) `class QuantumCircuit` - *Ordered sequence of quantum gate instructions.

Pure data structure: stores instructions, does not execute them.*
- `MeasurementResult` (line 1279) `class MeasurementResult` - *Non-destructive Born-rule measurement of the full register.

Contains the complete probability distribution over all 2^n basis states,
per-qubit marginals, and Bloch vectors. The state is never modified.*
- `QuantumComputer` (line 1333) `class QuantumComputer` - *Collapse-free quantum computer simulator with joint Hilbert space.

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
- `_make_logger` (line 58) `def _make_logger(name)` - *Create a module-level logger with a consistent formatter.*
- `_solve_eigenstate` (line 440) `def _solve_eigenstate(config, potential, n)` - *Solve the 1D marginal Hamiltonian and return the n-th eigenstate
as a normalized 2-channel (2, G, G) real tensor.*
- `_build_basis_amplitude` (line 461) `def _build_basis_amplitude(config, basis_idx)` - *Build the (2, G, G) spatial wavefunction for amplitude at basis index basis_idx.

Each computational basis state gets its own spatial eigenstate profile.
The excitation level is proportional to the popcount of the basis index.*
- `_single_qubit_unitary` (line 739) `def _single_qubit_unitary(state, qubit, u, backend)` - *Apply a 2x2 unitary u to qubit j in the joint Hilbert space.

For each pair of basis states (k0, k1) that differ only in bit j:
    alpha_{k0}' = u[0,0]*alpha_{k0} + u[0,1]*alpha_{k1}
    alpha_{k1}' = u[1,0]*alpha_{k0} + u[1,1]*alpha_{k1}

Complex scalar * (2,G,G) amplitude:
    (a+ib)(psi_r + i*psi_i) = (a*psi_r - b*psi_i) + i*(a*psi_i + b*psi_r)

This is exact, preserves unitarity, and correctly creates superpositions.*
- `_two_qubit_unitary` (line 789) `def _two_qubit_unitary(state, ctrl, tgt, u4)` - *Apply a 4x4 unitary in the {|00>,|01>,|10>,|11>} subspace of (ctrl, tgt).

For each group of 4 basis states sharing all bits except ctrl and tgt,
apply the 4x4 unitary to the amplitude quadruplet.

Ordering within the 4x4 block: |00>=0, |01>=1, |10>=2, |11>=3
(first bit = ctrl, second bit = tgt).

This correctly implements CNOT, CZ, SWAP and any 2-qubit gate.*
- `register_gate` (line 1179) `def register_gate(name, gate)` - *Register a custom gate without modifying existing code (Open/Closed Principle).*
- `_check` (line 1602) `def _check(label, condition)` - *Print PASS/FAIL for a single assertion. Returns True if passed.*
- `run_phase_tests` (line 1609) `def run_phase_tests(config)` - *Property-based test suite for quantum phase coherence and unitarity.

Each test has a known exact analytic answer derived from the unitary
algebra. Tests are designed to be sensitive to phase errors — circuits
where wrong relative phases produce DIFFERENT probabilities, not just
different phases that cancel out at measurement.

Returns the number of failed tests.*
- `_demo` (line 1836) `def _demo(config)` - *Run demo suite validating entanglement on all backends.*
- `__init__` (line 113) `def __init__(self, channels, grid_size)`
- `forward` (line 124) `def forward(self, x)` - *Apply spectral convolution via RFFT2.*
- `__init__` (line 150) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 159) `def forward(self, x)` - *Accepts (G,G), (1,G,G), or (B,1,G,G). Returns squeezed output.*
- `__init__` (line 176) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 188) `def forward(self, x)` - *(2,G,G) or (B,2,G,G) -> same shape.*
- `__init__` (line 205) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 217) `def forward(self, x)` - *(8,G,G) or (B,8,G,G) -> same shape.*
- `__init__` (line 236) `def __init__(self, representation, device)`
- `_init_matrices` (line 241) `def _init_matrices(self)`
- `to` (line 272) `def to(self, device)` - *Move all matrices to device.*
- `__init__` (line 305) `def __init__(self, amplitudes, n_qubits)`
- `normalize_` (line 317) `def normalize_(self)` - *In-place normalization: sum_k P(k) = 1.*
- `probabilities` (line 323) `def probabilities(self)` - *Return (2^n,) tensor of Born probabilities P(k) for each basis state.*
- `marginal_probability_one` (line 328) `def marginal_probability_one(self, qubit)` - *Marginal Born probability P(qubit_j = |1>).

Sums P(k) over all basis states k where bit j == 1.
Bit ordering: qubit 0 is the MSB of k.*
- `most_probable_basis_state` (line 343) `def most_probable_basis_state(self)` - *Return the index k with the highest probability.*
- `bloch_vector` (line 347) `def bloch_vector(self, qubit)` - *Compute the reduced Bloch vector for qubit j by partial trace.

rho_j[0,0] = P(qubit=0), rho_j[1,1] = P(qubit=1)
rho_j[0,1] = sum_{pairs} alpha_{k0}^* alpha_{k1} (off-diagonal coherence)
bx = 2 Re(rho_j[0,1]), by = -2 Im(rho_j[0,1]), bz = P(0) - P(1)*
- `clone` (line 385) `def clone(self)` - *Return a deep copy.*
- `__init__` (line 393) `def __init__(self, config)`
- `_grid` (line 397) `def _grid(self)`
- `harmonic` (line 402) `def harmonic(self)` - *V = k/2 * r^2.*
- `double_well` (line 410) `def double_well(self)` - *Double-well along x.*
- `coulomb` (line 417) `def coulomb(self)` - *Coulomb-like V ~ -1/r.*
- `periodic_lattice` (line 424) `def periodic_lattice(self)` - *Periodic cosine lattice.*
- `mixed` (line 429) `def mixed(self, seed)` - *Dirichlet-weighted mixture of all four potentials.*
- `__init__` (line 477) `def __init__(self, config)`
- `_empty` (line 480) `def _empty(self, n_qubits)`
- `all_zeros` (line 484) `def all_zeros(self, n_qubits)` - *Initialize register in |00...0>.*
- `basis_state` (line 492) `def basis_state(self, n_qubits, k)` - *Initialize register in computational basis state |k>.*
- `from_bitstring` (line 502) `def from_bitstring(self, bitstring)` - *Initialize in the basis state given by binary string.*
- `evolve_amplitude` (line 515) `def evolve_amplitude(self, amp, dt)` - *Evolve a single (2, G, G) wavefunction by dt under H.*
- `apply_phase` (line 519) `def apply_phase(self, amp, phase_angle)` - *Apply global phase e^{i*phi} to a (2, G, G) amplitude.*
- `__init__` (line 531) `def __init__(self, config)`
- `_load` (line 539) `def _load(self)`
- `_precompute_laplacian` (line 558) `def _precompute_laplacian(self)`
- `_apply_h` (line 565) `def _apply_h(self, field)`
- `evolve_amplitude` (line 573) `def evolve_amplitude(self, amp, dt)` - *dpsi/dt = -i H psi  =>  psi' = psi + dt * (-i H psi) = psi + dt*(H_i*r - H_r*i).*
- `apply_phase` (line 585) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 599) `def __init__(self, config, hamiltonian)`
- `_load` (line 606) `def _load(self)`
- `evolve_amplitude` (line 628) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 636) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 648) `def __init__(self, config, hamiltonian)`
- `_load` (line 657) `def _load(self)`
- `_precompute_dirac` (line 679) `def _precompute_dirac(self)`
- `_pack` (line 690) `def _pack(self, amp)`
- `_unpack` (line 701) `def _unpack(self, spinor)`
- `_analytical_dirac` (line 707) `def _analytical_dirac(self, spinor)`
- `evolve_amplitude` (line 722) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 735) `def apply_phase(self, amp, phase_angle)`
- `name` (line 849) `def name(self)` - *Gate identifier.*
- `apply` (line 853) `def apply(self, state, backend, targets, params)` - *Apply gate to joint state, return new joint state.*
- `name` (line 867) `def name(self)`
- `apply` (line 870) `def apply(self, state, backend, targets, params)`
- `name` (line 882) `def name(self)`
- `apply` (line 885) `def apply(self, state, backend, targets, params)`
- `name` (line 896) `def name(self)`
- `apply` (line 899) `def apply(self, state, backend, targets, params)`
- `name` (line 910) `def name(self)`
- `apply` (line 913) `def apply(self, state, backend, targets, params)`
- `name` (line 924) `def name(self)`
- `apply` (line 927) `def apply(self, state, backend, targets, params)`
- `name` (line 938) `def name(self)`
- `apply` (line 941) `def apply(self, state, backend, targets, params)`
- `name` (line 953) `def name(self)`
- `apply` (line 956) `def apply(self, state, backend, targets, params)`
- `name` (line 969) `def name(self)`
- `apply` (line 972) `def apply(self, state, backend, targets, params)`
- `name` (line 985) `def name(self)`
- `apply` (line 988) `def apply(self, state, backend, targets, params)`
- `name` (line 1007) `def name(self)`
- `apply` (line 1010) `def apply(self, state, backend, targets, params)`
- `name` (line 1030) `def name(self)`
- `apply` (line 1033) `def apply(self, state, backend, targets, params)`
- `name` (line 1053) `def name(self)`
- `apply` (line 1056) `def apply(self, state, backend, targets, params)`
- `name` (line 1076) `def name(self)`
- `apply` (line 1079) `def apply(self, state, backend, targets, params)`
- `name` (line 1115) `def name(self)`
- `apply` (line 1118) `def apply(self, state, backend, targets, params)`
- `name` (line 1139) `def name(self)`
- `apply` (line 1142) `def apply(self, state, backend, targets, params)`
- `__init__` (line 1200) `def __init__(self, n_qubits)`
- `h` (line 1206) `def h(self, q)`
- `x` (line 1209) `def x(self, q)`
- `y` (line 1212) `def y(self, q)`
- `z` (line 1215) `def z(self, q)`
- `s` (line 1218) `def s(self, q)`
- `t` (line 1221) `def t(self, q)`
- `rx` (line 1224) `def rx(self, q, theta)`
- `ry` (line 1227) `def ry(self, q, theta)`
- `rz` (line 1230) `def rz(self, q, theta)`
- `cnot` (line 1233) `def cnot(self, ctrl, tgt)`
- `cx` (line 1236) `def cx(self, ctrl, tgt)`
- `cz` (line 1239) `def cz(self, ctrl, tgt)`
- `swap` (line 1242) `def swap(self, a, b)`
- `toffoli` (line 1245) `def toffoli(self, c0, c1, tgt)`
- `ccx` (line 1248) `def ccx(self, c0, c1, tgt)`
- `evolve` (line 1251) `def evolve(self, qubits, dt, steps)`
- `barrier` (line 1254) `def barrier(self)`
- `_append` (line 1257) `def _append(self, gate_name, targets, params)`
- `depth` (line 1265) `def depth(self)`
- `__len__` (line 1268) `def __len__(self)`
- `__repr__` (line 1271) `def __repr__(self)`
- `probabilities` (line 1293) `def probabilities(self)` - *Alias: marginal P(|1>) per qubit index.*
- `most_probable_bitstring` (line 1297) `def most_probable_bitstring(self)` - *Return the bitstring with the highest probability.*
- `expectation_z` (line 1301) `def expectation_z(self, qubit)` - *<Z>_j = P(0) - P(1) in [-1, +1].*
- `entropy` (line 1305) `def entropy(self)` - *Shannon entropy of the full probability distribution in bits.*
- `__repr__` (line 1313) `def __repr__(self)`
- `__init__` (line 1361) `def __init__(self, config)`
- `_select_backend` (line 1375) `def _select_backend(self, name)`
- `_state_to_result` (line 1380) `def _state_to_result(self, state)`
- `run` (line 1388) `def run(self, circuit, backend, initial_states)` - *Execute a quantum circuit on the joint Hilbert space.

Args:
    circuit:        The QuantumCircuit to execute.
    backend:        Physics backend name.
    initial_states: Optional {qubit_idx: "0" or "1"}.

Returns:
    Non-destructive MeasurementResult with full distribution.*
- `run_with_state_snapshots` (line 1420) `def run_with_state_snapshots(self, circuit, backend, snapshot_after)` - *Execute circuit with non-destructive probability snapshots.

The state is never collapsed between snapshots.*
- `bell_state` (line 1449) `def bell_state(self, backend)` - *|Phi+> = (|00> + |11>) / sqrt(2).

Expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit.*
- `ghz_state` (line 1459) `def ghz_state(self, n_qubits, backend)` - *(|00...0> + |11...1>) / sqrt(2).

Expected: P(|00...0>)=P(|11...1>)=0.5, all others 0.*
- `quantum_fourier_transform` (line 1471) `def quantum_fourier_transform(self, n_qubits, backend)` - *QFT on |00...0>. Standard H + controlled-Rz decomposition.*
- `grover_oracle_search` (line 1481) `def grover_oracle_search(self, n_qubits, target_bitstring, backend, n_iterations)` - *Grover's search algorithm with correct phase oracle and diffusion operator.

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
- `variational_ansatz` (line 1559) `def variational_ansatz(self, n_qubits, n_layers, thetas, backend)` - *Hardware-efficient ansatz: Ry layers + CNOT chain. len(thetas)=n_qubits*n_layers.*
- `teleportation` (line 1572) `def teleportation(self, backend)` - *3-qubit teleportation protocol.

q0 prepared in Ry(pi/3). q2 should match q0's state after corrections.*
- `deutsch_jozsa` (line 1585) `def deutsch_jozsa(self, n_input_qubits, is_constant, backend)` - *Deutsch-Jozsa: constant -> all inputs |0>, balanced -> at least one |1>.*

#### `quantum_dash.py`
**Path:** `quantum_dash.py`

**Classes:**
- `ColorScheme` (line 106) `class ColorScheme(Enum)`
- `BrutalistConfig` (line 115) `class BrutalistConfig`
- `QuantumSnapshot` (line 239) `class QuantumSnapshot`
- `BackendComparison` (line 255) `class BackendComparison`
- `VisualizationOutput` (line 268) `class VisualizationOutput`
- `IVisualComponent` (line 275) `class IVisualComponent(ABC)`
- `ProbabilityVisualizer` (line 281) `class ProbabilityVisualizer(IVisualComponent)`
- `BlochSphereVisualizer` (line 340) `class BlochSphereVisualizer(IVisualComponent)`
- `PhaseSpaceVisualizer` (line 447) `class PhaseSpaceVisualizer(IVisualComponent)`
- `EntropyVisualizer` (line 509) `class EntropyVisualizer(IVisualComponent)`
- `BackendComparisonVisualizer` (line 583) `class BackendComparisonVisualizer(IVisualComponent)`
- `FidelityVisualizer` (line 633) `class FidelityVisualizer(IVisualComponent)`
- `QuantumStateAnalyzer` (line 674) `class QuantumStateAnalyzer`
- `StandardCircuits` (line 754) `class StandardCircuits`
- `CircuitExecutor` (line 806) `class CircuitExecutor`
- `FigureBuilder` (line 873) `class FigureBuilder`
- `QuantumVisualizer` (line 958) `class QuantumVisualizer`

**Functions:**
- `_make_logger` (line 91) `def _make_logger(name)`
- `main` (line 1199) `def main()`
- `colors` (line 168) `def colors(self)`
- `plotly_template` (line 234) `def plotly_template(self)`
- `render` (line 277) `def render(self, data, axes, config)`
- `render` (line 282) `def render(self, snapshot, axes, config)`
- `_render_empty` (line 318) `def _render_empty(self, axes, config)`
- `_generate_colors` (line 326) `def _generate_colors(self, probs, config)`
- `render` (line 341) `def render(self, snapshot, axes, config)`
- `_render_empty_sphere` (line 375) `def _render_empty_sphere(self, axes, config)`
- `_draw_sphere_wireframe` (line 381) `def _draw_sphere_wireframe(self, axes, config)`
- `_draw_axes` (line 402) `def _draw_axes(self, axes, config)`
- `_draw_bloch_vector` (line 417) `def _draw_bloch_vector(self, axes, bx, by, bz, color, qubit_idx, config)`
- `_draw_uncertainty_ring` (line 431) `def _draw_uncertainty_ring(self, axes, bx, by, bz, color, config)`
- `render` (line 448) `def render(self, snapshot, axes, config)`
- `_render_empty` (line 495) `def _render_empty(self, axes, config)`
- `render` (line 510) `def render(self, snapshots, axes, config)`
- `_render_empty` (line 556) `def _render_empty(self, axes, config)`
- `_interpolate_colors` (line 564) `def _interpolate_colors(self, color1, color2, n)`
- `_hex_to_rgb` (line 578) `def _hex_to_rgb(self, hex_color)`
- `render` (line 584) `def render(self, comparisons, axes, config)`
- `_render_empty` (line 624) `def _render_empty(self, axes, config)`
- `render` (line 634) `def render(self, comparisons, axes, config)`
- `_render_empty` (line 665) `def _render_empty(self, axes, config)`
- `__init__` (line 675) `def __init__(self, config)`
- `compute_probabilities` (line 678) `def compute_probabilities(self, state)`
- `compute_phases` (line 684) `def compute_phases(self, state)`
- `compute_entropy` (line 696) `def compute_entropy(self, probs)`
- `compute_bloch_vectors` (line 704) `def compute_bloch_vectors(self, state)`
- `create_snapshot` (line 713) `def create_snapshot(self, state, step, gate_name, backend_name)`
- `bell_state` (line 756) `def bell_state()`
- `ghz_state` (line 760) `def ghz_state(n_qubits)`
- `qft` (line 767) `def qft(n_qubits)`
- `grover_oracle` (line 782) `def grover_oracle(n_qubits, marked)`
- `grover_diffusion` (line 794) `def grover_diffusion(n_qubits)`
- `__init__` (line 807) `def __init__(self, qc, config)`
- `execute_sequence` (line 812) `def execute_sequence(self, gates, n_qubits, backend_name)`
- `compare_backends` (line 835) `def compare_backends(self, gates, n_qubits)`
- `__init__` (line 874) `def __init__(self, config)`
- `build_full_figure` (line 883) `def build_full_figure(self, snapshots, comparisons)`
- `build_summary_figure` (line 923) `def build_summary_figure(self, snapshots, comparisons)`
- `__init__` (line 959) `def __init__(self, config)`
- `_initialize` (line 966) `def _initialize(self)`
- `_init_quantum_computer` (line 976) `def _init_quantum_computer(self)`
- `visualize_bell_state` (line 1010) `def visualize_bell_state(self)`
- `visualize_ghz_state` (line 1016) `def visualize_ghz_state(self, n_qubits)`
- `visualize_qft` (line 1022) `def visualize_qft(self, n_qubits)`
- `visualize_grover` (line 1028) `def visualize_grover(self, n_qubits, marked_state)`
- `_execute_and_visualize` (line 1043) `def _execute_and_visualize(self, gates, n_qubits, name)`
- `_create_synthetic_snapshots` (line 1086) `def _create_synthetic_snapshots(self, n_qubits, gates)`
- `_create_synthetic_comparisons` (line 1129) `def _create_synthetic_comparisons(self, n_qubits, gates)`
- `_save_data` (line 1153) `def _save_data(self, path, snapshots, comparisons)`
- `run_all` (line 1174) `def run_all(self)`
- `_print_summary` (line 1189) `def _print_summary(self, results)`

#### `quantum_framework_core.py`
**Path:** `quantum_framework_core.py`

**Classes:**
- `HilbertPhase` (line 62) `class HilbertPhase(Enum)` - *Phase classification for Hilbert space compression.*
- `FrameworkConfig` (line 72) `class FrameworkConfig` - *Unified configuration for the quantum simulation framework.
Loads from TOML file with fallback to sensible defaults.*
- `AtomData` (line 218) `class AtomData` - *Atomic data structure.*
- `MoleculeData` (line 230) `class MoleculeData` - *Molecular data structure.*
- `OrbitalData` (line 250) `class OrbitalData` - *Atomic orbital data structure.*
- `ConfigLoader` (line 259) `class ConfigLoader` - *Configuration loader that parses TOML files and provides
access to atoms, molecules, and orbitals data.*
- `ITensorNetwork` (line 435) `class ITensorNetwork(ABC)` - *Abstract interface for tensor network quantum states.*
- `MPSCore` (line 480) `class MPSCore` - *Matrix Product State core tensor A^{[k]}_{i_k} with bond indices.

Shape: (chi_left, d, chi_right) where d=2 for qubits.
Memory per core: O(chi^2 * d) = O(chi^2)*
- `MPSState` (line 554) `class MPSState(ITensorNetwork)` - *Matrix Product State representation of n-qubit quantum state.

|psi> = sum_{i_1...i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} ... A^{[n]}_{i_n} |i_1...i_n>

Memory: O(n * chi^2 * d) vs O(d^n) for full statevector.

Example scaling:
    n=30, chi=16: ~30KB vs 8GB for statevector
    n=33, chi=16: ~33KB vs 64GB for statevector*
- `VacuumCore` (line 916) `class VacuumCore` - *Vacuum Core architecture for topological protection.

Projects irrelevant Hilbert subspace to zero, achieving
high sparsity (target 99.99%) while preserving quantum information.*
- `TopologicalProtector` (line 988) `class TopologicalProtector` - *Provides topological protection for quantum states.

Monitors:
    - Winding numbers
    - Berry phases
    - Edge state preservation*
- `SpectralLayer` (line 1040) `class SpectralLayer` - *Spectral convolution layer in frequency domain.*
- `HamiltonianBackboneNet` (line 1076) `class HamiltonianBackboneNet` - *Hamiltonian backbone network for spectral operations.*
- `SchrodingerSpectralNet` (line 1102) `class SchrodingerSpectralNet` - *Schrodinger network for wavefunction evolution.*
- `DiracSpectralNet` (line 1133) `class DiracSpectralNet` - *Dirac network for relativistic spinor evolution.*
- `GammaMatrices` (line 1164) `class GammaMatrices` - *Dirac gamma matrices in standard or Weyl representation.*
- `IPhysicsBackend` (line 1215) `class IPhysicsBackend(ABC)` - *Abstract interface for physics backends.*
- `HamiltonianBackend` (line 1229) `class HamiltonianBackend(IPhysicsBackend)` - *Hamiltonian backend using neural network for spectral operations.

Performs first-order Schrodinger time evolution:
    psi(t+dt) = psi(t) - i*dt*H*psi(t)*
- `SchrodingerBackend` (line 1305) `class SchrodingerBackend(IPhysicsBackend)` - *Schrodinger backend using learned 2-channel spectral network.

Falls back to HamiltonianBackend if checkpoint unavailable.*
- `DiracBackend` (line 1358) `class DiracBackend(IPhysicsBackend)` - *Dirac backend for relativistic spinor evolution.

Expands (2,G,G) amplitude to 4-component spinor, propagates,
then projects back.*
- `IQuantumGate` (line 1476) `class IQuantumGate(ABC)` - *Abstract interface for quantum gates.*
- `HadamardGate` (line 1496) `class HadamardGate(IQuantumGate)` - *Hadamard gate: H = [[1,1],[1,-1]] / sqrt(2).*
- `PauliXGate` (line 1511) `class PauliXGate(IQuantumGate)` - *Pauli-X gate: X = [[0,1],[1,0]].*
- `PauliYGate` (line 1525) `class PauliYGate(IQuantumGate)` - *Pauli-Y gate: Y = [[0,-i],[i,0]].*
- `PauliZGate` (line 1539) `class PauliZGate(IQuantumGate)` - *Pauli-Z gate: Z = [[1,0],[0,-1]].*
- `SGate` (line 1553) `class SGate(IQuantumGate)` - *S gate: S = [[1,0],[0,i]].*
- `TGate` (line 1567) `class TGate(IQuantumGate)` - *T gate: T = [[1,0],[0,e^{i*pi/4}]].*
- `RxGate` (line 1582) `class RxGate(IQuantumGate)` - *Rotation-X gate: Rx(theta) = exp(-i*theta/2 * X).*
- `RyGate` (line 1598) `class RyGate(IQuantumGate)` - *Rotation-Y gate: Ry(theta) = exp(-i*theta/2 * Y).*
- `RzGate` (line 1614) `class RzGate(IQuantumGate)` - *Rotation-Z gate: Rz(theta) = exp(-i*theta/2 * Z).*
- `CRzGate` (line 1631) `class CRzGate(IQuantumGate)` - *Controlled-Rz gate: applies Rz to target if control is |1>.*
- `CNOTGate` (line 1656) `class CNOTGate(IQuantumGate)` - *CNOT gate: flips target if control is |1>.*
- `CZGate` (line 1676) `class CZGate(IQuantumGate)` - *Controlled-Z gate: applies phase -1 to |11>.*
- `SWAPGate` (line 1696) `class SWAPGate(IQuantumGate)` - *SWAP gate: exchanges two qubits.*
- `CircuitInstruction` (line 1734) `class CircuitInstruction` - *Single instruction in a quantum circuit.*
- `QuantumCircuit` (line 1741) `class QuantumCircuit` - *Quantum circuit builder for MPS states.*
- `MPSQuantumComputer` (line 1816) `class MPSQuantumComputer` - *Main quantum computer using MPS representation.

Provides:
    - State preparation
    - Circuit execution
    - Backend selection
    - Memory-efficient simulation for up to 33+ qubits*

**Functions:**
- `_make_logger` (line 46) `def _make_logger(name, level)` - *Create a configured logger instance.*
- `run_scaling_benchmark` (line 2045) `def run_scaling_benchmark(config, max_qubits)` - *Run scaling benchmark to demonstrate MPS memory efficiency.

Returns:
    Dictionary with qubit counts, memory usage, compression ratios.*
- `run_grover_search` (line 2098) `def run_grover_search(qc, n_qubits, marked_states)` - *Run Grover's search algorithm and return results.

Implements the algorithm at the statevector level for correctness
(exact oracle via direct phase flip), then reads off probabilities.

Args:
    qc: MPSQuantumComputer instance (unused for computation, kept for API compatibility)
    n_qubits: number of qubits
    marked_states: list of integers representing marked computational basis states

Returns:
    dict with keys: probability, marked_states (as bit strings), speedup, iterations*
- `__post_init__` (line 139) `def __post_init__(self)` - *Initialize random seeds after configuration.*
- `from_toml` (line 147) `def from_toml(cls, toml_path)` - *Load configuration from TOML file.*
- `__init__` (line 265) `def __init__(self, config_path)`
- `_find_config` (line 274) `def _find_config(self)` - *Find configuration file in standard locations.*
- `_load` (line 286) `def _load(self)` - *Load configuration from TOML file.*
- `_load_defaults` (line 300) `def _load_defaults(self)` - *Load default configuration values.*
- `_parse_atoms` (line 318) `def _parse_atoms(self)` - *Parse atoms from configuration data.*
- `_parse_molecules` (line 332) `def _parse_molecules(self)` - *Parse molecules from configuration data.*
- `_parse_orbitals` (line 352) `def _parse_orbitals(self)` - *Parse orbitals from configuration data.*
- `_parse_experiments` (line 364) `def _parse_experiments(self)` - *Parse experiments from configuration data.*
- `get_atom` (line 375) `def get_atom(self, symbol)` - *Get atom data by symbol (case-insensitive).*
- `get_molecule` (line 384) `def get_molecule(self, name)` - *Get molecule data by name (case-insensitive).*
- `get_orbital` (line 393) `def get_orbital(self, name)` - *Get orbital data by name (case-insensitive).*
- `get_experiment` (line 402) `def get_experiment(self, name)` - *Get experiment data by name.*
- `atoms` (line 407) `def atoms(self)` - *Return all atoms.*
- `molecules` (line 412) `def molecules(self)` - *Return all molecules.*
- `orbitals` (line 417) `def orbitals(self)` - *Return all orbitals.*
- `experiments` (line 422) `def experiments(self)` - *Return all experiments.*
- `get_molecules_by_qubits` (line 426) `def get_molecules_by_qubits(self, max_qubits)` - *Get molecules that fit within qubit budget.*
- `get_atoms_by_qubits` (line 430) `def get_atoms_by_qubits(self, max_qubits)` - *Get atoms that fit within qubit budget.*
- `n_qubits` (line 440) `def n_qubits(self)` - *Return number of qubits.*
- `amplitude` (line 445) `def amplitude(self, basis_index)` - *Compute amplitude for a computational basis state.*
- `apply_single_qubit_gate` (line 450) `def apply_single_qubit_gate(self, qubit, gate)` - *Apply single-qubit gate in-place.*
- `apply_two_qubit_gate` (line 455) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)` - *Apply two-qubit gate in-place.*
- `norm` (line 460) `def norm(self)` - *Compute state norm.*
- `probabilities` (line 465) `def probabilities(self)` - *Compute measurement probabilities.*
- `entropy` (line 470) `def entropy(self)` - *Compute von Neumann entropy.*
- `memory_bytes` (line 475) `def memory_bytes(self)` - *Return memory usage in bytes.*
- `__init__` (line 488) `def __init__(self, chi_left, chi_right, d, device, dtype)`
- `_initialize` (line 504) `def _initialize(self)` - *Initialize core tensor for |0> product state (exact).*
- `tensor` (line 517) `def tensor(self)` - *Return the core tensor.*
- `tensor` (line 524) `def tensor(self, value)` - *Set the core tensor, preserving complex dtype when needed.*
- `left_canonicalize` (line 533) `def left_canonicalize(self)` - *Bring core to left-canonical form, return singular values.*
- `right_canonicalize` (line 543) `def right_canonicalize(self)` - *Bring core to right-canonical form, return singular values.*
- `__init__` (line 567) `def __init__(self, n_qubits, config)`
- `_initialize` (line 576) `def _initialize(self)` - *Initialize MPS with product state |00...0>.*
- `n_qubits` (line 600) `def n_qubits(self)`
- `_bond_dimension` (line 603) `def _bond_dimension(self, site)` - *Compute bond dimension at given site.*
- `amplitude` (line 610) `def amplitude(self, basis_index)` - *Compute amplitude for computational basis state.*
- `apply_single_qubit_gate` (line 631) `def apply_single_qubit_gate(self, qubit, gate)` - *Apply single-qubit gate in-place.

Works in complex128 so that gates with imaginary entries (Y, S, T, Rz…)
are handled correctly.  The core tensor is promoted to complex128 when
the result has a non-negligible imaginary component; otherwise it is
kept as the config dtype (typically float64).*
- `apply_two_qubit_gate` (line 655) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)` - *Apply two-qubit gate in-place.*
- `_swap_qubits_in_gate` (line 674) `def _swap_qubits_in_gate(self, gate)` - *Swap qubit ordering in two-qubit gate.*
- `_apply_adjacent_gate` (line 684) `def _apply_adjacent_gate(self, qubit, gate)` - *Apply gate to adjacent qubit pair.*
- `_apply_nonadjacent_gate` (line 747) `def _apply_nonadjacent_gate(self, qubit_a, qubit_b, gate)` - *Apply gate to non-adjacent qubit pair using SWAP network.*
- `norm` (line 762) `def norm(self)` - *Compute state norm.*
- `_canonicalize` (line 770) `def _canonicalize(self)` - *Bring MPS to canonical form.*
- `probabilities` (line 778) `def probabilities(self)` - *Compute measurement probabilities.*
- `entropy` (line 795) `def entropy(self)` - *Compute maximum entanglement entropy across all cuts.

For n=1 qubits, returns Shannon entropy of the probability distribution.
For n>1, returns the maximum entanglement entropy across all bipartite cuts.*
- `memory_bytes` (line 822) `def memory_bytes(self)` - *Return memory usage in bytes.*
- `entanglement_entropy` (line 829) `def entanglement_entropy(self, cut)` - *Compute entanglement entropy at given cut between qubits cut-1 and cut.
Uses Schmidt decomposition from the MPS bond.*
- `to_statevector` (line 874) `def to_statevector(self)` - *Convert MPS to full statevector (only for small systems).*
- `most_probable_bitstring` (line 890) `def most_probable_bitstring(self)` - *Return most probable basis state as bitstring.*
- `clone` (line 896) `def clone(self)` - *Return a deep copy.*
- `__init__` (line 924) `def __init__(self, n_qubits, config)`
- `_initialize` (line 935) `def _initialize(self)` - *Initialize vacuum core with ground state.*
- `_compute_berry_phases` (line 941) `def _compute_berry_phases(self)` - *Compute Berry phases between active states.*
- `add_active_state` (line 948) `def add_active_state(self, basis_index, winding_number)` - *Add a basis state to the active subspace.*
- `_compute_winding_number` (line 962) `def _compute_winding_number(self, basis_index)` - *Compute winding number for a basis state.*
- `is_topologically_protected` (line 968) `def is_topologically_protected(self, basis_index)` - *Check if state is topologically protected.*
- `sparsity` (line 973) `def sparsity(self)` - *Compute vacuum sparsity.*
- `project_to_active` (line 979) `def project_to_active(self, state)` - *Project state onto active subspace.*
- `__init__` (line 998) `def __init__(self, config)`
- `compute_winding_number` (line 1003) `def compute_winding_number(self, state, qubit)` - *Compute winding number for a qubit.*
- `compute_berry_phase` (line 1015) `def compute_berry_phase(self, state, qubit_a, qubit_b)` - *Compute Berry phase between two qubits.*
- `is_protected` (line 1031) `def is_protected(self, state, vacuum_core)` - *Check if state is topologically protected.*
- `__init__` (line 1043) `def __init__(self, channels, grid_size)`
- `forward` (line 1053) `def forward(self, x)` - *Apply spectral convolution via RFFT2.*
- `__init__` (line 1079) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 1088) `def forward(self, x)` - *Apply Hamiltonian backbone network.*
- `__init__` (line 1105) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 1119) `def forward(self, x)` - *Apply Schrodinger evolution network.*
- `__init__` (line 1136) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 1150) `def forward(self, x)` - *Apply Dirac evolution network.*
- `__init__` (line 1167) `def __init__(self, representation, device)`
- `_init_matrices` (line 1172) `def _init_matrices(self)` - *Initialize gamma matrices.*
- `to` (line 1208) `def to(self, device)` - *Move all matrices to device.*
- `evolve_amplitude` (line 1219) `def evolve_amplitude(self, amp, dt)` - *Evolve a single amplitude by time dt.*
- `apply_phase` (line 1224) `def apply_phase(self, amp, phase_angle)` - *Apply global phase to amplitude.*
- `__init__` (line 1237) `def __init__(self, config)`
- `_load` (line 1245) `def _load(self)` - *Load model from checkpoint.*
- `_precompute_laplacian` (line 1266) `def _precompute_laplacian(self)` - *Precompute Laplacian kernel for kinetic energy.*
- `_apply_h` (line 1274) `def _apply_h(self, field)` - *Apply Hamiltonian operator to field.*
- `evolve_amplitude` (line 1284) `def evolve_amplitude(self, amp, dt)` - *Evolve amplitude by time dt.*
- `apply_phase` (line 1298) `def apply_phase(self, amp, phase_angle)` - *Apply global phase.*
- `__init__` (line 1312) `def __init__(self, config, hamiltonian)`
- `_load` (line 1319) `def _load(self)` - *Load model from checkpoint.*
- `evolve_amplitude` (line 1342) `def evolve_amplitude(self, amp, dt)` - *Evolve amplitude by time dt.*
- `apply_phase` (line 1353) `def apply_phase(self, amp, phase_angle)` - *Apply global phase.*
- `__init__` (line 1366) `def __init__(self, config, hamiltonian)`
- `_load` (line 1375) `def _load(self)` - *Load model from checkpoint.*
- `_precompute_dirac` (line 1398) `def _precompute_dirac(self)` - *Precompute momentum grids for Dirac operator.*
- `_pack` (line 1407) `def _pack(self, amp)` - *Pack 2-channel amplitude to 4-component spinor.*
- `_unpack` (line 1421) `def _unpack(self, spinor)` - *Unpack 4-component spinor to 2-channel amplitude.*
- `_analytical_dirac` (line 1428) `def _analytical_dirac(self, spinor)` - *Apply analytical Dirac Hamiltonian to spinor.*
- `evolve_amplitude` (line 1447) `def evolve_amplitude(self, amp, dt)` - *Evolve amplitude by time dt using Dirac equation.*
- `apply_phase` (line 1463) `def apply_phase(self, amp, phase_angle)` - *Apply global phase.*
- `evolve_spinor` (line 1467) `def evolve_spinor(self, spinor, dt)` - *Evolve full 4-component spinor by time dt.*
- `name` (line 1481) `def name(self)` - *Return gate name.*
- `apply` (line 1486) `def apply(self, state, targets, params)` - *Apply gate to state and return new state.*
- `name` (line 1500) `def name(self)`
- `apply` (line 1503) `def apply(self, state, targets, params)`
- `name` (line 1515) `def name(self)`
- `apply` (line 1518) `def apply(self, state, targets, params)`
- `name` (line 1529) `def name(self)`
- `apply` (line 1532) `def apply(self, state, targets, params)`
- `name` (line 1543) `def name(self)`
- `apply` (line 1546) `def apply(self, state, targets, params)`
- `name` (line 1557) `def name(self)`
- `apply` (line 1560) `def apply(self, state, targets, params)`
- `name` (line 1571) `def name(self)`
- `apply` (line 1574) `def apply(self, state, targets, params)`
- `name` (line 1586) `def name(self)`
- `apply` (line 1589) `def apply(self, state, targets, params)`
- `name` (line 1602) `def name(self)`
- `apply` (line 1605) `def apply(self, state, targets, params)`
- `name` (line 1618) `def name(self)`
- `apply` (line 1621) `def apply(self, state, targets, params)`
- `name` (line 1635) `def name(self)`
- `apply` (line 1638) `def apply(self, state, targets, params)`
- `name` (line 1660) `def name(self)`
- `apply` (line 1663) `def apply(self, state, targets, params)`
- `name` (line 1680) `def name(self)`
- `apply` (line 1683) `def apply(self, state, targets, params)`
- `name` (line 1700) `def name(self)`
- `apply` (line 1703) `def apply(self, state, targets, params)`
- `__init__` (line 1744) `def __init__(self, n_qubits)`
- `_append` (line 1748) `def _append(self, gate_name, targets, params)` - *Append an instruction to the circuit.*
- `h` (line 1759) `def h(self, qubit)`
- `x` (line 1762) `def x(self, qubit)`
- `y` (line 1765) `def y(self, qubit)`
- `z` (line 1768) `def z(self, qubit)`
- `s` (line 1771) `def s(self, qubit)`
- `t` (line 1774) `def t(self, qubit)`
- `rx` (line 1777) `def rx(self, qubit, theta)`
- `ry` (line 1780) `def ry(self, qubit, theta)`
- `rz` (line 1783) `def rz(self, qubit, theta)`
- `crz` (line 1786) `def crz(self, control, target, theta)`
- `cnot` (line 1789) `def cnot(self, control, target)`
- `cz` (line 1792) `def cz(self, control, target)`
- `swap` (line 1795) `def swap(self, qubit1, qubit2)`
- `__len__` (line 1798) `def __len__(self)` - *Return number of instructions.*
- `__bool__` (line 1802) `def __bool__(self)` - *True if circuit has instructions.*
- `run` (line 1806) `def run(self, state)` - *Execute circuit on state.*
- `__init__` (line 1827) `def __init__(self, config)`
- `create_circuit` (line 1843) `def create_circuit(self, n_qubits)` - *Create a new quantum circuit.*
- `create_state` (line 1851) `def create_state(self, n_qubits)` - *Create initial state |00...0>.*
- `bell_state` (line 1865) `def bell_state(self, n_qubits)` - *Prepare Bell state |Phi+> = (|00> + |11>) / sqrt(2).*
- `ghz_state` (line 1873) `def ghz_state(self, n_qubits)` - *Prepare GHZ state (|00...0> + |11...1>) / sqrt(2).*
- `w_state` (line 1882) `def w_state(self, n_qubits)` - *Prepare W state: |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n

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
- `_build_w_state_direct` (line 1905) `def _build_w_state_direct(self, n_qubits, max_bond)` - *Build W state using direct statevector-to-MPS conversion via successive SVD.

This guarantees exact representation (up to numerical precision).*
- `run_circuit` (line 1985) `def run_circuit(self, circuit, initial_state)` - *Execute circuit on state.*
- `get_backend` (line 1995) `def get_backend(self, name)` - *Get physics backend by name.*
- `memory_usage` (line 1999) `def memory_usage(self, state)` - *Compute memory usage for a state.*
- `compression_ratio` (line 2012) `def compression_ratio(self, state)` - *Compute compression ratio vs full statevector.*
- `detect_phase` (line 2019) `def detect_phase(self, state)` - *Detect Hilbert space phase from state properties.*
- `_compute_average_bond_dimension` (line 2037) `def _compute_average_bond_dimension(self, state)` - *Compute average bond dimension across MPS cores.*

#### `quantum_framework_main.py`
**Path:** `quantum_framework_main.py`

**Functions:**
- `setup_logging` (line 52) `def setup_logging(verbose)` - *Configure logging level based on verbosity.*
- `run_benchmark` (line 61) `def run_benchmark(args, config)` - *Run scaling benchmark.*
- `run_experiment` (line 101) `def run_experiment(args, config, config_loader)` - *Run a specific experiment by name.*
- `run_molecular_simulation` (line 173) `def run_molecular_simulation(args, config, config_loader)` - *Run molecular simulation.*
- `run_orbital_visualization` (line 210) `def run_orbital_visualization(args, config, config_loader)` - *Run orbital visualization.*
- `print_info` (line 242) `def print_info(config, config_loader)` - *Print framework information.*
- `main` (line 292) `def main()` - *Main entry point.*

#### `quantum_framework_menu.py`
**Path:** `quantum_framework_menu.py`

**Classes:**
- `MenuSystem` (line 64) `class MenuSystem` - *Interactive menu system for the quantum simulation framework.

Provides structured access to all framework capabilities through
a hierarchical menu system with real-time feedback.*

**Functions:**
- `run_interactive_menu` (line 2357) `def run_interactive_menu(config, config_loader)` - *Run the interactive menu system.*
- `run_all_experiments` (line 2363) `def run_all_experiments(config, config_loader)` - *Run ALL experiments automatically for debugging.
This function executes all available experiments without user interaction.
Uses CORRECT physics formulas verified against experimental data.*
- `__init__` (line 72) `def __init__(self, config, config_loader)`
- `clear_screen` (line 79) `def clear_screen(self)` - *Clear the terminal screen.*
- `print_header` (line 83) `def print_header(self, title)` - *Print formatted header.*
- `print_menu` (line 90) `def print_menu(self, title, options)` - *Print formatted menu with options.*
- `get_input` (line 97) `def get_input(self, prompt)` - *Get user input with history tracking.*
- `pause` (line 106) `def pause(self, message)` - *Wait for user to press Enter.*
- `run` (line 113) `def run(self)` - *Run the main menu loop.*
- `_show_main_menu` (line 118) `def _show_main_menu(self)` - *Display main menu.*
- `_show_circuit_menu` (line 163) `def _show_circuit_menu(self)` - *Display quantum circuits menu.*
- `_custom_circuit` (line 197) `def _custom_circuit(self)` - *Create and run a custom circuit.*
- `_bell_state_demo` (line 279) `def _bell_state_demo(self)` - *Demonstrate Bell state preparation.*
- `_ghz_state_demo` (line 299) `def _ghz_state_demo(self)` - *Demonstrate GHZ state preparation.*
- `_w_state_demo` (line 328) `def _w_state_demo(self)` - *Demonstrate W state preparation.*
- `_single_qubit_gates_demo` (line 356) `def _single_qubit_gates_demo(self)` - *Demonstrate single qubit gates.*
- `_two_qubit_gates_demo` (line 397) `def _two_qubit_gates_demo(self)` - *Demonstrate two qubit gates.*
- `_show_entanglement_menu` (line 431) `def _show_entanglement_menu(self)` - *Display entanglement experiments menu.*
- `_bell_entropy_experiment` (line 459) `def _bell_entropy_experiment(self)` - *Measure entropy of Bell states.*
- `_ghz_scaling_experiment` (line 472) `def _ghz_scaling_experiment(self)` - *Study GHZ entanglement scaling with qubit count.*
- `_entropy_by_cut_experiment` (line 497) `def _entropy_by_cut_experiment(self)` - *Measure entanglement entropy at different cuts.*
- `_entropy_heatmap_experiment` (line 522) `def _entropy_heatmap_experiment(self)` - *Generate entropy heatmap for different states.*
- `_show_molecular_menu` (line 583) `def _show_molecular_menu(self)` - *Display molecular simulations menu.*
- `_list_molecules` (line 617) `def _list_molecules(self)` - *List all available molecules.*
- `_molecule_info` (line 636) `def _molecule_info(self)` - *Show detailed molecule information.*
- `_vqe_ground_state` (line 667) `def _vqe_ground_state(self)` - *Run VQE for molecular ground state.*
- `_energy_landscape` (line 751) `def _energy_landscape(self)` - *Plot molecular energy landscape.*
- `_bond_dissociation` (line 808) `def _bond_dissociation(self)` - *Simulate bond dissociation curve.*
- `_show_orbital_menu` (line 868) `def _show_orbital_menu(self)` - *Display orbital visualization menu.*
- `_list_orbitals` (line 899) `def _list_orbitals(self)` - *List all available orbitals.*
- `_visualize_orbital` (line 915) `def _visualize_orbital(self)` - *Visualize a single orbital.*
- `_generate_orbital_plot` (line 947) `def _generate_orbital_plot(self, orb, num_samples)` - *Generate orbital visualization plot.*
- `_compare_orbitals` (line 1064) `def _compare_orbitals(self)` - *Compare multiple orbitals.*
- `_radial_wavefunction` (line 1146) `def _radial_wavefunction(self)` - *Plot radial wavefunction.*
- `_angular_wavefunction` (line 1202) `def _angular_wavefunction(self)` - *Plot angular wavefunction.*
- `_show_relativistic_menu` (line 1264) `def _show_relativistic_menu(self)` - *Display relativistic physics menu.*
- `_dirac_energy_levels` (line 1292) `def _dirac_energy_levels(self)` - *Calculate Dirac energy levels.*
- `_dirac_energy` (line 1321) `def _dirac_energy(self, n, kappa, alpha, c)` - *Calculate Dirac energy level.*
- `_fine_structure` (line 1329) `def _fine_structure(self)` - *Calculate fine structure corrections.*
- `_zitterbewegung` (line 1347) `def _zitterbewegung(self)` - *Simulate Zitterbewegung.*
- `_spin_orbit` (line 1363) `def _spin_orbit(self)` - *Calculate spin-orbit coupling.*
- `_show_qed_menu` (line 1378) `def _show_qed_menu(self)` - *Display QED effects menu.*
- `_lamb_shift` (line 1406) `def _lamb_shift(self)` - *Calculate Lamb shift.*
- `_anomalous_moment` (line 1426) `def _anomalous_moment(self)` - *Calculate anomalous magnetic moment.*
- `_vacuum_polarization` (line 1453) `def _vacuum_polarization(self)` - *Calculate vacuum polarization effects.*
- `_full_qed` (line 1468) `def _full_qed(self)` - *Show full QED corrections.*
- `_show_algorithms_menu` (line 1482) `def _show_algorithms_menu(self)` - *Display quantum algorithms menu.*
- `_grover_search` (line 1510) `def _grover_search(self)` - *Demonstrate Grover's search algorithm.*
- `_apply_grover_iteration` (line 1558) `def _apply_grover_iteration(self, state, marked, n_qubits)` - *Apply one Grover iteration: oracle then diffusion.*
- `_qft_demo` (line 1607) `def _qft_demo(self)` - *Demonstrate Quantum Fourier Transform.*
- `_phase_estimation` (line 1669) `def _phase_estimation(self)` - *Demonstrate phase estimation.*
- `_vqe_demo` (line 1746) `def _vqe_demo(self)` - *Demonstrate VQE.*
- `_show_benchmark_menu` (line 1816) `def _show_benchmark_menu(self)` - *Display benchmarks menu.*
- `_mps_scaling_benchmark` (line 1844) `def _mps_scaling_benchmark(self)` - *Run MPS scaling benchmark.*
- `_gate_performance` (line 1872) `def _gate_performance(self)` - *Benchmark gate performance.*
- `_memory_comparison` (line 1907) `def _memory_comparison(self)` - *Compare memory usage.*
- `_entanglement_scaling` (line 1927) `def _entanglement_scaling(self)` - *Study entanglement scaling.*
- `_show_config_menu` (line 1953) `def _show_config_menu(self)` - *Display configuration menu.*
- `_view_config` (line 1984) `def _view_config(self)` - *View current configuration.*
- `_list_atoms` (line 1999) `def _list_atoms(self)` - *List available atoms.*
- `_list_molecules_config` (line 2016) `def _list_molecules_config(self)` - *List available molecules.*
- `_list_experiments` (line 2020) `def _list_experiments(self)` - *List available experiments.*
- `_system_info` (line 2036) `def _system_info(self)` - *Show system information.*
- `_show_particle_physics_menu` (line 2055) `def _show_particle_physics_menu(self)` - *Display particle physics menu (Higgs analysis).*
- `_run_higgs_analysis` (line 2077) `def _run_higgs_analysis(self)` - *Run the Higgs boson 4-lepton quantum analysis.*
- `_higgs_about` (line 2101) `def _higgs_about(self)` - *Show information about the Higgs analysis.*
- `_show_visualization_menu` (line 2129) `def _show_visualization_menu(self)` - *Display quantum visualization menu.*
- `_run_quantum_dash` (line 2154) `def _run_quantum_dash(self)` - *Run brutalist quantum state visualizer.*
- `_run_quantum_3dview` (line 2201) `def _run_quantum_3dview(self)` - *Run 3D holographic quantum dashboard.*
- `_run_quantum_visualizer` (line 2247) `def _run_quantum_visualizer(self)` - *Run the standard quantum state visualizer.*
- `_run_polarizability_vqe` (line 2285) `def _run_polarizability_vqe(self)` - *Run H2 polarizability / Stark effect VQE from app.py.*
- `_show_help` (line 2316) `def _show_help(self)` - *Show help information.*
- `_quit` (line 2350) `def _quit(self)` - *Exit the menu system.*
- `test_header` (line 2379) `def test_header(name)`
- `radial_wf` (line 951) `def radial_wf(n, l, r)`
- `spherical_harm_real` (line 959) `def spherical_harm_real(l, m, theta, phi)`
- `compute_energy` (line 1764) `def compute_energy(state, n_qubits)`

#### `quantum_framework_molecular.py`
**Path:** `quantum_framework_molecular.py`

**Classes:**
- `MoleculeData` (line 72) `class MoleculeData`
- `MoleculeBuilder` (line 85) `class MoleculeBuilder` - *Build molecule data for quantum chemistry calculations.*
- `ExactJWEnergy` (line 156) `class ExactJWEnergy` - *Exact Jordan-Wigner energy evaluator.

CORRECTED: Uses verified Hamiltonian coefficients from standard references.*
- `UCCSDAnsatz` (line 334) `class UCCSDAnsatz` - *Unitary Coupled Cluster Singles and Doubles ansatz.

For H2 in the 2-qubit active space model:
- Qubit 0 represents the bonding orbital occupation
- Qubit 1 represents the anti-bonding orbital occupation
- HF state: |10> (bonding occupied, anti-bonding empty)
- The double excitation |10> <-> |01> is mediated by X0X1 + Y0Y1 terms*
- `VQEResult` (line 476) `class VQEResult`
- `VQESolver` (line 510) `class VQESolver` - *Variational Quantum Eigensolver.

CORRECTED: Uses proper UCCSD ansatz and Hamiltonian evaluation.*

**Functions:**
- `_make_logger` (line 58) `def _make_logger(name)`
- `run_vqe_h2` (line 630) `def run_vqe_h2(max_iter)` - *Run VQE for H2 molecule - convenience function.*
- `h2_sto3g` (line 89) `def h2_sto3g(bond_length)` - *Build H2 molecule with STO-3G basis.*
- `_h2_pyscf` (line 96) `def _h2_pyscf(bond_length)` - *Build H2 using PySCF.*
- `_h2_hardcoded` (line 128) `def _h2_hardcoded(bond_length)` - *Build H2 with hardcoded values - CORRECTED for 2-qubit active space.*
- `__init__` (line 163) `def __init__(self, mol, n_qubits)`
- `_build_hamiltonian` (line 170) `def _build_hamiltonian(self)` - *Build the molecular Hamiltonian in JW representation.*
- `_build_openfermion_hamiltonian` (line 178) `def _build_openfermion_hamiltonian(self)` - *Build Hamiltonian using OpenFermion - FIXED API.*
- `_build_hardcoded_hamiltonian` (line 231) `def _build_hardcoded_hamiltonian(self)` - *Build hardcoded H2 Hamiltonian - CORRECTED COEFFICIENTS.

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
- `_apply_pauli` (line 276) `def _apply_pauli(self, state, pauli)` - *Apply Pauli operator to state vector - CORRECTED.*
- `expectation_value` (line 302) `def expectation_value(self, state)` - *Compute ⟨ψ|H|ψ⟩ for the given state.*
- `evaluate` (line 318) `def evaluate(self, amps)` - *Evaluate energy from amplitudes (supports both numpy and torch).*
- `__call__` (line 330) `def __call__(self, amps)`
- `__init__` (line 345) `def __init__(self, n_qubits, n_electrons)`
- `apply_double_excitation_2q` (line 373) `def apply_double_excitation_2q(self, state, theta)` - *Apply double excitation for 2-qubit H2 model.

This rotates between |10> and |01>:
|10> -> cos(theta)*|10> - sin(theta)*|01>
|01> -> sin(theta)*|10> + cos(theta)*|01>*
- `apply_single_excitation` (line 395) `def apply_single_excitation(self, state, o, v, theta)` - *Apply single excitation as Givens rotation.*
- `apply_double_excitation` (line 416) `def apply_double_excitation(self, state, o1, o2, v1, v2, theta)` - *Apply double excitation for 4+ qubit systems.*
- `apply` (line 443) `def apply(self, state, thetas)` - *Apply UCCSD ansatz to state.*
- `__repr__` (line 490) `def __repr__(self)`
- `__init__` (line 517) `def __init__(self, mol)`
- `prepare_hf_state` (line 528) `def prepare_hf_state(self)` - *Prepare Hartree-Fock state.

For H2 with 2 electrons in 4 spin-orbitals:
|1100⟩ means electrons in orbitals 0 and 1.*
- `run` (line 546) `def run(self, max_iter, tol)` - *Run VQE optimization.*
- `cost` (line 566) `def cost(thetas)`

#### `quantum_framework_molecular_fixed.py`
**Path:** `quantum_framework_molecular_fixed.py`

**Classes:**
- `MoleculeData` (line 64) `class MoleculeData`
- `MoleculeBuilder` (line 77) `class MoleculeBuilder` - *Build molecule data for quantum chemistry calculations.
Uses PySCF when available, falls back to hardcoded values.*
- `ExactJWEnergy` (line 147) `class ExactJWEnergy` - *Exact Jordan-Wigner energy evaluator.
Computes molecular energy using JW transformation.

FIXED: Proper Hamiltonian construction and expectation values.*
- `UCCSDAnsatz` (line 343) `class UCCSDAnsatz` - *Unitary Coupled Cluster Singles and Doubles ansatz.

FIXED: Correct excitation operator implementation.*
- `VQEResult` (line 457) `class VQEResult`
- `VQESolver` (line 491) `class VQESolver` - *Variational Quantum Eigensolver.

FIXED: Correct HF state preparation and energy evaluation.*

**Functions:**
- `_make_logger` (line 50) `def _make_logger(name)`
- `_get_sd_indices` (line 323) `def _get_sd_indices(n_electrons, n_qubits)` - *Get single and double excitation indices for UCCSD.*
- `run_vqe_demo` (line 614) `def run_vqe_demo()` - *Run a quick VQE demo to verify the fixes.*
- `h2_sto3g` (line 84) `def h2_sto3g(bond_length)` - *Build H2 molecule with STO-3G basis.*
- `_h2_pyscf` (line 91) `def _h2_pyscf(bond_length)` - *Build H2 using PySCF - FIXED atom string syntax.*
- `_h2_hardcoded` (line 126) `def _h2_hardcoded()` - *Build H2 with hardcoded values - FIXED coefficients.*
- `__init__` (line 155) `def __init__(self, mol, n_qubits)`
- `_build_hamiltonian` (line 162) `def _build_hamiltonian(self)` - *Build the molecular Hamiltonian in JW representation.*
- `_build_openfermion_hamiltonian` (line 169) `def _build_openfermion_hamiltonian(self)` - *Build Hamiltonian using OpenFermion - FIXED geometry.*
- `_build_hardcoded_hamiltonian` (line 215) `def _build_hardcoded_hamiltonian(self)` - *Build hardcoded H2 Hamiltonian - FIXED coefficients.

The H2/STO-3G Hamiltonian in JW form (standard reference):
H = g0*I + g1*Z0 + g2*Z1 + g3*Z0*Z1 + g4*X0*X1 + g5*Y0*Y1

With coefficients for bond length 0.735 Å:*
- `_apply_pauli` (line 247) `def _apply_pauli(self, state, pauli)` - *Apply Pauli operator to state vector.

FIXED: Correct amplitude indexing and phase handling.*
- `expectation_value` (line 281) `def expectation_value(self, state)` - *Compute ⟨ψ|H|ψ⟩ for the given state.

FIXED: Correct inner product calculation.*
- `evaluate` (line 305) `def evaluate(self, amps)` - *Evaluate energy from MPS amplitudes.*
- `__call__` (line 319) `def __call__(self, amps)`
- `__init__` (line 350) `def __init__(self, n_qubits, n_electrons, backend)`
- `apply_single_excitation` (line 359) `def apply_single_excitation(self, state, o, v, theta)` - *Apply single excitation operator exp(theta * (a_v† a_o - a_o† a_v)).

In JW, this is a Givens rotation between orbitals o and v.*
- `apply_double_excitation` (line 387) `def apply_double_excitation(self, state, o1, o2, v1, v2, theta)` - *Apply double excitation operator.

Simplified: applies pairwise excitation with rotation.*
- `apply` (line 419) `def apply(self, state, thetas)` - *Apply UCCSD ansatz to state.*
- `__repr__` (line 471) `def __repr__(self)`
- `__init__` (line 498) `def __init__(self, qc, config)`
- `prepare_hf_state` (line 502) `def prepare_hf_state(self, mol)` - *Prepare Hartree-Fock state.

FIXED: Correct bitstring with 0s and 1s.

For H2 with 2 electrons in 4 spin-orbitals:
|1100⟩ means electrons in orbitals 0 and 1 (occupied spin-orbitals)*
- `run` (line 525) `def run(self, mol, backend, max_iter, tol)` - *Run VQE optimization.*
- `cost` (line 556) `def cost(thetas)`

#### `quantum_framework_molecular_v2.py`
**Path:** `quantum_framework_molecular_v2.py`

**Classes:**
- `MolecularConfig` (line 95) `class MolecularConfig` - *Configuration for molecular VQE simulations.*
- `MoleculeData` (line 172) `class MoleculeData` - *Molecular data structure with all necessary information.*
- `MoleculeBuilder` (line 200) `class MoleculeBuilder` - *Build molecules using OpenFermion + PySCF.

NO hardcoded values - everything comes from quantum chemistry calculations.*
- `HamiltonianBuilder` (line 352) `class HamiltonianBuilder` - *Build molecular Hamiltonians using OpenFermion.

NO hardcoded coefficients - everything from first principles.*
- `CachedPauliOperation` (line 468) `class CachedPauliOperation` - *Precomputed Pauli operation for fast application.*
- `CachedHamiltonianEvaluator` (line 477) `class CachedHamiltonianEvaluator` - *Hamiltonian evaluator with cached Pauli operations.

Improvement: x10-100 speedup for repeated evaluations.*
- `SmartInitializer` (line 613) `class SmartInitializer` - *Smart parameter initialization for VQE.

Improvements:
- MP2 amplitude estimation
- Systematic parameter scan
- Parabolic refinement
- Sensitivity analysis*
- `UCCSDAnsatz` (line 746) `class UCCSDAnsatz` - *Unitary Coupled Cluster Singles and Doubles ansatz.

Features:
- Particle-conserving excitations
- Works with both direct and MPS representations
- Identity check (θ=0 → HF state)*
- `MPSState` (line 879) `class MPSState` - *Matrix Product State for scalable quantum simulation.

Features:
- Adaptive bond dimension
- Efficient gate application
- Entanglement tracking*
- `ParticleConservingState` (line 982) `class ParticleConservingState` - *State representation that preserves particle number symmetry.

Improvement: x4 reduction in Hilbert space, better convergence.*
- `VQEResult` (line 1064) `class VQEResult` - *VQE result container.*
- `VQESolver` (line 1104) `class VQESolver` - *Production VQE solver with all improvements.

Features:
- OpenFermion Hamiltonians (no hardcoded values)
- Precision mode flag (direct vs MPS)
- Smart initialization with MP2 + scan
- Cached Pauli operations
- Particle conservation option
- Backend integration*
- `BackendIntegrator` (line 1287) `class BackendIntegrator` - *Integration with existing backends (Hamiltonian, Schrodinger, Dirac).

Uses pre-trained models from QC repository.*
- `PseudoMolData` (line 301) `class PseudoMolData`

**Functions:**
- `_make_logger` (line 75) `def _make_logger(name)`
- `run_vqe` (line 1332) `def run_vqe(molecule, precision_mode, config_path)` - *Convenience function to run VQE.

Args:
    molecule: Molecule name ("H2", "H2O", "LiH")
    precision_mode: Use direct statevector (True) or MPS (False)
    config_path: Path to TOML config file
    **kwargs: Additional config overrides

Returns:
    VQEResult*
- `from_toml` (line 131) `def from_toml(cls, toml_path)` - *Load configuration from TOML file.*
- `build` (line 208) `def build(name, geometry, basis, charge, multiplicity, description)` - *Build molecule using OpenFermion.

Args:
    name: Molecule name (e.g., "H2", "H2O")
    geometry: List of (atom_symbol, (x, y, z)) in Angstrom
    basis: Basis set (e.g., "sto-3g", "6-31g")
    charge: Molecular charge
    multiplicity: Spin multiplicity
    description: Optional description

Returns:
    MoleculeData with all properties computed*
- `_run_pyscf_direct` (line 289) `def _run_pyscf_direct(geometry, basis, charge, multiplicity)` - *Run PySCF directly if openfermionpyscf not available.*
- `h2` (line 324) `def h2(bond_length, basis)` - *Build H2 molecule.*
- `h2o` (line 330) `def h2o(bond_length_oh, angle_hoh, basis)` - *Build H2O molecule.*
- `lih` (line 342) `def lih(bond_length, basis)` - *Build LiH molecule.*
- `build_jw_hamiltonian` (line 360) `def build_jw_hamiltonian(mol)` - *Build Jordan-Wigner transformed Hamiltonian using OpenFermion.

Args:
    mol: MoleculeData with geometry, basis, etc.

Returns:
    (pauli_terms, nuclear_repulsion)*
- `build_hamiltonian_matrix` (line 417) `def build_hamiltonian_matrix(mol, n_qubits)` - *Build full Hamiltonian matrix for small systems.

Args:
    mol: MoleculeData
    n_qubits: Number of qubits

Returns:
    Hamiltonian matrix (2^n_qubits, 2^n_qubits)*
- `_pauli_matrix` (line 441) `def _pauli_matrix(pauli_list, n_qubits)` - *Build matrix for a Pauli string.*
- `__init__` (line 484) `def __init__(self, mol, config)`
- `_precompute_operations` (line 503) `def _precompute_operations(self)` - *Precompute all Pauli operations for fast evaluation.*
- `_compute_pauli_mapping` (line 522) `def _compute_pauli_mapping(self, pauli_list)` - *Compute index mapping and phases for a Pauli string.*
- `apply_pauli_fast` (line 561) `def apply_pauli_fast(self, state, op)` - *Apply cached Pauli operation.*
- `expectation_value` (line 568) `def expectation_value(self, state)` - *Compute energy expectation value.*
- `batch_expectation` (line 587) `def batch_expectation(self, states)` - *Compute expectation for batch of states.*
- `__init__` (line 624) `def __init__(self, mol, ansatz, evaluator, config)`
- `estimate_mp2_amplitude` (line 630) `def estimate_mp2_amplitude(self)` - *Estimate doubles amplitude from MP2 theory.

θ_MP2 ≈ t_2^(1) / 2
where t_2^(1) = <ij||ab> / (ε_i + ε_j - ε_a - ε_b)*
- `scan_parameter_space` (line 657) `def scan_parameter_space(self, hf_state, n_samples, param_range)` - *Systematic scan of parameter space.

Returns:
    (best_thetas, best_energy)*
- `_parabolic_refinement` (line 712) `def _parabolic_refinement(self, x_vals, y_vals, best_idx, n_params)` - *Refine minimum using parabolic interpolation.*
- `initialize` (line 735) `def initialize(self, hf_state)` - *Complete initialization with all techniques.*
- `__init__` (line 756) `def __init__(self, n_qubits, n_electrons, config)`
- `_generate_excitations` (line 764) `def _generate_excitations(self)` - *Generate all single and double excitations.*
- `apply` (line 785) `def apply(self, state, thetas)` - *Apply UCCSD ansatz to state.

Args:
    state: State vector (2^n_qubits,)
    thetas: Parameters (n_params,)

Returns:
    Transformed state vector*
- `_apply_single` (line 817) `def _apply_single(self, state, i, a, theta)` - *Apply single excitation as Givens rotation.*
- `_apply_double` (line 838) `def _apply_double(self, state, i, j, a, b, theta)` - *Apply double excitation.*
- `verify_identity` (line 862) `def verify_identity(self, hf_state, evaluator, hf_energy)` - *Verify that θ=0 gives HF state.*
- `__init__` (line 889) `def __init__(self, n_qubits, config)`
- `to_statevector` (line 910) `def to_statevector(self)` - *Convert MPS to full statevector.*
- `from_statevector` (line 918) `def from_statevector(cls, state, n_qubits, config)` - *Create MPS from statevector.*
- `compute_entanglement` (line 953) `def compute_entanglement(self, bond_idx)` - *Compute entanglement entropy at bond.*
- `__init__` (line 989) `def __init__(self, n_qubits, n_particles)`
- `_generate_fock_states` (line 1000) `def _generate_fock_states(self)` - *Generate all states with fixed particle number.*
- `hf_state` (line 1013) `def hf_state(self)` - *Create HF state in subspace.*
- `apply_excitation` (line 1027) `def apply_excitation(self, state, occ, vir, theta)` - *Apply excitation preserving particle number.*
- `to_full_statevector` (line 1051) `def to_full_statevector(self, state)` - *Convert subspace state to full statevector.*
- `__repr__` (line 1081) `def __repr__(self)`
- `__init__` (line 1117) `def __init__(self, mol, config)`
- `prepare_hf_state` (line 1137) `def prepare_hf_state(self)` - *Prepare Hartree-Fock state.*
- `evaluate` (line 1152) `def evaluate(self, state)` - *Evaluate energy.*
- `apply_ansatz` (line 1159) `def apply_ansatz(self, state, thetas)` - *Apply UCCSD ansatz.*
- `run` (line 1184) `def run(self)` - *Run VQE optimization.*
- `__init__` (line 1294) `def __init__(self, config)`
- `_load_models` (line 1301) `def _load_models(self)` - *Load pre-trained backend models.*
- `get_backend_energy` (line 1318) `def get_backend_energy(self, state, backend_name)` - *Get energy estimate from backend model.*
- `cost` (line 1212) `def cost(thetas)`

#### `quantum_framework_physics.py`
**Path:** `quantum_framework_physics.py`

**Classes:**
- `SpectralLayer` (line 26) `class SpectralLayer` - *Spectral convolution in frequency domain.
Learns complex kernels that modulate Fourier coefficients.*
- `HamiltonianBackboneNet` (line 66) `class HamiltonianBackboneNet` - *Hamiltonian backbone: single-channel field -> H|psi>.
Shared by all physics backends as the H operator.*
- `SchrodingerSpectralNet` (line 92) `class SchrodingerSpectralNet` - *Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction.
Uses spectral convolution for physics-informed evolution.*
- `DiracSpectralNet` (line 120) `class DiracSpectralNet` - *Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor.
Handles 4-component spinor evolution for relativistic quantum mechanics.*
- `GammaMatrices` (line 150) `class GammaMatrices` - *Dirac gamma matrices in Dirac (standard) or Weyl representation.
gamma^0 = beta, gamma^i = beta * alpha_i*
- `PotentialGenerator` (line 250) `class PotentialGenerator` - *Spatial potentials for eigenstate initialization.*
- `DiracHamiltonianOperator` (line 294) `class DiracHamiltonianOperator` - *Dirac Hamiltonian operator for relativistic quantum mechanics.
H_Dirac = c * alpha . p + beta * m * c^2 + V(r)

In atomic units (c = 1/alpha ~ 137):
H = c * alpha . p + beta * m * c^2 + V*
- `LambShiftCalculator` (line 382) `class LambShiftCalculator` - *Calculates the Lamb shift using Bethe's formula.
The Lamb shift is the small difference between 2s_{1/2} and 2p_{1/2} energy levels
due to QED effects (vacuum fluctuations and self-energy).*
- `AnomalousMagneticMoment` (line 439) `class AnomalousMagneticMoment` - *Calculates the electron's anomalous magnetic moment (g-2).
The electron g-factor is slightly different from 2 due to QED effects:
g = 2(1 + a_e) where a_e = alpha/(2*pi) + higher-order terms*
- `DiracHydrogenAtom` (line 489) `class DiracHydrogenAtom` - *Relativistic hydrogen atom with Dirac equation.
Computes energy levels including fine structure.*
- `ZitterbewegungSimulator` (line 571) `class ZitterbewegungSimulator` - *Simulates the Zitterbewegung (trembling motion) of a relativistic electron.
In Dirac theory, the position operator has a term oscillating with frequency
~ 2mc^2/hbar, which is the interference between positive and negative energy states.*

**Functions:**
- `__init__` (line 32) `def __init__(self, channels, grid_size)`
- `forward` (line 43) `def forward(self, x)`
- `__init__` (line 72) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 81) `def forward(self, x)`
- `__init__` (line 98) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 109) `def forward(self, x)`
- `__init__` (line 126) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 139) `def forward(self, x)`
- `__init__` (line 156) `def __init__(self, representation, device)`
- `_init_matrices` (line 161) `def _init_matrices(self)`
- `to` (line 244) `def to(self, device)`
- `__init__` (line 253) `def __init__(self, grid_size, potential_depth, potential_width)`
- `_grid` (line 258) `def _grid(self)`
- `harmonic` (line 263) `def harmonic(self)`
- `double_well` (line 268) `def double_well(self)`
- `coulomb` (line 274) `def coulomb(self)`
- `periodic_lattice` (line 280) `def periodic_lattice(self)`
- `mixed` (line 284) `def mixed(self, seed)`
- `__init__` (line 303) `def __init__(self, grid_size, electron_mass, c_light, device)`
- `_precompute_operators` (line 311) `def _precompute_operators(self)`
- `apply_dirac_hamiltonian` (line 318) `def apply_dirac_hamiltonian(self, spinor, potential)` - *Apply Dirac Hamiltonian to 4-component spinor.

Args:
    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor
    potential: Optional scalar potential V(r)

Returns:
    H * psi with same shape as input*
- `time_evolution` (line 362) `def time_evolution(self, spinor, dt, potential, normalization_eps)` - *Time evolution of Dirac spinor using first-order split-step.
psi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi*
- `__init__` (line 389) `def __init__(self, alpha_fs, c_light, electron_mass)`
- `bethe_formula` (line 394) `def bethe_formula(self, n, l, Z)` - *Bethe's non-relativistic formula for Lamb shift.
Delta E_Lamb = (8*alpha^3 / 3*pi*n^3) * |psi_n(0)|^2 * ln(E_avg / E_n)*
- `_higher_l_shift` (line 407) `def _higher_l_shift(self, n, l, Z)`
- `full_lamb_shift` (line 412) `def full_lamb_shift(self, n, l, j, Z)` - *Calculate full Lamb shift including radiative corrections.
Delta E = Delta E_SE + Delta E_Uehling + Delta E_rel*
- `__init__` (line 446) `def __init__(self, alpha_fs)`
- `schwinger_term` (line 449) `def schwinger_term(self)`
- `second_order` (line 452) `def second_order(self)`
- `third_order` (line 456) `def third_order(self)`
- `fourth_order` (line 460) `def fourth_order(self)`
- `fifth_order` (line 464) `def fifth_order(self)`
- `calculate_a_e` (line 468) `def calculate_a_e(self, order)`
- `__init__` (line 495) `def __init__(self, c_light, alpha_fs)`
- `energy_level_dirac` (line 499) `def energy_level_dirac(self, n, kappa)` - *Exact Dirac energy level for hydrogen-like atom.
E = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)*
- `fine_structure_splitting` (line 512) `def fine_structure_splitting(self, n, l)` - *Calculate fine structure splitting for given n, l.
Returns energies for j = l+1/2 and j = l-1/2*
- `energy_spectrum` (line 535) `def energy_spectrum(self, n_max)`
- `__init__` (line 578) `def __init__(self, grid_size, c_light, electron_mass, device)`
- `create_gaussian_wave_packet` (line 585) `def create_gaussian_wave_packet(self, sigma, momentum)`
- `compute_position_expectation` (line 603) `def compute_position_expectation(self, spinor)`
- `compute_velocity_expectation` (line 615) `def compute_velocity_expectation(self, spinor)`

#### `quantum_framework_visualization.py`
**Path:** `quantum_framework_visualization.py`

**Classes:**
- `WavefunctionCalculator` (line 59) `class WavefunctionCalculator` - *Calculates hydrogen atom wavefunctions for visualization.
Uses analytical formulas for radial and angular parts.*
- `MonteCarloSampler` (line 115) `class MonteCarloSampler` - *Monte Carlo sampling for orbital visualization.
Uses rejection sampling to generate 3D point clouds.*
- `OrbitalVisualizer` (line 202) `class OrbitalVisualizer` - *High-resolution visualization of hydrogen orbitals.
Creates 2D projections and 3D scatter plots.*
- `EntangledHydrogenSampler` (line 290) `class EntangledHydrogenSampler` - *Monte Carlo sampler for entangled hydrogen states.
Samples from joint probability distribution of entangled orbitals.*
- `EntangledHydrogenVisualizer` (line 325) `class EntangledHydrogenVisualizer` - *Visualizer for entangled hydrogen states.
Creates high-resolution visualizations with multiple orbitals.*

**Functions:**
- `_make_logger` (line 46) `def _make_logger(name)`
- `__init__` (line 65) `def __init__(self, config)`
- `radial_wavefunction` (line 69) `def radial_wavefunction(n, l, r)`
- `spherical_harmonic_real` (line 81) `def spherical_harmonic_real(l, m, theta, phi)`
- `psi_3d` (line 92) `def psi_3d(self, n, l, m, r, theta, phi)`
- `psi_on_grid` (line 97) `def psi_on_grid(self, n, l, m)`
- `__init__` (line 121) `def __init__(self, config, wavefunction_calc)`
- `find_max_probability` (line 125) `def find_max_probability(self, n, l, m)`
- `sample` (line 149) `def sample(self, n, l, m, num_samples)`
- `__init__` (line 208) `def __init__(self, config)`
- `visualize` (line 211) `def visualize(self, data, save_path)`
- `__init__` (line 296) `def __init__(self, config, wavefunction_calc)`
- `sample_entangled_state` (line 301) `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples, entanglement_weight)`
- `__init__` (line 331) `def __init__(self, config)`
- `visualize` (line 334) `def visualize(self, data, quantum_result, save_path)`

#### `quantum_lab.py`
**Path:** `quantum_lab.py`

**Classes:**
- `OrbitalSpec` (line 431) `class OrbitalSpec` - *Definition of a real hydrogen orbital for the ASCII viewer.*
- `H2VQEEngine` (line 604) `class H2VQEEngine` - *Minimal live VQE for H2 in the 2-qubit active space.

The trial state cos(theta/2)|01> + sin(theta/2)|10> is prepared with a
real circuit (Ry, CNOT, X) on the MPS engine, and the energy is read
from the Jordan-Wigner H2 Hamiltonian of quantum_framework_molecular.*
- `Quiz` (line 659) `class Quiz`
- `Lesson` (line 667) `class Lesson`
- `QuantumLab` (line 672) `class QuantumLab` - *Interactive educational TUI driven by the real Q2C engine.*

**Functions:**
- `probability_bars` (line 325) `def probability_bars(probs, n_qubits, lang, max_rows)` - *Build a table of probability bars for a state's distribution.*
- `counts_bars` (line 353) `def counts_bars(counts, total, lang)` - *Build a table of measurement-count bars.*
- `draw_circuit` (line 369) `def draw_circuit(n_qubits, instructions)` - *Render an ASCII timeline of the circuit, one line per qubit.*
- `parse_angle` (line 394) `def parse_angle(token)` - *Parse an angle like '1.57', 'pi', '-pi/2' or '3*pi/4' into radians.*
- `sample_measurements` (line 414) `def sample_measurements(probs, n_qubits, n_samples, rng)` - *Draw measurement outcomes from a probability distribution.*
- `_radial` (line 440) `def _radial(n, l, r)` - *Hydrogen radial wavefunction R_nl(r) in atomic units.*
- `_ang_s` (line 452) `def _ang_s(x, y, z, r)`
- `_ang_pz` (line 456) `def _ang_pz(x, y, z, r)`
- `_ang_px` (line 460) `def _ang_px(x, y, z, r)`
- `_ang_dz2` (line 464) `def _ang_dz2(x, y, z, r)`
- `_ang_dxz` (line 469) `def _ang_dxz(x, y, z, r)`
- `_ang_dxy` (line 473) `def _ang_dxy(x, y, z, r)`
- `field_to_text` (line 490) `def field_to_text(psi)` - *Render a real scalar field as colored ASCII: brightness = |psi|^2, color = sign.*
- `render_orbital` (line 516) `def render_orbital(spec, rows, cols)` - *Render |psi|^2 of a hydrogen orbital on a plane slice as colored ASCII.*
- `render_h2_molecular_orbital` (line 536) `def render_h2_molecular_orbital(kind, rows, cols)` - *Render the bonding or antibonding LCAO molecular orbital of H2.*
- `landscape_plot` (line 556) `def landscape_plot(energies, e_hf, e_fci, marker, rows)` - *Draw an ASCII plot of E(theta) over one full period with HF and FCI
reference lines and an optional optimizer marker.*
- `launch_quantum_lab` (line 1474) `def launch_quantum_lab(config, loader, lang, lesson)` - *Entry point used both standalone and from quantum_framework_main.*
- `main` (line 1500) `def main()`
- `row_of` (line 567) `def row_of(e)`
- `__init__` (line 613) `def __init__(self, qc)`
- `ansatz_instructions` (line 621) `def ansatz_instructions(self, theta)`
- `energy` (line 624) `def energy(self, theta)`
- `correlation_pct` (line 633) `def correlation_pct(self, e)`
- `landscape` (line 636) `def landscape(self, cols)`
- `optimize` (line 640) `def optimize(self, theta0, lr, max_iters, tol)` - *Gradient descent; yields (iteration, theta, energy) live.*
- `__init__` (line 675) `def __init__(self, config, loader, lang)`
- `t` (line 683) `def t(self, key)`
- `pause` (line 690) `def pause(self)`
- `panel` (line 697) `def panel(self, body, title, style)`
- `show_state` (line 701) `def show_state(self, probs, n_qubits, title)`
- `run_quiz` (line 707) `def run_quiz(self, quiz)`
- `lessons` (line 728) `def lessons(self)`
- `_demo_superposition` (line 733) `def _demo_superposition(self)`
- `_demo_rotation` (line 744) `def _demo_rotation(self)`
- `_demo_measurement` (line 766) `def _demo_measurement(self)`
- `_demo_bell` (line 779) `def _demo_bell(self)`
- `_demo_ghz_w` (line 792) `def _demo_ghz_w(self)`
- `_demo_grover` (line 799) `def _demo_grover(self)`
- `_demo_molecule` (line 815) `def _demo_molecule(self)`
- `_vqe` (line 821) `def _vqe(self)`
- `_demo_h2_clouds` (line 826) `def _demo_h2_clouds(self)`
- `_demo_vqe_ansatz` (line 835) `def _demo_vqe_ansatz(self)`
- `_demo_vqe_landscape` (line 840) `def _demo_vqe_landscape(self)`
- `_demo_vqe_live` (line 847) `def _demo_vqe_live(self)`
- `_lessons_en` (line 875) `def _lessons_en(self)`
- `_lessons_es` (line 1015) `def _lessons_es(self)`
- `run_lesson` (line 1159) `def run_lesson(self, lesson)`
- `lessons_menu` (line 1175) `def lessons_menu(self)`
- `_rebuild_state` (line 1198) `def _rebuild_state(self, n_qubits, instructions)`
- `_playground_dashboard` (line 1216) `def _playground_dashboard(self, n_qubits, state, instructions)`
- `_show_amplitudes` (line 1228) `def _show_amplitudes(self, state, n_qubits)`
- `playground` (line 1249) `def playground(self)`
- `_molecule_card` (line 1346) `def _molecule_card(self, mol)`
- `molecule_explorer` (line 1365) `def molecule_explorer(self)`
- `orbital_viewer` (line 1383) `def orbital_viewer(self)`
- `chemistry_menu` (line 1403) `def chemistry_menu(self)`
- `glossary` (line 1422) `def glossary(self)`
- `banner` (line 1437) `def banner(self)`
- `main_menu` (line 1443) `def main_menu(self)`

#### `quantum_simulator.py`
**Path:** `quantum_simulator.py`

**Classes:**
- `FrameworkConfig` (line 68) `class FrameworkConfig`
- `AtomData` (line 164) `class AtomData`
- `MoleculeData` (line 174) `class MoleculeData`
- `OrbitalData` (line 192) `class OrbitalData`
- `ConfigLoader` (line 200) `class ConfigLoader`
- `SpectralLayer` (line 288) `class SpectralLayer`
- `HamiltonianBackboneNet` (line 303) `class HamiltonianBackboneNet`
- `SchrodingerSpectralNet` (line 321) `class SchrodingerSpectralNet`
- `DiracSpectralNet` (line 341) `class DiracSpectralNet`
- `GammaMatrices` (line 361) `class GammaMatrices`
- `JointHilbertState` (line 380) `class JointHilbertState`
- `IPhysicsBackend` (line 410) `class IPhysicsBackend(ABC)`
- `HamiltonianBackend` (line 420) `class HamiltonianBackend(IPhysicsBackend)`
- `SchrodingerBackend` (line 470) `class SchrodingerBackend(IPhysicsBackend)`
- `DiracBackend` (line 505) `class DiracBackend(IPhysicsBackend)`
- `IQuantumGate` (line 638) `class IQuantumGate(ABC)`
- `HadamardGate` (line 649) `class HadamardGate(IQuantumGate)`
- `PauliXGate` (line 662) `class PauliXGate(IQuantumGate)`
- `PauliYGate` (line 674) `class PauliYGate(IQuantumGate)`
- `PauliZGate` (line 686) `class PauliZGate(IQuantumGate)`
- `SGate` (line 698) `class SGate(IQuantumGate)`
- `TGate` (line 710) `class TGate(IQuantumGate)`
- `RxGate` (line 723) `class RxGate(IQuantumGate)`
- `RyGate` (line 737) `class RyGate(IQuantumGate)`
- `RzGate` (line 751) `class RzGate(IQuantumGate)`
- `CNOTGate` (line 766) `class CNOTGate(IQuantumGate)`
- `CZGate` (line 778) `class CZGate(IQuantumGate)`
- `SWAPGate` (line 790) `class SWAPGate(IQuantumGate)`
- `ToffoliGate` (line 802) `class ToffoliGate(IQuantumGate)`
- `CircuitInstruction` (line 832) `class CircuitInstruction`
- `QuantumCircuit` (line 838) `class QuantumCircuit`
- `QuantumResult` (line 894) `class QuantumResult`
- `PotentialGenerator` (line 908) `class PotentialGenerator`
- `JointStateFactory` (line 972) `class JointStateFactory`
- `QuantumComputer` (line 999) `class QuantumComputer`
- `WavefunctionCalculator` (line 1043) `class WavefunctionCalculator`
- `MonteCarloSampler` (line 1080) `class MonteCarloSampler`
- `DiracHydrogenAtom` (line 1159) `class DiracHydrogenAtom`
- `ZitterbewegungSimulator` (line 1200) `class ZitterbewegungSimulator`
- `OrbitalVisualizer` (line 1282) `class OrbitalVisualizer`
- `EntangledVisualizer` (line 1343) `class EntangledVisualizer`
- `QuantumSimulationFramework` (line 1409) `class QuantumSimulationFramework`
- `InteractiveMenu` (line 1535) `class InteractiveMenu`

**Functions:**
- `_make_logger` (line 54) `def _make_logger(name, level)`
- `_single_qubit_unitary` (line 587) `def _single_qubit_unitary(state, qubit, u, backend)`
- `_two_qubit_unitary` (line 611) `def _two_qubit_unitary(state, ctrl, tgt, u4)`
- `_solve_eigenstate` (line 949) `def _solve_eigenstate(config, potential, n)`
- `_build_basis_amplitude` (line 965) `def _build_basis_amplitude(config, basis_idx)`
- `main` (line 1868) `def main()`
- `from_toml` (line 112) `def from_toml(cls, toml_path)`
- `__init__` (line 201) `def __init__(self, config_path)`
- `_find_config` (line 209) `def _find_config(self)`
- `_load` (line 219) `def _load(self)`
- `_load_defaults` (line 229) `def _load_defaults(self)`
- `_parse_atoms` (line 236) `def _parse_atoms(self)`
- `_parse_molecules` (line 241) `def _parse_molecules(self)`
- `_parse_orbitals` (line 246) `def _parse_orbitals(self)`
- `get_atom` (line 251) `def get_atom(self, symbol)`
- `get_molecule` (line 259) `def get_molecule(self, name)`
- `get_orbital` (line 267) `def get_orbital(self, name)`
- `atoms` (line 276) `def atoms(self)`
- `molecules` (line 280) `def molecules(self)`
- `orbitals` (line 284) `def orbitals(self)`
- `__init__` (line 289) `def __init__(self, channels, grid_size)`
- `forward` (line 295) `def forward(self, x)`
- `__init__` (line 304) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 310) `def forward(self, x)`
- `__init__` (line 322) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 330) `def forward(self, x)`
- `__init__` (line 342) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 350) `def forward(self, x)`
- `__init__` (line 362) `def __init__(self, representation, device)`
- `__init__` (line 381) `def __init__(self, amplitudes, n_qubits)`
- `normalize_` (line 388) `def normalize_(self)`
- `probabilities` (line 393) `def probabilities(self)`
- `entropy` (line 397) `def entropy(self)`
- `most_probable_bitstring` (line 402) `def most_probable_bitstring(self)`
- `clone` (line 406) `def clone(self)`
- `evolve_amplitude` (line 412) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 416) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 421) `def __init__(self, config)`
- `_load` (line 429) `def _load(self)`
- `_precompute_laplacian` (line 443) `def _precompute_laplacian(self)`
- `_apply_h` (line 450) `def _apply_h(self, field)`
- `evolve_amplitude` (line 458) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 465) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 471) `def __init__(self, config, hamiltonian)`
- `_load` (line 478) `def _load(self)`
- `evolve_amplitude` (line 493) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 501) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 506) `def __init__(self, config, hamiltonian)`
- `_load` (line 515) `def _load(self)`
- `_precompute_dirac` (line 530) `def _precompute_dirac(self)`
- `_pack` (line 537) `def _pack(self, amp)`
- `_unpack` (line 547) `def _unpack(self, spinor)`
- `_analytical_dirac` (line 553) `def _analytical_dirac(self, spinor)`
- `evolve_amplitude` (line 564) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 577) `def apply_phase(self, amp, phase_angle)`
- `evolve_spinor` (line 580) `def evolve_spinor(self, spinor, dt)`
- `name` (line 641) `def name(self)`
- `apply` (line 645) `def apply(self, state, backend, targets, params)`
- `name` (line 651) `def name(self)`
- `apply` (line 654) `def apply(self, state, backend, targets, params)`
- `name` (line 664) `def name(self)`
- `apply` (line 667) `def apply(self, state, backend, targets, params)`
- `name` (line 676) `def name(self)`
- `apply` (line 679) `def apply(self, state, backend, targets, params)`
- `name` (line 688) `def name(self)`
- `apply` (line 691) `def apply(self, state, backend, targets, params)`
- `name` (line 700) `def name(self)`
- `apply` (line 703) `def apply(self, state, backend, targets, params)`
- `name` (line 712) `def name(self)`
- `apply` (line 715) `def apply(self, state, backend, targets, params)`
- `name` (line 725) `def name(self)`
- `apply` (line 728) `def apply(self, state, backend, targets, params)`
- `name` (line 739) `def name(self)`
- `apply` (line 742) `def apply(self, state, backend, targets, params)`
- `name` (line 753) `def name(self)`
- `apply` (line 756) `def apply(self, state, backend, targets, params)`
- `name` (line 768) `def name(self)`
- `apply` (line 771) `def apply(self, state, backend, targets, params)`
- `name` (line 780) `def name(self)`
- `apply` (line 783) `def apply(self, state, backend, targets, params)`
- `name` (line 792) `def name(self)`
- `apply` (line 795) `def apply(self, state, backend, targets, params)`
- `name` (line 804) `def name(self)`
- `apply` (line 807) `def apply(self, state, backend, targets, params)`
- `__init__` (line 839) `def __init__(self, n_qubits)`
- `_append` (line 843) `def _append(self, gate_name, targets, params)`
- `h` (line 846) `def h(self, qubit)`
- `x` (line 849) `def x(self, qubit)`
- `y` (line 852) `def y(self, qubit)`
- `z` (line 855) `def z(self, qubit)`
- `s` (line 858) `def s(self, qubit)`
- `t` (line 861) `def t(self, qubit)`
- `rx` (line 864) `def rx(self, qubit, theta)`
- `ry` (line 867) `def ry(self, qubit, theta)`
- `rz` (line 870) `def rz(self, qubit, theta)`
- `cnot` (line 873) `def cnot(self, control, target)`
- `cz` (line 876) `def cz(self, control, target)`
- `swap` (line 879) `def swap(self, qubit1, qubit2)`
- `ccx` (line 882) `def ccx(self, ctrl0, ctrl1, target)`
- `run` (line 885) `def run(self, state, backend)`
- `__init__` (line 895) `def __init__(self, state)`
- `entropy` (line 898) `def entropy(self)`
- `most_probable_bitstring` (line 901) `def most_probable_bitstring(self)`
- `probabilities` (line 904) `def probabilities(self)`
- `__init__` (line 909) `def __init__(self, config)`
- `_grid` (line 913) `def _grid(self)`
- `harmonic` (line 918) `def harmonic(self)`
- `double_well` (line 923) `def double_well(self)`
- `coulomb` (line 929) `def coulomb(self)`
- `periodic_lattice` (line 935) `def periodic_lattice(self)`
- `mixed` (line 939) `def mixed(self, seed)`
- `__init__` (line 973) `def __init__(self, config)`
- `_empty` (line 976) `def _empty(self, n_qubits)`
- `all_zeros` (line 979) `def all_zeros(self, n_qubits)`
- `basis_state` (line 986) `def basis_state(self, n_qubits, k)`
- `from_bitstring` (line 995) `def from_bitstring(self, bitstring)`
- `__init__` (line 1000) `def __init__(self, config)`
- `create_circuit` (line 1011) `def create_circuit(self, n_qubits)`
- `run_circuit` (line 1014) `def run_circuit(self, circuit, initial_state, backend)`
- `bell_state` (line 1021) `def bell_state(self, backend)`
- `ghz_state` (line 1027) `def ghz_state(self, n_qubits, backend)`
- `factory` (line 1035) `def factory(self)`
- `backends` (line 1039) `def backends(self)`
- `__init__` (line 1044) `def __init__(self, config)`
- `radial_wavefunction` (line 1048) `def radial_wavefunction(n, l, r)`
- `spherical_harmonic_real` (line 1060) `def spherical_harmonic_real(l, m, theta, phi)`
- `psi_3d` (line 1071) `def psi_3d(self, n, l, m, r, theta, phi)`
- `energy_analytical` (line 1076) `def energy_analytical(self, n)`
- `__init__` (line 1081) `def __init__(self, config, wavefunction_calc)`
- `find_max_probability` (line 1085) `def find_max_probability(self, n, l, m)`
- `sample_orbital` (line 1109) `def sample_orbital(self, n, l, m, num_samples, Z)`
- `sample_entangled_state` (line 1146) `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples)`
- `__init__` (line 1160) `def __init__(self, config)`
- `energy_level_dirac` (line 1165) `def energy_level_dirac(self, n, kappa, Z)`
- `energy_schrodinger` (line 1173) `def energy_schrodinger(self, n, Z)`
- `fine_structure_splitting` (line 1176) `def fine_structure_splitting(self, n, l, Z)`
- `energy_spectrum` (line 1184) `def energy_spectrum(self, n_max, Z)`
- `__init__` (line 1201) `def __init__(self, config, dirac_backend)`
- `create_gaussian_wave_packet` (line 1206) `def create_gaussian_wave_packet(self, sigma, momentum)`
- `compute_position_expectation` (line 1224) `def compute_position_expectation(self, spinor)`
- `compute_velocity_expectation` (line 1235) `def compute_velocity_expectation(self, spinor)`
- `simulate` (line 1246) `def simulate(self, duration, dt, sigma)`
- `__init__` (line 1283) `def __init__(self, config)`
- `visualize` (line 1286) `def visualize(self, data, save_path, title_suffix)`
- `__init__` (line 1344) `def __init__(self, config)`
- `visualize` (line 1347) `def visualize(self, data, quantum_result, save_path)`
- `__init__` (line 1410) `def __init__(self, config_path)`
- `_ensure_output_dir` (line 1422) `def _ensure_output_dir(self)`
- `list_available_atoms` (line 1425) `def list_available_atoms(self)`
- `list_available_molecules` (line 1428) `def list_available_molecules(self)`
- `list_available_orbitals` (line 1431) `def list_available_orbitals(self)`
- `get_atom` (line 1434) `def get_atom(self, symbol)`
- `get_molecule` (line 1437) `def get_molecule(self, name)`
- `get_orbital` (line 1440) `def get_orbital(self, name)`
- `run_quantum_circuit` (line 1443) `def run_quantum_circuit(self, circuit, backend)`
- `visualize_orbital` (line 1446) `def visualize_orbital(self, orbital_name, num_samples, save, Z, title_suffix)`
- `visualize_atom_orbitals` (line 1458) `def visualize_atom_orbitals(self, atom_symbol, num_samples, save)`
- `visualize_entangled_state` (line 1482) `def visualize_entangled_state(self, orbital1, orbital2, num_samples, save)`
- `compute_relativistic_energy` (line 1496) `def compute_relativistic_energy(self, n, l, Z)`
- `compute_energy_spectrum` (line 1499) `def compute_energy_spectrum(self, n_max, Z)`
- `run_zitterbewegung_simulation` (line 1502) `def run_zitterbewegung_simulation(self, duration, dt, sigma)`
- `run_all_demonstrations` (line 1505) `def run_all_demonstrations(self, num_samples)`
- `__init__` (line 1536) `def __init__(self, framework)`
- `display_header` (line 1540) `def display_header(self)`
- `display_main_menu` (line 1548) `def display_main_menu(self)`
- `get_user_choice` (line 1564) `def get_user_choice(self, prompt)`
- `_display_quantum_result` (line 1570) `def _display_quantum_result(self, result, n_qubits)`
- `orbital_menu` (line 1578) `def orbital_menu(self)`
- `atom_orbital_menu` (line 1602) `def atom_orbital_menu(self)`
- `entangled_menu` (line 1622) `def entangled_menu(self)`
- `quantum_circuit_menu` (line 1651) `def quantum_circuit_menu(self)`
- `relativistic_menu` (line 1737) `def relativistic_menu(self)`
- `zitterbewegung_menu` (line 1777) `def zitterbewegung_menu(self)`
- `molecular_menu` (line 1796) `def molecular_menu(self)`
- `atomic_menu` (line 1820) `def atomic_menu(self)`
- `run` (line 1840) `def run(self)`

#### `quantum_visualizer.py`
**Path:** `quantum_visualizer.py`

**Classes:**
- `VisualizerConfig` (line 109) `class VisualizerConfig`
- `QuantumStateSnapshot` (line 157) `class QuantumStateSnapshot`
- `BackendComparisonResult` (line 170) `class BackendComparisonResult`
- `VisualizationResult` (line 181) `class VisualizationResult`
- `IVisualizationComponent` (line 190) `class IVisualizationComponent(ABC)`
- `ProbabilityBarRenderer` (line 196) `class ProbabilityBarRenderer(IVisualizationComponent)`
- `BlochSphereRenderer` (line 227) `class BlochSphereRenderer(IVisualizationComponent)`
- `PhasePlotRenderer` (line 258) `class PhasePlotRenderer(IVisualizationComponent)`
- `EntropyPlotRenderer` (line 291) `class EntropyPlotRenderer(IVisualizationComponent)`
- `BackendComparisonRenderer` (line 313) `class BackendComparisonRenderer(IVisualizationComponent)`
- `QuantumStateAnalyzer` (line 341) `class QuantumStateAnalyzer`
- `CircuitExecutor` (line 397) `class CircuitExecutor`
- `StandardCircuits` (line 457) `class StandardCircuits`
- `FigureBuilder` (line 528) `class FigureBuilder`
- `QuantumVisualizer` (line 612) `class QuantumVisualizer`

**Functions:**
- `_make_logger` (line 93) `def _make_logger(name)`
- `main` (line 856) `def main()`
- `render` (line 192) `def render(self, data, axes, config)`
- `render` (line 197) `def render(self, data, axes, config)`
- `_get_colors` (line 218) `def _get_colors(self, probs, config)`
- `render` (line 228) `def render(self, data, axes, config)`
- `render` (line 259) `def render(self, data, axes, config)`
- `render` (line 292) `def render(self, snapshots, axes, config)`
- `render` (line 314) `def render(self, results, axes, config)`
- `__init__` (line 342) `def __init__(self, config)`
- `compute_probabilities` (line 345) `def compute_probabilities(self, state)`
- `compute_phases` (line 349) `def compute_phases(self, state)`
- `compute_entropy` (line 360) `def compute_entropy(self, probs)`
- `compute_bloch_vectors` (line 367) `def compute_bloch_vectors(self, state)`
- `create_snapshot` (line 374) `def create_snapshot(self, state, step, gate_name)`
- `__init__` (line 398) `def __init__(self, qc, config)`
- `execute_sequence` (line 403) `def execute_sequence(self, gates, n_qubits, backend_name)`
- `compare_backends` (line 423) `def compare_backends(self, gates, n_qubits, reference_backend)`
- `bell_state` (line 459) `def bell_state()`
- `ghz_state` (line 466) `def ghz_state(n_qubits)`
- `qft` (line 473) `def qft(n_qubits)`
- `grover_oracle` (line 488) `def grover_oracle(n_qubits, marked)`
- `grover_diffusion` (line 501) `def grover_diffusion(n_qubits)`
- `custom_sequence` (line 514) `def custom_sequence(sequence)`
- `__init__` (line 529) `def __init__(self, config)`
- `build_evolution_figure` (line 537) `def build_evolution_figure(self, snapshots, backend_results)`
- `build_summary_figure` (line 563) `def build_summary_figure(self, snapshots, backend_results)`
- `_render_backend_fidelity` (line 594) `def _render_backend_fidelity(self, results, axes)`
- `__init__` (line 613) `def __init__(self, config)`
- `_initialize` (line 620) `def _initialize(self)`
- `_init_quantum_computer` (line 629) `def _init_quantum_computer(self)`
- `visualize_bell_state` (line 654) `def visualize_bell_state(self)`
- `visualize_ghz_state` (line 683) `def visualize_ghz_state(self, n_qubits)`
- `visualize_qft` (line 711) `def visualize_qft(self, n_qubits)`
- `visualize_grover` (line 739) `def visualize_grover(self, n_qubits, marked_state)`
- `visualize_custom_circuit` (line 778) `def visualize_custom_circuit(self, gates, n_qubits, name)`
- `run_all_visualizations` (line 810) `def run_all_visualizations(self)`
- `_save_figure` (line 825) `def _save_figure(self, fig, name)`
- `_print_summary` (line 842) `def _print_summary(self, results)`

#### `relativistic_hydrogen.py`
**Path:** `relativistic_hydrogen.py`

**Classes:**
- `Config` (line 41) `class Config`
- `LoggerFactory` (line 95) `class LoggerFactory`
- `GammaMatrices` (line 113) `class GammaMatrices` - *Dirac gamma matrices in Dirac (standard) representation.
gamma^0 = beta, gamma^i = beta * alpha_i*
- `DiracHamiltonianOperator` (line 199) `class DiracHamiltonianOperator` - *Dirac Hamiltonian operator for relativistic quantum mechanics.
H_Dirac = c * alpha . p + beta * m * c^2 + V(r)

In atomic units (c = 1/alpha ~ 137):
H = c * alpha . p + beta * m * c^2 + V*
- `SpectralLayer` (line 310) `class SpectralLayer`
- `DiracSpectralNetwork` (line 351) `class DiracSpectralNetwork` - *Neural network for learning Dirac equation dynamics.
Handles 4-component spinors with real and imaginary parts (8 channels total).*
- `DiracModelWrapper` (line 393) `class DiracModelWrapper` - *Wrapper to load and use the trained Dirac model.*
- `DiracHydrogenAtom` (line 516) `class DiracHydrogenAtom` - *Relativistic hydrogen atom with Dirac equation.
Computes energy levels including fine structure.*
- `ZitterbewegungSimulator` (line 640) `class ZitterbewegungSimulator` - *Simulates the Zitterbewegung (trembling motion) of a relativistic electron.

In Dirac theory, the position operator has a term oscillating with frequency
~ 2mc^2/hbar, which is the interference between positive and negative energy states.

<x(t)> = <x(0)> + (p/m) * t + oscillating term
The oscillating term has amplitude ~ hbar/(2mc) ~ 10^-12 m*
- `DiracWavefunctionCalculator` (line 833) `class DiracWavefunctionCalculator` - *Calculate relativistic hydrogen wavefunctions.*
- `DiracMonteCarloSampler` (line 956) `class DiracMonteCarloSampler` - *Monte Carlo sampling for relativistic orbital visualization.*
- `DiracVisualizer` (line 1075) `class DiracVisualizer` - *Visualization suite for Dirac equation results.*
- `DiracValidationSuite` (line 1370) `class DiracValidationSuite` - *Complete validation suite for Dirac equation grokking.*

**Functions:**
- `main` (line 1613) `def main()`
- `create_logger` (line 97) `def create_logger(name, level)`
- `__init__` (line 118) `def __init__(self, device)`
- `_init_matrices` (line 122) `def _init_matrices(self)`
- `__init__` (line 207) `def __init__(self, config)`
- `_precompute_operators` (line 215) `def _precompute_operators(self)`
- `apply_dirac_hamiltonian` (line 223) `def apply_dirac_hamiltonian(self, spinor, potential)` - *Apply Dirac Hamiltonian to 4-component spinor.

Args:
    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor
    potential: Optional scalar potential V(r)

Returns:
    H * psi with same shape as input*
- `time_evolution` (line 282) `def time_evolution(self, spinor, dt, potential)` - *Time evolution of Dirac spinor using first-order split-step.
psi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi*
- `__init__` (line 311) `def __init__(self, channels, grid_size)`
- `forward` (line 322) `def forward(self, x)`
- `__init__` (line 356) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, spinor_components)`
- `forward` (line 379) `def forward(self, x)`
- `__init__` (line 397) `def __init__(self, config)`
- `_find_best_checkpoint` (line 406) `def _find_best_checkpoint(self)`
- `_load_model` (line 452) `def _load_model(self)`
- `apply_hamiltonian` (line 498) `def apply_hamiltonian(self, spinor, potential)` - *Apply Hamiltonian using analytical operator.
The NN model learns spinor evolution, but the Hamiltonian operator
is applied analytically for physical validation.*
- `evolve_spinor` (line 506) `def evolve_spinor(self, spinor, dt, potential)` - *Evolve spinor in time using the analytical Dirac operator.*
- `__init__` (line 521) `def __init__(self, config)`
- `energy_level_dirac` (line 526) `def energy_level_dirac(self, n, kappa)` - *Exact Dirac energy level for hydrogen-like atom.

E = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)

For hydrogen (Z=1):
E = mc^2 * [1 + (alpha^2 / (n - |kappa| + sqrt(kappa^2 - alpha^2)))^2]^(-1/2)

Args:
    n: Principal quantum number
    kappa: Relativistic quantum number (kappa = -(l+1) for j=l+1/2, kappa = l for j=l-1/2)

Returns:
    Energy in atomic units (relative to m*c^2)*
- `fine_structure_splitting` (line 556) `def fine_structure_splitting(self, n, l)` - *Calculate fine structure splitting for given n, l.

Fine structure includes:
1. Relativistic correction to kinetic energy
2. Spin-orbit coupling
3. Darwin term (for l=0)

Returns energies for j = l+1/2 and j = l-1/2*
- `energy_spectrum` (line 597) `def energy_spectrum(self, n_max)` - *Generate relativistic energy spectrum up to n_max.*
- `__init__` (line 650) `def __init__(self, config, model_wrapper)`
- `create_gaussian_wave_packet` (line 657) `def create_gaussian_wave_packet(self, sigma, momentum)` - *Create a Gaussian wave packet for a free particle.

For Dirac, we need a 4-component spinor that's a superposition
of positive energy states.*
- `compute_position_expectation` (line 702) `def compute_position_expectation(self, spinor)` - *Compute expectation value of position operator.
<x> = <psi| x |psi>*
- `compute_velocity_expectation` (line 724) `def compute_velocity_expectation(self, spinor)` - *Compute expectation value of velocity operator.
In Dirac theory, v = c * alpha

<v_x> = c * <psi| alpha_x |psi>*
- `simulate` (line 750) `def simulate(self, duration, dt, sigma)` - *Run Zitterbewegung simulation.

Returns time evolution of position and velocity showing the
oscillatory ZBW term.*
- `__init__` (line 837) `def __init__(self, config)`
- `radial_wavefunction_schrodinger` (line 843) `def radial_wavefunction_schrodinger(n, l, r)` - *Non-relativistic radial wavefunction for comparison.*
- `radial_wavefunction_dirac` (line 853) `def radial_wavefunction_dirac(self, n, kappa, r, Z)` - *Relativistic radial wavefunctions for hydrogen.

Returns (f, g) - small and large components.
For bound states, the Dirac radial functions are:
f(r) = sqrt((E+mc^2)/(2E)) * G(r)
g(r) = sqrt((E-mc^2)/(2E)) * F(r)

Simplified version using Sommerfeld fine-structure formula.*
- `spherical_harmonic_real` (line 900) `def spherical_harmonic_real(self, l, m, theta, phi)` - *Real spherical harmonics.*
- `spin_angular_function` (line 910) `def spin_angular_function(self, kappa, m_j, theta, phi)` - *Spin-angular functions Omega_{kappa,m_j}(theta, phi).

These couple the orbital and spin degrees of freedom.*
- `__init__` (line 960) `def __init__(self, config, model_wrapper)`
- `sample_orbital` (line 966) `def sample_orbital(self, n, l, j, num_samples)` - *Sample points from a relativistic hydrogen orbital.*
- `__init__` (line 1079) `def __init__(self, config)`
- `visualize_orbital` (line 1082) `def visualize_orbital(self, data, save_path)` - *Visualize relativistic orbital.*
- `visualize_energy_spectrum` (line 1213) `def visualize_energy_spectrum(self, spectrum, save_path)` - *Visualize relativistic energy spectrum with fine structure.*
- `visualize_zitterbewegung` (line 1296) `def visualize_zitterbewegung(self, zbw_data, save_path)` - *Visualize Zitterbewegung oscillation.*
- `__init__` (line 1374) `def __init__(self, config)`
- `print_header` (line 1398) `def print_header(self)`
- `validate_fine_structure` (line 1419) `def validate_fine_structure(self)` - *Validate fine structure energy corrections.*
- `validate_zitterbewegung` (line 1477) `def validate_zitterbewegung(self)` - *Validate Zitterbewegung simulation.*
- `validate_energy_spectrum` (line 1509) `def validate_energy_spectrum(self)` - *Validate complete energy spectrum.*
- `validate_orbital` (line 1524) `def validate_orbital(self, orbital_name, num_samples)` - *Validate single orbital visualization.*
- `run_full_validation` (line 1541) `def run_full_validation(self)` - *Run complete validation suite.*
- `interactive_mode` (line 1575) `def interactive_mode(self)` - *Run in interactive mode.*

#### `test_qc_integration.py`
**Path:** `test_qc_integration.py`

**Classes:**
- `TestIntegrationConfig` (line 70) `class TestIntegrationConfig` - *IntegrationConfig: centralised configuration with no hardcoded values.*
- `TestGateInstruction` (line 100) `class TestGateInstruction` - *GateInstruction: lightweight quantum gate descriptor.*
- `TestCircuitIR` (line 128) `class TestCircuitIR` - *CircuitIR: intermediate representation of a quantum circuit.*
- `TestOpenQasmAdapter` (line 165) `class TestOpenQasmAdapter` - *OpenQasmAdapter: OpenQASM 2.0 string <-> CircuitIR conversion.*
- `TestStandardCircuitFactory` (line 260) `class TestStandardCircuitFactory` - *StandardCircuitFactory: builds CircuitIR for common algorithms.*
- `TestIntegrationBridge` (line 292) `class TestIntegrationBridge` - *IntegrationBridge: facade for all format conversions.*
- `TestGateItem` (line 346) `class TestGateItem` - *GateItem: circuit builder gate representation.*
- `TestDashboardConfig` (line 372) `class TestDashboardConfig` - *DashboardConfig: centralised configuration for dashboard.*
- `TestVisualisationEngine` (line 400) `class TestVisualisationEngine` - *VisualisationEngine: renders figures from quantum state snapshots.*
- `TestSimulatorBackend` (line 441) `class TestSimulatorBackend` - *SimulatorBackend: lightweight wrapper around QC framework.*
- `TestSnapshotData` (line 473) `class TestSnapshotData` - *SnapshotData: quantum state snapshot for dashboard visualisation.*
- `TestSadPaths` (line 517) `class TestSadPaths` - *Edge cases and error conditions across all modules.*
- `TestEndToEnd` (line 585) `class TestEndToEnd` - *End-to-end scenarios combining multiple modules.*

**Functions:**
- `config` (line 42) `def config()`
- `bridge` (line 47) `def bridge(config)`
- `qasm_adapter` (line 52) `def qasm_adapter(config)`
- `bell_circuit` (line 57) `def bell_circuit()`
- `ghz_circuit` (line 62) `def ghz_circuit()`
- `test_default_config_has_supported_gates` (line 73) `def test_default_config_has_supported_gates(self, config)`
- `test_gate_name_map_is_complete` (line 79) `def test_gate_name_map_is_complete(self, config)`
- `test_reverse_gate_name_map_is_consistent` (line 83) `def test_reverse_gate_name_map_is_consistent(self, config)`
- `test_qasm_version_default` (line 87) `def test_qasm_version_default(self, config)`
- `test_max_qubits_defaults_are_positive` (line 90) `def test_max_qubits_defaults_are_positive(self, config)`
- `test_create_single_qubit_gate` (line 103) `def test_create_single_qubit_gate(self)`
- `test_create_two_qubit_gate` (line 109) `def test_create_two_qubit_gate(self)`
- `test_create_gate_with_params` (line 114) `def test_create_gate_with_params(self)`
- `test_targets_are_immutable` (line 118) `def test_targets_are_immutable(self)`
- `test_create_empty_circuit` (line 131) `def test_create_empty_circuit(self)`
- `test_append_gate` (line 136) `def test_append_gate(self)`
- `test_append_gate_out_of_range_raises` (line 141) `def test_append_gate_out_of_range_raises(self)`
- `test_multiple_gates` (line 146) `def test_multiple_gates(self)`
- `test_repr_includes_qubits_and_gates` (line 152) `def test_repr_includes_qubits_and_gates(self)`
- `test_export_bell_state_contains_header` (line 168) `def test_export_bell_state_contains_header(self, qasm_adapter, bell_circuit)`
- `test_export_bell_state_has_qreg_and_creg` (line 173) `def test_export_bell_state_has_qreg_and_creg(self, qasm_adapter, bell_circuit)`
- `test_export_bell_state_has_gates` (line 178) `def test_export_bell_state_has_gates(self, qasm_adapter, bell_circuit)`
- `test_export_ghz_state` (line 183) `def test_export_ghz_state(self, qasm_adapter, ghz_circuit)`
- `test_export_qft_has_swap` (line 189) `def test_export_qft_has_swap(self, qasm_adapter, config)`
- `test_export_parametric_gate` (line 194) `def test_export_parametric_gate(self, qasm_adapter)`
- `test_export_exceeds_max_qubits_raises` (line 201) `def test_export_exceeds_max_qubits_raises(self, qasm_adapter)`
- `test_import_bell_state_roundtrip` (line 206) `def test_import_bell_state_roundtrip(self, qasm_adapter, bell_circuit)`
- `test_import_ghz_roundtrip` (line 214) `def test_import_ghz_roundtrip(self, qasm_adapter, ghz_circuit)`
- `test_import_from_standard_qasm_string` (line 220) `def test_import_from_standard_qasm_string(self, qasm_adapter)`
- `test_import_with_parametric_gates` (line 234) `def test_import_with_parametric_gates(self, qasm_adapter)`
- `test_import_empty_qasm_returns_zero_qubit_circuit` (line 247) `def test_import_empty_qasm_returns_zero_qubit_circuit(self, qasm_adapter)`
- `test_bell_state_has_two_gates` (line 263) `def test_bell_state_has_two_gates(self)`
- `test_bell_state_has_two_qubits` (line 269) `def test_bell_state_has_two_qubits(self)`
- `test_ghz_state` (line 273) `def test_ghz_state(self)`
- `test_qft_three_qubits` (line 278) `def test_qft_three_qubits(self)`
- `test_grover_iterations` (line 283) `def test_grover_iterations(self)`
- `test_export_qasm_returns_string` (line 295) `def test_export_qasm_returns_string(self, bridge, bell_circuit)`
- `test_import_qasm_roundtrip` (line 300) `def test_import_qasm_roundtrip(self, bridge, bell_circuit)`
- `test_full_openqasm_roundtrip_bell` (line 306) `def test_full_openqasm_roundtrip_bell(self, bridge)`
- `test_full_openqasm_roundtrip_ghz` (line 313) `def test_full_openqasm_roundtrip_ghz(self, bridge)`
- `test_full_openqasm_roundtrip_qft` (line 320) `def test_full_openqasm_roundtrip_qft(self, bridge)`
- `test_qiskit_not_available_by_default` (line 327) `def test_qiskit_not_available_by_default(self, bridge)`
- `test_pennylane_not_available_by_default` (line 332) `def test_pennylane_not_available_by_default(self, bridge)`
- `test_export_qasm_custom_qreg_name` (line 337) `def test_export_qasm_custom_qreg_name(self, bridge, bell_circuit)`
- `test_create_single_qubit_gate` (line 349) `def test_create_single_qubit_gate(self)`
- `test_create_two_qubit_gate` (line 357) `def test_create_two_qubit_gate(self)`
- `test_create_parametric_gate` (line 362) `def test_create_parametric_gate(self)`
- `test_default_values` (line 375) `def test_default_values(self)`
- `test_gate_list_includes_standard_gates` (line 382) `def test_gate_list_includes_standard_gates(self)`
- `test_qasm_initial_contains_header` (line 389) `def test_qasm_initial_contains_header(self)`
- `test_engine_available_with_matplotlib` (line 403) `def test_engine_available_with_matplotlib(self)`
- `test_render_full_dashboard_returns_bytes` (line 415) `def test_render_full_dashboard_returns_bytes(self)`
- `test_synthetic_execute_returns_snapshots` (line 444) `def test_synthetic_execute_returns_snapshots(self)`
- `test_empty_circuit_returns_init_snapshot` (line 457) `def test_empty_circuit_returns_init_snapshot(self)`
- `test_create_snapshot` (line 476) `def test_create_snapshot(self)`
- `test_entropy_updates` (line 489) `def test_entropy_updates(self)`
- `test_probabilities_normalized` (line 500) `def test_probabilities_normalized(self)`
- `test_gate_instruction_empty_targets` (line 520) `def test_gate_instruction_empty_targets(self)`
- `test_circuit_ir_append_negative_qubit_raises` (line 524) `def test_circuit_ir_append_negative_qubit_raises(self)`
- `test_openqasm_import_empty_string` (line 529) `def test_openqasm_import_empty_string(self, qasm_adapter)`
- `test_openqasm_import_garbage_string` (line 534) `def test_openqasm_import_garbage_string(self, qasm_adapter)`
- `test_openqasm_export_zero_qubit_circuit` (line 538) `def test_openqasm_export_zero_qubit_circuit(self, qasm_adapter)`
- `test_standard_circuit_factory_qft_one_qubit` (line 543) `def test_standard_circuit_factory_qft_one_qubit(self)`
- `test_standard_circuit_factory_grover_minimal` (line 548) `def test_standard_circuit_factory_grover_minimal(self)`
- `test_circuit_ir_repr_no_gates` (line 552) `def test_circuit_ir_repr_no_gates(self)`
- `test_synthetic_snapshot_probabilities_sum_to_one` (line 557) `def test_synthetic_snapshot_probabilities_sum_to_one(self)`
- `test_visualisation_engine_handles_no_snapshots` (line 570) `def test_visualisation_engine_handles_no_snapshots(self)`
- `test_build_export_import_qasm_roundtrip` (line 588) `def test_build_export_import_qasm_roundtrip(self, bridge)`
- `test_qasm_to_circuitir_to_framework_mps` (line 598) `def test_qasm_to_circuitir_to_framework_mps(self, bridge)`
- `test_ghz_export_qasm_and_reimport_matches` (line 609) `def test_ghz_export_qasm_and_reimport_matches(self, bridge)`
- `test_qft_circuit_qasm_roundtrip` (line 616) `def test_qft_circuit_qasm_roundtrip(self, bridge)`
- `test_export_qasm_with_custom_names` (line 622) `def test_export_qasm_with_custom_names(self, bridge)`
- `test_import_qasm_preserves_gate_order` (line 628) `def test_import_qasm_preserves_gate_order(self, bridge)`
- `test_full_pipeline_build_export_import_to_mps` (line 644) `def test_full_pipeline_build_export_import_to_mps(self, bridge)`

#### `test_quantum_framework.py`
**Path:** `test_quantum_framework.py`

**Classes:**
- `TestFrameworkConfig` (line 72) `class TestFrameworkConfig` - *Tests for FrameworkConfig.*
- `TestMPSState` (line 95) `class TestMPSState` - *Tests for MPSState.*
- `TestBellState` (line 130) `class TestBellState` - *Tests for Bell state preparation.*
- `TestGHZState` (line 167) `class TestGHZState` - *Tests for GHZ state preparation.*
- `TestWState` (line 209) `class TestWState` - *Tests for W state preparation.*
- `TestQuantumGates` (line 261) `class TestQuantumGates` - *Tests for quantum gates.*
- `TestPhaseCoherence` (line 338) `class TestPhaseCoherence` - *Tests for phase coherence and unitarity.*
- `TestGroverAlgorithm` (line 403) `class TestGroverAlgorithm` - *Tests for Grover search algorithm.*
- `TestQFT` (line 431) `class TestQFT` - *Tests for Quantum Fourier Transform.*
- `TestMemoryScaling` (line 458) `class TestMemoryScaling` - *Tests for memory scaling.*
- `TestPrecisionMode` (line 501) `class TestPrecisionMode` - *Tests for precision mode.*
- `TestEdgeCases` (line 525) `class TestEdgeCases` - *Tests for edge cases.*

**Functions:**
- `config` (line 43) `def config()` - *Create default framework configuration.*
- `qc` (line 53) `def qc(config)` - *Create quantum computer instance.*
- `config_precision` (line 59) `def config_precision()` - *Create precision mode configuration.*
- `test_default_config` (line 75) `def test_default_config(self)` - *Test default configuration values.*
- `test_custom_config` (line 83) `def test_custom_config(self)` - *Test custom configuration values.*
- `test_create_state` (line 98) `def test_create_state(self, config)` - *Test state creation.*
- `test_initial_state` (line 104) `def test_initial_state(self, qc)` - *Test initial |00...0> state.*
- `test_clone_state` (line 113) `def test_clone_state(self, qc)` - *Test state cloning.*
- `test_bell_entropy` (line 133) `def test_bell_entropy(self, qc)` - *Test Bell state entropy.*
- `test_bell_probabilities` (line 141) `def test_bell_probabilities(self, qc)` - *Test Bell state probabilities.*
- `test_bell_entanglement` (line 154) `def test_bell_entanglement(self, qc)` - *Test Bell state entanglement.*
- `test_ghz_entropy` (line 170) `def test_ghz_entropy(self, qc)` - *Test GHZ state entropy.*
- `test_ghz_probabilities` (line 179) `def test_ghz_probabilities(self, qc)` - *Test GHZ state probabilities.*
- `test_ghz_scaling` (line 192) `def test_ghz_scaling(self, qc)` - *Test GHZ state memory scaling.*
- `test_w_state_probabilities` (line 212) `def test_w_state_probabilities(self, qc)` - *Test W state probabilities.*
- `test_w_state_entropy` (line 227) `def test_w_state_entropy(self, qc)` - *Test W state entropy.*
- `test_w_state_no_zero_probabilities` (line 247) `def test_w_state_no_zero_probabilities(self, qc)` - *Test that W state has non-zero probabilities for single-excitation states.*
- `test_hadamard_gate` (line 264) `def test_hadamard_gate(self, qc)` - *Test Hadamard gate.*
- `test_pauli_x_gate` (line 275) `def test_pauli_x_gate(self, qc)` - *Test Pauli-X gate.*
- `test_pauli_z_gate` (line 285) `def test_pauli_z_gate(self, qc)` - *Test Pauli-Z gate on |+> state.*
- `test_cnot_gate` (line 297) `def test_cnot_gate(self, qc)` - *Test CNOT gate.*
- `test_swap_gate` (line 309) `def test_swap_gate(self, qc)` - *Test SWAP gate.*
- `test_rotation_gates` (line 321) `def test_rotation_gates(self, qc)` - *Test rotation gates.*
- `test_hzh_equals_x` (line 341) `def test_hzh_equals_x(self, qc)` - *Test HZH = X identity.*
- `test_xx_equals_identity` (line 354) `def test_xx_equals_identity(self, qc)` - *Test XX = I identity.*
- `test_cnot_cnot_equals_identity` (line 366) `def test_cnot_cnot_equals_identity(self, qc)` - *Test CNOT CNOT = I identity.*
- `test_norm_preservation` (line 381) `def test_norm_preservation(self, qc)` - *Test that norm is preserved after gates.*
- `test_grover_3_qubits` (line 406) `def test_grover_3_qubits(self, qc)` - *Test Grover search on 3 qubits.*
- `test_grover_speedup` (line 417) `def test_grover_speedup(self, qc)` - *Test Grover speedup.*
- `test_qft_entropy` (line 434) `def test_qft_entropy(self, qc)` - *Test QFT entropy.*
- `test_memory_linear_scaling` (line 461) `def test_memory_linear_scaling(self)` - *Test that memory scales sub-exponentially with qubits (MPS property).*
- `test_compression_ratio` (line 478) `def test_compression_ratio(self)` - *Test MPS compression ratio vs full statevector for large n.*
- `test_precision_mode_config` (line 504) `def test_precision_mode_config(self)` - *Test precision mode configuration.*
- `test_precision_mode_bell_state` (line 510) `def test_precision_mode_bell_state(self, config_precision)` - *Test Bell state in precision mode.*
- `test_single_qubit` (line 528) `def test_single_qubit(self, qc)` - *Test single qubit operations.*
- `test_large_bond_dimension` (line 539) `def test_large_bond_dimension(self)` - *Test with large bond dimension.*
- `test_empty_circuit` (line 549) `def test_empty_circuit(self, qc)` - *Test empty circuit.*

#### `topological_hilbert_compression2.py`
**Path:** `topological_hilbert_compression2.py`

**Classes:**
- `HilbertPhase` (line 47) `class HilbertPhase(Enum)`
- `TopologicalCompressionConfig` (line 56) `class TopologicalCompressionConfig`
- `ITensorNetwork` (line 85) `class ITensorNetwork(ABC)`
- `MPSCore` (line 115) `class MPSCore` - *Matrix Product State core tensor A^{[k]}_{i_k} with left and right bond indices.
Shape: (chi_left, d, chi_right) where d=2 for qubits.*
- `MPSState` (line 162) `class MPSState(ITensorNetwork)` - *Matrix Product State representation of n-qubit quantum state.

|psi> = sum_{i_1...i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} ... A^{[n]}_{i_n} |i_1...i_n>

Memory: O(n * chi^2 * d) vs O(d^n) for full statevector.
For n=30, chi=16: ~30KB vs 8GB for statevector.*
- `VacuumCore` (line 351) `class VacuumCore` - *Vacuum Core architecture that projects irrelevant Hilbert subspace to zero.

Inspired by the HPU-Core achieving 99.996% sparsity:
- Active core: small subspace carrying quantum information
- Vacuum: 99%+ of Hilbert space forced to zero by regularization
- Protection: topological invariants prevent core destruction*
- `TopologicalProtector` (line 411) `class TopologicalProtector` - *Provides topological protection for quantum states via:
- Winding number monitoring
- Berry phase calculation
- Edge state preservation*
- `HybridBackend` (line 452) `class HybridBackend(ABC)`
- `DirectBackend` (line 466) `class DirectBackend(HybridBackend)` - *Direct tensor backend using JointHilbertState representation.
Limited to config.max_qubits_direct qubits due to exponential memory.*
- `MPSBackend` (line 515) `class MPSBackend(HybridBackend)` - *Matrix Product State backend for scalable quantum simulation.
Handles up to config.max_qubits_mps qubits with sub-exponential memory.*
- `TopologicalHilbertSimulator` (line 572) `class TopologicalHilbertSimulator` - *Main simulator implementing hybrid architecture for scalable quantum simulation.

Architecture:
    - n <= max_qubits_direct: HPU-Core direct tensor (exact wavefunction evolution)
    - n > max_qubits_direct: MPS with vacuum core compression*
- `Schrodinger20Experiment` (line 723) `class Schrodinger20Experiment` - *Validates topological compression on 20-qubit molecular ground state.

Target: H2O (10 electrons, ~20 qubits)
Metrics:
    - Energy error < 1 mHa vs Qiskit VQE
    - Inference time < 100ms
    - Memory < 50MB (vs 500MB for direct tensor)*

**Functions:**
- `main` (line 956) `def main()`
- `__post_init__` (line 80) `def __post_init__(self)`
- `amplitude` (line 87) `def amplitude(self, basis_index)`
- `apply_single_qubit_gate` (line 91) `def apply_single_qubit_gate(self, qubit, gate)`
- `apply_two_qubit_gate` (line 95) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)`
- `norm` (line 99) `def norm(self)`
- `probabilities` (line 103) `def probabilities(self)`
- `entropy` (line 107) `def entropy(self)`
- `memory_bytes` (line 111) `def memory_bytes(self)`
- `__init__` (line 121) `def __init__(self, chi_left, chi_right, d, device, dtype)`
- `_initialize` (line 130) `def _initialize(self)`
- `tensor` (line 136) `def tensor(self)`
- `tensor` (line 142) `def tensor(self, value)`
- `left_canonicalize` (line 147) `def left_canonicalize(self)`
- `right_canonicalize` (line 154) `def right_canonicalize(self)`
- `__init__` (line 172) `def __init__(self, n_qubits, config)`
- `_initialize` (line 181) `def _initialize(self)`
- `_bond_dimension` (line 195) `def _bond_dimension(self, site)`
- `amplitude` (line 198) `def amplitude(self, basis_index)`
- `apply_single_qubit_gate` (line 209) `def apply_single_qubit_gate(self, qubit, gate)`
- `apply_two_qubit_gate` (line 219) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)`
- `_swap_qubits_in_gate` (line 236) `def _swap_qubits_in_gate(self, gate)`
- `_apply_adjacent_gate` (line 245) `def _apply_adjacent_gate(self, qubit, gate)`
- `_apply_nonadjacent_gate` (line 285) `def _apply_nonadjacent_gate(self, qubit_a, qubit_b, gate)`
- `norm` (line 294) `def norm(self)`
- `_canonicalize` (line 301) `def _canonicalize(self)`
- `probabilities` (line 308) `def probabilities(self)`
- `entropy` (line 318) `def entropy(self)`
- `memory_bytes` (line 329) `def memory_bytes(self)`
- `entanglement_entropy` (line 335) `def entanglement_entropy(self, cut)`
- `__init__` (line 361) `def __init__(self, n_qubits, config)`
- `_initialize` (line 370) `def _initialize(self)`
- `_compute_berry_phases` (line 375) `def _compute_berry_phases(self)`
- `add_active_state` (line 381) `def add_active_state(self, basis_index, winding_number)`
- `_compute_winding_number` (line 389) `def _compute_winding_number(self, basis_index)`
- `is_topologically_protected` (line 394) `def is_topologically_protected(self, basis_index)`
- `sparsity` (line 398) `def sparsity(self)`
- `project_to_active` (line 403) `def project_to_active(self, state)`
- `__init__` (line 419) `def __init__(self, config)`
- `compute_winding_number` (line 424) `def compute_winding_number(self, state, qubit)`
- `compute_berry_phase` (line 433) `def compute_berry_phase(self, state, qubit_a, qubit_b)`
- `is_protected` (line 444) `def is_protected(self, state, vacuum_core)`
- `can_handle` (line 454) `def can_handle(self, n_qubits)`
- `create_state` (line 458) `def create_state(self, n_qubits)`
- `apply_gate` (line 462) `def apply_gate(self, state, gate_name, targets, params)`
- `__init__` (line 472) `def __init__(self, config)`
- `_load_quantum_computer` (line 479) `def _load_quantum_computer(self)`
- `can_handle` (line 497) `def can_handle(self, n_qubits)`
- `create_state` (line 500) `def create_state(self, n_qubits)`
- `apply_gate` (line 507) `def apply_gate(self, state, gate_name, targets, params)`
- `__init__` (line 521) `def __init__(self, config)`
- `_initialize_gate_cache` (line 526) `def _initialize_gate_cache(self)`
- `can_handle` (line 537) `def can_handle(self, n_qubits)`
- `create_state` (line 540) `def create_state(self, n_qubits)`
- `apply_gate` (line 543) `def apply_gate(self, state, gate_name, targets, params)`
- `__init__` (line 581) `def __init__(self, config)`
- `_select_backend` (line 590) `def _select_backend(self, n_qubits, force_mps)`
- `create_circuit` (line 602) `def create_circuit(self, n_qubits, force_mps)`
- `h` (line 610) `def h(self, qubit)`
- `x` (line 613) `def x(self, qubit)`
- `y` (line 616) `def y(self, qubit)`
- `z` (line 619) `def z(self, qubit)`
- `rx` (line 622) `def rx(self, qubit, theta)`
- `ry` (line 625) `def ry(self, qubit, theta)`
- `rz` (line 628) `def rz(self, qubit, theta)`
- `cnot` (line 631) `def cnot(self, control, target)`
- `cz` (line 634) `def cz(self, control, target)`
- `swap` (line 637) `def swap(self, qubit_a, qubit_b)`
- `run` (line 640) `def run(self)`
- `probabilities` (line 650) `def probabilities(self)`
- `entropy` (line 655) `def entropy(self)`
- `memory_usage` (line 666) `def memory_usage(self)`
- `compression_ratio` (line 682) `def compression_ratio(self)`
- `detect_phase` (line 691) `def detect_phase(self)`
- `_compute_average_bond_dimension` (line 714) `def _compute_average_bond_dimension(self)`
- `__init__` (line 734) `def __init__(self, config)`
- `run_bell_state` (line 739) `def run_bell_state(self, n_qubits, use_mps)`
- `run_ghz_state` (line 758) `def run_ghz_state(self, n_qubits, use_mps)`
- `run_w_state` (line 777) `def run_w_state(self, n_qubits, use_mps)`
- `prepare_ghz_state` (line 798) `def prepare_ghz_state(self, n_qubits, force_mps)` - *Prepare GHZ state directly without SWAP overhead.

GHZ: |00...0> + |11...1> normalized.
MPS representation:
- A^{[0]}_{i_0,α_1}: A[0,0,0]=1/√2, A[0,1,1]=1/√2, shape (1,2,2)
- A^{[k]}_{α_k,i_k,α_{k+1}}: A[0,0,0]=A[1,1,1]=1, shape (2,2,2)
- A^{[n-1]}_{α_{n-1},i_{n-1},0}: A[0,0,0]=A[1,1,0]=1, shape (2,2,1)*
- `run_scaling_benchmark` (line 844) `def run_scaling_benchmark(self, max_qubits, use_mps)`
- `run_all` (line 897) `def run_all(self)`
- `_print_summary` (line 909) `def _print_summary(self)`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
