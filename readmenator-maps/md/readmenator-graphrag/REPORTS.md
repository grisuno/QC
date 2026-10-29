# GraphRAG Community Reports

Entities: 2153 | Relationships: 5807 | Communities: 5 | Themes: 3 | Text units: 2026

Query with `readmenator . ask "<question>"` (local: BM25 + Personalized PageRank; global: map-reduce over these reports) or the MCP tool `readmenator.graphrag`.

## Project overview (`root`, root, rating 7.0)

43 files in 5 communities and 3 themes. Highest-impact communities: root: quantum_simulator (7.0), immersive (3.2), root: quantum_framework_core (2.5). God nodes: quantum_computer.py, quantum_framework_core.py, qc_dashboard.py, quantum_framework_menu.py, quantum_simulator.py.

- [root: quantum_simulator] rating 7.0: `quantum_framework_physics.py` ranks 1 by PageRank, 3 importers, 79 symbols: Quantum Framework Physics Module (canonical)
- [immersive] rating 3.2: `immersive/physics.py` ranks 1 by PageRank, 5 importers, 34 symbols: Physics kernels for the immersive viewer: hydrogenic tables, angular constants, qubit registers.
- [root: quantum_framework_core] rating 2.5: `quantum_framework_core.py` ranks 1 by PageRank, 14 importers, 157 symbols: Quantum Framework Core Module
- [unassigned files] rating 0.5: `immersive/matter.py` ranks 1 by PageRank, 0 importers, 19 symbols: Ab-initio fields for atoms and molecules (PySCF) sampled on a cubic grid for GPU volume rendering.
- [root: quantum_framework_molecular_v2] rating 0.4: `quantum_framework_molecular_v2.py` ranks 1 by PageRank, 2 importers, 63 symbols: Quantum Framework Molecular Module - Production Version
- Key entities: file:quantum_simulator.py, file:quantum_computer.py, file:immersive/scenes.py, file:immersive/backend.py, file:quantum_framework_core.py, sym:qc_integration.py::append@156
- Children: t0, t1, t2
- Root rating = highest community rating.

## root: quantum_simulator + immersive +1 (`t0`, theme, rating 7.0)

Theme of 3 communities and 38 files: root: quantum_simulator (16 files, rating 7.0); immersive (12 files, rating 3.2); root: quantum_framework_core (10 files, rating 2.5).

- [root: quantum_simulator] `quantum_framework_physics.py` ranks 1 by PageRank, 3 importers, 79 symbols: Quantum Framework Physics Module (canonical)
- [root: quantum_simulator] `quantum_computer.py` ranks 2 by PageRank, 21 importers, 123 symbols: Author: Gris Iscomeback License: AGPL v3  Collapse-Free Quantum Computer Simulator on Classical Hardware.
- [immersive] `immersive/physics.py` ranks 1 by PageRank, 5 importers, 34 symbols: Physics kernels for the immersive viewer: hydrogenic tables, angular constants, qubit registers.
- [immersive] `immersive/glkit.py` ranks 2 by PageRank, 4 importers, 30 symbols: Thin OpenGL helpers: program compilation, uniforms, fullscreen pass, orbit camera.
- [root: quantum_framework_core] `quantum_framework_core.py` ranks 1 by PageRank, 14 importers, 157 symbols: Quantum Framework Core Module
- [root: quantum_framework_core] `quantum_framework_molecular.py` ranks 2 by PageRank, 3 importers, 33 symbols: Quantum Framework Molecular Module
- Key entities: file:quantum_simulator.py, file:quantum_computer.py, file:immersive/scenes.py, file:immersive/backend.py, file:quantum_framework_core.py, sym:qc_integration.py::append@156
- Children: c0, c1, c2
- Theme rating = highest child community rating.

## unassigned files (`t1`, theme, rating 0.5)

Theme of 1 communities and 3 files: unassigned files (3 files, rating 0.5).

- [unassigned files] `immersive/matter.py` ranks 1 by PageRank, 0 importers, 19 symbols: Ab-initio fields for atoms and molecules (PySCF) sampled on a cubic grid for GPU volume rendering.
- [unassigned files] `install.sh` ranks 2 by PageRank, 0 importers, 0 symbols.
- Key entities: file:qc_launcher.py, file:immersive/matter.py
- Children: c4
- Theme rating = highest child community rating.

## root: quantum_framework_molecular_v2 (`t2`, theme, rating 0.4)

Theme of 1 communities and 2 files: root: quantum_framework_molecular_v2 (2 files, rating 0.4).

- [root: quantum_framework_molecular_v2] `quantum_framework_molecular_v2.py` ranks 1 by PageRank, 2 importers, 63 symbols: Quantum Framework Molecular Module - Production Version
- [root: quantum_framework_molecular_v2] `demo_molecular_vqe.py` ranks 2 by PageRank, 0 importers, 10 symbols: Quantum Framework Demo - Production Version
- Key entities: file:quantum_framework_molecular_v2.py, file:demo_molecular_vqe.py
- Children: c3
- Theme rating = highest child community rating.

## root: quantum_simulator (`c0`, community, rating 7.0)

16 files under root (py 16), mostly utility. Core file quantum_framework_physics.py (PageRank 0.1693, imported by 3 files): Quantum Framework Physics Module (canonical) Key abstractions: SpectralLayer, HamiltonianBackboneNet, SchrodingerSpectralNet, DiracSpectralNet, GammaMatrices, IPhysicsBackend. Depends on root: quantum_framework_core (2). Used by root: quantum_framework_core (9), immersive (2).

- `quantum_framework_physics.py` ranks 1 by PageRank, 3 importers, 79 symbols: Quantum Framework Physics Module (canonical)
- `quantum_computer.py` ranks 2 by PageRank, 21 importers, 123 symbols: Author: Gris Iscomeback License: AGPL v3  Collapse-Free Quantum Computer Simulator on Classical Hardware.
- `molecular_sim.py` ranks 3 by PageRank, 6 importers, 35 symbols: VERSIÓN CON OPENFERMION CORREGIDO
- Hotspot `quantum_computer.py`: 123 symbols, 34 connections (score 0.58).
- Hotspot `quantum_simulator.py`: 180 symbols, 18 connections (score 0.56).
- Taint: 6 paths reach this group via urllib.request.
- Surprising bridge: __init__.py <-> quantum_simulator.py (5 hops across communities).
- Key entities: file:quantum_simulator.py, file:quantum_computer.py, file:topological_hilbert_compression2.py, file:quantum_framework_physics.py, file:quantum_dash.py, file:relativistic_hydrogen.py, sym:molecular_sim.py::get@162, file:advanced_experiments.py
- Rating 7.0/10 = 7 x PageRank share 1.00 + 3 x risk 0.00. Internal imports: 33.

## immersive (`c1`, community, rating 3.2)

12 files under immersive (py 12), mostly utility. Core file immersive/physics.py (PageRank 0.0410, imported by 5 files): Physics kernels for the immersive viewer: hydrogenic tables, angular constants, qubit registers. Key abstractions: OrbitalPreset, Gate, QubitRegister, unit_vector_poly_args, angular_constants, ylm_from_table. Depends on root: quantum_framework_core (3), root: quantum_simulator (2).

- `immersive/physics.py` ranks 1 by PageRank, 5 importers, 34 symbols: Physics kernels for the immersive viewer: hydrogenic tables, angular constants, qubit registers.
- `immersive/glkit.py` ranks 2 by PageRank, 4 importers, 30 symbols: Thin OpenGL helpers: program compilation, uniforms, fullscreen pass, orbit camera.
- `immersive/app.py` ranks 3 by PageRank, 2 importers, 20 symbols: Immersive viewer entry: window, event loop, cinematic tour director.
- Hotspot `immersive/scenes.py`: 73 symbols, 17 connections (score 0.32).
- Hotspot `immersive/backend.py`: 35 symbols, 24 connections (score 0.30).
- Surprising bridge: __init__.py <-> orbital_visualizer2.py (5 hops across communities).
- Key entities: file:immersive/scenes.py, file:immersive/backend.py, file:immersive/physics.py, file:test_immersive.py, file:immersive/app.py, file:immersive/spin.py, file:immersive/glkit.py, file:immersive/hud.py
- Rating 3.2/10 = 7 x PageRank share 0.46 + 3 x risk 0.00. Internal imports: 24.

## root: quantum_framework_core (`c2`, community, rating 2.5)

10 files under root (py 10), mostly utility. Core file quantum_framework_core.py (PageRank 0.0481, imported by 14 files): Quantum Framework Core Module Key abstractions: HilbertPhase, FrameworkConfig, AtomData, MoleculeData, OrbitalData, ConfigLoader. Depends on root: quantum_simulator (9). Used by immersive (3), root: quantum_simulator (2).

- `quantum_framework_core.py` ranks 1 by PageRank, 14 importers, 157 symbols: Quantum Framework Core Module
- `quantum_framework_molecular.py` ranks 2 by PageRank, 3 importers, 33 symbols: Quantum Framework Molecular Module
- `qc_dashboard.py` ranks 3 by PageRank, 15 importers, 88 symbols: QC Dashboard - Streamlit web application for real-time quantum playground.
- Hotspot `qc_dashboard.py`: 88 symbols, 66 connections (score 0.80).
- Hotspot `quantum_framework_core.py`: 157 symbols, 31 connections (score 0.63).
- Surprising bridge: __init__.py <-> orbital_visualizer2.py (5 hops across communities).
- Key entities: file:quantum_framework_core.py, sym:qc_integration.py::append@156, file:qc_dashboard.py, file:test_qc_integration.py, file:quantum_lab.py, file:test_quantum_framework.py, file:qc_integration.py, file:quantum_framework_molecular.py
- Rating 2.5/10 = 7 x PageRank share 0.36 + 3 x risk 0.00. Internal imports: 35.

## unassigned files (`c4`, community, rating 0.5)

3 files under root (py 2, sh 1), mostly utility. Core file immersive/matter.py (PageRank 0.0111, imported by 0 files): Ab-initio fields for atoms and molecules (PySCF) sampled on a cubic grid for GPU volume rendering. Key abstractions: System, OrbitalInfo, Solved, AsyncSolver, pyscf_available, solve.

- `immersive/matter.py` ranks 1 by PageRank, 0 importers, 19 symbols: Ab-initio fields for atoms and molecules (PySCF) sampled on a cubic grid for GPU volume rendering.
- `install.sh` ranks 2 by PageRank, 0 importers, 0 symbols.
- `qc_launcher.py` ranks 3 by PageRank, 0 importers, 13 symbols: Q2C launcher: a Textual front door to the immersive GPU viewer and the rich Quantum Lab.
- Hotspot `qc_launcher.py`: 13 symbols, 11 connections (score 0.13).
- Hotspot `immersive/matter.py`: 19 symbols, 9 connections (score 0.12).
- Taint: 1 paths reach this group via subprocess.
- Key entities: file:qc_launcher.py, file:immersive/matter.py, sym:immersive/matter.py::solve@110, sym:immersive/matter.py::field_amplitude@160, sym:qc_launcher.py::refresh_detail@136, sym:qc_launcher.py::action_launch@167, sym:qc_launcher.py::t@111, sym:immersive/matter.py::prefetch@236
- Rating 0.5/10 = 7 x PageRank share 0.06 + 3 x risk 0.00. Internal imports: 0.

## root: quantum_framework_molecular_v2 (`c3`, community, rating 0.4)

2 files under root (py 2), mostly utility. Core file quantum_framework_molecular_v2.py (PageRank 0.0206, imported by 2 files): Quantum Framework Molecular Module - Production Version Key abstractions: MolecularConfig, MoleculeData, MoleculeBuilder, PseudoMolData, HamiltonianBuilder, CachedPauliOperation.

- `quantum_framework_molecular_v2.py` ranks 1 by PageRank, 2 importers, 63 symbols: Quantum Framework Molecular Module - Production Version
- `demo_molecular_vqe.py` ranks 2 by PageRank, 0 importers, 10 symbols: Quantum Framework Demo - Production Version
- Hotspot `quantum_framework_molecular_v2.py`: 63 symbols, 24 connections (score 0.36).
- Hotspot `demo_molecular_vqe.py`: 10 symbols, 11 connections (score 0.12).
- Key entities: file:quantum_framework_molecular_v2.py, file:demo_molecular_vqe.py, sym:quantum_framework_molecular_v2.py::run@1202, sym:quantum_framework_molecular_v2.py::__init__@485, sym:demo_molecular_vqe.py::print_header@47, sym:quantum_framework_molecular_v2.py::run_vqe@1349, sym:quantum_framework_molecular_v2.py::build@209, sym:quantum_framework_molecular_v2.py::scan_parameter_space@658
- Rating 0.4/10 = 7 x PageRank share 0.06 + 3 x risk 0.00. Internal imports: 2.
