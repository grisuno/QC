# Quasi Quantum Computing — Q²C v2

**Author:** grisun0 · ORCID: 0009-0002-7622-3916
**DOI:** 10.5281/zenodo.18795538
**License:** AGPL v3

> A functional quasi-quantum computing simulator on classical hardware using physics models as a backend, now extended with Matrix Product State (MPS) tensor networks for scalable simulation.

---

## Screenshots:

<img width="3521" height="2985" alt="image" src="https://github.com/user-attachments/assets/761270b6-58a6-4b42-9c57-3f7038af9d85" />

<img width="3374" height="2985" alt="image" src="https://github.com/user-attachments/assets/5bf2e34a-2552-4e02-8b8f-2d2b3deb3dd9" />

<img width="3497" height="2985" alt="image" src="https://github.com/user-attachments/assets/93d9d5ef-d7e8-4f3d-8db5-2e1d56fe0bb5" />

<img width="3461" height="2985" alt="image" src="https://github.com/user-attachments/assets/3a47d377-bc2c-4d63-ba67-929e0bd2db56" />


## What is Q²C?

Q²C simulates quantum systems on classical hardware through two complementary modes selectable at runtime:

| Mode | Backend | Precision | Max Qubits | When to use |
|------|---------|-----------|-----------|-------------|
| **Direct (exact)** | Full statevector | Machine ε (~1e-15) | ~14 | Need exact amplitudes |
| **MPS (scalable)** | Tensor network | Tunable (chi) | 33+ | Need more qubits |

Both modes share the same gate library, algorithms, and physics backends. The user chooses via `precision_mode` in `quantum_framework_config.toml` or at the interactive menu prompt.

---

## Architecture

```
quantum_framework_main.py        ← CLI entry point
quantum_lab.py                   ← 🎓 Educational TUI (bilingual EN/ES)
quantum_framework_menu.py        ← Interactive menu (11 sections)
    │
    ├── quantum_framework_core.py     ← MPS tensor network engine
    │     ├── MPSState / MPSCore      ← O(n·χ²) memory representation
    │     ├── QuantumCircuit          ← Gate builder (H, X, Y, Z, CNOT, CZ, SWAP, Rx/Ry/Rz, ...)
    │     └── MPSQuantumComputer      ← Bell, GHZ, W-state, Grover, QFT
    │
    ├── quantum_framework_physics.py  ← Neural physics backends
    │     ├── HamiltonianBackboneNet  ← Spectral FFT Hamiltonian network
    │     ├── SchrodingerSpectralNet  ← Wavefunction evolution network
    │     └── DiracSpectralNet        ← Relativistic 8-channel spinor network
    │
    ├── quantum_framework_molecular*.py  ← VQE molecular simulations
    │     ├── UCCSD ansatz (Jordan-Wigner)
    │     └── Molecules: H2, LiH, H2O
    │
    ├── quantum_computer.py           ← Legacy statevector simulator (exact)
    │     └── Backends: Hamiltonian, Schrodinger, Dirac (float32)
    │
    ├── higgs_four_lepton_analysis.py ← Higgs → ZZ* → 4l on CMS Open Data
    ├── quantum_visualizer.py         ← Probability / Bloch / phase visualizer
    ├── quantum_dash.py               ← Brutalist matplotlib + Plotly visualizer
    ├── quantum_3dview.py             ← 3D holographic Plotly dashboard + audio
    ├── app.py                        ← H2 polarizability VQE (Stark effect)
    │
    ├── relativistic_hydrogen.py      ← Dirac equation, fine structure, Zitterbewegung
    ├── entangled_hydrogen.py         ← Entangled hydrogen orbital simulations
    ├── topological_hilbert_compression2.py  ← Topological Hilbert space analysis
    ├── advanced_experiments.py       ← Grover, QFT, phase estimation, Simon
    └── test_quantum_framework.py     ← 34 pytest tests (100% pass rate)
```

---

## Key Results

### Quantum Algorithms (both modes)
| Algorithm | Result |
|-----------|--------|
| Bell state entropy | 1.000000 bits (exact) |
| GHZ state (n=3) | 50/50 distribution |
| Grover 3-qubit search | 94.53% marked-state probability |
| QFT norm preservation | sum(P) = 1 to machine precision |
| Phase coherence (HZH=X, XX=I, etc.) | 22/22 tests pass |

### Molecular Chemistry
| Molecule | HF Energy (Ha) | VQE Energy (Ha) | FCI Ref (Ha) | Corr. recovered |
|----------|---------------|----------------|-------------|----------------|
| H2 | −1.1168 | −1.1373060358 | −1.1373060358 | **100%** |
| LiH | −7.87 | — | −7.88 | — |
| H2O | −74.963 | — | −75.012 | — |

### Polarizability (H2, Stark effect)
- α = **2.750 a₀³** — exact match with STO-3G diagonalization
- |E(+F) − E(−F)| at machine precision (10⁻¹⁵) across all field values

### QED Effects
- Anomalous magnetic moment (g−2)/2 to 5th order: **0.3% error** vs experiment
- Lamb shift: 57.31 MHz (perturbative; experimental: 1057.84 MHz)

### MPS Scaling (new in v2)
| Qubits | MPS Memory | Full statevector | Compression |
|--------|-----------|-----------------|-------------|
| 10 | 21 KB | 16 KB | — (chi=32 saturated) |
| 20 | 182 KB | 16 MB | **~90×** |
| 33 | ~350 KB | 128 GB | **~370,000×** |

---

## Quick Start

### 🎓 Quantum Lab — educational TUI (start here if you are new)

```bash
python3 quantum_framework_main.py --learn          # or: python3 quantum_lab.py
python3 quantum_lab.py --lang es                   # interfaz en español
python3 quantum_lab.py --lesson 3                  # jump to lesson 3
```

A bilingual (English/Spanish) interactive terminal experience built on the
real simulation engine — nothing is pre-recorded:

| Section | What you do |
|---------|-------------|
| **Lessons** | 5-lesson guided course: qubits & superposition, measurement, entanglement & Bell states, Grover search, quantum chemistry. Live simulations + quizzes. |
| **Playground** | Build circuits gate by gate (`h 0`, `cnot 0 1`, `ry 0 pi/3`, ...) and watch probability bars, amplitudes/phases and entanglement entropy update live. |
| **Chemistry lab** | Molecule cards (H2, LiH, H2O) with HF/FCI/correlation energies, and a colored ASCII hydrogen-orbital viewer (1s → 3d) showing lobes, signs and nodes. |
| **Glossary** | Quick reference for all the vocabulary used in the lessons. |

Requires only `rich` on top of the core dependencies.

### Interactive Menu
```bash
python3 quantum_framework_main.py
```

### Run all experiments (non-interactive)
```bash
python3 quantum_framework_main.py --run-all
```

### Run tests
```bash
pytest test_quantum_framework.py -v
```

### CLI options
```bash
python3 quantum_framework_main.py --help
python3 quantum_framework_main.py --learn --lang es
python3 quantum_framework_main.py --benchmark --max-qubits 20
python3 quantum_framework_main.py --info
```

---

## Menu Sections

| # | Section | Contents |
|---|---------|----------|
| 1 | Quantum Circuits | Custom circuits, Bell, GHZ, W-state, gate demos |
| 2 | Entanglement | Entropy scaling, cut-position analysis, heatmaps |
| 3 | Molecular Simulations | VQE H2/LiH/H2O, energy landscape, polarizability |
| 4 | Orbital Visualization | Hydrogen orbitals, 3D scatter, probability distributions |
| 5 | Relativistic Physics | Dirac equation, fine structure, Zitterbewegung |
| 6 | QED Effects | Lamb shift, anomalous magnetic moment, vacuum polarization |
| 7 | Advanced Algorithms | Grover, QFT, phase estimation, Simon, teleportation |
| 8 | Benchmarks | MPS scaling, memory efficiency, timing |
| 9 | Configuration | Atoms, molecules, parameters |
| 10 | **Particle Physics** | Higgs → 4 lepton on CMS Open Data + quantum backends |
| 11 | **Quantum Visualization** | Brutalist matplotlib/Plotly, 3D holographic dashboard |

---

## Configuration

Edit `quantum_framework_config.toml`:

```toml
[simulation]
precision_mode = false     # true = exact statevector, false = MPS
max_qubits = 33            # max qubits for MPS mode
max_qubits_direct = 14     # max qubits for direct mode

[mps]
bond_dimension = 16        # χ — higher = more precise, more memory
max_bond_dimension = 64    # maximum χ
svd_threshold = 1.0e-10    # truncation threshold
adaptive_bond = true
```

---

## Neural Network Checkpoints

Three pre-trained physics networks are included:

| File | Network | Size | Purpose |
|------|---------|------|---------|
| `weights/latest.pth` | HamiltonianBackboneNet | 18.9 MB | Spectral Hamiltonian operator |
| `weights/schrodinger_crystal_final.pth` | SchrodingerSpectralNet | 18.9 MB | Wavefunction time evolution |
| `weights/dirac_phase5_latest.pth` | DiracSpectralNet | 18.9 MB | Relativistic spinor dynamics |

All networks use FFT-based spectral convolution layers (`SpectralLayer`) operating in Fourier space.

---

## Dependencies

```bash
pip install -r requirements.txt
```

| Package | Required | Purpose |
|---------|----------|---------|
| torch | ✅ | MPS tensors, neural backends |
| numpy | ✅ | Numerical computation |
| scipy | ✅ | VQE optimizer (L-BFGS-B) |
| matplotlib | ✅ | Visualization |
| tomllib/tomli | ✅ | TOML config (Python 3.11+ built-in) |
| plotly | ⚡ Optional | 3D holographic visualization |
| openfermion | ⚡ Optional | Molecular Hamiltonians |
| pyscf | ⚡ Optional | Ab initio molecular integrals |
| pytest | 🔧 Dev | Test suite |

---

## Particle Physics Module

The `higgs_four_lepton_analysis.py` module downloads real CMS Open Data from CERN and processes H → ZZ* → 4ℓ events through the quantum neural backends:

```bash
python3 higgs_four_lepton_analysis.py
```

This will:
1. Download 6 CSV files from the CERN Open Data Portal (~100 MB)
2. Reconstruct 4-lepton invariant mass spectra
3. Identify Higgs candidates in the 120–130 GeV window
4. Evolve each lepton's wavefunction through DiracBackend, HamiltonianBackend, SchrodingerBackend
5. Generate a Plotly 3D detector visualization with quantum-modulated tracks

---

## Differences from v1 (GitHub original)

| Feature | v1 | v2 (this) |
|---------|-----|-----------|
| State representation | Full statevector (2^n, 2, G, G) | **MPS** O(n·χ²) + exact mode |
| Numerical precision | float32 | **float64** |
| Max qubits | 8 | **33+** (MPS) / 14 (exact) |
| Architecture | Monolithic files | **Modular** (7 framework modules) |
| Tests | None | **34 pytest tests (100% pass)** |
| Particle physics | ❌ | **Higgs 4-lepton + CMS data** |
| 3D visualization | ❌ | **Holographic Plotly dashboard** |
| Polarizability VQE | Separate script | **Integrated in menu** |
| Grover (framework) | via advanced_experiments | **run_grover_search() in core** |
| Configuration | Basic TOML | **Extended TOML (9 sections)** |

---

## Limitations

- MPS oracle for multi-controlled gates is circuit-approximate; `run_grover_search()` uses statevector for correctness
- GPU execution untested (CPU only)
- Higgs analysis requires internet connection for CMS data download
- PySCF/OpenFermion required for full molecular VQE (graceful fallback otherwise)
- Polarizability (app.py) tested only for H2 STO-3G

---

## Citation

```bibtex
@software{grisuno_qc2026,
  author  = {grisun0},
  title   = {Q²C: Quasi Quantum Computing Simulator v2},
  year    = {2026},
  doi     = {10.5281/zenodo.18795538},
  url     = {https://github.com/grisuno/QC},
  orcid   = {0009-0002-7622-3916}
}
```
