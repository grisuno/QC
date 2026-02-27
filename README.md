# Quasi Quantum Computing

## Usage

```bash
python3 molecular_sim.py --molecule H2
```

```bash
python3 quantum_computer.py \
  --hamiltonian-checkpoint hamiltonian.pth \
  --schrodinger-checkpoint checkpoint_phase3_training_epoch_18921_20260224_154739.pth \
  --dirac-checkpoint       best_dirac.pth \
  --grid-size 16 \
  --hidden-dim 32 \
  --expansion-dim 64 \
  --device cpu

```

## Chapter QC.

This chapter is not about capability. It is about exposure.

I fixed the execution parameters deliberately. Grid size sixteen. Hidden dimension thirty-two. Expansion dimension sixty-four. CPU only. No stochasticity. No device noise. Three backends loaded independently from disk: Hamiltonian, Schrödinger, Dirac. Each checkpoint was restored without shared memory, without shared caches, without synchronized initialization beyond numerical libraries.

The system announced itself plainly: joint Hilbert space, no abstraction layer hiding state evolution. What followed was not a benchmark run but a constraint sweep.

The first observation was temporal stability. Across the entire execution window, no backend reloaded weights mid-run, no lazy initialization occurred after the first measurement, and no backend diverged in wall-clock behavior. This matters because silent reinitialization is a common source of false consistency in hybrid simulators. Here, the log timestamps show continuous execution with no discontinuities.

The Bell state preparation was identical across backends. The Hamiltonian backend reported P(|00>) = 0.5000 and P(|11>) = 0.5000 with Shannon entropy exactly 1.0000 bits. The Schrödinger backend produced the same distribution. The Dirac backend did as well. The most probable state differed only by ordering, never by value.

More importantly, the per-qubit marginals were symmetric: P(|1>) = 0.5000 for each qubit, with ⟨Z⟩ = 0.0000. The Bloch vectors collapsed to the origin. This indicates not only entanglement, but absence of bias introduced by basis ordering or backend-specific normalization.

The GHZ three-qubit state extended this behavior without degradation. Again, the probability mass was split exactly between |000> and |111>. The entropy remained at 1.0000 bit, not increasing spuriously with qubit count. Each marginal remained maximally mixed. No backend introduced asymmetry across qubits.

At this point, agreement is expected. The deviation appears when phase-sensitive algorithms are introduced.

The Deutsch–Jozsa constant oracle produced |000> with probability 0.5000 and |001> with probability 0.5000. This is not a mistake. The ancilla qubit remained in superposition while the input register collapsed deterministically. The per-qubit marginals confirmed this: q0 and q1 had P(|1>) = 0.0000 with ⟨Z⟩ = +1.0000, while q2 remained unbiased. This pattern was identical across all three backends.

The balanced oracle inverted this behavior. The most probable state shifted to |100> with probability 0.5000. Again, entropy remained at 1.0000 bit. The first qubit flipped deterministically, the second remained fixed, the third stayed unbiased. This is the expected interference pattern. No leakage was observed.

The quantum Fourier transform on three qubits is where simulators often diverge. Here, the measured distribution was uniform: eight basis states each at probability 0.1250. The Shannon entropy measured exactly 3.0000 bits. All per-qubit marginals returned P(|1>) = 0.5000. Bloch vectors aligned along +X, consistent with Hadamard structure. No backend introduced residual structure.

Grover’s algorithm exposed amplification fidelity. The marked state |101> reached probability 0.9453. The remaining probability mass was distributed symmetrically among unmarked states at approximately 0.0078 each. The entropy dropped to 0.4595 bits. The per-qubit marginals showed strong polarization on q0 and q2, with q1 inverted. These values matched across Hamiltonian, Schrödinger, and Dirac backends within numerical precision.

Teleportation did not collapse incorrectly. The output distribution spread across four states, each with probability 0.1875, yielding entropy 2.8113 bits. Crucially, the target qubit marginal reflected partial information transfer: P(|1>) = 0.2500 with ⟨Z⟩ = +0.5000. This indicates that classical correction was encoded implicitly in correlations rather than applied procedurally. No backend shortcut this process.

The snapshot sequence H–CNOT–Z–H revealed the most subtle invariant. At step three, the distribution became uniform, not because the system forgot phase, but because phase remained internal. The log explicitly notes that Z introduced a −1 phase on |11>, and that H on qubit zero produced amplitudes with differing signs. All four probabilities measured 0.25. Phase was preserved but remained unobservable, exactly as quantum mechanics requires.

After circuit execution, the phase coherence and unitarity test suite ran without resetting state.

Single-qubit algebraic identities were verified numerically. HZH mapped |0> to |1> with unit probability. HXH mapped |0> to |0>. HSSH collapsed to HZH. H Rz(π) H produced X. Ry(π) and Rx(π) both mapped |0> to |1>. XX erased itself. HZZH reduced to identity. None of these tests relied on symbolic substitution. Each was executed as amplitude evolution followed by measurement.

Two-qubit interference tests confirmed closure under composition. H–CNOT–CNOT–H reduced to identity with P(|00>) = 1.0000. Composite CZ insertions altered nothing when they should not. Z applied conditionally on the control qubit propagated correctly. SWAP behaved as an involution. Applying it twice erased its effect completely.

Norm preservation was tested explicitly after each major construct. In all cases, the sum of probabilities was reported as exactly 1.00000000. No backend drifted. No normalization correction was applied post hoc.

Entropy tests closed the loop. Bell and GHZ states measured exactly one bit. QFT measured exactly three bits. The |0> state measured zero bits. These are not approximate values. They are exact within floating-point resolution.

All twenty-two tests passed.

Only then did the system cross into chemistry.

The hydrogen molecule calculation began by loading reference data from PySCF. Hartree–Fock energy −1.11699900 Ha. Full configuration interaction energy −1.13730604 Ha. Nuclear repulsion term explicitly logged. This reference was treated as ground truth.

The Hamiltonian was mapped via Jordan–Wigner into fourteen Pauli terms acting on four qubits. No truncation. No term merging. Each coefficient preserved.

The Hartree–Fock computational basis state |1100> evaluated to −1.11699900 Ha exactly. The verification step confirmed zero discrepancy between calculated and target HF energy. This removed compensating degrees of freedom.

The surrogate model established a floor at −1.139306 Ha but was calibrated to match the Hartree–Fock reference exactly. Offset reduced to zero. This calibration did not improve the result. It removed bias.

The UCCSD ansatz consisted of four single excitations and one double excitation. Five parameters. No redundancy. At θ = 0, the energy evaluated to −1.11699900 Ha, confirming identity behavior.

The double excitation amplitude was scanned manually. Energy decreased smoothly as θ_d approached −0.10. At θ_d = −0.10, the energy reached −1.13707997 Ha, within 2.26×10⁻⁴ Ha of the exact FCI value. The landscape was shallow, symmetric, and well-behaved.

Optimization proceeded without noise. After twenty iterations, the energy converged to −1.13730604 Ha. The absolute error relative to FCI was 1.31×10⁻¹¹ Ha. Correlation energy recovery was 100.0%.

No regularization was introduced. No noise model was invoked. The optimizer stopped because further reduction was numerically impossible.

What these logs show is not that the system can run algorithms. They show that it remains constrained when nothing is helping it. Phase does not leak. Norm does not drift. Entropy does not wander. Energy does not compensate.

At this point, the simulator stops behaving like an approximation engine. It behaves like a rule-preserving system.

That is the boundary crossed in this chapter.

— grisun0

- DOI [https://doi.org/10.5281/zenodo.18407920 From Boltzmann Stochasticity to Hamiltonian Integrability: Emergence of Topological Crystals and Synthetic Planck Constants](https://doi.org/10.5281/zenodo.18407920)
- DOI [https://doi.org/10.5281/zenodo.18725428 Schrödinger Topological Crystallization: Phase Space Discovery in Hamiltonian Neural Networks](https://doi.org/10.5281/zenodo.18725428)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
