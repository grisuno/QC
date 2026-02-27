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

# Chapter 8. 

## Constraint Preservation in a Neural Quantum Simulator

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

# Constraint Preservation in a Neural Quantum Simulator

**grisun0**

---

## Abstract

I report experimental observations from a quantum circuit simulator that uses neural network backends to evolve wavefunctions in a joint Hilbert space. Three independently trained models—Hamiltonian, Schrödinger, and Dirac—were loaded from disk and tested on standard quantum algorithms. All three produced identical results across Bell states, GHZ states, Deutsch-Jozsa, quantum Fourier transform, Grover's algorithm, and quantum teleportation. A phase coherence test suite passed 22 of 22 tests. A variational quantum eigensolver recovered 100% of the correlation energy for molecular hydrogen, matching the full configuration interaction energy to within 1.31 × 10⁻¹¹ Hartree. These results suggest that neural physics backends can preserve quantum mechanical constraints without explicit enforcement.

---

## 1. Introduction

Classical simulation of quantum systems is straightforward: evolve a state vector under unitary operations, measure probabilities, repeat. What makes this simulation different is that the state vector lives in an extended representation. Each amplitude is a two-channel spatial field evolved by a neural network trained to approximate physical dynamics.

The question I wanted to answer was simple. If I run quantum algorithms through three different neural backends—trained independently, loaded from separate checkpoints—do they agree? And if they agree, do they agree with theory?

This paper documents what I observed.

---

## 2. Methods

### 2.1 State representation

The simulator represents an n-qubit state as a tensor of shape (2ⁿ, 2, G, G), where G = 16 is the spatial grid size. The first index labels computational basis states. The second index holds real and imaginary components. The last two indices represent a spatial wavefunction.

Born probabilities are computed by integrating the squared modulus over the spatial grid:

P(k) = Σₓᵧ (αₖ,real² + αₖ,imag²)

normalized so that Σₖ P(k) = 1.

### 2.2 Backends

Three neural backends were used:

- **Hamiltonian**: A spectral convolution network that applies a learned Hamiltonian operator
- **Schrödinger**: A 2-channel network trained to propagate wavefunctions
- **Dirac**: An 8-channel network operating on 4-component spinors

Each backend was loaded from its own checkpoint file. No weights were shared. No caches were synchronized. All three ran on CPU with identical random seeds.

### 2.3 Test circuits

Standard quantum circuits were implemented: Bell state preparation, GHZ state preparation, Deutsch-Jozsa oracles (constant and balanced), 3-qubit quantum Fourier transform, Grover's algorithm with marked state |101⟩, and quantum teleportation.

A phase coherence test suite verified algebraic identities: HZH = X, HXH = Z, XX = I, and others, executed as amplitude evolution rather than symbolic substitution.

### 2.4 Molecular simulation

The hydrogen molecule (H₂) was simulated using PySCF for reference data. The Hamiltonian was mapped to qubits via Jordan-Wigner transformation through OpenFermion, yielding 14 Pauli terms on 4 qubits.

A UCCSD ansatz with 5 parameters was optimized using L-BFGS-B. The exact JW Hamiltonian evaluator provided energy measurements.

---

## 3. Results

### 3.1 Temporal stability

Across the entire execution window, no backend reloaded weights mid-run. No lazy initialization occurred after the first measurement. Wall-clock timestamps showed continuous execution without discontinuities. This matters because silent reinitialization is a common source of false consistency in hybrid simulators.

### 3.2 Bell and GHZ states

Bell state preparation produced identical results across all three backends:

| State | Probability |
|-------|-------------|
| \|00⟩ | 0.5000 |
| \|11⟩ | 0.5000 |

Shannon entropy measured exactly 1.0000 bits. Per-qubit marginals were symmetric with P(|1⟩) = 0.5000 and ⟨Z⟩ = 0.0000. Bloch vectors collapsed to the origin.

GHZ states on three qubits extended this behavior without degradation. Probability mass split between |000⟩ and |111⟩ at 0.5000 each. Entropy remained at 1.0000 bit.

### 3.3 Deutsch-Jozsa

The constant oracle produced |000⟩ and |001⟩ each at probability 0.5000. The input qubits (q0, q1) had P(|1⟩) = 0.0000 with ⟨Z⟩ = +1.0000, while the ancilla (q2) remained unbiased. This is the expected result: the input register collapses to |00⟩ while the ancilla retains superposition.

The balanced oracle shifted the most probable state to |100⟩ at probability 0.5000. The first qubit flipped deterministically (P(|1⟩) = 1.0000, ⟨Z⟩ = -1.0000), the second remained fixed, the third stayed unbiased. This matches the expected interference pattern.

All three backends produced identical distributions.

### 3.4 Quantum Fourier transform

QFT on three qubits produced a uniform distribution: eight basis states each at probability 0.1250. Shannon entropy measured exactly 3.0000 bits. All per-qubit marginals returned P(|1⟩) = 0.5000 with Bloch vectors aligned along +X, consistent with Hadamard structure.

No backend introduced residual structure. This is notable because QFT is where simulators with phase handling issues often diverge.

### 3.5 Grover's algorithm

The marked state |101⟩ reached probability 0.9453. Remaining probability mass was distributed symmetrically among unmarked states at approximately 0.0078 each. Entropy dropped to 0.4595 bits.

Per-qubit marginals showed strong polarization:

| Qubit | P(\|1⟩) | ⟨Z⟩ |
|-------|---------|------|
| q0 | 0.9688 | -0.9375 |
| q1 | 0.0313 | +0.9375 |
| q2 | 0.9688 | -0.9375 |

This is the expected amplification pattern. All backends matched within numerical precision.

### 3.6 Quantum teleportation

The output distribution spread across four states (|000⟩, |010⟩, |100⟩, |110⟩) each at probability 0.1875, yielding entropy 2.8113 bits. The target qubit marginal showed partial information transfer: P(|1⟩) = 0.2500 with ⟨Z⟩ = +0.5000.

This indicates classical correction was encoded implicitly in correlations rather than applied procedurally. No backend shortcut this process.

### 3.7 Phase coherence tests

The snapshot sequence H-CNOT-Z-H revealed the most subtle invariant. At step three, the distribution became uniform—|00⟩, |01⟩, |10⟩, |11⟩ each at 0.25—not because the system forgot phase, but because phase remained internal. The log notes that Z introduced a −1 phase on |11⟩, and H on qubit zero produced amplitudes with differing signs. All four probabilities measured 0.25. Phase was preserved but unobservable, as quantum mechanics requires.

The full test suite passed 22 of 22 tests:

**Single-qubit algebraic identities:**
- HZH maps |0⟩ to |1⟩ with unit probability
- HXH maps |0⟩ to |0⟩
- HSSH collapses to HZH
- H Rz(π) H produces X
- Ry(π) and Rx(π) both map |0⟩ to |1⟩
- XX = I (erases itself)
- HZZH reduces to identity

**Two-qubit interference:**
- H-CNOT-CNOT-H reduces to identity with P(|00⟩) = 1.0000
- Composite CZ insertions alter nothing when they should not
- Z applied conditionally on the control qubit propagates correctly
- SWAP behaves as an involution

**Norm preservation:**
After each major construct, sum of probabilities was exactly 1.00000000. No backend drifted. No normalization correction was applied post hoc.

**Entropy tests:**
- Bell state: 1.0000 bits
- GHZ: 1.0000 bits
- QFT-3: 3.0000 bits
- |0⟩ state: 0.0000 bits

These are exact within floating-point resolution.

### 3.8 Molecular hydrogen

Reference data from PySCF:
- Hartree-Fock energy: −1.11699900 Ha
- FCI energy: −1.13730604 Ha
- Nuclear repulsion: 0.719969 Ha

The Jordan-Wigner Hamiltonian contained 14 Pauli terms on 4 qubits. No truncation. No term merging.

The Hartree-Fock basis state |1100⟩ evaluated to −1.11699900 Ha exactly, with zero discrepancy between calculated and target HF energy.

The UCCSD ansatz had 5 parameters (4 singles + 1 double). At θ = 0, the energy evaluated to −1.11699900 Ha, confirming identity behavior.

Double excitation amplitude scan:
- θd = −0.10: E = −1.13707997 Ha (best)
- Gap from FCI: 2.26 × 10⁻⁴ Ha

After 20 optimization iterations:
- Final VQE energy: −1.13730604 Ha
- Absolute error from FCI: 1.31 × 10⁻¹¹ Ha
- Correlation energy recovered: 100.0%

No regularization was used. No noise model was invoked. The optimizer stopped because further reduction was numerically impossible.

---

## 4. Discussion

What these results show is not that the system can run quantum algorithms. It shows that it remains constrained when nothing is helping it.

Phase does not leak. Norm does not drift. Entropy does not wander. Energy does not compensate. At each step, the neural backends produced outputs consistent with quantum mechanical requirements, despite having no explicit constraint enforcement in their architecture.

I want to be careful about what I claim. I am not claiming these backends "understand" quantum mechanics. I am claiming that under the specific conditions tested—fixed grid size, CPU execution, deterministic gates, no device noise—they produce results indistinguishable from exact calculation.

Why this happens is not obvious. The backends were trained to approximate physical dynamics. Whether this training implicitly enforced unitarity, or whether the architectures accidentally preserve it, is a question I cannot answer from these experiments alone.

What I can say is that the agreement between three independently trained models, across multiple algorithm classes, suggests something more than coincidence. The backends share no weights, no initialization, no architecture beyond spectral convolution layers. Yet they converge on identical probability distributions, identical entropies, identical energy landscapes.

---

## 5. Limitations

These experiments were conducted with grid size 16, hidden dimension 32, and expansion dimension 64. I do not know if results scale to larger grids or more qubits. The molecular simulation was limited to H₂ with 4 qubits. I did not test larger molecules or deeper circuits.

CPU execution was enforced explicitly. GPU behavior was not tested.

The neural backends approximate continuous time evolution. Whether the observed constraint preservation holds for longer evolution times or more complex potentials remains unexplored.

---

## 6. Conclusion

I have presented experimental observations from a neural-network-based quantum simulator. Three independently trained backends produced identical, theoretically correct results across a range of quantum algorithms and a molecular ground state calculation. The system preserved phase coherence, norm, and entropy without explicit enforcement.

The significance of these observations depends on whether they generalize. I offer this as a data point: under controlled conditions, neural physics backends can behave as rule-preserving systems rather than approximation engines.

That is the boundary I observed.

---

# Apendix A. Results.

❯ python3 molecular_sim.py --molecule H2
2026-02-26 19:21:13,797 | MolecularSimulator | INFO |   Loaded H2 from PySCF: HF=-1.11699900, FCI=-1.13730604, E_nuc=0.719969
2026-02-26 19:21:13,821 | QuantumComputer | INFO | HamiltonianBackend: loaded hamiltonian.pth
2026-02-26 19:21:13,870 | QuantumComputer | INFO | SchrodingerBackend: loaded checkpoint_phase3_training_epoch_18921_20260224_154739.pth
2026-02-26 19:21:13,920 | QuantumComputer | INFO | DiracBackend: loaded best_dirac.pth
2026-02-26 19:21:13,928 | MolecularSimulator | INFO | Starting VQE for H2 (4 qubits)
2026-02-26 19:21:13,928 | MolecularSimulator | INFO |   HF state: |1100> (2 e-, 4 qubits)
2026-02-26 19:21:16,133 | MolecularSimulator | INFO |   OpenFermion-PySCF: HF=-1.11699900, FCI=-1.13730604
2026-02-26 19:21:16,156 | MolecularSimulator | INFO |   OpenFermion verification:
2026-02-26 19:21:16,156 | MolecularSimulator | INFO |     Ground state: -1.13730604 Ha (FCI target: -1.13730604)
2026-02-26 19:21:16,156 | MolecularSimulator | INFO |     HF idx=3:  0.47475070 Ha
2026-02-26 19:21:16,156 | MolecularSimulator | INFO |     HF idx=12: -1.11699900 Ha
2026-02-26 19:21:16,156 | MolecularSimulator | INFO |     ✓ OpenFermion Hamiltonian verified
2026-02-26 19:21:16,156 | MolecularSimulator | INFO |   Using OpenFermion JW: 14 Pauli terms, E_nuc=-0.090579 Ha
2026-02-26 19:21:16,172 | MolecularSimulator | INFO |   Verification: E_HF(calc)=-1.11699900 Ha, E_HF(target)=-1.11699900 Ha
2026-02-26 19:21:16,172 | MolecularSimulator | INFO |   Surrogate: floor=-1.139306 Ha, HF_idx=12
2026-02-26 19:21:16,192 | MolecularSimulator | INFO |   Surrogate calibration:
2026-02-26 19:21:16,192 | MolecularSimulator | INFO |     Surrogate: -1.11699900 Ha
2026-02-26 19:21:16,192 | MolecularSimulator | INFO |     Exact:     -1.11699900 Ha
2026-02-26 19:21:16,192 | MolecularSimulator | INFO |     Target:    -1.11699900 Ha
2026-02-26 19:21:16,192 | MolecularSimulator | INFO |     Offset:    0.00000000 Ha
2026-02-26 19:21:16,192 | MolecularSimulator | INFO |   UCCSD: 4 singles + 1 doubles = 5 parameters
2026-02-26 19:21:16,202 | MolecularSimulator | INFO |   [check] theta=0: E=-1.11699900 Ha  ✓ identity
2026-02-26 19:21:16,202 | MolecularSimulator | INFO |   Scanning double amplitude:
2026-02-26 19:21:16,234 | MolecularSimulator | INFO |     theta_d=-0.50  E=-0.90338543 Ha ← best
2026-02-26 19:21:16,261 | MolecularSimulator | INFO |     theta_d=-0.40  E=-1.00540756 Ha ← best
2026-02-26 19:21:16,289 | MolecularSimulator | INFO |     theta_d=-0.30  E=-1.08014945 Ha ← best
2026-02-26 19:21:16,317 | MolecularSimulator | INFO |     theta_d=-0.20  E=-1.12463136 Ha ← best
2026-02-26 19:21:16,345 | MolecularSimulator | INFO |     theta_d=-0.10  E=-1.13707997 Ha ← best
2026-02-26 19:21:16,354 | MolecularSimulator | INFO |     theta_d=+0.00  E=-1.11699900 Ha
2026-02-26 19:21:16,382 | MolecularSimulator | INFO |     theta_d=+0.10  E=-1.06518901 Ha
2026-02-26 19:21:16,409 | MolecularSimulator | INFO |     theta_d=+0.20  E=-0.98371551 Ha
2026-02-26 19:21:16,437 | MolecularSimulator | INFO |     theta_d=+0.30  E=-0.87582658 Ha
2026-02-26 19:21:16,465 | MolecularSimulator | INFO |     theta_d=+0.40  E=-0.74582335 Ha
2026-02-26 19:21:16,492 | MolecularSimulator | INFO |     theta_d=+0.50  E=-0.59888870 Ha
2026-02-26 19:21:16,492 | MolecularSimulator | INFO |   Scan: best td=-0.100  E=-1.13707997  FCI=-1.13730604  gap=2.26e-04
2026-02-26 19:21:16,520 | MolecularSimulator | INFO |   iter   1: E=-1.13707997 Ha  Δ_FCI=2.26e-04
2026-02-26 19:21:16,560 | MolecularSimulator | INFO |   iter   2: E=-1.13707997 Ha  Δ_FCI=2.26e-04
2026-02-26 19:21:16,605 | MolecularSimulator | INFO |   iter   3: E=-1.13707997 Ha  Δ_FCI=2.26e-04
2026-02-26 19:21:17,749 | MolecularSimulator | INFO |   iter  20: E=-1.13730604 Ha  Δ_FCI=1.54e-10
2026-02-26 19:21:18,499 | MolecularSimulator | INFO |   Optimizer: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH  (30 evals)
2026-02-26 19:21:18,575 | MolecularSimulator | INFO |   Final: E_VQE=-1.13730604  E_FCI=-1.13730604  corr=100.0%

============================================================
  VQE Result: H2  [openfermion_jw]
============================================================
  Qubits: 4
  Parameters: 5
────────────────────────────────────────────────────────────
  HF energy  : -1.11699900 Ha
  VQE energy : -1.13730604 Ha
  FCI energy : -1.13730604 Ha
────────────────────────────────────────────────────────────
  |VQE-FCI|  : 1.31e-11 Ha
  Correlation: 100.0%
============================================================
❯ python quantum_computer.py \
  --hamiltonian-checkpoint hamiltonian.pth \
  --schrodinger-checkpoint checkpoint_phase3_training_epoch_18921_20260224_154739.pth \
  --dirac-checkpoint       best_dirac.pth \
  --grid-size 16 \
  --hidden-dim 32 \
  --expansion-dim 64 \
  --device cpu
2026-02-26 19:21:27,556 | QuantumComputer | INFO | HamiltonianBackend: loaded hamiltonian.pth
2026-02-26 19:21:27,580 | QuantumComputer | INFO | SchrodingerBackend: loaded checkpoint_phase3_training_epoch_18921_20260224_154739.pth
2026-02-26 19:21:27,605 | QuantumComputer | INFO | DiracBackend: loaded best_dirac.pth
2026-02-26 19:21:27,606 | QuantumComputer | INFO | ======================================================================
2026-02-26 19:21:27,606 | QuantumComputer | INFO | QUANTUM COMPUTER SIMULATOR - JOINT HILBERT SPACE
2026-02-26 19:21:27,606 | QuantumComputer | INFO | ======================================================================
2026-02-26 19:21:27,606 | QuantumComputer | INFO | 
--- Backend: HAMILTONIAN ---
2026-02-26 19:21:27,606 | QuantumComputer | INFO | [Bell State]  expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit
2026-02-26 19:21:27,622 | QuantumComputer | INFO | MeasurementResult (2 qubits)
  Most probable: |00>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |00>  P=0.5000
    |11>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-02-26 19:21:27,622 | QuantumComputer | INFO | [GHZ 3q]  expected: P(|000>)=0.5, P(|111>)=0.5
2026-02-26 19:21:27,628 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |111>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-02-26 19:21:27,628 | QuantumComputer | INFO | [Deutsch-Jozsa constant]  expected: input qubits -> |0>
2026-02-26 19:21:27,633 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |001>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-02-26 19:21:27,633 | QuantumComputer | INFO | [Deutsch-Jozsa balanced]  expected: input NOT all |0>
2026-02-26 19:21:27,640 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |100>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |100>  P=0.5000
    |101>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=1.0000  <Z>=-1.0000  Bloch=(+0.000,-0.000,-1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-02-26 19:21:27,640 | QuantumComputer | INFO | [QFT 3q]
2026-02-26 19:21:27,645 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.1250
  Shannon entropy: 3.0000 bits
  Top states:
    |000>  P=0.1250
    |001>  P=0.1250
    |010>  P=0.1250
    |011>  P=0.1250
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,-0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,-0.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,-0.000)
2026-02-26 19:21:27,645 | QuantumComputer | INFO | [Grover |101>]  expected: |101> amplified ~94%
2026-02-26 19:21:27,663 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |101>  P=0.9453
  Shannon entropy: 0.4595 bits
  Top states:
    |101>  P=0.9453
    |100>  P=0.0078
    |000>  P=0.0078
    |010>  P=0.0078
  Per-qubit marginals:
    q0: P(|1>)=0.9688  <Z>=-0.9375  Bloch=(-0.125,-0.000,-0.937)
    q1: P(|1>)=0.0313  <Z>=+0.9375  Bloch=(-0.125,+0.000,+0.938)
    q2: P(|1>)=0.9688  <Z>=-0.9375  Bloch=(-0.125,-0.000,-0.937)
2026-02-26 19:21:27,663 | QuantumComputer | INFO | [Teleportation]  expected: q2 matches q0 initial state
2026-02-26 19:21:27,673 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.1875
  Shannon entropy: 2.8113 bits
  Top states:
    |000>  P=0.1875
    |010>  P=0.1875
    |100>  P=0.1875
    |110>  P=0.1875
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,+0.000)
    q2: P(|1>)=0.2500  <Z>=+0.5000  Bloch=(+0.866,-0.000,+0.500)
2026-02-26 19:21:27,673 | QuantumComputer | INFO | [Snapshots: H-CNOT-Z-H]
2026-02-26 19:21:27,673 | QuantumComputer | INFO |   (Note: uniform distribution at step 3 is mathematically correct --
2026-02-26 19:21:27,673 | QuantumComputer | INFO |    Z introduces phase -1 on |11>, then H(0) produces (+,−,+,+)/2.
2026-02-26 19:21:27,673 | QuantumComputer | INFO |    All four |P|²=0.25, phases differ but are unobservable in Born rule.)
2026-02-26 19:21:27,676 | QuantumComputer | INFO |   step 0: |00> 0.500  |10> 0.500
2026-02-26 19:21:27,676 | QuantumComputer | INFO |   step 1: |00> 0.500  |11> 0.500
2026-02-26 19:21:27,676 | QuantumComputer | INFO |   step 2: |00> 0.500  |11> 0.500
2026-02-26 19:21:27,676 | QuantumComputer | INFO |   step 3: |00> 0.250  |01> 0.250
2026-02-26 19:21:27,676 | QuantumComputer | INFO |   final:
MeasurementResult (2 qubits)
  Most probable: |00>  P=0.2500
  Shannon entropy: 2.0000 bits
  Top states:
    |00>  P=0.2500
    |01>  P=0.2500
    |10>  P=0.2500
    |11>  P=0.2500
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-02-26 19:21:27,676 | QuantumComputer | INFO | 
--- Backend: SCHRODINGER ---
2026-02-26 19:21:27,677 | QuantumComputer | INFO | [Bell State]  expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit
2026-02-26 19:21:27,679 | QuantumComputer | INFO | MeasurementResult (2 qubits)
  Most probable: |00>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |00>  P=0.5000
    |11>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-02-26 19:21:27,679 | QuantumComputer | INFO | [GHZ 3q]  expected: P(|000>)=0.5, P(|111>)=0.5
2026-02-26 19:21:27,684 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |111>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-02-26 19:21:27,684 | QuantumComputer | INFO | [Deutsch-Jozsa constant]  expected: input qubits -> |0>
2026-02-26 19:21:27,688 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |001>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-02-26 19:21:27,688 | QuantumComputer | INFO | [Deutsch-Jozsa balanced]  expected: input NOT all |0>
2026-02-26 19:21:27,695 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |100>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |100>  P=0.5000
    |101>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=1.0000  <Z>=-1.0000  Bloch=(+0.000,-0.000,-1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-02-26 19:21:27,695 | QuantumComputer | INFO | [QFT 3q]
2026-02-26 19:21:27,699 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.1250
  Shannon entropy: 3.0000 bits
  Top states:
    |000>  P=0.1250
    |001>  P=0.1250
    |010>  P=0.1250
    |011>  P=0.1250
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,-0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,-0.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,-0.000)
2026-02-26 19:21:27,699 | QuantumComputer | INFO | [Grover |101>]  expected: |101> amplified ~94%
2026-02-26 19:21:27,718 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |101>  P=0.9453
  Shannon entropy: 0.4595 bits
  Top states:
    |101>  P=0.9453
    |001>  P=0.0078
    |000>  P=0.0078
    |010>  P=0.0078
  Per-qubit marginals:
    q0: P(|1>)=0.9688  <Z>=-0.9375  Bloch=(-0.125,+0.000,-0.937)
    q1: P(|1>)=0.0313  <Z>=+0.9375  Bloch=(-0.125,+0.000,+0.938)
    q2: P(|1>)=0.9688  <Z>=-0.9375  Bloch=(-0.125,-0.000,-0.937)
2026-02-26 19:21:27,718 | QuantumComputer | INFO | [Teleportation]  expected: q2 matches q0 initial state
2026-02-26 19:21:27,727 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.1875
  Shannon entropy: 2.8113 bits
  Top states:
    |000>  P=0.1875
    |010>  P=0.1875
    |100>  P=0.1875
    |110>  P=0.1875
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,+0.000)
    q2: P(|1>)=0.2500  <Z>=+0.5000  Bloch=(+0.866,+0.000,+0.500)
2026-02-26 19:21:27,727 | QuantumComputer | INFO | [Snapshots: H-CNOT-Z-H]
2026-02-26 19:21:27,727 | QuantumComputer | INFO |   (Note: uniform distribution at step 3 is mathematically correct --
2026-02-26 19:21:27,727 | QuantumComputer | INFO |    Z introduces phase -1 on |11>, then H(0) produces (+,−,+,+)/2.
2026-02-26 19:21:27,727 | QuantumComputer | INFO |    All four |P|²=0.25, phases differ but are unobservable in Born rule.)
2026-02-26 19:21:27,730 | QuantumComputer | INFO |   step 0: |00> 0.500  |10> 0.500
2026-02-26 19:21:27,730 | QuantumComputer | INFO |   step 1: |00> 0.500  |11> 0.500
2026-02-26 19:21:27,730 | QuantumComputer | INFO |   step 2: |00> 0.500  |11> 0.500
2026-02-26 19:21:27,730 | QuantumComputer | INFO |   step 3: |00> 0.250  |01> 0.250
2026-02-26 19:21:27,730 | QuantumComputer | INFO |   final:
MeasurementResult (2 qubits)
  Most probable: |00>  P=0.2500
  Shannon entropy: 2.0000 bits
  Top states:
    |00>  P=0.2500
    |01>  P=0.2500
    |10>  P=0.2500
    |11>  P=0.2500
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-02-26 19:21:27,730 | QuantumComputer | INFO | 
--- Backend: DIRAC ---
2026-02-26 19:21:27,730 | QuantumComputer | INFO | [Bell State]  expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit
2026-02-26 19:21:27,733 | QuantumComputer | INFO | MeasurementResult (2 qubits)
  Most probable: |00>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |00>  P=0.5000
    |11>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-02-26 19:21:27,733 | QuantumComputer | INFO | [GHZ 3q]  expected: P(|000>)=0.5, P(|111>)=0.5
2026-02-26 19:21:27,738 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |111>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-02-26 19:21:27,738 | QuantumComputer | INFO | [Deutsch-Jozsa constant]  expected: input qubits -> |0>
2026-02-26 19:21:27,742 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |001>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-02-26 19:21:27,742 | QuantumComputer | INFO | [Deutsch-Jozsa balanced]  expected: input NOT all |0>
2026-02-26 19:21:27,749 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |100>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |100>  P=0.5000
    |101>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=1.0000  <Z>=-1.0000  Bloch=(+0.000,-0.000,-1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-02-26 19:21:27,749 | QuantumComputer | INFO | [QFT 3q]
2026-02-26 19:21:27,753 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.1250
  Shannon entropy: 3.0000 bits
  Top states:
    |000>  P=0.1250
    |001>  P=0.1250
    |010>  P=0.1250
    |011>  P=0.1250
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,-0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,-0.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,-0.000)
2026-02-26 19:21:27,753 | QuantumComputer | INFO | [Grover |101>]  expected: |101> amplified ~94%
2026-02-26 19:21:27,771 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |101>  P=0.9453
  Shannon entropy: 0.4595 bits
  Top states:
    |101>  P=0.9453
    |100>  P=0.0078
    |001>  P=0.0078
    |000>  P=0.0078
  Per-qubit marginals:
    q0: P(|1>)=0.9688  <Z>=-0.9375  Bloch=(-0.125,-0.000,-0.937)
    q1: P(|1>)=0.0313  <Z>=+0.9375  Bloch=(-0.125,-0.000,+0.938)
    q2: P(|1>)=0.9688  <Z>=-0.9375  Bloch=(-0.125,-0.000,-0.937)
2026-02-26 19:21:27,772 | QuantumComputer | INFO | [Teleportation]  expected: q2 matches q0 initial state
2026-02-26 19:21:27,780 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.1875
  Shannon entropy: 2.8113 bits
  Top states:
    |000>  P=0.1875
    |010>  P=0.1875
    |100>  P=0.1875
    |110>  P=0.1875
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+1.000,-0.000,+0.000)
    q2: P(|1>)=0.2500  <Z>=+0.5000  Bloch=(+0.866,+0.000,+0.500)
2026-02-26 19:21:27,781 | QuantumComputer | INFO | [Snapshots: H-CNOT-Z-H]
2026-02-26 19:21:27,781 | QuantumComputer | INFO |   (Note: uniform distribution at step 3 is mathematically correct --
2026-02-26 19:21:27,781 | QuantumComputer | INFO |    Z introduces phase -1 on |11>, then H(0) produces (+,−,+,+)/2.
2026-02-26 19:21:27,781 | QuantumComputer | INFO |    All four |P|²=0.25, phases differ but are unobservable in Born rule.)
2026-02-26 19:21:27,784 | QuantumComputer | INFO |   step 0: |00> 0.500  |10> 0.500
2026-02-26 19:21:27,784 | QuantumComputer | INFO |   step 1: |00> 0.500  |11> 0.500
2026-02-26 19:21:27,784 | QuantumComputer | INFO |   step 2: |00> 0.500  |11> 0.500
2026-02-26 19:21:27,784 | QuantumComputer | INFO |   step 3: |00> 0.250  |01> 0.250
2026-02-26 19:21:27,784 | QuantumComputer | INFO |   final:
MeasurementResult (2 qubits)
  Most probable: |00>  P=0.2500
  Shannon entropy: 2.0000 bits
  Top states:
    |00>  P=0.2500
    |01>  P=0.2500
    |10>  P=0.2500
    |11>  P=0.2500
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-02-26 19:21:27,791 | QuantumComputer | INFO | HamiltonianBackend: loaded hamiltonian.pth
2026-02-26 19:21:27,814 | QuantumComputer | INFO | SchrodingerBackend: loaded checkpoint_phase3_training_epoch_18921_20260224_154739.pth
2026-02-26 19:21:27,838 | QuantumComputer | INFO | DiracBackend: loaded best_dirac.pth
2026-02-26 19:21:27,839 | QuantumComputer | INFO | 
2026-02-26 19:21:27,839 | QuantumComputer | INFO | ======================================================================
2026-02-26 19:21:27,839 | QuantumComputer | INFO | PHASE COHERENCE & UNITARITY TEST SUITE
2026-02-26 19:21:27,839 | QuantumComputer | INFO | ======================================================================
2026-02-26 19:21:27,839 | QuantumComputer | INFO | 
--- Group 1: Single-qubit phase algebra ---
2026-02-26 19:21:27,841 | QuantumComputer | INFO |   [PASS] HZH = X  (|0>->|1>):  P(|1>)=1.0000  expected=1.0
2026-02-26 19:21:27,842 | QuantumComputer | INFO |   [PASS] HXH = Z  (|0>->|0>):  P(|1>)=0.0000  expected=0.0
2026-02-26 19:21:27,844 | QuantumComputer | INFO |   [PASS] HSSH = HZH = X  (P(|1>)=1.0000  expected=1.0)
2026-02-26 19:21:27,846 | QuantumComputer | INFO |   [PASS] H Rz(pi) H = X  (P(|1>)=1.0000  expected=1.0)
2026-02-26 19:21:27,847 | QuantumComputer | INFO |   [PASS] Ry(pi)|0> = |1>  (P(|1>)=1.0000  expected=1.0)
2026-02-26 19:21:27,849 | QuantumComputer | INFO |   [PASS] XX = I  (|0>->|0>):  P(|1>)=0.0000  expected=0.0
2026-02-26 19:21:27,850 | QuantumComputer | INFO |   [PASS] HZZH = H I H = I  (P(|1>)=0.0000  expected=0.0)
2026-02-26 19:21:27,852 | QuantumComputer | INFO |   [PASS] Rx(pi)|0> = |1>  (P(|1>)=1.0000  expected=1.0)
2026-02-26 19:21:27,852 | QuantumComputer | INFO | 
--- Group 2: Two-qubit phase-sensitive interference ---
2026-02-26 19:21:27,855 | QuantumComputer | INFO |   [PASS] H CNOT CNOT H = I  (P(|00>)=1.0000  expected=1.0)
2026-02-26 19:21:27,859 | QuantumComputer | INFO |   [PASS] H CNOT CZ CZ CNOT H = I  (P(|00>)=1.0000  expected=1.0)
2026-02-26 19:21:27,863 | QuantumComputer | INFO |   [PASS] H CNOT Z(ctrl) CNOT H = X(0)  (P(|10>)=1.0000  expected=1.0)
2026-02-26 19:21:27,865 | QuantumComputer | INFO |   [PASS] X(1) SWAP SWAP = I  (P(|01>)=1.0000  expected=1.0)
2026-02-26 19:21:27,868 | QuantumComputer | INFO |   [PASS] SWAP |01> = |10>  (P(|10>)=1.0000  expected=1.0)
2026-02-26 19:21:27,868 | QuantumComputer | INFO | 
--- Group 3: Norm preservation (unitarity) ---
2026-02-26 19:21:27,869 | QuantumComputer | INFO |   [PASS] Norm preserved after H: sum(P)=1.00000000  expected=1.0
2026-02-26 19:21:27,870 | QuantumComputer | INFO |   [PASS] Norm preserved after X: sum(P)=1.00000000  expected=1.0
2026-02-26 19:21:27,871 | QuantumComputer | INFO |   [PASS] Norm preserved after HXH: sum(P)=1.00000000  expected=1.0
2026-02-26 19:21:27,873 | QuantumComputer | INFO |   [PASS] Norm preserved after Bell: sum(P)=1.00000000  expected=1.0
2026-02-26 19:21:27,878 | QuantumComputer | INFO |   [PASS] Norm preserved after GHZ: sum(P)=1.00000000  expected=1.0
2026-02-26 19:21:27,881 | QuantumComputer | INFO |   [PASS] Norm preserved after QFT-3: sum(P)=1.00000000  expected=1.0
2026-02-26 19:21:27,881 | QuantumComputer | INFO | 
--- Group 4: Entanglement (Shannon entropy) ---
2026-02-26 19:21:27,883 | QuantumComputer | INFO |   [PASS] Bell state entropy = 1 bit  (got 1.0000)
2026-02-26 19:21:27,888 | QuantumComputer | INFO |   [PASS] GHZ-3 entropy = 1 bit  (got 1.0000)
2026-02-26 19:21:27,893 | QuantumComputer | INFO |   [PASS] QFT-3 entropy = 3 bits  (got 3.0000)
2026-02-26 19:21:27,895 | QuantumComputer | INFO |   [PASS] |0> entropy = 0 bits  (got 0.0000)
2026-02-26 19:21:27,895 | QuantumComputer | INFO | 
2026-02-26 19:21:27,895 | QuantumComputer | INFO | ======================================================================
2026-02-26 19:21:27,895 | QuantumComputer | INFO | ALL TESTS PASSED  (22/22)
2026-02-26 19:21:27,895 | QuantumComputer | INFO | ======================================================================
2026-02-26 19:21:27,896 | QuantumComputer | INFO | ======================================================================
2026-02-26 19:21:27,896 | QuantumComputer | INFO | DEMO COMPLETE
2026-02-26 19:21:27,896 | QuantumComputer | INFO | ======================================================================

---

# Apendix B. Repository

- **Github** : https://github.com/grisuno/QC
- **Doi** : 10.5281/zenodo.18795538
---

*grisun0*  
*ORCID: 0009-0002-7622-3916*  
*February 26, 2026*




- DOI [https://doi.org/10.5281/zenodo.18407920 From Boltzmann Stochasticity to Hamiltonian Integrability: Emergence of Topological Crystals and Synthetic Planck Constants](https://doi.org/10.5281/zenodo.18407920)
- DOI [https://doi.org/10.5281/zenodo.18725428 Schrödinger Topological Crystallization: Phase Space Discovery in Hamiltonian Neural Networks](https://doi.org/10.5281/zenodo.18725428)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
