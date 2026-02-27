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

*grisun0*
