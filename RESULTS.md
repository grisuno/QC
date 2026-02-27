# RESULTS ADN USAGE

```text
python3 molecular_sim.py --molecule H2
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
```


