# RESULTS ADN USAGE

```text
 python3 quantum_computer.py \
  --hamiltonian-checkpoint hamiltonian.pth \
  --schrodinger-checkpoint checkpoint_phase3_training_epoch_18921_20260224_154739.pth \
  --dirac-checkpoint       best_dirac.pth \
  --grid-size 16 \
  --hidden-dim 32 \
  --expansion-dim 64 \
  --device cpu
2026-03-01 05:04:52,221 | QuantumComputer | INFO | HamiltonianBackend: loaded hamiltonian.pth
2026-03-01 05:04:52,246 | QuantumComputer | INFO | SchrodingerBackend: loaded checkpoint_phase3_training_epoch_18921_20260224_154739.pth
2026-03-01 05:04:52,271 | QuantumComputer | INFO | DiracBackend: loaded best_dirac.pth
2026-03-01 05:04:52,272 | QuantumComputer | INFO | ======================================================================
2026-03-01 05:04:52,272 | QuantumComputer | INFO | QUANTUM COMPUTER SIMULATOR - JOINT HILBERT SPACE
2026-03-01 05:04:52,272 | QuantumComputer | INFO | ======================================================================
2026-03-01 05:04:52,272 | QuantumComputer | INFO | 
--- Backend: HAMILTONIAN ---
2026-03-01 05:04:52,272 | QuantumComputer | INFO | [Bell State]  expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit
2026-03-01 05:04:52,276 | QuantumComputer | INFO | MeasurementResult (2 qubits)
  Most probable: |00>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |00>  P=0.5000
    |11>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-03-01 05:04:52,276 | QuantumComputer | INFO | [GHZ 3q]  expected: P(|000>)=0.5, P(|111>)=0.5
2026-03-01 05:04:52,282 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |111>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-03-01 05:04:52,282 | QuantumComputer | INFO | [Deutsch-Jozsa constant]  expected: input qubits -> |0>
2026-03-01 05:04:52,287 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |001>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-03-01 05:04:52,287 | QuantumComputer | INFO | [Deutsch-Jozsa balanced]  expected: input NOT all |0>
2026-03-01 05:04:52,293 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |100>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |100>  P=0.5000
    |101>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=1.0000  <Z>=-1.0000  Bloch=(+0.000,-0.000,-1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-03-01 05:04:52,294 | QuantumComputer | INFO | [QFT 3q]
2026-03-01 05:04:52,299 | QuantumComputer | INFO | MeasurementResult (3 qubits)
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
2026-03-01 05:04:52,299 | QuantumComputer | INFO | [Grover |101>]  expected: |101> amplified ~94%
2026-03-01 05:04:52,322 | QuantumComputer | INFO | MeasurementResult (3 qubits)
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
2026-03-01 05:04:52,322 | QuantumComputer | INFO | [Teleportation]  expected: q2 matches q0 initial state
2026-03-01 05:04:52,334 | QuantumComputer | INFO | MeasurementResult (3 qubits)
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
2026-03-01 05:04:52,334 | QuantumComputer | INFO | [Snapshots: H-CNOT-Z-H]
2026-03-01 05:04:52,334 | QuantumComputer | INFO |   (Note: uniform distribution at step 3 is mathematically correct --
2026-03-01 05:04:52,334 | QuantumComputer | INFO |    Z introduces phase -1 on |11>, then H(0) produces (+,−,+,+)/2.
2026-03-01 05:04:52,334 | QuantumComputer | INFO |    All four |P|²=0.25, phases differ but are unobservable in Born rule.)
2026-03-01 05:04:52,338 | QuantumComputer | INFO |   step 0: |00> 0.500  |10> 0.500
2026-03-01 05:04:52,338 | QuantumComputer | INFO |   step 1: |00> 0.500  |11> 0.500
2026-03-01 05:04:52,338 | QuantumComputer | INFO |   step 2: |00> 0.500  |11> 0.500
2026-03-01 05:04:52,338 | QuantumComputer | INFO |   step 3: |00> 0.250  |01> 0.250
2026-03-01 05:04:52,338 | QuantumComputer | INFO |   final:
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
2026-03-01 05:04:52,338 | QuantumComputer | INFO | 
--- Backend: SCHRODINGER ---
2026-03-01 05:04:52,338 | QuantumComputer | INFO | [Bell State]  expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit
2026-03-01 05:04:52,340 | QuantumComputer | INFO | MeasurementResult (2 qubits)
  Most probable: |00>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |00>  P=0.5000
    |11>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-03-01 05:04:52,340 | QuantumComputer | INFO | [GHZ 3q]  expected: P(|000>)=0.5, P(|111>)=0.5
2026-03-01 05:04:52,345 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |111>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-03-01 05:04:52,345 | QuantumComputer | INFO | [Deutsch-Jozsa constant]  expected: input qubits -> |0>
2026-03-01 05:04:52,350 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |001>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-03-01 05:04:52,350 | QuantumComputer | INFO | [Deutsch-Jozsa balanced]  expected: input NOT all |0>
2026-03-01 05:04:52,356 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |100>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |100>  P=0.5000
    |101>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=1.0000  <Z>=-1.0000  Bloch=(+0.000,-0.000,-1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-03-01 05:04:52,357 | QuantumComputer | INFO | [QFT 3q]
2026-03-01 05:04:52,362 | QuantumComputer | INFO | MeasurementResult (3 qubits)
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
2026-03-01 05:04:52,362 | QuantumComputer | INFO | [Grover |101>]  expected: |101> amplified ~94%
2026-03-01 05:04:52,384 | QuantumComputer | INFO | MeasurementResult (3 qubits)
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
2026-03-01 05:04:52,384 | QuantumComputer | INFO | [Teleportation]  expected: q2 matches q0 initial state
2026-03-01 05:04:52,393 | QuantumComputer | INFO | MeasurementResult (3 qubits)
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
2026-03-01 05:04:52,393 | QuantumComputer | INFO | [Snapshots: H-CNOT-Z-H]
2026-03-01 05:04:52,393 | QuantumComputer | INFO |   (Note: uniform distribution at step 3 is mathematically correct --
2026-03-01 05:04:52,393 | QuantumComputer | INFO |    Z introduces phase -1 on |11>, then H(0) produces (+,−,+,+)/2.
2026-03-01 05:04:52,393 | QuantumComputer | INFO |    All four |P|²=0.25, phases differ but are unobservable in Born rule.)
2026-03-01 05:04:52,396 | QuantumComputer | INFO |   step 0: |00> 0.500  |10> 0.500
2026-03-01 05:04:52,396 | QuantumComputer | INFO |   step 1: |00> 0.500  |11> 0.500
2026-03-01 05:04:52,396 | QuantumComputer | INFO |   step 2: |00> 0.500  |11> 0.500
2026-03-01 05:04:52,396 | QuantumComputer | INFO |   step 3: |00> 0.250  |01> 0.250
2026-03-01 05:04:52,396 | QuantumComputer | INFO |   final:
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
2026-03-01 05:04:52,396 | QuantumComputer | INFO | 
--- Backend: DIRAC ---
2026-03-01 05:04:52,396 | QuantumComputer | INFO | [Bell State]  expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit
2026-03-01 05:04:52,398 | QuantumComputer | INFO | MeasurementResult (2 qubits)
  Most probable: |00>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |00>  P=0.5000
    |11>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-03-01 05:04:52,398 | QuantumComputer | INFO | [GHZ 3q]  expected: P(|000>)=0.5, P(|111>)=0.5
2026-03-01 05:04:52,403 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |111>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q1: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(+0.000,-0.000,+0.000)
2026-03-01 05:04:52,403 | QuantumComputer | INFO | [Deutsch-Jozsa constant]  expected: input qubits -> |0>
2026-03-01 05:04:52,408 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |000>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |000>  P=0.5000
    |001>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-03-01 05:04:52,409 | QuantumComputer | INFO | [Deutsch-Jozsa balanced]  expected: input NOT all |0>
2026-03-01 05:04:52,415 | QuantumComputer | INFO | MeasurementResult (3 qubits)
  Most probable: |100>  P=0.5000
  Shannon entropy: 1.0000 bits
  Top states:
    |100>  P=0.5000
    |101>  P=0.5000
  Per-qubit marginals:
    q0: P(|1>)=1.0000  <Z>=-1.0000  Bloch=(+0.000,-0.000,-1.000)
    q1: P(|1>)=0.0000  <Z>=+1.0000  Bloch=(+0.000,-0.000,+1.000)
    q2: P(|1>)=0.5000  <Z>=+0.0000  Bloch=(-1.000,-0.000,+0.000)
2026-03-01 05:04:52,415 | QuantumComputer | INFO | [QFT 3q]
2026-03-01 05:04:52,420 | QuantumComputer | INFO | MeasurementResult (3 qubits)
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
2026-03-01 05:04:52,420 | QuantumComputer | INFO | [Grover |101>]  expected: |101> amplified ~94%
2026-03-01 05:04:52,441 | QuantumComputer | INFO | MeasurementResult (3 qubits)
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
2026-03-01 05:04:52,441 | QuantumComputer | INFO | [Teleportation]  expected: q2 matches q0 initial state
2026-03-01 05:04:52,450 | QuantumComputer | INFO | MeasurementResult (3 qubits)
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
2026-03-01 05:04:52,450 | QuantumComputer | INFO | [Snapshots: H-CNOT-Z-H]
2026-03-01 05:04:52,450 | QuantumComputer | INFO |   (Note: uniform distribution at step 3 is mathematically correct --
2026-03-01 05:04:52,450 | QuantumComputer | INFO |    Z introduces phase -1 on |11>, then H(0) produces (+,−,+,+)/2.
2026-03-01 05:04:52,450 | QuantumComputer | INFO |    All four |P|²=0.25, phases differ but are unobservable in Born rule.)
2026-03-01 05:04:52,454 | QuantumComputer | INFO |   step 0: |00> 0.500  |10> 0.500
2026-03-01 05:04:52,454 | QuantumComputer | INFO |   step 1: |00> 0.500  |11> 0.500
2026-03-01 05:04:52,454 | QuantumComputer | INFO |   step 2: |00> 0.500  |11> 0.500
2026-03-01 05:04:52,454 | QuantumComputer | INFO |   step 3: |00> 0.250  |01> 0.250
2026-03-01 05:04:52,454 | QuantumComputer | INFO |   final:
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
2026-03-01 05:04:52,461 | QuantumComputer | INFO | HamiltonianBackend: loaded hamiltonian.pth
2026-03-01 05:04:52,489 | QuantumComputer | INFO | SchrodingerBackend: loaded checkpoint_phase3_training_epoch_18921_20260224_154739.pth
2026-03-01 05:04:52,515 | QuantumComputer | INFO | DiracBackend: loaded best_dirac.pth
2026-03-01 05:04:52,516 | QuantumComputer | INFO | 
2026-03-01 05:04:52,516 | QuantumComputer | INFO | ======================================================================
2026-03-01 05:04:52,516 | QuantumComputer | INFO | PHASE COHERENCE & UNITARITY TEST SUITE
2026-03-01 05:04:52,516 | QuantumComputer | INFO | ======================================================================
2026-03-01 05:04:52,516 | QuantumComputer | INFO | 
--- Group 1: Single-qubit phase algebra ---
2026-03-01 05:04:52,518 | QuantumComputer | INFO |   [PASS] HZH = X  (|0>->|1>):  P(|1>)=1.0000  expected=1.0
2026-03-01 05:04:52,520 | QuantumComputer | INFO |   [PASS] HXH = Z  (|0>->|0>):  P(|1>)=0.0000  expected=0.0
2026-03-01 05:04:52,521 | QuantumComputer | INFO |   [PASS] HSSH = HZH = X  (P(|1>)=1.0000  expected=1.0)
2026-03-01 05:04:52,523 | QuantumComputer | INFO |   [PASS] H Rz(pi) H = X  (P(|1>)=1.0000  expected=1.0)
2026-03-01 05:04:52,524 | QuantumComputer | INFO |   [PASS] Ry(pi)|0> = |1>  (P(|1>)=1.0000  expected=1.0)
2026-03-01 05:04:52,525 | QuantumComputer | INFO |   [PASS] XX = I  (|0>->|0>):  P(|1>)=0.0000  expected=0.0
2026-03-01 05:04:52,527 | QuantumComputer | INFO |   [PASS] HZZH = H I H = I  (P(|1>)=0.0000  expected=0.0)
2026-03-01 05:04:52,528 | QuantumComputer | INFO |   [PASS] Rx(pi)|0> = |1>  (P(|1>)=1.0000  expected=1.0)
2026-03-01 05:04:52,528 | QuantumComputer | INFO | 
--- Group 2: Two-qubit phase-sensitive interference ---
2026-03-01 05:04:52,531 | QuantumComputer | INFO |   [PASS] H CNOT CNOT H = I  (P(|00>)=1.0000  expected=1.0)
2026-03-01 05:04:52,536 | QuantumComputer | INFO |   [PASS] H CNOT CZ CZ CNOT H = I  (P(|00>)=1.0000  expected=1.0)
2026-03-01 05:04:52,540 | QuantumComputer | INFO |   [PASS] H CNOT Z(ctrl) CNOT H = X(0)  (P(|10>)=1.0000  expected=1.0)
2026-03-01 05:04:52,543 | QuantumComputer | INFO |   [PASS] X(1) SWAP SWAP = I  (P(|01>)=1.0000  expected=1.0)
2026-03-01 05:04:52,545 | QuantumComputer | INFO |   [PASS] SWAP |01> = |10>  (P(|10>)=1.0000  expected=1.0)
2026-03-01 05:04:52,545 | QuantumComputer | INFO | 
--- Group 3: Norm preservation (unitarity) ---
2026-03-01 05:04:52,546 | QuantumComputer | INFO |   [PASS] Norm preserved after H: sum(P)=1.00000000  expected=1.0
2026-03-01 05:04:52,547 | QuantumComputer | INFO |   [PASS] Norm preserved after X: sum(P)=1.00000000  expected=1.0
2026-03-01 05:04:52,549 | QuantumComputer | INFO |   [PASS] Norm preserved after HXH: sum(P)=1.00000000  expected=1.0
2026-03-01 05:04:52,551 | QuantumComputer | INFO |   [PASS] Norm preserved after Bell: sum(P)=1.00000000  expected=1.0
2026-03-01 05:04:52,556 | QuantumComputer | INFO |   [PASS] Norm preserved after GHZ: sum(P)=1.00000000  expected=1.0
2026-03-01 05:04:52,559 | QuantumComputer | INFO |   [PASS] Norm preserved after QFT-3: sum(P)=1.00000000  expected=1.0
2026-03-01 05:04:52,559 | QuantumComputer | INFO | 
--- Group 4: Entanglement (Shannon entropy) ---
2026-03-01 05:04:52,562 | QuantumComputer | INFO |   [PASS] Bell state entropy = 1 bit  (got 1.0000)
2026-03-01 05:04:52,567 | QuantumComputer | INFO |   [PASS] GHZ-3 entropy = 1 bit  (got 1.0000)
2026-03-01 05:04:52,572 | QuantumComputer | INFO |   [PASS] QFT-3 entropy = 3 bits  (got 3.0000)
2026-03-01 05:04:52,573 | QuantumComputer | INFO |   [PASS] |0> entropy = 0 bits  (got 0.0000)
2026-03-01 05:04:52,574 | QuantumComputer | INFO | 
2026-03-01 05:04:52,574 | QuantumComputer | INFO | ======================================================================
2026-03-01 05:04:52,574 | QuantumComputer | INFO | ALL TESTS PASSED  (22/22)
2026-03-01 05:04:52,574 | QuantumComputer | INFO | ======================================================================
2026-03-01 05:04:52,576 | QuantumComputer | INFO | ======================================================================
2026-03-01 05:04:52,576 | QuantumComputer | INFO | DEMO COMPLETE
2026-03-01 05:04:52,576 | QuantumComputer | INFO | ======================================================================
❯ python3 molecular_sim.py --molecule H2
2026-03-01 05:05:05,540 | MolecularSimulator | INFO |   Loaded H2 from PySCF: HF=-1.11699900, FCI=-1.13730604, E_nuc=0.719969
2026-03-01 05:05:05,555 | QuantumComputer | INFO | HamiltonianBackend: loaded hamiltonian.pth
2026-03-01 05:05:05,577 | QuantumComputer | INFO | SchrodingerBackend: loaded checkpoint_phase3_training_epoch_18921_20260224_154739.pth
2026-03-01 05:05:05,602 | QuantumComputer | INFO | DiracBackend: loaded best_dirac.pth
2026-03-01 05:05:05,603 | MolecularSimulator | INFO | Starting VQE for H2 (4 qubits)
2026-03-01 05:05:05,603 | MolecularSimulator | INFO |   HF state: |1100> (2 e-, 4 qubits)
2026-03-01 05:05:06,500 | MolecularSimulator | INFO |   OpenFermion-PySCF: HF=-1.11699900, FCI=-1.13730604
2026-03-01 05:05:06,517 | MolecularSimulator | INFO |   OpenFermion verification:
2026-03-01 05:05:06,517 | MolecularSimulator | INFO |     Ground state: -1.13730604 Ha (FCI target: -1.13730604)
2026-03-01 05:05:06,517 | MolecularSimulator | INFO |     HF idx=3:  0.47475070 Ha
2026-03-01 05:05:06,517 | MolecularSimulator | INFO |     HF idx=12: -1.11699900 Ha
2026-03-01 05:05:06,517 | MolecularSimulator | INFO |     ✓ OpenFermion Hamiltonian verified
2026-03-01 05:05:06,517 | MolecularSimulator | INFO |   Using OpenFermion JW: 14 Pauli terms, E_nuc=-0.090579 Ha
2026-03-01 05:05:06,528 | MolecularSimulator | INFO |   Verification: E_HF(calc)=-1.11699900 Ha, E_HF(target)=-1.11699900 Ha
2026-03-01 05:05:06,528 | MolecularSimulator | INFO |   Surrogate: floor=-1.139306 Ha, HF_idx=12
2026-03-01 05:05:06,549 | MolecularSimulator | INFO |   Surrogate calibration:
2026-03-01 05:05:06,549 | MolecularSimulator | INFO |     Surrogate: -1.11699900 Ha
2026-03-01 05:05:06,550 | MolecularSimulator | INFO |     Exact:     -1.11699900 Ha
2026-03-01 05:05:06,550 | MolecularSimulator | INFO |     Target:    -1.11699900 Ha
2026-03-01 05:05:06,550 | MolecularSimulator | INFO |     Offset:    0.00000000 Ha
2026-03-01 05:05:06,550 | MolecularSimulator | INFO |   UCCSD: 4 singles + 1 doubles = 5 parameters
2026-03-01 05:05:06,560 | MolecularSimulator | INFO |   [check] theta=0: E=-1.11699900 Ha  ✓ identity
2026-03-01 05:05:06,560 | MolecularSimulator | INFO |   Scanning double amplitude:
2026-03-01 05:05:06,592 | MolecularSimulator | INFO |     theta_d=-0.50  E=-0.90338543 Ha ← best
2026-03-01 05:05:06,621 | MolecularSimulator | INFO |     theta_d=-0.40  E=-1.00540756 Ha ← best
2026-03-01 05:05:06,651 | MolecularSimulator | INFO |     theta_d=-0.30  E=-1.08014945 Ha ← best
2026-03-01 05:05:06,681 | MolecularSimulator | INFO |     theta_d=-0.20  E=-1.12463136 Ha ← best
2026-03-01 05:05:06,711 | MolecularSimulator | INFO |     theta_d=-0.10  E=-1.13707997 Ha ← best
2026-03-01 05:05:06,721 | MolecularSimulator | INFO |     theta_d=+0.00  E=-1.11699900 Ha
2026-03-01 05:05:06,750 | MolecularSimulator | INFO |     theta_d=+0.10  E=-1.06518901 Ha
2026-03-01 05:05:06,780 | MolecularSimulator | INFO |     theta_d=+0.20  E=-0.98371551 Ha
2026-03-01 05:05:06,810 | MolecularSimulator | INFO |     theta_d=+0.30  E=-0.87582658 Ha
2026-03-01 05:05:06,839 | MolecularSimulator | INFO |     theta_d=+0.40  E=-0.74582335 Ha
2026-03-01 05:05:06,868 | MolecularSimulator | INFO |     theta_d=+0.50  E=-0.59888870 Ha
2026-03-01 05:05:06,868 | MolecularSimulator | INFO |   Scan: best td=-0.100  E=-1.13707997  FCI=-1.13730604  gap=2.26e-04
2026-03-01 05:05:06,898 | MolecularSimulator | INFO |   iter   1: E=-1.13707997 Ha  Δ_FCI=2.26e-04
2026-03-01 05:05:06,941 | MolecularSimulator | INFO |   iter   2: E=-1.13707997 Ha  Δ_FCI=2.26e-04
2026-03-01 05:05:06,990 | MolecularSimulator | INFO |   iter   3: E=-1.13707997 Ha  Δ_FCI=2.26e-04
2026-03-01 05:05:08,182 | MolecularSimulator | INFO |   iter  20: E=-1.13730604 Ha  Δ_FCI=1.54e-10
2026-03-01 05:05:08,963 | MolecularSimulator | INFO |   Optimizer: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH  (30 evals)
2026-03-01 05:05:09,041 | MolecularSimulator | INFO |   Final: E_VQE=-1.13730604  E_FCI=-1.13730604  corr=100.0%

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
❯ python3 advanced_experiments.py
2026-03-01 05:05:19,670 | MolecularSimulator | INFO |   Loaded H2 from PySCF: HF=-1.11699900, FCI=-1.13730604, E_nuc=0.719969
2026-03-01 05:05:19,674 | AdvancedExperiments | INFO | 
======================================================================
2026-03-01 05:05:19,674 | AdvancedExperiments | INFO | ADVANCED QUANTUM EXPERIMENTS - INITIALIZING
2026-03-01 05:05:19,674 | AdvancedExperiments | INFO | ======================================================================
2026-03-01 05:05:19,674 | AdvancedExperiments | INFO | 
======================================================================
2026-03-01 05:05:19,674 | AdvancedExperiments | INFO | RUNNING ALL ADVANCED EXPERIMENTS
2026-03-01 05:05:19,674 | AdvancedExperiments | INFO | ======================================================================
2026-03-01 05:05:19,674 | AdvancedExperiments | INFO | 
######################################################################
2026-03-01 05:05:19,674 | AdvancedExperiments | INFO | # EXPERIMENT 1: GROVER'S ALGORITHM
2026-03-01 05:05:19,674 | AdvancedExperiments | INFO | ######################################################################
2026-03-01 05:05:19,687 | QuantumComputer | INFO | HamiltonianBackend: loaded weights/latest.pth
2026-03-01 05:05:19,714 | QuantumComputer | INFO | SchrodingerBackend: loaded weights/schrodinger_crystal_final.pth
2026-03-01 05:05:19,744 | QuantumComputer | INFO | DiracBackend: loaded weights/dirac_phase5_latest.pth
2026-03-01 05:05:19,745 | AdvancedExperiments | INFO |   Backend: schrodinger
2026-03-01 05:05:19,745 | AdvancedExperiments | INFO | Grover Search initialized:
2026-03-01 05:05:19,745 | AdvancedExperiments | INFO |   Qubits: 3
2026-03-01 05:05:19,745 | AdvancedExperiments | INFO |   Marked state: |101> (decimal 5)
2026-03-01 05:05:19,745 | AdvancedExperiments | INFO |   Iterations: 2 (optimal: 2)
2026-03-01 05:05:19,745 | AdvancedExperiments | INFO | 
============================================================
2026-03-01 05:05:19,745 | AdvancedExperiments | INFO | RUNNING GROVER'S ALGORITHM
2026-03-01 05:05:19,745 | AdvancedExperiments | INFO | ============================================================
2026-03-01 05:05:19,747 | AdvancedExperiments | INFO | 
Initial state: |000>
2026-03-01 05:05:19,750 | AdvancedExperiments | INFO | After Hadamard: entropy = 3.0000 bits
2026-03-01 05:05:19,750 | AdvancedExperiments | INFO | Initial probability of |101>: 0.1250
2026-03-01 05:05:19,756 | AdvancedExperiments | INFO | 
Iteration 1:
2026-03-01 05:05:19,756 | AdvancedExperiments | INFO |   P(|101>) = 0.7813
2026-03-01 05:05:19,756 | AdvancedExperiments | INFO |   Entropy = 1.3720 bits
2026-03-01 05:05:19,756 | AdvancedExperiments | INFO |   Most probable: |101>
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO | 
Iteration 2:
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO |   P(|101>) = 0.9453
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO |   Entropy = 0.4595 bits
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO |   Most probable: |101>
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO | 
============================================================
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO | RESULTS
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO | ============================================================
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO |   Target state:    |101>
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO |   Measured state:  |101>
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO |   Success prob:    0.9453
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO |   Classical prob:  0.1250
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO |   Speedup:         7.56x
2026-03-01 05:05:19,761 | AdvancedExperiments | INFO |   Status: SUCCESS!
2026-03-01 05:05:19,762 | AdvancedExperiments | INFO | 
######################################################################
2026-03-01 05:05:19,762 | AdvancedExperiments | INFO | # EXPERIMENT 2: QED EFFECTS
2026-03-01 05:05:19,762 | AdvancedExperiments | INFO | ######################################################################
2026-03-01 05:05:19,762 | AdvancedExperiments | INFO | Lamb Shift Calculator initialized
2026-03-01 05:05:19,762 | AdvancedExperiments | INFO |   Fine structure constant: α = 0.0072973526
2026-03-01 05:05:19,762 | AdvancedExperiments | INFO | Anomalous Magnetic Moment Calculator initialized
2026-03-01 05:05:19,762 | AdvancedExperiments | INFO | 
QED Effects Experiment initialized
2026-03-01 05:05:19,762 | AdvancedExperiments | INFO |   α = 0.0072973526
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
======================================================================
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | QUANTUM ELECTRODYNAMICS (QED) EFFECTS ANALYSIS
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | ======================================================================
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
============================================================
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | LAMB SHIFT: 2s_{1/2} vs 2p_{1/2} (Z=1)
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | ============================================================
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
2s_{1/2} Lamb shift: 57.47 MHz
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 2p_{1/2} Lamb shift: 0.1598 MHz
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
Splitting (calculated): 57.31 MHz
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | Splitting (experimental): 1057.84 MHz
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
============================================================
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | ELECTRON ANOMALOUS MAGNETIC MOMENT (g-2)
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | ============================================================
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
Contributions:
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Order 1 (Schwinger): 0.001161409733
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Order 2:             0.000001772305
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Order 3:             0.000000014804
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Order 4:             -0.000000000056
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Order 5:             0.000000000001
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
  Total (calculated):  0.001163196787
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Experimental:        0.001159652181
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Error:               3.54e-06
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Relative error:      3.06e-03
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
g-factor:
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Calculated:   2.002326393574
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO |   Experimental: 2.002319304363
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
Hydrogen Energy Levels with QED Corrections:
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
======================================================================
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | QED ANALYSIS COMPLETE
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | ======================================================================
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | 
######################################################################
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | # EXPERIMENT 3: POLYATOMIC MOLECULES
2026-03-01 05:05:19,763 | AdvancedExperiments | INFO | ######################################################################
2026-03-01 05:05:19,772 | QuantumComputer | INFO | HamiltonianBackend: loaded weights/latest.pth
2026-03-01 05:05:19,798 | QuantumComputer | INFO | SchrodingerBackend: loaded weights/schrodinger_crystal_final.pth
2026-03-01 05:05:19,824 | QuantumComputer | INFO | DiracBackend: loaded weights/dirac_phase5_latest.pth
2026-03-01 05:05:19,825 | AdvancedExperiments | INFO | Polyatomic VQE Solver initialized
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO | Polyatomic Molecule Experiment initialized
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO |   Available molecules: ['H2O', 'NH3', 'CH4']
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO | 
======================================================================
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO | POLYATOMIC MOLECULE ANALYSIS: H2O
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO | ======================================================================
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO | 
Molecule: H2O
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO | Description: H2O: r_OH=0.9575 Å, ∠HOH=104.5°
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO | 
Geometry:
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO |   O: (0.0000, 0.0000, 0.0000) Å
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO |   H: (0.9575, 0.0000, 0.0000) Å
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO |   H: (-0.2397, 0.9270, 0.0000) Å
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO | 
Running PySCF for H2O
2026-03-01 05:05:19,826 | AdvancedExperiments | INFO |   Geometry: O 0.000000 0.000000 0.000000; H 0.957500 0.000000 0.000000; H -0.239739 0.927001 0.000000
2026-03-01 05:05:19,827 | AdvancedExperiments | INFO |   Electrons: 10
2026-03-01 05:05:19,827 | AdvancedExperiments | INFO |   Orbitals: 7
2026-03-01 05:05:19,827 | AdvancedExperiments | INFO |   Qubits (spin): 14
2026-03-01 05:05:19,854 | AdvancedExperiments | INFO |   HF Energy: -74.96297761 Ha
2026-03-01 05:05:20,037 | AdvancedExperiments | INFO |   FCI Energy: -75.01249437 Ha
2026-03-01 05:05:20,037 | AdvancedExperiments | INFO | 
Correlation Energy: 0.049517 Ha
2026-03-01 05:05:20,037 | AdvancedExperiments | INFO | 
======================================================================
2026-03-01 05:05:20,037 | AdvancedExperiments | INFO | ANALYSIS COMPLETE
2026-03-01 05:05:20,038 | AdvancedExperiments | INFO | ======================================================================
2026-03-01 05:05:20,038 | AdvancedExperiments | INFO | 
======================================================================
2026-03-01 05:05:20,038 | AdvancedExperiments | INFO | ALL EXPERIMENTS COMPLETE
2026-03-01 05:05:20,038 | AdvancedExperiments | INFO | ======================================================================

❯ python3 app.py
2026-03-01 16:51:27,096 | MolecularSimulator | INFO |   Loaded H2 from PySCF: HF=-1.11699900, FCI=-1.13730604, E_nuc=0.719969
2026-03-01 16:51:28,097 | QuantumComputer | INFO | HamiltonianBackend: loaded weights/latest.pth
2026-03-01 16:51:28,122 | QuantumComputer | INFO | SchrodingerBackend: loaded weights/schrodinger_crystal_final.pth
2026-03-01 16:51:28,148 | QuantumComputer | INFO | DiracBackend: loaded weights/dirac_phase5_latest.pth
Dipole MO matrix:
[[-3.28091797e-10 -9.27833470e-01]
 [-9.27833470e-01 -3.28091741e-10]]
Dipole: 4 Pauli terms, identity=0.000000

Building base Hamiltonian...
2026-03-01 16:51:28,197 | MolecularSimulator | INFO |   OpenFermion-PySCF: HF=-1.11699900, FCI=-1.13730604
2026-03-01 16:51:28,214 | MolecularSimulator | INFO |   OpenFermion verification:
2026-03-01 16:51:28,214 | MolecularSimulator | INFO |     Ground state: -1.13730604 Ha (FCI target: -1.13730604)
2026-03-01 16:51:28,214 | MolecularSimulator | INFO |     HF idx=3:  0.47475070 Ha
2026-03-01 16:51:28,215 | MolecularSimulator | INFO |     HF idx=12: -1.11699900 Ha
2026-03-01 16:51:28,215 | MolecularSimulator | INFO |     ✓ OpenFermion Hamiltonian verified
2026-03-01 16:51:28,215 | MolecularSimulator | INFO |   Using OpenFermion JW: 14 Pauli terms, E_nuc=-0.090579 Ha
2026-03-01 16:51:28,226 | MolecularSimulator | INFO |   Verification: E_HF(calc)=-1.11699900 Ha, E_HF(target)=-1.11699900 Ha
  Singles: [(0, 2), (0, 3), (1, 2), (1, 3)]
  Doubles: [(0, 1, 2, 3)]
  Total params: 5

============================================================
ANSATZ VERIFICATION
============================================================

  Single (0→2), θ=0.3:
    |1100> (2e): 14.448608
    |0110> (2e): 1.382571

  Single (0→3), θ=0.3:
    |1100> (2e): 14.428297
    |0101> (2e): 1.380628

  Single (1→2), θ=0.3:
    |1100> (2e): 14.509766
    |1010> (2e): 1.388424

  Single (1→3), θ=0.3:
    |1100> (2e): 14.486293
    |1001> (2e): 1.386178

  Dipole scan with single (0→2):
    θ₀=-0.30: E=-1.07150715, <μ>=-0.40752082
    θ₀=-0.10: E=-1.12979672, <μ>=-0.14338610
    θ₀=+0.00: E=-1.13730595, <μ>=+0.00000000
    θ₀=+0.10: E=-1.12979672, <μ>=+0.14338609
    θ₀=+0.30: E=-1.07150715, <μ>=+0.40752084

============================================================
STEP 1: ZERO-FIELD REFERENCE
============================================================
E(0) = -1.1373060358
θ = [-6.77278357e-09 -1.03333271e-08 -8.66274926e-09 -5.65363690e-08
 -1.11768514e-01]
ΔE_FCI = 2.89e-15

--- Diagnostic: same θ, different H(F) ---
  [zero-field θ] F=+0.00000: <H₀>=-1.1373060358, <μ>=0.0000000925, -F<μ>=-0.0000000000, E=-1.1373060358
  [zero-field θ] F=+0.00500: <H₀>=-1.1373060358, <μ>=0.0000000925, -F<μ>=-0.0000000005, E=-1.1373060362
  [zero-field θ] F=+0.01000: <H₀>=-1.1373060358, <μ>=0.0000000925, -F<μ>=-0.0000000009, E=-1.1373060367
  [zero-field θ] F=+0.02000: <H₀>=-1.1373060358, <μ>=0.0000000925, -F<μ>=-0.0000000019, E=-1.1373060376

============================================================
STEP 2: FIELD SWEEP
============================================================
  F=-0.00500: E=-1.1373404123, ΔE=-0.0000343765
  F=+0.00500: E=-1.1373404123, ΔE=-0.0000343765
  F=-0.01000: E=-1.1374435409, ΔE=-0.0001375052
  F=+0.01000: E=-1.1374435409, ΔE=-0.0001375052
  F=-0.01500: E=-1.1376154189, ΔE=-0.0003093832
  F=+0.01500: E=-1.1376154189, ΔE=-0.0003093832
  F=-0.02000: E=-1.1378560417, ΔE=-0.0005500059
  F=+0.02000: E=-1.1378560417, ΔE=-0.0005500059

============================================================
POLARIZABILITY ANALYSIS
============================================================

         F               E(F)             ΔE
  -0.02000      -1.1378560417  -0.0005500059
  -0.01500      -1.1376154189  -0.0003093832
  -0.01000      -1.1374435409  -0.0001375052
  -0.00500      -1.1373404123  -0.0000343765
  +0.00000      -1.1373060358  +0.0000000000
  +0.00500      -1.1373404123  -0.0000343765
  +0.01000      -1.1374435409  -0.0001375052
  +0.01500      -1.1376154189  -0.0003093832
  +0.02000      -1.1378560417  -0.0005500059

Symmetry |E(+F)-E(-F)|:
  ±0.0050: 2.22e-16
  ±0.0100: 3.11e-15
  ±0.0150: 2.36e-12
  ±0.0200: 1.11e-15

  α (VQE)  = 2.7500 a₀³
  α (exact diag, STO-3G) ≈ 2.750 a₀³
  Error = 0.0%

```


