# root: quantum_simulator

*Community 0 | 11 files | cohesion 0.60*

## Definition

This community groups 11 file(s) rooted at `root` with dominant language py (cohesion 0.60). Central symbols: `AdvancedExperimentRunner`, `AnomalousMagneticMoment`, `AtomData`, `BellState`, `CNOTGate`, `CZGate`, `CircuitInstruction`, `Config`. Core file: `quantum_simulator.py` (219 symbols). Documented purpose: Advanced Quantum Experiments - Extension Pack.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `advanced_experiments.py` | py | utility | 56 | yes |
| `app.py` | py | utility | 21 | yes |
| `entangled_hydrogen.py` | py | utility | 43 | yes |
| `higgs_four_lepton_analysis.py` | py | utility | 44 | yes |
| `higgs_quantum_analysis.py` | py | utility | 44 | yes |
| `molecular_sim.py` | py | utility | 26 | yes |
| `polarizability_v3.py` | py | utility | 21 | yes |
| `quantum_computer.py` | py | utility | 163 | yes |
| `quantum_simulator.py` | py | utility | 219 | yes |
| `relativistic_hydrogen.py` | py | utility | 58 | yes |
| `topological_hilbert_compression2.py` | py | utility | 95 | yes |

## Key Symbols

- `_make_logger` (function, `advanced_experiments.py:121`) `def _make_logger(name, level)`
- `GroverConfig` (class, `advanced_experiments.py:141`) `class GroverConfig` - Configuration for Grover's algorithm experiments.
- `GroverOracle` (class, `advanced_experiments.py:159`) `class GroverOracle` - Oracle for Grover's algorithm.
- `__init__` (method, `advanced_experiments.py:167`) `def __init__(self, n_qubits, marked_state)`
- `_validate` (method, `advanced_experiments.py:172`) `def _validate(self)`
- `apply` (method, `advanced_experiments.py:176`) `def apply(self, state, backend)` - Apply oracle: flip phase of marked state.
- `GroverDiffusionOperator` (class, `advanced_experiments.py:194`) `class GroverDiffusionOperator` - Diffusion operator (Grover diffusion / inversion about mean).
- `__init__` (method, `advanced_experiments.py:203`) `def __init__(self, n_qubits)`
- `apply` (method, `advanced_experiments.py:206`) `def apply(self, state, backend)` - Apply diffusion operator using Hadamard + Oracle on \|0> + Hadamard.
- `GroverSearch` (class, `advanced_experiments.py:240`) `class GroverSearch` - Complete Grover's algorithm implementation using existing quantum_computer.py infrastructure.
- `__init__` (method, `advanced_experiments.py:245`) `def __init__(self, config)`
- `_calculate_entropy` (method, `advanced_experiments.py:266`) `def _calculate_entropy(self, probs)` - Calculate Shannon entropy from probability distribution.
- `_init_quantum_computer` (method, `advanced_experiments.py:278`) `def _init_quantum_computer(self)` - Initialize quantum computer using existing quantum_computer.py infrastructure.
- `run` (method, `advanced_experiments.py:300`) `def run(self)` - Run Grover's search algorithm.
- `QEDConfig` (class, `advanced_experiments.py:404`) `class QEDConfig` - Configuration for QED effects calculations.
- `LambShiftCalculator` (class, `advanced_experiments.py:430`) `class LambShiftCalculator` - Calculates the Lamb shift using Bethe's formula and more accurate methods.
- `__init__` (method, `advanced_experiments.py:440`) `def __init__(self, config)`
- `bethe_formula` (method, `advanced_experiments.py:462`) `def bethe_formula(self, n, l, Z)` - Bethe's non-relativistic formula for Lamb shift.
- `_higher_l_shift` (method, `advanced_experiments.py:505`) `def _higher_l_shift(self, n, l, Z)` - Approximate Lamb shift for l > 0.
- `full_lamb_shift` (method, `advanced_experiments.py:518`) `def full_lamb_shift(self, n, l, j, Z)` - Calculate full Lamb shift including radiative corrections.
- `compare_2s_2p` (method, `advanced_experiments.py:558`) `def compare_2s_2p(self, Z)` - Compare Lamb shift for 2s_{1/2} and 2p_{1/2} states.
- `AnomalousMagneticMoment` (class, `advanced_experiments.py:595`) `class AnomalousMagneticMoment` - Calculates the electron's anomalous magnetic moment (g-2).
- `__init__` (method, `advanced_experiments.py:605`) `def __init__(self, config)`
- `schwinger_term` (method, `advanced_experiments.py:611`) `def schwinger_term(self)` - Schwinger's first-order result: a_e = α/(2π)
- `second_order` (method, `advanced_experiments.py:619`) `def second_order(self)` - Second-order correction: (α/π)^2 * C_2
- `third_order` (method, `advanced_experiments.py:627`) `def third_order(self)` - Third-order correction: (α/π)^3 * C_3
- `fourth_order` (method, `advanced_experiments.py:635`) `def fourth_order(self)` - Fourth-order correction: (α/π)^4 * C_4
- `fifth_order` (method, `advanced_experiments.py:643`) `def fifth_order(self)` - Fifth-order correction: (α/π)^5 * C_5
- `calculate_a_e` (method, `advanced_experiments.py:651`) `def calculate_a_e(self, order)` - Calculate anomalous magnetic moment to specified order.
- `full_report` (method, `advanced_experiments.py:686`) `def full_report(self)` - Generate a full report on g-2 calculations.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 19
- Cross-boundary resolved imports (EXTRACTED): 13

## Connections

- [EXTRACTED] depends_on community 1 <-> 0 (strength 0.9): Extracted import edge crosses communities: qc_dashboard.py imports quantum_computer.py.
- [EXTRACTED] depends_on community 2 <-> 0 (strength 0.9): Extracted import edge crosses communities: quantum_3dview.py imports quantum_computer.py.
- [INFERRED] bridges community 0 <-> 2 (strength 0.6): Inferred cross-community bridge: advanced_experiments.py reaches quantum_lab.py in 4 hops.
- [INFERRED] bridges community 2 <-> 0 (strength 0.5): Inferred cross-community bridge: quantum_lab.py reaches quantum_simulator.py in 5 hops.
- [INFERRED] bridges community 2 <-> 0 (strength 0.5): Inferred cross-community bridge: quantum_lab.py reaches relativistic_hydrogen.py in 5 hops.
- [INFERRED] bridges community 0 <-> 1 (strength 0.5): Inferred cross-community bridge: quantum_simulator.py reaches test_quantum_framework.py in 5 hops.
- [INFERRED] bridges community 0 <-> 1 (strength 0.5): Inferred cross-community bridge: relativistic_hydrogen.py reaches test_quantum_framework.py in 5 hops.
- [INFERRED] shares_context community 0 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (root: quantum_simulator) and community 3 (root: quantum_framework_molecular_v2).
- [INFERRED] shares_context community 0 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (root: quantum_simulator) and community 4 (orphans).

## Risks

- [taint medium] `higgs_four_lepton_analysis.py` -> `higgs_four_lepton_analysis.py` via `urllib.request` (0 hops)
- [taint medium] `higgs_four_lepton_analysis.py` -> `quantum_computer.py` via `urllib.request` (1 hops)
- [taint medium] `higgs_quantum_analysis.py` -> `higgs_quantum_analysis.py` via `urllib.request` (0 hops)
- [taint medium] `higgs_quantum_analysis.py` -> `quantum_computer.py` via `urllib.request` (1 hops)

## Open Questions

- Is the dangerous import `urllib.request` in `higgs_four_lepton_analysis.py` still required, or can it be isolated?
- What would break if the most connected file in root: quantum_simulator changed?
- Should root: quantum_simulator be split, given cohesion 0.60?

## Sources

- `advanced_experiments.py`
- `app.py`
- `entangled_hydrogen.py`
- `higgs_four_lepton_analysis.py`
- `higgs_quantum_analysis.py`
- `molecular_sim.py`
- `polarizability_v3.py`
- `quantum_computer.py`
- `quantum_simulator.py`
- `relativistic_hydrogen.py`
- `topological_hilbert_compression2.py`
