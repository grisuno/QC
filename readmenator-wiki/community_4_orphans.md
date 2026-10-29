# orphans

*Community 4 | 3 files | cohesion 0.00*

## Definition

This community groups 3 file(s) rooted at `root` with dominant language py (cohesion 0.00). Central symbols: `AsyncSolver`, `Experience`, `Launcher`, `OrbitalInfo`, `Solved`, `System`, `__init__`, `_ao_values`. Core file: `immersive/matter.py` (19 symbols). Documented purpose: Ab-initio fields for atoms and molecules (PySCF) sampled on a cubic grid for GPU volume rendering..

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `immersive/matter.py` | py | utility | 19 | yes |
| `install.sh` | sh | utility | 0 | no |
| `qc_launcher.py` | py | utility | 13 | yes |

## Key Symbols

- `System` (class, `immersive/matter.py:30`) `class System`
- `_mol` (function, `immersive/matter.py:39`) `def _mol(key, title, atoms, spin, en, es)`
- `OrbitalInfo` (class, `immersive/matter.py:77`) `class OrbitalInfo`
- `Solved` (class, `immersive/matter.py:85`) `class Solved`
- `pyscf_available` (function, `immersive/matter.py:102`) `def pyscf_available()`
- `solve` (function, `immersive/matter.py:110`) `def solve(system)`
- `_ao_values` (function, `immersive/matter.py:139`) `def _ao_values(solved, n)`
- `_evaluate` (function, `immersive/matter.py:149`) `def _evaluate(solved, quadratic)` - Evaluate sum_ab ao_a ao_b D_ab (density-like) chunk-wise; `quadratic` returns the AO-basis matrix.
- `field_amplitude` (function, `immersive/matter.py:160`) `def field_amplitude(solved, kind)` - Signed amplitude a(r) with density a^2: sqrt(rho), signed sqrt(spin density), or an orbital.
- `exposure` (function, `immersive/matter.py:192`) `def exposure(amp, half)` - 1 / (99.7th percentile of a^2 inside the bounding sphere).
- `bonds` (function, `immersive/matter.py:201`) `def bonds(symbols, coords)`
- `AsyncSolver` (class, `immersive/matter.py:211`) `class AsyncSolver` - Runs PySCF off the render thread and caches finished systems.
- `__init__` (method, `immersive/matter.py:214`) `def __init__(self)`
- `request` (method, `immersive/matter.py:220`) `def request(self, system)`
- `work` (function, `immersive/matter.py:224`) `def work()`
- `prefetch` (method, `immersive/matter.py:236`) `def prefetch(self, system, kinds)` - Solve `system` and sample the given fields in one background job.
- `work` (function, `immersive/matter.py:242`) `def work()`
- `request_field` (method, `immersive/matter.py:261`) `def request_field(self, system, kind)`
- `work` (function, `immersive/matter.py:269`) `def work()`
- `Experience` (class, `qc_launcher.py:38`) `class Experience`
- `Launcher` (class, `qc_launcher.py:87`) `class Launcher(App)`
- `__init__` (method, `qc_launcher.py:105`) `def __init__(self)`
- `t` (method, `qc_launcher.py:111`) `def t(self, key)`
- `compose` (method, `qc_launcher.py:114`) `def compose(self)`
- `_options` (method, `qc_launcher.py:128`) `def _options(self)`
- `on_mount` (method, `qc_launcher.py:131`) `def on_mount(self)`
- `refresh_detail` (method, `qc_launcher.py:136`) `def refresh_detail(self)`
- `on_option_list_option_highlighted` (method, `qc_launcher.py:143`) `def on_option_list_option_highlighted(self, event)`
- `on_option_list_option_selected` (method, `qc_launcher.py:147`) `def on_option_list_option_selected(self, event)`
- `action_toggle_lang` (method, `qc_launcher.py:151`) `def action_toggle_lang(self)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (root: quantum_simulator) and community 4 (orphans).
- [INFERRED] shares_context community 1 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (immersive) and community 4 (orphans).
- [INFERRED] shares_context community 2 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 2 (root: quantum_framework_core) and community 4 (orphans).
- [INFERRED] shares_context community 3 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 3 (root: quantum_framework_molecular_v2) and community 4 (orphans).

## Risks

- [taint high] `qc_launcher.py` -> `qc_launcher.py` via `subprocess` (0 hops)

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- Is the dangerous import `subprocess` in `qc_launcher.py` still required, or can it be isolated?
- What would break if the most connected file in orphans changed?
- Should orphans be split, given cohesion 0.00?

## Sources

- `immersive/matter.py`
- `install.sh`
- `qc_launcher.py`
