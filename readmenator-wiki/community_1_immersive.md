# immersive

*Community 1 | 12 files | cohesion 0.81*

## Definition

This community groups 12 file(s) rooted at `immersive` with dominant language py (cohesion 0.81). Central symbols: `App`, `AtomScene`, `BackendHub`, `BackendRegister`, `BlochScene`, `DiracBackendLink`, `EntangleScene`, `FullscreenPass`. Core file: `immersive/scenes.py` (73 symbols). Documented purpose: Immersive viewer entry: window, event loop, cinematic tour director..

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `immersive/__init__.py` | py | utility | 0 | no |
| `immersive/app.py` | py | utility | 20 | yes |
| `immersive/backend.py` | py | utility | 35 | yes |
| `immersive/glkit.py` | py | utility | 30 | yes |
| `immersive/hud.py` | py | utility | 19 | yes |
| `immersive/physics.py` | py | utility | 34 | yes |
| `immersive/postfx.py` | py | utility | 11 | yes |
| `immersive/scenes.py` | py | utility | 73 | yes |
| `immersive/shaders.py` | py | utility | 2 | yes |
| `immersive/spin.py` | py | utility | 22 | yes |
| `qc_immersive.py` | py | utility | 0 | yes |
| `test_immersive.py` | py | testing | 22 | yes |

## Key Symbols

- `Step` (class, `immersive/app.py:64`) `class Step`
- `build_tour` (function, `immersive/app.py:71`) `def build_tour()`
- `cmd` (function, `immersive/app.py:72`) `def cmd()`
- `spin` (function, `immersive/app.py:75`) `def spin(speed)`
- `App` (class, `immersive/app.py:92`) `class App`
- `__init__` (method, `immersive/app.py:93`) `def __init__(self, args)`
- `go` (method, `immersive/app.py:147`) `def go(self, name, instant)`
- `run_command` (method, `immersive/app.py:157`) `def run_command(self, cmd)`
- `start_tour` (method, `immersive/app.py:169`) `def start_tour(self)`
- `stop_tour` (method, `immersive/app.py:176`) `def stop_tour(self)`
- `advance_tour` (method, `immersive/app.py:182`) `def advance_tour(self)`
- `update_tour` (method, `immersive/app.py:197`) `def update_tour(self, dt)`
- `screenshot` (method, `immersive/app.py:214`) `def screenshot(self, path)`
- `handle_event` (method, `immersive/app.py:224`) `def handle_event(self, ev)`
- `update` (method, `immersive/app.py:264`) `def update(self, dt)`
- `draw_chrome` (method, `immersive/app.py:286`) `def draw_chrome(self)`
- `frame` (method, `immersive/app.py:329`) `def frame(self, dt)`
- `run` (method, `immersive/app.py:344`) `def run(self)`
- `parse_args` (function, `immersive/app.py:369`) `def parse_args(argv)`
- `main` (function, `immersive/app.py:382`) `def main(argv)`
- `BackendHub` (class, `immersive/backend.py:31`) `class BackendHub` - Loads the project's QuantumComputer once and exposes its physics backends.
- `shared` (method, `immersive/backend.py:39`) `def shared(cls)`
- `preload` (method, `immersive/backend.py:46`) `def preload(cls)`
- `__init__` (method, `immersive/backend.py:49`) `def __init__(self)`
- `backend` (method, `immersive/backend.py:67`) `def backend(self, name)`
- `new_state` (method, `immersive/backend.py:70`) `def new_state(self, n_qubits)`
- `_torch` (function, `immersive/backend.py:74`) `def _torch()`
- `to_complex` (function, `immersive/backend.py:80`) `def to_complex(state)`
- `BackendRegister` (class, `immersive/backend.py:85`) `class BackendRegister` - Animated n-qubit register whose state is a JointHilbertState from the project's simulator.
- `__init__` (method, `immersive/backend.py:88`) `def __init__(self, n_qubits, backend, seconds_per_gate)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 24
- Cross-boundary resolved imports (EXTRACTED): 5

## Connections

- [EXTRACTED] depends_on community 1 <-> 2 (strength 0.9): Extracted import edge crosses communities: immersive/backend.py imports quantum_lab.py.
- [EXTRACTED] depends_on community 1 <-> 0 (strength 0.9): Extracted import edge crosses communities: immersive/backend.py imports quantum_computer.py.
- [INFERRED] bridges community 1 <-> 2 (strength 0.5): Inferred cross-community bridge: immersive/__init__.py reaches orbital_visualizer2.py in 5 hops.
- [INFERRED] bridges community 1 <-> 2 (strength 0.5): Inferred cross-community bridge: immersive/__init__.py reaches quantum_framework_visualization.py in 5 hops.
- [INFERRED] bridges community 1 <-> 0 (strength 0.5): Inferred cross-community bridge: immersive/__init__.py reaches quantum_simulator.py in 5 hops.
- [INFERRED] bridges community 1 <-> 2 (strength 0.5): Inferred cross-community bridge: immersive/glkit.py reaches orbital_visualizer2.py in 5 hops.
- [INFERRED] bridges community 1 <-> 2 (strength 0.5): Inferred cross-community bridge: immersive/glkit.py reaches quantum_framework_visualization.py in 5 hops.
- [INFERRED] shares_context community 1 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (immersive) and community 3 (root: quantum_framework_molecular_v2).
- [INFERRED] shares_context community 1 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (immersive) and community 4 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `immersive/__init__.py`)? What purpose do they serve?
- What would break if the most connected file in immersive changed?
- Should immersive be split, given cohesion 0.81?

## Sources

- `immersive/__init__.py`
- `immersive/app.py`
- `immersive/backend.py`
- `immersive/glkit.py`
- `immersive/hud.py`
- `immersive/physics.py`
- `immersive/postfx.py`
- `immersive/scenes.py`
- `immersive/shaders.py`
- `immersive/spin.py`
- `qc_immersive.py`
- `test_immersive.py`
