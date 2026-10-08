# root: quantum_framework_menu

*Community 2 | 6 files | cohesion 0.39*

## Definition

This community groups 6 file(s) rooted at `root` with dominant language py (cohesion 0.39). Central symbols: `BackendComparisonRenderer`, `BackendComparisonResult`, `BlochSphereRenderer`, `BrutalConfig`, `BrutalDashboard`, `BrutalTheme`, `CircuitExecutor`, `EntropyPlotRenderer`. Core file: `quantum_framework_menu.py` (78 symbols). Documented purpose: Ultra-High Fidelity Quantum Visualization.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `quantum_3dview.py` | py | presentation | 27 | yes |
| `quantum_framework_main.py` | py | utility | 7 | yes |
| `quantum_framework_menu.py` | py | utility | 78 | yes |
| `quantum_framework_molecular.py` | py | utility | 29 | yes |
| `quantum_lab.py` | py | utility | 64 | yes |
| `quantum_visualizer.py` | py | utility | 54 | yes |

## Key Symbols

- `BrutalTheme` (class, `quantum_3dview.py:32`) `class BrutalTheme(Enum)`
- `BrutalConfig` (class, `quantum_3dview.py:39`) `class BrutalConfig`
- `colors` (method, `quantum_3dview.py:52`) `def colors(self)`
- `QuantumHologram` (class, `quantum_3dview.py:84`) `class QuantumHologram` - Visualizador holográfico 3D de estados cuánticos
- `__init__` (method, `quantum_3dview.py:87`) `def __init__(self, config)`
- `create_amplitude_hologram` (method, `quantum_3dview.py:92`) `def create_amplitude_hologram(self, snapshots, backend_comparison)` - Crea visualización holográfica de amplitudes en 3D con efecto de partículas
- `_add_holographic_field` (method, `quantum_3dview.py:148`) `def _add_holographic_field(self, fig, snapshots, row, col)` - Añade campo de amplitudes 3D con efecto de partículas flotantes
- `_add_entropy_trails` (method, `quantum_3dview.py:245`) `def _add_entropy_trails(self, fig, snapshots, row, col)` - Añade trazas de entropía con efecto de cola luminosa
- `_add_bloch_sphere_holographic` (method, `quantum_3dview.py:292`) `def _add_bloch_sphere_holographic(self, fig, snapshot, backend_name, row, col)` - Esfera de Bloch con efectos de holograma y partículas orbitales
- `_interpolate_color` (method, `quantum_3dview.py:389`) `def _interpolate_color(self, color1, color2, factor)` - Interpola entre dos colores hex
- `hex_to_rgb` (method, `quantum_3dview.py:391`) `def hex_to_rgb(hex_color)`
- `rgb_to_hex` (method, `quantum_3dview.py:395`) `def rgb_to_hex(rgb)`
- `QuantumNeuralTopology` (class, `quantum_3dview.py:403`) `class QuantumNeuralTopology` - Visualiza la topología interna de las redes neuronales cuánticas
- `__init__` (method, `quantum_3dview.py:406`) `def __init__(self, model)`
- `create_topology_map` (method, `quantum_3dview.py:410`) `def create_topology_map(self)` - Crea mapa 3D de la arquitectura neuronal con activaciones en tiempo real
- `_extract_layers` (method, `quantum_3dview.py:477`) `def _extract_layers(self, model)` - Extrae información de capas del modelo PyTorch
- `_generate_quantum_topology` (method, `quantum_3dview.py:490`) `def _generate_quantum_topology(self)` - Genera topología representativa de backend cuántico
- `QuantumSonification` (class, `quantum_3dview.py:500`) `class QuantumSonification` - Convierte estados cuánticos en audio para percepción alternativa
- `__init__` (method, `quantum_3dview.py:503`) `def __init__(self, sample_rate)`
- `state_to_audio` (method, `quantum_3dview.py:506`) `def state_to_audio(self, snapshot, duration)` - Convierte un estado cuántico en onda de audio
- `_adsr_envelope` (method, `quantum_3dview.py:538`) `def _adsr_envelope(self, length, intensity)` - Genera envolvente ADSR proporcional a la intensidad del estado
- `BrutalDashboard` (class, `quantum_3dview.py:560`) `class BrutalDashboard` - Dashboard interactivo completo con todas las visualizaciones brutales
- `__init__` (method, `quantum_3dview.py:563`) `def __init__(self, config)`
- `generate_full_report` (method, `quantum_3dview.py:570`) `def generate_full_report(self, snapshots, backend_comparison)` - Genera reporte completo con múltiples visualizaciones
- `_save_audio` (method, `quantum_3dview.py:605`) `def _save_audio(self, audio, path)` - Guarda audio como WAV
- `demo_brutal` (method, `quantum_3dview.py:614`) `def demo_brutal()` - Demostración de visualización brutal
- `create_synthetic_snapshots` (method, `quantum_3dview.py:653`) `def create_synthetic_snapshots()` - Crea datos sintéticos para demostración
- `setup_logging` (function, `quantum_framework_main.py:52`) `def setup_logging(verbose)` - Configure logging level based on verbosity.
- `run_benchmark` (function, `quantum_framework_main.py:61`) `def run_benchmark(args, config)` - Run scaling benchmark.
- `run_experiment` (function, `quantum_framework_main.py:101`) `def run_experiment(args, config, config_loader)` - Run a specific experiment by name.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 7
- Cross-boundary resolved imports (EXTRACTED): 11

## Connections

- [EXTRACTED] depends_on community 1 <-> 2 (strength 0.9): Extracted import edge crosses communities: qc_dashboard.py imports quantum_3dview.py.
- [EXTRACTED] depends_on community 2 <-> 0 (strength 0.9): Extracted import edge crosses communities: quantum_3dview.py imports quantum_computer.py.
- [INFERRED] bridges community 0 <-> 2 (strength 0.6): Inferred cross-community bridge: advanced_experiments.py reaches quantum_lab.py in 4 hops.
- [INFERRED] bridges community 2 <-> 0 (strength 0.5): Inferred cross-community bridge: quantum_lab.py reaches quantum_simulator.py in 5 hops.
- [INFERRED] bridges community 2 <-> 0 (strength 0.5): Inferred cross-community bridge: quantum_lab.py reaches relativistic_hydrogen.py in 5 hops.
- [INFERRED] shares_context community 2 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 2 (root: quantum_framework_menu) and community 3 (root: quantum_framework_molecular_v2).
- [INFERRED] shares_context community 2 <-> 4 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 2 (root: quantum_framework_menu) and community 4 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in root: quantum_framework_menu changed?
- Should root: quantum_framework_menu be split, given cohesion 0.39?

## Sources

- `quantum_3dview.py`
- `quantum_framework_main.py`
- `quantum_framework_menu.py`
- `quantum_framework_molecular.py`
- `quantum_lab.py`
- `quantum_visualizer.py`
