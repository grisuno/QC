# Project Memory

> Cross-session context for agents. Sections 1-6 are regenerated from the source tree with zero LLM tokens: declared rules are quoted verbatim with `file:line`, measured baselines come from the scan. Section 7 is written by agents and humans and is preserved across rebuilds.

Generated from 43 files at commit `587dd9bbb732`. Read this first, then `readmenator-wiki/index.md`, then `readmenator . ask "<question>"` for anything specific.

## 1. Purpose and domain

- What it is: Q²C simulates quantum systems on classical hardware through two complementary modes selectable at runtime: (`README.md:13`)
- Domain vocabulary (term, files): `quantum` (28), `run` (26), `state` (23), `config` (22), `energy` (19), `all` (18), `single` (18), `author` (18), `gris` (18), `iscomeback` (18), `backend` (17), `entropy` (17), `molecular` (17), `create` (17), `make` (17)
- Subsystem `root: quantum_simulator`: 16 files, core `quantum_simulator.py`: Quantum Simulation Framework - Complete Edition
- Subsystem `immersive`: 12 files, core `immersive/scenes.py`: Interactive scenes: hydrogen orbital clouds, H2 molecular orbitals, Bloch sphere, entanglement.
- Subsystem `root: quantum_framework_core`: 10 files, core `quantum_framework_core.py`: Quantum Framework Core Module
- Subsystem `root: quantum_framework_molecular_v2`: 2 files, core `quantum_framework_molecular_v2.py`: Quantum Framework Molecular Module - Production Version
- Business rules that the code cannot show live in section 7: record them there.

## 2. Workflow

Detected commands:
- `python -m pytest -q` (tests/ layout)
- `CI workflow github-actions-demo.yml` (workflows/github-actions-demo.yml)

Session protocol:
1. Start: read this file, then `readmenator . fresh` (exit 1 means run `readmenator . --rebuild`).
2. Orient: `readmenator-wiki/index.md`; for a question use `readmenator . ask "<question>"` (local = entities + sources, `--global` = community reports).
3. Before editing a file: `grep -n '<file>' readmenator-agent/GOTCHAS.md readmenator-agent/SECURITY.md`.
4. After the change: run the tests above, then `readmenator . --rebuild` so the maps, wiki, and this file stay true.
5. End: record decisions, business rules, and gotchas with `readmenator . remember "<note>" --kind decision`.

## 3. Rules and constraints

Declared:
- none declared in instruction files (add them to AGENTS.md or record them in section 7)

Measured baseline:
- Security findings at medium or above: 0 (see `readmenator-agent/SECURITY.md`); do not add new ones.
- Dependency cycles: 0; layer violations: 0 (see `readmenator-agent/GOTCHAS.md`).

## 4. Style norms

Declared:
- none declared in instruction files (add them to AGENTS.md or record them in section 7)

Measured baseline:
- py: 42 files, 1985 symbols; docstrings on 36% of symbols; functions snake_case (100%); types PascalCase (100%); median file 686 lines, max 3232.
- sh: 1 files, 0 symbols; median file 4 lines, max 4.
- Tests: 3 files under .; follow the existing naming (e.g. `test_immersive.py`).

## 5. Minimum deliverables

Declared:
- none declared in instruction files (add them to AGENTS.md or record them in section 7)

Measured baseline:
- Tests pass: `python -m pytest -q`.
- Docstring coverage stays at or above 36%.
- No new security findings at medium or above (current: 0).
- No new dependency cycles (current: 0).
- Files stay under 300 lines where possible (`readmenator . lint`).
- Docs refreshed: `readmenator . --rebuild`, and decisions recorded in section 7.

## 6. Risks to respect

- God nodes (changes ripple widely): `quantum_computer.py`, `quantum_framework_core.py`, `qc_dashboard.py`, `quantum_framework_menu.py`, `quantum_simulator.py`
- Hotspots (complex and central): `qc_dashboard.py`, `quantum_framework_core.py`, `quantum_computer.py`, `test_qc_integration.py`, `quantum_simulator.py`
- Full blast radius: `readmenator-agent/GOTCHAS.md`; findings: `readmenator-agent/SECURITY.md`.

## 7. Session log (preserved across rebuilds)

Append with `readmenator . remember "<note>" --kind <kind>` (kinds: business, decision, rule, workflow, style, deliverable, gotcha, todo, note) or the MCP tool `readmenator.remember`. Record business rules, decisions and their reasons, workflow changes, and anything the next session must not rediscover.

<!-- readmenator:memory:notes:begin -->
<!-- readmenator:memory:notes:end -->
