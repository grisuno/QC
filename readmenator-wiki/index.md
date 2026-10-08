# Second Brain

*Last synthesized: 2026-10-07 | 30 files | 5 concept pages | offline, zero tokens*

> Raw sources -> readmenator wiki -> links (Karpathy LLM Wiki Pattern, deterministic).
> Start here, then open one community page. Prefer grep over full reads.

## Vault Overview

The codebase centres on `quantum_computer.py`, `quantum_framework_core.py`, `qc_dashboard.py`. Architecturally it is 3 layers, dominant utility (26 files) across 5 import-based communities. Recorded risk surface: 0 security findings and 0 dependency cycles.

Surprising tissue lives between root: quantum_simulator, root: quantum_framework_core, root: quantum_framework_menu: 3 extracted cross-community imports and 12 inferred bridges. Follow `connections.json` sorted by strength before refactoring.

Open work clusters around documentation (97% file coverage), 0 security findings, 4 taint paths, and 5 suggested exploration questions in `queries.md`.

## Stats

| Metric | Value |
|--------|-------|
| Files | 30 |
| Symbols | 1790 |
| Resolved imports | 78 |
| Languages | py, sh |
| Communities | 5 |
| Doc coverage | 97% (29/30 files) |
| Security findings | 0 |
| Estimated read cost | ~24992 tokens (chars/4, offline so $0) |

## Reading Order

1. Skim Stats and God Nodes below for blast radius.
2. Open the largest community page first, then follow Connections.
3. Use `queries.md` for the next question; log the answer there.

```
grep -rn '<keyword>' index.md community_*.md
readmenator query "<question>" --target readmenator_QC_tckz33e2
```

## Concept Wiki

- [root: quantum_simulator (11 files, cohesion 0.60)](./community_0_root_quantum_simulator.md)
- [root: quantum_framework_core (8 files, cohesion 0.53)](./community_1_root_quantum_framework_core.md)
- [root: quantum_framework_menu (6 files, cohesion 0.39)](./community_2_root_quantum_framework_menu.md)
- [root: quantum_framework_molecular_v2 (2 files, cohesion 1.00)](./community_3_root_quantum_framework_molecular_v2.md)
- [orphans (3 files, cohesion 0.00)](./community_4_orphans.md)

## God Nodes

| File | Score |
|------|-------|
| `quantum_computer.py` | 42.3 |
| `quantum_framework_core.py` | 33.6 |
| `qc_dashboard.py` | 24.8 |
| `quantum_simulator.py` | 23.9 |
| `quantum_framework_menu.py` | 23.8 |

## Strongest Connections

- 1 -> 0: depends_on (strength 0.9, EXTRACTED)
- 1 -> 2: depends_on (strength 0.9, EXTRACTED)
- 2 -> 0: depends_on (strength 0.9, EXTRACTED)
- 0 -> 2: bridges (strength 0.6, INFERRED)
- 2 -> 0: bridges (strength 0.5, INFERRED)
- 2 -> 0: bridges (strength 0.5, INFERRED)
- 0 -> 1: bridges (strength 0.5, INFERRED)
- 0 -> 1: bridges (strength 0.5, INFERRED)
- 0 -> 3: shares_context (strength 0.5, INFERRED)
- 0 -> 4: shares_context (strength 0.5, INFERRED)

## Navigation Tips

- Obsidian Graph View works: every community page links back here.
- `connections.json` is machine-readable for GraphRAG pipelines.
- `REPORT.md` states what was extracted vs inferred and current limits.
- Regenerate offline: `readmenator . --rebuild` (no network, no tokens).
