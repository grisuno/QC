# Second Brain

*Last synthesized: 2026-10-29 | 43 files | 5 concept pages | offline, zero tokens*

> Raw sources -> readmenator wiki -> links (Karpathy LLM Wiki Pattern, deterministic).
> Start here, then open one community page. Prefer grep over full reads.

## Vault Overview

The codebase centres on `quantum_computer.py`, `quantum_framework_core.py`, `qc_dashboard.py`. Architecturally it is 3 layers, dominant utility (38 files) across 5 import-based communities. Recorded risk surface: 0 security findings and 0 dependency cycles.

Surprising tissue lives between root: quantum_simulator, immersive, root: quantum_framework_core: 3 extracted cross-community imports and 12 inferred bridges. Follow `connections.json` sorted by strength before refactoring.

Open work clusters around documentation (95% file coverage), 0 security findings, 7 taint paths, and 5 suggested exploration questions in `queries.md`.

## Stats

| Metric | Value |
|--------|-------|
| Files | 43 |
| Symbols | 1985 |
| Resolved imports | 110 |
| Languages | py, sh |
| Communities | 5 |
| Doc coverage | 95% (41/43 files) |
| Security findings | 0 |
| Estimated read cost | ~28004 tokens (chars/4, offline so $0) |

## Reading Order

1. Skim Stats and God Nodes below for blast radius.
2. Open the largest community page first, then follow Connections.
3. Use `queries.md` for the next question; log the answer there.

```
grep -rn '<keyword>' index.md community_*.md
readmenator query "<question>" --target QC
```

## Concept Wiki

- [root: quantum_simulator (16 files, cohesion 0.74)](./community_0_root_quantum_simulator.md)
- [immersive (12 files, cohesion 0.81)](./community_1_immersive.md)
- [root: quantum_framework_core (10 files, cohesion 0.54)](./community_2_root_quantum_framework_core.md)
- [root: quantum_framework_molecular_v2 (2 files, cohesion 1.00)](./community_3_root_quantum_framework_molecular_v2.md)
- [orphans (3 files, cohesion 0.00)](./community_4_orphans.md)

## God Nodes

| File | Score |
|------|-------|
| `quantum_computer.py` | 42.3 |
| `quantum_framework_core.py` | 33.7 |
| `qc_dashboard.py` | 24.8 |
| `quantum_framework_menu.py` | 23.8 |
| `quantum_simulator.py` | 22.0 |

## Strongest Connections

- 1 -> 2: depends_on (strength 0.9, EXTRACTED)
- 1 -> 0: depends_on (strength 0.9, EXTRACTED)
- 2 -> 0: depends_on (strength 0.9, EXTRACTED)
- 1 -> 2: bridges (strength 0.5, INFERRED)
- 1 -> 2: bridges (strength 0.5, INFERRED)
- 1 -> 0: bridges (strength 0.5, INFERRED)
- 1 -> 2: bridges (strength 0.5, INFERRED)
- 1 -> 2: bridges (strength 0.5, INFERRED)
- 0 -> 3: shares_context (strength 0.5, INFERRED)
- 0 -> 4: shares_context (strength 0.5, INFERRED)

## Navigation Tips

- Obsidian Graph View works: every community page links back here.
- `connections.json` is machine-readable for GraphRAG pipelines.
- `REPORT.md` states what was extracted vs inferred and current limits.
- Regenerate offline: `readmenator . --rebuild` (no network, no tokens).
