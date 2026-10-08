# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. 30 files, 1790 symbols, 512 imports. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM, Ruby, Swift, Kotlin, Scala, Lua, Elixir.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Start here:** Statistics Dashboard for scope, God Nodes for blast radius, Architecture Reference for per-file API. Agents: prefer `readmenator-agent/INDEX.md` + `SYMBOLS.md`.

**Wiki:** prefer `readmenator-wiki/index.md` for progressive disclosure: one synthesis page per community, `connections.json` with EXTRACTED vs INFERRED confidence, `queries.md` log, `REPORT.md` audit.

**Confidence:** EXTRACTED = parsed from source, INFERRED = heuristic bridge, AMBIGUOUS = reported, never hidden. See `readmenator-wiki/REPORT.md`.

**Total Files Parsed:** 30 | **Total Symbols Extracted:** 1790 | **Total Imports:** 512
 | **Resolved Imports:** 78

<!-- ranking_model: v1.0 | weights: {ppr:0.45,auth:0.2,test:0.15,doc:0.1,fresh:0.1} | alpha:0.85 | commit:1e0fd0b | date:2026-07-18 -->


## Table of Contents

1. [Statistics Dashboard](#statistics-dashboard)
2. [Architectural Layers](#architectural-layers)
3. [Ranked Context](#ranked-context)
4. [God Nodes](#god-nodes)
5. [Community Analysis](#community-analysis)
6. [Surprising Connections](#surprising-connections)
7. [Suggested Questions](#suggested-questions)
8. [Taint Propagation Map](#taint-propagation-map)
9. [Hotspot Analysis](#hotspot-analysis)
10. [Change Impact Analysis](#change-impact-analysis)
11. [Suggested Linting Rules](#suggested-linting-rules)
12. [Concept Graph](#concept-graph)
13. [Orphans](#orphans)
14. [Query Recipes](#query-recipes)
15. [Structural Knowledge Map](#structural-knowledge-map)
16. [UML Class Diagram](#uml-class-diagram)
17. [Code Property Graph](#code-property-graph)
18. [Architecture Reference](#architecture-reference)
    - [PY (29 files)](#py-29-files)
    - [SH (1 files)](#sh-1-files)

---

## Statistics Dashboard

| Metric | Value |
|--------|-------|
| Total Files | 30 |
| Total Symbols | 1790 |
| Total Imports | 512 |
| Call Edges | 11473 |
| Inheritance Edges | 110 |
| Languages | 2 |
| Avg Symbols/File | 59.7 |
| Avg Imports/File | 17.1 |
| Resolved Imports | 78 |

### Top Files by Import Count (Fan-Out)

| File | Imports | Symbols | Language |
|------|---------|---------|----------|
| `qc_dashboard.py` | 38 | 88 | py |
| `quantum_framework_menu.py` | 25 | 78 | py |
| `quantum_dash.py` | 24 | 71 | py |
| `quantum_lab.py` | 23 | 64 | py |
| `quantum_visualizer.py` | 23 | 54 | py |
| `test_qc_integration.py` | 23 | 87 | py |
| `entangled_hydrogen.py` | 22 | 43 | py |
| `quantum_framework_molecular_v2.py` | 22 | 61 | py |
| `molecular_sim.py` | 20 | 26 | py |
| `relativistic_hydrogen.py` | 20 | 58 | py |

---

## Architectural Layers

Auto-detected from path patterns, naming conventions, and imported frameworks.

| Layer | Files |
|-------|-------|
| utility | 26 |
| presentation | 2 |
| testing | 2 |

### utility

- `advanced_experiments.py` (py, 56 symbols)
- `app.py` (py, 21 symbols)
- `demo_molecular_vqe.py` (py, 10 symbols)
- `entangled_hydrogen.py` (py, 43 symbols)
- `higgs_four_lepton_analysis.py` (py, 44 symbols)
- `higgs_quantum_analysis.py` (py, 44 symbols)
- `install.sh` (sh, 0 symbols)
- `molecular_sim.py` (py, 26 symbols)
- `orbital_visualizer2.py` (py, 17 symbols)
- `polarizability_v3.py` (py, 21 symbols)
- `qc_dashboard.py` (py, 88 symbols)
- `qc_integration.py` (py, 61 symbols)
- `quantum_computer.py` (py, 163 symbols)
- `quantum_framework_core.py` (py, 196 symbols)
- `quantum_framework_main.py` (py, 7 symbols)
- *... and 11 more*

### presentation

- `quantum_3dview.py` (py, 27 symbols)
- `quantum_dash.py` (py, 71 symbols)

### testing

- `test_qc_integration.py` (py, 87 symbols)
- `test_quantum_framework.py` (py, 49 symbols)

---

## Ranked Context

Files ranked by composite score for the current query context. The ranking combines Personalized PageRank (query relevance), global authority, test coverage, documentation coverage, and code freshness. Model: v1.0.

| Rank | File | Composite | PPR | Authority | Test | Doc |
|------|------|-----------|-----|-----------|------|-----|
| 1 | `quantum_computer.py` | 0.1937 | 0.2225 | 0.2225 | 0.00 | 0.49 |
| 2 | `quantum_framework_main.py` | 0.1277 | 0.0207 | 0.0207 | 0.00 | 1.14 |
| 3 | `quantum_framework_core.py` | 0.1242 | 0.0796 | 0.0796 | 0.00 | 0.72 |
| 4 | `demo_molecular_vqe.py` | 0.1234 | 0.0207 | 0.0207 | 0.00 | 1.10 |
| 5 | `test_quantum_framework.py` | 0.1155 | 0.0207 | 0.0207 | 0.00 | 1.02 |
| 6 | `quantum_framework_menu.py` | 0.1121 | 0.0265 | 0.0265 | 0.00 | 0.95 |
| 7 | `quantum_framework_molecular_v2.py` | 0.1085 | 0.0382 | 0.0382 | 0.00 | 0.84 |
| 8 | `advanced_experiments.py` | 0.1020 | 0.0306 | 0.0306 | 0.00 | 0.82 |
| 9 | `quantum_framework_molecular.py` | 0.0953 | 0.0352 | 0.0352 | 0.00 | 0.72 |
| 10 | `quantum_3dview.py` | 0.0874 | 0.0262 | 0.0262 | 0.00 | 0.70 |

---

## God Nodes

Most architecturally central files ranked by combined import/export degree and symbol richness.

| File | Score | Connections | PageRank |
|------|-------|-------------|----------|
| `quantum_computer.py` | 42.3 | | 0.2225 |
| `quantum_framework_core.py` | 33.6 | | 0.0796 |
| `qc_dashboard.py` | 24.8 | | 0.0000 |
| `quantum_simulator.py` | 23.9 | | 0.0000 |
| `quantum_framework_menu.py` | 23.8 | | 0.0265 |
| `molecular_sim.py` | 16.6 | | 0.0000 |
| `advanced_experiments.py` | 15.6 | | 0.0306 |
| `quantum_visualizer.py` | 15.4 | | 0.0000 |
| `quantum_dash.py` | 15.1 | | 0.0000 |
| `test_qc_integration.py` | 14.7 | | 0.0000 |

---

## Community Analysis

Files grouped by import-based community detection. Cohesion measures how tightly connected each community is internally.

### root: quantum_simulator (Cohesion: 0.60)

**11 files** in this community:

- `advanced_experiments.py` (py, 56 symbols)
- `app.py` (py, 21 symbols)
- `entangled_hydrogen.py` (py, 43 symbols)
- `higgs_four_lepton_analysis.py` (py, 44 symbols)
- `higgs_quantum_analysis.py` (py, 44 symbols)
- `molecular_sim.py` (py, 26 symbols)
- `polarizability_v3.py` (py, 21 symbols)
- `quantum_computer.py` (py, 163 symbols)
- `quantum_simulator.py` (py, 219 symbols)
- `relativistic_hydrogen.py` (py, 58 symbols)
- `topological_hilbert_compression2.py` (py, 95 symbols)

### root: quantum_framework_core (Cohesion: 0.53)

**8 files** in this community:

- `orbital_visualizer2.py` (py, 17 symbols)
- `qc_dashboard.py` (py, 88 symbols)
- `qc_integration.py` (py, 61 symbols)
- `quantum_dash.py` (py, 71 symbols)
- `quantum_framework_core.py` (py, 196 symbols)
- `quantum_framework_visualization.py` (py, 20 symbols)
- `test_qc_integration.py` (py, 87 symbols)
- `test_quantum_framework.py` (py, 49 symbols)

### root: quantum_framework_menu (Cohesion: 0.39)

**6 files** in this community:

- `quantum_3dview.py` (py, 27 symbols)
- `quantum_framework_main.py` (py, 7 symbols)
- `quantum_framework_menu.py` (py, 78 symbols)
- `quantum_framework_molecular.py` (py, 29 symbols)
- `quantum_lab.py` (py, 64 symbols)
- `quantum_visualizer.py` (py, 54 symbols)

### root: quantum_framework_molecular_v2 (Cohesion: 1.00)

**2 files** in this community:

- `demo_molecular_vqe.py` (py, 10 symbols)
- `quantum_framework_molecular_v2.py` (py, 61 symbols)

---

## Surprising Connections

Files in different communities connected through 3+ indirect hops.

- `quantum_lab.py` <-> `quantum_simulator.py` (5 hops, across 3 communities)
- `quantum_lab.py` <-> `relativistic_hydrogen.py` (5 hops, across 3 communities)
- `quantum_simulator.py` <-> `test_quantum_framework.py` (5 hops, across 2 communities)
- `relativistic_hydrogen.py` <-> `test_quantum_framework.py` (5 hops, across 2 communities)
- `advanced_experiments.py` <-> `quantum_lab.py` (4 hops, across 3 communities)

---

## Suggested Questions

Auto-generated exploration prompts based on graph structure:

- What does quantum_computer.py depend on, and what depends on it? (13 connections)
- What does quantum_framework_core.py depend on, and what depends on it? (7 connections)
- What does qc_dashboard.py depend on, and what depends on it? (8 connections)
- How are the 11 files in 'root: quantum_simulator' related to each other?
- Why are quantum_lab.py and quantum_simulator.py connected through 5 hops across 3 communities?

---

## Taint Propagation Map

Taint analysis traces how dangerous imports propagate through the codebase via transitive dependencies. Source files import dangerous modules directly; sink files receive the danger indirectly.

**Taint Sources:** 2 | **Taint Sinks:** 3 | **Propagation Paths:** 4

- `higgs_four_lepton_analysis.py` imports `urllib.request` (0 hop to `higgs_four_lepton_analysis.py`) [medium]
  Path: higgs_four_lepton_analysis.py
- `higgs_four_lepton_analysis.py` imports `urllib.request` (1 hop to `quantum_computer.py`) [medium]
  Path: higgs_four_lepton_analysis.py -> quantum_computer.py
- `higgs_quantum_analysis.py` imports `urllib.request` (0 hop to `higgs_quantum_analysis.py`) [medium]
  Path: higgs_quantum_analysis.py
- `higgs_quantum_analysis.py` imports `urllib.request` (1 hop to `quantum_computer.py`) [medium]
  Path: higgs_quantum_analysis.py -> quantum_computer.py

---

## Hotspot Analysis

Files ranked by combined complexity (symbol count) and centrality (connection count). High-scoring files are architecturally critical and may need refactoring attention.

| File | Complexity | Centrality | Combined | Symbols | Connections |
|------|-----------|------------|----------|---------|-------------|
| `quantum_computer.py` | 0.744 | 0.500 | 0.598 | 163 | 33 |
| `quantum_framework_main.py` | 0.032 | 0.197 | 0.131 | 7 | 13 |
| `quantum_framework_core.py` | 0.895 | 0.455 | 0.631 | 196 | 30 |
| `demo_molecular_vqe.py` | 0.046 | 0.167 | 0.118 | 10 | 11 |
| `test_quantum_framework.py` | 0.224 | 0.197 | 0.208 | 49 | 13 |
| `quantum_framework_menu.py` | 0.356 | 0.500 | 0.443 | 78 | 33 |
| `quantum_framework_molecular_v2.py` | 0.279 | 0.364 | 0.330 | 61 | 24 |
| `advanced_experiments.py` | 0.256 | 0.364 | 0.321 | 56 | 24 |
| `quantum_framework_molecular.py` | 0.132 | 0.258 | 0.207 | 29 | 17 |
| `quantum_3dview.py` | 0.123 | 0.303 | 0.231 | 27 | 20 |
| `qc_dashboard.py` | 0.402 | 1.000 | 0.761 | 88 | 66 |
| `quantum_simulator.py` | 1.000 | 0.273 | 0.564 | 219 | 18 |
| `test_qc_integration.py` | 0.397 | 0.621 | 0.532 | 87 | 41 |
| `quantum_dash.py` | 0.324 | 0.424 | 0.384 | 71 | 28 |
| `topological_hilbert_compression2.py` | 0.434 | 0.303 | 0.355 | 95 | 20 |

---

## Concept Graph

Semantic second-brain layer: nouns are concept nodes, verbs are edges. Each noun maps atomically to a file set (EXTRACTED); each verb aggregates structural imports, calls, and inherits into consumes, invokes, extends, depends_on, or bridges (INFERRED).

**50 concepts, 100 relations.**

| Concept | Files | Mentions |
|---------|-------|----------|
| `quantum` | 26 | 229 |
| `run` | 24 | 126 |
| `state` | 21 | 227 |
| `config` | 21 | 58 |
| `author` | 19 | 19 |
| `gris` | 19 | 19 |
| `iscomeback` | 19 | 19 |
| `agpl` | 18 | 18 |
| `license` | 18 | 18 |
| `energy` | 17 | 76 |
| `all` | 17 | 66 |
| `molecular` | 17 | 55 |
| `hamiltonian` | 16 | 85 |
| `compute` | 16 | 71 |
| `create` | 16 | 47 |
| `single` | 16 | 41 |
| `make` | 16 | 17 |
| `framework` | 15 | 63 |
| `using` | 15 | 59 |
| `entropy` | 15 | 56 |
| `full` | 15 | 35 |
| `uses` | 15 | 35 |
| `logger` | 15 | 19 |
| `apply` | 14 | 150 |
| `circuit` | 14 | 103 |
| `qubits` | 14 | 69 |
| `build` | 14 | 60 |
| `wavefunction` | 14 | 42 |
| `data` | 14 | 41 |
| `states` | 14 | 40 |

### Verb Edges

| Source | Verb | Target | Strength | Evidence |
|--------|------|--------|----------|----------|
| `quantum` | `depends_on` | `run` | 1.00 | 10 |
| `quantum` | `depends_on` | `hamiltonian` | 0.95 | 10 |
| `quantum` | `depends_on` | `build` | 0.92 | 10 |
| `quantum` | `depends_on` | `logger` | 0.92 | 10 |
| `quantum` | `depends_on` | `state` | 0.92 | 10 |
| `run` | `depends_on` | `hamiltonian` | 0.92 | 10 |
| `run` | `depends_on` | `quantum` | 0.92 | 10 |
| `config` | `depends_on` | `quantum` | 0.90 | 10 |
| `config` | `depends_on` | `run` | 0.90 | 10 |
| `quantum` | `depends_on` | `config` | 0.90 | 10 |
| `quantum` | `depends_on` | `make` | 0.90 | 10 |
| `run` | `depends_on` | `build` | 0.90 | 10 |
| `run` | `depends_on` | `logger` | 0.90 | 10 |
| `state` | `depends_on` | `run` | 0.90 | 10 |
| `run` | `depends_on` | `make` | 0.87 | 10 |
| `run` | `depends_on` | `state` | 0.87 | 10 |
| `state` | `depends_on` | `hamiltonian` | 0.87 | 10 |
| `state` | `depends_on` | `logger` | 0.87 | 10 |
| `config` | `depends_on` | `hamiltonian` | 0.85 | 10 |
| `config` | `depends_on` | `state` | 0.85 | 10 |
| `quantum` | `depends_on` | `backend` | 0.85 | 10 |
| `quantum` | `depends_on` | `data` | 0.85 | 10 |
| `run` | `depends_on` | `config` | 0.85 | 10 |
| `run` | `depends_on` | `data` | 0.85 | 10 |
| `state` | `depends_on` | `build` | 0.85 | 10 |
| `state` | `depends_on` | `make` | 0.85 | 10 |
| `state` | `depends_on` | `quantum` | 0.85 | 10 |
| `all` | `depends_on` | `run` | 0.82 | 10 |
| `config` | `depends_on` | `build` | 0.82 | 10 |
| `config` | `depends_on` | `logger` | 0.82 | 10 |

### Dialectic Prompts

- Thesis: `agpl` centralizes 18 files; Antithesis: `all` pulls 17 files with 11 shared (Jaccard 0.46); Synthesis: should they merge, split by layer, or keep `bridges` explicit?
- Thesis: `agpl` centralizes 18 files; Antithesis: `amplitude` pulls 11 files with 8 shared (Jaccard 0.38); Synthesis: should they merge, split by layer, or keep `bridges` explicit?
- Thesis: `agpl` centralizes 18 files; Antithesis: `apply` pulls 14 files with 9 shared (Jaccard 0.39); Synthesis: should they merge, split by layer, or keep `bridges` explicit?
- Thesis: `agpl` centralizes 18 files; Antithesis: `author` pulls 19 files with 18 shared (Jaccard 0.95); Synthesis: should they merge, split by layer, or keep `bridges` explicit?
- Thesis: `agpl` centralizes 18 files; Antithesis: `backend` pulls 13 files with 8 shared (Jaccard 0.35); Synthesis: should they merge, split by layer, or keep `bridges` explicit?
- Thesis: `agpl` centralizes 18 files; Antithesis: `bell` pulls 12 files with 9 shared (Jaccard 0.43); Synthesis: should they merge, split by layer, or keep `bridges` explicit?
- Thesis: `agpl` centralizes 18 files; Antithesis: `build` pulls 14 files with 9 shared (Jaccard 0.39); Synthesis: should they merge, split by layer, or keep `bridges` explicit?
- Thesis: `agpl` centralizes 18 files; Antithesis: `builder` pulls 12 files with 7 shared (Jaccard 0.30); Synthesis: should they merge, split by layer, or keep `bridges` explicit?
- Thesis: `agpl` centralizes 18 files; Antithesis: `circuit` pulls 14 files with 8 shared (Jaccard 0.33); Synthesis: should they merge, split by layer, or keep `bridges` explicit?
- Thesis: `agpl` centralizes 18 files; Antithesis: `compute` pulls 16 files with 12 shared (Jaccard 0.55); Synthesis: should they merge, split by layer, or keep `bridges` explicit?

---

## Change Impact Analysis

Files sorted by how many other files would be affected if they changed. High-impact files should be changed with caution.

| File | Direct Dependents | Transitive Dependents | Total Impact |
|------|------------------|----------------------|--------------|
| `quantum_computer.py` | 13 | 3 | 16 |
| `molecular_sim.py` | 6 | 5 | 11 |
| `relativistic_hydrogen.py` | 2 | 6 | 8 |
| `quantum_framework_core.py` | 7 | 0 | 7 |
| `quantum_simulator.py` | 1 | 6 | 7 |
| `advanced_experiments.py` | 1 | 5 | 6 |
| `quantum_visualizer.py` | 2 | 3 | 5 |
| `quantum_3dview.py` | 2 | 2 | 4 |
| `quantum_dash.py` | 2 | 2 | 4 |
| `quantum_framework_molecular.py` | 2 | 1 | 3 |
| `app.py` | 1 | 1 | 2 |
| `higgs_four_lepton_analysis.py` | 1 | 1 | 2 |
| `orbital_visualizer2.py` | 1 | 1 | 2 |
| `qc_integration.py` | 2 | 0 | 2 |
| `quantum_framework_visualization.py` | 1 | 1 | 2 |

---

## Suggested Linting Rules

Automatically suggested linting and security rules based on patterns detected in the codebase. These can be exported as Semgrep rules using the `--export-rules` flag.

| Rule ID | Severity | Description | Language | Matches |
|---------|----------|-------------|----------|---------|
| `RM002` | warning | Bare except clause catches all exceptions including SystemExit | python | 4 |
| `RM001` | info | Large number of functions in py: 1459 total | py | 1459 |
| `RM003` | info | Print statement found (consider logging instead) | python | 1013 |

---

## Orphans

Files with no documentation or low connectivity. These are candidates for documentation investment or cleanup.

- `install.sh` (0 symbols, no doc)

---

## Query Recipes

Example queries you can run against this knowledge base using the ranking engine:

```
# Find files most relevant to a concept
readmenator query "Where is the import resolver implemented?"

# Rank files by relevance to a topic
readmenator query "How does documentation generation work?"

# Explain why a file ranks highly
readmenator query "explain readmenator/_documentation.py"

# Trace dependency paths with ranked context
readmenator query "path from CLI to exporter"
```

The ranking model uses the following signals:

- **Personalized PageRank** (45% weight): query-specific relevance via seed propagation
- **Global Authority** (20% weight): structural importance via standard PageRank
- **Test Coverage** (15% weight): fraction of symbols referenced in test files
- **Doc Coverage** (10% weight): presence of docstrings and file-level docs
- **Freshness** (10% weight): recent modification activity

Results include score decomposition and justification paths for each ranked item.

---

## Structural Knowledge Map

```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
    subgraph community_1 ["root: quantum_framework_core"]
    qc_dashboard_py["qc_dashboard.py (py)"]
    class qc_dashboard_py mod;
    qc_dashboard_py_DashboardConfig["DashboardConfig"]
    class qc_dashboard_py_DashboardConfig cls;
    qc_dashboard_py --> qc_dashboard_py_DashboardConfig
    qc_dashboard_py_GateItem["GateItem"]
    class qc_dashboard_py_GateItem cls;
    qc_dashboard_py --> qc_dashboard_py_GateItem
    qc_dashboard_py_SnapshotData["SnapshotData"]
    class qc_dashboard_py_SnapshotData cls;
    qc_dashboard_py --> qc_dashboard_py_SnapshotData
    qc_dashboard_py_H2VQESolver["H2VQESolver"]
    class qc_dashboard_py_H2VQESolver cls;
    qc_dashboard_py --> qc_dashboard_py_H2VQESolver
    qc_dashboard_py_VisualisationEngine["VisualisationEngine"]
    class qc_dashboard_py_VisualisationEngine cls;
    qc_dashboard_py --> qc_dashboard_py_VisualisationEngine
    test_qc_integration_py["test_qc_integration.py (py)"]
    class test_qc_integration_py mod;
    end
    subgraph community_2 ["root: quantum_framework_menu"]
    quantum_framework_menu_py["quantum_framework_menu.py (py)"]
    class quantum_framework_menu_py mod;
    quantum_dash_py["quantum_dash.py (py)"]
    class quantum_dash_py mod;
    quantum_visualizer_py["quantum_visualizer.py (py)"]
    class quantum_visualizer_py mod;
    end
    subgraph community_0 ["root: quantum_simulator"]
    entangled_hydrogen_py["entangled_hydrogen.py (py)"]
    class entangled_hydrogen_py mod;
    quantum_lab_py["quantum_lab.py (py)"]
    class quantum_lab_py mod;
    advanced_experiments_py["advanced_experiments.py (py)"]
    class advanced_experiments_py mod;
    molecular_sim_py["molecular_sim.py (py)"]
    class molecular_sim_py mod;
    end
    subgraph community_3 ["root: quantum_framework_molecular_v2"]
    quantum_framework_molecular_v2_py["quantum_framework_molecular_v2.py (py)"]
    class quantum_framework_molecular_v2_py mod;
    topological_hilbert_compression2_py["topological_hilbert_compression2.py (py)"]
    class topological_hilbert_compression2_py mod;
    relativistic_hydrogen_py["relativistic_hydrogen.py (py)"]
    class relativistic_hydrogen_py mod;
    higgs_four_lepton_analysis_py["higgs_four_lepton_analysis.py (py)"]
    class higgs_four_lepton_analysis_py mod;
    higgs_quantum_analysis_py["higgs_quantum_analysis.py (py)"]
    class higgs_quantum_analysis_py mod;
    qc_integration_py["qc_integration.py (py)"]
    class qc_integration_py mod;
    quantum_3dview_py["quantum_3dview.py (py)"]
    class quantum_3dview_py mod;
    quantum_simulator_py["quantum_simulator.py (py)"]
    class quantum_simulator_py mod;
    quantum_framework_core_py["quantum_framework_core.py (py)"]
    class quantum_framework_core_py mod;
    quantum_framework_molecular_fixed_py["quantum_framework_molecular_fixed.py (py)"]
    class quantum_framework_molecular_fixed_py mod;
    app_py["app.py (py)"]
    class app_py mod;
    polarizability_v3_py["polarizability_v3.py (py)"]
    class polarizability_v3_py mod;
    quantum_framework_molecular_py["quantum_framework_molecular.py (py)"]
    class quantum_framework_molecular_py mod;
    quantum_framework_visualization_py["quantum_framework_visualization.py (py)"]
    class quantum_framework_visualization_py mod;
    quantum_computer_py["quantum_computer.py (py)"]
    class quantum_computer_py mod;
    test_quantum_framework_py["test_quantum_framework.py (py)"]
    class test_quantum_framework_py mod;
    quantum_framework_main_py["quantum_framework_main.py (py)"]
    class quantum_framework_main_py mod;
    orbital_visualizer2_py["orbital_visualizer2.py (py)"]
    class orbital_visualizer2_py mod;
    demo_molecular_vqe_py["demo_molecular_vqe.py (py)"]
    class demo_molecular_vqe_py mod;
    quantum_framework_physics_py["quantum_framework_physics.py (py)"]
    class quantum_framework_physics_py mod;
    install_sh["install.sh (sh)"]
    class install_sh mod;
    end
    advanced_experiments_py -- resolved_imports --> quantum_computer_py
    advanced_experiments_py -- resolved_imports --> quantum_simulator_py
    advanced_experiments_py -- resolved_imports --> relativistic_hydrogen_py
    advanced_experiments_py -- resolved_imports --> molecular_sim_py
    app_py -- resolved_imports --> quantum_computer_py
    app_py -- resolved_imports --> molecular_sim_py
    demo_molecular_vqe_py -- resolved_imports --> quantum_framework_molecular_v2_py
    demo_molecular_vqe_py -- resolved_imports --> quantum_framework_molecular_v2_py
    entangled_hydrogen_py -- resolved_imports --> quantum_computer_py
    entangled_hydrogen_py -- resolved_imports --> quantum_computer_py
    entangled_hydrogen_py -- resolved_imports --> molecular_sim_py
    entangled_hydrogen_py -- resolved_imports --> relativistic_hydrogen_py
    higgs_four_lepton_analysis_py -- resolved_imports --> quantum_computer_py
    higgs_quantum_analysis_py -- resolved_imports --> quantum_computer_py
    molecular_sim_py -- resolved_imports --> quantum_computer_py
    molecular_sim_py -- resolved_imports --> quantum_computer_py
    molecular_sim_py -- resolved_imports --> quantum_computer_py
    polarizability_v3_py -- resolved_imports --> quantum_computer_py
    polarizability_v3_py -- resolved_imports --> molecular_sim_py
    qc_dashboard_py -- resolved_imports --> quantum_framework_core_py
    qc_dashboard_py -- resolved_imports --> quantum_computer_py
    qc_dashboard_py -- resolved_imports --> quantum_framework_core_py
    qc_dashboard_py -- resolved_imports --> quantum_framework_core_py
    qc_dashboard_py -- resolved_imports --> quantum_computer_py
    qc_dashboard_py -- resolved_imports --> orbital_visualizer2_py
    qc_dashboard_py -- resolved_imports --> quantum_framework_visualization_py
    qc_dashboard_py -- resolved_imports --> quantum_dash_py
    qc_dashboard_py -- resolved_imports --> quantum_3dview_py
    qc_dashboard_py -- resolved_imports --> qc_integration_py
    qc_dashboard_py -- resolved_imports --> quantum_computer_py
    qc_dashboard_py -- resolved_imports --> qc_integration_py
    qc_dashboard_py -- resolved_imports --> qc_integration_py
    qc_integration_py -- resolved_imports --> quantum_framework_core_py
    qc_integration_py -- resolved_imports --> quantum_framework_core_py
    qc_integration_py -- resolved_imports --> quantum_computer_py
    qc_integration_py -- resolved_imports --> quantum_computer_py
    quantum_3dview_py -- resolved_imports --> quantum_computer_py
    quantum_3dview_py -- resolved_imports --> quantum_visualizer_py
    quantum_dash_py -- resolved_imports --> quantum_computer_py
    quantum_dash_py -- resolved_imports --> molecular_sim_py
    quantum_framework_main_py -- resolved_imports --> quantum_framework_core_py
    quantum_framework_main_py -- resolved_imports --> quantum_framework_menu_py
    quantum_framework_main_py -- resolved_imports --> quantum_lab_py
    quantum_framework_menu_py -- resolved_imports --> quantum_framework_core_py
    quantum_framework_menu_py -- resolved_imports --> quantum_framework_molecular_py
    quantum_framework_menu_py -- resolved_imports --> higgs_four_lepton_analysis_py
    quantum_framework_menu_py -- resolved_imports --> quantum_dash_py
    quantum_framework_menu_py -- resolved_imports --> quantum_3dview_py
    quantum_framework_menu_py -- resolved_imports --> quantum_visualizer_py
    quantum_framework_menu_py -- resolved_imports --> app_py
    quantum_lab_py -- resolved_imports --> quantum_framework_core_py
    quantum_lab_py -- resolved_imports --> quantum_framework_molecular_py
    quantum_visualizer_py -- resolved_imports --> quantum_computer_py
    quantum_visualizer_py -- resolved_imports --> molecular_sim_py
    quantum_visualizer_py -- resolved_imports --> advanced_experiments_py
    test_qc_integration_py -- resolved_imports --> qc_integration_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> qc_dashboard_py
    test_qc_integration_py -- resolved_imports --> quantum_framework_core_py
    test_qc_integration_py -- resolved_imports --> quantum_framework_core_py
    test_quantum_framework_py -- resolved_imports --> quantum_framework_core_py
    test_quantum_framework_py -- resolved_imports --> quantum_framework_core_py
    test_quantum_framework_py -- resolved_imports --> quantum_framework_core_py
    topological_hilbert_compression2_py -- resolved_imports --> quantum_computer_py
    topological_hilbert_compression2_py -- resolved_imports --> quantum_computer_py
    ext___future__["__future__"]
    class ext___future__ ext;
    advanced_experiments_py -.->|imports| ext___future__
    ext_logging["logging"]
    class ext_logging ext;
    advanced_experiments_py -.->|imports| ext_logging
    ext_math["math"]
    class ext_math ext;
    advanced_experiments_py -.->|imports| ext_math
    ext_os["os"]
    class ext_os ext;
    advanced_experiments_py -.->|imports| ext_os
    ext_sys["sys"]
    class ext_sys ext;
    advanced_experiments_py -.->|imports| ext_sys
    ext_warnings["warnings"]
    class ext_warnings ext;
    advanced_experiments_py -.->|imports| ext_warnings
    ext_dataclasses["dataclasses"]
    class ext_dataclasses ext;
    advanced_experiments_py -.->|imports| ext_dataclasses
    ext_typing["typing"]
    class ext_typing ext;
    advanced_experiments_py -.->|imports| ext_typing
    ext_abc["abc"]
    class ext_abc ext;
    advanced_experiments_py -.->|imports| ext_abc
    ext_numpy["numpy"]
    class ext_numpy ext;
    advanced_experiments_py -.->|imports| ext_numpy
    ext_torch["torch"]
    class ext_torch ext;
    advanced_experiments_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    advanced_experiments_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    advanced_experiments_py -.->|imports| ext_torch_nn_functional
    ext_quantum_computer["quantum_computer"]
    class ext_quantum_computer ext;
    advanced_experiments_py -.->|imports| ext_quantum_computer
    ext_quantum_simulator["quantum_simulator"]
    class ext_quantum_simulator ext;
    advanced_experiments_py -.->|imports| ext_quantum_simulator
    ext_relativistic_hydrogen["relativistic_hydrogen"]
    class ext_relativistic_hydrogen ext;
    advanced_experiments_py -.->|imports| ext_relativistic_hydrogen
    ext_molecular_sim["molecular_sim"]
    class ext_molecular_sim ext;
    advanced_experiments_py -.->|imports| ext_molecular_sim
    ext_argparse["argparse"]
    class ext_argparse ext;
    advanced_experiments_py -.->|imports| ext_argparse
    ext_pyscf["pyscf"]
    class ext_pyscf ext;
    advanced_experiments_py -.->|imports| ext_pyscf
    app_py -.->|imports| ext___future__
    app_py -.->|imports| ext_warnings
    app_py -.->|imports| ext_math
    app_py -.->|imports| ext_numpy
    app_py -.->|imports| ext_sys
    app_py -.->|imports| ext_dataclasses
    app_py -.->|imports| ext_typing
    app_py -.->|imports| ext_torch
    app_py -.->|imports| ext_quantum_computer
    app_py -.->|imports| ext_molecular_sim
    app_py -.->|imports| ext_pyscf
    ext_openfermion_transforms["openfermion.transforms"]
    class ext_openfermion_transforms ext;
    app_py -.->|imports| ext_openfermion_transforms
    ext_openfermion_ops["openfermion.ops"]
    class ext_openfermion_ops ext;
    app_py -.->|imports| ext_openfermion_ops
    ext_scipy_optimize["scipy.optimize"]
    class ext_scipy_optimize ext;
    app_py -.->|imports| ext_scipy_optimize
    demo_molecular_vqe_py -.->|imports| ext_logging
    demo_molecular_vqe_py -.->|imports| ext_sys
    ext_time["time"]
    class ext_time ext;
    demo_molecular_vqe_py -.->|imports| ext_time
    demo_molecular_vqe_py -.->|imports| ext_typing
    ext_quantum_framework_molecular_v2["quantum_framework_molecular_v2"]
    class ext_quantum_framework_molecular_v2 ext;
    demo_molecular_vqe_py -.->|imports| ext_quantum_framework_molecular_v2
    demo_molecular_vqe_py -.->|imports| ext_quantum_framework_molecular_v2
    demo_molecular_vqe_py -.->|imports| ext_numpy
    demo_molecular_vqe_py -.->|imports| ext_os
    ext_traceback["traceback"]
    class ext_traceback ext;
    demo_molecular_vqe_py -.->|imports| ext_traceback
    entangled_hydrogen_py -.->|imports| ext___future__
    entangled_hydrogen_py -.->|imports| ext_logging
    entangled_hydrogen_py -.->|imports| ext_math
    entangled_hydrogen_py -.->|imports| ext_os
    entangled_hydrogen_py -.->|imports| ext_sys
    entangled_hydrogen_py -.->|imports| ext_warnings
    entangled_hydrogen_py -.->|imports| ext_abc
    entangled_hydrogen_py -.->|imports| ext_dataclasses
    entangled_hydrogen_py -.->|imports| ext_typing
    entangled_hydrogen_py -.->|imports| ext_numpy
    entangled_hydrogen_py -.->|imports| ext_torch
    entangled_hydrogen_py -.->|imports| ext_torch_nn
    entangled_hydrogen_py -.->|imports| ext_torch_nn_functional
    ext_scipy_special["scipy.special"]
    class ext_scipy_special ext;
    entangled_hydrogen_py -.->|imports| ext_scipy_special
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    entangled_hydrogen_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib["matplotlib"]
    class ext_matplotlib ext;
    entangled_hydrogen_py -.->|imports| ext_matplotlib
    ext_matplotlib_colors["matplotlib.colors"]
    class ext_matplotlib_colors ext;
    entangled_hydrogen_py -.->|imports| ext_matplotlib_colors
    entangled_hydrogen_py -.->|imports| ext_argparse
    entangled_hydrogen_py -.->|imports| ext_quantum_computer
    entangled_hydrogen_py -.->|imports| ext_quantum_computer
    entangled_hydrogen_py -.->|imports| ext_molecular_sim
    entangled_hydrogen_py -.->|imports| ext_relativistic_hydrogen
    higgs_four_lepton_analysis_py -.->|imports| ext___future__
    ext_csv["csv"]
    class ext_csv ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_csv
    higgs_four_lepton_analysis_py -.->|imports| ext_logging
    higgs_four_lepton_analysis_py -.->|imports| ext_math
    higgs_four_lepton_analysis_py -.->|imports| ext_os
    higgs_four_lepton_analysis_py -.->|imports| ext_sys
    ext_urllib_request["urllib.request"]
    class ext_urllib_request ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_urllib_request
    ext_urllib_error["urllib.error"]
    class ext_urllib_error ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_urllib_error
    higgs_four_lepton_analysis_py -.->|imports| ext_dataclasses
    ext_enum["enum"]
    class ext_enum ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_enum
    ext_pathlib["pathlib"]
    class ext_pathlib ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_pathlib
    higgs_four_lepton_analysis_py -.->|imports| ext_typing
    higgs_four_lepton_analysis_py -.->|imports| ext_numpy
    higgs_four_lepton_analysis_py -.->|imports| ext_quantum_computer
    higgs_four_lepton_analysis_py -.->|imports| ext_torch
    higgs_four_lepton_analysis_py -.->|imports| ext_torch_nn_functional
    ext_plotly_graph_objects["plotly.graph_objects"]
    class ext_plotly_graph_objects ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_plotly_graph_objects
    ext_plotly_subplots["plotly.subplots"]
    class ext_plotly_subplots ext;
    higgs_four_lepton_analysis_py -.->|imports| ext_plotly_subplots
    higgs_quantum_analysis_py -.->|imports| ext___future__
    higgs_quantum_analysis_py -.->|imports| ext_csv
    higgs_quantum_analysis_py -.->|imports| ext_logging
    higgs_quantum_analysis_py -.->|imports| ext_math
    higgs_quantum_analysis_py -.->|imports| ext_os
    higgs_quantum_analysis_py -.->|imports| ext_sys
    higgs_quantum_analysis_py -.->|imports| ext_urllib_request
    higgs_quantum_analysis_py -.->|imports| ext_urllib_error
    higgs_quantum_analysis_py -.->|imports| ext_dataclasses
    higgs_quantum_analysis_py -.->|imports| ext_enum
    higgs_quantum_analysis_py -.->|imports| ext_pathlib
    higgs_quantum_analysis_py -.->|imports| ext_typing
    higgs_quantum_analysis_py -.->|imports| ext_numpy
    higgs_quantum_analysis_py -.->|imports| ext_quantum_computer
    higgs_quantum_analysis_py -.->|imports| ext_torch
    higgs_quantum_analysis_py -.->|imports| ext_torch_nn_functional
    higgs_quantum_analysis_py -.->|imports| ext_plotly_graph_objects
    higgs_quantum_analysis_py -.->|imports| ext_plotly_subplots
    molecular_sim_py -.->|imports| ext___future__
    molecular_sim_py -.->|imports| ext_logging
    molecular_sim_py -.->|imports| ext_math
    molecular_sim_py -.->|imports| ext_os
    molecular_sim_py -.->|imports| ext_sys
    molecular_sim_py -.->|imports| ext_dataclasses
    molecular_sim_py -.->|imports| ext_typing
    molecular_sim_py -.->|imports| ext_numpy
    molecular_sim_py -.->|imports| ext_torch
    molecular_sim_py -.->|imports| ext_quantum_computer
    molecular_sim_py -.->|imports| ext_argparse
    molecular_sim_py -.->|imports| ext_quantum_computer
    molecular_sim_py -.->|imports| ext_pyscf
    ext_openfermion["openfermion"]
    class ext_openfermion ext;
    molecular_sim_py -.->|imports| ext_openfermion
    molecular_sim_py -.->|imports| ext_openfermion_transforms
    ext_openfermion_linalg["openfermion.linalg"]
    class ext_openfermion_linalg ext;
    molecular_sim_py -.->|imports| ext_openfermion_linalg
    ext_openfermionpyscf["openfermionpyscf"]
    class ext_openfermionpyscf ext;
    molecular_sim_py -.->|imports| ext_openfermionpyscf
    molecular_sim_py -.->|imports| ext_openfermion
    molecular_sim_py -.->|imports| ext_quantum_computer
    molecular_sim_py -.->|imports| ext_scipy_optimize
    orbital_visualizer2_py -.->|imports| ext_numpy
    orbital_visualizer2_py -.->|imports| ext_scipy_special
    orbital_visualizer2_py -.->|imports| ext_matplotlib_pyplot
    orbital_visualizer2_py -.->|imports| ext_matplotlib
    orbital_visualizer2_py -.->|imports| ext_torch
    orbital_visualizer2_py -.->|imports| ext_os
    orbital_visualizer2_py -.->|imports| ext_sys
    orbital_visualizer2_py -.->|imports| ext_warnings
    orbital_visualizer2_py -.->|imports| ext_typing
    ext_schrodinger_crystal_fixed2["schrodinger_crystal_fixed2"]
    class ext_schrodinger_crystal_fixed2 ext;
    orbital_visualizer2_py -.->|imports| ext_schrodinger_crystal_fixed2
    orbital_visualizer2_py -.->|imports| ext_plotly_graph_objects
    orbital_visualizer2_py -.->|imports| ext_traceback
    polarizability_v3_py -.->|imports| ext___future__
    polarizability_v3_py -.->|imports| ext_warnings
    polarizability_v3_py -.->|imports| ext_math
    polarizability_v3_py -.->|imports| ext_numpy
    polarizability_v3_py -.->|imports| ext_sys
    polarizability_v3_py -.->|imports| ext_dataclasses
    polarizability_v3_py -.->|imports| ext_typing
    polarizability_v3_py -.->|imports| ext_torch
    polarizability_v3_py -.->|imports| ext_quantum_computer
    polarizability_v3_py -.->|imports| ext_molecular_sim
    polarizability_v3_py -.->|imports| ext_pyscf
    polarizability_v3_py -.->|imports| ext_openfermion_transforms
    polarizability_v3_py -.->|imports| ext_openfermion_ops
    polarizability_v3_py -.->|imports| ext_scipy_optimize
    qc_dashboard_py -.->|imports| ext___future__
    ext_io["io"]
    class ext_io ext;
    qc_dashboard_py -.->|imports| ext_io
    qc_dashboard_py -.->|imports| ext_logging
    qc_dashboard_py -.->|imports| ext_math
    qc_dashboard_py -.->|imports| ext_os
    qc_dashboard_py -.->|imports| ext_sys
    ext_tempfile["tempfile"]
    class ext_tempfile ext;
    qc_dashboard_py -.->|imports| ext_tempfile
    qc_dashboard_py -.->|imports| ext_dataclasses
    qc_dashboard_py -.->|imports| ext_pathlib
    qc_dashboard_py -.->|imports| ext_typing
    qc_dashboard_py -.->|imports| ext_numpy
    qc_dashboard_py -.->|imports| ext_matplotlib
    qc_dashboard_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib_patches["matplotlib.patches"]
    class ext_matplotlib_patches ext;
    qc_dashboard_py -.->|imports| ext_matplotlib_patches
    ext_quantum_framework_core["quantum_framework_core"]
    class ext_quantum_framework_core ext;
    qc_dashboard_py -.->|imports| ext_quantum_framework_core
    qc_dashboard_py -.->|imports| ext_quantum_computer
    qc_dashboard_py -.->|imports| ext_tempfile
    qc_dashboard_py -.->|imports| ext_tempfile
    qc_dashboard_py -.->|imports| ext_tempfile
    ext_streamlit["streamlit"]
    class ext_streamlit ext;
    qc_dashboard_py -.->|imports| ext_streamlit
    qc_dashboard_py -.->|imports| ext_matplotlib
    qc_dashboard_py -.->|imports| ext_matplotlib_pyplot
    ext_mpl_toolkits_mplot3d["mpl_toolkits.mplot3d"]
    class ext_mpl_toolkits_mplot3d ext;
    qc_dashboard_py -.->|imports| ext_mpl_toolkits_mplot3d
    qc_dashboard_py -.->|imports| ext_quantum_framework_core
    qc_dashboard_py -.->|imports| ext_quantum_framework_core
    qc_dashboard_py -.->|imports| ext_quantum_computer
    qc_dashboard_py -.->|imports| ext_plotly_graph_objects
    qc_dashboard_py -.->|imports| ext_plotly_subplots
    ext_orbital_visualizer2["orbital_visualizer2"]
    class ext_orbital_visualizer2 ext;
    qc_dashboard_py -.->|imports| ext_orbital_visualizer2
    qc_dashboard_py -.->|imports| ext_numpy
    ext_quantum_framework_visualization["quantum_framework_visualization"]
    class ext_quantum_framework_visualization ext;
    qc_dashboard_py -.->|imports| ext_quantum_framework_visualization
    ext_quantum_dash["quantum_dash"]
    class ext_quantum_dash ext;
    qc_dashboard_py -.->|imports| ext_quantum_dash
    ext_quantum_3dview["quantum_3dview"]
    class ext_quantum_3dview ext;
    qc_dashboard_py -.->|imports| ext_quantum_3dview
    qc_dashboard_py -.->|imports| ext_streamlit
    ext_qc_integration["qc_integration"]
    class ext_qc_integration ext;
    qc_dashboard_py -.->|imports| ext_qc_integration
    qc_dashboard_py -.->|imports| ext_quantum_computer
    qc_dashboard_py -.->|imports| ext_qc_integration
    qc_dashboard_py -.->|imports| ext_qc_integration
    qc_integration_py -.->|imports| ext___future__
    qc_integration_py -.->|imports| ext_logging
    qc_integration_py -.->|imports| ext_math
    ext_re["re"]
    class ext_re ext;
    qc_integration_py -.->|imports| ext_re
    qc_integration_py -.->|imports| ext_abc
    qc_integration_py -.->|imports| ext_dataclasses
    qc_integration_py -.->|imports| ext_typing
    qc_integration_py -.->|imports| ext_argparse
    qc_integration_py -.->|imports| ext_quantum_framework_core
    qc_integration_py -.->|imports| ext_quantum_framework_core
    qc_integration_py -.->|imports| ext_quantum_computer
    qc_integration_py -.->|imports| ext_quantum_computer
    ext_qiskit["qiskit"]
    class ext_qiskit ext;
    qc_integration_py -.->|imports| ext_qiskit
    ext_pennylane["pennylane"]
    class ext_pennylane ext;
    qc_integration_py -.->|imports| ext_pennylane
    quantum_3dview_py -.->|imports| ext_numpy
    quantum_3dview_py -.->|imports| ext_torch
    quantum_3dview_py -.->|imports| ext_plotly_graph_objects
    quantum_3dview_py -.->|imports| ext_plotly_subplots
    ext_plotly_express["plotly.express"]
    class ext_plotly_express ext;
    quantum_3dview_py -.->|imports| ext_plotly_express
    quantum_3dview_py -.->|imports| ext_dataclasses
    quantum_3dview_py -.->|imports| ext_typing
    quantum_3dview_py -.->|imports| ext_enum
    ext_colorsys["colorsys"]
    class ext_colorsys ext;
    quantum_3dview_py -.->|imports| ext_colorsys
    ext_json["json"]
    class ext_json ext;
    quantum_3dview_py -.->|imports| ext_json
    quantum_3dview_py -.->|imports| ext_pathlib
    quantum_3dview_py -.->|imports| ext_warnings
    quantum_3dview_py -.->|imports| ext_quantum_computer
    ext_quantum_visualizer["quantum_visualizer"]
    class ext_quantum_visualizer ext;
    quantum_3dview_py -.->|imports| ext_quantum_visualizer
    ext_IPython_display["IPython.display"]
    class ext_IPython_display ext;
    quantum_3dview_py -.->|imports| ext_IPython_display
    ext_scipy_io["scipy.io"]
    class ext_scipy_io ext;
    quantum_3dview_py -.->|imports| ext_scipy_io
    quantum_computer_py -.->|imports| ext___future__
    quantum_computer_py -.->|imports| ext_logging
    quantum_computer_py -.->|imports| ext_math
    quantum_computer_py -.->|imports| ext_os
    quantum_computer_py -.->|imports| ext_warnings
    quantum_computer_py -.->|imports| ext_abc
    quantum_computer_py -.->|imports| ext_dataclasses
    quantum_computer_py -.->|imports| ext_typing
    quantum_computer_py -.->|imports| ext_numpy
    quantum_computer_py -.->|imports| ext_torch
    quantum_computer_py -.->|imports| ext_torch_nn
    quantum_computer_py -.->|imports| ext_torch_nn_functional
    quantum_computer_py -.->|imports| ext_argparse
    quantum_dash_py -.->|imports| ext___future__
    quantum_dash_py -.->|imports| ext_logging
    quantum_dash_py -.->|imports| ext_math
    quantum_dash_py -.->|imports| ext_os
    quantum_dash_py -.->|imports| ext_sys
    quantum_dash_py -.->|imports| ext_warnings
    quantum_dash_py -.->|imports| ext_abc
    quantum_dash_py -.->|imports| ext_dataclasses
    quantum_dash_py -.->|imports| ext_enum
    quantum_dash_py -.->|imports| ext_pathlib
    quantum_dash_py -.->|imports| ext_typing
    quantum_dash_py -.->|imports| ext_numpy
    quantum_dash_py -.->|imports| ext_torch
    quantum_dash_py -.->|imports| ext_plotly_graph_objects
    quantum_dash_py -.->|imports| ext_plotly_subplots
    quantum_dash_py -.->|imports| ext_plotly_express
    quantum_dash_py -.->|imports| ext_matplotlib
    quantum_dash_py -.->|imports| ext_matplotlib_pyplot
    quantum_dash_py -.->|imports| ext_matplotlib_patches
    quantum_dash_py -.->|imports| ext_mpl_toolkits_mplot3d
    quantum_dash_py -.->|imports| ext_mpl_toolkits_mplot3d
    quantum_dash_py -.->|imports| ext_argparse
    quantum_dash_py -.->|imports| ext_quantum_computer
    quantum_dash_py -.->|imports| ext_molecular_sim
    quantum_framework_core_py -.->|imports| ext___future__
    quantum_framework_core_py -.->|imports| ext_logging
    quantum_framework_core_py -.->|imports| ext_math
    quantum_framework_core_py -.->|imports| ext_os
    quantum_framework_core_py -.->|imports| ext_warnings
    quantum_framework_core_py -.->|imports| ext_abc
    quantum_framework_core_py -.->|imports| ext_dataclasses
    quantum_framework_core_py -.->|imports| ext_enum
    quantum_framework_core_py -.->|imports| ext_typing
    quantum_framework_core_py -.->|imports| ext_numpy
    quantum_framework_core_py -.->|imports| ext_torch
    quantum_framework_core_py -.->|imports| ext_torch_nn
    quantum_framework_core_py -.->|imports| ext_torch_nn_functional
    ext_tomllib["tomllib"]
    class ext_tomllib ext;
    quantum_framework_core_py -.->|imports| ext_tomllib
    quantum_framework_core_py -.->|imports| ext_time
    quantum_framework_core_py -.->|imports| ext_argparse
    ext_tomli["tomli"]
    class ext_tomli ext;
    quantum_framework_core_py -.->|imports| ext_tomli
    quantum_framework_main_py -.->|imports| ext___future__
    quantum_framework_main_py -.->|imports| ext_argparse
    quantum_framework_main_py -.->|imports| ext_logging
    quantum_framework_main_py -.->|imports| ext_os
    quantum_framework_main_py -.->|imports| ext_sys
    quantum_framework_main_py -.->|imports| ext_typing
    quantum_framework_main_py -.->|imports| ext_torch
    quantum_framework_main_py -.->|imports| ext_quantum_framework_core
    ext_quantum_framework_menu["quantum_framework_menu"]
    class ext_quantum_framework_menu ext;
    quantum_framework_main_py -.->|imports| ext_quantum_framework_menu
    ext_quantum_lab["quantum_lab"]
    class ext_quantum_lab ext;
    quantum_framework_main_py -.->|imports| ext_quantum_lab
    quantum_framework_menu_py -.->|imports| ext___future__
    quantum_framework_menu_py -.->|imports| ext_logging
    quantum_framework_menu_py -.->|imports| ext_math
    quantum_framework_menu_py -.->|imports| ext_os
    quantum_framework_menu_py -.->|imports| ext_sys
    quantum_framework_menu_py -.->|imports| ext_time
    quantum_framework_menu_py -.->|imports| ext_dataclasses
    quantum_framework_menu_py -.->|imports| ext_typing
    quantum_framework_menu_py -.->|imports| ext_numpy
    quantum_framework_menu_py -.->|imports| ext_torch
    quantum_framework_menu_py -.->|imports| ext_quantum_framework_core
    ext_quantum_framework_molecular["quantum_framework_molecular"]
    class ext_quantum_framework_molecular ext;
    quantum_framework_menu_py -.->|imports| ext_quantum_framework_molecular
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    quantum_framework_menu_py -.->|imports| ext_matplotlib_pyplot
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    quantum_framework_menu_py -.->|imports| ext_scipy_special
    ext_higgs_four_lepton_analysis["higgs_four_lepton_analysis"]
    class ext_higgs_four_lepton_analysis ext;
    quantum_framework_menu_py -.->|imports| ext_higgs_four_lepton_analysis
    quantum_framework_menu_py -.->|imports| ext_quantum_dash
    quantum_framework_menu_py -.->|imports| ext_quantum_3dview
    quantum_framework_menu_py -.->|imports| ext_json
    quantum_framework_menu_py -.->|imports| ext_quantum_visualizer
    ext_app["app"]
    class ext_app ext;
    quantum_framework_menu_py -.->|imports| ext_app
    quantum_framework_menu_py -.->|imports| ext_traceback
    quantum_framework_molecular_py -.->|imports| ext___future__
    quantum_framework_molecular_py -.->|imports| ext_logging
    quantum_framework_molecular_py -.->|imports| ext_math
    quantum_framework_molecular_py -.->|imports| ext_os
    quantum_framework_molecular_py -.->|imports| ext_warnings
    quantum_framework_molecular_py -.->|imports| ext_dataclasses
    quantum_framework_molecular_py -.->|imports| ext_typing
    quantum_framework_molecular_py -.->|imports| ext_numpy
    quantum_framework_molecular_py -.->|imports| ext_torch
    quantum_framework_molecular_py -.->|imports| ext_pyscf
    quantum_framework_molecular_py -.->|imports| ext_openfermion
    quantum_framework_molecular_py -.->|imports| ext_openfermion_transforms
    quantum_framework_molecular_py -.->|imports| ext_openfermion_ops
    quantum_framework_molecular_py -.->|imports| ext_openfermionpyscf
    quantum_framework_molecular_py -.->|imports| ext_scipy_optimize
    quantum_framework_molecular_fixed_py -.->|imports| ext___future__
    quantum_framework_molecular_fixed_py -.->|imports| ext_logging
    quantum_framework_molecular_fixed_py -.->|imports| ext_math
    quantum_framework_molecular_fixed_py -.->|imports| ext_os
    quantum_framework_molecular_fixed_py -.->|imports| ext_warnings
    quantum_framework_molecular_fixed_py -.->|imports| ext_dataclasses
    quantum_framework_molecular_fixed_py -.->|imports| ext_typing
    quantum_framework_molecular_fixed_py -.->|imports| ext_numpy
    quantum_framework_molecular_fixed_py -.->|imports| ext_torch
    quantum_framework_molecular_fixed_py -.->|imports| ext_pyscf
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermion
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermion_transforms
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermion_linalg
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermion_ops
    quantum_framework_molecular_fixed_py -.->|imports| ext_openfermionpyscf
    quantum_framework_molecular_fixed_py -.->|imports| ext_scipy_optimize
    quantum_framework_molecular_v2_py -.->|imports| ext___future__
    quantum_framework_molecular_v2_py -.->|imports| ext_logging
    quantum_framework_molecular_v2_py -.->|imports| ext_math
    quantum_framework_molecular_v2_py -.->|imports| ext_os
    quantum_framework_molecular_v2_py -.->|imports| ext_warnings
    quantum_framework_molecular_v2_py -.->|imports| ext_dataclasses
    quantum_framework_molecular_v2_py -.->|imports| ext_typing
    quantum_framework_molecular_v2_py -.->|imports| ext_numpy
    quantum_framework_molecular_v2_py -.->|imports| ext_torch
    quantum_framework_molecular_v2_py -.->|imports| ext_pyscf
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermion
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermion_transforms
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermion_ops
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermion_linalg
    quantum_framework_molecular_v2_py -.->|imports| ext_openfermionpyscf
    quantum_framework_molecular_v2_py -.->|imports| ext_scipy_optimize
    quantum_framework_molecular_v2_py -.->|imports| ext_tomllib
    quantum_framework_molecular_v2_py -.->|imports| ext_argparse
    quantum_framework_molecular_v2_py -.->|imports| ext_tomli
    quantum_framework_molecular_v2_py -.->|imports| ext_math
    ext_itertools["itertools"]
    class ext_itertools ext;
    quantum_framework_molecular_v2_py -.->|imports| ext_itertools
    quantum_framework_molecular_v2_py -.->|imports| ext_time
    quantum_framework_physics_py -.->|imports| ext___future__
    quantum_framework_physics_py -.->|imports| ext_math
    quantum_framework_physics_py -.->|imports| ext_warnings
    quantum_framework_physics_py -.->|imports| ext_typing
    quantum_framework_physics_py -.->|imports| ext_numpy
    quantum_framework_physics_py -.->|imports| ext_torch
    quantum_framework_physics_py -.->|imports| ext_torch_nn
    quantum_framework_physics_py -.->|imports| ext_torch_nn_functional
    quantum_framework_visualization_py -.->|imports| ext___future__
    quantum_framework_visualization_py -.->|imports| ext_logging
    quantum_framework_visualization_py -.->|imports| ext_math
    quantum_framework_visualization_py -.->|imports| ext_os
    quantum_framework_visualization_py -.->|imports| ext_warnings
    quantum_framework_visualization_py -.->|imports| ext_dataclasses
    quantum_framework_visualization_py -.->|imports| ext_typing
    quantum_framework_visualization_py -.->|imports| ext_numpy
    quantum_framework_visualization_py -.->|imports| ext_torch
    quantum_framework_visualization_py -.->|imports| ext_scipy_special
    quantum_framework_visualization_py -.->|imports| ext_matplotlib_pyplot
    quantum_framework_visualization_py -.->|imports| ext_matplotlib
    quantum_framework_visualization_py -.->|imports| ext_matplotlib_colors
    quantum_framework_visualization_py -.->|imports| ext_plotly_graph_objects
    quantum_lab_py -.->|imports| ext___future__
    quantum_lab_py -.->|imports| ext_argparse
    quantum_lab_py -.->|imports| ext_math
    quantum_lab_py -.->|imports| ext_os
    quantum_lab_py -.->|imports| ext_sys
    quantum_lab_py -.->|imports| ext_time
    quantum_lab_py -.->|imports| ext_dataclasses
    quantum_lab_py -.->|imports| ext_typing
    quantum_lab_py -.->|imports| ext_numpy
    quantum_lab_py -.->|imports| ext_quantum_framework_core
    quantum_lab_py -.->|imports| ext_torch
    ext_rich_console["rich.console"]
    class ext_rich_console ext;
    quantum_lab_py -.->|imports| ext_rich_console
    ext_rich_panel["rich.panel"]
    class ext_rich_panel ext;
    quantum_lab_py -.->|imports| ext_rich_panel
    ext_rich_table["rich.table"]
    class ext_rich_table ext;
    quantum_lab_py -.->|imports| ext_rich_table
    ext_rich_text["rich.text"]
    class ext_rich_text ext;
    quantum_lab_py -.->|imports| ext_rich_text
    ext_rich_rule["rich.rule"]
    class ext_rich_rule ext;
    quantum_lab_py -.->|imports| ext_rich_rule
    ext_rich_prompt["rich.prompt"]
    class ext_rich_prompt ext;
    quantum_lab_py -.->|imports| ext_rich_prompt
    ext_rich_align["rich.align"]
    class ext_rich_align ext;
    quantum_lab_py -.->|imports| ext_rich_align
    ext_rich_columns["rich.columns"]
    class ext_rich_columns ext;
    quantum_lab_py -.->|imports| ext_rich_columns
    ext_rich_live["rich.live"]
    class ext_rich_live ext;
    quantum_lab_py -.->|imports| ext_rich_live
    quantum_lab_py -.->|imports| ext_scipy_special
    quantum_lab_py -.->|imports| ext_math
    quantum_lab_py -.->|imports| ext_quantum_framework_molecular
    quantum_simulator_py -.->|imports| ext___future__
    quantum_simulator_py -.->|imports| ext_logging
    quantum_simulator_py -.->|imports| ext_math
    quantum_simulator_py -.->|imports| ext_os
    quantum_simulator_py -.->|imports| ext_warnings
    quantum_simulator_py -.->|imports| ext_abc
    quantum_simulator_py -.->|imports| ext_dataclasses
    quantum_simulator_py -.->|imports| ext_enum
    quantum_simulator_py -.->|imports| ext_typing
    quantum_simulator_py -.->|imports| ext_numpy
    quantum_simulator_py -.->|imports| ext_torch
    quantum_simulator_py -.->|imports| ext_torch_nn
    quantum_simulator_py -.->|imports| ext_torch_nn_functional
    quantum_simulator_py -.->|imports| ext_tomllib
    quantum_simulator_py -.->|imports| ext_scipy_special
    quantum_simulator_py -.->|imports| ext_matplotlib_pyplot
    quantum_simulator_py -.->|imports| ext_tomli
    quantum_visualizer_py -.->|imports| ext___future__
    quantum_visualizer_py -.->|imports| ext_logging
    quantum_visualizer_py -.->|imports| ext_math
    quantum_visualizer_py -.->|imports| ext_os
    quantum_visualizer_py -.->|imports| ext_sys
    quantum_visualizer_py -.->|imports| ext_warnings
    quantum_visualizer_py -.->|imports| ext_abc
    quantum_visualizer_py -.->|imports| ext_dataclasses
    quantum_visualizer_py -.->|imports| ext_typing
    quantum_visualizer_py -.->|imports| ext_numpy
    quantum_visualizer_py -.->|imports| ext_torch
    quantum_visualizer_py -.->|imports| ext_quantum_computer
    quantum_visualizer_py -.->|imports| ext_molecular_sim
    ext_advanced_experiments["advanced_experiments"]
    class ext_advanced_experiments ext;
    quantum_visualizer_py -.->|imports| ext_advanced_experiments
    quantum_visualizer_py -.->|imports| ext_argparse
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    quantum_visualizer_py -.->|imports| ext_matplotlib_patches
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    quantum_visualizer_py -.->|imports| ext_mpl_toolkits_mplot3d
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    quantum_visualizer_py -.->|imports| ext_mpl_toolkits_mplot3d
    quantum_visualizer_py -.->|imports| ext_matplotlib_pyplot
    relativistic_hydrogen_py -.->|imports| ext_numpy
    relativistic_hydrogen_py -.->|imports| ext_scipy_special
    ext_scipy["scipy"]
    class ext_scipy ext;
    relativistic_hydrogen_py -.->|imports| ext_scipy
    relativistic_hydrogen_py -.->|imports| ext_matplotlib_pyplot
    relativistic_hydrogen_py -.->|imports| ext_matplotlib
    relativistic_hydrogen_py -.->|imports| ext_matplotlib_colors
    relativistic_hydrogen_py -.->|imports| ext_torch
    relativistic_hydrogen_py -.->|imports| ext_torch_nn
    relativistic_hydrogen_py -.->|imports| ext_torch_nn_functional
    relativistic_hydrogen_py -.->|imports| ext_os
    relativistic_hydrogen_py -.->|imports| ext_sys
    relativistic_hydrogen_py -.->|imports| ext_warnings
    relativistic_hydrogen_py -.->|imports| ext_json
    relativistic_hydrogen_py -.->|imports| ext_typing
    relativistic_hydrogen_py -.->|imports| ext_dataclasses
    relativistic_hydrogen_py -.->|imports| ext_abc
    relativistic_hydrogen_py -.->|imports| ext_logging
    relativistic_hydrogen_py -.->|imports| ext_math
    ext_glob["glob"]
    class ext_glob ext;
    relativistic_hydrogen_py -.->|imports| ext_glob
    relativistic_hydrogen_py -.->|imports| ext_traceback
    test_qc_integration_py -.->|imports| ext___future__
    test_qc_integration_py -.->|imports| ext_math
    test_qc_integration_py -.->|imports| ext_typing
    ext_pytest["pytest"]
    class ext_pytest ext;
    test_qc_integration_py -.->|imports| ext_pytest
    test_qc_integration_py -.->|imports| ext_numpy
    test_qc_integration_py -.->|imports| ext_qc_integration
    ext_qc_dashboard["qc_dashboard"]
    class ext_qc_dashboard ext;
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_qc_dashboard
    test_qc_integration_py -.->|imports| ext_quantum_framework_core
    test_qc_integration_py -.->|imports| ext_quantum_framework_core
    test_quantum_framework_py -.->|imports| ext_math
    test_quantum_framework_py -.->|imports| ext_sys
    test_quantum_framework_py -.->|imports| ext_os
    test_quantum_framework_py -.->|imports| ext_typing
    test_quantum_framework_py -.->|imports| ext_pytest
    test_quantum_framework_py -.->|imports| ext_numpy
    test_quantum_framework_py -.->|imports| ext_torch
    test_quantum_framework_py -.->|imports| ext_quantum_framework_core
    test_quantum_framework_py -.->|imports| ext_quantum_framework_core
    test_quantum_framework_py -.->|imports| ext_quantum_framework_core
    topological_hilbert_compression2_py -.->|imports| ext___future__
    topological_hilbert_compression2_py -.->|imports| ext_logging
    topological_hilbert_compression2_py -.->|imports| ext_math
    topological_hilbert_compression2_py -.->|imports| ext_os
    topological_hilbert_compression2_py -.->|imports| ext_sys
    topological_hilbert_compression2_py -.->|imports| ext_warnings
    topological_hilbert_compression2_py -.->|imports| ext_abc
    topological_hilbert_compression2_py -.->|imports| ext_dataclasses
    topological_hilbert_compression2_py -.->|imports| ext_enum
    topological_hilbert_compression2_py -.->|imports| ext_typing
    topological_hilbert_compression2_py -.->|imports| ext_numpy
    topological_hilbert_compression2_py -.->|imports| ext_torch
    topological_hilbert_compression2_py -.->|imports| ext_torch_nn
    topological_hilbert_compression2_py -.->|imports| ext_torch_nn_functional
    topological_hilbert_compression2_py -.->|imports| ext_argparse
    topological_hilbert_compression2_py -.->|imports| ext_quantum_computer
    topological_hilbert_compression2_py -.->|imports| ext_time
    topological_hilbert_compression2_py -.->|imports| ext_quantum_computer
```

---

## UML Class Diagram

Auto-generated Mermaid class diagram from parsed class-level symbols. Shows classes, structs, interfaces, traits, and their methods with inheritance and dependency relationships.

```mermaid
classDiagram
  class advanced_experiments_py_GroverConfig {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_GroverOracle {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_GroverDiffusionOperator {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_GroverSearch {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_QEDConfig {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_LambShiftCalculator {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_AnomalousMagneticMoment {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_QEDEffectsExperiment {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_PolyatomicMoleculeData {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_MoleculeBuilder {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_PolyatomicVQE {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_PolyatomicExperiment {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class advanced_experiments_py_AdvancedExperimentRunner {
    <<class>>
    +_make_logger(name, level)
    +main()
    +__init__(self, n_qubits, marked_state)
    +_validate(self)
    +apply(self, state, backend)
    +__init__(self, n_qubits)
    +apply(self, state, backend)
    +__init__(self, config)
    +_calculate_entropy(self, probs)
    +_init_quantum_computer(self)
  }
  class app_py_VQEResult {
    <<class>>
    +_sd_indices(n_e, n_q)
    +_run_circuit(circuit, backend, state)
    +givens_single_excitation(state, o, v, theta, n_qubits, backend)
    +particle_conserving_ansatz(state, thetas, singles, doubles, backend)
    +__init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)
    +_to_scalar(amps)
    +_apply_pauli(self, amps, pauli)
    +eval_dipole(self, amps_raw)
    +__call__(self, amps)
    +__init__(self, bond_length_angstrom)
  }
  class app_py_StarkEvaluator {
    <<class>>
    +_sd_indices(n_e, n_q)
    +_run_circuit(circuit, backend, state)
    +givens_single_excitation(state, o, v, theta, n_qubits, backend)
    +particle_conserving_ansatz(state, thetas, singles, doubles, backend)
    +__init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)
    +_to_scalar(amps)
    +_apply_pauli(self, amps, pauli)
    +eval_dipole(self, amps_raw)
    +__call__(self, amps)
    +__init__(self, bond_length_angstrom)
  }
  class app_py_DipoleOperatorBuilder {
    <<class>>
    +_sd_indices(n_e, n_q)
    +_run_circuit(circuit, backend, state)
    +givens_single_excitation(state, o, v, theta, n_qubits, backend)
    +particle_conserving_ansatz(state, thetas, singles, doubles, backend)
    +__init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)
    +_to_scalar(amps)
    +_apply_pauli(self, amps, pauli)
    +eval_dipole(self, amps_raw)
    +__call__(self, amps)
    +__init__(self, bond_length_angstrom)
  }
  class app_py_PolarizabilityCalculator {
    <<class>>
    +_sd_indices(n_e, n_q)
    +_run_circuit(circuit, backend, state)
    +givens_single_excitation(state, o, v, theta, n_qubits, backend)
    +particle_conserving_ansatz(state, thetas, singles, doubles, backend)
    +__init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)
    +_to_scalar(amps)
    +_apply_pauli(self, amps, pauli)
    +eval_dipole(self, amps_raw)
    +__call__(self, amps)
    +__init__(self, bond_length_angstrom)
  }
  class entangled_hydrogen_py_EntangledHydrogenConfig {
    <<class>>
    +_make_logger(name)
    +main()
    +name(self)
    +prepare(self, n_qubits)
    +get_theoretical_entropy(self)
    +name(self)
    +prepare(self, qc, backend)
    +get_theoretical_entropy(self)
    +__init__(self, n_qubits)
    +name(self)
  }
  class entangled_hydrogen_py_IEntangledState {
    <<class>>
    +_make_logger(name)
    +main()
    +name(self)
    +prepare(self, n_qubits)
    +get_theoretical_entropy(self)
    +name(self)
    +prepare(self, qc, backend)
    +get_theoretical_entropy(self)
    +__init__(self, n_qubits)
    +name(self)
  }
  class entangled_hydrogen_py_BellState {
    <<class>>
    +_make_logger(name)
    +main()
    +name(self)
    +prepare(self, n_qubits)
    +get_theoretical_entropy(self)
    +name(self)
    +prepare(self, qc, backend)
    +get_theoretical_entropy(self)
    +__init__(self, n_qubits)
    +name(self)
  }
  class entangled_hydrogen_py_GHZState {
    <<class>>
    +_make_logger(name)
    +main()
    +name(self)
    +prepare(self, n_qubits)
    +get_theoretical_entropy(self)
    +name(self)
    +prepare(self, qc, backend)
    +get_theoretical_entropy(self)
    +__init__(self, n_qubits)
    +name(self)
  }
  class entangled_hydrogen_py_WState {
    <<class>>
    +_make_logger(name)
    +main()
    +name(self)
    +prepare(self, n_qubits)
    +get_theoretical_entropy(self)
    +name(self)
    +prepare(self, qc, backend)
    +get_theoretical_entropy(self)
    +__init__(self, n_qubits)
    +name(self)
  }
  class entangled_hydrogen_py_WavefunctionCalculator {
    <<class>>
    +_make_logger(name)
    +main()
    +name(self)
    +prepare(self, n_qubits)
    +get_theoretical_entropy(self)
    +name(self)
    +prepare(self, qc, backend)
    +get_theoretical_entropy(self)
    +__init__(self, n_qubits)
    +name(self)
  }
  class entangled_hydrogen_py_EntangledHydrogenSampler {
    <<class>>
    +_make_logger(name)
    +main()
    +name(self)
    +prepare(self, n_qubits)
    +get_theoretical_entropy(self)
    +name(self)
    +prepare(self, qc, backend)
    +get_theoretical_entropy(self)
    +__init__(self, n_qubits)
    +name(self)
  }
  class entangled_hydrogen_py_EntangledHydrogenVisualizer {
    <<class>>
    +_make_logger(name)
    +main()
    +name(self)
    +prepare(self, n_qubits)
    +get_theoretical_entropy(self)
    +name(self)
    +prepare(self, qc, backend)
    +get_theoretical_entropy(self)
    +__init__(self, n_qubits)
    +name(self)
  }
  class entangled_hydrogen_py_EntangledHydrogenExperiment {
    <<class>>
    +_make_logger(name)
    +main()
    +name(self)
    +prepare(self, n_qubits)
    +get_theoretical_entropy(self)
    +name(self)
    +prepare(self, qc, backend)
    +get_theoretical_entropy(self)
    +__init__(self, n_qubits)
    +name(self)
  }
  class higgs_four_lepton_analysis_py_LeptonType {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_four_lepton_analysis_py_EventType {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_four_lepton_analysis_py_Config {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_four_lepton_analysis_py_FourMomentum {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_four_lepton_analysis_py_Lepton {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_four_lepton_analysis_py_Event {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_four_lepton_analysis_py_QuantumSpinorProcessor {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_four_lepton_analysis_py_EventParser {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_four_lepton_analysis_py_Visualizer {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_four_lepton_analysis_py_HiggsQuantumAnalysis {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_LeptonType {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_EventType {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_Config {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_FourMomentum {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_Lepton {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_Event {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_QuantumSpinorProcessor {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_EventParser {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_Visualizer {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class higgs_quantum_analysis_py_HiggsQuantumAnalysis {
    <<class>>
    +_make_logger(name)
    +main()
    +__post_init__(self)
    +from_energy_momentum(cls, E, px, py, pz)
    +__add__(self, other)
    +pt(self)
    +eta(self)
    +phi(self)
    +energy(self)
    +mass(self)
  }
  class molecular_sim_py_MoleculeData {
    <<class>>
    +_make_logger(name)
    +_h2_sto3g_pyscf()
    +_h2_sto3g_hardcoded()
    +build_jw_hamiltonian_of(mol)
    +prepare_hf(mol, factory, backend)
    +_sd_indices(n_e, n_q)
    +uccsd(state, thetas, singles, doubles, backend, runner)
    +__init__(self, mol, n_qubits)
    +_to_scalar(amps)
    +_verify_hf(self)
  }
  class molecular_sim_py_ExactJWEnergy {
    <<class>>
    +_make_logger(name)
    +_h2_sto3g_pyscf()
    +_h2_sto3g_hardcoded()
    +build_jw_hamiltonian_of(mol)
    +prepare_hf(mol, factory, backend)
    +_sd_indices(n_e, n_q)
    +uccsd(state, thetas, singles, doubles, backend, runner)
    +__init__(self, mol, n_qubits)
    +_to_scalar(amps)
    +_verify_hf(self)
  }
  class molecular_sim_py_SurrogateEnergy {
    <<class>>
    +_make_logger(name)
    +_h2_sto3g_pyscf()
    +_h2_sto3g_hardcoded()
    +build_jw_hamiltonian_of(mol)
    +prepare_hf(mol, factory, backend)
    +_sd_indices(n_e, n_q)
    +uccsd(state, thetas, singles, doubles, backend, runner)
    +__init__(self, mol, n_qubits)
    +_to_scalar(amps)
    +_verify_hf(self)
  }
  class molecular_sim_py_VQEResult {
    <<class>>
    +_make_logger(name)
    +_h2_sto3g_pyscf()
    +_h2_sto3g_hardcoded()
    +build_jw_hamiltonian_of(mol)
    +prepare_hf(mol, factory, backend)
    +_sd_indices(n_e, n_q)
    +uccsd(state, thetas, singles, doubles, backend, runner)
    +__init__(self, mol, n_qubits)
    +_to_scalar(amps)
    +_verify_hf(self)
  }
```

---

## Code Property Graph

Machine-readable Code Property Graph (CPG) in JSON-LD format. This block allows AI agents to parse the full structural graph without additional file reads. Compatible with GraphRAG pipelines.

```json
{"@context": "https://schema.org", "analysis": {"communities": [{"cohesion": 0.6, "id": 0, "label": "root: quantum_simulator", "size": 11}, {"cohesion": 0.526, "id": 1, "label": "root: quantum_framework_core", "size": 8}, {"cohesion": 0.389, "id": 2, "label": "root: quantum_framework_menu", "size": 6}, {"cohesion": 1.0, "id": 3, "label": "root: quantum_framework_molecular_v2", "size": 2}], "god_nodes": [{"node_id": "quantum_computer.py", "score": 42.3}, {"node_id": "quantum_framework_core.py", "score": 33.6}, {"node_id": "qc_dashboard.py", "score": 24.8}, {"node_id": "quantum_simulator.py", "score": 23.9}, {"node_id": "quantum_framework_menu.py", "score": 23.8}, {"node_id": "molecular_sim.py", "score": 16.6}, {"node_id": "advanced_experiments.py", "score": 15.6}, {"node_id": "quantum_visualizer.py", "score": 15.4}, {"node_id": "quantum_dash.py", "score": 15.1}, {"node_id": "test_qc_integration.py", "score": 14.7}], "surprising_connections": [{"hops": 5, "source": "quantum_lab.py", "target": "quantum_simulator.py"}, {"hops": 5, "source": "quantum_lab.py", "target": "relativistic_hydrogen.py"}, {"hops": 5, "source": "quantum_simulator.py", "target": "test_quantum_framework.py"}, {"hops": 5, "source": "relativistic_hydrogen.py", "target": "test_quantum_framework.py"}, {"hops": 4, "source": "advanced_experiments.py", "target": "quantum_lab.py"}]}, "edges": [{"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "quantum_simulator"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "relativistic_hydrogen"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "molecular_sim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "advanced_experiments.py", "target": "pyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "molecular_sim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "pyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "openfermion.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "openfermion.ops"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_molecular_vqe.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_molecular_vqe.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_molecular_vqe.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_molecular_vqe.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_molecular_vqe.py", "target": "quantum_framework_molecular_v2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_molecular_vqe.py", "target": "quantum_framework_molecular_v2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_molecular_vqe.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_molecular_vqe.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_molecular_vqe.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "matplotlib.colors"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "molecular_sim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "entangled_hydrogen.py", "target": "relativistic_hydrogen"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "csv"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "urllib.request"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "urllib.error"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "enum"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "plotly.graph_objects"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_four_lepton_analysis.py", "target": "plotly.subplots"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "csv"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "urllib.request"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "urllib.error"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "enum"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "plotly.graph_objects"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "higgs_quantum_analysis.py", "target": "plotly.subplots"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "pyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "openfermion"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "openfermion.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "openfermion.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "openfermionpyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "openfermion"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "molecular_sim.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "schrodinger_crystal_fixed2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "plotly.graph_objects"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "orbital_visualizer2.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "molecular_sim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "pyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "openfermion.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "openfermion.ops"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polarizability_v3.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "io"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "tempfile"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "tempfile"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "tempfile"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "tempfile"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "streamlit"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "mpl_toolkits.mplot3d"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "plotly.graph_objects"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "plotly.subplots"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "orbital_visualizer2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "quantum_framework_visualization"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "quantum_dash"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "quantum_3dview"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "streamlit"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "qc_integration"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "qc_integration"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_dashboard.py", "target": "qc_integration"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "re"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "qiskit"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "qc_integration.py", "target": "pennylane"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "plotly.graph_objects"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "plotly.subplots"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "plotly.express"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "enum"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "colorsys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "quantum_visualizer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "IPython.display"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_3dview.py", "target": "scipy.io"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_computer.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "enum"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "plotly.graph_objects"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "plotly.subplots"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "plotly.express"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "mpl_toolkits.mplot3d"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "mpl_toolkits.mplot3d"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_dash.py", "target": "molecular_sim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "enum"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "tomllib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_core.py", "target": "tomli"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "quantum_framework_menu"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_main.py", "target": "quantum_lab"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "quantum_framework_molecular"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "higgs_four_lepton_analysis"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "quantum_dash"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "quantum_3dview"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "quantum_visualizer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "app"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_menu.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "pyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "openfermion"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "openfermion.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "openfermion.ops"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "openfermionpyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "pyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "openfermion"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "openfermion.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "openfermion.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "openfermion.ops"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "openfermionpyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_fixed.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "pyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "openfermion"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "openfermion.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "openfermion.ops"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "openfermion.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "openfermionpyscf"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "tomllib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "tomli"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "itertools"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_molecular_v2.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_physics.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_physics.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_physics.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_physics.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_physics.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_physics.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_physics.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_physics.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "matplotlib.colors"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_framework_visualization.py", "target": "plotly.graph_objects"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "rich.console"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "rich.panel"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "rich.table"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "rich.text"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "rich.rule"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "rich.prompt"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "rich.align"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "rich.columns"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "rich.live"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_lab.py", "target": "quantum_framework_molecular"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "enum"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "tomllib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_simulator.py", "target": "tomli"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "molecular_sim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "advanced_experiments"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "mpl_toolkits.mplot3d"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "mpl_toolkits.mplot3d"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quantum_visualizer.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "scipy.special"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "matplotlib.colors"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "relativistic_hydrogen.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "pytest"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_integration"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "qc_dashboard"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_qc_integration.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "pytest"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_quantum_framework.py", "target": "quantum_framework_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "enum"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "topological_hilbert_compression2.py", "target": "quantum_computer"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "advanced_experiments.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "advanced_experiments.py", "target": "quantum_simulator.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "advanced_experiments.py", "target": "relativistic_hydrogen.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "advanced_experiments.py", "target": "molecular_sim.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "app.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "app.py", "target": "molecular_sim.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "demo_molecular_vqe.py", "target": "quantum_framework_molecular_v2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "demo_molecular_vqe.py", "target": "quantum_framework_molecular_v2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "entangled_hydrogen.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "entangled_hydrogen.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "entangled_hydrogen.py", "target": "molecular_sim.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "entangled_hydrogen.py", "target": "relativistic_hydrogen.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "higgs_four_lepton_analysis.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "higgs_quantum_analysis.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "molecular_sim.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "molecular_sim.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "molecular_sim.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "polarizability_v3.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "polarizability_v3.py", "target": "molecular_sim.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "orbital_visualizer2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "quantum_framework_visualization.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "quantum_dash.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "quantum_3dview.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "qc_integration.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "qc_integration.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_dashboard.py", "target": "qc_integration.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_integration.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_integration.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_integration.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "qc_integration.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_3dview.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_3dview.py", "target": "quantum_visualizer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_dash.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_dash.py", "target": "molecular_sim.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_main.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_main.py", "target": "quantum_framework_menu.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_main.py", "target": "quantum_lab.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_menu.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_menu.py", "target": "quantum_framework_molecular.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_menu.py", "target": "higgs_four_lepton_analysis.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_menu.py", "target": "quantum_dash.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_menu.py", "target": "quantum_3dview.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_menu.py", "target": "quantum_visualizer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_framework_menu.py", "target": "app.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_lab.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_lab.py", "target": "quantum_framework_molecular.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_visualizer.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_visualizer.py", "target": "molecular_sim.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "quantum_visualizer.py", "target": "advanced_experiments.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_integration.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "qc_dashboard.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_qc_integration.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_quantum_framework.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_quantum_framework.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_quantum_framework.py", "target": "quantum_framework_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "topological_hilbert_compression2.py", "target": "quantum_computer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "topological_hilbert_compression2.py", "target": "quantum_computer.py"}], "generator": "readmenator", "metadata": {"edge_count": 12173, "file_count": 30, "language_count": 2, "symbol_count": 1790}, "nodes": [{"doc": "Advanced Quantum Experiments - Extension Pack ============================================== Extends the existing quantum simulator with:  1. GROVER'S ALGORITHM - Quantum search using quantum_computer.py 2. QED EFFECTS - Lamb shift, anomalous magnetic moment using relativistic_hydrogen.py 3. POLYATOMIC MOLECULES - H2O, NH3 using molecular_sim.py infrastructure  All built on top of existing PyTorch infrastructure.  Author: Gris Iscomeback License: AGPL v3", "id": "advanced_experiments.py", "kind": "module", "label": "advanced_experiments.py", "language": "py", "sha256": "87fd681b0912c4f7", "symbol_count": 56, "symbols": [{"kind": "function", "line": 121, "name": "_make_logger", "signature": "def _make_logger(name, level)"}, {"doc": "Configuration for Grover's algorithm experiments.", "kind": "class", "line": 141, "name": "GroverConfig", "signature": "class GroverConfig"}, {"doc": "Oracle for Grover's algorithm.\nMarks the target state by applying a phase flip.\n\nUses the existing MCZGate from quantum_computer.py for multi-controlled Z.", "kind": "class", "line": 159, "name": "GroverOracle", "signature": "class GroverOracle"}, {"doc": "Diffusion operator (Grover diffusion / inversion about mean).\n\nD = 2|s><s| - I where |s> = H^⊗n |0>\n\nImplemented using the existing gate infrastructure.", "kind": "class", "line": 194, "name": "GroverDiffusionOperator", "signature": "class GroverDiffusionOperator"}, {"doc": "Complete Grover's algorithm implementation using existing quantum_computer.py infrastructure.", "kind": "class", "line": 240, "name": "GroverSearch", "signature": "class GroverSearch"}, {"doc": "Configuration for QED effects calculations.", "kind": "class", "line": 404, "name": "QEDConfig", "signature": "class QEDConfig"}, {"doc": "Calculates the Lamb shift using Bethe's formula and more accurate methods.\n\nThe Lamb shift is the small difference between 2s_{1/2} and 2p_{1/2} energy levels\ndue to QED effects (vacuum fluctuations and self-energy).\n\nUses the existing Dirac infrastructure from relativistic_hydrogen.py", "kind": "class", "line": 430, "name": "LambShiftCalculator", "signature": "class LambShiftCalculator"}, {"doc": "Calculates the electron's anomalous magnetic moment (g-2).\n\nThe electron g-factor is slightly different from 2 due to QED effects:\ng = 2(1 + a_e) where a_e = α/(2π) + higher-order terms\n\nUses the existing Dirac infrastructure for baseline calculations.", "kind": "class", "line": 595, "name": "AnomalousMagneticMoment", "signature": "class AnomalousMagneticMoment"}, {"doc": "Complete QED effects experiment combining Lamb shift and g-2.\nUses existing relativistic_hydrogen.py infrastructure.", "kind": "class", "line": 722, "name": "QEDEffectsExperiment", "signature": "class QEDEffectsExperiment"}, {"doc": "Data for polyatomic molecules.", "kind": "class", "line": 831, "name": "PolyatomicMoleculeData", "signature": "class PolyatomicMoleculeData"}, {"doc": "Build molecule data for VQE calculations.\nUses existing molecular_sim.py infrastructure.", "kind": "class", "line": 849, "name": "MoleculeBuilder", "signature": "class MoleculeBuilder"}, {"doc": "VQE solver for polyatomic molecules.\nUses existing molecular_sim.py infrastructure.", "kind": "class", "line": 970, "name": "PolyatomicVQE", "signature": "class PolyatomicVQE"}, {"doc": "Complete polyatomic molecule experiment.", "kind": "class", "line": 1105, "name": "PolyatomicExperiment", "signature": "class PolyatomicExperiment"}, {"doc": "Runs all three advanced experiments.", "kind": "class", "line": 1223, "name": "AdvancedExperimentRunner", "signature": "class AdvancedExperimentRunner"}, {"doc": "Main entry point.", "kind": "method", "line": 1304, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 167, "name": "__init__", "signature": "def __init__(self, n_qubits, marked_state)"}, {"kind": "method", "line": 172, "name": "_validate", "signature": "def _validate(self)"}, {"doc": "Apply oracle: flip phase of marked state.\n|x> -> (-1)^{f(x)} |x> where f(x)=1 only for marked state.\n\nUses amplitude-level phase manipulation for exact implementation.", "kind": "method", "line": 176, "name": "apply", "signature": "def apply(self, state, backend)"}, {"kind": "method", "line": 203, "name": "__init__", "signature": "def __init__(self, n_qubits)"}, {"doc": "Apply diffusion operator using Hadamard + Oracle on |0> + Hadamard.\n\nD = H^⊗n (2|0><0| - I) H^⊗n", "kind": "method", "line": 206, "name": "apply", "signature": "def apply(self, state, backend)"}, {"kind": "method", "line": 245, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate Shannon entropy from probability distribution.\nH = -sum(p * log2(p))", "kind": "method", "line": 266, "name": "_calculate_entropy", "signature": "def _calculate_entropy(self, probs)"}, {"doc": "Initialize quantum computer using existing quantum_computer.py infrastructure.", "kind": "method", "line": 278, "name": "_init_quantum_computer", "signature": "def _init_quantum_computer(self)"}, {"doc": "Run Grover's search algorithm.\n\nReturns:\n    Dictionary with results including success probability and evolution history.", "kind": "method", "line": 300, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 440, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Bethe's non-relativistic formula for Lamb shift.\n\nΔE_Lamb = (8α^3 / 3πn^3) * |ψ_n(0)|^2 * ln(E_avg / E_n)\n\nFor s-states: |ψ_n(0)|^2 = Z^3 / (π n^3 a0^3)\n\nArgs:\n    n: Principal quantum number\n    l: Angular momentum quantum number\n    Z: Nuclear charge (default 1 for hydrogen)\n\nReturns:\n    Lamb shift in atomic units", "kind": "method", "line": 462, "name": "bethe_formula", "signature": "def bethe_formula(self, n, l, Z)"}, {"doc": "Approximate Lamb shift for l > 0.\nMuch smaller than for s-states.", "kind": "method", "line": 505, "name": "_higher_l_shift", "signature": "def _higher_l_shift(self, n, l, Z)"}, {"doc": "Calculate full Lamb shift including radiative corrections.\n\nΔE = ΔE_SE + ΔE_Uehling + ΔE_rel\n\nWhere:\n- ΔE_SE: Self-energy (main contribution)\n- ΔE_Uehling: Vacuum polarization (Uehling potential)\n- ΔE_rel: Relativistic corrections", "kind": "method", "line": 518, "name": "full_lamb_shift", "signature": "def full_lamb_shift(self, n, l, j, Z)"}, {"doc": "Compare Lamb shift for 2s_{1/2} and 2p_{1/2} states.\n\nThis is the classic Lamb shift measurement: the 2s_{1/2} - 2p_{1/2} splitting.\nExperimentally: ~1057.8 MHz", "kind": "method", "line": 558, "name": "compare_2s_2p", "signature": "def compare_2s_2p(self, Z)"}, {"kind": "method", "line": 605, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Schwinger's first-order result: a_e = α/(2π)\n\nThis is the leading QED correction.", "kind": "method", "line": 611, "name": "schwinger_term", "signature": "def schwinger_term(self)"}, {"doc": "Second-order correction: (α/π)^2 * C_2\nC_2 ≈ 0.328478965...", "kind": "method", "line": 619, "name": "second_order", "signature": "def second_order(self)"}, {"doc": "Third-order correction: (α/π)^3 * C_3\nC_3 ≈ 1.181241456...", "kind": "method", "line": 627, "name": "third_order", "signature": "def third_order(self)"}, {"doc": "Fourth-order correction: (α/π)^4 * C_4\nC_4 ≈ -1.9144(35)", "kind": "method", "line": 635, "name": "fourth_order", "signature": "def fourth_order(self)"}, {"doc": "Fifth-order correction: (α/π)^5 * C_5\nC_5 ≈ 7.7(1.1)", "kind": "method", "line": 643, "name": "fifth_order", "signature": "def fifth_order(self)"}, {"doc": "Calculate anomalous magnetic moment to specified order.\n\nArgs:\n    order: Maximum order to include (1-5)\n\nReturns:\n    Dictionary with contributions at each order", "kind": "method", "line": 651, "name": "calculate_a_e", "signature": "def calculate_a_e(self, order)"}, {"doc": "Generate a full report on g-2 calculations.", "kind": "method", "line": 686, "name": "full_report", "signature": "def full_report(self)"}, {"kind": "method", "line": 728, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Run complete QED analysis.", "kind": "method", "line": 736, "name": "run_full_analysis", "signature": "def run_full_analysis(self)"}, {"doc": "Calculate hydrogen energy levels including QED corrections.", "kind": "method", "line": 761, "name": "_calculate_energy_levels", "signature": "def _calculate_energy_levels(self)"}, {"doc": "Calculate Dirac energy level.\nUses existing relativistic_hydrogen.py if available.", "kind": "method", "line": 805, "name": "_dirac_energy", "signature": "def _dirac_energy(self, n, kappa)"}, {"doc": "Build water molecule geometry.\n\nH2O geometry:\n    H1 at (0, 0, 0)\n    O  at (r_OH, 0, 0)\n    H2 at (r_OH + r_OH*cos(θ), r_OH*sin(θ), 0)", "kind": "method", "line": 866, "name": "h2o", "signature": "def h2o(bond_length, angle_deg)"}, {"doc": "Build ammonia molecule geometry.\n\nNH3 has trigonal pyramidal geometry.", "kind": "method", "line": 898, "name": "nh3", "signature": "def nh3(bond_length, angle_deg)"}, {"doc": "Build methane molecule geometry.\n\nCH4 has tetrahedral geometry.", "kind": "method", "line": 933, "name": "ch4", "signature": "def ch4(bond_length)"}, {"kind": "method", "line": 976, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Run PySCF calculation for the molecule.", "kind": "method", "line": 999, "name": "run_pyscf", "signature": "def run_pyscf(self, molecule)"}, {"doc": "Return hardcoded reference values for common molecules.", "kind": "method", "line": 1066, "name": "_hardcoded_values", "signature": "def _hardcoded_values(self, molecule)"}, {"kind": "method", "line": 1110, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Run complete analysis for a molecule.", "kind": "method", "line": 1123, "name": "run_analysis", "signature": "def run_analysis(self, molecule_name)"}, {"doc": "Run analysis for all molecules.", "kind": "method", "line": 1164, "name": "run_all", "signature": "def run_all(self)"}, {"doc": "Scan potential energy surface by varying bond length.", "kind": "method", "line": 1175, "name": "scan_bond_length", "signature": "def scan_bond_length(self, molecule_name, r_min, r_max, n_points)"}, {"kind": "method", "line": 1228, "name": "__init__", "signature": "def __init__(self)"}, {"doc": "Run Grover's algorithm experiment.", "kind": "method", "line": 1235, "name": "run_grover", "signature": "def run_grover(self, n_qubits, marked_state)"}, {"doc": "Run QED effects experiment.", "kind": "method", "line": 1252, "name": "run_qed", "signature": "def run_qed(self)"}, {"doc": "Run polyatomic molecule experiment.", "kind": "method", "line": 1266, "name": "run_polyatomic", "signature": "def run_polyatomic(self, molecule)"}, {"doc": "Run all experiments.", "kind": "method", "line": 1278, "name": "run_all", "signature": "def run_all(self)"}]}, {"doc": "app.py — Corrected with proper particle-conserving ansatz =======================================================================  Root cause identified: The uccsd() function in molecular_sim.py implements singles excitations incorrectly. For excitation (o→v), it does: CNOT ladder → RY(v) → CNOT ladder inverse This ADDS an electron at v without REMOVING from o, producing |1110⟩ instead of |0110⟩. The states reachable by uccsd never include |0110⟩ or |1001⟩, which are exactly the states the dipole operator connects to |1100⟩ (HF). Hence <μ>=0 always, giving α=0.", "id": "app.py", "kind": "module", "label": "app.py", "language": "py", "sha256": "cf21176222bbe708", "symbol_count": 21, "symbols": [{"kind": "class", "line": 39, "name": "VQEResult", "signature": "class VQEResult"}, {"kind": "method", "line": 46, "name": "_sd_indices", "signature": "def _sd_indices(n_e, n_q)"}, {"kind": "method", "line": 55, "name": "_run_circuit", "signature": "def _run_circuit(circuit, backend, state)"}, {"doc": "Apply a particle-conserving single excitation rotation between \nqubits o (occupied) and v (virtual).\n\nRotates in the {|1_o 0_v⟩, |0_o 1_v⟩} subspace:\n    |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩\n    |0_o 1_v⟩ → -sin(θ)|1_o 0_v⟩ + cos(θ)|0_o 1_v⟩\n\nFor adjacent qubits: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)\nFor non-adjacent: SWAP chain to make adjacent, apply, SWAP back.", "kind": "method", "line": 61, "name": "givens_single_excitation", "signature": "def givens_single_excitation(state, o, v, theta, n_qubits, backend)"}, {"doc": "Particle-conserving UCCSD-like ansatz:\n- Singles: Givens rotations (correct particle conservation)\n- Doubles: reuse uccsd's double excitation (works correctly)", "kind": "method", "line": 109, "name": "particle_conserving_ansatz", "signature": "def particle_conserving_ansatz(state, thetas, singles, doubles, backend)"}, {"kind": "class", "line": 151, "name": "StarkEvaluator", "signature": "class StarkEvaluator"}, {"kind": "class", "line": 203, "name": "DipoleOperatorBuilder", "signature": "class DipoleOperatorBuilder"}, {"kind": "class", "line": 237, "name": "PolarizabilityCalculator", "signature": "class PolarizabilityCalculator"}, {"kind": "method", "line": 152, "name": "__init__", "signature": "def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)"}, {"kind": "method", "line": 161, "name": "_to_scalar", "signature": "def _to_scalar(amps)"}, {"kind": "method", "line": 165, "name": "_apply_pauli", "signature": "def _apply_pauli(self, amps, pauli)"}, {"kind": "method", "line": 184, "name": "eval_dipole", "signature": "def eval_dipole(self, amps_raw)"}, {"kind": "method", "line": 197, "name": "__call__", "signature": "def __call__(self, amps)"}, {"kind": "method", "line": 204, "name": "__init__", "signature": "def __init__(self, bond_length_angstrom)"}, {"kind": "method", "line": 238, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 258, "name": "_evaluator", "signature": "def _evaluator(self, field)"}, {"kind": "method", "line": 262, "name": "_get_state", "signature": "def _get_state(self, theta)"}, {"kind": "method", "line": 267, "name": "_diagnose", "signature": "def _diagnose(self, field, theta, label)"}, {"kind": "method", "line": 277, "name": "_optimize", "signature": "def _optimize(self, field, theta_init, n_restarts)"}, {"kind": "method", "line": 303, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 290, "name": "cost", "signature": "def cost(th)"}]}, {"doc": "Quantum Framework Demo - Production Version =========================================== Demonstrates all improvements and features of the refactored molecular module.  Improvements implemented: 1. OpenFermion for all Hamiltonians (NO hardcoded values) 2. Precision mode flag (direct vs MPS) 3. Smart initialization with MP2 + parameter scan 4. Cached Pauli operations (x10-100 speedup) 5. Particle-conserving subspace 6. TOML configuration", "id": "demo_molecular_vqe.py", "kind": "module", "label": "demo_molecular_vqe.py", "language": "py", "sha256": "8caf21b459f1f4b0", "symbol_count": 10, "symbols": [{"doc": "Print formatted header.", "kind": "function", "line": 47, "name": "print_header", "signature": "def print_header(title)"}, {"doc": "Check and report available dependencies.", "kind": "function", "line": 54, "name": "check_dependencies", "signature": "def check_dependencies()"}, {"doc": "Demo: H2 with direct statevector (precision mode).", "kind": "function", "line": 78, "name": "demo_h2_direct", "signature": "def demo_h2_direct()"}, {"doc": "Demo: H2 with MPS compression.", "kind": "function", "line": 104, "name": "demo_h2_mps", "signature": "def demo_h2_mps()"}, {"doc": "Demo: Compare direct vs MPS.", "kind": "function", "line": 131, "name": "demo_comparison", "signature": "def demo_comparison()"}, {"doc": "Demo: OpenFermion molecule builder.", "kind": "function", "line": 167, "name": "demo_molecule_builder", "signature": "def demo_molecule_builder()"}, {"doc": "Demo: Smart parameter initialization.", "kind": "function", "line": 197, "name": "demo_smart_initialization", "signature": "def demo_smart_initialization()"}, {"doc": "Demo: Cached Pauli operations.", "kind": "function", "line": 231, "name": "demo_cached_operations", "signature": "def demo_cached_operations()"}, {"doc": "Demo: Configuration from TOML.", "kind": "function", "line": 277, "name": "demo_config_from_toml", "signature": "def demo_config_from_toml()"}, {"doc": "Run all demos.", "kind": "function", "line": 302, "name": "main", "signature": "def main()"}]}, {"doc": "Entangled Hydrogen Visualization System ======================================== Demonstrates entangled hydrogen states using the trained quantum computer and molecular simulation backends.  This script IMPORTS and USES the existing modules: - quantum_computer.py for quantum state preparation and evolution - molecular_sim.py for molecular energy evaluation - relativistic_hydrogen.py for Dirac relativistic calculations - orbital_visualizer2.py for orbital visualization components  Author: Gris Iscomeback License: AGPL v3", "id": "entangled_hydrogen.py", "kind": "module", "label": "entangled_hydrogen.py", "language": "py", "sha256": "feaa00c962591a10", "symbol_count": 43, "symbols": [{"doc": "Create a module-level logger with consistent formatter.", "kind": "function", "line": 44, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"doc": "Configuration for entangled hydrogen visualization system.\n\nAll parameters are parametric and configurable from this class.", "kind": "class", "line": 61, "name": "EntangledHydrogenConfig", "signature": "class EntangledHydrogenConfig"}, {"doc": "Abstract interface for entangled quantum states.", "kind": "class", "line": 114, "name": "IEntangledState", "signature": "class IEntangledState(ABC)"}, {"doc": "Bell state: |Phi+> = (|00> + |11>) / sqrt(2).", "kind": "class", "line": 131, "name": "BellState", "signature": "class BellState(IEntangledState)"}, {"doc": "GHZ state: (|00...0> + |11...1>) / sqrt(2).", "kind": "class", "line": 145, "name": "GHZState", "signature": "class GHZState(IEntangledState)"}, {"doc": "W state: (|001> + |010> + |100>) / sqrt(3).", "kind": "class", "line": 162, "name": "WState", "signature": "class WState(IEntangledState)"}, {"doc": "Calculates hydrogen atom wavefunctions for entangled state visualization.\nUses the same implementation as orbital_visualizer2.py.", "kind": "class", "line": 194, "name": "WavefunctionCalculator", "signature": "class WavefunctionCalculator"}, {"doc": "Monte Carlo sampler for entangled hydrogen states.\nSamples from the joint probability distribution of entangled orbitals.", "kind": "class", "line": 253, "name": "EntangledHydrogenSampler", "signature": "class EntangledHydrogenSampler"}, {"doc": "Visualizer for entangled hydrogen states.\nCreates high-resolution visualizations similar to orbital_visualizer2.py.", "kind": "class", "line": 408, "name": "EntangledHydrogenVisualizer", "signature": "class EntangledHydrogenVisualizer"}, {"doc": "Main experiment class for entangled hydrogen visualization.\n\nUses the existing quantum_computer.py, molecular_sim.py, and\nvisualization components from orbital_visualizer2.py and relativistic_hydrogen.py.", "kind": "class", "line": 569, "name": "EntangledHydrogenExperiment", "signature": "class EntangledHydrogenExperiment"}, {"doc": "Main entry point for entangled hydrogen visualization.", "kind": "method", "line": 893, "name": "main", "signature": "def main()"}, {"doc": "Return the name of the entangled state.", "kind": "method", "line": 119, "name": "name", "signature": "def name(self)"}, {"doc": "Prepare the entangled state on n qubits.", "kind": "method", "line": 123, "name": "prepare", "signature": "def prepare(self, n_qubits)"}, {"doc": "Return the theoretical Shannon entropy in bits.", "kind": "method", "line": 127, "name": "get_theoretical_entropy", "signature": "def get_theoretical_entropy(self)"}, {"kind": "method", "line": 135, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 138, "name": "prepare", "signature": "def prepare(self, qc, backend)"}, {"kind": "method", "line": 141, "name": "get_theoretical_entropy", "signature": "def get_theoretical_entropy(self)"}, {"kind": "method", "line": 148, "name": "__init__", "signature": "def __init__(self, n_qubits)"}, {"kind": "method", "line": 152, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 155, "name": "prepare", "signature": "def prepare(self, qc, backend)"}, {"kind": "method", "line": 158, "name": "get_theoretical_entropy", "signature": "def get_theoretical_entropy(self)"}, {"kind": "method", "line": 165, "name": "__init__", "signature": "def __init__(self, n_qubits)"}, {"kind": "method", "line": 169, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 172, "name": "prepare", "signature": "def prepare(self, qc, backend, factory)"}, {"kind": "method", "line": 190, "name": "get_theoretical_entropy", "signature": "def get_theoretical_entropy(self)"}, {"kind": "method", "line": 200, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate non-relativistic radial wavefunction R_nl(r).", "kind": "method", "line": 204, "name": "radial_wavefunction", "signature": "def radial_wavefunction(n, l, r)"}, {"doc": "Calculate real spherical harmonics Y_lm(theta, phi).", "kind": "method", "line": 215, "name": "spherical_harmonic_real", "signature": "def spherical_harmonic_real(l, m, theta, phi)"}, {"doc": "Calculate wavefunction on 2D grid for quantum computer processing.", "kind": "method", "line": 225, "name": "psi_on_grid", "signature": "def psi_on_grid(self, n, l, m)"}, {"doc": "Calculate full 3D wavefunction psi_nlm(r, theta, phi).", "kind": "method", "line": 245, "name": "psi_3d", "signature": "def psi_3d(self, n, l, m, r, theta, phi)"}, {"kind": "method", "line": 259, "name": "__init__", "signature": "def __init__(self, config, wavefunction_calc)"}, {"doc": "Find maximum probability for rejection sampling.", "kind": "method", "line": 263, "name": "find_max_probability", "signature": "def find_max_probability(self, n, l, m)"}, {"doc": "Sample points from a single hydrogen orbital using Monte Carlo.", "kind": "method", "line": 295, "name": "sample_orbital", "signature": "def sample_orbital(self, n, l, m, num_samples)"}, {"doc": "Sample from entangled hydrogen state.\n\nCreates a superposition of two orbitals with entanglement_weight:\n|psi> = sqrt(1-w) * |n1,l1,m1> + sqrt(w) * |n2,l2,m2>\n\nFor true entanglement visualization, we create a joint state:\n|Psi> = (|n1,l1,m1>|n2,l2,m2> + |n2,l2,m2>|n1,l1,m1>) / sqrt(2)", "kind": "method", "line": 362, "name": "sample_entangled_state", "signature": "def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples, entanglement_weight)"}, {"kind": "method", "line": 414, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Create visualization of entangled hydrogen state.", "kind": "method", "line": 417, "name": "visualize", "signature": "def visualize(self, data, quantum_result, save_path)"}, {"kind": "method", "line": 577, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Initialize the quantum computer using the existing quantum_computer.py module.", "kind": "method", "line": 591, "name": "_initialize_quantum_computer", "signature": "def _initialize_quantum_computer(self)"}, {"doc": "Run Bell state entangled hydrogen visualization.\n\nCreates a Bell state and correlates it with hydrogen orbitals.", "kind": "method", "line": 630, "name": "run_bell_entangled_hydrogen", "signature": "def run_bell_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples, suffix)"}, {"doc": "Run GHZ state entangled hydrogen visualization.\n\nCreates a GHZ state with n qubits and correlates with n orbitals.", "kind": "method", "line": 672, "name": "run_ghz_entangled_hydrogen", "signature": "def run_ghz_entangled_hydrogen(self, orbitals, backend, num_samples)"}, {"doc": "Run entangled hydrogen with molecular energy evaluation using molecular_sim.py.", "kind": "method", "line": 745, "name": "run_entangled_h_with_molecular_energy", "signature": "def run_entangled_h_with_molecular_energy(self, n1, l1, m1, n2, l2, m2, backend, num_samples)"}, {"doc": "Run entangled hydrogen with relativistic Dirac calculations.\nUses components from relativistic_hydrogen.py.", "kind": "method", "line": 794, "name": "run_relativistic_entangled_hydrogen", "signature": "def run_relativistic_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples)"}, {"doc": "Run all available entangled hydrogen demonstrations.", "kind": "method", "line": 852, "name": "run_all_demonstrations", "signature": "def run_all_demonstrations(self, num_samples)"}]}, {"doc": "Higgs to Four Lepton Analysis - Quantum Backend Integration =================================================================  This script ACTUALLY uses the user's quantum computing backends: - HamiltonianBackend: Spectral neural network for Hamiltonian operations - SchrodingerBackend: Wave function evolution network - DiracBackend: Relativistic spinor network with gamma matrices  No fake implementations. No placeholders. Uses the neural networks.  Author: Gris Iscomeback License: AGPL v3", "id": "higgs_four_lepton_analysis.py", "kind": "module", "label": "higgs_four_lepton_analysis.py", "language": "py", "sha256": "29535e31af9a4593", "symbol_count": 44, "symbols": [{"kind": "function", "line": 57, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"kind": "class", "line": 72, "name": "LeptonType", "signature": "class LeptonType(Enum)"}, {"kind": "class", "line": 78, "name": "EventType", "signature": "class EventType(Enum)"}, {"kind": "class", "line": 86, "name": "Config", "signature": "class Config"}, {"kind": "class", "line": 125, "name": "FourMomentum", "signature": "class FourMomentum"}, {"kind": "class", "line": 164, "name": "Lepton", "signature": "class Lepton"}, {"kind": "class", "line": 183, "name": "Event", "signature": "class Event"}, {"doc": "Uses the ACTUAL DiracBackend from quantum_computer.py for spinor calculations.\n\nThis is NOT a fake implementation - it uses the neural network\nspectral layers trained (or randomly initialized) for Dirac spinor evolution.", "kind": "class", "line": 205, "name": "QuantumSpinorProcessor", "signature": "class QuantumSpinorProcessor"}, {"doc": "Parse CMS CSV files with actual column names.", "kind": "class", "line": 377, "name": "EventParser", "signature": "class EventParser"}, {"doc": "3D visualization using Plotly with quantum-processed trajectories.", "kind": "class", "line": 436, "name": "Visualizer", "signature": "class Visualizer"}, {"doc": "Main analysis class using quantum backends from quantum_computer.py", "kind": "class", "line": 612, "name": "HiggsQuantumAnalysis", "signature": "class HiggsQuantumAnalysis"}, {"kind": "method", "line": 747, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 136, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 151, "name": "from_energy_momentum", "signature": "def from_energy_momentum(cls, E, px, py, pz)"}, {"kind": "method", "line": 154, "name": "__add__", "signature": "def __add__(self, other)"}, {"kind": "method", "line": 171, "name": "pt", "signature": "def pt(self)"}, {"kind": "method", "line": 173, "name": "eta", "signature": "def eta(self)"}, {"kind": "method", "line": 175, "name": "phi", "signature": "def phi(self)"}, {"kind": "method", "line": 177, "name": "energy", "signature": "def energy(self)"}, {"kind": "method", "line": 179, "name": "mass", "signature": "def mass(self)"}, {"kind": "method", "line": 193, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 200, "name": "check_higgs", "signature": "def check_higgs(self, cfg)"}, {"kind": "method", "line": 213, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Precompute k-space grids for Dirac equation.", "kind": "method", "line": 247, "name": "_precompute_momentum_grids", "signature": "def _precompute_momentum_grids(self)"}, {"doc": "Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)\nsuitable for processing by the DiracBackend.\n\nThe wavefunction encodes the momentum as a plane wave with the correct\nrelativistic dispersion relation.", "kind": "method", "line": 255, "name": "momentum_to_spinor_wavefunction", "signature": "def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)"}, {"doc": "Evolve a wavefunction using the DiracBackend neural network.\n\nThis applies the actual spectral layers from the trained (or random)\nDirac network.", "kind": "method", "line": 296, "name": "evolve_with_dirac_backend", "signature": "def evolve_with_dirac_backend(self, psi, steps)"}, {"doc": "Evolve using the SchrodingerBackend neural network.", "kind": "method", "line": 308, "name": "evolve_with_schrodinger_backend", "signature": "def evolve_with_schrodinger_backend(self, psi, steps)"}, {"doc": "Evolve using the HamiltonianBackend neural network.", "kind": "method", "line": 315, "name": "evolve_with_hamiltonian_backend", "signature": "def evolve_with_hamiltonian_backend(self, psi, steps)"}, {"doc": "Compute the Dirac current using the actual neural network backends.\n\nReturns (jx, jy, jz, info_dict) where info contains quantum observables.", "kind": "method", "line": 322, "name": "compute_dirac_current", "signature": "def compute_dirac_current(self, px, py, pz, energy, mass, charge)"}, {"doc": "Compute complex amplitude from wavefunction for helicity analysis.", "kind": "method", "line": 370, "name": "compute_spinor_amplitude", "signature": "def compute_spinor_amplitude(self, psi)"}, {"kind": "method", "line": 380, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 383, "name": "parse_file", "signature": "def parse_file(self, filepath, event_type)"}, {"kind": "method", "line": 396, "name": "_parse_row", "signature": "def _parse_row(self, row, event_type)"}, {"kind": "method", "line": 439, "name": "__init__", "signature": "def __init__(self, config, quantum_processor)"}, {"doc": "Compute helical trajectory using actual DiracBackend evolution.", "kind": "method", "line": 443, "name": "compute_quantum_helix", "signature": "def compute_quantum_helix(self, px, py, pz, charge, mass, energy)"}, {"kind": "method", "line": 474, "name": "create_visualization", "signature": "def create_visualization(self, events, output_path)"}, {"kind": "method", "line": 566, "name": "_create_detector", "signature": "def _create_detector(self)"}, {"kind": "method", "line": 581, "name": "_create_explosion", "signature": "def _create_explosion(self, vx, vy, vz, energy)"}, {"kind": "method", "line": 617, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 630, "name": "fetch_data", "signature": "def fetch_data(self)"}, {"kind": "method", "line": 645, "name": "load_events", "signature": "def load_events(self)"}, {"doc": "Run analysis using actual quantum backends.", "kind": "method", "line": 672, "name": "analyze_with_quantum_backends", "signature": "def analyze_with_quantum_backends(self)"}, {"kind": "method", "line": 728, "name": "generate_visualization", "signature": "def generate_visualization(self)"}, {"kind": "method", "line": 734, "name": "run", "signature": "def run(self)"}]}, {"doc": "Higgs to Four Lepton Analysis - Quantum Backend Integration =================================================================  This script ACTUALLY uses the user's quantum computing backends: - HamiltonianBackend: Spectral neural network for Hamiltonian operations - SchrodingerBackend: Wave function evolution network - DiracBackend: Relativistic spinor network with gamma matrices  No fake implementations. No placeholders. Uses the neural networks.  Author: Gris Iscomeback License: AGPL v3", "id": "higgs_quantum_analysis.py", "kind": "module", "label": "higgs_quantum_analysis.py", "language": "py", "sha256": "6ca9f4646633e223", "symbol_count": 44, "symbols": [{"kind": "function", "line": 57, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"kind": "class", "line": 72, "name": "LeptonType", "signature": "class LeptonType(Enum)"}, {"kind": "class", "line": 78, "name": "EventType", "signature": "class EventType(Enum)"}, {"kind": "class", "line": 86, "name": "Config", "signature": "class Config"}, {"kind": "class", "line": 125, "name": "FourMomentum", "signature": "class FourMomentum"}, {"kind": "class", "line": 164, "name": "Lepton", "signature": "class Lepton"}, {"kind": "class", "line": 183, "name": "Event", "signature": "class Event"}, {"doc": "Uses the ACTUAL DiracBackend from quantum_computer.py for spinor calculations.\n\nThis is NOT a fake implementation - it uses the neural network\nspectral layers trained (or randomly initialized) for Dirac spinor evolution.", "kind": "class", "line": 205, "name": "QuantumSpinorProcessor", "signature": "class QuantumSpinorProcessor"}, {"doc": "Parse CMS CSV files with actual column names.", "kind": "class", "line": 377, "name": "EventParser", "signature": "class EventParser"}, {"doc": "3D visualization using Plotly with quantum-processed trajectories.", "kind": "class", "line": 436, "name": "Visualizer", "signature": "class Visualizer"}, {"doc": "Main analysis class using quantum backends from quantum_computer.py", "kind": "class", "line": 612, "name": "HiggsQuantumAnalysis", "signature": "class HiggsQuantumAnalysis"}, {"kind": "method", "line": 747, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 136, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 151, "name": "from_energy_momentum", "signature": "def from_energy_momentum(cls, E, px, py, pz)"}, {"kind": "method", "line": 154, "name": "__add__", "signature": "def __add__(self, other)"}, {"kind": "method", "line": 171, "name": "pt", "signature": "def pt(self)"}, {"kind": "method", "line": 173, "name": "eta", "signature": "def eta(self)"}, {"kind": "method", "line": 175, "name": "phi", "signature": "def phi(self)"}, {"kind": "method", "line": 177, "name": "energy", "signature": "def energy(self)"}, {"kind": "method", "line": 179, "name": "mass", "signature": "def mass(self)"}, {"kind": "method", "line": 193, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 200, "name": "check_higgs", "signature": "def check_higgs(self, cfg)"}, {"kind": "method", "line": 213, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Precompute k-space grids for Dirac equation.", "kind": "method", "line": 247, "name": "_precompute_momentum_grids", "signature": "def _precompute_momentum_grids(self)"}, {"doc": "Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)\nsuitable for processing by the DiracBackend.\n\nThe wavefunction encodes the momentum as a plane wave with the correct\nrelativistic dispersion relation.", "kind": "method", "line": 255, "name": "momentum_to_spinor_wavefunction", "signature": "def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)"}, {"doc": "Evolve a wavefunction using the DiracBackend neural network.\n\nThis applies the actual spectral layers from the trained (or random)\nDirac network.", "kind": "method", "line": 296, "name": "evolve_with_dirac_backend", "signature": "def evolve_with_dirac_backend(self, psi, steps)"}, {"doc": "Evolve using the SchrodingerBackend neural network.", "kind": "method", "line": 308, "name": "evolve_with_schrodinger_backend", "signature": "def evolve_with_schrodinger_backend(self, psi, steps)"}, {"doc": "Evolve using the HamiltonianBackend neural network.", "kind": "method", "line": 315, "name": "evolve_with_hamiltonian_backend", "signature": "def evolve_with_hamiltonian_backend(self, psi, steps)"}, {"doc": "Compute the Dirac current using the actual neural network backends.\n\nReturns (jx, jy, jz, info_dict) where info contains quantum observables.", "kind": "method", "line": 322, "name": "compute_dirac_current", "signature": "def compute_dirac_current(self, px, py, pz, energy, mass, charge)"}, {"doc": "Compute complex amplitude from wavefunction for helicity analysis.", "kind": "method", "line": 370, "name": "compute_spinor_amplitude", "signature": "def compute_spinor_amplitude(self, psi)"}, {"kind": "method", "line": 380, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 383, "name": "parse_file", "signature": "def parse_file(self, filepath, event_type)"}, {"kind": "method", "line": 396, "name": "_parse_row", "signature": "def _parse_row(self, row, event_type)"}, {"kind": "method", "line": 439, "name": "__init__", "signature": "def __init__(self, config, quantum_processor)"}, {"doc": "Compute helical trajectory using actual DiracBackend evolution.", "kind": "method", "line": 443, "name": "compute_quantum_helix", "signature": "def compute_quantum_helix(self, px, py, pz, charge, mass, energy)"}, {"kind": "method", "line": 474, "name": "create_visualization", "signature": "def create_visualization(self, events, output_path)"}, {"kind": "method", "line": 566, "name": "_create_detector", "signature": "def _create_detector(self)"}, {"kind": "method", "line": 581, "name": "_create_explosion", "signature": "def _create_explosion(self, vx, vy, vz, energy)"}, {"kind": "method", "line": 617, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 630, "name": "fetch_data", "signature": "def fetch_data(self)"}, {"kind": "method", "line": 645, "name": "load_events", "signature": "def load_events(self)"}, {"doc": "Run analysis using actual quantum backends.", "kind": "method", "line": 672, "name": "analyze_with_quantum_backends", "signature": "def analyze_with_quantum_backends(self)"}, {"kind": "method", "line": 728, "name": "generate_visualization", "signature": "def generate_visualization(self)"}, {"kind": "method", "line": 734, "name": "run", "signature": "def run(self)"}]}, {"id": "install.sh", "kind": "module", "label": "install.sh", "language": "sh", "sha256": "c907d80fd6734993", "symbol_count": 0, "symbols": []}, {"doc": "molecular_simulator.py - VERSIÓN CON OPENFERMION CORREGIDO", "id": "molecular_sim.py", "kind": "module", "label": "molecular_sim.py", "language": "py", "sha256": "ae4e90ef6f4fcc4a", "symbol_count": 26, "symbols": [{"kind": "function", "line": 18, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"kind": "class", "line": 35, "name": "MoleculeData", "signature": "class MoleculeData"}, {"doc": "H2 con datos PySCF.", "kind": "method", "line": 41, "name": "_h2_sto3g_pyscf", "signature": "def _h2_sto3g_pyscf()"}, {"doc": "H2 con datos hardcodeados.", "kind": "method", "line": 75, "name": "_h2_sto3g_hardcoded", "signature": "def _h2_sto3g_hardcoded()"}, {"doc": "Construye Hamiltoniano JW usando OpenFermion correctamente.\nUsa MolecularData de OpenFermion para asegurar consistencia.", "kind": "method", "line": 111, "name": "build_jw_hamiltonian_of", "signature": "def build_jw_hamiltonian_of(mol)"}, {"doc": "Evaluador exacto usando OpenFermion JW.", "kind": "class", "line": 187, "name": "ExactJWEnergy", "signature": "class ExactJWEnergy"}, {"doc": "Backend neuronal con calibración.", "kind": "class", "line": 265, "name": "SurrogateEnergy", "signature": "class SurrogateEnergy"}, {"kind": "method", "line": 304, "name": "prepare_hf", "signature": "def prepare_hf(mol, factory, backend)"}, {"kind": "method", "line": 312, "name": "_sd_indices", "signature": "def _sd_indices(n_e, n_q)"}, {"doc": "UCCSD ansatz.\n\n- Singles: cadena JW estándar, conserva número de partículas.\n- Doubles: rotación Givens en el subespacio {|HF⟩, |exc⟩} determinado\n  dinámicamente desde las amplitudes (robusto ante cambio de convención).\n- theta=0 para cualquier parámetro → circuito vacío → identidad exacta.", "kind": "method", "line": 321, "name": "uccsd", "signature": "def uccsd(state, thetas, singles, doubles, backend, runner)"}, {"kind": "class", "line": 371, "name": "VQEResult", "signature": "class VQEResult"}, {"kind": "class", "line": 390, "name": "VQESolver", "signature": "class VQESolver"}, {"kind": "method", "line": 190, "name": "__init__", "signature": "def __init__(self, mol, n_qubits)"}, {"doc": "(dim,2,G,G) → (dim,2)  ó  (dim,2) → (dim,2).\nIntegra la función de onda espacial sobre la grilla manteniendo\nla estructura re/im que necesita el evaluador JW.", "kind": "method", "line": 203, "name": "_to_scalar", "signature": "def _to_scalar(amps)"}, {"kind": "method", "line": 213, "name": "_verify_hf", "signature": "def _verify_hf(self)"}, {"kind": "method", "line": 223, "name": "_apply", "signature": "def _apply(self, amps, pauli)"}, {"kind": "method", "line": 243, "name": "_evaluate", "signature": "def _evaluate(self, amps)"}, {"kind": "method", "line": 257, "name": "__call__", "signature": "def __call__(self, amps)"}, {"kind": "method", "line": 268, "name": "__init__", "signature": "def __init__(self, mol, n_qubits, exact_eval, backend)"}, {"kind": "method", "line": 277, "name": "calibrate", "signature": "def calibrate(self, hf_amps)"}, {"kind": "method", "line": 290, "name": "cost_with_barrier", "signature": "def cost_with_barrier(self, amps)"}, {"kind": "method", "line": 377, "name": "__repr__", "signature": "def __repr__(self)"}, {"kind": "method", "line": 391, "name": "__init__", "signature": "def __init__(self, qc, config)"}, {"kind": "method", "line": 395, "name": "_run", "signature": "def _run(self, circ, be, state)"}, {"kind": "method", "line": 403, "name": "run", "signature": "def run(self, mol, backend, max_iter, tol)"}, {"kind": "method", "line": 451, "name": "cost", "signature": "def cost(thetas)"}]}, {"doc": "Hydrogen Orbital Visualizer ============================ HIGH RESOLUTION visualization with Hamiltonian NN. FIXED: Large image size, proper point sizes, high quality output.", "id": "orbital_visualizer2.py", "kind": "module", "label": "orbital_visualizer2.py", "language": "py", "sha256": "6af587728e08ac34", "symbol_count": 17, "symbols": [{"kind": "class", "line": 44, "name": "Config", "signature": "class Config"}, {"doc": "Calculates hydrogen atom wavefunctions.", "kind": "class", "line": 83, "name": "WavefunctionCalculator", "signature": "class WavefunctionCalculator"}, {"doc": "Uses YOUR TRAINED MODEL for calculations.", "kind": "class", "line": 126, "name": "HamiltonianNNProcessor", "signature": "class HamiltonianNNProcessor"}, {"doc": "Monte Carlo sampling for orbital visualization.", "kind": "class", "line": 160, "name": "MonteCarloSampler", "signature": "class MonteCarloSampler"}, {"doc": "HIGH RESOLUTION visualization - NOT 16x16!", "kind": "class", "line": 265, "name": "OrbitalVisualizer", "signature": "class OrbitalVisualizer"}, {"kind": "method", "line": 433, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 87, "name": "radial_wavefunction", "signature": "def radial_wavefunction(n, l, r)"}, {"kind": "method", "line": 97, "name": "spherical_harmonic_real", "signature": "def spherical_harmonic_real(l, m, theta, phi)"}, {"kind": "method", "line": 107, "name": "psi_on_grid", "signature": "def psi_on_grid(n, l, m, grid_size)"}, {"kind": "method", "line": 129, "name": "__init__", "signature": "def __init__(self, engine)"}, {"kind": "method", "line": 133, "name": "is_model_loaded", "signature": "def is_model_loaded(self)"}, {"kind": "method", "line": 136, "name": "compute_expected_energy", "signature": "def compute_expected_energy(self, n, l, m)"}, {"kind": "method", "line": 163, "name": "__init__", "signature": "def __init__(self, hamiltonian_processor)"}, {"kind": "method", "line": 166, "name": "find_max_probability", "signature": "def find_max_probability(self, n, l, m)"}, {"kind": "method", "line": 195, "name": "sample", "signature": "def sample(self, n, l, m, num_samples)"}, {"kind": "method", "line": 268, "name": "visualize", "signature": "def visualize(self, data, save_path, hamiltonian_processor)"}, {"kind": "method", "line": 396, "name": "_plotly", "signature": "def _plotly(self, X, Y, Z, prob_norm, phases, n, l, m)"}]}, {"doc": "polarizability_v3.py — Corrected with proper particle-conserving ansatz =======================================================================  Root cause identified: The uccsd() function in molecular_sim.py implements singles excitations incorrectly. For excitation (o→v), it does: CNOT ladder → RY(v) → CNOT ladder inverse This ADDS an electron at v without REMOVING from o, producing |1110⟩ instead of |0110⟩. The states reachable by uccsd never include |0110⟩ or |1001⟩, which are exactly the states the dipole operator connects to |1100⟩ (HF). Hence <μ>=0 always, giving α=0.  Fix: Implement proper Givens-rotation-based singles that conserve particle number. For adjacent qubits, Givens(o,v,θ) is: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o) This rotates in the {|10⟩, |01⟩} subspace: |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩  For non-adjacent qubits, we SWAP to make them adjacent, apply Givens, then SWAP back. This preserves all quantum numbers.  We keep the doubles excitation from uccsd since it works correctly (verified: it produces |1100⟩↔|0011⟩ rotation properly).", "id": "polarizability_v3.py", "kind": "module", "label": "polarizability_v3.py", "language": "py", "sha256": "13aa04d08c3e233f", "symbol_count": 21, "symbols": [{"kind": "class", "line": 51, "name": "VQEResult", "signature": "class VQEResult"}, {"kind": "method", "line": 58, "name": "_sd_indices", "signature": "def _sd_indices(n_e, n_q)"}, {"kind": "method", "line": 67, "name": "_run_circuit", "signature": "def _run_circuit(circuit, backend, state)"}, {"doc": "Apply a particle-conserving single excitation rotation between \nqubits o (occupied) and v (virtual).\n\nRotates in the {|1_o 0_v⟩, |0_o 1_v⟩} subspace:\n    |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩\n    |0_o 1_v⟩ → -sin(θ)|1_o 0_v⟩ + cos(θ)|0_o 1_v⟩\n\nFor adjacent qubits: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)\nFor non-adjacent: SWAP chain to make adjacent, apply, SWAP back.", "kind": "method", "line": 73, "name": "givens_single_excitation", "signature": "def givens_single_excitation(state, o, v, theta, n_qubits, backend)"}, {"doc": "Particle-conserving UCCSD-like ansatz:\n- Singles: Givens rotations (correct particle conservation)\n- Doubles: reuse uccsd's double excitation (works correctly)", "kind": "method", "line": 121, "name": "particle_conserving_ansatz", "signature": "def particle_conserving_ansatz(state, thetas, singles, doubles, backend)"}, {"kind": "class", "line": 163, "name": "StarkEvaluator", "signature": "class StarkEvaluator"}, {"kind": "class", "line": 215, "name": "DipoleOperatorBuilder", "signature": "class DipoleOperatorBuilder"}, {"kind": "class", "line": 249, "name": "PolarizabilityCalculator", "signature": "class PolarizabilityCalculator"}, {"kind": "method", "line": 164, "name": "__init__", "signature": "def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)"}, {"kind": "method", "line": 173, "name": "_to_scalar", "signature": "def _to_scalar(amps)"}, {"kind": "method", "line": 177, "name": "_apply_pauli", "signature": "def _apply_pauli(self, amps, pauli)"}, {"kind": "method", "line": 196, "name": "eval_dipole", "signature": "def eval_dipole(self, amps_raw)"}, {"kind": "method", "line": 209, "name": "__call__", "signature": "def __call__(self, amps)"}, {"kind": "method", "line": 216, "name": "__init__", "signature": "def __init__(self, bond_length_angstrom)"}, {"kind": "method", "line": 250, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 270, "name": "_evaluator", "signature": "def _evaluator(self, field)"}, {"kind": "method", "line": 274, "name": "_get_state", "signature": "def _get_state(self, theta)"}, {"kind": "method", "line": 279, "name": "_diagnose", "signature": "def _diagnose(self, field, theta, label)"}, {"kind": "method", "line": 289, "name": "_optimize", "signature": "def _optimize(self, field, theta_init, n_restarts)"}, {"kind": "method", "line": 315, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 302, "name": "cost", "signature": "def cost(th)"}]}, {"doc": "QC Dashboard - Streamlit web application for real-time quantum playground.  Provides an interactive web-based quantum circuit builder with: - Interactive circuit construction via gate buttons - Real-time state visualisation (probability bars, Bloch spheres, phase space) - OpenQASM code editor with live preview and export/import - Entropy evolution tracking + entanglement analysis (cut-position sweep, scaling) - Molecular H2 VQE simulation (energy convergence, landscape, orbital visualisation) - 3D visualisation via Plotly (Bloch spheres, probability bars, state vector) - Backend comparison (MPS, statevector)  Architecture ------------ DashboardConfig         -- centralised configuration (no magic numbers) VisualisationEngine     -- renders matplotlib figures from snapshots SimulatorBackend        -- thin wrapper around MPS / statevector / synthetic H2VQESolver             -- self-contained H2 VQE (numpy only, no PySCF) DashboardApp            -- top-level Streamlit application orchestrator  Usage ----- streamlit run qc_dashboard.py  Or programmatically: from qc_dashboard import DashboardApp, DashboardConfig app = DashboardApp(DashboardConfig()) app.run()", "id": "qc_dashboard.py", "kind": "module", "label": "qc_dashboard.py", "language": "py", "sha256": "3c6bac18bbd4fb97", "symbol_count": 88, "symbols": [{"doc": "Centralised configuration for the dashboard.", "kind": "class", "line": 55, "name": "DashboardConfig", "signature": "class DashboardConfig"}, {"doc": "A gate placed in the circuit builder.", "kind": "class", "line": 120, "name": "GateItem", "signature": "class GateItem"}, {"doc": "Quantum state snapshot for visualisation.", "kind": "class", "line": 130, "name": "SnapshotData", "signature": "class SnapshotData"}, {"doc": "Self-contained H2 VQE solver using the hardcoded STO-3G Hamiltonian.\n\nThe Hamiltonian in the Jordan-Wigner 2-qubit active space:\n    H = E_nuc + hZZ * Z0Z1 + hXX * X0X1 + hYY * Y0Y1\n\nReference energies:\n    E_HF = -1.11675928 Ha,  E_FCI = -1.13728383 Ha,  E_nuc = 0.71996899 Ha\n\nUsage\n-----\n    solver = H2VQESolver()\n    result = solver.run_vqe()\n    print(result[\"energy\"])  # converged VQE energy\n\n    landscape = solver.energy_landscape()\n    print(landscape[\"bond_lengths\"], landscape[\"energies\"])", "kind": "class", "line": 147, "name": "H2VQESolver", "signature": "class H2VQESolver"}, {"doc": "Renders figures from quantum state snapshots using matplotlib.", "kind": "class", "line": 340, "name": "VisualisationEngine", "signature": "class VisualisationEngine"}, {"doc": "Thin wrapper around the QC framework simulator for dashboard use.", "kind": "class", "line": 725, "name": "SimulatorBackend", "signature": "class SimulatorBackend"}, {"doc": "Optional Plotly-based 3D visualisations.", "kind": "class", "line": 979, "name": "Plotly3DEngine", "signature": "class Plotly3DEngine"}, {"doc": "Run a function that creates a matplotlib figure and capture it as PNG bytes.", "kind": "method", "line": 1133, "name": "_capture_mpl_fig", "signature": "def _capture_mpl_fig(func)"}, {"doc": "Wrapper around the repo's real orbital_visualizer2.py scripts.", "kind": "class", "line": 1151, "name": "RealOrbitalEngine", "signature": "class RealOrbitalEngine"}, {"doc": "Wrapper around the repo's real quantum_dash.py and quantum_3dview.py.", "kind": "class", "line": 1269, "name": "BrutalVizEngine", "signature": "class BrutalVizEngine"}, {"doc": "Compare circuit execution across all available backends.", "kind": "class", "line": 1344, "name": "BackendComparator", "signature": "class BackendComparator"}, {"doc": "Streamlit-based interactive quantum playground.", "kind": "class", "line": 1371, "name": "DashboardApp", "signature": "class DashboardApp"}, {"doc": "Launch the Streamlit dashboard.", "kind": "method", "line": 1994, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 166, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 173, "name": "_pauli_operators", "signature": "def _pauli_operators(n_qubits, qubit, op)"}, {"kind": "method", "line": 189, "name": "_pauli_string_matrix", "signature": "def _pauli_string_matrix(paulis, n_qubits)"}, {"doc": "Build H2 Hamiltonian matrix for given bond length (Angstrom).\n\nThe coefficients scale with bond length to reproduce the Morse-like well.", "kind": "method", "line": 206, "name": "_build_hamiltonian", "signature": "def _build_hamiltonian(self, bond_length)"}, {"doc": "UCC-like ansatz for H2: |psi(theta)> = cos(theta)*|10> + sin(theta)*|01>.", "kind": "method", "line": 229, "name": "_ansatz_state", "signature": "def _ansatz_state(theta)"}, {"kind": "method", "line": 241, "name": "_energy", "signature": "def _energy(self, theta, h_matrix, e_nuc)"}, {"kind": "method", "line": 248, "name": "run_vqe", "signature": "def run_vqe(self, bond_length, max_iter)"}, {"doc": "Sweep bond length and return VQE energy curve.", "kind": "method", "line": 293, "name": "energy_landscape", "signature": "def energy_landscape(self)"}, {"doc": "Compute hydrogen 1s orbital wavefunction along the internuclear axis.", "kind": "method", "line": 318, "name": "orbital_wavefunction", "signature": "def orbital_wavefunction(bond_length, grid_points)"}, {"kind": "method", "line": 343, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 349, "name": "_init_plotting", "signature": "def _init_plotting(self)"}, {"kind": "method", "line": 362, "name": "available", "signature": "def available(self)"}, {"kind": "method", "line": 365, "name": "render_full_dashboard", "signature": "def render_full_dashboard(self, snapshots, current)"}, {"kind": "method", "line": 403, "name": "render_entropy_chart", "signature": "def render_entropy_chart(self, snapshots)"}, {"doc": "Entropy vs cut position for the latest snapshot.", "kind": "method", "line": 429, "name": "render_entanglement_profile", "signature": "def render_entanglement_profile(self, snapshots)"}, {"kind": "method", "line": 474, "name": "render_vqe_convergence", "signature": "def render_vqe_convergence(self, convergence, e_hf, e_fci)"}, {"kind": "method", "line": 501, "name": "render_energy_landscape", "signature": "def render_energy_landscape(self, bond_lengths, vqe_energies, hf_energies, fci_energies)"}, {"kind": "method", "line": 533, "name": "render_orbital_plot", "signature": "def render_orbital_plot(self, orbital_data)"}, {"kind": "method", "line": 562, "name": "render_entropy_scaling", "signature": "def render_entropy_scaling(self, data)"}, {"kind": "method", "line": 589, "name": "_render_probabilities", "signature": "def _render_probabilities(self, snap, ax)"}, {"kind": "method", "line": 606, "name": "_render_bloch_sphere", "signature": "def _render_bloch_sphere(self, snap, ax)"}, {"kind": "method", "line": 635, "name": "_render_phase_space", "signature": "def _render_phase_space(self, snap, ax)"}, {"kind": "method", "line": 670, "name": "render_orbital_2d_projections", "signature": "def render_orbital_2d_projections(self, data)"}, {"kind": "method", "line": 715, "name": "_hex_to_rgb", "signature": "def _hex_to_rgb(h)"}, {"kind": "method", "line": 728, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 737, "name": "_init_framework", "signature": "def _init_framework(self)"}, {"kind": "method", "line": 760, "name": "_get_mps_gate_registry", "signature": "def _get_mps_gate_registry()"}, {"kind": "method", "line": 768, "name": "_get_sv_gate_registry", "signature": "def _get_sv_gate_registry()"}, {"kind": "method", "line": 775, "name": "execute_circuit", "signature": "def execute_circuit(self, gates, n_qubits)"}, {"kind": "method", "line": 788, "name": "_mps_execute", "signature": "def _mps_execute(self, gates, n_qubits)"}, {"kind": "method", "line": 839, "name": "_sv_execute", "signature": "def _sv_execute(self, gates, n_qubits)"}, {"kind": "method", "line": 886, "name": "_snapshot_from_mps", "signature": "def _snapshot_from_mps(self, state, step, gate_name)"}, {"kind": "method", "line": 898, "name": "_snapshot_from_sv", "signature": "def _snapshot_from_sv(self, state, step, gate_name)"}, {"kind": "method", "line": 917, "name": "_compute_bloch_mps", "signature": "def _compute_bloch_mps(state, n_qubits)"}, {"kind": "method", "line": 933, "name": "_synthetic_execute", "signature": "def _synthetic_execute(self, gates, n_qubits)"}, {"kind": "method", "line": 982, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 988, "name": "_init_plotly", "signature": "def _init_plotly(self)"}, {"kind": "method", "line": 998, "name": "available", "signature": "def available(self)"}, {"kind": "method", "line": 1001, "name": "render_bloch_3d", "signature": "def render_bloch_3d(self, bloch_vectors)"}, {"kind": "method", "line": 1049, "name": "render_probability_3d", "signature": "def render_probability_3d(self, probabilities, n_qubits)"}, {"doc": "3D scatter plot: X=real, Y=imaginary, Z=probability.", "kind": "method", "line": 1080, "name": "render_state_3d", "signature": "def render_state_3d(self, probabilities, phases)"}, {"kind": "method", "line": 1154, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 1163, "name": "_init_real", "signature": "def _init_real(self)"}, {"kind": "method", "line": 1210, "name": "available", "signature": "def available(self)"}, {"kind": "method", "line": 1214, "name": "entangled_available", "signature": "def entangled_available(self)"}, {"kind": "method", "line": 1217, "name": "sample", "signature": "def sample(self, n, l, m, num_samples)"}, {"kind": "method", "line": 1222, "name": "render_to_bytes", "signature": "def render_to_bytes(self, data)"}, {"kind": "method", "line": 1240, "name": "sample_entangled", "signature": "def sample_entangled(self, n1, l1, m1, n2, l2, m2, num_samples)"}, {"kind": "method", "line": 1250, "name": "render_entangled_to_bytes", "signature": "def render_entangled_to_bytes(self, data)"}, {"kind": "method", "line": 1272, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 1279, "name": "_init_real", "signature": "def _init_real(self)"}, {"kind": "method", "line": 1297, "name": "dash_available", "signature": "def dash_available(self)"}, {"kind": "method", "line": 1301, "name": "hologram_available", "signature": "def hologram_available(self)"}, {"kind": "method", "line": 1304, "name": "run_brutal_viz", "signature": "def run_brutal_viz(self, circuit_name)"}, {"kind": "method", "line": 1327, "name": "render_hologram", "signature": "def render_hologram(self, snapshots, backend_comp)"}, {"kind": "method", "line": 1347, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1351, "name": "run_comparison", "signature": "def run_comparison(self, gates, n_qubits)"}, {"kind": "method", "line": 1374, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1384, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 1399, "name": "_ensure_streamlit", "signature": "def _ensure_streamlit()"}, {"kind": "method", "line": 1408, "name": "_init_session", "signature": "def _init_session(st)"}, {"kind": "method", "line": 1432, "name": "_render_ui", "signature": "def _render_ui(self, st)"}, {"kind": "method", "line": 1447, "name": "_render_sidebar_controls", "signature": "def _render_sidebar_controls(self, st)"}, {"kind": "method", "line": 1535, "name": "_render_main_panel", "signature": "def _render_main_panel(self, st)"}, {"kind": "method", "line": 1563, "name": "_render_playground_tab", "signature": "def _render_playground_tab(self, st)"}, {"kind": "method", "line": 1606, "name": "_render_qasm_tab", "signature": "def _render_qasm_tab(self, st)"}, {"kind": "method", "line": 1669, "name": "_render_entropy_tab", "signature": "def _render_entropy_tab(self, st)"}, {"kind": "method", "line": 1688, "name": "_render_entanglement_tab", "signature": "def _render_entanglement_tab(self, st)"}, {"kind": "method", "line": 1732, "name": "_render_molecule_tab", "signature": "def _render_molecule_tab(self, st)"}, {"kind": "method", "line": 1803, "name": "_render_3d_tab", "signature": "def _render_3d_tab(self, st)"}, {"kind": "method", "line": 1855, "name": "_render_orbital_tab", "signature": "def _render_orbital_tab(self, st)"}, {"kind": "method", "line": 1958, "name": "_auto_run", "signature": "def _auto_run(self, st)"}, {"kind": "method", "line": 1969, "name": "_build_qasm_from_gates", "signature": "def _build_qasm_from_gates(self, gates)"}, {"kind": "class", "line": 1176, "name": "_FakeConfig", "signature": "class _FakeConfig"}, {"kind": "method", "line": 1936, "name": "_parse_orb", "signature": "def _parse_orb(k)"}]}, {"doc": "QC Integration Bridge - OpenQASM, Qiskit, and PennyLane interoperability layer.  Provides bidirectional conversion between the QC quantum circuit representation and standard quantum computing formats: - OpenQASM 2.0 (export/import) - Qiskit QuantumCircuit (export/import when qiskit is installed) - PennyLane QNode / tape (export/import when pennylane is installed)  Architecture ------------ IntegrationConfig    -- centralised configuration (no hardcoded values) IQCAdapter          -- abstract interface for each target format OpenQasmAdapter     -- OpenQASM 2.0 string <-> internal circuit QiskitAdapter       -- Qiskit QuantumCircuit <-> internal circuit PennyLaneAdapter    -- PennyLane operations <-> internal circuit IntegrationBridge   -- facade that delegates to the correct adapter  Usage ----- from qc_integration import IntegrationBridge, IntegrationConfig  config = IntegrationConfig() bridge = IntegrationBridge(config)  # Export to OpenQASM qasm_str = bridge.export_qasm(circuit)  # Import from OpenQASM", "id": "qc_integration.py", "kind": "module", "label": "qc_integration.py", "language": "py", "sha256": "a643f5ad88ff8726", "symbol_count": 61, "symbols": [{"doc": "Centralised configuration for the integration bridge.\n\nAll tunable parameters live here -- no hardcoded values in logic.", "kind": "class", "line": 77, "name": "IntegrationConfig", "signature": "class IntegrationConfig"}, {"doc": "A single quantum gate instruction.", "kind": "class", "line": 130, "name": "GateInstruction", "signature": "class GateInstruction"}, {"doc": "Intermediate representation of a quantum circuit.\n\nThis is the lingua franca between the QC framework and external formats.", "kind": "class", "line": 146, "name": "CircuitIR", "signature": "class CircuitIR"}, {"doc": "Interface for a format-specific adapter (Interface Segregation).", "kind": "class", "line": 181, "name": "IQCAdapter", "signature": "class IQCAdapter(ABC)"}, {"doc": "Adapter for OpenQASM 2.0 format.", "kind": "class", "line": 226, "name": "OpenQasmAdapter", "signature": "class OpenQasmAdapter(IQCAdapter)"}, {"doc": "Adapter for Qiskit QuantumCircuit format.", "kind": "class", "line": 358, "name": "QiskitAdapter", "signature": "class QiskitAdapter(IQCAdapter)"}, {"doc": "Adapter for PennyLane format.", "kind": "class", "line": 455, "name": "PennyLaneAdapter", "signature": "class PennyLaneAdapter(IQCAdapter)"}, {"doc": "Converts between CircuitIR and the actual QC framework circuit types.\n\nSupports both:\n  - quantum_framework_core.MPSQuantumComputer / QuantumCircuit (MPS)\n  - quantum_computer.QuantumComputer / QuantumCircuit (statevector)", "kind": "class", "line": 557, "name": "FrameworkAdapter", "signature": "class FrameworkAdapter"}, {"doc": "Build CircuitIR instances for common quantum algorithms.", "kind": "class", "line": 659, "name": "StandardCircuitFactory", "signature": "class StandardCircuitFactory"}, {"doc": "Facade that exposes all format conversions through a single API.\n\nUsage\n-----\n    bridge = IntegrationBridge()\n    cir = StandardCircuitFactory.bell_state()\n\n    # OpenQASM\n    qasm_str = bridge.export_qasm(cir)\n    assert isinstance(qasm_str, str)\n\n    cir_restored = bridge.import_qasm(qasm_str)\n    assert cir_restored.n_qubits == cir.n_qubits\n\n    # Qiskit (if installed)\n    if bridge._qiskit_available:\n        qc_qiskit = bridge.to_qiskit(cir)\n        cir_back = bridge.from_qiskit(qc_qiskit)\n\n    # PennyLane (if installed)\n    if bridge._pennylane_available:\n        qnode = bridge.to_pennylane(cir)\n        result = qnode()", "kind": "class", "line": 727, "name": "IntegrationBridge", "signature": "class IntegrationBridge"}, {"doc": "Command-line interface for the integration bridge.", "kind": "method", "line": 851, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 115, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 137, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 141, "name": "num_qubits", "signature": "def num_qubits(self)"}, {"kind": "method", "line": 156, "name": "append", "signature": "def append(self, gate)"}, {"kind": "method", "line": 164, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 167, "name": "__repr__", "signature": "def __repr__(self)"}, {"doc": "Export a CircuitIR to the target format.", "kind": "method", "line": 185, "name": "export", "signature": "def export(self, circuit)"}, {"doc": "Import from the target format into a CircuitIR.", "kind": "method", "line": 189, "name": "import_", "signature": "def import_(self, data)"}, {"kind": "method", "line": 238, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 243, "name": "export", "signature": "def export(self, circuit)"}, {"doc": "Parse an OpenQASM 2.0 string into a CircuitIR.", "kind": "method", "line": 277, "name": "import_", "signature": "def import_(self, data)"}, {"kind": "method", "line": 339, "name": "_param_names_for_gate", "signature": "def _param_names_for_gate(name)"}, {"kind": "method", "line": 361, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 364, "name": "export", "signature": "def export(self, circuit)"}, {"kind": "method", "line": 379, "name": "import_", "signature": "def import_(self, data)"}, {"kind": "method", "line": 403, "name": "_ensure_qiskit", "signature": "def _ensure_qiskit()"}, {"kind": "method", "line": 414, "name": "_build_qiskit_method_map", "signature": "def _build_qiskit_method_map(qc)"}, {"kind": "method", "line": 433, "name": "_build_reverse_gate_map", "signature": "def _build_reverse_gate_map()"}, {"kind": "method", "line": 458, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 461, "name": "export", "signature": "def export(self, circuit)"}, {"kind": "method", "line": 483, "name": "import_", "signature": "def import_(self, data)"}, {"kind": "method", "line": 506, "name": "_ensure_pennylane", "signature": "def _ensure_pennylane()"}, {"kind": "method", "line": 517, "name": "_build_gate_ops", "signature": "def _build_gate_ops(pl)"}, {"kind": "method", "line": 535, "name": "_build_reverse_ops", "signature": "def _build_reverse_ops(pl)"}, {"kind": "method", "line": 565, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Extract a CircuitIR from a framework circuit object.", "kind": "method", "line": 568, "name": "to_circuit_ir", "signature": "def to_circuit_ir(self, circuit, framework_type)"}, {"doc": "Build a framework circuit object from a CircuitIR.", "kind": "method", "line": 584, "name": "from_circuit_ir", "signature": "def from_circuit_ir(self, cir, framework_type)"}, {"kind": "method", "line": 598, "name": "_detect_framework", "signature": "def _detect_framework(circuit)"}, {"kind": "method", "line": 607, "name": "_from_mps_circuit", "signature": "def _from_mps_circuit(circuit)"}, {"kind": "method", "line": 623, "name": "_to_mps_circuit", "signature": "def _to_mps_circuit(cir)"}, {"kind": "method", "line": 631, "name": "_from_sv_circuit", "signature": "def _from_sv_circuit(circuit)"}, {"kind": "method", "line": 647, "name": "_to_sv_circuit", "signature": "def _to_sv_circuit(cir)"}, {"kind": "method", "line": 663, "name": "bell_state", "signature": "def bell_state()"}, {"kind": "method", "line": 670, "name": "ghz_state", "signature": "def ghz_state(n_qubits)"}, {"kind": "method", "line": 678, "name": "qft", "signature": "def qft(n_qubits)"}, {"kind": "method", "line": 693, "name": "w_state", "signature": "def w_state(n_qubits)"}, {"kind": "method", "line": 701, "name": "grover", "signature": "def grover(n_qubits, marked, iterations)"}, {"kind": "method", "line": 753, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 763, "name": "_init_optional_adapters", "signature": "def _init_optional_adapters(self)"}, {"kind": "method", "line": 780, "name": "export_qasm", "signature": "def export_qasm(self, circuit)"}, {"kind": "method", "line": 783, "name": "import_qasm", "signature": "def import_qasm(self, qasm_str)"}, {"kind": "method", "line": 788, "name": "to_qiskit", "signature": "def to_qiskit(self, circuit)"}, {"kind": "method", "line": 793, "name": "from_qiskit", "signature": "def from_qiskit(self, qiskit_circuit)"}, {"kind": "method", "line": 800, "name": "to_pennylane", "signature": "def to_pennylane(self, circuit)"}, {"kind": "method", "line": 805, "name": "from_pennylane", "signature": "def from_pennylane(self, pennylane_data)"}, {"kind": "method", "line": 812, "name": "to_circuit_ir", "signature": "def to_circuit_ir(self, circuit, framework_type)"}, {"kind": "method", "line": 819, "name": "from_circuit_ir", "signature": "def from_circuit_ir(self, cir, framework_type)"}, {"kind": "method", "line": 828, "name": "export_qasm_from_framework", "signature": "def export_qasm_from_framework(self, circuit, framework_type)"}, {"kind": "method", "line": 837, "name": "import_qasm_to_framework", "signature": "def import_qasm_to_framework(self, qasm_str, framework_type)"}, {"kind": "method", "line": 467, "name": "circuit_fn", "signature": "def circuit_fn()"}]}, {"doc": "quantum_brutalist.py - Ultra-High Fidelity Quantum Visualization ================================================================ Real-time holographic quantum state visualization with particle effects, neural network topology mapping, and immersive 3D interaction.  Author: Gris Iscomeback", "id": "quantum_3dview.py", "kind": "module", "label": "quantum_3dview.py", "language": "py", "sha256": "64f907fe6d0c2638", "symbol_count": 27, "symbols": [{"kind": "class", "line": 32, "name": "BrutalTheme", "signature": "class BrutalTheme(Enum)"}, {"kind": "class", "line": 39, "name": "BrutalConfig", "signature": "class BrutalConfig"}, {"doc": "Visualizador holográfico 3D de estados cuánticos", "kind": "class", "line": 84, "name": "QuantumHologram", "signature": "class QuantumHologram"}, {"doc": "Visualiza la topología interna de las redes neuronales cuánticas", "kind": "class", "line": 403, "name": "QuantumNeuralTopology", "signature": "class QuantumNeuralTopology"}, {"doc": "Convierte estados cuánticos en audio para percepción alternativa", "kind": "class", "line": 500, "name": "QuantumSonification", "signature": "class QuantumSonification"}, {"doc": "Dashboard interactivo completo con todas las visualizaciones brutales", "kind": "class", "line": 560, "name": "BrutalDashboard", "signature": "class BrutalDashboard"}, {"doc": "Demostración de visualización brutal", "kind": "method", "line": 614, "name": "demo_brutal", "signature": "def demo_brutal()"}, {"doc": "Crea datos sintéticos para demostración", "kind": "method", "line": 653, "name": "create_synthetic_snapshots", "signature": "def create_synthetic_snapshots()"}, {"kind": "method", "line": 52, "name": "colors", "signature": "def colors(self)"}, {"kind": "method", "line": 87, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Crea visualización holográfica de amplitudes en 3D con efecto de partículas", "kind": "method", "line": 92, "name": "create_amplitude_hologram", "signature": "def create_amplitude_hologram(self, snapshots, backend_comparison)"}, {"doc": "Añade campo de amplitudes 3D con efecto de partículas flotantes", "kind": "method", "line": 148, "name": "_add_holographic_field", "signature": "def _add_holographic_field(self, fig, snapshots, row, col)"}, {"doc": "Añade trazas de entropía con efecto de cola luminosa", "kind": "method", "line": 245, "name": "_add_entropy_trails", "signature": "def _add_entropy_trails(self, fig, snapshots, row, col)"}, {"doc": "Esfera de Bloch con efectos de holograma y partículas orbitales", "kind": "method", "line": 292, "name": "_add_bloch_sphere_holographic", "signature": "def _add_bloch_sphere_holographic(self, fig, snapshot, backend_name, row, col)"}, {"doc": "Interpola entre dos colores hex", "kind": "method", "line": 389, "name": "_interpolate_color", "signature": "def _interpolate_color(self, color1, color2, factor)"}, {"kind": "method", "line": 406, "name": "__init__", "signature": "def __init__(self, model)"}, {"doc": "Crea mapa 3D de la arquitectura neuronal con activaciones en tiempo real", "kind": "method", "line": 410, "name": "create_topology_map", "signature": "def create_topology_map(self)"}, {"doc": "Extrae información de capas del modelo PyTorch", "kind": "method", "line": 477, "name": "_extract_layers", "signature": "def _extract_layers(self, model)"}, {"doc": "Genera topología representativa de backend cuántico", "kind": "method", "line": 490, "name": "_generate_quantum_topology", "signature": "def _generate_quantum_topology(self)"}, {"kind": "method", "line": 503, "name": "__init__", "signature": "def __init__(self, sample_rate)"}, {"doc": "Convierte un estado cuántico en onda de audio\n- Amplitudes controlan volumen\n- Fases controlan paneo estéreo\n- Probabilidades controlan frecuencia", "kind": "method", "line": 506, "name": "state_to_audio", "signature": "def state_to_audio(self, snapshot, duration)"}, {"doc": "Genera envolvente ADSR proporcional a la intensidad del estado", "kind": "method", "line": 538, "name": "_adsr_envelope", "signature": "def _adsr_envelope(self, length, intensity)"}, {"kind": "method", "line": 563, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Genera reporte completo con múltiples visualizaciones", "kind": "method", "line": 570, "name": "generate_full_report", "signature": "def generate_full_report(self, snapshots, backend_comparison)"}, {"doc": "Guarda audio como WAV", "kind": "method", "line": 605, "name": "_save_audio", "signature": "def _save_audio(self, audio, path)"}, {"kind": "method", "line": 391, "name": "hex_to_rgb", "signature": "def hex_to_rgb(hex_color)"}, {"kind": "method", "line": 395, "name": "rgb_to_hex", "signature": "def rgb_to_hex(rgb)"}]}, {"doc": "quantum_computer.py  Author: Gris Iscomeback License: AGPL v3  Collapse-Free Quantum Computer Simulator on Classical Hardware.  The state of n qubits lives in the JOINT Hilbert space C^(2^n). The state vector has 2^n complex amplitudes: one per computational basis state. This is the only representation that correctly supports entanglement.  Each amplitude alpha_k (k in {0,...,2^n - 1}) is encoded as a 2D spatial wavefunction on a (G, G) grid, using the neural physics backends as the time-evolution engine. The joint state tensor has shape:  amplitudes: (2^n, 2, G, G) dim 0 : computational basis index  (2^n states) dim 1 : real / imaginary channel   (2 channels) dim 2 : spatial x                  (G points) dim 3 : spatial y                  (G points)  Single-qubit gates act via einsum on the qubit index within dim 0. Two-qubit gates (CNOT, CZ, SWAP) permute and mix amplitude pairs in dim 0. Measurement reads Born probabilities from norm-squared without collapsing.  Architecture (SOLID): - IQuantumGate         : gate abstraction (Interface Segregation) - IPhysicsBackend      : physics engine abstraction (Dependency Inversion)", "id": "quantum_computer.py", "kind": "module", "label": "quantum_computer.py", "language": "py", "sha256": "d019133208e599e7", "symbol_count": 163, "symbols": [{"doc": "Create a module-level logger with a consistent formatter.", "kind": "function", "line": 58, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"doc": "Global configuration for the quantum computer simulator.\n\ngrid_size, hidden_dim, expansion_dim, num_spectral_layers must match\nthe values used when training the checkpoint files.", "kind": "class", "line": 74, "name": "SimulatorConfig", "signature": "class SimulatorConfig"}, {"doc": "Spectral convolution in frequency domain.\n\nLearns complex kernels that modulate Fourier coefficients.\nArchitecture is identical to the training scripts.", "kind": "class", "line": 105, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"doc": "Hamiltonian backbone: single-channel field -> H|psi>.\nShared by all physics backends as the H operator.", "kind": "class", "line": 144, "name": "HamiltonianBackboneNet", "signature": "class HamiltonianBackboneNet(Module)"}, {"doc": "Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction.", "kind": "class", "line": 171, "name": "SchrodingerSpectralNet", "signature": "class SchrodingerSpectralNet(Module)"}, {"doc": "Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor.", "kind": "class", "line": 200, "name": "DiracSpectralNet", "signature": "class DiracSpectralNet(Module)"}, {"doc": "Dirac gamma matrices in Dirac or Weyl representation.", "kind": "class", "line": 233, "name": "GammaMatrices", "signature": "class GammaMatrices"}, {"doc": "Joint quantum state of n qubits in the full 2^n dimensional Hilbert space.\n\nThe state is stored as a tensor of shape (2^n, 2, G, G):\n    - dim 0: computational basis index k in {0, ..., 2^n - 1}\n             bit j of k is the state of qubit j (MSB = qubit 0)\n    - dim 1: channel 0 = real part, channel 1 = imaginary part\n    - dim 2: spatial x (G grid points)\n    - dim 3: spatial y (G grid points)\n\nEach amplitude alpha_k is a spatial wavefunction. The overall quantum\namplitude for basis state |k> is the complex field alpha_k(x,y).\nThe Born probability of measuring |k> is:\n\n    P(k) = integral |alpha_k(x,y)|^2 dx dy\n         = sum_{x,y} (alpha_k_real^2 + alpha_k_imag^2)\n\nnormalized so that sum_k P(k) = 1.\n\nThis representation correctly supports:\n    - Superposition: multiple k indices have non-zero amplitude\n    - Entanglement: amplitudes do not factorize across qubits\n    - Coherent multi-qubit gates: exact permutation and mixing of amplitudes", "kind": "class", "line": 279, "name": "JointHilbertState", "signature": "class JointHilbertState"}, {"doc": "Spatial potentials for eigenstate initialization.", "kind": "class", "line": 390, "name": "PotentialGenerator", "signature": "class PotentialGenerator"}, {"doc": "Solve the 1D marginal Hamiltonian and return the n-th eigenstate\nas a normalized 2-channel (2, G, G) real tensor.", "kind": "method", "line": 440, "name": "_solve_eigenstate", "signature": "def _solve_eigenstate(config, potential, n)"}, {"doc": "Build the (2, G, G) spatial wavefunction for amplitude at basis index basis_idx.\n\nEach computational basis state gets its own spatial eigenstate profile.\nThe excitation level is proportional to the popcount of the basis index.", "kind": "method", "line": 461, "name": "_build_basis_amplitude", "signature": "def _build_basis_amplitude(config, basis_idx)"}, {"doc": "Builds JointHilbertState tensors for common initial conditions.", "kind": "class", "line": 474, "name": "JointStateFactory", "signature": "class JointStateFactory"}, {"doc": "Abstract physics backend for spatial wavefunction evolution.", "kind": "class", "line": 511, "name": "IPhysicsBackend", "signature": "class IPhysicsBackend(ABC)"}, {"doc": "Physics backend driven by the Hamiltonian neural network.\n\nPerforms first-order Schrodinger time evolution:\n    psi(t+dt) = psi(t) - i*dt*H*psi(t)", "kind": "class", "line": 523, "name": "HamiltonianBackend", "signature": "class HamiltonianBackend(IPhysicsBackend)"}, {"doc": "Physics backend driven by the Schrodinger network.\n\nUses the learned 2-channel spectral network for wavefunction propagation.\nFalls back to HamiltonianBackend if checkpoint is unavailable.", "kind": "class", "line": 591, "name": "SchrodingerBackend", "signature": "class SchrodingerBackend(IPhysicsBackend)"}, {"doc": "Physics backend driven by the Dirac network.\n\nExpands each (2,G,G) amplitude to a 4-component spinor, propagates\nvia the Dirac network, then projects back to (2,G,G).", "kind": "class", "line": 640, "name": "DiracBackend", "signature": "class DiracBackend(IPhysicsBackend)"}, {"doc": "Apply a 2x2 unitary u to qubit j in the joint Hilbert space.\n\nFor each pair of basis states (k0, k1) that differ only in bit j:\n    alpha_{k0}' = u[0,0]*alpha_{k0} + u[0,1]*alpha_{k1}\n    alpha_{k1}' = u[1,0]*alpha_{k0} + u[1,1]*alpha_{k1}\n\nComplex scalar * (2,G,G) amplitude:\n    (a+ib)(psi_r + i*psi_i) = (a*psi_r - b*psi_i) + i*(a*psi_i + b*psi_r)\n\nThis is exact, preserves unitarity, and correctly creates superpositions.", "kind": "method", "line": 739, "name": "_single_qubit_unitary", "signature": "def _single_qubit_unitary(state, qubit, u, backend)"}, {"doc": "Apply a 4x4 unitary in the {|00>,|01>,|10>,|11>} subspace of (ctrl, tgt).\n\nFor each group of 4 basis states sharing all bits except ctrl and tgt,\napply the 4x4 unitary to the amplitude quadruplet.\n\nOrdering within the 4x4 block: |00>=0, |01>=1, |10>=2, |11>=3\n(first bit = ctrl, second bit = tgt).\n\nThis correctly implements CNOT, CZ, SWAP and any 2-qubit gate.", "kind": "method", "line": 789, "name": "_two_qubit_unitary", "signature": "def _two_qubit_unitary(state, ctrl, tgt, u4)"}, {"doc": "Abstract quantum gate operating on the joint Hilbert space.", "kind": "class", "line": 844, "name": "IQuantumGate", "signature": "class IQuantumGate(ABC)"}, {"doc": "H = [[1,1],[1,-1]] / sqrt(2).", "kind": "class", "line": 863, "name": "HadamardGate", "signature": "class HadamardGate(IQuantumGate)"}, {"doc": "X = [[0,1],[1,0]].", "kind": "class", "line": 878, "name": "PauliXGate", "signature": "class PauliXGate(IQuantumGate)"}, {"doc": "Y = [[0,-i],[i,0]].", "kind": "class", "line": 892, "name": "PauliYGate", "signature": "class PauliYGate(IQuantumGate)"}, {"doc": "Z = [[1,0],[0,-1]].", "kind": "class", "line": 906, "name": "PauliZGate", "signature": "class PauliZGate(IQuantumGate)"}, {"doc": "S = [[1,0],[0,i]].", "kind": "class", "line": 920, "name": "SGate", "signature": "class SGate(IQuantumGate)"}, {"doc": "T = [[1,0],[0,e^{i*pi/4}]].", "kind": "class", "line": 934, "name": "TGate", "signature": "class TGate(IQuantumGate)"}, {"doc": "Rx(theta) = exp(-i*theta/2 * X).", "kind": "class", "line": 949, "name": "RxGate", "signature": "class RxGate(IQuantumGate)"}, {"doc": "Ry(theta) = exp(-i*theta/2 * Y).", "kind": "class", "line": 965, "name": "RyGate", "signature": "class RyGate(IQuantumGate)"}, {"doc": "Rz(theta) = exp(-i*theta/2 * Z).", "kind": "class", "line": 981, "name": "RzGate", "signature": "class RzGate(IQuantumGate)"}, {"doc": "CNOT: |ctrl tgt> -> |ctrl, ctrl XOR tgt>.\n\n4x4 matrix (|00>,|01>,|10>,|11>):\n    |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>", "kind": "class", "line": 998, "name": "CNOTGate", "signature": "class CNOTGate(IQuantumGate)"}, {"doc": "CZ: applies phase -1 to |11>.\n\n4x4 matrix: diag(1, 1, 1, -1).", "kind": "class", "line": 1022, "name": "CZGate", "signature": "class CZGate(IQuantumGate)"}, {"doc": "SWAP: exchanges two qubits.\n\n4x4 matrix: |01>->|10>, |10>->|01>, others unchanged.", "kind": "class", "line": 1045, "name": "SWAPGate", "signature": "class SWAPGate(IQuantumGate)"}, {"doc": "Toffoli (CCX): flips target iff both controls are |1>.\n\nExact amplitude permutation in the 8-element 3-qubit subspace.", "kind": "class", "line": 1068, "name": "ToffoliGate", "signature": "class ToffoliGate(IQuantumGate)"}, {"doc": "Multi-Controlled Z gate: applies phase -1 to the single basis state\nwhere ALL qubits in targets are |1>.\n\nThis is the exact oracle primitive needed by Grover's algorithm.\nFor n target qubits it marks the state |11...1> with a global phase of -1\nand leaves all other basis states unchanged.\n\nImplementation: iterate over all basis states k; if every bit\ncorresponding to a qubit in targets is set to 1, negate that amplitude\n(multiply real and imaginary parts by -1).\n\ntargets: list of qubit indices that must all be |1> for the phase flip.", "kind": "class", "line": 1098, "name": "MCZGate", "signature": "class MCZGate(IQuantumGate)"}, {"doc": "Free Hamiltonian evolution applied to every amplitude in the joint state.\n\nUses the active physics backend.\nparams: {\"dt\": float, \"steps\": int}", "kind": "class", "line": 1130, "name": "EvolveGate", "signature": "class EvolveGate(IQuantumGate)"}, {"doc": "Register a custom gate without modifying existing code (Open/Closed Principle).", "kind": "method", "line": 1179, "name": "register_gate", "signature": "def register_gate(name, gate)"}, {"doc": "Single gate instruction.", "kind": "class", "line": 1186, "name": "CircuitInstruction", "signature": "class CircuitInstruction"}, {"doc": "Ordered sequence of quantum gate instructions.\n\nPure data structure: stores instructions, does not execute them.", "kind": "class", "line": 1193, "name": "QuantumCircuit", "signature": "class QuantumCircuit"}, {"doc": "Non-destructive Born-rule measurement of the full register.\n\nContains the complete probability distribution over all 2^n basis states,\nper-qubit marginals, and Bloch vectors. The state is never modified.", "kind": "class", "line": 1279, "name": "MeasurementResult", "signature": "class MeasurementResult"}, {"doc": "Collapse-free quantum computer simulator with joint Hilbert space.\n\nThe n-qubit state is stored as a (2^n, 2, G, G) tensor. All gates\noperate via exact unitary transformations on amplitude pairs. Measurement\nis non-destructive Born-rule readout — the state is never collapsed.\n\nBackends:\n    \"hamiltonian\" : Hamiltonian NN spectral operator\n    \"schrodinger\" : Schrodinger evolution network\n    \"dirac\"       : Dirac relativistic spinor network\n\nUsage:\n    config = SimulatorConfig(\n        hamiltonian_checkpoint=\"weights/latest.pth\",\n        schrodinger_checkpoint=\"weights/schrodinger_crystal_final.pth\",\n        dirac_checkpoint=\"weights/dirac_phase5_latest.pth\",\n    )\n    qc = QuantumComputer(config)\n    circuit = QuantumCircuit(2)\n    circuit.h(0).cnot(0, 1)\n    result = qc.run(circuit, backend=\"schrodinger\")\n    print(result)\n    # Most probable: |00> P=0.5  or  |11> P=0.5\n    # Shannon entropy: 1.0000 bits  <- true entanglement", "kind": "class", "line": 1333, "name": "QuantumComputer", "signature": "class QuantumComputer"}, {"doc": "Print PASS/FAIL for a single assertion. Returns True if passed.", "kind": "method", "line": 1602, "name": "_check", "signature": "def _check(label, condition)"}, {"doc": "Property-based test suite for quantum phase coherence and unitarity.\n\nEach test has a known exact analytic answer derived from the unitary\nalgebra. Tests are designed to be sensitive to phase errors — circuits\nwhere wrong relative phases produce DIFFERENT probabilities, not just\ndifferent phases that cancel out at measurement.\n\nReturns the number of failed tests.", "kind": "method", "line": 1609, "name": "run_phase_tests", "signature": "def run_phase_tests(config)"}, {"doc": "Run demo suite validating entanglement on all backends.", "kind": "method", "line": 1836, "name": "_demo", "signature": "def _demo(config)"}, {"kind": "method", "line": 113, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"doc": "Apply spectral convolution via RFFT2.", "kind": "method", "line": 124, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 150, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers)"}, {"doc": "Accepts (G,G), (1,G,G), or (B,1,G,G). Returns squeezed output.", "kind": "method", "line": 159, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 176, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)"}, {"doc": "(2,G,G) or (B,2,G,G) -> same shape.", "kind": "method", "line": 188, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 205, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)"}, {"doc": "(8,G,G) or (B,8,G,G) -> same shape.", "kind": "method", "line": 217, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 236, "name": "__init__", "signature": "def __init__(self, representation, device)"}, {"kind": "method", "line": 241, "name": "_init_matrices", "signature": "def _init_matrices(self)"}, {"doc": "Move all matrices to device.", "kind": "method", "line": 272, "name": "to", "signature": "def to(self, device)"}, {"kind": "method", "line": 305, "name": "__init__", "signature": "def __init__(self, amplitudes, n_qubits)"}, {"doc": "In-place normalization: sum_k P(k) = 1.", "kind": "method", "line": 317, "name": "normalize_", "signature": "def normalize_(self)"}, {"doc": "Return (2^n,) tensor of Born probabilities P(k) for each basis state.", "kind": "method", "line": 323, "name": "probabilities", "signature": "def probabilities(self)"}, {"doc": "Marginal Born probability P(qubit_j = |1>).\n\nSums P(k) over all basis states k where bit j == 1.\nBit ordering: qubit 0 is the MSB of k.", "kind": "method", "line": 328, "name": "marginal_probability_one", "signature": "def marginal_probability_one(self, qubit)"}, {"doc": "Return the index k with the highest probability.", "kind": "method", "line": 343, "name": "most_probable_basis_state", "signature": "def most_probable_basis_state(self)"}, {"doc": "Compute the reduced Bloch vector for qubit j by partial trace.\n\nrho_j[0,0] = P(qubit=0), rho_j[1,1] = P(qubit=1)\nrho_j[0,1] = sum_{pairs} alpha_{k0}^* alpha_{k1} (off-diagonal coherence)\nbx = 2 Re(rho_j[0,1]), by = -2 Im(rho_j[0,1]), bz = P(0) - P(1)", "kind": "method", "line": 347, "name": "bloch_vector", "signature": "def bloch_vector(self, qubit)"}, {"doc": "Return a deep copy.", "kind": "method", "line": 385, "name": "clone", "signature": "def clone(self)"}, {"kind": "method", "line": 393, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 397, "name": "_grid", "signature": "def _grid(self)"}, {"doc": "V = k/2 * r^2.", "kind": "method", "line": 402, "name": "harmonic", "signature": "def harmonic(self)"}, {"doc": "Double-well along x.", "kind": "method", "line": 410, "name": "double_well", "signature": "def double_well(self)"}, {"doc": "Coulomb-like V ~ -1/r.", "kind": "method", "line": 417, "name": "coulomb", "signature": "def coulomb(self)"}, {"doc": "Periodic cosine lattice.", "kind": "method", "line": 424, "name": "periodic_lattice", "signature": "def periodic_lattice(self)"}, {"doc": "Dirichlet-weighted mixture of all four potentials.", "kind": "method", "line": 429, "name": "mixed", "signature": "def mixed(self, seed)"}, {"kind": "method", "line": 477, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 480, "name": "_empty", "signature": "def _empty(self, n_qubits)"}, {"doc": "Initialize register in |00...0>.", "kind": "method", "line": 484, "name": "all_zeros", "signature": "def all_zeros(self, n_qubits)"}, {"doc": "Initialize register in computational basis state |k>.", "kind": "method", "line": 492, "name": "basis_state", "signature": "def basis_state(self, n_qubits, k)"}, {"doc": "Initialize in the basis state given by binary string.", "kind": "method", "line": 502, "name": "from_bitstring", "signature": "def from_bitstring(self, bitstring)"}, {"doc": "Evolve a single (2, G, G) wavefunction by dt under H.", "kind": "method", "line": 515, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"doc": "Apply global phase e^{i*phi} to a (2, G, G) amplitude.", "kind": "method", "line": 519, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 531, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 539, "name": "_load", "signature": "def _load(self)"}, {"kind": "method", "line": 558, "name": "_precompute_laplacian", "signature": "def _precompute_laplacian(self)"}, {"kind": "method", "line": 565, "name": "_apply_h", "signature": "def _apply_h(self, field)"}, {"doc": "dpsi/dt = -i H psi  =>  psi' = psi + dt * (-i H psi) = psi + dt*(H_i*r - H_r*i).", "kind": "method", "line": 573, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"kind": "method", "line": 585, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 599, "name": "__init__", "signature": "def __init__(self, config, hamiltonian)"}, {"kind": "method", "line": 606, "name": "_load", "signature": "def _load(self)"}, {"kind": "method", "line": 628, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"kind": "method", "line": 636, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 648, "name": "__init__", "signature": "def __init__(self, config, hamiltonian)"}, {"kind": "method", "line": 657, "name": "_load", "signature": "def _load(self)"}, {"kind": "method", "line": 679, "name": "_precompute_dirac", "signature": "def _precompute_dirac(self)"}, {"kind": "method", "line": 690, "name": "_pack", "signature": "def _pack(self, amp)"}, {"kind": "method", "line": 701, "name": "_unpack", "signature": "def _unpack(self, spinor)"}, {"kind": "method", "line": 707, "name": "_analytical_dirac", "signature": "def _analytical_dirac(self, spinor)"}, {"kind": "method", "line": 722, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"kind": "method", "line": 735, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"doc": "Gate identifier.", "kind": "method", "line": 849, "name": "name", "signature": "def name(self)"}, {"doc": "Apply gate to joint state, return new joint state.", "kind": "method", "line": 853, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 867, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 870, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 882, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 885, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 896, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 899, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 910, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 913, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 924, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 927, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 938, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 941, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 953, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 956, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 969, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 972, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 985, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 988, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 1007, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1010, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 1030, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1033, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 1053, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1056, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 1076, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1079, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 1115, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1118, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 1139, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1142, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 1200, "name": "__init__", "signature": "def __init__(self, n_qubits)"}, {"kind": "method", "line": 1206, "name": "h", "signature": "def h(self, q)"}, {"kind": "method", "line": 1209, "name": "x", "signature": "def x(self, q)"}, {"kind": "method", "line": 1212, "name": "y", "signature": "def y(self, q)"}, {"kind": "method", "line": 1215, "name": "z", "signature": "def z(self, q)"}, {"kind": "method", "line": 1218, "name": "s", "signature": "def s(self, q)"}, {"kind": "method", "line": 1221, "name": "t", "signature": "def t(self, q)"}, {"kind": "method", "line": 1224, "name": "rx", "signature": "def rx(self, q, theta)"}, {"kind": "method", "line": 1227, "name": "ry", "signature": "def ry(self, q, theta)"}, {"kind": "method", "line": 1230, "name": "rz", "signature": "def rz(self, q, theta)"}, {"kind": "method", "line": 1233, "name": "cnot", "signature": "def cnot(self, ctrl, tgt)"}, {"kind": "method", "line": 1236, "name": "cx", "signature": "def cx(self, ctrl, tgt)"}, {"kind": "method", "line": 1239, "name": "cz", "signature": "def cz(self, ctrl, tgt)"}, {"kind": "method", "line": 1242, "name": "swap", "signature": "def swap(self, a, b)"}, {"kind": "method", "line": 1245, "name": "toffoli", "signature": "def toffoli(self, c0, c1, tgt)"}, {"kind": "method", "line": 1248, "name": "ccx", "signature": "def ccx(self, c0, c1, tgt)"}, {"kind": "method", "line": 1251, "name": "evolve", "signature": "def evolve(self, qubits, dt, steps)"}, {"kind": "method", "line": 1254, "name": "barrier", "signature": "def barrier(self)"}, {"kind": "method", "line": 1257, "name": "_append", "signature": "def _append(self, gate_name, targets, params)"}, {"kind": "method", "line": 1265, "name": "depth", "signature": "def depth(self)"}, {"kind": "method", "line": 1268, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 1271, "name": "__repr__", "signature": "def __repr__(self)"}, {"doc": "Alias: marginal P(|1>) per qubit index.", "kind": "method", "line": 1293, "name": "probabilities", "signature": "def probabilities(self)"}, {"doc": "Return the bitstring with the highest probability.", "kind": "method", "line": 1297, "name": "most_probable_bitstring", "signature": "def most_probable_bitstring(self)"}, {"doc": "<Z>_j = P(0) - P(1) in [-1, +1].", "kind": "method", "line": 1301, "name": "expectation_z", "signature": "def expectation_z(self, qubit)"}, {"doc": "Shannon entropy of the full probability distribution in bits.", "kind": "method", "line": 1305, "name": "entropy", "signature": "def entropy(self)"}, {"kind": "method", "line": 1313, "name": "__repr__", "signature": "def __repr__(self)"}, {"kind": "method", "line": 1361, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1375, "name": "_select_backend", "signature": "def _select_backend(self, name)"}, {"kind": "method", "line": 1380, "name": "_state_to_result", "signature": "def _state_to_result(self, state)"}, {"doc": "Execute a quantum circuit on the joint Hilbert space.\n\nArgs:\n    circuit:        The QuantumCircuit to execute.\n    backend:        Physics backend name.\n    initial_states: Optional {qubit_idx: \"0\" or \"1\"}.\n\nReturns:\n    Non-destructive MeasurementResult with full distribution.", "kind": "method", "line": 1388, "name": "run", "signature": "def run(self, circuit, backend, initial_states)"}, {"doc": "Execute circuit with non-destructive probability snapshots.\n\nThe state is never collapsed between snapshots.", "kind": "method", "line": 1420, "name": "run_with_state_snapshots", "signature": "def run_with_state_snapshots(self, circuit, backend, snapshot_after)"}, {"doc": "|Phi+> = (|00> + |11>) / sqrt(2).\n\nExpected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit.", "kind": "method", "line": 1449, "name": "bell_state", "signature": "def bell_state(self, backend)"}, {"doc": "(|00...0> + |11...1>) / sqrt(2).\n\nExpected: P(|00...0>)=P(|11...1>)=0.5, all others 0.", "kind": "method", "line": 1459, "name": "ghz_state", "signature": "def ghz_state(self, n_qubits, backend)"}, {"doc": "QFT on |00...0>. Standard H + controlled-Rz decomposition.", "kind": "method", "line": 1471, "name": "quantum_fourier_transform", "signature": "def quantum_fourier_transform(self, n_qubits, backend)"}, {"doc": "Grover's search algorithm with correct phase oracle and diffusion operator.\n\nThe optimal number of iterations is floor(pi/4 * sqrt(2^n)) which gives\nthe highest probability of measuring the target state.\n\nOracle construction for target |t_0 t_1 ... t_{n-1}>:\n    1. Apply X to every qubit i where t_i == '0'.\n       This maps the target bitstring to |11...1>.\n    2. Apply MCZ on all n qubits.\n       MCZ flips the phase of |11...1> -> exactly the target state\n       (after the X conjugation) gets phase -1.\n    3. Undo the X gates from step 1.\n\nDiffusion operator (inversion about the mean):\n    H^n  X^n  MCZ  X^n  H^n\n\nBoth steps use MCZGate which applies phase -1 to the unique basis state\nwhere all specified qubits are |1>. This is exact for any n.\n\nArgs:\n    n_qubits:        Number of qubits.\n    target_bitstring: Binary string of length n_qubits.\n    backend:         Physics backend name.\n    n_iterations:    Number of Grover iterations. Defaults to\n                     max(1, round(pi/4 * sqrt(2^n_qubits))).\n\nReturns:\n    MeasurementResult. The target bitstring should have the highest\n    probability after the optimal number of iterations.", "kind": "method", "line": 1481, "name": "grover_oracle_search", "signature": "def grover_oracle_search(self, n_qubits, target_bitstring, backend, n_iterations)"}, {"doc": "Hardware-efficient ansatz: Ry layers + CNOT chain. len(thetas)=n_qubits*n_layers.", "kind": "method", "line": 1559, "name": "variational_ansatz", "signature": "def variational_ansatz(self, n_qubits, n_layers, thetas, backend)"}, {"doc": "3-qubit teleportation protocol.\n\nq0 prepared in Ry(pi/3). q2 should match q0's state after corrections.", "kind": "method", "line": 1572, "name": "teleportation", "signature": "def teleportation(self, backend)"}, {"doc": "Deutsch-Jozsa: constant -> all inputs |0>, balanced -> at least one |1>.", "kind": "method", "line": 1585, "name": "deutsch_jozsa", "signature": "def deutsch_jozsa(self, n_input_qubits, is_constant, backend)"}]}, {"doc": "quantum_brutalist_viz.py - Production Quantum State Visualizer ============================================================== High-fidelity 3D holographic visualization of quantum states using trained neural network backends (Hamiltonian, Schrodinger, Dirac).  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_dash.py", "kind": "module", "label": "quantum_dash.py", "language": "py", "sha256": "6fb179a76d62cdd9", "symbol_count": 71, "symbols": [{"kind": "function", "line": 91, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"kind": "class", "line": 106, "name": "ColorScheme", "signature": "class ColorScheme(Enum)"}, {"kind": "class", "line": 115, "name": "BrutalistConfig", "signature": "class BrutalistConfig"}, {"kind": "class", "line": 239, "name": "QuantumSnapshot", "signature": "class QuantumSnapshot"}, {"kind": "class", "line": 255, "name": "BackendComparison", "signature": "class BackendComparison"}, {"kind": "class", "line": 268, "name": "VisualizationOutput", "signature": "class VisualizationOutput"}, {"kind": "class", "line": 275, "name": "IVisualComponent", "signature": "class IVisualComponent(ABC)"}, {"kind": "class", "line": 281, "name": "ProbabilityVisualizer", "signature": "class ProbabilityVisualizer(IVisualComponent)"}, {"kind": "class", "line": 340, "name": "BlochSphereVisualizer", "signature": "class BlochSphereVisualizer(IVisualComponent)"}, {"kind": "class", "line": 447, "name": "PhaseSpaceVisualizer", "signature": "class PhaseSpaceVisualizer(IVisualComponent)"}, {"kind": "class", "line": 509, "name": "EntropyVisualizer", "signature": "class EntropyVisualizer(IVisualComponent)"}, {"kind": "class", "line": 583, "name": "BackendComparisonVisualizer", "signature": "class BackendComparisonVisualizer(IVisualComponent)"}, {"kind": "class", "line": 633, "name": "FidelityVisualizer", "signature": "class FidelityVisualizer(IVisualComponent)"}, {"kind": "class", "line": 674, "name": "QuantumStateAnalyzer", "signature": "class QuantumStateAnalyzer"}, {"kind": "class", "line": 754, "name": "StandardCircuits", "signature": "class StandardCircuits"}, {"kind": "class", "line": 806, "name": "CircuitExecutor", "signature": "class CircuitExecutor"}, {"kind": "class", "line": 873, "name": "FigureBuilder", "signature": "class FigureBuilder"}, {"kind": "class", "line": 958, "name": "QuantumVisualizer", "signature": "class QuantumVisualizer"}, {"kind": "method", "line": 1199, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 168, "name": "colors", "signature": "def colors(self)"}, {"kind": "method", "line": 234, "name": "plotly_template", "signature": "def plotly_template(self)"}, {"kind": "method", "line": 277, "name": "render", "signature": "def render(self, data, axes, config)"}, {"kind": "method", "line": 282, "name": "render", "signature": "def render(self, snapshot, axes, config)"}, {"kind": "method", "line": 318, "name": "_render_empty", "signature": "def _render_empty(self, axes, config)"}, {"kind": "method", "line": 326, "name": "_generate_colors", "signature": "def _generate_colors(self, probs, config)"}, {"kind": "method", "line": 341, "name": "render", "signature": "def render(self, snapshot, axes, config)"}, {"kind": "method", "line": 375, "name": "_render_empty_sphere", "signature": "def _render_empty_sphere(self, axes, config)"}, {"kind": "method", "line": 381, "name": "_draw_sphere_wireframe", "signature": "def _draw_sphere_wireframe(self, axes, config)"}, {"kind": "method", "line": 402, "name": "_draw_axes", "signature": "def _draw_axes(self, axes, config)"}, {"kind": "method", "line": 417, "name": "_draw_bloch_vector", "signature": "def _draw_bloch_vector(self, axes, bx, by, bz, color, qubit_idx, config)"}, {"kind": "method", "line": 431, "name": "_draw_uncertainty_ring", "signature": "def _draw_uncertainty_ring(self, axes, bx, by, bz, color, config)"}, {"kind": "method", "line": 448, "name": "render", "signature": "def render(self, snapshot, axes, config)"}, {"kind": "method", "line": 495, "name": "_render_empty", "signature": "def _render_empty(self, axes, config)"}, {"kind": "method", "line": 510, "name": "render", "signature": "def render(self, snapshots, axes, config)"}, {"kind": "method", "line": 556, "name": "_render_empty", "signature": "def _render_empty(self, axes, config)"}, {"kind": "method", "line": 564, "name": "_interpolate_colors", "signature": "def _interpolate_colors(self, color1, color2, n)"}, {"kind": "method", "line": 578, "name": "_hex_to_rgb", "signature": "def _hex_to_rgb(self, hex_color)"}, {"kind": "method", "line": 584, "name": "render", "signature": "def render(self, comparisons, axes, config)"}, {"kind": "method", "line": 624, "name": "_render_empty", "signature": "def _render_empty(self, axes, config)"}, {"kind": "method", "line": 634, "name": "render", "signature": "def render(self, comparisons, axes, config)"}, {"kind": "method", "line": 665, "name": "_render_empty", "signature": "def _render_empty(self, axes, config)"}, {"kind": "method", "line": 675, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 678, "name": "compute_probabilities", "signature": "def compute_probabilities(self, state)"}, {"kind": "method", "line": 684, "name": "compute_phases", "signature": "def compute_phases(self, state)"}, {"kind": "method", "line": 696, "name": "compute_entropy", "signature": "def compute_entropy(self, probs)"}, {"kind": "method", "line": 704, "name": "compute_bloch_vectors", "signature": "def compute_bloch_vectors(self, state)"}, {"kind": "method", "line": 713, "name": "create_snapshot", "signature": "def create_snapshot(self, state, step, gate_name, backend_name)"}, {"kind": "method", "line": 756, "name": "bell_state", "signature": "def bell_state()"}, {"kind": "method", "line": 760, "name": "ghz_state", "signature": "def ghz_state(n_qubits)"}, {"kind": "method", "line": 767, "name": "qft", "signature": "def qft(n_qubits)"}, {"kind": "method", "line": 782, "name": "grover_oracle", "signature": "def grover_oracle(n_qubits, marked)"}, {"kind": "method", "line": 794, "name": "grover_diffusion", "signature": "def grover_diffusion(n_qubits)"}, {"kind": "method", "line": 807, "name": "__init__", "signature": "def __init__(self, qc, config)"}, {"kind": "method", "line": 812, "name": "execute_sequence", "signature": "def execute_sequence(self, gates, n_qubits, backend_name)"}, {"kind": "method", "line": 835, "name": "compare_backends", "signature": "def compare_backends(self, gates, n_qubits)"}, {"kind": "method", "line": 874, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 883, "name": "build_full_figure", "signature": "def build_full_figure(self, snapshots, comparisons)"}, {"kind": "method", "line": 923, "name": "build_summary_figure", "signature": "def build_summary_figure(self, snapshots, comparisons)"}, {"kind": "method", "line": 959, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 966, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 976, "name": "_init_quantum_computer", "signature": "def _init_quantum_computer(self)"}, {"kind": "method", "line": 1010, "name": "visualize_bell_state", "signature": "def visualize_bell_state(self)"}, {"kind": "method", "line": 1016, "name": "visualize_ghz_state", "signature": "def visualize_ghz_state(self, n_qubits)"}, {"kind": "method", "line": 1022, "name": "visualize_qft", "signature": "def visualize_qft(self, n_qubits)"}, {"kind": "method", "line": 1028, "name": "visualize_grover", "signature": "def visualize_grover(self, n_qubits, marked_state)"}, {"kind": "method", "line": 1043, "name": "_execute_and_visualize", "signature": "def _execute_and_visualize(self, gates, n_qubits, name)"}, {"kind": "method", "line": 1086, "name": "_create_synthetic_snapshots", "signature": "def _create_synthetic_snapshots(self, n_qubits, gates)"}, {"kind": "method", "line": 1129, "name": "_create_synthetic_comparisons", "signature": "def _create_synthetic_comparisons(self, n_qubits, gates)"}, {"kind": "method", "line": 1153, "name": "_save_data", "signature": "def _save_data(self, path, snapshots, comparisons)"}, {"kind": "method", "line": 1174, "name": "run_all", "signature": "def run_all(self)"}, {"kind": "method", "line": 1189, "name": "_print_summary", "signature": "def _print_summary(self, results)"}]}, {"doc": "Quantum Framework Core Module ============================= Unified core module for quantum simulation using MPS-based Hilbert space representation. Achieves sub-exponential memory scaling O(n * chi^2) instead of O(2^n) for full statevector representation.  This module consolidates: - Configuration loading from TOML - MPS (Matrix Product State) tensor network representation - Physics backends (Hamiltonian, Schrodinger, Dirac) - Quantum gate registry with MPS-compatible operations - Quantum circuit builder and executor  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_framework_core.py", "kind": "module", "label": "quantum_framework_core.py", "language": "py", "sha256": "801a97f4a0285964", "symbol_count": 196, "symbols": [{"doc": "Create a configured logger instance.", "kind": "function", "line": 46, "name": "_make_logger", "signature": "def _make_logger(name, level)"}, {"doc": "Phase classification for Hilbert space compression.", "kind": "class", "line": 62, "name": "HilbertPhase", "signature": "class HilbertPhase(Enum)"}, {"doc": "Unified configuration for the quantum simulation framework.\nLoads from TOML file with fallback to sensible defaults.", "kind": "class", "line": 72, "name": "FrameworkConfig", "signature": "class FrameworkConfig"}, {"doc": "Atomic data structure.", "kind": "class", "line": 218, "name": "AtomData", "signature": "class AtomData"}, {"doc": "Molecular data structure.", "kind": "class", "line": 230, "name": "MoleculeData", "signature": "class MoleculeData"}, {"doc": "Atomic orbital data structure.", "kind": "class", "line": 250, "name": "OrbitalData", "signature": "class OrbitalData"}, {"doc": "Configuration loader that parses TOML files and provides\naccess to atoms, molecules, and orbitals data.", "kind": "class", "line": 259, "name": "ConfigLoader", "signature": "class ConfigLoader"}, {"doc": "Abstract interface for tensor network quantum states.", "kind": "class", "line": 435, "name": "ITensorNetwork", "signature": "class ITensorNetwork(ABC)"}, {"doc": "Matrix Product State core tensor A^{[k]}_{i_k} with bond indices.\n\nShape: (chi_left, d, chi_right) where d=2 for qubits.\nMemory per core: O(chi^2 * d) = O(chi^2)", "kind": "class", "line": 480, "name": "MPSCore", "signature": "class MPSCore"}, {"doc": "Matrix Product State representation of n-qubit quantum state.\n\n|psi> = sum_{i_1...i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} ... A^{[n]}_{i_n} |i_1...i_n>\n\nMemory: O(n * chi^2 * d) vs O(d^n) for full statevector.\n\nExample scaling:\n    n=30, chi=16: ~30KB vs 8GB for statevector\n    n=33, chi=16: ~33KB vs 64GB for statevector", "kind": "class", "line": 554, "name": "MPSState", "signature": "class MPSState(ITensorNetwork)"}, {"doc": "Vacuum Core architecture for topological protection.\n\nProjects irrelevant Hilbert subspace to zero, achieving\nhigh sparsity (target 99.99%) while preserving quantum information.", "kind": "class", "line": 916, "name": "VacuumCore", "signature": "class VacuumCore"}, {"doc": "Provides topological protection for quantum states.\n\nMonitors:\n    - Winding numbers\n    - Berry phases\n    - Edge state preservation", "kind": "class", "line": 988, "name": "TopologicalProtector", "signature": "class TopologicalProtector"}, {"doc": "Spectral convolution layer in frequency domain.", "kind": "class", "line": 1040, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"doc": "Hamiltonian backbone network for spectral operations.", "kind": "class", "line": 1076, "name": "HamiltonianBackboneNet", "signature": "class HamiltonianBackboneNet(Module)"}, {"doc": "Schrodinger network for wavefunction evolution.", "kind": "class", "line": 1102, "name": "SchrodingerSpectralNet", "signature": "class SchrodingerSpectralNet(Module)"}, {"doc": "Dirac network for relativistic spinor evolution.", "kind": "class", "line": 1133, "name": "DiracSpectralNet", "signature": "class DiracSpectralNet(Module)"}, {"doc": "Dirac gamma matrices in standard or Weyl representation.", "kind": "class", "line": 1164, "name": "GammaMatrices", "signature": "class GammaMatrices"}, {"doc": "Abstract interface for physics backends.", "kind": "class", "line": 1215, "name": "IPhysicsBackend", "signature": "class IPhysicsBackend(ABC)"}, {"doc": "Hamiltonian backend using neural network for spectral operations.\n\nPerforms first-order Schrodinger time evolution:\n    psi(t+dt) = psi(t) - i*dt*H*psi(t)", "kind": "class", "line": 1229, "name": "HamiltonianBackend", "signature": "class HamiltonianBackend(IPhysicsBackend)"}, {"doc": "Schrodinger backend using learned 2-channel spectral network.\n\nFalls back to HamiltonianBackend if checkpoint unavailable.", "kind": "class", "line": 1305, "name": "SchrodingerBackend", "signature": "class SchrodingerBackend(IPhysicsBackend)"}, {"doc": "Dirac backend for relativistic spinor evolution.\n\nExpands (2,G,G) amplitude to 4-component spinor, propagates,\nthen projects back.", "kind": "class", "line": 1358, "name": "DiracBackend", "signature": "class DiracBackend(IPhysicsBackend)"}, {"doc": "Abstract interface for quantum gates.", "kind": "class", "line": 1476, "name": "IQuantumGate", "signature": "class IQuantumGate(ABC)"}, {"doc": "Hadamard gate: H = [[1,1],[1,-1]] / sqrt(2).", "kind": "class", "line": 1496, "name": "HadamardGate", "signature": "class HadamardGate(IQuantumGate)"}, {"doc": "Pauli-X gate: X = [[0,1],[1,0]].", "kind": "class", "line": 1511, "name": "PauliXGate", "signature": "class PauliXGate(IQuantumGate)"}, {"doc": "Pauli-Y gate: Y = [[0,-i],[i,0]].", "kind": "class", "line": 1525, "name": "PauliYGate", "signature": "class PauliYGate(IQuantumGate)"}, {"doc": "Pauli-Z gate: Z = [[1,0],[0,-1]].", "kind": "class", "line": 1539, "name": "PauliZGate", "signature": "class PauliZGate(IQuantumGate)"}, {"doc": "S gate: S = [[1,0],[0,i]].", "kind": "class", "line": 1553, "name": "SGate", "signature": "class SGate(IQuantumGate)"}, {"doc": "T gate: T = [[1,0],[0,e^{i*pi/4}]].", "kind": "class", "line": 1567, "name": "TGate", "signature": "class TGate(IQuantumGate)"}, {"doc": "Rotation-X gate: Rx(theta) = exp(-i*theta/2 * X).", "kind": "class", "line": 1582, "name": "RxGate", "signature": "class RxGate(IQuantumGate)"}, {"doc": "Rotation-Y gate: Ry(theta) = exp(-i*theta/2 * Y).", "kind": "class", "line": 1598, "name": "RyGate", "signature": "class RyGate(IQuantumGate)"}, {"doc": "Rotation-Z gate: Rz(theta) = exp(-i*theta/2 * Z).", "kind": "class", "line": 1614, "name": "RzGate", "signature": "class RzGate(IQuantumGate)"}, {"doc": "Controlled-Rz gate: applies Rz to target if control is |1>.", "kind": "class", "line": 1631, "name": "CRzGate", "signature": "class CRzGate(IQuantumGate)"}, {"doc": "CNOT gate: flips target if control is |1>.", "kind": "class", "line": 1656, "name": "CNOTGate", "signature": "class CNOTGate(IQuantumGate)"}, {"doc": "Controlled-Z gate: applies phase -1 to |11>.", "kind": "class", "line": 1676, "name": "CZGate", "signature": "class CZGate(IQuantumGate)"}, {"doc": "SWAP gate: exchanges two qubits.", "kind": "class", "line": 1696, "name": "SWAPGate", "signature": "class SWAPGate(IQuantumGate)"}, {"doc": "Single instruction in a quantum circuit.", "kind": "class", "line": 1734, "name": "CircuitInstruction", "signature": "class CircuitInstruction"}, {"doc": "Quantum circuit builder for MPS states.", "kind": "class", "line": 1741, "name": "QuantumCircuit", "signature": "class QuantumCircuit"}, {"doc": "Main quantum computer using MPS representation.\n\nProvides:\n    - State preparation\n    - Circuit execution\n    - Backend selection\n    - Memory-efficient simulation for up to 33+ qubits", "kind": "class", "line": 1816, "name": "MPSQuantumComputer", "signature": "class MPSQuantumComputer"}, {"doc": "Run scaling benchmark to demonstrate MPS memory efficiency.\n\nReturns:\n    Dictionary with qubit counts, memory usage, compression ratios.", "kind": "method", "line": 2045, "name": "run_scaling_benchmark", "signature": "def run_scaling_benchmark(config, max_qubits)"}, {"doc": "Run Grover's search algorithm and return results.\n\nImplements the algorithm at the statevector level for correctness\n(exact oracle via direct phase flip), then reads off probabilities.\n\nArgs:\n    qc: MPSQuantumComputer instance (unused for computation, kept for API compatibility)\n    n_qubits: number of qubits\n    marked_states: list of integers representing marked computational basis states\n\nReturns:\n    dict with keys: probability, marked_states (as bit strings), speedup, iterations", "kind": "method", "line": 2098, "name": "run_grover_search", "signature": "def run_grover_search(qc, n_qubits, marked_states)"}, {"doc": "Initialize random seeds after configuration.", "kind": "method", "line": 139, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "Load configuration from TOML file.", "kind": "method", "line": 147, "name": "from_toml", "signature": "def from_toml(cls, toml_path)"}, {"kind": "method", "line": 265, "name": "__init__", "signature": "def __init__(self, config_path)"}, {"doc": "Find configuration file in standard locations.", "kind": "method", "line": 274, "name": "_find_config", "signature": "def _find_config(self)"}, {"doc": "Load configuration from TOML file.", "kind": "method", "line": 286, "name": "_load", "signature": "def _load(self)"}, {"doc": "Load default configuration values.", "kind": "method", "line": 300, "name": "_load_defaults", "signature": "def _load_defaults(self)"}, {"doc": "Parse atoms from configuration data.", "kind": "method", "line": 318, "name": "_parse_atoms", "signature": "def _parse_atoms(self)"}, {"doc": "Parse molecules from configuration data.", "kind": "method", "line": 332, "name": "_parse_molecules", "signature": "def _parse_molecules(self)"}, {"doc": "Parse orbitals from configuration data.", "kind": "method", "line": 352, "name": "_parse_orbitals", "signature": "def _parse_orbitals(self)"}, {"doc": "Parse experiments from configuration data.", "kind": "method", "line": 364, "name": "_parse_experiments", "signature": "def _parse_experiments(self)"}, {"doc": "Get atom data by symbol (case-insensitive).", "kind": "method", "line": 375, "name": "get_atom", "signature": "def get_atom(self, symbol)"}, {"doc": "Get molecule data by name (case-insensitive).", "kind": "method", "line": 384, "name": "get_molecule", "signature": "def get_molecule(self, name)"}, {"doc": "Get orbital data by name (case-insensitive).", "kind": "method", "line": 393, "name": "get_orbital", "signature": "def get_orbital(self, name)"}, {"doc": "Get experiment data by name.", "kind": "method", "line": 402, "name": "get_experiment", "signature": "def get_experiment(self, name)"}, {"doc": "Return all atoms.", "kind": "method", "line": 407, "name": "atoms", "signature": "def atoms(self)"}, {"doc": "Return all molecules.", "kind": "method", "line": 412, "name": "molecules", "signature": "def molecules(self)"}, {"doc": "Return all orbitals.", "kind": "method", "line": 417, "name": "orbitals", "signature": "def orbitals(self)"}, {"doc": "Return all experiments.", "kind": "method", "line": 422, "name": "experiments", "signature": "def experiments(self)"}, {"doc": "Get molecules that fit within qubit budget.", "kind": "method", "line": 426, "name": "get_molecules_by_qubits", "signature": "def get_molecules_by_qubits(self, max_qubits)"}, {"doc": "Get atoms that fit within qubit budget.", "kind": "method", "line": 430, "name": "get_atoms_by_qubits", "signature": "def get_atoms_by_qubits(self, max_qubits)"}, {"doc": "Return number of qubits.", "kind": "method", "line": 440, "name": "n_qubits", "signature": "def n_qubits(self)"}, {"doc": "Compute amplitude for a computational basis state.", "kind": "method", "line": 445, "name": "amplitude", "signature": "def amplitude(self, basis_index)"}, {"doc": "Apply single-qubit gate in-place.", "kind": "method", "line": 450, "name": "apply_single_qubit_gate", "signature": "def apply_single_qubit_gate(self, qubit, gate)"}, {"doc": "Apply two-qubit gate in-place.", "kind": "method", "line": 455, "name": "apply_two_qubit_gate", "signature": "def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)"}, {"doc": "Compute state norm.", "kind": "method", "line": 460, "name": "norm", "signature": "def norm(self)"}, {"doc": "Compute measurement probabilities.", "kind": "method", "line": 465, "name": "probabilities", "signature": "def probabilities(self)"}, {"doc": "Compute von Neumann entropy.", "kind": "method", "line": 470, "name": "entropy", "signature": "def entropy(self)"}, {"doc": "Return memory usage in bytes.", "kind": "method", "line": 475, "name": "memory_bytes", "signature": "def memory_bytes(self)"}, {"kind": "method", "line": 488, "name": "__init__", "signature": "def __init__(self, chi_left, chi_right, d, device, dtype)"}, {"doc": "Initialize core tensor for |0> product state (exact).", "kind": "method", "line": 504, "name": "_initialize", "signature": "def _initialize(self)"}, {"doc": "Return the core tensor.", "kind": "method", "line": 517, "name": "tensor", "signature": "def tensor(self)"}, {"doc": "Set the core tensor, preserving complex dtype when needed.", "kind": "method", "line": 524, "name": "tensor", "signature": "def tensor(self, value)"}, {"doc": "Bring core to left-canonical form, return singular values.", "kind": "method", "line": 533, "name": "left_canonicalize", "signature": "def left_canonicalize(self)"}, {"doc": "Bring core to right-canonical form, return singular values.", "kind": "method", "line": 543, "name": "right_canonicalize", "signature": "def right_canonicalize(self)"}, {"kind": "method", "line": 567, "name": "__init__", "signature": "def __init__(self, n_qubits, config)"}, {"doc": "Initialize MPS with product state |00...0>.", "kind": "method", "line": 576, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 600, "name": "n_qubits", "signature": "def n_qubits(self)"}, {"doc": "Compute bond dimension at given site.", "kind": "method", "line": 603, "name": "_bond_dimension", "signature": "def _bond_dimension(self, site)"}, {"doc": "Compute amplitude for computational basis state.", "kind": "method", "line": 610, "name": "amplitude", "signature": "def amplitude(self, basis_index)"}, {"doc": "Apply single-qubit gate in-place.\n\nWorks in complex128 so that gates with imaginary entries (Y, S, T, Rz…)\nare handled correctly.  The core tensor is promoted to complex128 when\nthe result has a non-negligible imaginary component; otherwise it is\nkept as the config dtype (typically float64).", "kind": "method", "line": 631, "name": "apply_single_qubit_gate", "signature": "def apply_single_qubit_gate(self, qubit, gate)"}, {"doc": "Apply two-qubit gate in-place.", "kind": "method", "line": 655, "name": "apply_two_qubit_gate", "signature": "def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)"}, {"doc": "Swap qubit ordering in two-qubit gate.", "kind": "method", "line": 674, "name": "_swap_qubits_in_gate", "signature": "def _swap_qubits_in_gate(self, gate)"}, {"doc": "Apply gate to adjacent qubit pair.", "kind": "method", "line": 684, "name": "_apply_adjacent_gate", "signature": "def _apply_adjacent_gate(self, qubit, gate)"}, {"doc": "Apply gate to non-adjacent qubit pair using SWAP network.", "kind": "method", "line": 747, "name": "_apply_nonadjacent_gate", "signature": "def _apply_nonadjacent_gate(self, qubit_a, qubit_b, gate)"}, {"doc": "Compute state norm.", "kind": "method", "line": 762, "name": "norm", "signature": "def norm(self)"}, {"doc": "Bring MPS to canonical form.", "kind": "method", "line": 770, "name": "_canonicalize", "signature": "def _canonicalize(self)"}, {"doc": "Compute measurement probabilities.", "kind": "method", "line": 778, "name": "probabilities", "signature": "def probabilities(self)"}, {"doc": "Compute maximum entanglement entropy across all cuts.\n\nFor n=1 qubits, returns Shannon entropy of the probability distribution.\nFor n>1, returns the maximum entanglement entropy across all bipartite cuts.", "kind": "method", "line": 795, "name": "entropy", "signature": "def entropy(self)"}, {"doc": "Return memory usage in bytes.", "kind": "method", "line": 822, "name": "memory_bytes", "signature": "def memory_bytes(self)"}, {"doc": "Compute entanglement entropy at given cut between qubits cut-1 and cut.\nUses Schmidt decomposition from the MPS bond.", "kind": "method", "line": 829, "name": "entanglement_entropy", "signature": "def entanglement_entropy(self, cut)"}, {"doc": "Convert MPS to full statevector (only for small systems).", "kind": "method", "line": 874, "name": "to_statevector", "signature": "def to_statevector(self)"}, {"doc": "Return most probable basis state as bitstring.", "kind": "method", "line": 890, "name": "most_probable_bitstring", "signature": "def most_probable_bitstring(self)"}, {"doc": "Return a deep copy.", "kind": "method", "line": 896, "name": "clone", "signature": "def clone(self)"}, {"kind": "method", "line": 924, "name": "__init__", "signature": "def __init__(self, n_qubits, config)"}, {"doc": "Initialize vacuum core with ground state.", "kind": "method", "line": 935, "name": "_initialize", "signature": "def _initialize(self)"}, {"doc": "Compute Berry phases between active states.", "kind": "method", "line": 941, "name": "_compute_berry_phases", "signature": "def _compute_berry_phases(self)"}, {"doc": "Add a basis state to the active subspace.", "kind": "method", "line": 948, "name": "add_active_state", "signature": "def add_active_state(self, basis_index, winding_number)"}, {"doc": "Compute winding number for a basis state.", "kind": "method", "line": 962, "name": "_compute_winding_number", "signature": "def _compute_winding_number(self, basis_index)"}, {"doc": "Check if state is topologically protected.", "kind": "method", "line": 968, "name": "is_topologically_protected", "signature": "def is_topologically_protected(self, basis_index)"}, {"doc": "Compute vacuum sparsity.", "kind": "method", "line": 973, "name": "sparsity", "signature": "def sparsity(self)"}, {"doc": "Project state onto active subspace.", "kind": "method", "line": 979, "name": "project_to_active", "signature": "def project_to_active(self, state)"}, {"kind": "method", "line": 998, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Compute winding number for a qubit.", "kind": "method", "line": 1003, "name": "compute_winding_number", "signature": "def compute_winding_number(self, state, qubit)"}, {"doc": "Compute Berry phase between two qubits.", "kind": "method", "line": 1015, "name": "compute_berry_phase", "signature": "def compute_berry_phase(self, state, qubit_a, qubit_b)"}, {"doc": "Check if state is topologically protected.", "kind": "method", "line": 1031, "name": "is_protected", "signature": "def is_protected(self, state, vacuum_core)"}, {"kind": "method", "line": 1043, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"doc": "Apply spectral convolution via RFFT2.", "kind": "method", "line": 1053, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 1079, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers)"}, {"doc": "Apply Hamiltonian backbone network.", "kind": "method", "line": 1088, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 1105, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)"}, {"doc": "Apply Schrodinger evolution network.", "kind": "method", "line": 1119, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 1136, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)"}, {"doc": "Apply Dirac evolution network.", "kind": "method", "line": 1150, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 1167, "name": "__init__", "signature": "def __init__(self, representation, device)"}, {"doc": "Initialize gamma matrices.", "kind": "method", "line": 1172, "name": "_init_matrices", "signature": "def _init_matrices(self)"}, {"doc": "Move all matrices to device.", "kind": "method", "line": 1208, "name": "to", "signature": "def to(self, device)"}, {"doc": "Evolve a single amplitude by time dt.", "kind": "method", "line": 1219, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"doc": "Apply global phase to amplitude.", "kind": "method", "line": 1224, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 1237, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Load model from checkpoint.", "kind": "method", "line": 1245, "name": "_load", "signature": "def _load(self)"}, {"doc": "Precompute Laplacian kernel for kinetic energy.", "kind": "method", "line": 1266, "name": "_precompute_laplacian", "signature": "def _precompute_laplacian(self)"}, {"doc": "Apply Hamiltonian operator to field.", "kind": "method", "line": 1274, "name": "_apply_h", "signature": "def _apply_h(self, field)"}, {"doc": "Evolve amplitude by time dt.", "kind": "method", "line": 1284, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"doc": "Apply global phase.", "kind": "method", "line": 1298, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 1312, "name": "__init__", "signature": "def __init__(self, config, hamiltonian)"}, {"doc": "Load model from checkpoint.", "kind": "method", "line": 1319, "name": "_load", "signature": "def _load(self)"}, {"doc": "Evolve amplitude by time dt.", "kind": "method", "line": 1342, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"doc": "Apply global phase.", "kind": "method", "line": 1353, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 1366, "name": "__init__", "signature": "def __init__(self, config, hamiltonian)"}, {"doc": "Load model from checkpoint.", "kind": "method", "line": 1375, "name": "_load", "signature": "def _load(self)"}, {"doc": "Precompute momentum grids for Dirac operator.", "kind": "method", "line": 1398, "name": "_precompute_dirac", "signature": "def _precompute_dirac(self)"}, {"doc": "Pack 2-channel amplitude to 4-component spinor.", "kind": "method", "line": 1407, "name": "_pack", "signature": "def _pack(self, amp)"}, {"doc": "Unpack 4-component spinor to 2-channel amplitude.", "kind": "method", "line": 1421, "name": "_unpack", "signature": "def _unpack(self, spinor)"}, {"doc": "Apply analytical Dirac Hamiltonian to spinor.", "kind": "method", "line": 1428, "name": "_analytical_dirac", "signature": "def _analytical_dirac(self, spinor)"}, {"doc": "Evolve amplitude by time dt using Dirac equation.", "kind": "method", "line": 1447, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"doc": "Apply global phase.", "kind": "method", "line": 1463, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"doc": "Evolve full 4-component spinor by time dt.", "kind": "method", "line": 1467, "name": "evolve_spinor", "signature": "def evolve_spinor(self, spinor, dt)"}, {"doc": "Return gate name.", "kind": "method", "line": 1481, "name": "name", "signature": "def name(self)"}, {"doc": "Apply gate to state and return new state.", "kind": "method", "line": 1486, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1500, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1503, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1515, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1518, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1529, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1532, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1543, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1546, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1557, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1560, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1571, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1574, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1586, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1589, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1602, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1605, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1618, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1621, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1635, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1638, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1660, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1663, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1680, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1683, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1700, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 1703, "name": "apply", "signature": "def apply(self, state, targets, params)"}, {"kind": "method", "line": 1744, "name": "__init__", "signature": "def __init__(self, n_qubits)"}, {"doc": "Append an instruction to the circuit.", "kind": "method", "line": 1748, "name": "_append", "signature": "def _append(self, gate_name, targets, params)"}, {"kind": "method", "line": 1759, "name": "h", "signature": "def h(self, qubit)"}, {"kind": "method", "line": 1762, "name": "x", "signature": "def x(self, qubit)"}, {"kind": "method", "line": 1765, "name": "y", "signature": "def y(self, qubit)"}, {"kind": "method", "line": 1768, "name": "z", "signature": "def z(self, qubit)"}, {"kind": "method", "line": 1771, "name": "s", "signature": "def s(self, qubit)"}, {"kind": "method", "line": 1774, "name": "t", "signature": "def t(self, qubit)"}, {"kind": "method", "line": 1777, "name": "rx", "signature": "def rx(self, qubit, theta)"}, {"kind": "method", "line": 1780, "name": "ry", "signature": "def ry(self, qubit, theta)"}, {"kind": "method", "line": 1783, "name": "rz", "signature": "def rz(self, qubit, theta)"}, {"kind": "method", "line": 1786, "name": "crz", "signature": "def crz(self, control, target, theta)"}, {"kind": "method", "line": 1789, "name": "cnot", "signature": "def cnot(self, control, target)"}, {"kind": "method", "line": 1792, "name": "cz", "signature": "def cz(self, control, target)"}, {"kind": "method", "line": 1795, "name": "swap", "signature": "def swap(self, qubit1, qubit2)"}, {"doc": "Return number of instructions.", "kind": "method", "line": 1798, "name": "__len__", "signature": "def __len__(self)"}, {"doc": "True if circuit has instructions.", "kind": "method", "line": 1802, "name": "__bool__", "signature": "def __bool__(self)"}, {"doc": "Execute circuit on state.", "kind": "method", "line": 1806, "name": "run", "signature": "def run(self, state)"}, {"kind": "method", "line": 1827, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Create a new quantum circuit.", "kind": "method", "line": 1843, "name": "create_circuit", "signature": "def create_circuit(self, n_qubits)"}, {"doc": "Create initial state |00...0>.", "kind": "method", "line": 1851, "name": "create_state", "signature": "def create_state(self, n_qubits)"}, {"doc": "Prepare Bell state |Phi+> = (|00> + |11>) / sqrt(2).", "kind": "method", "line": 1865, "name": "bell_state", "signature": "def bell_state(self, n_qubits)"}, {"doc": "Prepare GHZ state (|00...0> + |11...1>) / sqrt(2).", "kind": "method", "line": 1873, "name": "ghz_state", "signature": "def ghz_state(self, n_qubits)"}, {"doc": "Prepare W state: |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n\n\nUses direct statevector-to-MPS conversion via successive SVD.\nThis guarantees exact representation (up to numerical precision).\n\nThe W state has a unique entanglement structure:\n- The entanglement entropy for a cut separating k qubits from (n-k) is:\n  S(k) = H({k/n, (n-k)/n}) where H is the binary entropy function\n- Maximum entropy is 1 bit when n is even and k = n/2\n- For W₃: S_max ≈ 0.9183 bits (H({1/3, 2/3}))\n- This is DIFFERENT from GHZ which has entropy = 1 bit for any cut\n\nReturns:\n    MPSState: The W state in MPS representation", "kind": "method", "line": 1882, "name": "w_state", "signature": "def w_state(self, n_qubits)"}, {"doc": "Build W state using direct statevector-to-MPS conversion via successive SVD.\n\nThis guarantees exact representation (up to numerical precision).", "kind": "method", "line": 1905, "name": "_build_w_state_direct", "signature": "def _build_w_state_direct(self, n_qubits, max_bond)"}, {"doc": "Execute circuit on state.", "kind": "method", "line": 1985, "name": "run_circuit", "signature": "def run_circuit(self, circuit, initial_state)"}, {"doc": "Get physics backend by name.", "kind": "method", "line": 1995, "name": "get_backend", "signature": "def get_backend(self, name)"}, {"doc": "Compute memory usage for a state.", "kind": "method", "line": 1999, "name": "memory_usage", "signature": "def memory_usage(self, state)"}, {"doc": "Compute compression ratio vs full statevector.", "kind": "method", "line": 2012, "name": "compression_ratio", "signature": "def compression_ratio(self, state)"}, {"doc": "Detect Hilbert space phase from state properties.", "kind": "method", "line": 2019, "name": "detect_phase", "signature": "def detect_phase(self, state)"}, {"doc": "Compute average bond dimension across MPS cores.", "kind": "method", "line": 2037, "name": "_compute_average_bond_dimension", "signature": "def _compute_average_bond_dimension(self, state)"}]}, {"doc": "Quantum Simulation Framework - Main Entry Point ================================================ Main entry point for the quantum simulation framework with command-line interface and interactive menu support.  Usage: python quantum_framework_main.py [OPTIONS]  Options: --config PATH       Path to TOML configuration file --learn             Launch the educational Quantum Lab TUI (EN/ES) --interactive       Launch interactive menu mode --all               Run ALL experiments automatically (for debugging) --experiment NAME   Run specific experiment by name --benchmark         Run scaling benchmark --max-qubits N      Maximum qubits for benchmark --molecule NAME     Specify molecule for experiments --orbital NAME      Specify orbital for visualization --qubits N          Number of qubits for experiments --output DIR        Output directory --verbose           Enable verbose logging --help              Show help message  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_framework_main.py", "kind": "module", "label": "quantum_framework_main.py", "language": "py", "sha256": "4b5fbbbac4dc043d", "symbol_count": 7, "symbols": [{"doc": "Configure logging level based on verbosity.", "kind": "function", "line": 52, "name": "setup_logging", "signature": "def setup_logging(verbose)"}, {"doc": "Run scaling benchmark.", "kind": "function", "line": 61, "name": "run_benchmark", "signature": "def run_benchmark(args, config)"}, {"doc": "Run a specific experiment by name.", "kind": "function", "line": 101, "name": "run_experiment", "signature": "def run_experiment(args, config, config_loader)"}, {"doc": "Run molecular simulation.", "kind": "function", "line": 173, "name": "run_molecular_simulation", "signature": "def run_molecular_simulation(args, config, config_loader)"}, {"doc": "Run orbital visualization.", "kind": "function", "line": 210, "name": "run_orbital_visualization", "signature": "def run_orbital_visualization(args, config, config_loader)"}, {"doc": "Print framework information.", "kind": "function", "line": 242, "name": "print_info", "signature": "def print_info(config, config_loader)"}, {"doc": "Main entry point.", "kind": "function", "line": 292, "name": "main", "signature": "def main()"}]}, {"doc": "Quantum Framework Interactive Menu System ========================================= Interactive menu system for accessing all framework capabilities including quantum circuits, molecular simulations, orbital visualization, and advanced physics experiments.  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_framework_menu.py", "kind": "module", "label": "quantum_framework_menu.py", "language": "py", "sha256": "84285f5898e861b8", "symbol_count": 78, "symbols": [{"doc": "Interactive menu system for the quantum simulation framework.\n\nProvides structured access to all framework capabilities through\na hierarchical menu system with real-time feedback.", "kind": "class", "line": 64, "name": "MenuSystem", "signature": "class MenuSystem"}, {"doc": "Run the interactive menu system.", "kind": "method", "line": 2357, "name": "run_interactive_menu", "signature": "def run_interactive_menu(config, config_loader)"}, {"doc": "Run ALL experiments automatically for debugging.\nThis function executes all available experiments without user interaction.\nUses CORRECT physics formulas verified against experimental data.", "kind": "method", "line": 2363, "name": "run_all_experiments", "signature": "def run_all_experiments(config, config_loader)"}, {"kind": "method", "line": 72, "name": "__init__", "signature": "def __init__(self, config, config_loader)"}, {"doc": "Clear the terminal screen.", "kind": "method", "line": 79, "name": "clear_screen", "signature": "def clear_screen(self)"}, {"doc": "Print formatted header.", "kind": "method", "line": 83, "name": "print_header", "signature": "def print_header(self, title)"}, {"doc": "Print formatted menu with options.", "kind": "method", "line": 90, "name": "print_menu", "signature": "def print_menu(self, title, options)"}, {"doc": "Get user input with history tracking.", "kind": "method", "line": 97, "name": "get_input", "signature": "def get_input(self, prompt)"}, {"doc": "Wait for user to press Enter.", "kind": "method", "line": 106, "name": "pause", "signature": "def pause(self, message)"}, {"doc": "Run the main menu loop.", "kind": "method", "line": 113, "name": "run", "signature": "def run(self)"}, {"doc": "Display main menu.", "kind": "method", "line": 118, "name": "_show_main_menu", "signature": "def _show_main_menu(self)"}, {"doc": "Display quantum circuits menu.", "kind": "method", "line": 163, "name": "_show_circuit_menu", "signature": "def _show_circuit_menu(self)"}, {"doc": "Create and run a custom circuit.", "kind": "method", "line": 197, "name": "_custom_circuit", "signature": "def _custom_circuit(self)"}, {"doc": "Demonstrate Bell state preparation.", "kind": "method", "line": 279, "name": "_bell_state_demo", "signature": "def _bell_state_demo(self)"}, {"doc": "Demonstrate GHZ state preparation.", "kind": "method", "line": 299, "name": "_ghz_state_demo", "signature": "def _ghz_state_demo(self)"}, {"doc": "Demonstrate W state preparation.", "kind": "method", "line": 328, "name": "_w_state_demo", "signature": "def _w_state_demo(self)"}, {"doc": "Demonstrate single qubit gates.", "kind": "method", "line": 356, "name": "_single_qubit_gates_demo", "signature": "def _single_qubit_gates_demo(self)"}, {"doc": "Demonstrate two qubit gates.", "kind": "method", "line": 397, "name": "_two_qubit_gates_demo", "signature": "def _two_qubit_gates_demo(self)"}, {"doc": "Display entanglement experiments menu.", "kind": "method", "line": 431, "name": "_show_entanglement_menu", "signature": "def _show_entanglement_menu(self)"}, {"doc": "Measure entropy of Bell states.", "kind": "method", "line": 459, "name": "_bell_entropy_experiment", "signature": "def _bell_entropy_experiment(self)"}, {"doc": "Study GHZ entanglement scaling with qubit count.", "kind": "method", "line": 472, "name": "_ghz_scaling_experiment", "signature": "def _ghz_scaling_experiment(self)"}, {"doc": "Measure entanglement entropy at different cuts.", "kind": "method", "line": 497, "name": "_entropy_by_cut_experiment", "signature": "def _entropy_by_cut_experiment(self)"}, {"doc": "Generate entropy heatmap for different states.", "kind": "method", "line": 522, "name": "_entropy_heatmap_experiment", "signature": "def _entropy_heatmap_experiment(self)"}, {"doc": "Display molecular simulations menu.", "kind": "method", "line": 583, "name": "_show_molecular_menu", "signature": "def _show_molecular_menu(self)"}, {"doc": "List all available molecules.", "kind": "method", "line": 617, "name": "_list_molecules", "signature": "def _list_molecules(self)"}, {"doc": "Show detailed molecule information.", "kind": "method", "line": 636, "name": "_molecule_info", "signature": "def _molecule_info(self)"}, {"doc": "Run VQE for molecular ground state.", "kind": "method", "line": 667, "name": "_vqe_ground_state", "signature": "def _vqe_ground_state(self)"}, {"doc": "Plot molecular energy landscape.", "kind": "method", "line": 751, "name": "_energy_landscape", "signature": "def _energy_landscape(self)"}, {"doc": "Simulate bond dissociation curve.", "kind": "method", "line": 808, "name": "_bond_dissociation", "signature": "def _bond_dissociation(self)"}, {"doc": "Display orbital visualization menu.", "kind": "method", "line": 868, "name": "_show_orbital_menu", "signature": "def _show_orbital_menu(self)"}, {"doc": "List all available orbitals.", "kind": "method", "line": 899, "name": "_list_orbitals", "signature": "def _list_orbitals(self)"}, {"doc": "Visualize a single orbital.", "kind": "method", "line": 915, "name": "_visualize_orbital", "signature": "def _visualize_orbital(self)"}, {"doc": "Generate orbital visualization plot.", "kind": "method", "line": 947, "name": "_generate_orbital_plot", "signature": "def _generate_orbital_plot(self, orb, num_samples)"}, {"doc": "Compare multiple orbitals.", "kind": "method", "line": 1064, "name": "_compare_orbitals", "signature": "def _compare_orbitals(self)"}, {"doc": "Plot radial wavefunction.", "kind": "method", "line": 1146, "name": "_radial_wavefunction", "signature": "def _radial_wavefunction(self)"}, {"doc": "Plot angular wavefunction.", "kind": "method", "line": 1202, "name": "_angular_wavefunction", "signature": "def _angular_wavefunction(self)"}, {"doc": "Display relativistic physics menu.", "kind": "method", "line": 1264, "name": "_show_relativistic_menu", "signature": "def _show_relativistic_menu(self)"}, {"doc": "Calculate Dirac energy levels.", "kind": "method", "line": 1292, "name": "_dirac_energy_levels", "signature": "def _dirac_energy_levels(self)"}, {"doc": "Calculate Dirac energy level.", "kind": "method", "line": 1321, "name": "_dirac_energy", "signature": "def _dirac_energy(self, n, kappa, alpha, c)"}, {"doc": "Calculate fine structure corrections.", "kind": "method", "line": 1329, "name": "_fine_structure", "signature": "def _fine_structure(self)"}, {"doc": "Simulate Zitterbewegung.", "kind": "method", "line": 1347, "name": "_zitterbewegung", "signature": "def _zitterbewegung(self)"}, {"doc": "Calculate spin-orbit coupling.", "kind": "method", "line": 1363, "name": "_spin_orbit", "signature": "def _spin_orbit(self)"}, {"doc": "Display QED effects menu.", "kind": "method", "line": 1378, "name": "_show_qed_menu", "signature": "def _show_qed_menu(self)"}, {"doc": "Calculate Lamb shift.", "kind": "method", "line": 1406, "name": "_lamb_shift", "signature": "def _lamb_shift(self)"}, {"doc": "Calculate anomalous magnetic moment.", "kind": "method", "line": 1426, "name": "_anomalous_moment", "signature": "def _anomalous_moment(self)"}, {"doc": "Calculate vacuum polarization effects.", "kind": "method", "line": 1453, "name": "_vacuum_polarization", "signature": "def _vacuum_polarization(self)"}, {"doc": "Show full QED corrections.", "kind": "method", "line": 1468, "name": "_full_qed", "signature": "def _full_qed(self)"}, {"doc": "Display quantum algorithms menu.", "kind": "method", "line": 1482, "name": "_show_algorithms_menu", "signature": "def _show_algorithms_menu(self)"}, {"doc": "Demonstrate Grover's search algorithm.", "kind": "method", "line": 1510, "name": "_grover_search", "signature": "def _grover_search(self)"}, {"doc": "Apply one Grover iteration: oracle then diffusion.", "kind": "method", "line": 1558, "name": "_apply_grover_iteration", "signature": "def _apply_grover_iteration(self, state, marked, n_qubits)"}, {"doc": "Demonstrate Quantum Fourier Transform.", "kind": "method", "line": 1607, "name": "_qft_demo", "signature": "def _qft_demo(self)"}, {"doc": "Demonstrate phase estimation.", "kind": "method", "line": 1669, "name": "_phase_estimation", "signature": "def _phase_estimation(self)"}, {"doc": "Demonstrate VQE.", "kind": "method", "line": 1746, "name": "_vqe_demo", "signature": "def _vqe_demo(self)"}, {"doc": "Display benchmarks menu.", "kind": "method", "line": 1816, "name": "_show_benchmark_menu", "signature": "def _show_benchmark_menu(self)"}, {"doc": "Run MPS scaling benchmark.", "kind": "method", "line": 1844, "name": "_mps_scaling_benchmark", "signature": "def _mps_scaling_benchmark(self)"}, {"doc": "Benchmark gate performance.", "kind": "method", "line": 1872, "name": "_gate_performance", "signature": "def _gate_performance(self)"}, {"doc": "Compare memory usage.", "kind": "method", "line": 1907, "name": "_memory_comparison", "signature": "def _memory_comparison(self)"}, {"doc": "Study entanglement scaling.", "kind": "method", "line": 1927, "name": "_entanglement_scaling", "signature": "def _entanglement_scaling(self)"}, {"doc": "Display configuration menu.", "kind": "method", "line": 1953, "name": "_show_config_menu", "signature": "def _show_config_menu(self)"}, {"doc": "View current configuration.", "kind": "method", "line": 1984, "name": "_view_config", "signature": "def _view_config(self)"}, {"doc": "List available atoms.", "kind": "method", "line": 1999, "name": "_list_atoms", "signature": "def _list_atoms(self)"}, {"doc": "List available molecules.", "kind": "method", "line": 2016, "name": "_list_molecules_config", "signature": "def _list_molecules_config(self)"}, {"doc": "List available experiments.", "kind": "method", "line": 2020, "name": "_list_experiments", "signature": "def _list_experiments(self)"}, {"doc": "Show system information.", "kind": "method", "line": 2036, "name": "_system_info", "signature": "def _system_info(self)"}, {"doc": "Display particle physics menu (Higgs analysis).", "kind": "method", "line": 2055, "name": "_show_particle_physics_menu", "signature": "def _show_particle_physics_menu(self)"}, {"doc": "Run the Higgs boson 4-lepton quantum analysis.", "kind": "method", "line": 2077, "name": "_run_higgs_analysis", "signature": "def _run_higgs_analysis(self)"}, {"doc": "Show information about the Higgs analysis.", "kind": "method", "line": 2101, "name": "_higgs_about", "signature": "def _higgs_about(self)"}, {"doc": "Display quantum visualization menu.", "kind": "method", "line": 2129, "name": "_show_visualization_menu", "signature": "def _show_visualization_menu(self)"}, {"doc": "Run brutalist quantum state visualizer.", "kind": "method", "line": 2154, "name": "_run_quantum_dash", "signature": "def _run_quantum_dash(self)"}, {"doc": "Run 3D holographic quantum dashboard.", "kind": "method", "line": 2201, "name": "_run_quantum_3dview", "signature": "def _run_quantum_3dview(self)"}, {"doc": "Run the standard quantum state visualizer.", "kind": "method", "line": 2247, "name": "_run_quantum_visualizer", "signature": "def _run_quantum_visualizer(self)"}, {"doc": "Run H2 polarizability / Stark effect VQE from app.py.", "kind": "method", "line": 2285, "name": "_run_polarizability_vqe", "signature": "def _run_polarizability_vqe(self)"}, {"doc": "Show help information.", "kind": "method", "line": 2316, "name": "_show_help", "signature": "def _show_help(self)"}, {"doc": "Exit the menu system.", "kind": "method", "line": 2350, "name": "_quit", "signature": "def _quit(self)"}, {"kind": "method", "line": 2379, "name": "test_header", "signature": "def test_header(name)"}, {"kind": "method", "line": 951, "name": "radial_wf", "signature": "def radial_wf(n, l, r)"}, {"kind": "method", "line": 959, "name": "spherical_harm_real", "signature": "def spherical_harm_real(l, m, theta, phi)"}, {"kind": "method", "line": 1764, "name": "compute_energy", "signature": "def compute_energy(state, n_qubits)"}]}, {"doc": "Quantum Framework Molecular Module - CORRECTED VERSION ======================================================= Molecular simulation with VQE and UCCSD ansatz using MPS representation.  CRITICAL FIXES: - Corrected H2 Hamiltonian coefficients (verified against PySCF/OpenFermion) - Fixed OpenFermion compatibility (InteractionOperator API) - Proper nuclear repulsion handling - UCCSD ansatz with correct excitation operators  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_framework_molecular.py", "kind": "module", "label": "quantum_framework_molecular.py", "language": "py", "sha256": "e64f3598dc2b412d", "symbol_count": 29, "symbols": [{"kind": "function", "line": 58, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"kind": "class", "line": 72, "name": "MoleculeData", "signature": "class MoleculeData"}, {"doc": "Build molecule data for quantum chemistry calculations.", "kind": "class", "line": 85, "name": "MoleculeBuilder", "signature": "class MoleculeBuilder"}, {"doc": "Exact Jordan-Wigner energy evaluator.\n\nCORRECTED: Uses verified Hamiltonian coefficients from standard references.", "kind": "class", "line": 156, "name": "ExactJWEnergy", "signature": "class ExactJWEnergy"}, {"doc": "Unitary Coupled Cluster Singles and Doubles ansatz.\n\nFor H2 in the 2-qubit active space model:\n- Qubit 0 represents the bonding orbital occupation\n- Qubit 1 represents the anti-bonding orbital occupation\n- HF state: |10> (bonding occupied, anti-bonding empty)\n- The double excitation |10> <-> |01> is mediated by X0X1 + Y0Y1 terms", "kind": "class", "line": 334, "name": "UCCSDAnsatz", "signature": "class UCCSDAnsatz"}, {"kind": "class", "line": 476, "name": "VQEResult", "signature": "class VQEResult"}, {"doc": "Variational Quantum Eigensolver.\n\nCORRECTED: Uses proper UCCSD ansatz and Hamiltonian evaluation.", "kind": "class", "line": 510, "name": "VQESolver", "signature": "class VQESolver"}, {"doc": "Run VQE for H2 molecule - convenience function.", "kind": "method", "line": 630, "name": "run_vqe_h2", "signature": "def run_vqe_h2(max_iter)"}, {"doc": "Build H2 molecule with STO-3G basis.", "kind": "method", "line": 89, "name": "h2_sto3g", "signature": "def h2_sto3g(bond_length)"}, {"doc": "Build H2 using PySCF.", "kind": "method", "line": 96, "name": "_h2_pyscf", "signature": "def _h2_pyscf(bond_length)"}, {"doc": "Build H2 with hardcoded values - CORRECTED for 2-qubit active space.", "kind": "method", "line": 128, "name": "_h2_hardcoded", "signature": "def _h2_hardcoded(bond_length)"}, {"kind": "method", "line": 163, "name": "__init__", "signature": "def __init__(self, mol, n_qubits)"}, {"doc": "Build the molecular Hamiltonian in JW representation.", "kind": "method", "line": 170, "name": "_build_hamiltonian", "signature": "def _build_hamiltonian(self)"}, {"doc": "Build Hamiltonian using OpenFermion - FIXED API.", "kind": "method", "line": 178, "name": "_build_openfermion_hamiltonian", "signature": "def _build_openfermion_hamiltonian(self)"}, {"doc": "Build hardcoded H2 Hamiltonian - CORRECTED COEFFICIENTS.\n\nThe H2/STO-3G Hamiltonian in the minimal active space (2 qubits)\nwith Jordan-Wigner transformation.\n\nDerived from reference energies:\n- E_HF = -1.11675928 Ha\n- E_FCI = -1.13728383 Ha\n- E_nuc = 0.71996899 Ha\n\nThe Hamiltonian has the form:\nH = E_nuc + h1*Z0 + h2*Z1 + h3*Z0*Z1 + h4*X0*X1 + h5*Y0*Y1\n\nFor the 2-qubit active space model where:\n- |10> is the HF state (bonding orbital occupied)\n- The ground state is a superposition of |10> and |01>", "kind": "method", "line": 231, "name": "_build_hardcoded_hamiltonian", "signature": "def _build_hardcoded_hamiltonian(self)"}, {"doc": "Apply Pauli operator to state vector - CORRECTED.", "kind": "method", "line": 276, "name": "_apply_pauli", "signature": "def _apply_pauli(self, state, pauli)"}, {"doc": "Compute ⟨ψ|H|ψ⟩ for the given state.", "kind": "method", "line": 302, "name": "expectation_value", "signature": "def expectation_value(self, state)"}, {"doc": "Evaluate energy from amplitudes (supports both numpy and torch).", "kind": "method", "line": 318, "name": "evaluate", "signature": "def evaluate(self, amps)"}, {"kind": "method", "line": 330, "name": "__call__", "signature": "def __call__(self, amps)"}, {"kind": "method", "line": 345, "name": "__init__", "signature": "def __init__(self, n_qubits, n_electrons)"}, {"doc": "Apply double excitation for 2-qubit H2 model.\n\nThis rotates between |10> and |01>:\n|10> -> cos(theta)*|10> - sin(theta)*|01>\n|01> -> sin(theta)*|10> + cos(theta)*|01>", "kind": "method", "line": 373, "name": "apply_double_excitation_2q", "signature": "def apply_double_excitation_2q(self, state, theta)"}, {"doc": "Apply single excitation as Givens rotation.", "kind": "method", "line": 395, "name": "apply_single_excitation", "signature": "def apply_single_excitation(self, state, o, v, theta)"}, {"doc": "Apply double excitation for 4+ qubit systems.", "kind": "method", "line": 416, "name": "apply_double_excitation", "signature": "def apply_double_excitation(self, state, o1, o2, v1, v2, theta)"}, {"doc": "Apply UCCSD ansatz to state.", "kind": "method", "line": 443, "name": "apply", "signature": "def apply(self, state, thetas)"}, {"kind": "method", "line": 490, "name": "__repr__", "signature": "def __repr__(self)"}, {"kind": "method", "line": 517, "name": "__init__", "signature": "def __init__(self, mol)"}, {"doc": "Prepare Hartree-Fock state.\n\nFor H2 with 2 electrons in 4 spin-orbitals:\n|1100⟩ means electrons in orbitals 0 and 1.", "kind": "method", "line": 528, "name": "prepare_hf_state", "signature": "def prepare_hf_state(self)"}, {"doc": "Run VQE optimization.", "kind": "method", "line": 546, "name": "run", "signature": "def run(self, max_iter, tol)"}, {"kind": "method", "line": 566, "name": "cost", "signature": "def cost(thetas)"}]}, {"doc": "Quantum Framework Molecular Module - FIXED VERSION ==================================================== Molecular simulation with VQE and UCCSD ansatz using MPS representation.  FIXES: - Corrected OpenFermion geometry (charge=0 for H2 neutral) - Fixed hardcoded Hamiltonian coefficients - Fixed _apply_pauli for correct amplitude indexing - Fixed HF state preparation (0s and 1s correctly) - Fixed PySCF atom string syntax - Fixed n_orbitals and n_qubits for H2/STO-3G  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_framework_molecular_fixed.py", "kind": "module", "label": "quantum_framework_molecular_fixed.py", "language": "py", "sha256": "a9d93bfd91ff15a7", "symbol_count": 29, "symbols": [{"kind": "function", "line": 50, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"kind": "class", "line": 64, "name": "MoleculeData", "signature": "class MoleculeData"}, {"doc": "Build molecule data for quantum chemistry calculations.\nUses PySCF when available, falls back to hardcoded values.", "kind": "class", "line": 77, "name": "MoleculeBuilder", "signature": "class MoleculeBuilder"}, {"doc": "Exact Jordan-Wigner energy evaluator.\nComputes molecular energy using JW transformation.\n\nFIXED: Proper Hamiltonian construction and expectation values.", "kind": "class", "line": 147, "name": "ExactJWEnergy", "signature": "class ExactJWEnergy"}, {"doc": "Get single and double excitation indices for UCCSD.", "kind": "method", "line": 323, "name": "_get_sd_indices", "signature": "def _get_sd_indices(n_electrons, n_qubits)"}, {"doc": "Unitary Coupled Cluster Singles and Doubles ansatz.\n\nFIXED: Correct excitation operator implementation.", "kind": "class", "line": 343, "name": "UCCSDAnsatz", "signature": "class UCCSDAnsatz"}, {"kind": "class", "line": 457, "name": "VQEResult", "signature": "class VQEResult"}, {"doc": "Variational Quantum Eigensolver.\n\nFIXED: Correct HF state preparation and energy evaluation.", "kind": "class", "line": 491, "name": "VQESolver", "signature": "class VQESolver"}, {"doc": "Run a quick VQE demo to verify the fixes.", "kind": "method", "line": 614, "name": "run_vqe_demo", "signature": "def run_vqe_demo()"}, {"doc": "Build H2 molecule with STO-3G basis.", "kind": "method", "line": 84, "name": "h2_sto3g", "signature": "def h2_sto3g(bond_length)"}, {"doc": "Build H2 using PySCF - FIXED atom string syntax.", "kind": "method", "line": 91, "name": "_h2_pyscf", "signature": "def _h2_pyscf(bond_length)"}, {"doc": "Build H2 with hardcoded values - FIXED coefficients.", "kind": "method", "line": 126, "name": "_h2_hardcoded", "signature": "def _h2_hardcoded()"}, {"kind": "method", "line": 155, "name": "__init__", "signature": "def __init__(self, mol, n_qubits)"}, {"doc": "Build the molecular Hamiltonian in JW representation.", "kind": "method", "line": 162, "name": "_build_hamiltonian", "signature": "def _build_hamiltonian(self)"}, {"doc": "Build Hamiltonian using OpenFermion - FIXED geometry.", "kind": "method", "line": 169, "name": "_build_openfermion_hamiltonian", "signature": "def _build_openfermion_hamiltonian(self)"}, {"doc": "Build hardcoded H2 Hamiltonian - FIXED coefficients.\n\nThe H2/STO-3G Hamiltonian in JW form (standard reference):\nH = g0*I + g1*Z0 + g2*Z1 + g3*Z0*Z1 + g4*X0*X1 + g5*Y0*Y1\n\nWith coefficients for bond length 0.735 Å:", "kind": "method", "line": 215, "name": "_build_hardcoded_hamiltonian", "signature": "def _build_hardcoded_hamiltonian(self)"}, {"doc": "Apply Pauli operator to state vector.\n\nFIXED: Correct amplitude indexing and phase handling.", "kind": "method", "line": 247, "name": "_apply_pauli", "signature": "def _apply_pauli(self, state, pauli)"}, {"doc": "Compute ⟨ψ|H|ψ⟩ for the given state.\n\nFIXED: Correct inner product calculation.", "kind": "method", "line": 281, "name": "expectation_value", "signature": "def expectation_value(self, state)"}, {"doc": "Evaluate energy from MPS amplitudes.", "kind": "method", "line": 305, "name": "evaluate", "signature": "def evaluate(self, amps)"}, {"kind": "method", "line": 319, "name": "__call__", "signature": "def __call__(self, amps)"}, {"kind": "method", "line": 350, "name": "__init__", "signature": "def __init__(self, n_qubits, n_electrons, backend)"}, {"doc": "Apply single excitation operator exp(theta * (a_v† a_o - a_o† a_v)).\n\nIn JW, this is a Givens rotation between orbitals o and v.", "kind": "method", "line": 359, "name": "apply_single_excitation", "signature": "def apply_single_excitation(self, state, o, v, theta)"}, {"doc": "Apply double excitation operator.\n\nSimplified: applies pairwise excitation with rotation.", "kind": "method", "line": 387, "name": "apply_double_excitation", "signature": "def apply_double_excitation(self, state, o1, o2, v1, v2, theta)"}, {"doc": "Apply UCCSD ansatz to state.", "kind": "method", "line": 419, "name": "apply", "signature": "def apply(self, state, thetas)"}, {"kind": "method", "line": 471, "name": "__repr__", "signature": "def __repr__(self)"}, {"kind": "method", "line": 498, "name": "__init__", "signature": "def __init__(self, qc, config)"}, {"doc": "Prepare Hartree-Fock state.\n\nFIXED: Correct bitstring with 0s and 1s.\n\nFor H2 with 2 electrons in 4 spin-orbitals:\n|1100⟩ means electrons in orbitals 0 and 1 (occupied spin-orbitals)", "kind": "method", "line": 502, "name": "prepare_hf_state", "signature": "def prepare_hf_state(self, mol)"}, {"doc": "Run VQE optimization.", "kind": "method", "line": 525, "name": "run", "signature": "def run(self, mol, backend, max_iter, tol)"}, {"kind": "method", "line": 556, "name": "cost", "signature": "def cost(thetas)"}]}, {"doc": "Quantum Framework Molecular Module - Production Version ======================================================== Molecular VQE simulation with OpenFermion integration, MPS optimization, and multiple precision modes.  Features: - OpenFermion for all Hamiltonians (NO hardcoded values) - Precision mode flag: direct statevector vs MPS compression - Integration with Schrodinger, Dirac, Hamiltonian backends - Smart initialization with MP2 + parameter scan - Cached Pauli operations for x10-100 speedup - Particle-conserving MPS for stability - Adaptive bond dimension - TOML configuration  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_framework_molecular_v2.py", "kind": "module", "label": "quantum_framework_molecular_v2.py", "language": "py", "sha256": "1ba4ab363bb2adf0", "symbol_count": 61, "symbols": [{"kind": "function", "line": 75, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"doc": "Configuration for molecular VQE simulations.", "kind": "class", "line": 95, "name": "MolecularConfig", "signature": "class MolecularConfig"}, {"doc": "Molecular data structure with all necessary information.", "kind": "class", "line": 172, "name": "MoleculeData", "signature": "class MoleculeData"}, {"doc": "Build molecules using OpenFermion + PySCF.\n\nNO hardcoded values - everything comes from quantum chemistry calculations.", "kind": "class", "line": 200, "name": "MoleculeBuilder", "signature": "class MoleculeBuilder"}, {"doc": "Build molecular Hamiltonians using OpenFermion.\n\nNO hardcoded coefficients - everything from first principles.", "kind": "class", "line": 352, "name": "HamiltonianBuilder", "signature": "class HamiltonianBuilder"}, {"doc": "Precomputed Pauli operation for fast application.", "kind": "class", "line": 468, "name": "CachedPauliOperation", "signature": "class CachedPauliOperation"}, {"doc": "Hamiltonian evaluator with cached Pauli operations.\n\nImprovement: x10-100 speedup for repeated evaluations.", "kind": "class", "line": 477, "name": "CachedHamiltonianEvaluator", "signature": "class CachedHamiltonianEvaluator"}, {"doc": "Smart parameter initialization for VQE.\n\nImprovements:\n- MP2 amplitude estimation\n- Systematic parameter scan\n- Parabolic refinement\n- Sensitivity analysis", "kind": "class", "line": 613, "name": "SmartInitializer", "signature": "class SmartInitializer"}, {"doc": "Unitary Coupled Cluster Singles and Doubles ansatz.\n\nFeatures:\n- Particle-conserving excitations\n- Works with both direct and MPS representations\n- Identity check (θ=0 → HF state)", "kind": "class", "line": 746, "name": "UCCSDAnsatz", "signature": "class UCCSDAnsatz"}, {"doc": "Matrix Product State for scalable quantum simulation.\n\nFeatures:\n- Adaptive bond dimension\n- Efficient gate application\n- Entanglement tracking", "kind": "class", "line": 879, "name": "MPSState", "signature": "class MPSState"}, {"doc": "State representation that preserves particle number symmetry.\n\nImprovement: x4 reduction in Hilbert space, better convergence.", "kind": "class", "line": 982, "name": "ParticleConservingState", "signature": "class ParticleConservingState"}, {"doc": "VQE result container.", "kind": "class", "line": 1064, "name": "VQEResult", "signature": "class VQEResult"}, {"doc": "Production VQE solver with all improvements.\n\nFeatures:\n- OpenFermion Hamiltonians (no hardcoded values)\n- Precision mode flag (direct vs MPS)\n- Smart initialization with MP2 + scan\n- Cached Pauli operations\n- Particle conservation option\n- Backend integration", "kind": "class", "line": 1104, "name": "VQESolver", "signature": "class VQESolver"}, {"doc": "Integration with existing backends (Hamiltonian, Schrodinger, Dirac).\n\nUses pre-trained models from QC repository.", "kind": "class", "line": 1287, "name": "BackendIntegrator", "signature": "class BackendIntegrator"}, {"doc": "Convenience function to run VQE.\n\nArgs:\n    molecule: Molecule name (\"H2\", \"H2O\", \"LiH\")\n    precision_mode: Use direct statevector (True) or MPS (False)\n    config_path: Path to TOML config file\n    **kwargs: Additional config overrides\n\nReturns:\n    VQEResult", "kind": "method", "line": 1332, "name": "run_vqe", "signature": "def run_vqe(molecule, precision_mode, config_path)"}, {"doc": "Load configuration from TOML file.", "kind": "method", "line": 131, "name": "from_toml", "signature": "def from_toml(cls, toml_path)"}, {"doc": "Build molecule using OpenFermion.\n\nArgs:\n    name: Molecule name (e.g., \"H2\", \"H2O\")\n    geometry: List of (atom_symbol, (x, y, z)) in Angstrom\n    basis: Basis set (e.g., \"sto-3g\", \"6-31g\")\n    charge: Molecular charge\n    multiplicity: Spin multiplicity\n    description: Optional description\n\nReturns:\n    MoleculeData with all properties computed", "kind": "method", "line": 208, "name": "build", "signature": "def build(name, geometry, basis, charge, multiplicity, description)"}, {"doc": "Run PySCF directly if openfermionpyscf not available.", "kind": "method", "line": 289, "name": "_run_pyscf_direct", "signature": "def _run_pyscf_direct(geometry, basis, charge, multiplicity)"}, {"doc": "Build H2 molecule.", "kind": "method", "line": 324, "name": "h2", "signature": "def h2(bond_length, basis)"}, {"doc": "Build H2O molecule.", "kind": "method", "line": 330, "name": "h2o", "signature": "def h2o(bond_length_oh, angle_hoh, basis)"}, {"doc": "Build LiH molecule.", "kind": "method", "line": 342, "name": "lih", "signature": "def lih(bond_length, basis)"}, {"doc": "Build Jordan-Wigner transformed Hamiltonian using OpenFermion.\n\nArgs:\n    mol: MoleculeData with geometry, basis, etc.\n\nReturns:\n    (pauli_terms, nuclear_repulsion)", "kind": "method", "line": 360, "name": "build_jw_hamiltonian", "signature": "def build_jw_hamiltonian(mol)"}, {"doc": "Build full Hamiltonian matrix for small systems.\n\nArgs:\n    mol: MoleculeData\n    n_qubits: Number of qubits\n\nReturns:\n    Hamiltonian matrix (2^n_qubits, 2^n_qubits)", "kind": "method", "line": 417, "name": "build_hamiltonian_matrix", "signature": "def build_hamiltonian_matrix(mol, n_qubits)"}, {"doc": "Build matrix for a Pauli string.", "kind": "method", "line": 441, "name": "_pauli_matrix", "signature": "def _pauli_matrix(pauli_list, n_qubits)"}, {"kind": "method", "line": 484, "name": "__init__", "signature": "def __init__(self, mol, config)"}, {"doc": "Precompute all Pauli operations for fast evaluation.", "kind": "method", "line": 503, "name": "_precompute_operations", "signature": "def _precompute_operations(self)"}, {"doc": "Compute index mapping and phases for a Pauli string.", "kind": "method", "line": 522, "name": "_compute_pauli_mapping", "signature": "def _compute_pauli_mapping(self, pauli_list)"}, {"doc": "Apply cached Pauli operation.", "kind": "method", "line": 561, "name": "apply_pauli_fast", "signature": "def apply_pauli_fast(self, state, op)"}, {"doc": "Compute energy expectation value.", "kind": "method", "line": 568, "name": "expectation_value", "signature": "def expectation_value(self, state)"}, {"doc": "Compute expectation for batch of states.", "kind": "method", "line": 587, "name": "batch_expectation", "signature": "def batch_expectation(self, states)"}, {"kind": "method", "line": 624, "name": "__init__", "signature": "def __init__(self, mol, ansatz, evaluator, config)"}, {"doc": "Estimate doubles amplitude from MP2 theory.\n\nθ_MP2 ≈ t_2^(1) / 2\nwhere t_2^(1) = <ij||ab> / (ε_i + ε_j - ε_a - ε_b)", "kind": "method", "line": 630, "name": "estimate_mp2_amplitude", "signature": "def estimate_mp2_amplitude(self)"}, {"doc": "Systematic scan of parameter space.\n\nReturns:\n    (best_thetas, best_energy)", "kind": "method", "line": 657, "name": "scan_parameter_space", "signature": "def scan_parameter_space(self, hf_state, n_samples, param_range)"}, {"doc": "Refine minimum using parabolic interpolation.", "kind": "method", "line": 712, "name": "_parabolic_refinement", "signature": "def _parabolic_refinement(self, x_vals, y_vals, best_idx, n_params)"}, {"doc": "Complete initialization with all techniques.", "kind": "method", "line": 735, "name": "initialize", "signature": "def initialize(self, hf_state)"}, {"kind": "method", "line": 756, "name": "__init__", "signature": "def __init__(self, n_qubits, n_electrons, config)"}, {"doc": "Generate all single and double excitations.", "kind": "method", "line": 764, "name": "_generate_excitations", "signature": "def _generate_excitations(self)"}, {"doc": "Apply UCCSD ansatz to state.\n\nArgs:\n    state: State vector (2^n_qubits,)\n    thetas: Parameters (n_params,)\n\nReturns:\n    Transformed state vector", "kind": "method", "line": 785, "name": "apply", "signature": "def apply(self, state, thetas)"}, {"doc": "Apply single excitation as Givens rotation.", "kind": "method", "line": 817, "name": "_apply_single", "signature": "def _apply_single(self, state, i, a, theta)"}, {"doc": "Apply double excitation.", "kind": "method", "line": 838, "name": "_apply_double", "signature": "def _apply_double(self, state, i, j, a, b, theta)"}, {"doc": "Verify that θ=0 gives HF state.", "kind": "method", "line": 862, "name": "verify_identity", "signature": "def verify_identity(self, hf_state, evaluator, hf_energy)"}, {"kind": "method", "line": 889, "name": "__init__", "signature": "def __init__(self, n_qubits, config)"}, {"doc": "Convert MPS to full statevector.", "kind": "method", "line": 910, "name": "to_statevector", "signature": "def to_statevector(self)"}, {"doc": "Create MPS from statevector.", "kind": "method", "line": 918, "name": "from_statevector", "signature": "def from_statevector(cls, state, n_qubits, config)"}, {"doc": "Compute entanglement entropy at bond.", "kind": "method", "line": 953, "name": "compute_entanglement", "signature": "def compute_entanglement(self, bond_idx)"}, {"kind": "method", "line": 989, "name": "__init__", "signature": "def __init__(self, n_qubits, n_particles)"}, {"doc": "Generate all states with fixed particle number.", "kind": "method", "line": 1000, "name": "_generate_fock_states", "signature": "def _generate_fock_states(self)"}, {"doc": "Create HF state in subspace.", "kind": "method", "line": 1013, "name": "hf_state", "signature": "def hf_state(self)"}, {"doc": "Apply excitation preserving particle number.", "kind": "method", "line": 1027, "name": "apply_excitation", "signature": "def apply_excitation(self, state, occ, vir, theta)"}, {"doc": "Convert subspace state to full statevector.", "kind": "method", "line": 1051, "name": "to_full_statevector", "signature": "def to_full_statevector(self, state)"}, {"kind": "method", "line": 1081, "name": "__repr__", "signature": "def __repr__(self)"}, {"kind": "method", "line": 1117, "name": "__init__", "signature": "def __init__(self, mol, config)"}, {"doc": "Prepare Hartree-Fock state.", "kind": "method", "line": 1137, "name": "prepare_hf_state", "signature": "def prepare_hf_state(self)"}, {"doc": "Evaluate energy.", "kind": "method", "line": 1152, "name": "evaluate", "signature": "def evaluate(self, state)"}, {"doc": "Apply UCCSD ansatz.", "kind": "method", "line": 1159, "name": "apply_ansatz", "signature": "def apply_ansatz(self, state, thetas)"}, {"doc": "Run VQE optimization.", "kind": "method", "line": 1184, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 1294, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Load pre-trained backend models.", "kind": "method", "line": 1301, "name": "_load_models", "signature": "def _load_models(self)"}, {"doc": "Get energy estimate from backend model.", "kind": "method", "line": 1318, "name": "get_backend_energy", "signature": "def get_backend_energy(self, state, backend_name)"}, {"kind": "class", "line": 301, "name": "PseudoMolData", "signature": "class PseudoMolData"}, {"kind": "method", "line": 1212, "name": "cost", "signature": "def cost(thetas)"}]}, {"doc": "Quantum Framework Physics Module ================================ Neural network backends for quantum physics simulations. Provides spectral layers, Hamiltonian networks, and Dirac operators.  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_framework_physics.py", "kind": "module", "label": "quantum_framework_physics.py", "language": "py", "sha256": "59443c4d5099f38d", "symbol_count": 52, "symbols": [{"doc": "Spectral convolution in frequency domain.\nLearns complex kernels that modulate Fourier coefficients.", "kind": "class", "line": 26, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"doc": "Hamiltonian backbone: single-channel field -> H|psi>.\nShared by all physics backends as the H operator.", "kind": "class", "line": 66, "name": "HamiltonianBackboneNet", "signature": "class HamiltonianBackboneNet(Module)"}, {"doc": "Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction.\nUses spectral convolution for physics-informed evolution.", "kind": "class", "line": 92, "name": "SchrodingerSpectralNet", "signature": "class SchrodingerSpectralNet(Module)"}, {"doc": "Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor.\nHandles 4-component spinor evolution for relativistic quantum mechanics.", "kind": "class", "line": 120, "name": "DiracSpectralNet", "signature": "class DiracSpectralNet(Module)"}, {"doc": "Dirac gamma matrices in Dirac (standard) or Weyl representation.\ngamma^0 = beta, gamma^i = beta * alpha_i", "kind": "class", "line": 150, "name": "GammaMatrices", "signature": "class GammaMatrices"}, {"doc": "Spatial potentials for eigenstate initialization.", "kind": "class", "line": 250, "name": "PotentialGenerator", "signature": "class PotentialGenerator"}, {"doc": "Dirac Hamiltonian operator for relativistic quantum mechanics.\nH_Dirac = c * alpha . p + beta * m * c^2 + V(r)\n\nIn atomic units (c = 1/alpha ~ 137):\nH = c * alpha . p + beta * m * c^2 + V", "kind": "class", "line": 294, "name": "DiracHamiltonianOperator", "signature": "class DiracHamiltonianOperator"}, {"doc": "Calculates the Lamb shift using Bethe's formula.\nThe Lamb shift is the small difference between 2s_{1/2} and 2p_{1/2} energy levels\ndue to QED effects (vacuum fluctuations and self-energy).", "kind": "class", "line": 382, "name": "LambShiftCalculator", "signature": "class LambShiftCalculator"}, {"doc": "Calculates the electron's anomalous magnetic moment (g-2).\nThe electron g-factor is slightly different from 2 due to QED effects:\ng = 2(1 + a_e) where a_e = alpha/(2*pi) + higher-order terms", "kind": "class", "line": 439, "name": "AnomalousMagneticMoment", "signature": "class AnomalousMagneticMoment"}, {"doc": "Relativistic hydrogen atom with Dirac equation.\nComputes energy levels including fine structure.", "kind": "class", "line": 489, "name": "DiracHydrogenAtom", "signature": "class DiracHydrogenAtom"}, {"doc": "Simulates the Zitterbewegung (trembling motion) of a relativistic electron.\nIn Dirac theory, the position operator has a term oscillating with frequency\n~ 2mc^2/hbar, which is the interference between positive and negative energy states.", "kind": "class", "line": 571, "name": "ZitterbewegungSimulator", "signature": "class ZitterbewegungSimulator"}, {"kind": "method", "line": 32, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"kind": "method", "line": 43, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 72, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers)"}, {"kind": "method", "line": 81, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 98, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)"}, {"kind": "method", "line": 109, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 126, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)"}, {"kind": "method", "line": 139, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 156, "name": "__init__", "signature": "def __init__(self, representation, device)"}, {"kind": "method", "line": 161, "name": "_init_matrices", "signature": "def _init_matrices(self)"}, {"kind": "method", "line": 244, "name": "to", "signature": "def to(self, device)"}, {"kind": "method", "line": 253, "name": "__init__", "signature": "def __init__(self, grid_size, potential_depth, potential_width)"}, {"kind": "method", "line": 258, "name": "_grid", "signature": "def _grid(self)"}, {"kind": "method", "line": 263, "name": "harmonic", "signature": "def harmonic(self)"}, {"kind": "method", "line": 268, "name": "double_well", "signature": "def double_well(self)"}, {"kind": "method", "line": 274, "name": "coulomb", "signature": "def coulomb(self)"}, {"kind": "method", "line": 280, "name": "periodic_lattice", "signature": "def periodic_lattice(self)"}, {"kind": "method", "line": 284, "name": "mixed", "signature": "def mixed(self, seed)"}, {"kind": "method", "line": 303, "name": "__init__", "signature": "def __init__(self, grid_size, electron_mass, c_light, device)"}, {"kind": "method", "line": 311, "name": "_precompute_operators", "signature": "def _precompute_operators(self)"}, {"doc": "Apply Dirac Hamiltonian to 4-component spinor.\n\nArgs:\n    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor\n    potential: Optional scalar potential V(r)\n\nReturns:\n    H * psi with same shape as input", "kind": "method", "line": 318, "name": "apply_dirac_hamiltonian", "signature": "def apply_dirac_hamiltonian(self, spinor, potential)"}, {"doc": "Time evolution of Dirac spinor using first-order split-step.\npsi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi", "kind": "method", "line": 362, "name": "time_evolution", "signature": "def time_evolution(self, spinor, dt, potential, normalization_eps)"}, {"kind": "method", "line": 389, "name": "__init__", "signature": "def __init__(self, alpha_fs, c_light, electron_mass)"}, {"doc": "Bethe's non-relativistic formula for Lamb shift.\nDelta E_Lamb = (8*alpha^3 / 3*pi*n^3) * |psi_n(0)|^2 * ln(E_avg / E_n)", "kind": "method", "line": 394, "name": "bethe_formula", "signature": "def bethe_formula(self, n, l, Z)"}, {"kind": "method", "line": 407, "name": "_higher_l_shift", "signature": "def _higher_l_shift(self, n, l, Z)"}, {"doc": "Calculate full Lamb shift including radiative corrections.\nDelta E = Delta E_SE + Delta E_Uehling + Delta E_rel", "kind": "method", "line": 412, "name": "full_lamb_shift", "signature": "def full_lamb_shift(self, n, l, j, Z)"}, {"kind": "method", "line": 446, "name": "__init__", "signature": "def __init__(self, alpha_fs)"}, {"kind": "method", "line": 449, "name": "schwinger_term", "signature": "def schwinger_term(self)"}, {"kind": "method", "line": 452, "name": "second_order", "signature": "def second_order(self)"}, {"kind": "method", "line": 456, "name": "third_order", "signature": "def third_order(self)"}, {"kind": "method", "line": 460, "name": "fourth_order", "signature": "def fourth_order(self)"}, {"kind": "method", "line": 464, "name": "fifth_order", "signature": "def fifth_order(self)"}, {"kind": "method", "line": 468, "name": "calculate_a_e", "signature": "def calculate_a_e(self, order)"}, {"kind": "method", "line": 495, "name": "__init__", "signature": "def __init__(self, c_light, alpha_fs)"}, {"doc": "Exact Dirac energy level for hydrogen-like atom.\nE = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)", "kind": "method", "line": 499, "name": "energy_level_dirac", "signature": "def energy_level_dirac(self, n, kappa)"}, {"doc": "Calculate fine structure splitting for given n, l.\nReturns energies for j = l+1/2 and j = l-1/2", "kind": "method", "line": 512, "name": "fine_structure_splitting", "signature": "def fine_structure_splitting(self, n, l)"}, {"kind": "method", "line": 535, "name": "energy_spectrum", "signature": "def energy_spectrum(self, n_max)"}, {"kind": "method", "line": 578, "name": "__init__", "signature": "def __init__(self, grid_size, c_light, electron_mass, device)"}, {"kind": "method", "line": 585, "name": "create_gaussian_wave_packet", "signature": "def create_gaussian_wave_packet(self, sigma, momentum)"}, {"kind": "method", "line": 603, "name": "compute_position_expectation", "signature": "def compute_position_expectation(self, spinor)"}, {"kind": "method", "line": 615, "name": "compute_velocity_expectation", "signature": "def compute_velocity_expectation(self, spinor)"}]}, {"doc": "Quantum Framework Visualization Module ======================================== Orbital visualization, Monte Carlo sampling, and entangled state visualization.  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_framework_visualization.py", "kind": "module", "label": "quantum_framework_visualization.py", "language": "py", "sha256": "57b15a048568c119", "symbol_count": 20, "symbols": [{"kind": "function", "line": 46, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"doc": "Calculates hydrogen atom wavefunctions for visualization.\nUses analytical formulas for radial and angular parts.", "kind": "class", "line": 59, "name": "WavefunctionCalculator", "signature": "class WavefunctionCalculator"}, {"doc": "Monte Carlo sampling for orbital visualization.\nUses rejection sampling to generate 3D point clouds.", "kind": "class", "line": 115, "name": "MonteCarloSampler", "signature": "class MonteCarloSampler"}, {"doc": "High-resolution visualization of hydrogen orbitals.\nCreates 2D projections and 3D scatter plots.", "kind": "class", "line": 202, "name": "OrbitalVisualizer", "signature": "class OrbitalVisualizer"}, {"doc": "Monte Carlo sampler for entangled hydrogen states.\nSamples from joint probability distribution of entangled orbitals.", "kind": "class", "line": 290, "name": "EntangledHydrogenSampler", "signature": "class EntangledHydrogenSampler"}, {"doc": "Visualizer for entangled hydrogen states.\nCreates high-resolution visualizations with multiple orbitals.", "kind": "class", "line": 325, "name": "EntangledHydrogenVisualizer", "signature": "class EntangledHydrogenVisualizer"}, {"kind": "method", "line": 65, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 69, "name": "radial_wavefunction", "signature": "def radial_wavefunction(n, l, r)"}, {"kind": "method", "line": 81, "name": "spherical_harmonic_real", "signature": "def spherical_harmonic_real(l, m, theta, phi)"}, {"kind": "method", "line": 92, "name": "psi_3d", "signature": "def psi_3d(self, n, l, m, r, theta, phi)"}, {"kind": "method", "line": 97, "name": "psi_on_grid", "signature": "def psi_on_grid(self, n, l, m)"}, {"kind": "method", "line": 121, "name": "__init__", "signature": "def __init__(self, config, wavefunction_calc)"}, {"kind": "method", "line": 125, "name": "find_max_probability", "signature": "def find_max_probability(self, n, l, m)"}, {"kind": "method", "line": 149, "name": "sample", "signature": "def sample(self, n, l, m, num_samples)"}, {"kind": "method", "line": 208, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 211, "name": "visualize", "signature": "def visualize(self, data, save_path)"}, {"kind": "method", "line": 296, "name": "__init__", "signature": "def __init__(self, config, wavefunction_calc)"}, {"kind": "method", "line": 301, "name": "sample_entangled_state", "signature": "def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples, entanglement_weight)"}, {"kind": "method", "line": 331, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 334, "name": "visualize", "signature": "def visualize(self, data, quantum_result, save_path)"}]}, {"doc": "Q2C Quantum Lab - Interactive Educational TUI ============================================= A guided, bilingual (English/Spanish) terminal experience for learning quantum computing and quantum chemistry with the Q2C simulator.  It is built on top of the real simulation engine (quantum_framework_core): every probability bar, entropy value and orbital picture you see is computed live, not pre-recorded.  Usage: python quantum_lab.py                # interactive, asks for language python quantum_lab.py --lang es      # start in Spanish python quantum_lab.py --lesson 3     # jump straight into lesson 3  Sections: 1. Lessons     - guided course: qubits, measurement, entanglement, Grover search and quantum chemistry 2. Playground  - build circuits gate by gate, watch the state evolve 3. Chemistry   - molecule cards and an ASCII hydrogen-orbital viewer 4. Glossary    - quick reference of the vocabulary used everywhere  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_lab.py", "kind": "module", "label": "quantum_lab.py", "language": "py", "sha256": "5eae9ecad5919795", "symbol_count": 64, "symbols": [{"doc": "Build a table of probability bars for a state's distribution.", "kind": "function", "line": 325, "name": "probability_bars", "signature": "def probability_bars(probs, n_qubits, lang, max_rows)"}, {"doc": "Build a table of measurement-count bars.", "kind": "function", "line": 353, "name": "counts_bars", "signature": "def counts_bars(counts, total, lang)"}, {"doc": "Render an ASCII timeline of the circuit, one line per qubit.", "kind": "function", "line": 369, "name": "draw_circuit", "signature": "def draw_circuit(n_qubits, instructions)"}, {"doc": "Parse an angle like '1.57', 'pi', '-pi/2' or '3*pi/4' into radians.", "kind": "function", "line": 394, "name": "parse_angle", "signature": "def parse_angle(token)"}, {"doc": "Draw measurement outcomes from a probability distribution.", "kind": "function", "line": 414, "name": "sample_measurements", "signature": "def sample_measurements(probs, n_qubits, n_samples, rng)"}, {"doc": "Definition of a real hydrogen orbital for the ASCII viewer.", "kind": "class", "line": 431, "name": "OrbitalSpec", "signature": "class OrbitalSpec"}, {"doc": "Hydrogen radial wavefunction R_nl(r) in atomic units.", "kind": "method", "line": 440, "name": "_radial", "signature": "def _radial(n, l, r)"}, {"kind": "method", "line": 452, "name": "_ang_s", "signature": "def _ang_s(x, y, z, r)"}, {"kind": "method", "line": 456, "name": "_ang_pz", "signature": "def _ang_pz(x, y, z, r)"}, {"kind": "method", "line": 460, "name": "_ang_px", "signature": "def _ang_px(x, y, z, r)"}, {"kind": "method", "line": 464, "name": "_ang_dz2", "signature": "def _ang_dz2(x, y, z, r)"}, {"kind": "method", "line": 469, "name": "_ang_dxz", "signature": "def _ang_dxz(x, y, z, r)"}, {"kind": "method", "line": 473, "name": "_ang_dxy", "signature": "def _ang_dxy(x, y, z, r)"}, {"doc": "Render a real scalar field as colored ASCII: brightness = |psi|^2, color = sign.", "kind": "method", "line": 490, "name": "field_to_text", "signature": "def field_to_text(psi)"}, {"doc": "Render |psi|^2 of a hydrogen orbital on a plane slice as colored ASCII.", "kind": "method", "line": 516, "name": "render_orbital", "signature": "def render_orbital(spec, rows, cols)"}, {"doc": "Render the bonding or antibonding LCAO molecular orbital of H2.", "kind": "method", "line": 536, "name": "render_h2_molecular_orbital", "signature": "def render_h2_molecular_orbital(kind, rows, cols)"}, {"doc": "Draw an ASCII plot of E(theta) over one full period with HF and FCI\nreference lines and an optional optimizer marker.", "kind": "method", "line": 556, "name": "landscape_plot", "signature": "def landscape_plot(energies, e_hf, e_fci, marker, rows)"}, {"doc": "Minimal live VQE for H2 in the 2-qubit active space.\n\nThe trial state cos(theta/2)|01> + sin(theta/2)|10> is prepared with a\nreal circuit (Ry, CNOT, X) on the MPS engine, and the energy is read\nfrom the Jordan-Wigner H2 Hamiltonian of quantum_framework_molecular.", "kind": "class", "line": 604, "name": "H2VQEEngine", "signature": "class H2VQEEngine"}, {"kind": "class", "line": 659, "name": "Quiz", "signature": "class Quiz"}, {"kind": "class", "line": 667, "name": "Lesson", "signature": "class Lesson"}, {"doc": "Interactive educational TUI driven by the real Q2C engine.", "kind": "class", "line": 672, "name": "QuantumLab", "signature": "class QuantumLab"}, {"doc": "Entry point used both standalone and from quantum_framework_main.", "kind": "method", "line": 1474, "name": "launch_quantum_lab", "signature": "def launch_quantum_lab(config, loader, lang, lesson)"}, {"kind": "method", "line": 1500, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 567, "name": "row_of", "signature": "def row_of(e)"}, {"kind": "method", "line": 613, "name": "__init__", "signature": "def __init__(self, qc)"}, {"kind": "method", "line": 621, "name": "ansatz_instructions", "signature": "def ansatz_instructions(self, theta)"}, {"kind": "method", "line": 624, "name": "energy", "signature": "def energy(self, theta)"}, {"kind": "method", "line": 633, "name": "correlation_pct", "signature": "def correlation_pct(self, e)"}, {"kind": "method", "line": 636, "name": "landscape", "signature": "def landscape(self, cols)"}, {"doc": "Gradient descent; yields (iteration, theta, energy) live.", "kind": "method", "line": 640, "name": "optimize", "signature": "def optimize(self, theta0, lr, max_iters, tol)"}, {"kind": "method", "line": 675, "name": "__init__", "signature": "def __init__(self, config, loader, lang)"}, {"kind": "method", "line": 683, "name": "t", "signature": "def t(self, key)"}, {"kind": "method", "line": 690, "name": "pause", "signature": "def pause(self)"}, {"kind": "method", "line": 697, "name": "panel", "signature": "def panel(self, body, title, style)"}, {"kind": "method", "line": 701, "name": "show_state", "signature": "def show_state(self, probs, n_qubits, title)"}, {"kind": "method", "line": 707, "name": "run_quiz", "signature": "def run_quiz(self, quiz)"}, {"kind": "method", "line": 728, "name": "lessons", "signature": "def lessons(self)"}, {"kind": "method", "line": 733, "name": "_demo_superposition", "signature": "def _demo_superposition(self)"}, {"kind": "method", "line": 744, "name": "_demo_rotation", "signature": "def _demo_rotation(self)"}, {"kind": "method", "line": 766, "name": "_demo_measurement", "signature": "def _demo_measurement(self)"}, {"kind": "method", "line": 779, "name": "_demo_bell", "signature": "def _demo_bell(self)"}, {"kind": "method", "line": 792, "name": "_demo_ghz_w", "signature": "def _demo_ghz_w(self)"}, {"kind": "method", "line": 799, "name": "_demo_grover", "signature": "def _demo_grover(self)"}, {"kind": "method", "line": 815, "name": "_demo_molecule", "signature": "def _demo_molecule(self)"}, {"kind": "method", "line": 821, "name": "_vqe", "signature": "def _vqe(self)"}, {"kind": "method", "line": 826, "name": "_demo_h2_clouds", "signature": "def _demo_h2_clouds(self)"}, {"kind": "method", "line": 835, "name": "_demo_vqe_ansatz", "signature": "def _demo_vqe_ansatz(self)"}, {"kind": "method", "line": 840, "name": "_demo_vqe_landscape", "signature": "def _demo_vqe_landscape(self)"}, {"kind": "method", "line": 847, "name": "_demo_vqe_live", "signature": "def _demo_vqe_live(self)"}, {"kind": "method", "line": 875, "name": "_lessons_en", "signature": "def _lessons_en(self)"}, {"kind": "method", "line": 1015, "name": "_lessons_es", "signature": "def _lessons_es(self)"}, {"kind": "method", "line": 1159, "name": "run_lesson", "signature": "def run_lesson(self, lesson)"}, {"kind": "method", "line": 1175, "name": "lessons_menu", "signature": "def lessons_menu(self)"}, {"kind": "method", "line": 1198, "name": "_rebuild_state", "signature": "def _rebuild_state(self, n_qubits, instructions)"}, {"kind": "method", "line": 1216, "name": "_playground_dashboard", "signature": "def _playground_dashboard(self, n_qubits, state, instructions)"}, {"kind": "method", "line": 1228, "name": "_show_amplitudes", "signature": "def _show_amplitudes(self, state, n_qubits)"}, {"kind": "method", "line": 1249, "name": "playground", "signature": "def playground(self)"}, {"kind": "method", "line": 1346, "name": "_molecule_card", "signature": "def _molecule_card(self, mol)"}, {"kind": "method", "line": 1365, "name": "molecule_explorer", "signature": "def molecule_explorer(self)"}, {"kind": "method", "line": 1383, "name": "orbital_viewer", "signature": "def orbital_viewer(self)"}, {"kind": "method", "line": 1403, "name": "chemistry_menu", "signature": "def chemistry_menu(self)"}, {"kind": "method", "line": 1422, "name": "glossary", "signature": "def glossary(self)"}, {"kind": "method", "line": 1437, "name": "banner", "signature": "def banner(self)"}, {"kind": "method", "line": 1443, "name": "main_menu", "signature": "def main_menu(self)"}]}, {"doc": "Quantum Simulation Framework - Complete Edition ================================================ Self-contained quantum simulation framework with: - Quantum circuit simulation (Hamiltonian, Schrodinger, Dirac backends) - Atomic orbital visualization with Monte Carlo sampling - Molecular VQE/UCCSD simulation - Relativistic hydrogen with fine structure and Zitterbewegung - Entangled state visualization - Multi-atom orbital support  All configuration via TOML file - no hardcoded values.", "id": "quantum_simulator.py", "kind": "module", "label": "quantum_simulator.py", "language": "py", "sha256": "c44a434f3891288e", "symbol_count": 219, "symbols": [{"kind": "function", "line": 54, "name": "_make_logger", "signature": "def _make_logger(name, level)"}, {"kind": "class", "line": 68, "name": "FrameworkConfig", "signature": "class FrameworkConfig"}, {"kind": "class", "line": 164, "name": "AtomData", "signature": "class AtomData"}, {"kind": "class", "line": 174, "name": "MoleculeData", "signature": "class MoleculeData"}, {"kind": "class", "line": 192, "name": "OrbitalData", "signature": "class OrbitalData"}, {"kind": "class", "line": 200, "name": "ConfigLoader", "signature": "class ConfigLoader"}, {"kind": "class", "line": 288, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"kind": "class", "line": 303, "name": "HamiltonianBackboneNet", "signature": "class HamiltonianBackboneNet(Module)"}, {"kind": "class", "line": 321, "name": "SchrodingerSpectralNet", "signature": "class SchrodingerSpectralNet(Module)"}, {"kind": "class", "line": 341, "name": "DiracSpectralNet", "signature": "class DiracSpectralNet(Module)"}, {"kind": "class", "line": 361, "name": "GammaMatrices", "signature": "class GammaMatrices"}, {"kind": "class", "line": 380, "name": "JointHilbertState", "signature": "class JointHilbertState"}, {"kind": "class", "line": 410, "name": "IPhysicsBackend", "signature": "class IPhysicsBackend(ABC)"}, {"kind": "class", "line": 420, "name": "HamiltonianBackend", "signature": "class HamiltonianBackend(IPhysicsBackend)"}, {"kind": "class", "line": 470, "name": "SchrodingerBackend", "signature": "class SchrodingerBackend(IPhysicsBackend)"}, {"kind": "class", "line": 505, "name": "DiracBackend", "signature": "class DiracBackend(IPhysicsBackend)"}, {"kind": "method", "line": 587, "name": "_single_qubit_unitary", "signature": "def _single_qubit_unitary(state, qubit, u, backend)"}, {"kind": "method", "line": 611, "name": "_two_qubit_unitary", "signature": "def _two_qubit_unitary(state, ctrl, tgt, u4)"}, {"kind": "class", "line": 638, "name": "IQuantumGate", "signature": "class IQuantumGate(ABC)"}, {"kind": "class", "line": 649, "name": "HadamardGate", "signature": "class HadamardGate(IQuantumGate)"}, {"kind": "class", "line": 662, "name": "PauliXGate", "signature": "class PauliXGate(IQuantumGate)"}, {"kind": "class", "line": 674, "name": "PauliYGate", "signature": "class PauliYGate(IQuantumGate)"}, {"kind": "class", "line": 686, "name": "PauliZGate", "signature": "class PauliZGate(IQuantumGate)"}, {"kind": "class", "line": 698, "name": "SGate", "signature": "class SGate(IQuantumGate)"}, {"kind": "class", "line": 710, "name": "TGate", "signature": "class TGate(IQuantumGate)"}, {"kind": "class", "line": 723, "name": "RxGate", "signature": "class RxGate(IQuantumGate)"}, {"kind": "class", "line": 737, "name": "RyGate", "signature": "class RyGate(IQuantumGate)"}, {"kind": "class", "line": 751, "name": "RzGate", "signature": "class RzGate(IQuantumGate)"}, {"kind": "class", "line": 766, "name": "CNOTGate", "signature": "class CNOTGate(IQuantumGate)"}, {"kind": "class", "line": 778, "name": "CZGate", "signature": "class CZGate(IQuantumGate)"}, {"kind": "class", "line": 790, "name": "SWAPGate", "signature": "class SWAPGate(IQuantumGate)"}, {"kind": "class", "line": 802, "name": "ToffoliGate", "signature": "class ToffoliGate(IQuantumGate)"}, {"kind": "class", "line": 832, "name": "CircuitInstruction", "signature": "class CircuitInstruction"}, {"kind": "class", "line": 838, "name": "QuantumCircuit", "signature": "class QuantumCircuit"}, {"kind": "class", "line": 894, "name": "QuantumResult", "signature": "class QuantumResult"}, {"kind": "class", "line": 908, "name": "PotentialGenerator", "signature": "class PotentialGenerator"}, {"kind": "method", "line": 949, "name": "_solve_eigenstate", "signature": "def _solve_eigenstate(config, potential, n)"}, {"kind": "method", "line": 965, "name": "_build_basis_amplitude", "signature": "def _build_basis_amplitude(config, basis_idx)"}, {"kind": "class", "line": 972, "name": "JointStateFactory", "signature": "class JointStateFactory"}, {"kind": "class", "line": 999, "name": "QuantumComputer", "signature": "class QuantumComputer"}, {"kind": "class", "line": 1043, "name": "WavefunctionCalculator", "signature": "class WavefunctionCalculator"}, {"kind": "class", "line": 1080, "name": "MonteCarloSampler", "signature": "class MonteCarloSampler"}, {"kind": "class", "line": 1159, "name": "DiracHydrogenAtom", "signature": "class DiracHydrogenAtom"}, {"kind": "class", "line": 1200, "name": "ZitterbewegungSimulator", "signature": "class ZitterbewegungSimulator"}, {"kind": "class", "line": 1282, "name": "OrbitalVisualizer", "signature": "class OrbitalVisualizer"}, {"kind": "class", "line": 1343, "name": "EntangledVisualizer", "signature": "class EntangledVisualizer"}, {"kind": "class", "line": 1409, "name": "QuantumSimulationFramework", "signature": "class QuantumSimulationFramework"}, {"kind": "class", "line": 1535, "name": "InteractiveMenu", "signature": "class InteractiveMenu"}, {"kind": "method", "line": 1868, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 112, "name": "from_toml", "signature": "def from_toml(cls, toml_path)"}, {"kind": "method", "line": 201, "name": "__init__", "signature": "def __init__(self, config_path)"}, {"kind": "method", "line": 209, "name": "_find_config", "signature": "def _find_config(self)"}, {"kind": "method", "line": 219, "name": "_load", "signature": "def _load(self)"}, {"kind": "method", "line": 229, "name": "_load_defaults", "signature": "def _load_defaults(self)"}, {"kind": "method", "line": 236, "name": "_parse_atoms", "signature": "def _parse_atoms(self)"}, {"kind": "method", "line": 241, "name": "_parse_molecules", "signature": "def _parse_molecules(self)"}, {"kind": "method", "line": 246, "name": "_parse_orbitals", "signature": "def _parse_orbitals(self)"}, {"kind": "method", "line": 251, "name": "get_atom", "signature": "def get_atom(self, symbol)"}, {"kind": "method", "line": 259, "name": "get_molecule", "signature": "def get_molecule(self, name)"}, {"kind": "method", "line": 267, "name": "get_orbital", "signature": "def get_orbital(self, name)"}, {"kind": "method", "line": 276, "name": "atoms", "signature": "def atoms(self)"}, {"kind": "method", "line": 280, "name": "molecules", "signature": "def molecules(self)"}, {"kind": "method", "line": 284, "name": "orbitals", "signature": "def orbitals(self)"}, {"kind": "method", "line": 289, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"kind": "method", "line": 295, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 304, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers)"}, {"kind": "method", "line": 310, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 322, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)"}, {"kind": "method", "line": 330, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 342, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)"}, {"kind": "method", "line": 350, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 362, "name": "__init__", "signature": "def __init__(self, representation, device)"}, {"kind": "method", "line": 381, "name": "__init__", "signature": "def __init__(self, amplitudes, n_qubits)"}, {"kind": "method", "line": 388, "name": "normalize_", "signature": "def normalize_(self)"}, {"kind": "method", "line": 393, "name": "probabilities", "signature": "def probabilities(self)"}, {"kind": "method", "line": 397, "name": "entropy", "signature": "def entropy(self)"}, {"kind": "method", "line": 402, "name": "most_probable_bitstring", "signature": "def most_probable_bitstring(self)"}, {"kind": "method", "line": 406, "name": "clone", "signature": "def clone(self)"}, {"kind": "method", "line": 412, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"kind": "method", "line": 416, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 421, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 429, "name": "_load", "signature": "def _load(self)"}, {"kind": "method", "line": 443, "name": "_precompute_laplacian", "signature": "def _precompute_laplacian(self)"}, {"kind": "method", "line": 450, "name": "_apply_h", "signature": "def _apply_h(self, field)"}, {"kind": "method", "line": 458, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"kind": "method", "line": 465, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 471, "name": "__init__", "signature": "def __init__(self, config, hamiltonian)"}, {"kind": "method", "line": 478, "name": "_load", "signature": "def _load(self)"}, {"kind": "method", "line": 493, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"kind": "method", "line": 501, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 506, "name": "__init__", "signature": "def __init__(self, config, hamiltonian)"}, {"kind": "method", "line": 515, "name": "_load", "signature": "def _load(self)"}, {"kind": "method", "line": 530, "name": "_precompute_dirac", "signature": "def _precompute_dirac(self)"}, {"kind": "method", "line": 537, "name": "_pack", "signature": "def _pack(self, amp)"}, {"kind": "method", "line": 547, "name": "_unpack", "signature": "def _unpack(self, spinor)"}, {"kind": "method", "line": 553, "name": "_analytical_dirac", "signature": "def _analytical_dirac(self, spinor)"}, {"kind": "method", "line": 564, "name": "evolve_amplitude", "signature": "def evolve_amplitude(self, amp, dt)"}, {"kind": "method", "line": 577, "name": "apply_phase", "signature": "def apply_phase(self, amp, phase_angle)"}, {"kind": "method", "line": 580, "name": "evolve_spinor", "signature": "def evolve_spinor(self, spinor, dt)"}, {"kind": "method", "line": 641, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 645, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 651, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 654, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 664, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 667, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 676, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 679, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 688, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 691, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 700, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 703, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 712, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 715, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 725, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 728, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 739, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 742, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 753, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 756, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 768, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 771, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 780, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 783, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 792, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 795, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 804, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 807, "name": "apply", "signature": "def apply(self, state, backend, targets, params)"}, {"kind": "method", "line": 839, "name": "__init__", "signature": "def __init__(self, n_qubits)"}, {"kind": "method", "line": 843, "name": "_append", "signature": "def _append(self, gate_name, targets, params)"}, {"kind": "method", "line": 846, "name": "h", "signature": "def h(self, qubit)"}, {"kind": "method", "line": 849, "name": "x", "signature": "def x(self, qubit)"}, {"kind": "method", "line": 852, "name": "y", "signature": "def y(self, qubit)"}, {"kind": "method", "line": 855, "name": "z", "signature": "def z(self, qubit)"}, {"kind": "method", "line": 858, "name": "s", "signature": "def s(self, qubit)"}, {"kind": "method", "line": 861, "name": "t", "signature": "def t(self, qubit)"}, {"kind": "method", "line": 864, "name": "rx", "signature": "def rx(self, qubit, theta)"}, {"kind": "method", "line": 867, "name": "ry", "signature": "def ry(self, qubit, theta)"}, {"kind": "method", "line": 870, "name": "rz", "signature": "def rz(self, qubit, theta)"}, {"kind": "method", "line": 873, "name": "cnot", "signature": "def cnot(self, control, target)"}, {"kind": "method", "line": 876, "name": "cz", "signature": "def cz(self, control, target)"}, {"kind": "method", "line": 879, "name": "swap", "signature": "def swap(self, qubit1, qubit2)"}, {"kind": "method", "line": 882, "name": "ccx", "signature": "def ccx(self, ctrl0, ctrl1, target)"}, {"kind": "method", "line": 885, "name": "run", "signature": "def run(self, state, backend)"}, {"kind": "method", "line": 895, "name": "__init__", "signature": "def __init__(self, state)"}, {"kind": "method", "line": 898, "name": "entropy", "signature": "def entropy(self)"}, {"kind": "method", "line": 901, "name": "most_probable_bitstring", "signature": "def most_probable_bitstring(self)"}, {"kind": "method", "line": 904, "name": "probabilities", "signature": "def probabilities(self)"}, {"kind": "method", "line": 909, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 913, "name": "_grid", "signature": "def _grid(self)"}, {"kind": "method", "line": 918, "name": "harmonic", "signature": "def harmonic(self)"}, {"kind": "method", "line": 923, "name": "double_well", "signature": "def double_well(self)"}, {"kind": "method", "line": 929, "name": "coulomb", "signature": "def coulomb(self)"}, {"kind": "method", "line": 935, "name": "periodic_lattice", "signature": "def periodic_lattice(self)"}, {"kind": "method", "line": 939, "name": "mixed", "signature": "def mixed(self, seed)"}, {"kind": "method", "line": 973, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 976, "name": "_empty", "signature": "def _empty(self, n_qubits)"}, {"kind": "method", "line": 979, "name": "all_zeros", "signature": "def all_zeros(self, n_qubits)"}, {"kind": "method", "line": 986, "name": "basis_state", "signature": "def basis_state(self, n_qubits, k)"}, {"kind": "method", "line": 995, "name": "from_bitstring", "signature": "def from_bitstring(self, bitstring)"}, {"kind": "method", "line": 1000, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1011, "name": "create_circuit", "signature": "def create_circuit(self, n_qubits)"}, {"kind": "method", "line": 1014, "name": "run_circuit", "signature": "def run_circuit(self, circuit, initial_state, backend)"}, {"kind": "method", "line": 1021, "name": "bell_state", "signature": "def bell_state(self, backend)"}, {"kind": "method", "line": 1027, "name": "ghz_state", "signature": "def ghz_state(self, n_qubits, backend)"}, {"kind": "method", "line": 1035, "name": "factory", "signature": "def factory(self)"}, {"kind": "method", "line": 1039, "name": "backends", "signature": "def backends(self)"}, {"kind": "method", "line": 1044, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1048, "name": "radial_wavefunction", "signature": "def radial_wavefunction(n, l, r)"}, {"kind": "method", "line": 1060, "name": "spherical_harmonic_real", "signature": "def spherical_harmonic_real(l, m, theta, phi)"}, {"kind": "method", "line": 1071, "name": "psi_3d", "signature": "def psi_3d(self, n, l, m, r, theta, phi)"}, {"kind": "method", "line": 1076, "name": "energy_analytical", "signature": "def energy_analytical(self, n)"}, {"kind": "method", "line": 1081, "name": "__init__", "signature": "def __init__(self, config, wavefunction_calc)"}, {"kind": "method", "line": 1085, "name": "find_max_probability", "signature": "def find_max_probability(self, n, l, m)"}, {"kind": "method", "line": 1109, "name": "sample_orbital", "signature": "def sample_orbital(self, n, l, m, num_samples, Z)"}, {"kind": "method", "line": 1146, "name": "sample_entangled_state", "signature": "def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples)"}, {"kind": "method", "line": 1160, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1165, "name": "energy_level_dirac", "signature": "def energy_level_dirac(self, n, kappa, Z)"}, {"kind": "method", "line": 1173, "name": "energy_schrodinger", "signature": "def energy_schrodinger(self, n, Z)"}, {"kind": "method", "line": 1176, "name": "fine_structure_splitting", "signature": "def fine_structure_splitting(self, n, l, Z)"}, {"kind": "method", "line": 1184, "name": "energy_spectrum", "signature": "def energy_spectrum(self, n_max, Z)"}, {"kind": "method", "line": 1201, "name": "__init__", "signature": "def __init__(self, config, dirac_backend)"}, {"kind": "method", "line": 1206, "name": "create_gaussian_wave_packet", "signature": "def create_gaussian_wave_packet(self, sigma, momentum)"}, {"kind": "method", "line": 1224, "name": "compute_position_expectation", "signature": "def compute_position_expectation(self, spinor)"}, {"kind": "method", "line": 1235, "name": "compute_velocity_expectation", "signature": "def compute_velocity_expectation(self, spinor)"}, {"kind": "method", "line": 1246, "name": "simulate", "signature": "def simulate(self, duration, dt, sigma)"}, {"kind": "method", "line": 1283, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1286, "name": "visualize", "signature": "def visualize(self, data, save_path, title_suffix)"}, {"kind": "method", "line": 1344, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1347, "name": "visualize", "signature": "def visualize(self, data, quantum_result, save_path)"}, {"kind": "method", "line": 1410, "name": "__init__", "signature": "def __init__(self, config_path)"}, {"kind": "method", "line": 1422, "name": "_ensure_output_dir", "signature": "def _ensure_output_dir(self)"}, {"kind": "method", "line": 1425, "name": "list_available_atoms", "signature": "def list_available_atoms(self)"}, {"kind": "method", "line": 1428, "name": "list_available_molecules", "signature": "def list_available_molecules(self)"}, {"kind": "method", "line": 1431, "name": "list_available_orbitals", "signature": "def list_available_orbitals(self)"}, {"kind": "method", "line": 1434, "name": "get_atom", "signature": "def get_atom(self, symbol)"}, {"kind": "method", "line": 1437, "name": "get_molecule", "signature": "def get_molecule(self, name)"}, {"kind": "method", "line": 1440, "name": "get_orbital", "signature": "def get_orbital(self, name)"}, {"kind": "method", "line": 1443, "name": "run_quantum_circuit", "signature": "def run_quantum_circuit(self, circuit, backend)"}, {"kind": "method", "line": 1446, "name": "visualize_orbital", "signature": "def visualize_orbital(self, orbital_name, num_samples, save, Z, title_suffix)"}, {"kind": "method", "line": 1458, "name": "visualize_atom_orbitals", "signature": "def visualize_atom_orbitals(self, atom_symbol, num_samples, save)"}, {"kind": "method", "line": 1482, "name": "visualize_entangled_state", "signature": "def visualize_entangled_state(self, orbital1, orbital2, num_samples, save)"}, {"kind": "method", "line": 1496, "name": "compute_relativistic_energy", "signature": "def compute_relativistic_energy(self, n, l, Z)"}, {"kind": "method", "line": 1499, "name": "compute_energy_spectrum", "signature": "def compute_energy_spectrum(self, n_max, Z)"}, {"kind": "method", "line": 1502, "name": "run_zitterbewegung_simulation", "signature": "def run_zitterbewegung_simulation(self, duration, dt, sigma)"}, {"kind": "method", "line": 1505, "name": "run_all_demonstrations", "signature": "def run_all_demonstrations(self, num_samples)"}, {"kind": "method", "line": 1536, "name": "__init__", "signature": "def __init__(self, framework)"}, {"kind": "method", "line": 1540, "name": "display_header", "signature": "def display_header(self)"}, {"kind": "method", "line": 1548, "name": "display_main_menu", "signature": "def display_main_menu(self)"}, {"kind": "method", "line": 1564, "name": "get_user_choice", "signature": "def get_user_choice(self, prompt)"}, {"kind": "method", "line": 1570, "name": "_display_quantum_result", "signature": "def _display_quantum_result(self, result, n_qubits)"}, {"kind": "method", "line": 1578, "name": "orbital_menu", "signature": "def orbital_menu(self)"}, {"kind": "method", "line": 1602, "name": "atom_orbital_menu", "signature": "def atom_orbital_menu(self)"}, {"kind": "method", "line": 1622, "name": "entangled_menu", "signature": "def entangled_menu(self)"}, {"kind": "method", "line": 1651, "name": "quantum_circuit_menu", "signature": "def quantum_circuit_menu(self)"}, {"kind": "method", "line": 1737, "name": "relativistic_menu", "signature": "def relativistic_menu(self)"}, {"kind": "method", "line": 1777, "name": "zitterbewegung_menu", "signature": "def zitterbewegung_menu(self)"}, {"kind": "method", "line": 1796, "name": "molecular_menu", "signature": "def molecular_menu(self)"}, {"kind": "method", "line": 1820, "name": "atomic_menu", "signature": "def atomic_menu(self)"}, {"kind": "method", "line": 1840, "name": "run", "signature": "def run(self)"}]}, {"doc": "quantum_visualizer.py - Production Quantum State Visualizer ============================================================ Brutal real-time visualization of quantum state evolution using trained neural network backends (Hamiltonian, Schrodinger, Dirac).  Imports and uses existing quantum_computer.py, molecular_sim.py, and advanced_experiments.py infrastructure.  Author: Gris Iscomeback License: AGPL v3", "id": "quantum_visualizer.py", "kind": "module", "label": "quantum_visualizer.py", "language": "py", "sha256": "204c7ff3e9fc697b", "symbol_count": 54, "symbols": [{"kind": "function", "line": 93, "name": "_make_logger", "signature": "def _make_logger(name)"}, {"kind": "class", "line": 109, "name": "VisualizerConfig", "signature": "class VisualizerConfig"}, {"kind": "class", "line": 157, "name": "QuantumStateSnapshot", "signature": "class QuantumStateSnapshot"}, {"kind": "class", "line": 170, "name": "BackendComparisonResult", "signature": "class BackendComparisonResult"}, {"kind": "class", "line": 181, "name": "VisualizationResult", "signature": "class VisualizationResult"}, {"kind": "class", "line": 190, "name": "IVisualizationComponent", "signature": "class IVisualizationComponent(ABC)"}, {"kind": "class", "line": 196, "name": "ProbabilityBarRenderer", "signature": "class ProbabilityBarRenderer(IVisualizationComponent)"}, {"kind": "class", "line": 227, "name": "BlochSphereRenderer", "signature": "class BlochSphereRenderer(IVisualizationComponent)"}, {"kind": "class", "line": 258, "name": "PhasePlotRenderer", "signature": "class PhasePlotRenderer(IVisualizationComponent)"}, {"kind": "class", "line": 291, "name": "EntropyPlotRenderer", "signature": "class EntropyPlotRenderer(IVisualizationComponent)"}, {"kind": "class", "line": 313, "name": "BackendComparisonRenderer", "signature": "class BackendComparisonRenderer(IVisualizationComponent)"}, {"kind": "class", "line": 341, "name": "QuantumStateAnalyzer", "signature": "class QuantumStateAnalyzer"}, {"kind": "class", "line": 397, "name": "CircuitExecutor", "signature": "class CircuitExecutor"}, {"kind": "class", "line": 457, "name": "StandardCircuits", "signature": "class StandardCircuits"}, {"kind": "class", "line": 528, "name": "FigureBuilder", "signature": "class FigureBuilder"}, {"kind": "class", "line": 612, "name": "QuantumVisualizer", "signature": "class QuantumVisualizer"}, {"kind": "method", "line": 856, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 192, "name": "render", "signature": "def render(self, data, axes, config)"}, {"kind": "method", "line": 197, "name": "render", "signature": "def render(self, data, axes, config)"}, {"kind": "method", "line": 218, "name": "_get_colors", "signature": "def _get_colors(self, probs, config)"}, {"kind": "method", "line": 228, "name": "render", "signature": "def render(self, data, axes, config)"}, {"kind": "method", "line": 259, "name": "render", "signature": "def render(self, data, axes, config)"}, {"kind": "method", "line": 292, "name": "render", "signature": "def render(self, snapshots, axes, config)"}, {"kind": "method", "line": 314, "name": "render", "signature": "def render(self, results, axes, config)"}, {"kind": "method", "line": 342, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 345, "name": "compute_probabilities", "signature": "def compute_probabilities(self, state)"}, {"kind": "method", "line": 349, "name": "compute_phases", "signature": "def compute_phases(self, state)"}, {"kind": "method", "line": 360, "name": "compute_entropy", "signature": "def compute_entropy(self, probs)"}, {"kind": "method", "line": 367, "name": "compute_bloch_vectors", "signature": "def compute_bloch_vectors(self, state)"}, {"kind": "method", "line": 374, "name": "create_snapshot", "signature": "def create_snapshot(self, state, step, gate_name)"}, {"kind": "method", "line": 398, "name": "__init__", "signature": "def __init__(self, qc, config)"}, {"kind": "method", "line": 403, "name": "execute_sequence", "signature": "def execute_sequence(self, gates, n_qubits, backend_name)"}, {"kind": "method", "line": 423, "name": "compare_backends", "signature": "def compare_backends(self, gates, n_qubits, reference_backend)"}, {"kind": "method", "line": 459, "name": "bell_state", "signature": "def bell_state()"}, {"kind": "method", "line": 466, "name": "ghz_state", "signature": "def ghz_state(n_qubits)"}, {"kind": "method", "line": 473, "name": "qft", "signature": "def qft(n_qubits)"}, {"kind": "method", "line": 488, "name": "grover_oracle", "signature": "def grover_oracle(n_qubits, marked)"}, {"kind": "method", "line": 501, "name": "grover_diffusion", "signature": "def grover_diffusion(n_qubits)"}, {"kind": "method", "line": 514, "name": "custom_sequence", "signature": "def custom_sequence(sequence)"}, {"kind": "method", "line": 529, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 537, "name": "build_evolution_figure", "signature": "def build_evolution_figure(self, snapshots, backend_results)"}, {"kind": "method", "line": 563, "name": "build_summary_figure", "signature": "def build_summary_figure(self, snapshots, backend_results)"}, {"kind": "method", "line": 594, "name": "_render_backend_fidelity", "signature": "def _render_backend_fidelity(self, results, axes)"}, {"kind": "method", "line": 613, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 620, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 629, "name": "_init_quantum_computer", "signature": "def _init_quantum_computer(self)"}, {"kind": "method", "line": 654, "name": "visualize_bell_state", "signature": "def visualize_bell_state(self)"}, {"kind": "method", "line": 683, "name": "visualize_ghz_state", "signature": "def visualize_ghz_state(self, n_qubits)"}, {"kind": "method", "line": 711, "name": "visualize_qft", "signature": "def visualize_qft(self, n_qubits)"}, {"kind": "method", "line": 739, "name": "visualize_grover", "signature": "def visualize_grover(self, n_qubits, marked_state)"}, {"kind": "method", "line": 778, "name": "visualize_custom_circuit", "signature": "def visualize_custom_circuit(self, gates, n_qubits, name)"}, {"kind": "method", "line": 810, "name": "run_all_visualizations", "signature": "def run_all_visualizations(self)"}, {"kind": "method", "line": 825, "name": "_save_figure", "signature": "def _save_figure(self, fig, name)"}, {"kind": "method", "line": 842, "name": "_print_summary", "signature": "def _print_summary(self, results)"}]}, {"doc": "Dirac Relativistic Hydrogen Visualizer ====================================== Validation suite for Dirac equation grokking via Hamiltonian Topological Crystallization. Extends the Schrodinger visualization architecture to handle relativistic quantum mechanics.  Features: - Relativistic Hydrogen Atom energy levels (Fine Structure) - Zitterbewegung (Trembling Motion) reconstruction - Spin-orbit coupling visualization - 4-component spinor evolution", "id": "relativistic_hydrogen.py", "kind": "module", "label": "relativistic_hydrogen.py", "language": "py", "sha256": "87a769c3724afcf9", "symbol_count": 58, "symbols": [{"kind": "class", "line": 41, "name": "Config", "signature": "class Config"}, {"kind": "class", "line": 95, "name": "LoggerFactory", "signature": "class LoggerFactory"}, {"doc": "Dirac gamma matrices in Dirac (standard) representation.\ngamma^0 = beta, gamma^i = beta * alpha_i", "kind": "class", "line": 113, "name": "GammaMatrices", "signature": "class GammaMatrices"}, {"doc": "Dirac Hamiltonian operator for relativistic quantum mechanics.\nH_Dirac = c * alpha . p + beta * m * c^2 + V(r)\n\nIn atomic units (c = 1/alpha ~ 137):\nH = c * alpha . p + beta * m * c^2 + V", "kind": "class", "line": 199, "name": "DiracHamiltonianOperator", "signature": "class DiracHamiltonianOperator"}, {"kind": "class", "line": 310, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"doc": "Neural network for learning Dirac equation dynamics.\nHandles 4-component spinors with real and imaginary parts (8 channels total).", "kind": "class", "line": 351, "name": "DiracSpectralNetwork", "signature": "class DiracSpectralNetwork(Module)"}, {"doc": "Wrapper to load and use the trained Dirac model.", "kind": "class", "line": 393, "name": "DiracModelWrapper", "signature": "class DiracModelWrapper"}, {"doc": "Relativistic hydrogen atom with Dirac equation.\nComputes energy levels including fine structure.", "kind": "class", "line": 516, "name": "DiracHydrogenAtom", "signature": "class DiracHydrogenAtom"}, {"doc": "Simulates the Zitterbewegung (trembling motion) of a relativistic electron.\n\nIn Dirac theory, the position operator has a term oscillating with frequency\n~ 2mc^2/hbar, which is the interference between positive and negative energy states.\n\n<x(t)> = <x(0)> + (p/m) * t + oscillating term\nThe oscillating term has amplitude ~ hbar/(2mc) ~ 10^-12 m", "kind": "class", "line": 640, "name": "ZitterbewegungSimulator", "signature": "class ZitterbewegungSimulator"}, {"doc": "Calculate relativistic hydrogen wavefunctions.", "kind": "class", "line": 833, "name": "DiracWavefunctionCalculator", "signature": "class DiracWavefunctionCalculator"}, {"doc": "Monte Carlo sampling for relativistic orbital visualization.", "kind": "class", "line": 956, "name": "DiracMonteCarloSampler", "signature": "class DiracMonteCarloSampler"}, {"doc": "Visualization suite for Dirac equation results.", "kind": "class", "line": 1075, "name": "DiracVisualizer", "signature": "class DiracVisualizer"}, {"doc": "Complete validation suite for Dirac equation grokking.", "kind": "class", "line": 1370, "name": "DiracValidationSuite", "signature": "class DiracValidationSuite"}, {"kind": "method", "line": 1613, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 97, "name": "create_logger", "signature": "def create_logger(name, level)"}, {"kind": "method", "line": 118, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 122, "name": "_init_matrices", "signature": "def _init_matrices(self)"}, {"kind": "method", "line": 207, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 215, "name": "_precompute_operators", "signature": "def _precompute_operators(self)"}, {"doc": "Apply Dirac Hamiltonian to 4-component spinor.\n\nArgs:\n    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor\n    potential: Optional scalar potential V(r)\n\nReturns:\n    H * psi with same shape as input", "kind": "method", "line": 223, "name": "apply_dirac_hamiltonian", "signature": "def apply_dirac_hamiltonian(self, spinor, potential)"}, {"doc": "Time evolution of Dirac spinor using first-order split-step.\npsi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi", "kind": "method", "line": 282, "name": "time_evolution", "signature": "def time_evolution(self, spinor, dt, potential)"}, {"kind": "method", "line": 311, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"kind": "method", "line": 322, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 356, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, spinor_components)"}, {"kind": "method", "line": 379, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 397, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 406, "name": "_find_best_checkpoint", "signature": "def _find_best_checkpoint(self)"}, {"kind": "method", "line": 452, "name": "_load_model", "signature": "def _load_model(self)"}, {"doc": "Apply Hamiltonian using analytical operator.\nThe NN model learns spinor evolution, but the Hamiltonian operator\nis applied analytically for physical validation.", "kind": "method", "line": 498, "name": "apply_hamiltonian", "signature": "def apply_hamiltonian(self, spinor, potential)"}, {"doc": "Evolve spinor in time using the analytical Dirac operator.", "kind": "method", "line": 506, "name": "evolve_spinor", "signature": "def evolve_spinor(self, spinor, dt, potential)"}, {"kind": "method", "line": 521, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Exact Dirac energy level for hydrogen-like atom.\n\nE = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)\n\nFor hydrogen (Z=1):\nE = mc^2 * [1 + (alpha^2 / (n - |kappa| + sqrt(kappa^2 - alpha^2)))^2]^(-1/2)\n\nArgs:\n    n: Principal quantum number\n    kappa: Relativistic quantum number (kappa = -(l+1) for j=l+1/2, kappa = l for j=l-1/2)\n\nReturns:\n    Energy in atomic units (relative to m*c^2)", "kind": "method", "line": 526, "name": "energy_level_dirac", "signature": "def energy_level_dirac(self, n, kappa)"}, {"doc": "Calculate fine structure splitting for given n, l.\n\nFine structure includes:\n1. Relativistic correction to kinetic energy\n2. Spin-orbit coupling\n3. Darwin term (for l=0)\n\nReturns energies for j = l+1/2 and j = l-1/2", "kind": "method", "line": 556, "name": "fine_structure_splitting", "signature": "def fine_structure_splitting(self, n, l)"}, {"doc": "Generate relativistic energy spectrum up to n_max.", "kind": "method", "line": 597, "name": "energy_spectrum", "signature": "def energy_spectrum(self, n_max)"}, {"kind": "method", "line": 650, "name": "__init__", "signature": "def __init__(self, config, model_wrapper)"}, {"doc": "Create a Gaussian wave packet for a free particle.\n\nFor Dirac, we need a 4-component spinor that's a superposition\nof positive energy states.", "kind": "method", "line": 657, "name": "create_gaussian_wave_packet", "signature": "def create_gaussian_wave_packet(self, sigma, momentum)"}, {"doc": "Compute expectation value of position operator.\n<x> = <psi| x |psi>", "kind": "method", "line": 702, "name": "compute_position_expectation", "signature": "def compute_position_expectation(self, spinor)"}, {"doc": "Compute expectation value of velocity operator.\nIn Dirac theory, v = c * alpha\n\n<v_x> = c * <psi| alpha_x |psi>", "kind": "method", "line": 724, "name": "compute_velocity_expectation", "signature": "def compute_velocity_expectation(self, spinor)"}, {"doc": "Run Zitterbewegung simulation.\n\nReturns time evolution of position and velocity showing the\noscillatory ZBW term.", "kind": "method", "line": 750, "name": "simulate", "signature": "def simulate(self, duration, dt, sigma)"}, {"kind": "method", "line": 837, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Non-relativistic radial wavefunction for comparison.", "kind": "method", "line": 843, "name": "radial_wavefunction_schrodinger", "signature": "def radial_wavefunction_schrodinger(n, l, r)"}, {"doc": "Relativistic radial wavefunctions for hydrogen.\n\nReturns (f, g) - small and large components.\nFor bound states, the Dirac radial functions are:\nf(r) = sqrt((E+mc^2)/(2E)) * G(r)\ng(r) = sqrt((E-mc^2)/(2E)) * F(r)\n\nSimplified version using Sommerfeld fine-structure formula.", "kind": "method", "line": 853, "name": "radial_wavefunction_dirac", "signature": "def radial_wavefunction_dirac(self, n, kappa, r, Z)"}, {"doc": "Real spherical harmonics.", "kind": "method", "line": 900, "name": "spherical_harmonic_real", "signature": "def spherical_harmonic_real(self, l, m, theta, phi)"}, {"doc": "Spin-angular functions Omega_{kappa,m_j}(theta, phi).\n\nThese couple the orbital and spin degrees of freedom.", "kind": "method", "line": 910, "name": "spin_angular_function", "signature": "def spin_angular_function(self, kappa, m_j, theta, phi)"}, {"kind": "method", "line": 960, "name": "__init__", "signature": "def __init__(self, config, model_wrapper)"}, {"doc": "Sample points from a relativistic hydrogen orbital.", "kind": "method", "line": 966, "name": "sample_orbital", "signature": "def sample_orbital(self, n, l, j, num_samples)"}, {"kind": "method", "line": 1079, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Visualize relativistic orbital.", "kind": "method", "line": 1082, "name": "visualize_orbital", "signature": "def visualize_orbital(self, data, save_path)"}, {"doc": "Visualize relativistic energy spectrum with fine structure.", "kind": "method", "line": 1213, "name": "visualize_energy_spectrum", "signature": "def visualize_energy_spectrum(self, spectrum, save_path)"}, {"doc": "Visualize Zitterbewegung oscillation.", "kind": "method", "line": 1296, "name": "visualize_zitterbewegung", "signature": "def visualize_zitterbewegung(self, zbw_data, save_path)"}, {"kind": "method", "line": 1374, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1398, "name": "print_header", "signature": "def print_header(self)"}, {"doc": "Validate fine structure energy corrections.", "kind": "method", "line": 1419, "name": "validate_fine_structure", "signature": "def validate_fine_structure(self)"}, {"doc": "Validate Zitterbewegung simulation.", "kind": "method", "line": 1477, "name": "validate_zitterbewegung", "signature": "def validate_zitterbewegung(self)"}, {"doc": "Validate complete energy spectrum.", "kind": "method", "line": 1509, "name": "validate_energy_spectrum", "signature": "def validate_energy_spectrum(self)"}, {"doc": "Validate single orbital visualization.", "kind": "method", "line": 1524, "name": "validate_orbital", "signature": "def validate_orbital(self, orbital_name, num_samples)"}, {"doc": "Run complete validation suite.", "kind": "method", "line": 1541, "name": "run_full_validation", "signature": "def run_full_validation(self)"}, {"doc": "Run in interactive mode.", "kind": "method", "line": 1575, "name": "interactive_mode", "signature": "def interactive_mode(self)"}]}, {"doc": "BDD-style integration tests for qc_integration.py and qc_dashboard.py.  Uses pytest with descriptive test names following the pattern: test_<scenario>_<expected_behavior>  Coverage: - qc_integration: IntegrationConfig, GateInstruction, CircuitIR, OpenQasmAdapter, StandardCircuitFactory, IntegrationBridge - qc_dashboard: DashboardConfig, GateItem, VisualisationEngine, SimulatorBackend, DashboardApp  Run: pytest test_qc_integration.py -v pytest test_qc_integration.py -v -k \"qasm or bridge\"", "id": "test_qc_integration.py", "kind": "module", "label": "test_qc_integration.py", "language": "py", "sha256": "4926b9b66934415b", "symbol_count": 87, "symbols": [{"kind": "function", "line": 42, "name": "config", "signature": "def config()"}, {"kind": "function", "line": 47, "name": "bridge", "signature": "def bridge(config)"}, {"kind": "function", "line": 52, "name": "qasm_adapter", "signature": "def qasm_adapter(config)"}, {"kind": "function", "line": 57, "name": "bell_circuit", "signature": "def bell_circuit()"}, {"kind": "function", "line": 62, "name": "ghz_circuit", "signature": "def ghz_circuit()"}, {"doc": "IntegrationConfig: centralised configuration with no hardcoded values.", "kind": "class", "line": 70, "name": "TestIntegrationConfig", "signature": "class TestIntegrationConfig"}, {"doc": "GateInstruction: lightweight quantum gate descriptor.", "kind": "class", "line": 100, "name": "TestGateInstruction", "signature": "class TestGateInstruction"}, {"doc": "CircuitIR: intermediate representation of a quantum circuit.", "kind": "class", "line": 128, "name": "TestCircuitIR", "signature": "class TestCircuitIR"}, {"doc": "OpenQasmAdapter: OpenQASM 2.0 string <-> CircuitIR conversion.", "kind": "class", "line": 165, "name": "TestOpenQasmAdapter", "signature": "class TestOpenQasmAdapter"}, {"doc": "StandardCircuitFactory: builds CircuitIR for common algorithms.", "kind": "class", "line": 260, "name": "TestStandardCircuitFactory", "signature": "class TestStandardCircuitFactory"}, {"doc": "IntegrationBridge: facade for all format conversions.", "kind": "class", "line": 292, "name": "TestIntegrationBridge", "signature": "class TestIntegrationBridge"}, {"doc": "GateItem: circuit builder gate representation.", "kind": "class", "line": 346, "name": "TestGateItem", "signature": "class TestGateItem"}, {"doc": "DashboardConfig: centralised configuration for dashboard.", "kind": "class", "line": 372, "name": "TestDashboardConfig", "signature": "class TestDashboardConfig"}, {"doc": "VisualisationEngine: renders figures from quantum state snapshots.", "kind": "class", "line": 400, "name": "TestVisualisationEngine", "signature": "class TestVisualisationEngine"}, {"doc": "SimulatorBackend: lightweight wrapper around QC framework.", "kind": "class", "line": 441, "name": "TestSimulatorBackend", "signature": "class TestSimulatorBackend"}, {"doc": "SnapshotData: quantum state snapshot for dashboard visualisation.", "kind": "class", "line": 473, "name": "TestSnapshotData", "signature": "class TestSnapshotData"}, {"doc": "Edge cases and error conditions across all modules.", "kind": "class", "line": 517, "name": "TestSadPaths", "signature": "class TestSadPaths"}, {"doc": "End-to-end scenarios combining multiple modules.", "kind": "class", "line": 585, "name": "TestEndToEnd", "signature": "class TestEndToEnd"}, {"kind": "method", "line": 73, "name": "test_default_config_has_supported_gates", "signature": "def test_default_config_has_supported_gates(self, config)"}, {"kind": "method", "line": 79, "name": "test_gate_name_map_is_complete", "signature": "def test_gate_name_map_is_complete(self, config)"}, {"kind": "method", "line": 83, "name": "test_reverse_gate_name_map_is_consistent", "signature": "def test_reverse_gate_name_map_is_consistent(self, config)"}, {"kind": "method", "line": 87, "name": "test_qasm_version_default", "signature": "def test_qasm_version_default(self, config)"}, {"kind": "method", "line": 90, "name": "test_max_qubits_defaults_are_positive", "signature": "def test_max_qubits_defaults_are_positive(self, config)"}, {"kind": "method", "line": 103, "name": "test_create_single_qubit_gate", "signature": "def test_create_single_qubit_gate(self)"}, {"kind": "method", "line": 109, "name": "test_create_two_qubit_gate", "signature": "def test_create_two_qubit_gate(self)"}, {"kind": "method", "line": 114, "name": "test_create_gate_with_params", "signature": "def test_create_gate_with_params(self)"}, {"kind": "method", "line": 118, "name": "test_targets_are_immutable", "signature": "def test_targets_are_immutable(self)"}, {"kind": "method", "line": 131, "name": "test_create_empty_circuit", "signature": "def test_create_empty_circuit(self)"}, {"kind": "method", "line": 136, "name": "test_append_gate", "signature": "def test_append_gate(self)"}, {"kind": "method", "line": 141, "name": "test_append_gate_out_of_range_raises", "signature": "def test_append_gate_out_of_range_raises(self)"}, {"kind": "method", "line": 146, "name": "test_multiple_gates", "signature": "def test_multiple_gates(self)"}, {"kind": "method", "line": 152, "name": "test_repr_includes_qubits_and_gates", "signature": "def test_repr_includes_qubits_and_gates(self)"}, {"kind": "method", "line": 168, "name": "test_export_bell_state_contains_header", "signature": "def test_export_bell_state_contains_header(self, qasm_adapter, bell_circuit)"}, {"kind": "method", "line": 173, "name": "test_export_bell_state_has_qreg_and_creg", "signature": "def test_export_bell_state_has_qreg_and_creg(self, qasm_adapter, bell_circuit)"}, {"kind": "method", "line": 178, "name": "test_export_bell_state_has_gates", "signature": "def test_export_bell_state_has_gates(self, qasm_adapter, bell_circuit)"}, {"kind": "method", "line": 183, "name": "test_export_ghz_state", "signature": "def test_export_ghz_state(self, qasm_adapter, ghz_circuit)"}, {"kind": "method", "line": 189, "name": "test_export_qft_has_swap", "signature": "def test_export_qft_has_swap(self, qasm_adapter, config)"}, {"kind": "method", "line": 194, "name": "test_export_parametric_gate", "signature": "def test_export_parametric_gate(self, qasm_adapter)"}, {"kind": "method", "line": 201, "name": "test_export_exceeds_max_qubits_raises", "signature": "def test_export_exceeds_max_qubits_raises(self, qasm_adapter)"}, {"kind": "method", "line": 206, "name": "test_import_bell_state_roundtrip", "signature": "def test_import_bell_state_roundtrip(self, qasm_adapter, bell_circuit)"}, {"kind": "method", "line": 214, "name": "test_import_ghz_roundtrip", "signature": "def test_import_ghz_roundtrip(self, qasm_adapter, ghz_circuit)"}, {"kind": "method", "line": 220, "name": "test_import_from_standard_qasm_string", "signature": "def test_import_from_standard_qasm_string(self, qasm_adapter)"}, {"kind": "method", "line": 234, "name": "test_import_with_parametric_gates", "signature": "def test_import_with_parametric_gates(self, qasm_adapter)"}, {"kind": "method", "line": 247, "name": "test_import_empty_qasm_returns_zero_qubit_circuit", "signature": "def test_import_empty_qasm_returns_zero_qubit_circuit(self, qasm_adapter)"}, {"kind": "method", "line": 263, "name": "test_bell_state_has_two_gates", "signature": "def test_bell_state_has_two_gates(self)"}, {"kind": "method", "line": 269, "name": "test_bell_state_has_two_qubits", "signature": "def test_bell_state_has_two_qubits(self)"}, {"kind": "method", "line": 273, "name": "test_ghz_state", "signature": "def test_ghz_state(self)"}, {"kind": "method", "line": 278, "name": "test_qft_three_qubits", "signature": "def test_qft_three_qubits(self)"}, {"kind": "method", "line": 283, "name": "test_grover_iterations", "signature": "def test_grover_iterations(self)"}, {"kind": "method", "line": 295, "name": "test_export_qasm_returns_string", "signature": "def test_export_qasm_returns_string(self, bridge, bell_circuit)"}, {"kind": "method", "line": 300, "name": "test_import_qasm_roundtrip", "signature": "def test_import_qasm_roundtrip(self, bridge, bell_circuit)"}, {"kind": "method", "line": 306, "name": "test_full_openqasm_roundtrip_bell", "signature": "def test_full_openqasm_roundtrip_bell(self, bridge)"}, {"kind": "method", "line": 313, "name": "test_full_openqasm_roundtrip_ghz", "signature": "def test_full_openqasm_roundtrip_ghz(self, bridge)"}, {"kind": "method", "line": 320, "name": "test_full_openqasm_roundtrip_qft", "signature": "def test_full_openqasm_roundtrip_qft(self, bridge)"}, {"kind": "method", "line": 327, "name": "test_qiskit_not_available_by_default", "signature": "def test_qiskit_not_available_by_default(self, bridge)"}, {"kind": "method", "line": 332, "name": "test_pennylane_not_available_by_default", "signature": "def test_pennylane_not_available_by_default(self, bridge)"}, {"kind": "method", "line": 337, "name": "test_export_qasm_custom_qreg_name", "signature": "def test_export_qasm_custom_qreg_name(self, bridge, bell_circuit)"}, {"kind": "method", "line": 349, "name": "test_create_single_qubit_gate", "signature": "def test_create_single_qubit_gate(self)"}, {"kind": "method", "line": 357, "name": "test_create_two_qubit_gate", "signature": "def test_create_two_qubit_gate(self)"}, {"kind": "method", "line": 362, "name": "test_create_parametric_gate", "signature": "def test_create_parametric_gate(self)"}, {"kind": "method", "line": 375, "name": "test_default_values", "signature": "def test_default_values(self)"}, {"kind": "method", "line": 382, "name": "test_gate_list_includes_standard_gates", "signature": "def test_gate_list_includes_standard_gates(self)"}, {"kind": "method", "line": 389, "name": "test_qasm_initial_contains_header", "signature": "def test_qasm_initial_contains_header(self)"}, {"kind": "method", "line": 403, "name": "test_engine_available_with_matplotlib", "signature": "def test_engine_available_with_matplotlib(self)"}, {"kind": "method", "line": 415, "name": "test_render_full_dashboard_returns_bytes", "signature": "def test_render_full_dashboard_returns_bytes(self)"}, {"kind": "method", "line": 444, "name": "test_synthetic_execute_returns_snapshots", "signature": "def test_synthetic_execute_returns_snapshots(self)"}, {"kind": "method", "line": 457, "name": "test_empty_circuit_returns_init_snapshot", "signature": "def test_empty_circuit_returns_init_snapshot(self)"}, {"kind": "method", "line": 476, "name": "test_create_snapshot", "signature": "def test_create_snapshot(self)"}, {"kind": "method", "line": 489, "name": "test_entropy_updates", "signature": "def test_entropy_updates(self)"}, {"kind": "method", "line": 500, "name": "test_probabilities_normalized", "signature": "def test_probabilities_normalized(self)"}, {"kind": "method", "line": 520, "name": "test_gate_instruction_empty_targets", "signature": "def test_gate_instruction_empty_targets(self)"}, {"kind": "method", "line": 524, "name": "test_circuit_ir_append_negative_qubit_raises", "signature": "def test_circuit_ir_append_negative_qubit_raises(self)"}, {"kind": "method", "line": 529, "name": "test_openqasm_import_empty_string", "signature": "def test_openqasm_import_empty_string(self, qasm_adapter)"}, {"kind": "method", "line": 534, "name": "test_openqasm_import_garbage_string", "signature": "def test_openqasm_import_garbage_string(self, qasm_adapter)"}, {"kind": "method", "line": 538, "name": "test_openqasm_export_zero_qubit_circuit", "signature": "def test_openqasm_export_zero_qubit_circuit(self, qasm_adapter)"}, {"kind": "method", "line": 543, "name": "test_standard_circuit_factory_qft_one_qubit", "signature": "def test_standard_circuit_factory_qft_one_qubit(self)"}, {"kind": "method", "line": 548, "name": "test_standard_circuit_factory_grover_minimal", "signature": "def test_standard_circuit_factory_grover_minimal(self)"}, {"kind": "method", "line": 552, "name": "test_circuit_ir_repr_no_gates", "signature": "def test_circuit_ir_repr_no_gates(self)"}, {"kind": "method", "line": 557, "name": "test_synthetic_snapshot_probabilities_sum_to_one", "signature": "def test_synthetic_snapshot_probabilities_sum_to_one(self)"}, {"kind": "method", "line": 570, "name": "test_visualisation_engine_handles_no_snapshots", "signature": "def test_visualisation_engine_handles_no_snapshots(self)"}, {"kind": "method", "line": 588, "name": "test_build_export_import_qasm_roundtrip", "signature": "def test_build_export_import_qasm_roundtrip(self, bridge)"}, {"kind": "method", "line": 598, "name": "test_qasm_to_circuitir_to_framework_mps", "signature": "def test_qasm_to_circuitir_to_framework_mps(self, bridge)"}, {"kind": "method", "line": 609, "name": "test_ghz_export_qasm_and_reimport_matches", "signature": "def test_ghz_export_qasm_and_reimport_matches(self, bridge)"}, {"kind": "method", "line": 616, "name": "test_qft_circuit_qasm_roundtrip", "signature": "def test_qft_circuit_qasm_roundtrip(self, bridge)"}, {"kind": "method", "line": 622, "name": "test_export_qasm_with_custom_names", "signature": "def test_export_qasm_with_custom_names(self, bridge)"}, {"kind": "method", "line": 628, "name": "test_import_qasm_preserves_gate_order", "signature": "def test_import_qasm_preserves_gate_order(self, bridge)"}, {"kind": "method", "line": 644, "name": "test_full_pipeline_build_export_import_to_mps", "signature": "def test_full_pipeline_build_export_import_to_mps(self, bridge)"}]}, {"doc": "Quantum Framework Test Suite ============================ Comprehensive pytest tests for the MPS quantum simulation framework.  Run with: pytest test_quantum_framework.py -v pytest test_quantum_framework.py -v -k \"bell\"  # Run only bell state tests pytest test_quantum_framework.py -v --tb=short  # Short traceback  Author: Gris Iscomeback License: AGPL v3", "id": "test_quantum_framework.py", "kind": "module", "label": "test_quantum_framework.py", "language": "py", "sha256": "e8e440d9324c2d14", "symbol_count": 49, "symbols": [{"doc": "Create default framework configuration.", "kind": "function", "line": 43, "name": "config", "signature": "def config()"}, {"doc": "Create quantum computer instance.", "kind": "function", "line": 53, "name": "qc", "signature": "def qc(config)"}, {"doc": "Create precision mode configuration.", "kind": "function", "line": 59, "name": "config_precision", "signature": "def config_precision()"}, {"doc": "Tests for FrameworkConfig.", "kind": "class", "line": 72, "name": "TestFrameworkConfig", "signature": "class TestFrameworkConfig"}, {"doc": "Tests for MPSState.", "kind": "class", "line": 95, "name": "TestMPSState", "signature": "class TestMPSState"}, {"doc": "Tests for Bell state preparation.", "kind": "class", "line": 130, "name": "TestBellState", "signature": "class TestBellState"}, {"doc": "Tests for GHZ state preparation.", "kind": "class", "line": 167, "name": "TestGHZState", "signature": "class TestGHZState"}, {"doc": "Tests for W state preparation.", "kind": "class", "line": 209, "name": "TestWState", "signature": "class TestWState"}, {"doc": "Tests for quantum gates.", "kind": "class", "line": 261, "name": "TestQuantumGates", "signature": "class TestQuantumGates"}, {"doc": "Tests for phase coherence and unitarity.", "kind": "class", "line": 338, "name": "TestPhaseCoherence", "signature": "class TestPhaseCoherence"}, {"doc": "Tests for Grover search algorithm.", "kind": "class", "line": 403, "name": "TestGroverAlgorithm", "signature": "class TestGroverAlgorithm"}, {"doc": "Tests for Quantum Fourier Transform.", "kind": "class", "line": 431, "name": "TestQFT", "signature": "class TestQFT"}, {"doc": "Tests for memory scaling.", "kind": "class", "line": 458, "name": "TestMemoryScaling", "signature": "class TestMemoryScaling"}, {"doc": "Tests for precision mode.", "kind": "class", "line": 501, "name": "TestPrecisionMode", "signature": "class TestPrecisionMode"}, {"doc": "Tests for edge cases.", "kind": "class", "line": 525, "name": "TestEdgeCases", "signature": "class TestEdgeCases"}, {"doc": "Test default configuration values.", "kind": "method", "line": 75, "name": "test_default_config", "signature": "def test_default_config(self)"}, {"doc": "Test custom configuration values.", "kind": "method", "line": 83, "name": "test_custom_config", "signature": "def test_custom_config(self)"}, {"doc": "Test state creation.", "kind": "method", "line": 98, "name": "test_create_state", "signature": "def test_create_state(self, config)"}, {"doc": "Test initial |00...0> state.", "kind": "method", "line": 104, "name": "test_initial_state", "signature": "def test_initial_state(self, qc)"}, {"doc": "Test state cloning.", "kind": "method", "line": 113, "name": "test_clone_state", "signature": "def test_clone_state(self, qc)"}, {"doc": "Test Bell state entropy.", "kind": "method", "line": 133, "name": "test_bell_entropy", "signature": "def test_bell_entropy(self, qc)"}, {"doc": "Test Bell state probabilities.", "kind": "method", "line": 141, "name": "test_bell_probabilities", "signature": "def test_bell_probabilities(self, qc)"}, {"doc": "Test Bell state entanglement.", "kind": "method", "line": 154, "name": "test_bell_entanglement", "signature": "def test_bell_entanglement(self, qc)"}, {"doc": "Test GHZ state entropy.", "kind": "method", "line": 170, "name": "test_ghz_entropy", "signature": "def test_ghz_entropy(self, qc)"}, {"doc": "Test GHZ state probabilities.", "kind": "method", "line": 179, "name": "test_ghz_probabilities", "signature": "def test_ghz_probabilities(self, qc)"}, {"doc": "Test GHZ state memory scaling.", "kind": "method", "line": 192, "name": "test_ghz_scaling", "signature": "def test_ghz_scaling(self, qc)"}, {"doc": "Test W state probabilities.", "kind": "method", "line": 212, "name": "test_w_state_probabilities", "signature": "def test_w_state_probabilities(self, qc)"}, {"doc": "Test W state entropy.", "kind": "method", "line": 227, "name": "test_w_state_entropy", "signature": "def test_w_state_entropy(self, qc)"}, {"doc": "Test that W state has non-zero probabilities for single-excitation states.", "kind": "method", "line": 247, "name": "test_w_state_no_zero_probabilities", "signature": "def test_w_state_no_zero_probabilities(self, qc)"}, {"doc": "Test Hadamard gate.", "kind": "method", "line": 264, "name": "test_hadamard_gate", "signature": "def test_hadamard_gate(self, qc)"}, {"doc": "Test Pauli-X gate.", "kind": "method", "line": 275, "name": "test_pauli_x_gate", "signature": "def test_pauli_x_gate(self, qc)"}, {"doc": "Test Pauli-Z gate on |+> state.", "kind": "method", "line": 285, "name": "test_pauli_z_gate", "signature": "def test_pauli_z_gate(self, qc)"}, {"doc": "Test CNOT gate.", "kind": "method", "line": 297, "name": "test_cnot_gate", "signature": "def test_cnot_gate(self, qc)"}, {"doc": "Test SWAP gate.", "kind": "method", "line": 309, "name": "test_swap_gate", "signature": "def test_swap_gate(self, qc)"}, {"doc": "Test rotation gates.", "kind": "method", "line": 321, "name": "test_rotation_gates", "signature": "def test_rotation_gates(self, qc)"}, {"doc": "Test HZH = X identity.", "kind": "method", "line": 341, "name": "test_hzh_equals_x", "signature": "def test_hzh_equals_x(self, qc)"}, {"doc": "Test XX = I identity.", "kind": "method", "line": 354, "name": "test_xx_equals_identity", "signature": "def test_xx_equals_identity(self, qc)"}, {"doc": "Test CNOT CNOT = I identity.", "kind": "method", "line": 366, "name": "test_cnot_cnot_equals_identity", "signature": "def test_cnot_cnot_equals_identity(self, qc)"}, {"doc": "Test that norm is preserved after gates.", "kind": "method", "line": 381, "name": "test_norm_preservation", "signature": "def test_norm_preservation(self, qc)"}, {"doc": "Test Grover search on 3 qubits.", "kind": "method", "line": 406, "name": "test_grover_3_qubits", "signature": "def test_grover_3_qubits(self, qc)"}, {"doc": "Test Grover speedup.", "kind": "method", "line": 417, "name": "test_grover_speedup", "signature": "def test_grover_speedup(self, qc)"}, {"doc": "Test QFT entropy.", "kind": "method", "line": 434, "name": "test_qft_entropy", "signature": "def test_qft_entropy(self, qc)"}, {"doc": "Test that memory scales sub-exponentially with qubits (MPS property).", "kind": "method", "line": 461, "name": "test_memory_linear_scaling", "signature": "def test_memory_linear_scaling(self)"}, {"doc": "Test MPS compression ratio vs full statevector for large n.", "kind": "method", "line": 478, "name": "test_compression_ratio", "signature": "def test_compression_ratio(self)"}, {"doc": "Test precision mode configuration.", "kind": "method", "line": 504, "name": "test_precision_mode_config", "signature": "def test_precision_mode_config(self)"}, {"doc": "Test Bell state in precision mode.", "kind": "method", "line": 510, "name": "test_precision_mode_bell_state", "signature": "def test_precision_mode_bell_state(self, config_precision)"}, {"doc": "Test single qubit operations.", "kind": "method", "line": 528, "name": "test_single_qubit", "signature": "def test_single_qubit(self, qc)"}, {"doc": "Test with large bond dimension.", "kind": "method", "line": 539, "name": "test_large_bond_dimension", "signature": "def test_large_bond_dimension(self)"}, {"doc": "Test empty circuit.", "kind": "method", "line": 549, "name": "test_empty_circuit", "signature": "def test_empty_circuit(self, qc)"}]}, {"doc": "Topological Hilbert Space Compression ====================================== Implements Matrix Product State (MPS) factorization with vacuum-core architecture for scaling quantum simulations beyond 20 qubits.  Architecture: - MPS/TTN factorization reduces O(2^n) to O(n * chi^2) - Vacuum Core projects irrelevant Hilbert subspace to zero - Hybrid backend routes subcircuits to optimal processors - Topological protection via winding numbers and Berry phases  Author: Gris Iscomeback License: AGPL v3", "id": "topological_hilbert_compression2.py", "kind": "module", "label": "topological_hilbert_compression2.py", "language": "py", "sha256": "28c78ab5c56f9538", "symbol_count": 95, "symbols": [{"kind": "class", "line": 47, "name": "HilbertPhase", "signature": "class HilbertPhase(Enum)"}, {"kind": "class", "line": 56, "name": "TopologicalCompressionConfig", "signature": "class TopologicalCompressionConfig"}, {"kind": "class", "line": 85, "name": "ITensorNetwork", "signature": "class ITensorNetwork(ABC)"}, {"doc": "Matrix Product State core tensor A^{[k]}_{i_k} with left and right bond indices.\nShape: (chi_left, d, chi_right) where d=2 for qubits.", "kind": "class", "line": 115, "name": "MPSCore", "signature": "class MPSCore"}, {"doc": "Matrix Product State representation of n-qubit quantum state.\n\n|psi> = sum_{i_1...i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} ... A^{[n]}_{i_n} |i_1...i_n>\n\nMemory: O(n * chi^2 * d) vs O(d^n) for full statevector.\nFor n=30, chi=16: ~30KB vs 8GB for statevector.", "kind": "class", "line": 162, "name": "MPSState", "signature": "class MPSState(ITensorNetwork)"}, {"doc": "Vacuum Core architecture that projects irrelevant Hilbert subspace to zero.\n\nInspired by the HPU-Core achieving 99.996% sparsity:\n- Active core: small subspace carrying quantum information\n- Vacuum: 99%+ of Hilbert space forced to zero by regularization\n- Protection: topological invariants prevent core destruction", "kind": "class", "line": 351, "name": "VacuumCore", "signature": "class VacuumCore"}, {"doc": "Provides topological protection for quantum states via:\n- Winding number monitoring\n- Berry phase calculation\n- Edge state preservation", "kind": "class", "line": 411, "name": "TopologicalProtector", "signature": "class TopologicalProtector"}, {"kind": "class", "line": 452, "name": "HybridBackend", "signature": "class HybridBackend(ABC)"}, {"doc": "Direct tensor backend using JointHilbertState representation.\nLimited to config.max_qubits_direct qubits due to exponential memory.", "kind": "class", "line": 466, "name": "DirectBackend", "signature": "class DirectBackend(HybridBackend)"}, {"doc": "Matrix Product State backend for scalable quantum simulation.\nHandles up to config.max_qubits_mps qubits with sub-exponential memory.", "kind": "class", "line": 515, "name": "MPSBackend", "signature": "class MPSBackend(HybridBackend)"}, {"doc": "Main simulator implementing hybrid architecture for scalable quantum simulation.\n\nArchitecture:\n    - n <= max_qubits_direct: HPU-Core direct tensor (exact wavefunction evolution)\n    - n > max_qubits_direct: MPS with vacuum core compression", "kind": "class", "line": 572, "name": "TopologicalHilbertSimulator", "signature": "class TopologicalHilbertSimulator"}, {"doc": "Validates topological compression on 20-qubit molecular ground state.\n\nTarget: H2O (10 electrons, ~20 qubits)\nMetrics:\n    - Energy error < 1 mHa vs Qiskit VQE\n    - Inference time < 100ms\n    - Memory < 50MB (vs 500MB for direct tensor)", "kind": "class", "line": 723, "name": "Schrodinger20Experiment", "signature": "class Schrodinger20Experiment"}, {"kind": "method", "line": 956, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 80, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 87, "name": "amplitude", "signature": "def amplitude(self, basis_index)"}, {"kind": "method", "line": 91, "name": "apply_single_qubit_gate", "signature": "def apply_single_qubit_gate(self, qubit, gate)"}, {"kind": "method", "line": 95, "name": "apply_two_qubit_gate", "signature": "def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)"}, {"kind": "method", "line": 99, "name": "norm", "signature": "def norm(self)"}, {"kind": "method", "line": 103, "name": "probabilities", "signature": "def probabilities(self)"}, {"kind": "method", "line": 107, "name": "entropy", "signature": "def entropy(self)"}, {"kind": "method", "line": 111, "name": "memory_bytes", "signature": "def memory_bytes(self)"}, {"kind": "method", "line": 121, "name": "__init__", "signature": "def __init__(self, chi_left, chi_right, d, device, dtype)"}, {"kind": "method", "line": 130, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 136, "name": "tensor", "signature": "def tensor(self)"}, {"kind": "method", "line": 142, "name": "tensor", "signature": "def tensor(self, value)"}, {"kind": "method", "line": 147, "name": "left_canonicalize", "signature": "def left_canonicalize(self)"}, {"kind": "method", "line": 154, "name": "right_canonicalize", "signature": "def right_canonicalize(self)"}, {"kind": "method", "line": 172, "name": "__init__", "signature": "def __init__(self, n_qubits, config)"}, {"kind": "method", "line": 181, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 195, "name": "_bond_dimension", "signature": "def _bond_dimension(self, site)"}, {"kind": "method", "line": 198, "name": "amplitude", "signature": "def amplitude(self, basis_index)"}, {"kind": "method", "line": 209, "name": "apply_single_qubit_gate", "signature": "def apply_single_qubit_gate(self, qubit, gate)"}, {"kind": "method", "line": 219, "name": "apply_two_qubit_gate", "signature": "def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)"}, {"kind": "method", "line": 236, "name": "_swap_qubits_in_gate", "signature": "def _swap_qubits_in_gate(self, gate)"}, {"kind": "method", "line": 245, "name": "_apply_adjacent_gate", "signature": "def _apply_adjacent_gate(self, qubit, gate)"}, {"kind": "method", "line": 285, "name": "_apply_nonadjacent_gate", "signature": "def _apply_nonadjacent_gate(self, qubit_a, qubit_b, gate)"}, {"kind": "method", "line": 294, "name": "norm", "signature": "def norm(self)"}, {"kind": "method", "line": 301, "name": "_canonicalize", "signature": "def _canonicalize(self)"}, {"kind": "method", "line": 308, "name": "probabilities", "signature": "def probabilities(self)"}, {"kind": "method", "line": 318, "name": "entropy", "signature": "def entropy(self)"}, {"kind": "method", "line": 329, "name": "memory_bytes", "signature": "def memory_bytes(self)"}, {"kind": "method", "line": 335, "name": "entanglement_entropy", "signature": "def entanglement_entropy(self, cut)"}, {"kind": "method", "line": 361, "name": "__init__", "signature": "def __init__(self, n_qubits, config)"}, {"kind": "method", "line": 370, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 375, "name": "_compute_berry_phases", "signature": "def _compute_berry_phases(self)"}, {"kind": "method", "line": 381, "name": "add_active_state", "signature": "def add_active_state(self, basis_index, winding_number)"}, {"kind": "method", "line": 389, "name": "_compute_winding_number", "signature": "def _compute_winding_number(self, basis_index)"}, {"kind": "method", "line": 394, "name": "is_topologically_protected", "signature": "def is_topologically_protected(self, basis_index)"}, {"kind": "method", "line": 398, "name": "sparsity", "signature": "def sparsity(self)"}, {"kind": "method", "line": 403, "name": "project_to_active", "signature": "def project_to_active(self, state)"}, {"kind": "method", "line": 419, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 424, "name": "compute_winding_number", "signature": "def compute_winding_number(self, state, qubit)"}, {"kind": "method", "line": 433, "name": "compute_berry_phase", "signature": "def compute_berry_phase(self, state, qubit_a, qubit_b)"}, {"kind": "method", "line": 444, "name": "is_protected", "signature": "def is_protected(self, state, vacuum_core)"}, {"kind": "method", "line": 454, "name": "can_handle", "signature": "def can_handle(self, n_qubits)"}, {"kind": "method", "line": 458, "name": "create_state", "signature": "def create_state(self, n_qubits)"}, {"kind": "method", "line": 462, "name": "apply_gate", "signature": "def apply_gate(self, state, gate_name, targets, params)"}, {"kind": "method", "line": 472, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 479, "name": "_load_quantum_computer", "signature": "def _load_quantum_computer(self)"}, {"kind": "method", "line": 497, "name": "can_handle", "signature": "def can_handle(self, n_qubits)"}, {"kind": "method", "line": 500, "name": "create_state", "signature": "def create_state(self, n_qubits)"}, {"kind": "method", "line": 507, "name": "apply_gate", "signature": "def apply_gate(self, state, gate_name, targets, params)"}, {"kind": "method", "line": 521, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 526, "name": "_initialize_gate_cache", "signature": "def _initialize_gate_cache(self)"}, {"kind": "method", "line": 537, "name": "can_handle", "signature": "def can_handle(self, n_qubits)"}, {"kind": "method", "line": 540, "name": "create_state", "signature": "def create_state(self, n_qubits)"}, {"kind": "method", "line": 543, "name": "apply_gate", "signature": "def apply_gate(self, state, gate_name, targets, params)"}, {"kind": "method", "line": 581, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 590, "name": "_select_backend", "signature": "def _select_backend(self, n_qubits, force_mps)"}, {"kind": "method", "line": 602, "name": "create_circuit", "signature": "def create_circuit(self, n_qubits, force_mps)"}, {"kind": "method", "line": 610, "name": "h", "signature": "def h(self, qubit)"}, {"kind": "method", "line": 613, "name": "x", "signature": "def x(self, qubit)"}, {"kind": "method", "line": 616, "name": "y", "signature": "def y(self, qubit)"}, {"kind": "method", "line": 619, "name": "z", "signature": "def z(self, qubit)"}, {"kind": "method", "line": 622, "name": "rx", "signature": "def rx(self, qubit, theta)"}, {"kind": "method", "line": 625, "name": "ry", "signature": "def ry(self, qubit, theta)"}, {"kind": "method", "line": 628, "name": "rz", "signature": "def rz(self, qubit, theta)"}, {"kind": "method", "line": 631, "name": "cnot", "signature": "def cnot(self, control, target)"}, {"kind": "method", "line": 634, "name": "cz", "signature": "def cz(self, control, target)"}, {"kind": "method", "line": 637, "name": "swap", "signature": "def swap(self, qubit_a, qubit_b)"}, {"kind": "method", "line": 640, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 650, "name": "probabilities", "signature": "def probabilities(self)"}, {"kind": "method", "line": 655, "name": "entropy", "signature": "def entropy(self)"}, {"kind": "method", "line": 666, "name": "memory_usage", "signature": "def memory_usage(self)"}, {"kind": "method", "line": 682, "name": "compression_ratio", "signature": "def compression_ratio(self)"}, {"kind": "method", "line": 691, "name": "detect_phase", "signature": "def detect_phase(self)"}, {"kind": "method", "line": 714, "name": "_compute_average_bond_dimension", "signature": "def _compute_average_bond_dimension(self)"}, {"kind": "method", "line": 734, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 739, "name": "run_bell_state", "signature": "def run_bell_state(self, n_qubits, use_mps)"}, {"kind": "method", "line": 758, "name": "run_ghz_state", "signature": "def run_ghz_state(self, n_qubits, use_mps)"}, {"kind": "method", "line": 777, "name": "run_w_state", "signature": "def run_w_state(self, n_qubits, use_mps)"}, {"doc": "Prepare GHZ state directly without SWAP overhead.\n\nGHZ: |00...0> + |11...1> normalized.\nMPS representation:\n- A^{[0]}_{i_0,α_1}: A[0,0,0]=1/√2, A[0,1,1]=1/√2, shape (1,2,2)\n- A^{[k]}_{α_k,i_k,α_{k+1}}: A[0,0,0]=A[1,1,1]=1, shape (2,2,2)\n- A^{[n-1]}_{α_{n-1},i_{n-1},0}: A[0,0,0]=A[1,1,0]=1, shape (2,2,1)", "kind": "method", "line": 798, "name": "prepare_ghz_state", "signature": "def prepare_ghz_state(self, n_qubits, force_mps)"}, {"kind": "method", "line": 844, "name": "run_scaling_benchmark", "signature": "def run_scaling_benchmark(self, max_qubits, use_mps)"}, {"kind": "method", "line": 897, "name": "run_all", "signature": "def run_all(self)"}, {"kind": "method", "line": 909, "name": "_print_summary", "signature": "def _print_summary(self)"}]}], "type": "CodePropertyGraph", "version": "1.0"}
```

---

## Architecture Reference

### PY (29 files)

#### `advanced_experiments.py`
**Path:** `advanced_experiments.py`
**File Doc:** *Advanced Quantum Experiments - Extension Pack ============================================== Extends the existing quantum simulator with:  1. GROVER'S ALGORITHM - Quantum search using quantum_computer.py 2. QED EFFECTS - Lamb shift, anomalous magnetic moment using relativistic_hydrogen.py 3. POLYATOMIC MOLECULES - H2O, NH3 using molecular_sim.py infrastructure  All built on top of existing PyTorch infrastructure.  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `GroverConfig` (line 141) `class GroverConfig` - *Configuration for Grover's algorithm experiments.*
- `GroverOracle` (line 159) `class GroverOracle` - *Oracle for Grover's algorithm.
Marks the target state by applying a phase flip.

Uses the existing MCZGate from quantum_computer.py for multi-controlled Z.*
- `GroverDiffusionOperator` (line 194) `class GroverDiffusionOperator` - *Diffusion operator (Grover diffusion / inversion about mean).

D = 2|s><s| - I where |s> = H^⊗n |0>

Implemented using the existing gate infrastructure.*
- `GroverSearch` (line 240) `class GroverSearch` - *Complete Grover's algorithm implementation using existing quantum_computer.py infrastructure.*
- `QEDConfig` (line 404) `class QEDConfig` - *Configuration for QED effects calculations.*
- `LambShiftCalculator` (line 430) `class LambShiftCalculator` - *Calculates the Lamb shift using Bethe's formula and more accurate methods.

The Lamb shift is the small difference between 2s_{1/2} and 2p_{1/2} energy levels
due to QED effects (vacuum fluctuations and self-energy).

Uses the existing Dirac infrastructure from relativistic_hydrogen.py*
- `AnomalousMagneticMoment` (line 595) `class AnomalousMagneticMoment` - *Calculates the electron's anomalous magnetic moment (g-2).

The electron g-factor is slightly different from 2 due to QED effects:
g = 2(1 + a_e) where a_e = α/(2π) + higher-order terms

Uses the existing Dirac infrastructure for baseline calculations.*
- `QEDEffectsExperiment` (line 722) `class QEDEffectsExperiment` - *Complete QED effects experiment combining Lamb shift and g-2.
Uses existing relativistic_hydrogen.py infrastructure.*
- `PolyatomicMoleculeData` (line 831) `class PolyatomicMoleculeData` - *Data for polyatomic molecules.*
- `MoleculeBuilder` (line 849) `class MoleculeBuilder` - *Build molecule data for VQE calculations.
Uses existing molecular_sim.py infrastructure.*
- `PolyatomicVQE` (line 970) `class PolyatomicVQE` - *VQE solver for polyatomic molecules.
Uses existing molecular_sim.py infrastructure.*
- `PolyatomicExperiment` (line 1105) `class PolyatomicExperiment` - *Complete polyatomic molecule experiment.*
- `AdvancedExperimentRunner` (line 1223) `class AdvancedExperimentRunner` - *Runs all three advanced experiments.*

**Functions:**
- `_make_logger` (line 121) `def _make_logger(name, level)`

**Methods:**
- `main` (line 1304) `def main()` - *Main entry point.*
- `__init__` (line 167) `def __init__(self, n_qubits, marked_state)`
- `_validate` (line 172) `def _validate(self)`
- `apply` (line 176) `def apply(self, state, backend)` - *Apply oracle: flip phase of marked state.
|x> -> (-1)^{f(x)} |x> where f(x)=1 only for marked state.

Uses amplitude-level phase manipulation for exact implementation.*
- `__init__` (line 203) `def __init__(self, n_qubits)`
- `apply` (line 206) `def apply(self, state, backend)` - *Apply diffusion operator using Hadamard + Oracle on |0> + Hadamard.

D = H^⊗n (2|0><0| - I) H^⊗n*
- `__init__` (line 245) `def __init__(self, config)`
- `_calculate_entropy` (line 266) `def _calculate_entropy(self, probs)` - *Calculate Shannon entropy from probability distribution.
H = -sum(p * log2(p))*
- `_init_quantum_computer` (line 278) `def _init_quantum_computer(self)` - *Initialize quantum computer using existing quantum_computer.py infrastructure.*
- `run` (line 300) `def run(self)` - *Run Grover's search algorithm.

Returns:
    Dictionary with results including success probability and evolution history.*
- `__init__` (line 440) `def __init__(self, config)`
- `bethe_formula` (line 462) `def bethe_formula(self, n, l, Z)` - *Bethe's non-relativistic formula for Lamb shift.

ΔE_Lamb = (8α^3 / 3πn^3) * |ψ_n(0)|^2 * ln(E_avg / E_n)

For s-states: |ψ_n(0)|^2 = Z^3 / (π n^3 a0^3)

Args:
    n: Principal quantum number
    l: Angular momentum quantum number
    Z: Nuclear charge (default 1 for hydrogen)

Returns:
    Lamb shift in atomic units*
- `_higher_l_shift` (line 505) `def _higher_l_shift(self, n, l, Z)` - *Approximate Lamb shift for l > 0.
Much smaller than for s-states.*
- `full_lamb_shift` (line 518) `def full_lamb_shift(self, n, l, j, Z)` - *Calculate full Lamb shift including radiative corrections.

ΔE = ΔE_SE + ΔE_Uehling + ΔE_rel

Where:
- ΔE_SE: Self-energy (main contribution)
- ΔE_Uehling: Vacuum polarization (Uehling potential)
- ΔE_rel: Relativistic corrections*
- `compare_2s_2p` (line 558) `def compare_2s_2p(self, Z)` - *Compare Lamb shift for 2s_{1/2} and 2p_{1/2} states.

This is the classic Lamb shift measurement: the 2s_{1/2} - 2p_{1/2} splitting.
Experimentally: ~1057.8 MHz*
- `__init__` (line 605) `def __init__(self, config)`
- `schwinger_term` (line 611) `def schwinger_term(self)` - *Schwinger's first-order result: a_e = α/(2π)

This is the leading QED correction.*
- `second_order` (line 619) `def second_order(self)` - *Second-order correction: (α/π)^2 * C_2
C_2 ≈ 0.328478965...*
- `third_order` (line 627) `def third_order(self)` - *Third-order correction: (α/π)^3 * C_3
C_3 ≈ 1.181241456...*
- `fourth_order` (line 635) `def fourth_order(self)` - *Fourth-order correction: (α/π)^4 * C_4
C_4 ≈ -1.9144(35)*
- `fifth_order` (line 643) `def fifth_order(self)` - *Fifth-order correction: (α/π)^5 * C_5
C_5 ≈ 7.7(1.1)*
- `calculate_a_e` (line 651) `def calculate_a_e(self, order)` - *Calculate anomalous magnetic moment to specified order.

Args:
    order: Maximum order to include (1-5)

Returns:
    Dictionary with contributions at each order*
- `full_report` (line 686) `def full_report(self)` - *Generate a full report on g-2 calculations.*
- `__init__` (line 728) `def __init__(self, config)`
- `run_full_analysis` (line 736) `def run_full_analysis(self)` - *Run complete QED analysis.*
- `_calculate_energy_levels` (line 761) `def _calculate_energy_levels(self)` - *Calculate hydrogen energy levels including QED corrections.*
- `_dirac_energy` (line 805) `def _dirac_energy(self, n, kappa)` - *Calculate Dirac energy level.
Uses existing relativistic_hydrogen.py if available.*
- `h2o` (line 866) `def h2o(bond_length, angle_deg)` - *Build water molecule geometry.

H2O geometry:
    H1 at (0, 0, 0)
    O  at (r_OH, 0, 0)
    H2 at (r_OH + r_OH*cos(θ), r_OH*sin(θ), 0)*
- `nh3` (line 898) `def nh3(bond_length, angle_deg)` - *Build ammonia molecule geometry.

NH3 has trigonal pyramidal geometry.*
- `ch4` (line 933) `def ch4(bond_length)` - *Build methane molecule geometry.

CH4 has tetrahedral geometry.*
- `__init__` (line 976) `def __init__(self, config)`
- `run_pyscf` (line 999) `def run_pyscf(self, molecule)` - *Run PySCF calculation for the molecule.*
- `_hardcoded_values` (line 1066) `def _hardcoded_values(self, molecule)` - *Return hardcoded reference values for common molecules.*
- `__init__` (line 1110) `def __init__(self, config)`
- `run_analysis` (line 1123) `def run_analysis(self, molecule_name)` - *Run complete analysis for a molecule.*
- `run_all` (line 1164) `def run_all(self)` - *Run analysis for all molecules.*
- `scan_bond_length` (line 1175) `def scan_bond_length(self, molecule_name, r_min, r_max, n_points)` - *Scan potential energy surface by varying bond length.*
- `__init__` (line 1228) `def __init__(self)`
- `run_grover` (line 1235) `def run_grover(self, n_qubits, marked_state)` - *Run Grover's algorithm experiment.*
- `run_qed` (line 1252) `def run_qed(self)` - *Run QED effects experiment.*
- `run_polyatomic` (line 1266) `def run_polyatomic(self, molecule)` - *Run polyatomic molecule experiment.*
- `run_all` (line 1278) `def run_all(self)` - *Run all experiments.*

#### `app.py`
**Path:** `app.py`
**File Doc:** *app.py — Corrected with proper particle-conserving ansatz =======================================================================  Root cause identified: The uccsd() function in molecular_sim.py implements singles excitations incorrectly. For excitation (o→v), it does: CNOT ladder → RY(v) → CNOT ladder inverse This ADDS an electron at v without REMOVING from o, producing |1110⟩ instead of |0110⟩. The states reachable by uccsd never include |0110⟩ or |1001⟩, which are exactly the states the dipole operator connects to |1100⟩ (HF). Hence <μ>=0 always, giving α=0.*

**Classes:**
- `VQEResult` (line 39) `class VQEResult`
- `StarkEvaluator` (line 151) `class StarkEvaluator`
- `DipoleOperatorBuilder` (line 203) `class DipoleOperatorBuilder`
- `PolarizabilityCalculator` (line 237) `class PolarizabilityCalculator`

**Methods:**
- `_sd_indices` (line 46) `def _sd_indices(n_e, n_q)`
- `_run_circuit` (line 55) `def _run_circuit(circuit, backend, state)`
- `givens_single_excitation` (line 61) `def givens_single_excitation(state, o, v, theta, n_qubits, backend)` - *Apply a particle-conserving single excitation rotation between 
qubits o (occupied) and v (virtual).

Rotates in the {|1_o 0_v⟩, |0_o 1_v⟩} subspace:
    |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩
    |0_o 1_v⟩ → -sin(θ)|1_o 0_v⟩ + cos(θ)|0_o 1_v⟩

For adjacent qubits: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)
For non-adjacent: SWAP chain to make adjacent, apply, SWAP back.*
- `particle_conserving_ansatz` (line 109) `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)` - *Particle-conserving UCCSD-like ansatz:
- Singles: Givens rotations (correct particle conservation)
- Doubles: reuse uccsd's double excitation (works correctly)*
- `__init__` (line 152) `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- `_to_scalar` (line 161) `def _to_scalar(amps)`
- `_apply_pauli` (line 165) `def _apply_pauli(self, amps, pauli)`
- `eval_dipole` (line 184) `def eval_dipole(self, amps_raw)`
- `__call__` (line 197) `def __call__(self, amps)`
- `__init__` (line 204) `def __init__(self, bond_length_angstrom)`
- `__init__` (line 238) `def __init__(self)`
- `_evaluator` (line 258) `def _evaluator(self, field)`
- `_get_state` (line 262) `def _get_state(self, theta)`
- `_diagnose` (line 267) `def _diagnose(self, field, theta, label)`
- `_optimize` (line 277) `def _optimize(self, field, theta_init, n_restarts)`
- `run` (line 303) `def run(self)`
- `cost` (line 290) `def cost(th)`

#### `demo_molecular_vqe.py`
**Path:** `demo_molecular_vqe.py`
**File Doc:** *Quantum Framework Demo - Production Version =========================================== Demonstrates all improvements and features of the refactored molecular module.  Improvements implemented: 1. OpenFermion for all Hamiltonians (NO hardcoded values) 2. Precision mode flag (direct vs MPS) 3. Smart initialization with MP2 + parameter scan 4. Cached Pauli operations (x10-100 speedup) 5. Particle-conserving subspace 6. TOML configuration*

**Functions:**
- `print_header` (line 47) `def print_header(title)` - *Print formatted header.*
- `check_dependencies` (line 54) `def check_dependencies()` - *Check and report available dependencies.*
- `demo_h2_direct` (line 78) `def demo_h2_direct()` - *Demo: H2 with direct statevector (precision mode).*
- `demo_h2_mps` (line 104) `def demo_h2_mps()` - *Demo: H2 with MPS compression.*
- `demo_comparison` (line 131) `def demo_comparison()` - *Demo: Compare direct vs MPS.*
- `demo_molecule_builder` (line 167) `def demo_molecule_builder()` - *Demo: OpenFermion molecule builder.*
- `demo_smart_initialization` (line 197) `def demo_smart_initialization()` - *Demo: Smart parameter initialization.*
- `demo_cached_operations` (line 231) `def demo_cached_operations()` - *Demo: Cached Pauli operations.*
- `demo_config_from_toml` (line 277) `def demo_config_from_toml()` - *Demo: Configuration from TOML.*
- `main` (line 302) `def main()` - *Run all demos.*

#### `entangled_hydrogen.py`
**Path:** `entangled_hydrogen.py`
**File Doc:** *Entangled Hydrogen Visualization System ======================================== Demonstrates entangled hydrogen states using the trained quantum computer and molecular simulation backends.  This script IMPORTS and USES the existing modules: - quantum_computer.py for quantum state preparation and evolution - molecular_sim.py for molecular energy evaluation - relativistic_hydrogen.py for Dirac relativistic calculations - orbital_visualizer2.py for orbital visualization components  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `EntangledHydrogenConfig` (line 61) `class EntangledHydrogenConfig` - *Configuration for entangled hydrogen visualization system.

All parameters are parametric and configurable from this class.*
- `IEntangledState` (line 114) `class IEntangledState(ABC)` - *Abstract interface for entangled quantum states.*
- `BellState` (line 131) `class BellState(IEntangledState)` - *Bell state: |Phi+> = (|00> + |11>) / sqrt(2).*
- `GHZState` (line 145) `class GHZState(IEntangledState)` - *GHZ state: (|00...0> + |11...1>) / sqrt(2).*
- `WState` (line 162) `class WState(IEntangledState)` - *W state: (|001> + |010> + |100>) / sqrt(3).*
- `WavefunctionCalculator` (line 194) `class WavefunctionCalculator` - *Calculates hydrogen atom wavefunctions for entangled state visualization.
Uses the same implementation as orbital_visualizer2.py.*
- `EntangledHydrogenSampler` (line 253) `class EntangledHydrogenSampler` - *Monte Carlo sampler for entangled hydrogen states.
Samples from the joint probability distribution of entangled orbitals.*
- `EntangledHydrogenVisualizer` (line 408) `class EntangledHydrogenVisualizer` - *Visualizer for entangled hydrogen states.
Creates high-resolution visualizations similar to orbital_visualizer2.py.*
- `EntangledHydrogenExperiment` (line 569) `class EntangledHydrogenExperiment` - *Main experiment class for entangled hydrogen visualization.

Uses the existing quantum_computer.py, molecular_sim.py, and
visualization components from orbital_visualizer2.py and relativistic_hydrogen.py.*

**Functions:**
- `_make_logger` (line 44) `def _make_logger(name)` - *Create a module-level logger with consistent formatter.*

**Methods:**
- `main` (line 893) `def main()` - *Main entry point for entangled hydrogen visualization.*
- `name` (line 119) `def name(self)` - *Return the name of the entangled state.*
- `prepare` (line 123) `def prepare(self, n_qubits)` - *Prepare the entangled state on n qubits.*
- `get_theoretical_entropy` (line 127) `def get_theoretical_entropy(self)` - *Return the theoretical Shannon entropy in bits.*
- `name` (line 135) `def name(self)`
- `prepare` (line 138) `def prepare(self, qc, backend)`
- `get_theoretical_entropy` (line 141) `def get_theoretical_entropy(self)`
- `__init__` (line 148) `def __init__(self, n_qubits)`
- `name` (line 152) `def name(self)`
- `prepare` (line 155) `def prepare(self, qc, backend)`
- `get_theoretical_entropy` (line 158) `def get_theoretical_entropy(self)`
- `__init__` (line 165) `def __init__(self, n_qubits)`
- `name` (line 169) `def name(self)`
- `prepare` (line 172) `def prepare(self, qc, backend, factory)`
- `get_theoretical_entropy` (line 190) `def get_theoretical_entropy(self)`
- `__init__` (line 200) `def __init__(self, config)`
- `radial_wavefunction` (line 204) `def radial_wavefunction(n, l, r)` - *Calculate non-relativistic radial wavefunction R_nl(r).*
- `spherical_harmonic_real` (line 215) `def spherical_harmonic_real(l, m, theta, phi)` - *Calculate real spherical harmonics Y_lm(theta, phi).*
- `psi_on_grid` (line 225) `def psi_on_grid(self, n, l, m)` - *Calculate wavefunction on 2D grid for quantum computer processing.*
- `psi_3d` (line 245) `def psi_3d(self, n, l, m, r, theta, phi)` - *Calculate full 3D wavefunction psi_nlm(r, theta, phi).*
- `__init__` (line 259) `def __init__(self, config, wavefunction_calc)`
- `find_max_probability` (line 263) `def find_max_probability(self, n, l, m)` - *Find maximum probability for rejection sampling.*
- `sample_orbital` (line 295) `def sample_orbital(self, n, l, m, num_samples)` - *Sample points from a single hydrogen orbital using Monte Carlo.*
- `sample_entangled_state` (line 362) `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples, entanglement_weight)` - *Sample from entangled hydrogen state.

Creates a superposition of two orbitals with entanglement_weight:
|psi> = sqrt(1-w) * |n1,l1,m1> + sqrt(w) * |n2,l2,m2>

For true entanglement visualization, we create a joint state:
|Psi> = (|n1,l1,m1>|n2,l2,m2> + |n2,l2,m2>|n1,l1,m1>) / sqrt(2)*
- `__init__` (line 414) `def __init__(self, config)`
- `visualize` (line 417) `def visualize(self, data, quantum_result, save_path)` - *Create visualization of entangled hydrogen state.*
- `__init__` (line 577) `def __init__(self, config)`
- `_initialize_quantum_computer` (line 591) `def _initialize_quantum_computer(self)` - *Initialize the quantum computer using the existing quantum_computer.py module.*
- `run_bell_entangled_hydrogen` (line 630) `def run_bell_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples, suffix)` - *Run Bell state entangled hydrogen visualization.

Creates a Bell state and correlates it with hydrogen orbitals.*
- `run_ghz_entangled_hydrogen` (line 672) `def run_ghz_entangled_hydrogen(self, orbitals, backend, num_samples)` - *Run GHZ state entangled hydrogen visualization.

Creates a GHZ state with n qubits and correlates with n orbitals.*
- `run_entangled_h_with_molecular_energy` (line 745) `def run_entangled_h_with_molecular_energy(self, n1, l1, m1, n2, l2, m2, backend, num_samples)` - *Run entangled hydrogen with molecular energy evaluation using molecular_sim.py.*
- `run_relativistic_entangled_hydrogen` (line 794) `def run_relativistic_entangled_hydrogen(self, n1, l1, m1, n2, l2, m2, backend, num_samples)` - *Run entangled hydrogen with relativistic Dirac calculations.
Uses components from relativistic_hydrogen.py.*
- `run_all_demonstrations` (line 852) `def run_all_demonstrations(self, num_samples)` - *Run all available entangled hydrogen demonstrations.*

#### `higgs_four_lepton_analysis.py`
**Path:** `higgs_four_lepton_analysis.py`
**File Doc:** *Higgs to Four Lepton Analysis - Quantum Backend Integration =================================================================  This script ACTUALLY uses the user's quantum computing backends: - HamiltonianBackend: Spectral neural network for Hamiltonian operations - SchrodingerBackend: Wave function evolution network - DiracBackend: Relativistic spinor network with gamma matrices  No fake implementations. No placeholders. Uses the neural networks.  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `LeptonType` (line 72) `class LeptonType(Enum)`
- `EventType` (line 78) `class EventType(Enum)`
- `Config` (line 86) `class Config`
- `FourMomentum` (line 125) `class FourMomentum`
- `Lepton` (line 164) `class Lepton`
- `Event` (line 183) `class Event`
- `QuantumSpinorProcessor` (line 205) `class QuantumSpinorProcessor` - *Uses the ACTUAL DiracBackend from quantum_computer.py for spinor calculations.

This is NOT a fake implementation - it uses the neural network
spectral layers trained (or randomly initialized) for Dirac spinor evolution.*
- `EventParser` (line 377) `class EventParser` - *Parse CMS CSV files with actual column names.*
- `Visualizer` (line 436) `class Visualizer` - *3D visualization using Plotly with quantum-processed trajectories.*
- `HiggsQuantumAnalysis` (line 612) `class HiggsQuantumAnalysis` - *Main analysis class using quantum backends from quantum_computer.py*

**Functions:**
- `_make_logger` (line 57) `def _make_logger(name)`

**Methods:**
- `main` (line 747) `def main()`
- `__post_init__` (line 136) `def __post_init__(self)`
- `from_energy_momentum` (line 151) `def from_energy_momentum(cls, E, px, py, pz)`
- `__add__` (line 154) `def __add__(self, other)`
- `pt` (line 171) `def pt(self)`
- `eta` (line 173) `def eta(self)`
- `phi` (line 175) `def phi(self)`
- `energy` (line 177) `def energy(self)`
- `mass` (line 179) `def mass(self)`
- `__post_init__` (line 193) `def __post_init__(self)`
- `check_higgs` (line 200) `def check_higgs(self, cfg)`
- `__init__` (line 213) `def __init__(self, config)`
- `_precompute_momentum_grids` (line 247) `def _precompute_momentum_grids(self)` - *Precompute k-space grids for Dirac equation.*
- `momentum_to_spinor_wavefunction` (line 255) `def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)` - *Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)
suitable for processing by the DiracBackend.

The wavefunction encodes the momentum as a plane wave with the correct
relativistic dispersion relation.*
- `evolve_with_dirac_backend` (line 296) `def evolve_with_dirac_backend(self, psi, steps)` - *Evolve a wavefunction using the DiracBackend neural network.

This applies the actual spectral layers from the trained (or random)
Dirac network.*
- `evolve_with_schrodinger_backend` (line 308) `def evolve_with_schrodinger_backend(self, psi, steps)` - *Evolve using the SchrodingerBackend neural network.*
- `evolve_with_hamiltonian_backend` (line 315) `def evolve_with_hamiltonian_backend(self, psi, steps)` - *Evolve using the HamiltonianBackend neural network.*
- `compute_dirac_current` (line 322) `def compute_dirac_current(self, px, py, pz, energy, mass, charge)` - *Compute the Dirac current using the actual neural network backends.

Returns (jx, jy, jz, info_dict) where info contains quantum observables.*
- `compute_spinor_amplitude` (line 370) `def compute_spinor_amplitude(self, psi)` - *Compute complex amplitude from wavefunction for helicity analysis.*
- `__init__` (line 380) `def __init__(self, config)`
- `parse_file` (line 383) `def parse_file(self, filepath, event_type)`
- `_parse_row` (line 396) `def _parse_row(self, row, event_type)`
- `__init__` (line 439) `def __init__(self, config, quantum_processor)`
- `compute_quantum_helix` (line 443) `def compute_quantum_helix(self, px, py, pz, charge, mass, energy)` - *Compute helical trajectory using actual DiracBackend evolution.*
- `create_visualization` (line 474) `def create_visualization(self, events, output_path)`
- `_create_detector` (line 566) `def _create_detector(self)`
- `_create_explosion` (line 581) `def _create_explosion(self, vx, vy, vz, energy)`
- `__init__` (line 617) `def __init__(self, config)`
- `fetch_data` (line 630) `def fetch_data(self)`
- `load_events` (line 645) `def load_events(self)`
- `analyze_with_quantum_backends` (line 672) `def analyze_with_quantum_backends(self)` - *Run analysis using actual quantum backends.*
- `generate_visualization` (line 728) `def generate_visualization(self)`
- `run` (line 734) `def run(self)`

#### `higgs_quantum_analysis.py`
**Path:** `higgs_quantum_analysis.py`
**File Doc:** *Higgs to Four Lepton Analysis - Quantum Backend Integration =================================================================  This script ACTUALLY uses the user's quantum computing backends: - HamiltonianBackend: Spectral neural network for Hamiltonian operations - SchrodingerBackend: Wave function evolution network - DiracBackend: Relativistic spinor network with gamma matrices  No fake implementations. No placeholders. Uses the neural networks.  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `LeptonType` (line 72) `class LeptonType(Enum)`
- `EventType` (line 78) `class EventType(Enum)`
- `Config` (line 86) `class Config`
- `FourMomentum` (line 125) `class FourMomentum`
- `Lepton` (line 164) `class Lepton`
- `Event` (line 183) `class Event`
- `QuantumSpinorProcessor` (line 205) `class QuantumSpinorProcessor` - *Uses the ACTUAL DiracBackend from quantum_computer.py for spinor calculations.

This is NOT a fake implementation - it uses the neural network
spectral layers trained (or randomly initialized) for Dirac spinor evolution.*
- `EventParser` (line 377) `class EventParser` - *Parse CMS CSV files with actual column names.*
- `Visualizer` (line 436) `class Visualizer` - *3D visualization using Plotly with quantum-processed trajectories.*
- `HiggsQuantumAnalysis` (line 612) `class HiggsQuantumAnalysis` - *Main analysis class using quantum backends from quantum_computer.py*

**Functions:**
- `_make_logger` (line 57) `def _make_logger(name)`

**Methods:**
- `main` (line 747) `def main()`
- `__post_init__` (line 136) `def __post_init__(self)`
- `from_energy_momentum` (line 151) `def from_energy_momentum(cls, E, px, py, pz)`
- `__add__` (line 154) `def __add__(self, other)`
- `pt` (line 171) `def pt(self)`
- `eta` (line 173) `def eta(self)`
- `phi` (line 175) `def phi(self)`
- `energy` (line 177) `def energy(self)`
- `mass` (line 179) `def mass(self)`
- `__post_init__` (line 193) `def __post_init__(self)`
- `check_higgs` (line 200) `def check_higgs(self, cfg)`
- `__init__` (line 213) `def __init__(self, config)`
- `_precompute_momentum_grids` (line 247) `def _precompute_momentum_grids(self)` - *Precompute k-space grids for Dirac equation.*
- `momentum_to_spinor_wavefunction` (line 255) `def momentum_to_spinor_wavefunction(self, px, py, pz, energy, mass, charge)` - *Convert particle momentum to a 2-channel spatial wavefunction (2, G, G)
suitable for processing by the DiracBackend.

The wavefunction encodes the momentum as a plane wave with the correct
relativistic dispersion relation.*
- `evolve_with_dirac_backend` (line 296) `def evolve_with_dirac_backend(self, psi, steps)` - *Evolve a wavefunction using the DiracBackend neural network.

This applies the actual spectral layers from the trained (or random)
Dirac network.*
- `evolve_with_schrodinger_backend` (line 308) `def evolve_with_schrodinger_backend(self, psi, steps)` - *Evolve using the SchrodingerBackend neural network.*
- `evolve_with_hamiltonian_backend` (line 315) `def evolve_with_hamiltonian_backend(self, psi, steps)` - *Evolve using the HamiltonianBackend neural network.*
- `compute_dirac_current` (line 322) `def compute_dirac_current(self, px, py, pz, energy, mass, charge)` - *Compute the Dirac current using the actual neural network backends.

Returns (jx, jy, jz, info_dict) where info contains quantum observables.*
- `compute_spinor_amplitude` (line 370) `def compute_spinor_amplitude(self, psi)` - *Compute complex amplitude from wavefunction for helicity analysis.*
- `__init__` (line 380) `def __init__(self, config)`
- `parse_file` (line 383) `def parse_file(self, filepath, event_type)`
- `_parse_row` (line 396) `def _parse_row(self, row, event_type)`
- `__init__` (line 439) `def __init__(self, config, quantum_processor)`
- `compute_quantum_helix` (line 443) `def compute_quantum_helix(self, px, py, pz, charge, mass, energy)` - *Compute helical trajectory using actual DiracBackend evolution.*
- `create_visualization` (line 474) `def create_visualization(self, events, output_path)`
- `_create_detector` (line 566) `def _create_detector(self)`
- `_create_explosion` (line 581) `def _create_explosion(self, vx, vy, vz, energy)`
- `__init__` (line 617) `def __init__(self, config)`
- `fetch_data` (line 630) `def fetch_data(self)`
- `load_events` (line 645) `def load_events(self)`
- `analyze_with_quantum_backends` (line 672) `def analyze_with_quantum_backends(self)` - *Run analysis using actual quantum backends.*
- `generate_visualization` (line 728) `def generate_visualization(self)`
- `run` (line 734) `def run(self)`

#### `molecular_sim.py`
**Path:** `molecular_sim.py`
**File Doc:** *molecular_simulator.py - VERSIÓN CON OPENFERMION CORREGIDO*

**Classes:**
- `MoleculeData` (line 35) `class MoleculeData`
- `ExactJWEnergy` (line 187) `class ExactJWEnergy` - *Evaluador exacto usando OpenFermion JW.*
- `SurrogateEnergy` (line 265) `class SurrogateEnergy` - *Backend neuronal con calibración.*
- `VQEResult` (line 371) `class VQEResult`
- `VQESolver` (line 390) `class VQESolver`

**Functions:**
- `_make_logger` (line 18) `def _make_logger(name)`

**Methods:**
- `_h2_sto3g_pyscf` (line 41) `def _h2_sto3g_pyscf()` - *H2 con datos PySCF.*
- `_h2_sto3g_hardcoded` (line 75) `def _h2_sto3g_hardcoded()` - *H2 con datos hardcodeados.*
- `build_jw_hamiltonian_of` (line 111) `def build_jw_hamiltonian_of(mol)` - *Construye Hamiltoniano JW usando OpenFermion correctamente.
Usa MolecularData de OpenFermion para asegurar consistencia.*
- `prepare_hf` (line 304) `def prepare_hf(mol, factory, backend)`
- `_sd_indices` (line 312) `def _sd_indices(n_e, n_q)`
- `uccsd` (line 321) `def uccsd(state, thetas, singles, doubles, backend, runner)` - *UCCSD ansatz.

- Singles: cadena JW estándar, conserva número de partículas.
- Doubles: rotación Givens en el subespacio {|HF⟩, |exc⟩} determinado
  dinámicamente desde las amplitudes (robusto ante cambio de convención).
- theta=0 para cualquier parámetro → circuito vacío → identidad exacta.*
- `__init__` (line 190) `def __init__(self, mol, n_qubits)`
- `_to_scalar` (line 203) `def _to_scalar(amps)` - *(dim,2,G,G) → (dim,2)  ó  (dim,2) → (dim,2).
Integra la función de onda espacial sobre la grilla manteniendo
la estructura re/im que necesita el evaluador JW.*
- `_verify_hf` (line 213) `def _verify_hf(self)`
- `_apply` (line 223) `def _apply(self, amps, pauli)`
- `_evaluate` (line 243) `def _evaluate(self, amps)`
- `__call__` (line 257) `def __call__(self, amps)`
- `__init__` (line 268) `def __init__(self, mol, n_qubits, exact_eval, backend)`
- `calibrate` (line 277) `def calibrate(self, hf_amps)`
- `cost_with_barrier` (line 290) `def cost_with_barrier(self, amps)`
- `__repr__` (line 377) `def __repr__(self)`
- `__init__` (line 391) `def __init__(self, qc, config)`
- `_run` (line 395) `def _run(self, circ, be, state)`
- `run` (line 403) `def run(self, mol, backend, max_iter, tol)`
- `cost` (line 451) `def cost(thetas)`

#### `orbital_visualizer2.py`
**Path:** `orbital_visualizer2.py`
**File Doc:** *Hydrogen Orbital Visualizer ============================ HIGH RESOLUTION visualization with Hamiltonian NN. FIXED: Large image size, proper point sizes, high quality output.*

**Classes:**
- `Config` (line 44) `class Config`
- `WavefunctionCalculator` (line 83) `class WavefunctionCalculator` - *Calculates hydrogen atom wavefunctions.*
- `HamiltonianNNProcessor` (line 126) `class HamiltonianNNProcessor` - *Uses YOUR TRAINED MODEL for calculations.*
- `MonteCarloSampler` (line 160) `class MonteCarloSampler` - *Monte Carlo sampling for orbital visualization.*
- `OrbitalVisualizer` (line 265) `class OrbitalVisualizer` - *HIGH RESOLUTION visualization - NOT 16x16!*

**Methods:**
- `main` (line 433) `def main()`
- `radial_wavefunction` (line 87) `def radial_wavefunction(n, l, r)`
- `spherical_harmonic_real` (line 97) `def spherical_harmonic_real(l, m, theta, phi)`
- `psi_on_grid` (line 107) `def psi_on_grid(n, l, m, grid_size)`
- `__init__` (line 129) `def __init__(self, engine)`
- `is_model_loaded` (line 133) `def is_model_loaded(self)`
- `compute_expected_energy` (line 136) `def compute_expected_energy(self, n, l, m)`
- `__init__` (line 163) `def __init__(self, hamiltonian_processor)`
- `find_max_probability` (line 166) `def find_max_probability(self, n, l, m)`
- `sample` (line 195) `def sample(self, n, l, m, num_samples)`
- `visualize` (line 268) `def visualize(self, data, save_path, hamiltonian_processor)`
- `_plotly` (line 396) `def _plotly(self, X, Y, Z, prob_norm, phases, n, l, m)`

#### `polarizability_v3.py`
**Path:** `polarizability_v3.py`
**File Doc:** *polarizability_v3.py — Corrected with proper particle-conserving ansatz =======================================================================  Root cause identified: The uccsd() function in molecular_sim.py implements singles excitations incorrectly. For excitation (o→v), it does: CNOT ladder → RY(v) → CNOT ladder inverse This ADDS an electron at v without REMOVING from o, producing |1110⟩ instead of |0110⟩. The states reachable by uccsd never include |0110⟩ or |1001⟩, which are exactly the states the dipole operator connects to |1100⟩ (HF). Hence <μ>=0 always, giving α=0.  Fix: Implement proper Givens-rotation-based singles that conserve particle number. For adjacent qubits, Givens(o,v,θ) is: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o) This rotates in the {|10⟩, |01⟩} subspace: |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩  For non-adjacent qubits, we SWAP to make them adjacent, apply Givens, then SWAP back. This preserves all quantum numbers.  We keep the doubles excitation from uccsd since it works correctly (verified: it produces |1100⟩↔|0011⟩ rotation properly).*

**Classes:**
- `VQEResult` (line 51) `class VQEResult`
- `StarkEvaluator` (line 163) `class StarkEvaluator`
- `DipoleOperatorBuilder` (line 215) `class DipoleOperatorBuilder`
- `PolarizabilityCalculator` (line 249) `class PolarizabilityCalculator`

**Methods:**
- `_sd_indices` (line 58) `def _sd_indices(n_e, n_q)`
- `_run_circuit` (line 67) `def _run_circuit(circuit, backend, state)`
- `givens_single_excitation` (line 73) `def givens_single_excitation(state, o, v, theta, n_qubits, backend)` - *Apply a particle-conserving single excitation rotation between 
qubits o (occupied) and v (virtual).

Rotates in the {|1_o 0_v⟩, |0_o 1_v⟩} subspace:
    |1_o 0_v⟩ → cos(θ)|1_o 0_v⟩ + sin(θ)|0_o 1_v⟩
    |0_o 1_v⟩ → -sin(θ)|1_o 0_v⟩ + cos(θ)|0_o 1_v⟩

For adjacent qubits: CNOT(v,o) → RY(v, 2θ) → CNOT(v,o)
For non-adjacent: SWAP chain to make adjacent, apply, SWAP back.*
- `particle_conserving_ansatz` (line 121) `def particle_conserving_ansatz(state, thetas, singles, doubles, backend)` - *Particle-conserving UCCSD-like ansatz:
- Singles: Givens rotations (correct particle conservation)
- Doubles: reuse uccsd's double excitation (works correctly)*
- `__init__` (line 164) `def __init__(self, base_hamiltonian, dipole_paulis, dipole_identity, field, n_qubits)`
- `_to_scalar` (line 173) `def _to_scalar(amps)`
- `_apply_pauli` (line 177) `def _apply_pauli(self, amps, pauli)`
- `eval_dipole` (line 196) `def eval_dipole(self, amps_raw)`
- `__call__` (line 209) `def __call__(self, amps)`
- `__init__` (line 216) `def __init__(self, bond_length_angstrom)`
- `__init__` (line 250) `def __init__(self)`
- `_evaluator` (line 270) `def _evaluator(self, field)`
- `_get_state` (line 274) `def _get_state(self, theta)`
- `_diagnose` (line 279) `def _diagnose(self, field, theta, label)`
- `_optimize` (line 289) `def _optimize(self, field, theta_init, n_restarts)`
- `run` (line 315) `def run(self)`
- `cost` (line 302) `def cost(th)`

#### `qc_dashboard.py`
**Path:** `qc_dashboard.py`
**File Doc:** *QC Dashboard - Streamlit web application for real-time quantum playground.  Provides an interactive web-based quantum circuit builder with: - Interactive circuit construction via gate buttons - Real-time state visualisation (probability bars, Bloch spheres, phase space) - OpenQASM code editor with live preview and export/import - Entropy evolution tracking + entanglement analysis (cut-position sweep, scaling) - Molecular H2 VQE simulation (energy convergence, landscape, orbital visualisation) - 3D visualisation via Plotly (Bloch spheres, probability bars, state vector) - Backend comparison (MPS, statevector)  Architecture ------------ DashboardConfig         -- centralised configuration (no magic numbers) VisualisationEngine     -- renders matplotlib figures from snapshots SimulatorBackend        -- thin wrapper around MPS / statevector / synthetic H2VQESolver             -- self-contained H2 VQE (numpy only, no PySCF) DashboardApp            -- top-level Streamlit application orchestrator  Usage ----- streamlit run qc_dashboard.py  Or programmatically: from qc_dashboard import DashboardApp, DashboardConfig app = DashboardApp(DashboardConfig()) app.run()*

**Classes:**
- `DashboardConfig` (line 55) `class DashboardConfig` - *Centralised configuration for the dashboard.*
- `GateItem` (line 120) `class GateItem` - *A gate placed in the circuit builder.*
- `SnapshotData` (line 130) `class SnapshotData` - *Quantum state snapshot for visualisation.*
- `H2VQESolver` (line 147) `class H2VQESolver` - *Self-contained H2 VQE solver using the hardcoded STO-3G Hamiltonian.

The Hamiltonian in the Jordan-Wigner 2-qubit active space:
    H = E_nuc + hZZ * Z0Z1 + hXX * X0X1 + hYY * Y0Y1

Reference energies:
    E_HF = -1.11675928 Ha,  E_FCI = -1.13728383 Ha,  E_nuc = 0.71996899 Ha

Usage
-----
    solver = H2VQESolver()
    result = solver.run_vqe()
    print(result["energy"])  # converged VQE energy

    landscape = solver.energy_landscape()
    print(landscape["bond_lengths"], landscape["energies"])*
- `VisualisationEngine` (line 340) `class VisualisationEngine` - *Renders figures from quantum state snapshots using matplotlib.*
- `SimulatorBackend` (line 725) `class SimulatorBackend` - *Thin wrapper around the QC framework simulator for dashboard use.*
- `Plotly3DEngine` (line 979) `class Plotly3DEngine` - *Optional Plotly-based 3D visualisations.*
- `RealOrbitalEngine` (line 1151) `class RealOrbitalEngine` - *Wrapper around the repo's real orbital_visualizer2.py scripts.*
- `BrutalVizEngine` (line 1269) `class BrutalVizEngine` - *Wrapper around the repo's real quantum_dash.py and quantum_3dview.py.*
- `BackendComparator` (line 1344) `class BackendComparator` - *Compare circuit execution across all available backends.*
- `DashboardApp` (line 1371) `class DashboardApp` - *Streamlit-based interactive quantum playground.*
- `_FakeConfig` (line 1176) `class _FakeConfig`

**Methods:**
- `_capture_mpl_fig` (line 1133) `def _capture_mpl_fig(func)` - *Run a function that creates a matplotlib figure and capture it as PNG bytes.*
- `main` (line 1994) `def main()` - *Launch the Streamlit dashboard.*
- `__init__` (line 166) `def __init__(self, config)`
- `_pauli_operators` (line 173) `def _pauli_operators(n_qubits, qubit, op)`
- `_pauli_string_matrix` (line 189) `def _pauli_string_matrix(paulis, n_qubits)`
- `_build_hamiltonian` (line 206) `def _build_hamiltonian(self, bond_length)` - *Build H2 Hamiltonian matrix for given bond length (Angstrom).

The coefficients scale with bond length to reproduce the Morse-like well.*
- `_ansatz_state` (line 229) `def _ansatz_state(theta)` - *UCC-like ansatz for H2: |psi(theta)> = cos(theta)*|10> + sin(theta)*|01>.*
- `_energy` (line 241) `def _energy(self, theta, h_matrix, e_nuc)`
- `run_vqe` (line 248) `def run_vqe(self, bond_length, max_iter)`
- `energy_landscape` (line 293) `def energy_landscape(self)` - *Sweep bond length and return VQE energy curve.*
- `orbital_wavefunction` (line 318) `def orbital_wavefunction(bond_length, grid_points)` - *Compute hydrogen 1s orbital wavefunction along the internuclear axis.*
- `__init__` (line 343) `def __init__(self, config)`
- `_init_plotting` (line 349) `def _init_plotting(self)`
- `available` (line 362) `def available(self)`
- `render_full_dashboard` (line 365) `def render_full_dashboard(self, snapshots, current)`
- `render_entropy_chart` (line 403) `def render_entropy_chart(self, snapshots)`
- `render_entanglement_profile` (line 429) `def render_entanglement_profile(self, snapshots)` - *Entropy vs cut position for the latest snapshot.*
- `render_vqe_convergence` (line 474) `def render_vqe_convergence(self, convergence, e_hf, e_fci)`
- `render_energy_landscape` (line 501) `def render_energy_landscape(self, bond_lengths, vqe_energies, hf_energies, fci_energies)`
- `render_orbital_plot` (line 533) `def render_orbital_plot(self, orbital_data)`
- `render_entropy_scaling` (line 562) `def render_entropy_scaling(self, data)`
- `_render_probabilities` (line 589) `def _render_probabilities(self, snap, ax)`
- `_render_bloch_sphere` (line 606) `def _render_bloch_sphere(self, snap, ax)`
- `_render_phase_space` (line 635) `def _render_phase_space(self, snap, ax)`
- `render_orbital_2d_projections` (line 670) `def render_orbital_2d_projections(self, data)`
- `_hex_to_rgb` (line 715) `def _hex_to_rgb(h)`
- `__init__` (line 728) `def __init__(self, config)`
- `_init_framework` (line 737) `def _init_framework(self)`
- `_get_mps_gate_registry` (line 760) `def _get_mps_gate_registry()`
- `_get_sv_gate_registry` (line 768) `def _get_sv_gate_registry()`
- `execute_circuit` (line 775) `def execute_circuit(self, gates, n_qubits)`
- `_mps_execute` (line 788) `def _mps_execute(self, gates, n_qubits)`
- `_sv_execute` (line 839) `def _sv_execute(self, gates, n_qubits)`
- `_snapshot_from_mps` (line 886) `def _snapshot_from_mps(self, state, step, gate_name)`
- `_snapshot_from_sv` (line 898) `def _snapshot_from_sv(self, state, step, gate_name)`
- `_compute_bloch_mps` (line 917) `def _compute_bloch_mps(state, n_qubits)`
- `_synthetic_execute` (line 933) `def _synthetic_execute(self, gates, n_qubits)`
- `__init__` (line 982) `def __init__(self, config)`
- `_init_plotly` (line 988) `def _init_plotly(self)`
- `available` (line 998) `def available(self)`
- `render_bloch_3d` (line 1001) `def render_bloch_3d(self, bloch_vectors)`
- `render_probability_3d` (line 1049) `def render_probability_3d(self, probabilities, n_qubits)`
- `render_state_3d` (line 1080) `def render_state_3d(self, probabilities, phases)` - *3D scatter plot: X=real, Y=imaginary, Z=probability.*
- `__init__` (line 1154) `def __init__(self)`
- `_init_real` (line 1163) `def _init_real(self)`
- `available` (line 1210) `def available(self)`
- `entangled_available` (line 1214) `def entangled_available(self)`
- `sample` (line 1217) `def sample(self, n, l, m, num_samples)`
- `render_to_bytes` (line 1222) `def render_to_bytes(self, data)`
- `sample_entangled` (line 1240) `def sample_entangled(self, n1, l1, m1, n2, l2, m2, num_samples)`
- `render_entangled_to_bytes` (line 1250) `def render_entangled_to_bytes(self, data)`
- `__init__` (line 1272) `def __init__(self)`
- `_init_real` (line 1279) `def _init_real(self)`
- `dash_available` (line 1297) `def dash_available(self)`
- `hologram_available` (line 1301) `def hologram_available(self)`
- `run_brutal_viz` (line 1304) `def run_brutal_viz(self, circuit_name)`
- `render_hologram` (line 1327) `def render_hologram(self, snapshots, backend_comp)`
- `__init__` (line 1347) `def __init__(self, config)`
- `run_comparison` (line 1351) `def run_comparison(self, gates, n_qubits)`
- `__init__` (line 1374) `def __init__(self, config)`
- `run` (line 1384) `def run(self)`
- `_ensure_streamlit` (line 1399) `def _ensure_streamlit()`
- `_init_session` (line 1408) `def _init_session(st)`
- `_render_ui` (line 1432) `def _render_ui(self, st)`
- `_render_sidebar_controls` (line 1447) `def _render_sidebar_controls(self, st)`
- `_render_main_panel` (line 1535) `def _render_main_panel(self, st)`
- `_render_playground_tab` (line 1563) `def _render_playground_tab(self, st)`
- `_render_qasm_tab` (line 1606) `def _render_qasm_tab(self, st)`
- `_render_entropy_tab` (line 1669) `def _render_entropy_tab(self, st)`
- `_render_entanglement_tab` (line 1688) `def _render_entanglement_tab(self, st)`
- `_render_molecule_tab` (line 1732) `def _render_molecule_tab(self, st)`
- `_render_3d_tab` (line 1803) `def _render_3d_tab(self, st)`
- `_render_orbital_tab` (line 1855) `def _render_orbital_tab(self, st)`
- `_auto_run` (line 1958) `def _auto_run(self, st)`
- `_build_qasm_from_gates` (line 1969) `def _build_qasm_from_gates(self, gates)`
- `_parse_orb` (line 1936) `def _parse_orb(k)`

#### `qc_integration.py`
**Path:** `qc_integration.py`
**File Doc:** *QC Integration Bridge - OpenQASM, Qiskit, and PennyLane interoperability layer.  Provides bidirectional conversion between the QC quantum circuit representation and standard quantum computing formats: - OpenQASM 2.0 (export/import) - Qiskit QuantumCircuit (export/import when qiskit is installed) - PennyLane QNode / tape (export/import when pennylane is installed)  Architecture ------------ IntegrationConfig    -- centralised configuration (no hardcoded values) IQCAdapter          -- abstract interface for each target format OpenQasmAdapter     -- OpenQASM 2.0 string <-> internal circuit QiskitAdapter       -- Qiskit QuantumCircuit <-> internal circuit PennyLaneAdapter    -- PennyLane operations <-> internal circuit IntegrationBridge   -- facade that delegates to the correct adapter  Usage ----- from qc_integration import IntegrationBridge, IntegrationConfig  config = IntegrationConfig() bridge = IntegrationBridge(config)  # Export to OpenQASM qasm_str = bridge.export_qasm(circuit)  # Import from OpenQASM*

**Classes:**
- `IntegrationConfig` (line 77) `class IntegrationConfig` - *Centralised configuration for the integration bridge.

All tunable parameters live here -- no hardcoded values in logic.*
- `GateInstruction` (line 130) `class GateInstruction` - *A single quantum gate instruction.*
- `CircuitIR` (line 146) `class CircuitIR` - *Intermediate representation of a quantum circuit.

This is the lingua franca between the QC framework and external formats.*
- `IQCAdapter` (line 181) `class IQCAdapter(ABC)` - *Interface for a format-specific adapter (Interface Segregation).*
- `OpenQasmAdapter` (line 226) `class OpenQasmAdapter(IQCAdapter)` - *Adapter for OpenQASM 2.0 format.*
- `QiskitAdapter` (line 358) `class QiskitAdapter(IQCAdapter)` - *Adapter for Qiskit QuantumCircuit format.*
- `PennyLaneAdapter` (line 455) `class PennyLaneAdapter(IQCAdapter)` - *Adapter for PennyLane format.*
- `FrameworkAdapter` (line 557) `class FrameworkAdapter` - *Converts between CircuitIR and the actual QC framework circuit types.

Supports both:
  - quantum_framework_core.MPSQuantumComputer / QuantumCircuit (MPS)
  - quantum_computer.QuantumComputer / QuantumCircuit (statevector)*
- `StandardCircuitFactory` (line 659) `class StandardCircuitFactory` - *Build CircuitIR instances for common quantum algorithms.*
- `IntegrationBridge` (line 727) `class IntegrationBridge` - *Facade that exposes all format conversions through a single API.

Usage
-----
    bridge = IntegrationBridge()
    cir = StandardCircuitFactory.bell_state()

    # OpenQASM
    qasm_str = bridge.export_qasm(cir)
    assert isinstance(qasm_str, str)

    cir_restored = bridge.import_qasm(qasm_str)
    assert cir_restored.n_qubits == cir.n_qubits

    # Qiskit (if installed)
    if bridge._qiskit_available:
        qc_qiskit = bridge.to_qiskit(cir)
        cir_back = bridge.from_qiskit(qc_qiskit)

    # PennyLane (if installed)
    if bridge._pennylane_available:
        qnode = bridge.to_pennylane(cir)
        result = qnode()*

**Methods:**
- `main` (line 851) `def main()` - *Command-line interface for the integration bridge.*
- `__post_init__` (line 115) `def __post_init__(self)`
- `__post_init__` (line 137) `def __post_init__(self)`
- `num_qubits` (line 141) `def num_qubits(self)`
- `append` (line 156) `def append(self, gate)`
- `__len__` (line 164) `def __len__(self)`
- `__repr__` (line 167) `def __repr__(self)`
- `export` (line 185) `def export(self, circuit)` - *Export a CircuitIR to the target format.*
- `import_` (line 189) `def import_(self, data)` - *Import from the target format into a CircuitIR.*
- `__init__` (line 238) `def __init__(self, config)`
- `export` (line 243) `def export(self, circuit)`
- `import_` (line 277) `def import_(self, data)` - *Parse an OpenQASM 2.0 string into a CircuitIR.*
- `_param_names_for_gate` (line 339) `def _param_names_for_gate(name)`
- `__init__` (line 361) `def __init__(self, config)`
- `export` (line 364) `def export(self, circuit)`
- `import_` (line 379) `def import_(self, data)`
- `_ensure_qiskit` (line 403) `def _ensure_qiskit()`
- `_build_qiskit_method_map` (line 414) `def _build_qiskit_method_map(qc)`
- `_build_reverse_gate_map` (line 433) `def _build_reverse_gate_map()`
- `__init__` (line 458) `def __init__(self, config)`
- `export` (line 461) `def export(self, circuit)`
- `import_` (line 483) `def import_(self, data)`
- `_ensure_pennylane` (line 506) `def _ensure_pennylane()`
- `_build_gate_ops` (line 517) `def _build_gate_ops(pl)`
- `_build_reverse_ops` (line 535) `def _build_reverse_ops(pl)`
- `__init__` (line 565) `def __init__(self, config)`
- `to_circuit_ir` (line 568) `def to_circuit_ir(self, circuit, framework_type)` - *Extract a CircuitIR from a framework circuit object.*
- `from_circuit_ir` (line 584) `def from_circuit_ir(self, cir, framework_type)` - *Build a framework circuit object from a CircuitIR.*
- `_detect_framework` (line 598) `def _detect_framework(circuit)`
- `_from_mps_circuit` (line 607) `def _from_mps_circuit(circuit)`
- `_to_mps_circuit` (line 623) `def _to_mps_circuit(cir)`
- `_from_sv_circuit` (line 631) `def _from_sv_circuit(circuit)`
- `_to_sv_circuit` (line 647) `def _to_sv_circuit(cir)`
- `bell_state` (line 663) `def bell_state()`
- `ghz_state` (line 670) `def ghz_state(n_qubits)`
- `qft` (line 678) `def qft(n_qubits)`
- `w_state` (line 693) `def w_state(n_qubits)`
- `grover` (line 701) `def grover(n_qubits, marked, iterations)`
- `__init__` (line 753) `def __init__(self, config)`
- `_init_optional_adapters` (line 763) `def _init_optional_adapters(self)`
- `export_qasm` (line 780) `def export_qasm(self, circuit)`
- `import_qasm` (line 783) `def import_qasm(self, qasm_str)`
- `to_qiskit` (line 788) `def to_qiskit(self, circuit)`
- `from_qiskit` (line 793) `def from_qiskit(self, qiskit_circuit)`
- `to_pennylane` (line 800) `def to_pennylane(self, circuit)`
- `from_pennylane` (line 805) `def from_pennylane(self, pennylane_data)`
- `to_circuit_ir` (line 812) `def to_circuit_ir(self, circuit, framework_type)`
- `from_circuit_ir` (line 819) `def from_circuit_ir(self, cir, framework_type)`
- `export_qasm_from_framework` (line 828) `def export_qasm_from_framework(self, circuit, framework_type)`
- `import_qasm_to_framework` (line 837) `def import_qasm_to_framework(self, qasm_str, framework_type)`
- `circuit_fn` (line 467) `def circuit_fn()`

#### `quantum_3dview.py`
**Path:** `quantum_3dview.py`
**File Doc:** *quantum_brutalist.py - Ultra-High Fidelity Quantum Visualization ================================================================ Real-time holographic quantum state visualization with particle effects, neural network topology mapping, and immersive 3D interaction.  Author: Gris Iscomeback*

**Classes:**
- `BrutalTheme` (line 32) `class BrutalTheme(Enum)`
- `BrutalConfig` (line 39) `class BrutalConfig`
- `QuantumHologram` (line 84) `class QuantumHologram` - *Visualizador holográfico 3D de estados cuánticos*
- `QuantumNeuralTopology` (line 403) `class QuantumNeuralTopology` - *Visualiza la topología interna de las redes neuronales cuánticas*
- `QuantumSonification` (line 500) `class QuantumSonification` - *Convierte estados cuánticos en audio para percepción alternativa*
- `BrutalDashboard` (line 560) `class BrutalDashboard` - *Dashboard interactivo completo con todas las visualizaciones brutales*

**Methods:**
- `demo_brutal` (line 614) `def demo_brutal()` - *Demostración de visualización brutal*
- `create_synthetic_snapshots` (line 653) `def create_synthetic_snapshots()` - *Crea datos sintéticos para demostración*
- `colors` (line 52) `def colors(self)`
- `__init__` (line 87) `def __init__(self, config)`
- `create_amplitude_hologram` (line 92) `def create_amplitude_hologram(self, snapshots, backend_comparison)` - *Crea visualización holográfica de amplitudes en 3D con efecto de partículas*
- `_add_holographic_field` (line 148) `def _add_holographic_field(self, fig, snapshots, row, col)` - *Añade campo de amplitudes 3D con efecto de partículas flotantes*
- `_add_entropy_trails` (line 245) `def _add_entropy_trails(self, fig, snapshots, row, col)` - *Añade trazas de entropía con efecto de cola luminosa*
- `_add_bloch_sphere_holographic` (line 292) `def _add_bloch_sphere_holographic(self, fig, snapshot, backend_name, row, col)` - *Esfera de Bloch con efectos de holograma y partículas orbitales*
- `_interpolate_color` (line 389) `def _interpolate_color(self, color1, color2, factor)` - *Interpola entre dos colores hex*
- `__init__` (line 406) `def __init__(self, model)`
- `create_topology_map` (line 410) `def create_topology_map(self)` - *Crea mapa 3D de la arquitectura neuronal con activaciones en tiempo real*
- `_extract_layers` (line 477) `def _extract_layers(self, model)` - *Extrae información de capas del modelo PyTorch*
- `_generate_quantum_topology` (line 490) `def _generate_quantum_topology(self)` - *Genera topología representativa de backend cuántico*
- `__init__` (line 503) `def __init__(self, sample_rate)`
- `state_to_audio` (line 506) `def state_to_audio(self, snapshot, duration)` - *Convierte un estado cuántico en onda de audio
- Amplitudes controlan volumen
- Fases controlan paneo estéreo
- Probabilidades controlan frecuencia*
- `_adsr_envelope` (line 538) `def _adsr_envelope(self, length, intensity)` - *Genera envolvente ADSR proporcional a la intensidad del estado*
- `__init__` (line 563) `def __init__(self, config)`
- `generate_full_report` (line 570) `def generate_full_report(self, snapshots, backend_comparison)` - *Genera reporte completo con múltiples visualizaciones*
- `_save_audio` (line 605) `def _save_audio(self, audio, path)` - *Guarda audio como WAV*
- `hex_to_rgb` (line 391) `def hex_to_rgb(hex_color)`
- `rgb_to_hex` (line 395) `def rgb_to_hex(rgb)`

#### `quantum_computer.py`
**Path:** `quantum_computer.py`
**File Doc:** *quantum_computer.py  Author: Gris Iscomeback License: AGPL v3  Collapse-Free Quantum Computer Simulator on Classical Hardware.  The state of n qubits lives in the JOINT Hilbert space C^(2^n). The state vector has 2^n complex amplitudes: one per computational basis state. This is the only representation that correctly supports entanglement.  Each amplitude alpha_k (k in {0,...,2^n - 1}) is encoded as a 2D spatial wavefunction on a (G, G) grid, using the neural physics backends as the time-evolution engine. The joint state tensor has shape:  amplitudes: (2^n, 2, G, G) dim 0 : computational basis index  (2^n states) dim 1 : real / imaginary channel   (2 channels) dim 2 : spatial x                  (G points) dim 3 : spatial y                  (G points)  Single-qubit gates act via einsum on the qubit index within dim 0. Two-qubit gates (CNOT, CZ, SWAP) permute and mix amplitude pairs in dim 0. Measurement reads Born probabilities from norm-squared without collapsing.  Architecture (SOLID): - IQuantumGate         : gate abstraction (Interface Segregation) - IPhysicsBackend      : physics engine abstraction (Dependency Inversion)*

**Classes:**
- `SimulatorConfig` (line 74) `class SimulatorConfig` - *Global configuration for the quantum computer simulator.

grid_size, hidden_dim, expansion_dim, num_spectral_layers must match
the values used when training the checkpoint files.*
- `SpectralLayer` (line 105) `class SpectralLayer(Module)` - *Spectral convolution in frequency domain.

Learns complex kernels that modulate Fourier coefficients.
Architecture is identical to the training scripts.*
- `HamiltonianBackboneNet` (line 144) `class HamiltonianBackboneNet(Module)` - *Hamiltonian backbone: single-channel field -> H|psi>.
Shared by all physics backends as the H operator.*
- `SchrodingerSpectralNet` (line 171) `class SchrodingerSpectralNet(Module)` - *Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction.*
- `DiracSpectralNet` (line 200) `class DiracSpectralNet(Module)` - *Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor.*
- `GammaMatrices` (line 233) `class GammaMatrices` - *Dirac gamma matrices in Dirac or Weyl representation.*
- `JointHilbertState` (line 279) `class JointHilbertState` - *Joint quantum state of n qubits in the full 2^n dimensional Hilbert space.

The state is stored as a tensor of shape (2^n, 2, G, G):
    - dim 0: computational basis index k in {0, ..., 2^n - 1}
             bit j of k is the state of qubit j (MSB = qubit 0)
    - dim 1: channel 0 = real part, channel 1 = imaginary part
    - dim 2: spatial x (G grid points)
    - dim 3: spatial y (G grid points)

Each amplitude alpha_k is a spatial wavefunction. The overall quantum
amplitude for basis state |k> is the complex field alpha_k(x,y).
The Born probability of measuring |k> is:

    P(k) = integral |alpha_k(x,y)|^2 dx dy
         = sum_{x,y} (alpha_k_real^2 + alpha_k_imag^2)

normalized so that sum_k P(k) = 1.

This representation correctly supports:
    - Superposition: multiple k indices have non-zero amplitude
    - Entanglement: amplitudes do not factorize across qubits
    - Coherent multi-qubit gates: exact permutation and mixing of amplitudes*
- `PotentialGenerator` (line 390) `class PotentialGenerator` - *Spatial potentials for eigenstate initialization.*
- `JointStateFactory` (line 474) `class JointStateFactory` - *Builds JointHilbertState tensors for common initial conditions.*
- `IPhysicsBackend` (line 511) `class IPhysicsBackend(ABC)` - *Abstract physics backend for spatial wavefunction evolution.*
- `HamiltonianBackend` (line 523) `class HamiltonianBackend(IPhysicsBackend)` - *Physics backend driven by the Hamiltonian neural network.

Performs first-order Schrodinger time evolution:
    psi(t+dt) = psi(t) - i*dt*H*psi(t)*
- `SchrodingerBackend` (line 591) `class SchrodingerBackend(IPhysicsBackend)` - *Physics backend driven by the Schrodinger network.

Uses the learned 2-channel spectral network for wavefunction propagation.
Falls back to HamiltonianBackend if checkpoint is unavailable.*
- `DiracBackend` (line 640) `class DiracBackend(IPhysicsBackend)` - *Physics backend driven by the Dirac network.

Expands each (2,G,G) amplitude to a 4-component spinor, propagates
via the Dirac network, then projects back to (2,G,G).*
- `IQuantumGate` (line 844) `class IQuantumGate(ABC)` - *Abstract quantum gate operating on the joint Hilbert space.*
- `HadamardGate` (line 863) `class HadamardGate(IQuantumGate)` - *H = [[1,1],[1,-1]] / sqrt(2).*
- `PauliXGate` (line 878) `class PauliXGate(IQuantumGate)` - *X = [[0,1],[1,0]].*
- `PauliYGate` (line 892) `class PauliYGate(IQuantumGate)` - *Y = [[0,-i],[i,0]].*
- `PauliZGate` (line 906) `class PauliZGate(IQuantumGate)` - *Z = [[1,0],[0,-1]].*
- `SGate` (line 920) `class SGate(IQuantumGate)` - *S = [[1,0],[0,i]].*
- `TGate` (line 934) `class TGate(IQuantumGate)` - *T = [[1,0],[0,e^{i*pi/4}]].*
- `RxGate` (line 949) `class RxGate(IQuantumGate)` - *Rx(theta) = exp(-i*theta/2 * X).*
- `RyGate` (line 965) `class RyGate(IQuantumGate)` - *Ry(theta) = exp(-i*theta/2 * Y).*
- `RzGate` (line 981) `class RzGate(IQuantumGate)` - *Rz(theta) = exp(-i*theta/2 * Z).*
- `CNOTGate` (line 998) `class CNOTGate(IQuantumGate)` - *CNOT: |ctrl tgt> -> |ctrl, ctrl XOR tgt>.

4x4 matrix (|00>,|01>,|10>,|11>):
    |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>*
- `CZGate` (line 1022) `class CZGate(IQuantumGate)` - *CZ: applies phase -1 to |11>.

4x4 matrix: diag(1, 1, 1, -1).*
- `SWAPGate` (line 1045) `class SWAPGate(IQuantumGate)` - *SWAP: exchanges two qubits.

4x4 matrix: |01>->|10>, |10>->|01>, others unchanged.*
- `ToffoliGate` (line 1068) `class ToffoliGate(IQuantumGate)` - *Toffoli (CCX): flips target iff both controls are |1>.

Exact amplitude permutation in the 8-element 3-qubit subspace.*
- `MCZGate` (line 1098) `class MCZGate(IQuantumGate)` - *Multi-Controlled Z gate: applies phase -1 to the single basis state
where ALL qubits in targets are |1>.

This is the exact oracle primitive needed by Grover's algorithm.
For n target qubits it marks the state |11...1> with a global phase of -1
and leaves all other basis states unchanged.

Implementation: iterate over all basis states k; if every bit
corresponding to a qubit in targets is set to 1, negate that amplitude
(multiply real and imaginary parts by -1).

targets: list of qubit indices that must all be |1> for the phase flip.*
- `EvolveGate` (line 1130) `class EvolveGate(IQuantumGate)` - *Free Hamiltonian evolution applied to every amplitude in the joint state.

Uses the active physics backend.
params: {"dt": float, "steps": int}*
- `CircuitInstruction` (line 1186) `class CircuitInstruction` - *Single gate instruction.*
- `QuantumCircuit` (line 1193) `class QuantumCircuit` - *Ordered sequence of quantum gate instructions.

Pure data structure: stores instructions, does not execute them.*
- `MeasurementResult` (line 1279) `class MeasurementResult` - *Non-destructive Born-rule measurement of the full register.

Contains the complete probability distribution over all 2^n basis states,
per-qubit marginals, and Bloch vectors. The state is never modified.*
- `QuantumComputer` (line 1333) `class QuantumComputer` - *Collapse-free quantum computer simulator with joint Hilbert space.

The n-qubit state is stored as a (2^n, 2, G, G) tensor. All gates
operate via exact unitary transformations on amplitude pairs. Measurement
is non-destructive Born-rule readout — the state is never collapsed.

Backends:
    "hamiltonian" : Hamiltonian NN spectral operator
    "schrodinger" : Schrodinger evolution network
    "dirac"       : Dirac relativistic spinor network

Usage:
    config = SimulatorConfig(
        hamiltonian_checkpoint="weights/latest.pth",
        schrodinger_checkpoint="weights/schrodinger_crystal_final.pth",
        dirac_checkpoint="weights/dirac_phase5_latest.pth",
    )
    qc = QuantumComputer(config)
    circuit = QuantumCircuit(2)
    circuit.h(0).cnot(0, 1)
    result = qc.run(circuit, backend="schrodinger")
    print(result)
    # Most probable: |00> P=0.5  or  |11> P=0.5
    # Shannon entropy: 1.0000 bits  <- true entanglement*

**Functions:**
- `_make_logger` (line 58) `def _make_logger(name)` - *Create a module-level logger with a consistent formatter.*

**Methods:**
- `_solve_eigenstate` (line 440) `def _solve_eigenstate(config, potential, n)` - *Solve the 1D marginal Hamiltonian and return the n-th eigenstate
as a normalized 2-channel (2, G, G) real tensor.*
- `_build_basis_amplitude` (line 461) `def _build_basis_amplitude(config, basis_idx)` - *Build the (2, G, G) spatial wavefunction for amplitude at basis index basis_idx.

Each computational basis state gets its own spatial eigenstate profile.
The excitation level is proportional to the popcount of the basis index.*
- `_single_qubit_unitary` (line 739) `def _single_qubit_unitary(state, qubit, u, backend)` - *Apply a 2x2 unitary u to qubit j in the joint Hilbert space.

For each pair of basis states (k0, k1) that differ only in bit j:
    alpha_{k0}' = u[0,0]*alpha_{k0} + u[0,1]*alpha_{k1}
    alpha_{k1}' = u[1,0]*alpha_{k0} + u[1,1]*alpha_{k1}

Complex scalar * (2,G,G) amplitude:
    (a+ib)(psi_r + i*psi_i) = (a*psi_r - b*psi_i) + i*(a*psi_i + b*psi_r)

This is exact, preserves unitarity, and correctly creates superpositions.*
- `_two_qubit_unitary` (line 789) `def _two_qubit_unitary(state, ctrl, tgt, u4)` - *Apply a 4x4 unitary in the {|00>,|01>,|10>,|11>} subspace of (ctrl, tgt).

For each group of 4 basis states sharing all bits except ctrl and tgt,
apply the 4x4 unitary to the amplitude quadruplet.

Ordering within the 4x4 block: |00>=0, |01>=1, |10>=2, |11>=3
(first bit = ctrl, second bit = tgt).

This correctly implements CNOT, CZ, SWAP and any 2-qubit gate.*
- `register_gate` (line 1179) `def register_gate(name, gate)` - *Register a custom gate without modifying existing code (Open/Closed Principle).*
- `_check` (line 1602) `def _check(label, condition)` - *Print PASS/FAIL for a single assertion. Returns True if passed.*
- `run_phase_tests` (line 1609) `def run_phase_tests(config)` - *Property-based test suite for quantum phase coherence and unitarity.

Each test has a known exact analytic answer derived from the unitary
algebra. Tests are designed to be sensitive to phase errors — circuits
where wrong relative phases produce DIFFERENT probabilities, not just
different phases that cancel out at measurement.

Returns the number of failed tests.*
- `_demo` (line 1836) `def _demo(config)` - *Run demo suite validating entanglement on all backends.*
- `__init__` (line 113) `def __init__(self, channels, grid_size)`
- `forward` (line 124) `def forward(self, x)` - *Apply spectral convolution via RFFT2.*
- `__init__` (line 150) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 159) `def forward(self, x)` - *Accepts (G,G), (1,G,G), or (B,1,G,G). Returns squeezed output.*
- `__init__` (line 176) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 188) `def forward(self, x)` - *(2,G,G) or (B,2,G,G) -> same shape.*
- `__init__` (line 205) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 217) `def forward(self, x)` - *(8,G,G) or (B,8,G,G) -> same shape.*
- `__init__` (line 236) `def __init__(self, representation, device)`
- `_init_matrices` (line 241) `def _init_matrices(self)`
- `to` (line 272) `def to(self, device)` - *Move all matrices to device.*
- `__init__` (line 305) `def __init__(self, amplitudes, n_qubits)`
- `normalize_` (line 317) `def normalize_(self)` - *In-place normalization: sum_k P(k) = 1.*
- `probabilities` (line 323) `def probabilities(self)` - *Return (2^n,) tensor of Born probabilities P(k) for each basis state.*
- `marginal_probability_one` (line 328) `def marginal_probability_one(self, qubit)` - *Marginal Born probability P(qubit_j = |1>).

Sums P(k) over all basis states k where bit j == 1.
Bit ordering: qubit 0 is the MSB of k.*
- `most_probable_basis_state` (line 343) `def most_probable_basis_state(self)` - *Return the index k with the highest probability.*
- `bloch_vector` (line 347) `def bloch_vector(self, qubit)` - *Compute the reduced Bloch vector for qubit j by partial trace.

rho_j[0,0] = P(qubit=0), rho_j[1,1] = P(qubit=1)
rho_j[0,1] = sum_{pairs} alpha_{k0}^* alpha_{k1} (off-diagonal coherence)
bx = 2 Re(rho_j[0,1]), by = -2 Im(rho_j[0,1]), bz = P(0) - P(1)*
- `clone` (line 385) `def clone(self)` - *Return a deep copy.*
- `__init__` (line 393) `def __init__(self, config)`
- `_grid` (line 397) `def _grid(self)`
- `harmonic` (line 402) `def harmonic(self)` - *V = k/2 * r^2.*
- `double_well` (line 410) `def double_well(self)` - *Double-well along x.*
- `coulomb` (line 417) `def coulomb(self)` - *Coulomb-like V ~ -1/r.*
- `periodic_lattice` (line 424) `def periodic_lattice(self)` - *Periodic cosine lattice.*
- `mixed` (line 429) `def mixed(self, seed)` - *Dirichlet-weighted mixture of all four potentials.*
- `__init__` (line 477) `def __init__(self, config)`
- `_empty` (line 480) `def _empty(self, n_qubits)`
- `all_zeros` (line 484) `def all_zeros(self, n_qubits)` - *Initialize register in |00...0>.*
- `basis_state` (line 492) `def basis_state(self, n_qubits, k)` - *Initialize register in computational basis state |k>.*
- `from_bitstring` (line 502) `def from_bitstring(self, bitstring)` - *Initialize in the basis state given by binary string.*
- `evolve_amplitude` (line 515) `def evolve_amplitude(self, amp, dt)` - *Evolve a single (2, G, G) wavefunction by dt under H.*
- `apply_phase` (line 519) `def apply_phase(self, amp, phase_angle)` - *Apply global phase e^{i*phi} to a (2, G, G) amplitude.*
- `__init__` (line 531) `def __init__(self, config)`
- `_load` (line 539) `def _load(self)`
- `_precompute_laplacian` (line 558) `def _precompute_laplacian(self)`
- `_apply_h` (line 565) `def _apply_h(self, field)`
- `evolve_amplitude` (line 573) `def evolve_amplitude(self, amp, dt)` - *dpsi/dt = -i H psi  =>  psi' = psi + dt * (-i H psi) = psi + dt*(H_i*r - H_r*i).*
- `apply_phase` (line 585) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 599) `def __init__(self, config, hamiltonian)`
- `_load` (line 606) `def _load(self)`
- `evolve_amplitude` (line 628) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 636) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 648) `def __init__(self, config, hamiltonian)`
- `_load` (line 657) `def _load(self)`
- `_precompute_dirac` (line 679) `def _precompute_dirac(self)`
- `_pack` (line 690) `def _pack(self, amp)`
- `_unpack` (line 701) `def _unpack(self, spinor)`
- `_analytical_dirac` (line 707) `def _analytical_dirac(self, spinor)`
- `evolve_amplitude` (line 722) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 735) `def apply_phase(self, amp, phase_angle)`
- `name` (line 849) `def name(self)` - *Gate identifier.*
- `apply` (line 853) `def apply(self, state, backend, targets, params)` - *Apply gate to joint state, return new joint state.*
- `name` (line 867) `def name(self)`
- `apply` (line 870) `def apply(self, state, backend, targets, params)`
- `name` (line 882) `def name(self)`
- `apply` (line 885) `def apply(self, state, backend, targets, params)`
- `name` (line 896) `def name(self)`
- `apply` (line 899) `def apply(self, state, backend, targets, params)`
- `name` (line 910) `def name(self)`
- `apply` (line 913) `def apply(self, state, backend, targets, params)`
- `name` (line 924) `def name(self)`
- `apply` (line 927) `def apply(self, state, backend, targets, params)`
- `name` (line 938) `def name(self)`
- `apply` (line 941) `def apply(self, state, backend, targets, params)`
- `name` (line 953) `def name(self)`
- `apply` (line 956) `def apply(self, state, backend, targets, params)`
- `name` (line 969) `def name(self)`
- `apply` (line 972) `def apply(self, state, backend, targets, params)`
- `name` (line 985) `def name(self)`
- `apply` (line 988) `def apply(self, state, backend, targets, params)`
- `name` (line 1007) `def name(self)`
- `apply` (line 1010) `def apply(self, state, backend, targets, params)`
- `name` (line 1030) `def name(self)`
- `apply` (line 1033) `def apply(self, state, backend, targets, params)`
- `name` (line 1053) `def name(self)`
- `apply` (line 1056) `def apply(self, state, backend, targets, params)`
- `name` (line 1076) `def name(self)`
- `apply` (line 1079) `def apply(self, state, backend, targets, params)`
- `name` (line 1115) `def name(self)`
- `apply` (line 1118) `def apply(self, state, backend, targets, params)`
- `name` (line 1139) `def name(self)`
- `apply` (line 1142) `def apply(self, state, backend, targets, params)`
- `__init__` (line 1200) `def __init__(self, n_qubits)`
- `h` (line 1206) `def h(self, q)`
- `x` (line 1209) `def x(self, q)`
- `y` (line 1212) `def y(self, q)`
- `z` (line 1215) `def z(self, q)`
- `s` (line 1218) `def s(self, q)`
- `t` (line 1221) `def t(self, q)`
- `rx` (line 1224) `def rx(self, q, theta)`
- `ry` (line 1227) `def ry(self, q, theta)`
- `rz` (line 1230) `def rz(self, q, theta)`
- `cnot` (line 1233) `def cnot(self, ctrl, tgt)`
- `cx` (line 1236) `def cx(self, ctrl, tgt)`
- `cz` (line 1239) `def cz(self, ctrl, tgt)`
- `swap` (line 1242) `def swap(self, a, b)`
- `toffoli` (line 1245) `def toffoli(self, c0, c1, tgt)`
- `ccx` (line 1248) `def ccx(self, c0, c1, tgt)`
- `evolve` (line 1251) `def evolve(self, qubits, dt, steps)`
- `barrier` (line 1254) `def barrier(self)`
- `_append` (line 1257) `def _append(self, gate_name, targets, params)`
- `depth` (line 1265) `def depth(self)`
- `__len__` (line 1268) `def __len__(self)`
- `__repr__` (line 1271) `def __repr__(self)`
- `probabilities` (line 1293) `def probabilities(self)` - *Alias: marginal P(|1>) per qubit index.*
- `most_probable_bitstring` (line 1297) `def most_probable_bitstring(self)` - *Return the bitstring with the highest probability.*
- `expectation_z` (line 1301) `def expectation_z(self, qubit)` - *<Z>_j = P(0) - P(1) in [-1, +1].*
- `entropy` (line 1305) `def entropy(self)` - *Shannon entropy of the full probability distribution in bits.*
- `__repr__` (line 1313) `def __repr__(self)`
- `__init__` (line 1361) `def __init__(self, config)`
- `_select_backend` (line 1375) `def _select_backend(self, name)`
- `_state_to_result` (line 1380) `def _state_to_result(self, state)`
- `run` (line 1388) `def run(self, circuit, backend, initial_states)` - *Execute a quantum circuit on the joint Hilbert space.

Args:
    circuit:        The QuantumCircuit to execute.
    backend:        Physics backend name.
    initial_states: Optional {qubit_idx: "0" or "1"}.

Returns:
    Non-destructive MeasurementResult with full distribution.*
- `run_with_state_snapshots` (line 1420) `def run_with_state_snapshots(self, circuit, backend, snapshot_after)` - *Execute circuit with non-destructive probability snapshots.

The state is never collapsed between snapshots.*
- `bell_state` (line 1449) `def bell_state(self, backend)` - *|Phi+> = (|00> + |11>) / sqrt(2).

Expected: P(|00>)=0.5, P(|11>)=0.5, entropy=1 bit.*
- `ghz_state` (line 1459) `def ghz_state(self, n_qubits, backend)` - *(|00...0> + |11...1>) / sqrt(2).

Expected: P(|00...0>)=P(|11...1>)=0.5, all others 0.*
- `quantum_fourier_transform` (line 1471) `def quantum_fourier_transform(self, n_qubits, backend)` - *QFT on |00...0>. Standard H + controlled-Rz decomposition.*
- `grover_oracle_search` (line 1481) `def grover_oracle_search(self, n_qubits, target_bitstring, backend, n_iterations)` - *Grover's search algorithm with correct phase oracle and diffusion operator.

The optimal number of iterations is floor(pi/4 * sqrt(2^n)) which gives
the highest probability of measuring the target state.

Oracle construction for target |t_0 t_1 ... t_{n-1}>:
    1. Apply X to every qubit i where t_i == '0'.
       This maps the target bitstring to |11...1>.
    2. Apply MCZ on all n qubits.
       MCZ flips the phase of |11...1> -> exactly the target state
       (after the X conjugation) gets phase -1.
    3. Undo the X gates from step 1.

Diffusion operator (inversion about the mean):
    H^n  X^n  MCZ  X^n  H^n

Both steps use MCZGate which applies phase -1 to the unique basis state
where all specified qubits are |1>. This is exact for any n.

Args:
    n_qubits:        Number of qubits.
    target_bitstring: Binary string of length n_qubits.
    backend:         Physics backend name.
    n_iterations:    Number of Grover iterations. Defaults to
                     max(1, round(pi/4 * sqrt(2^n_qubits))).

Returns:
    MeasurementResult. The target bitstring should have the highest
    probability after the optimal number of iterations.*
- `variational_ansatz` (line 1559) `def variational_ansatz(self, n_qubits, n_layers, thetas, backend)` - *Hardware-efficient ansatz: Ry layers + CNOT chain. len(thetas)=n_qubits*n_layers.*
- `teleportation` (line 1572) `def teleportation(self, backend)` - *3-qubit teleportation protocol.

q0 prepared in Ry(pi/3). q2 should match q0's state after corrections.*
- `deutsch_jozsa` (line 1585) `def deutsch_jozsa(self, n_input_qubits, is_constant, backend)` - *Deutsch-Jozsa: constant -> all inputs |0>, balanced -> at least one |1>.*

#### `quantum_dash.py`
**Path:** `quantum_dash.py`
**File Doc:** *quantum_brutalist_viz.py - Production Quantum State Visualizer ============================================================== High-fidelity 3D holographic visualization of quantum states using trained neural network backends (Hamiltonian, Schrodinger, Dirac).  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `ColorScheme` (line 106) `class ColorScheme(Enum)`
- `BrutalistConfig` (line 115) `class BrutalistConfig`
- `QuantumSnapshot` (line 239) `class QuantumSnapshot`
- `BackendComparison` (line 255) `class BackendComparison`
- `VisualizationOutput` (line 268) `class VisualizationOutput`
- `IVisualComponent` (line 275) `class IVisualComponent(ABC)`
- `ProbabilityVisualizer` (line 281) `class ProbabilityVisualizer(IVisualComponent)`
- `BlochSphereVisualizer` (line 340) `class BlochSphereVisualizer(IVisualComponent)`
- `PhaseSpaceVisualizer` (line 447) `class PhaseSpaceVisualizer(IVisualComponent)`
- `EntropyVisualizer` (line 509) `class EntropyVisualizer(IVisualComponent)`
- `BackendComparisonVisualizer` (line 583) `class BackendComparisonVisualizer(IVisualComponent)`
- `FidelityVisualizer` (line 633) `class FidelityVisualizer(IVisualComponent)`
- `QuantumStateAnalyzer` (line 674) `class QuantumStateAnalyzer`
- `StandardCircuits` (line 754) `class StandardCircuits`
- `CircuitExecutor` (line 806) `class CircuitExecutor`
- `FigureBuilder` (line 873) `class FigureBuilder`
- `QuantumVisualizer` (line 958) `class QuantumVisualizer`

**Functions:**
- `_make_logger` (line 91) `def _make_logger(name)`

**Methods:**
- `main` (line 1199) `def main()`
- `colors` (line 168) `def colors(self)`
- `plotly_template` (line 234) `def plotly_template(self)`
- `render` (line 277) `def render(self, data, axes, config)`
- `render` (line 282) `def render(self, snapshot, axes, config)`
- `_render_empty` (line 318) `def _render_empty(self, axes, config)`
- `_generate_colors` (line 326) `def _generate_colors(self, probs, config)`
- `render` (line 341) `def render(self, snapshot, axes, config)`
- `_render_empty_sphere` (line 375) `def _render_empty_sphere(self, axes, config)`
- `_draw_sphere_wireframe` (line 381) `def _draw_sphere_wireframe(self, axes, config)`
- `_draw_axes` (line 402) `def _draw_axes(self, axes, config)`
- `_draw_bloch_vector` (line 417) `def _draw_bloch_vector(self, axes, bx, by, bz, color, qubit_idx, config)`
- `_draw_uncertainty_ring` (line 431) `def _draw_uncertainty_ring(self, axes, bx, by, bz, color, config)`
- `render` (line 448) `def render(self, snapshot, axes, config)`
- `_render_empty` (line 495) `def _render_empty(self, axes, config)`
- `render` (line 510) `def render(self, snapshots, axes, config)`
- `_render_empty` (line 556) `def _render_empty(self, axes, config)`
- `_interpolate_colors` (line 564) `def _interpolate_colors(self, color1, color2, n)`
- `_hex_to_rgb` (line 578) `def _hex_to_rgb(self, hex_color)`
- `render` (line 584) `def render(self, comparisons, axes, config)`
- `_render_empty` (line 624) `def _render_empty(self, axes, config)`
- `render` (line 634) `def render(self, comparisons, axes, config)`
- `_render_empty` (line 665) `def _render_empty(self, axes, config)`
- `__init__` (line 675) `def __init__(self, config)`
- `compute_probabilities` (line 678) `def compute_probabilities(self, state)`
- `compute_phases` (line 684) `def compute_phases(self, state)`
- `compute_entropy` (line 696) `def compute_entropy(self, probs)`
- `compute_bloch_vectors` (line 704) `def compute_bloch_vectors(self, state)`
- `create_snapshot` (line 713) `def create_snapshot(self, state, step, gate_name, backend_name)`
- `bell_state` (line 756) `def bell_state()`
- `ghz_state` (line 760) `def ghz_state(n_qubits)`
- `qft` (line 767) `def qft(n_qubits)`
- `grover_oracle` (line 782) `def grover_oracle(n_qubits, marked)`
- `grover_diffusion` (line 794) `def grover_diffusion(n_qubits)`
- `__init__` (line 807) `def __init__(self, qc, config)`
- `execute_sequence` (line 812) `def execute_sequence(self, gates, n_qubits, backend_name)`
- `compare_backends` (line 835) `def compare_backends(self, gates, n_qubits)`
- `__init__` (line 874) `def __init__(self, config)`
- `build_full_figure` (line 883) `def build_full_figure(self, snapshots, comparisons)`
- `build_summary_figure` (line 923) `def build_summary_figure(self, snapshots, comparisons)`
- `__init__` (line 959) `def __init__(self, config)`
- `_initialize` (line 966) `def _initialize(self)`
- `_init_quantum_computer` (line 976) `def _init_quantum_computer(self)`
- `visualize_bell_state` (line 1010) `def visualize_bell_state(self)`
- `visualize_ghz_state` (line 1016) `def visualize_ghz_state(self, n_qubits)`
- `visualize_qft` (line 1022) `def visualize_qft(self, n_qubits)`
- `visualize_grover` (line 1028) `def visualize_grover(self, n_qubits, marked_state)`
- `_execute_and_visualize` (line 1043) `def _execute_and_visualize(self, gates, n_qubits, name)`
- `_create_synthetic_snapshots` (line 1086) `def _create_synthetic_snapshots(self, n_qubits, gates)`
- `_create_synthetic_comparisons` (line 1129) `def _create_synthetic_comparisons(self, n_qubits, gates)`
- `_save_data` (line 1153) `def _save_data(self, path, snapshots, comparisons)`
- `run_all` (line 1174) `def run_all(self)`
- `_print_summary` (line 1189) `def _print_summary(self, results)`

#### `quantum_framework_core.py`
**Path:** `quantum_framework_core.py`
**File Doc:** *Quantum Framework Core Module ============================= Unified core module for quantum simulation using MPS-based Hilbert space representation. Achieves sub-exponential memory scaling O(n * chi^2) instead of O(2^n) for full statevector representation.  This module consolidates: - Configuration loading from TOML - MPS (Matrix Product State) tensor network representation - Physics backends (Hamiltonian, Schrodinger, Dirac) - Quantum gate registry with MPS-compatible operations - Quantum circuit builder and executor  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `HilbertPhase` (line 62) `class HilbertPhase(Enum)` - *Phase classification for Hilbert space compression.*
- `FrameworkConfig` (line 72) `class FrameworkConfig` - *Unified configuration for the quantum simulation framework.
Loads from TOML file with fallback to sensible defaults.*
- `AtomData` (line 218) `class AtomData` - *Atomic data structure.*
- `MoleculeData` (line 230) `class MoleculeData` - *Molecular data structure.*
- `OrbitalData` (line 250) `class OrbitalData` - *Atomic orbital data structure.*
- `ConfigLoader` (line 259) `class ConfigLoader` - *Configuration loader that parses TOML files and provides
access to atoms, molecules, and orbitals data.*
- `ITensorNetwork` (line 435) `class ITensorNetwork(ABC)` - *Abstract interface for tensor network quantum states.*
- `MPSCore` (line 480) `class MPSCore` - *Matrix Product State core tensor A^{[k]}_{i_k} with bond indices.

Shape: (chi_left, d, chi_right) where d=2 for qubits.
Memory per core: O(chi^2 * d) = O(chi^2)*
- `MPSState` (line 554) `class MPSState(ITensorNetwork)` - *Matrix Product State representation of n-qubit quantum state.

|psi> = sum_{i_1...i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} ... A^{[n]}_{i_n} |i_1...i_n>

Memory: O(n * chi^2 * d) vs O(d^n) for full statevector.

Example scaling:
    n=30, chi=16: ~30KB vs 8GB for statevector
    n=33, chi=16: ~33KB vs 64GB for statevector*
- `VacuumCore` (line 916) `class VacuumCore` - *Vacuum Core architecture for topological protection.

Projects irrelevant Hilbert subspace to zero, achieving
high sparsity (target 99.99%) while preserving quantum information.*
- `TopologicalProtector` (line 988) `class TopologicalProtector` - *Provides topological protection for quantum states.

Monitors:
    - Winding numbers
    - Berry phases
    - Edge state preservation*
- `SpectralLayer` (line 1040) `class SpectralLayer(Module)` - *Spectral convolution layer in frequency domain.*
- `HamiltonianBackboneNet` (line 1076) `class HamiltonianBackboneNet(Module)` - *Hamiltonian backbone network for spectral operations.*
- `SchrodingerSpectralNet` (line 1102) `class SchrodingerSpectralNet(Module)` - *Schrodinger network for wavefunction evolution.*
- `DiracSpectralNet` (line 1133) `class DiracSpectralNet(Module)` - *Dirac network for relativistic spinor evolution.*
- `GammaMatrices` (line 1164) `class GammaMatrices` - *Dirac gamma matrices in standard or Weyl representation.*
- `IPhysicsBackend` (line 1215) `class IPhysicsBackend(ABC)` - *Abstract interface for physics backends.*
- `HamiltonianBackend` (line 1229) `class HamiltonianBackend(IPhysicsBackend)` - *Hamiltonian backend using neural network for spectral operations.

Performs first-order Schrodinger time evolution:
    psi(t+dt) = psi(t) - i*dt*H*psi(t)*
- `SchrodingerBackend` (line 1305) `class SchrodingerBackend(IPhysicsBackend)` - *Schrodinger backend using learned 2-channel spectral network.

Falls back to HamiltonianBackend if checkpoint unavailable.*
- `DiracBackend` (line 1358) `class DiracBackend(IPhysicsBackend)` - *Dirac backend for relativistic spinor evolution.

Expands (2,G,G) amplitude to 4-component spinor, propagates,
then projects back.*
- `IQuantumGate` (line 1476) `class IQuantumGate(ABC)` - *Abstract interface for quantum gates.*
- `HadamardGate` (line 1496) `class HadamardGate(IQuantumGate)` - *Hadamard gate: H = [[1,1],[1,-1]] / sqrt(2).*
- `PauliXGate` (line 1511) `class PauliXGate(IQuantumGate)` - *Pauli-X gate: X = [[0,1],[1,0]].*
- `PauliYGate` (line 1525) `class PauliYGate(IQuantumGate)` - *Pauli-Y gate: Y = [[0,-i],[i,0]].*
- `PauliZGate` (line 1539) `class PauliZGate(IQuantumGate)` - *Pauli-Z gate: Z = [[1,0],[0,-1]].*
- `SGate` (line 1553) `class SGate(IQuantumGate)` - *S gate: S = [[1,0],[0,i]].*
- `TGate` (line 1567) `class TGate(IQuantumGate)` - *T gate: T = [[1,0],[0,e^{i*pi/4}]].*
- `RxGate` (line 1582) `class RxGate(IQuantumGate)` - *Rotation-X gate: Rx(theta) = exp(-i*theta/2 * X).*
- `RyGate` (line 1598) `class RyGate(IQuantumGate)` - *Rotation-Y gate: Ry(theta) = exp(-i*theta/2 * Y).*
- `RzGate` (line 1614) `class RzGate(IQuantumGate)` - *Rotation-Z gate: Rz(theta) = exp(-i*theta/2 * Z).*
- `CRzGate` (line 1631) `class CRzGate(IQuantumGate)` - *Controlled-Rz gate: applies Rz to target if control is |1>.*
- `CNOTGate` (line 1656) `class CNOTGate(IQuantumGate)` - *CNOT gate: flips target if control is |1>.*
- `CZGate` (line 1676) `class CZGate(IQuantumGate)` - *Controlled-Z gate: applies phase -1 to |11>.*
- `SWAPGate` (line 1696) `class SWAPGate(IQuantumGate)` - *SWAP gate: exchanges two qubits.*
- `CircuitInstruction` (line 1734) `class CircuitInstruction` - *Single instruction in a quantum circuit.*
- `QuantumCircuit` (line 1741) `class QuantumCircuit` - *Quantum circuit builder for MPS states.*
- `MPSQuantumComputer` (line 1816) `class MPSQuantumComputer` - *Main quantum computer using MPS representation.

Provides:
    - State preparation
    - Circuit execution
    - Backend selection
    - Memory-efficient simulation for up to 33+ qubits*

**Functions:**
- `_make_logger` (line 46) `def _make_logger(name, level)` - *Create a configured logger instance.*

**Methods:**
- `run_scaling_benchmark` (line 2045) `def run_scaling_benchmark(config, max_qubits)` - *Run scaling benchmark to demonstrate MPS memory efficiency.

Returns:
    Dictionary with qubit counts, memory usage, compression ratios.*
- `run_grover_search` (line 2098) `def run_grover_search(qc, n_qubits, marked_states)` - *Run Grover's search algorithm and return results.

Implements the algorithm at the statevector level for correctness
(exact oracle via direct phase flip), then reads off probabilities.

Args:
    qc: MPSQuantumComputer instance (unused for computation, kept for API compatibility)
    n_qubits: number of qubits
    marked_states: list of integers representing marked computational basis states

Returns:
    dict with keys: probability, marked_states (as bit strings), speedup, iterations*
- `__post_init__` (line 139) `def __post_init__(self)` - *Initialize random seeds after configuration.*
- `from_toml` (line 147) `def from_toml(cls, toml_path)` - *Load configuration from TOML file.*
- `__init__` (line 265) `def __init__(self, config_path)`
- `_find_config` (line 274) `def _find_config(self)` - *Find configuration file in standard locations.*
- `_load` (line 286) `def _load(self)` - *Load configuration from TOML file.*
- `_load_defaults` (line 300) `def _load_defaults(self)` - *Load default configuration values.*
- `_parse_atoms` (line 318) `def _parse_atoms(self)` - *Parse atoms from configuration data.*
- `_parse_molecules` (line 332) `def _parse_molecules(self)` - *Parse molecules from configuration data.*
- `_parse_orbitals` (line 352) `def _parse_orbitals(self)` - *Parse orbitals from configuration data.*
- `_parse_experiments` (line 364) `def _parse_experiments(self)` - *Parse experiments from configuration data.*
- `get_atom` (line 375) `def get_atom(self, symbol)` - *Get atom data by symbol (case-insensitive).*
- `get_molecule` (line 384) `def get_molecule(self, name)` - *Get molecule data by name (case-insensitive).*
- `get_orbital` (line 393) `def get_orbital(self, name)` - *Get orbital data by name (case-insensitive).*
- `get_experiment` (line 402) `def get_experiment(self, name)` - *Get experiment data by name.*
- `atoms` (line 407) `def atoms(self)` - *Return all atoms.*
- `molecules` (line 412) `def molecules(self)` - *Return all molecules.*
- `orbitals` (line 417) `def orbitals(self)` - *Return all orbitals.*
- `experiments` (line 422) `def experiments(self)` - *Return all experiments.*
- `get_molecules_by_qubits` (line 426) `def get_molecules_by_qubits(self, max_qubits)` - *Get molecules that fit within qubit budget.*
- `get_atoms_by_qubits` (line 430) `def get_atoms_by_qubits(self, max_qubits)` - *Get atoms that fit within qubit budget.*
- `n_qubits` (line 440) `def n_qubits(self)` - *Return number of qubits.*
- `amplitude` (line 445) `def amplitude(self, basis_index)` - *Compute amplitude for a computational basis state.*
- `apply_single_qubit_gate` (line 450) `def apply_single_qubit_gate(self, qubit, gate)` - *Apply single-qubit gate in-place.*
- `apply_two_qubit_gate` (line 455) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)` - *Apply two-qubit gate in-place.*
- `norm` (line 460) `def norm(self)` - *Compute state norm.*
- `probabilities` (line 465) `def probabilities(self)` - *Compute measurement probabilities.*
- `entropy` (line 470) `def entropy(self)` - *Compute von Neumann entropy.*
- `memory_bytes` (line 475) `def memory_bytes(self)` - *Return memory usage in bytes.*
- `__init__` (line 488) `def __init__(self, chi_left, chi_right, d, device, dtype)`
- `_initialize` (line 504) `def _initialize(self)` - *Initialize core tensor for |0> product state (exact).*
- `tensor` (line 517) `def tensor(self)` - *Return the core tensor.*
- `tensor` (line 524) `def tensor(self, value)` - *Set the core tensor, preserving complex dtype when needed.*
- `left_canonicalize` (line 533) `def left_canonicalize(self)` - *Bring core to left-canonical form, return singular values.*
- `right_canonicalize` (line 543) `def right_canonicalize(self)` - *Bring core to right-canonical form, return singular values.*
- `__init__` (line 567) `def __init__(self, n_qubits, config)`
- `_initialize` (line 576) `def _initialize(self)` - *Initialize MPS with product state |00...0>.*
- `n_qubits` (line 600) `def n_qubits(self)`
- `_bond_dimension` (line 603) `def _bond_dimension(self, site)` - *Compute bond dimension at given site.*
- `amplitude` (line 610) `def amplitude(self, basis_index)` - *Compute amplitude for computational basis state.*
- `apply_single_qubit_gate` (line 631) `def apply_single_qubit_gate(self, qubit, gate)` - *Apply single-qubit gate in-place.

Works in complex128 so that gates with imaginary entries (Y, S, T, Rz…)
are handled correctly.  The core tensor is promoted to complex128 when
the result has a non-negligible imaginary component; otherwise it is
kept as the config dtype (typically float64).*
- `apply_two_qubit_gate` (line 655) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)` - *Apply two-qubit gate in-place.*
- `_swap_qubits_in_gate` (line 674) `def _swap_qubits_in_gate(self, gate)` - *Swap qubit ordering in two-qubit gate.*
- `_apply_adjacent_gate` (line 684) `def _apply_adjacent_gate(self, qubit, gate)` - *Apply gate to adjacent qubit pair.*
- `_apply_nonadjacent_gate` (line 747) `def _apply_nonadjacent_gate(self, qubit_a, qubit_b, gate)` - *Apply gate to non-adjacent qubit pair using SWAP network.*
- `norm` (line 762) `def norm(self)` - *Compute state norm.*
- `_canonicalize` (line 770) `def _canonicalize(self)` - *Bring MPS to canonical form.*
- `probabilities` (line 778) `def probabilities(self)` - *Compute measurement probabilities.*
- `entropy` (line 795) `def entropy(self)` - *Compute maximum entanglement entropy across all cuts.

For n=1 qubits, returns Shannon entropy of the probability distribution.
For n>1, returns the maximum entanglement entropy across all bipartite cuts.*
- `memory_bytes` (line 822) `def memory_bytes(self)` - *Return memory usage in bytes.*
- `entanglement_entropy` (line 829) `def entanglement_entropy(self, cut)` - *Compute entanglement entropy at given cut between qubits cut-1 and cut.
Uses Schmidt decomposition from the MPS bond.*
- `to_statevector` (line 874) `def to_statevector(self)` - *Convert MPS to full statevector (only for small systems).*
- `most_probable_bitstring` (line 890) `def most_probable_bitstring(self)` - *Return most probable basis state as bitstring.*
- `clone` (line 896) `def clone(self)` - *Return a deep copy.*
- `__init__` (line 924) `def __init__(self, n_qubits, config)`
- `_initialize` (line 935) `def _initialize(self)` - *Initialize vacuum core with ground state.*
- `_compute_berry_phases` (line 941) `def _compute_berry_phases(self)` - *Compute Berry phases between active states.*
- `add_active_state` (line 948) `def add_active_state(self, basis_index, winding_number)` - *Add a basis state to the active subspace.*
- `_compute_winding_number` (line 962) `def _compute_winding_number(self, basis_index)` - *Compute winding number for a basis state.*
- `is_topologically_protected` (line 968) `def is_topologically_protected(self, basis_index)` - *Check if state is topologically protected.*
- `sparsity` (line 973) `def sparsity(self)` - *Compute vacuum sparsity.*
- `project_to_active` (line 979) `def project_to_active(self, state)` - *Project state onto active subspace.*
- `__init__` (line 998) `def __init__(self, config)`
- `compute_winding_number` (line 1003) `def compute_winding_number(self, state, qubit)` - *Compute winding number for a qubit.*
- `compute_berry_phase` (line 1015) `def compute_berry_phase(self, state, qubit_a, qubit_b)` - *Compute Berry phase between two qubits.*
- `is_protected` (line 1031) `def is_protected(self, state, vacuum_core)` - *Check if state is topologically protected.*
- `__init__` (line 1043) `def __init__(self, channels, grid_size)`
- `forward` (line 1053) `def forward(self, x)` - *Apply spectral convolution via RFFT2.*
- `__init__` (line 1079) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 1088) `def forward(self, x)` - *Apply Hamiltonian backbone network.*
- `__init__` (line 1105) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 1119) `def forward(self, x)` - *Apply Schrodinger evolution network.*
- `__init__` (line 1136) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 1150) `def forward(self, x)` - *Apply Dirac evolution network.*
- `__init__` (line 1167) `def __init__(self, representation, device)`
- `_init_matrices` (line 1172) `def _init_matrices(self)` - *Initialize gamma matrices.*
- `to` (line 1208) `def to(self, device)` - *Move all matrices to device.*
- `evolve_amplitude` (line 1219) `def evolve_amplitude(self, amp, dt)` - *Evolve a single amplitude by time dt.*
- `apply_phase` (line 1224) `def apply_phase(self, amp, phase_angle)` - *Apply global phase to amplitude.*
- `__init__` (line 1237) `def __init__(self, config)`
- `_load` (line 1245) `def _load(self)` - *Load model from checkpoint.*
- `_precompute_laplacian` (line 1266) `def _precompute_laplacian(self)` - *Precompute Laplacian kernel for kinetic energy.*
- `_apply_h` (line 1274) `def _apply_h(self, field)` - *Apply Hamiltonian operator to field.*
- `evolve_amplitude` (line 1284) `def evolve_amplitude(self, amp, dt)` - *Evolve amplitude by time dt.*
- `apply_phase` (line 1298) `def apply_phase(self, amp, phase_angle)` - *Apply global phase.*
- `__init__` (line 1312) `def __init__(self, config, hamiltonian)`
- `_load` (line 1319) `def _load(self)` - *Load model from checkpoint.*
- `evolve_amplitude` (line 1342) `def evolve_amplitude(self, amp, dt)` - *Evolve amplitude by time dt.*
- `apply_phase` (line 1353) `def apply_phase(self, amp, phase_angle)` - *Apply global phase.*
- `__init__` (line 1366) `def __init__(self, config, hamiltonian)`
- `_load` (line 1375) `def _load(self)` - *Load model from checkpoint.*
- `_precompute_dirac` (line 1398) `def _precompute_dirac(self)` - *Precompute momentum grids for Dirac operator.*
- `_pack` (line 1407) `def _pack(self, amp)` - *Pack 2-channel amplitude to 4-component spinor.*
- `_unpack` (line 1421) `def _unpack(self, spinor)` - *Unpack 4-component spinor to 2-channel amplitude.*
- `_analytical_dirac` (line 1428) `def _analytical_dirac(self, spinor)` - *Apply analytical Dirac Hamiltonian to spinor.*
- `evolve_amplitude` (line 1447) `def evolve_amplitude(self, amp, dt)` - *Evolve amplitude by time dt using Dirac equation.*
- `apply_phase` (line 1463) `def apply_phase(self, amp, phase_angle)` - *Apply global phase.*
- `evolve_spinor` (line 1467) `def evolve_spinor(self, spinor, dt)` - *Evolve full 4-component spinor by time dt.*
- `name` (line 1481) `def name(self)` - *Return gate name.*
- `apply` (line 1486) `def apply(self, state, targets, params)` - *Apply gate to state and return new state.*
- `name` (line 1500) `def name(self)`
- `apply` (line 1503) `def apply(self, state, targets, params)`
- `name` (line 1515) `def name(self)`
- `apply` (line 1518) `def apply(self, state, targets, params)`
- `name` (line 1529) `def name(self)`
- `apply` (line 1532) `def apply(self, state, targets, params)`
- `name` (line 1543) `def name(self)`
- `apply` (line 1546) `def apply(self, state, targets, params)`
- `name` (line 1557) `def name(self)`
- `apply` (line 1560) `def apply(self, state, targets, params)`
- `name` (line 1571) `def name(self)`
- `apply` (line 1574) `def apply(self, state, targets, params)`
- `name` (line 1586) `def name(self)`
- `apply` (line 1589) `def apply(self, state, targets, params)`
- `name` (line 1602) `def name(self)`
- `apply` (line 1605) `def apply(self, state, targets, params)`
- `name` (line 1618) `def name(self)`
- `apply` (line 1621) `def apply(self, state, targets, params)`
- `name` (line 1635) `def name(self)`
- `apply` (line 1638) `def apply(self, state, targets, params)`
- `name` (line 1660) `def name(self)`
- `apply` (line 1663) `def apply(self, state, targets, params)`
- `name` (line 1680) `def name(self)`
- `apply` (line 1683) `def apply(self, state, targets, params)`
- `name` (line 1700) `def name(self)`
- `apply` (line 1703) `def apply(self, state, targets, params)`
- `__init__` (line 1744) `def __init__(self, n_qubits)`
- `_append` (line 1748) `def _append(self, gate_name, targets, params)` - *Append an instruction to the circuit.*
- `h` (line 1759) `def h(self, qubit)`
- `x` (line 1762) `def x(self, qubit)`
- `y` (line 1765) `def y(self, qubit)`
- `z` (line 1768) `def z(self, qubit)`
- `s` (line 1771) `def s(self, qubit)`
- `t` (line 1774) `def t(self, qubit)`
- `rx` (line 1777) `def rx(self, qubit, theta)`
- `ry` (line 1780) `def ry(self, qubit, theta)`
- `rz` (line 1783) `def rz(self, qubit, theta)`
- `crz` (line 1786) `def crz(self, control, target, theta)`
- `cnot` (line 1789) `def cnot(self, control, target)`
- `cz` (line 1792) `def cz(self, control, target)`
- `swap` (line 1795) `def swap(self, qubit1, qubit2)`
- `__len__` (line 1798) `def __len__(self)` - *Return number of instructions.*
- `__bool__` (line 1802) `def __bool__(self)` - *True if circuit has instructions.*
- `run` (line 1806) `def run(self, state)` - *Execute circuit on state.*
- `__init__` (line 1827) `def __init__(self, config)`
- `create_circuit` (line 1843) `def create_circuit(self, n_qubits)` - *Create a new quantum circuit.*
- `create_state` (line 1851) `def create_state(self, n_qubits)` - *Create initial state |00...0>.*
- `bell_state` (line 1865) `def bell_state(self, n_qubits)` - *Prepare Bell state |Phi+> = (|00> + |11>) / sqrt(2).*
- `ghz_state` (line 1873) `def ghz_state(self, n_qubits)` - *Prepare GHZ state (|00...0> + |11...1>) / sqrt(2).*
- `w_state` (line 1882) `def w_state(self, n_qubits)` - *Prepare W state: |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n

Uses direct statevector-to-MPS conversion via successive SVD.
This guarantees exact representation (up to numerical precision).

The W state has a unique entanglement structure:
- The entanglement entropy for a cut separating k qubits from (n-k) is:
  S(k) = H({k/n, (n-k)/n}) where H is the binary entropy function
- Maximum entropy is 1 bit when n is even and k = n/2
- For W₃: S_max ≈ 0.9183 bits (H({1/3, 2/3}))
- This is DIFFERENT from GHZ which has entropy = 1 bit for any cut

Returns:
    MPSState: The W state in MPS representation*
- `_build_w_state_direct` (line 1905) `def _build_w_state_direct(self, n_qubits, max_bond)` - *Build W state using direct statevector-to-MPS conversion via successive SVD.

This guarantees exact representation (up to numerical precision).*
- `run_circuit` (line 1985) `def run_circuit(self, circuit, initial_state)` - *Execute circuit on state.*
- `get_backend` (line 1995) `def get_backend(self, name)` - *Get physics backend by name.*
- `memory_usage` (line 1999) `def memory_usage(self, state)` - *Compute memory usage for a state.*
- `compression_ratio` (line 2012) `def compression_ratio(self, state)` - *Compute compression ratio vs full statevector.*
- `detect_phase` (line 2019) `def detect_phase(self, state)` - *Detect Hilbert space phase from state properties.*
- `_compute_average_bond_dimension` (line 2037) `def _compute_average_bond_dimension(self, state)` - *Compute average bond dimension across MPS cores.*

#### `quantum_framework_main.py`
**Path:** `quantum_framework_main.py`
**File Doc:** *Quantum Simulation Framework - Main Entry Point ================================================ Main entry point for the quantum simulation framework with command-line interface and interactive menu support.  Usage: python quantum_framework_main.py [OPTIONS]  Options: --config PATH       Path to TOML configuration file --learn             Launch the educational Quantum Lab TUI (EN/ES) --interactive       Launch interactive menu mode --all               Run ALL experiments automatically (for debugging) --experiment NAME   Run specific experiment by name --benchmark         Run scaling benchmark --max-qubits N      Maximum qubits for benchmark --molecule NAME     Specify molecule for experiments --orbital NAME      Specify orbital for visualization --qubits N          Number of qubits for experiments --output DIR        Output directory --verbose           Enable verbose logging --help              Show help message  Author: Gris Iscomeback License: AGPL v3*

**Functions:**
- `setup_logging` (line 52) `def setup_logging(verbose)` - *Configure logging level based on verbosity.*
- `run_benchmark` (line 61) `def run_benchmark(args, config)` - *Run scaling benchmark.*
- `run_experiment` (line 101) `def run_experiment(args, config, config_loader)` - *Run a specific experiment by name.*
- `run_molecular_simulation` (line 173) `def run_molecular_simulation(args, config, config_loader)` - *Run molecular simulation.*
- `run_orbital_visualization` (line 210) `def run_orbital_visualization(args, config, config_loader)` - *Run orbital visualization.*
- `print_info` (line 242) `def print_info(config, config_loader)` - *Print framework information.*
- `main` (line 292) `def main()` - *Main entry point.*

#### `quantum_framework_menu.py`
**Path:** `quantum_framework_menu.py`
**File Doc:** *Quantum Framework Interactive Menu System ========================================= Interactive menu system for accessing all framework capabilities including quantum circuits, molecular simulations, orbital visualization, and advanced physics experiments.  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `MenuSystem` (line 64) `class MenuSystem` - *Interactive menu system for the quantum simulation framework.

Provides structured access to all framework capabilities through
a hierarchical menu system with real-time feedback.*

**Methods:**
- `run_interactive_menu` (line 2357) `def run_interactive_menu(config, config_loader)` - *Run the interactive menu system.*
- `run_all_experiments` (line 2363) `def run_all_experiments(config, config_loader)` - *Run ALL experiments automatically for debugging.
This function executes all available experiments without user interaction.
Uses CORRECT physics formulas verified against experimental data.*
- `__init__` (line 72) `def __init__(self, config, config_loader)`
- `clear_screen` (line 79) `def clear_screen(self)` - *Clear the terminal screen.*
- `print_header` (line 83) `def print_header(self, title)` - *Print formatted header.*
- `print_menu` (line 90) `def print_menu(self, title, options)` - *Print formatted menu with options.*
- `get_input` (line 97) `def get_input(self, prompt)` - *Get user input with history tracking.*
- `pause` (line 106) `def pause(self, message)` - *Wait for user to press Enter.*
- `run` (line 113) `def run(self)` - *Run the main menu loop.*
- `_show_main_menu` (line 118) `def _show_main_menu(self)` - *Display main menu.*
- `_show_circuit_menu` (line 163) `def _show_circuit_menu(self)` - *Display quantum circuits menu.*
- `_custom_circuit` (line 197) `def _custom_circuit(self)` - *Create and run a custom circuit.*
- `_bell_state_demo` (line 279) `def _bell_state_demo(self)` - *Demonstrate Bell state preparation.*
- `_ghz_state_demo` (line 299) `def _ghz_state_demo(self)` - *Demonstrate GHZ state preparation.*
- `_w_state_demo` (line 328) `def _w_state_demo(self)` - *Demonstrate W state preparation.*
- `_single_qubit_gates_demo` (line 356) `def _single_qubit_gates_demo(self)` - *Demonstrate single qubit gates.*
- `_two_qubit_gates_demo` (line 397) `def _two_qubit_gates_demo(self)` - *Demonstrate two qubit gates.*
- `_show_entanglement_menu` (line 431) `def _show_entanglement_menu(self)` - *Display entanglement experiments menu.*
- `_bell_entropy_experiment` (line 459) `def _bell_entropy_experiment(self)` - *Measure entropy of Bell states.*
- `_ghz_scaling_experiment` (line 472) `def _ghz_scaling_experiment(self)` - *Study GHZ entanglement scaling with qubit count.*
- `_entropy_by_cut_experiment` (line 497) `def _entropy_by_cut_experiment(self)` - *Measure entanglement entropy at different cuts.*
- `_entropy_heatmap_experiment` (line 522) `def _entropy_heatmap_experiment(self)` - *Generate entropy heatmap for different states.*
- `_show_molecular_menu` (line 583) `def _show_molecular_menu(self)` - *Display molecular simulations menu.*
- `_list_molecules` (line 617) `def _list_molecules(self)` - *List all available molecules.*
- `_molecule_info` (line 636) `def _molecule_info(self)` - *Show detailed molecule information.*
- `_vqe_ground_state` (line 667) `def _vqe_ground_state(self)` - *Run VQE for molecular ground state.*
- `_energy_landscape` (line 751) `def _energy_landscape(self)` - *Plot molecular energy landscape.*
- `_bond_dissociation` (line 808) `def _bond_dissociation(self)` - *Simulate bond dissociation curve.*
- `_show_orbital_menu` (line 868) `def _show_orbital_menu(self)` - *Display orbital visualization menu.*
- `_list_orbitals` (line 899) `def _list_orbitals(self)` - *List all available orbitals.*
- `_visualize_orbital` (line 915) `def _visualize_orbital(self)` - *Visualize a single orbital.*
- `_generate_orbital_plot` (line 947) `def _generate_orbital_plot(self, orb, num_samples)` - *Generate orbital visualization plot.*
- `_compare_orbitals` (line 1064) `def _compare_orbitals(self)` - *Compare multiple orbitals.*
- `_radial_wavefunction` (line 1146) `def _radial_wavefunction(self)` - *Plot radial wavefunction.*
- `_angular_wavefunction` (line 1202) `def _angular_wavefunction(self)` - *Plot angular wavefunction.*
- `_show_relativistic_menu` (line 1264) `def _show_relativistic_menu(self)` - *Display relativistic physics menu.*
- `_dirac_energy_levels` (line 1292) `def _dirac_energy_levels(self)` - *Calculate Dirac energy levels.*
- `_dirac_energy` (line 1321) `def _dirac_energy(self, n, kappa, alpha, c)` - *Calculate Dirac energy level.*
- `_fine_structure` (line 1329) `def _fine_structure(self)` - *Calculate fine structure corrections.*
- `_zitterbewegung` (line 1347) `def _zitterbewegung(self)` - *Simulate Zitterbewegung.*
- `_spin_orbit` (line 1363) `def _spin_orbit(self)` - *Calculate spin-orbit coupling.*
- `_show_qed_menu` (line 1378) `def _show_qed_menu(self)` - *Display QED effects menu.*
- `_lamb_shift` (line 1406) `def _lamb_shift(self)` - *Calculate Lamb shift.*
- `_anomalous_moment` (line 1426) `def _anomalous_moment(self)` - *Calculate anomalous magnetic moment.*
- `_vacuum_polarization` (line 1453) `def _vacuum_polarization(self)` - *Calculate vacuum polarization effects.*
- `_full_qed` (line 1468) `def _full_qed(self)` - *Show full QED corrections.*
- `_show_algorithms_menu` (line 1482) `def _show_algorithms_menu(self)` - *Display quantum algorithms menu.*
- `_grover_search` (line 1510) `def _grover_search(self)` - *Demonstrate Grover's search algorithm.*
- `_apply_grover_iteration` (line 1558) `def _apply_grover_iteration(self, state, marked, n_qubits)` - *Apply one Grover iteration: oracle then diffusion.*
- `_qft_demo` (line 1607) `def _qft_demo(self)` - *Demonstrate Quantum Fourier Transform.*
- `_phase_estimation` (line 1669) `def _phase_estimation(self)` - *Demonstrate phase estimation.*
- `_vqe_demo` (line 1746) `def _vqe_demo(self)` - *Demonstrate VQE.*
- `_show_benchmark_menu` (line 1816) `def _show_benchmark_menu(self)` - *Display benchmarks menu.*
- `_mps_scaling_benchmark` (line 1844) `def _mps_scaling_benchmark(self)` - *Run MPS scaling benchmark.*
- `_gate_performance` (line 1872) `def _gate_performance(self)` - *Benchmark gate performance.*
- `_memory_comparison` (line 1907) `def _memory_comparison(self)` - *Compare memory usage.*
- `_entanglement_scaling` (line 1927) `def _entanglement_scaling(self)` - *Study entanglement scaling.*
- `_show_config_menu` (line 1953) `def _show_config_menu(self)` - *Display configuration menu.*
- `_view_config` (line 1984) `def _view_config(self)` - *View current configuration.*
- `_list_atoms` (line 1999) `def _list_atoms(self)` - *List available atoms.*
- `_list_molecules_config` (line 2016) `def _list_molecules_config(self)` - *List available molecules.*
- `_list_experiments` (line 2020) `def _list_experiments(self)` - *List available experiments.*
- `_system_info` (line 2036) `def _system_info(self)` - *Show system information.*
- `_show_particle_physics_menu` (line 2055) `def _show_particle_physics_menu(self)` - *Display particle physics menu (Higgs analysis).*
- `_run_higgs_analysis` (line 2077) `def _run_higgs_analysis(self)` - *Run the Higgs boson 4-lepton quantum analysis.*
- `_higgs_about` (line 2101) `def _higgs_about(self)` - *Show information about the Higgs analysis.*
- `_show_visualization_menu` (line 2129) `def _show_visualization_menu(self)` - *Display quantum visualization menu.*
- `_run_quantum_dash` (line 2154) `def _run_quantum_dash(self)` - *Run brutalist quantum state visualizer.*
- `_run_quantum_3dview` (line 2201) `def _run_quantum_3dview(self)` - *Run 3D holographic quantum dashboard.*
- `_run_quantum_visualizer` (line 2247) `def _run_quantum_visualizer(self)` - *Run the standard quantum state visualizer.*
- `_run_polarizability_vqe` (line 2285) `def _run_polarizability_vqe(self)` - *Run H2 polarizability / Stark effect VQE from app.py.*
- `_show_help` (line 2316) `def _show_help(self)` - *Show help information.*
- `_quit` (line 2350) `def _quit(self)` - *Exit the menu system.*
- `test_header` (line 2379) `def test_header(name)`
- `radial_wf` (line 951) `def radial_wf(n, l, r)`
- `spherical_harm_real` (line 959) `def spherical_harm_real(l, m, theta, phi)`
- `compute_energy` (line 1764) `def compute_energy(state, n_qubits)`

#### `quantum_framework_molecular.py`
**Path:** `quantum_framework_molecular.py`
**File Doc:** *Quantum Framework Molecular Module - CORRECTED VERSION ======================================================= Molecular simulation with VQE and UCCSD ansatz using MPS representation.  CRITICAL FIXES: - Corrected H2 Hamiltonian coefficients (verified against PySCF/OpenFermion) - Fixed OpenFermion compatibility (InteractionOperator API) - Proper nuclear repulsion handling - UCCSD ansatz with correct excitation operators  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `MoleculeData` (line 72) `class MoleculeData`
- `MoleculeBuilder` (line 85) `class MoleculeBuilder` - *Build molecule data for quantum chemistry calculations.*
- `ExactJWEnergy` (line 156) `class ExactJWEnergy` - *Exact Jordan-Wigner energy evaluator.

CORRECTED: Uses verified Hamiltonian coefficients from standard references.*
- `UCCSDAnsatz` (line 334) `class UCCSDAnsatz` - *Unitary Coupled Cluster Singles and Doubles ansatz.

For H2 in the 2-qubit active space model:
- Qubit 0 represents the bonding orbital occupation
- Qubit 1 represents the anti-bonding orbital occupation
- HF state: |10> (bonding occupied, anti-bonding empty)
- The double excitation |10> <-> |01> is mediated by X0X1 + Y0Y1 terms*
- `VQEResult` (line 476) `class VQEResult`
- `VQESolver` (line 510) `class VQESolver` - *Variational Quantum Eigensolver.

CORRECTED: Uses proper UCCSD ansatz and Hamiltonian evaluation.*

**Functions:**
- `_make_logger` (line 58) `def _make_logger(name)`

**Methods:**
- `run_vqe_h2` (line 630) `def run_vqe_h2(max_iter)` - *Run VQE for H2 molecule - convenience function.*
- `h2_sto3g` (line 89) `def h2_sto3g(bond_length)` - *Build H2 molecule with STO-3G basis.*
- `_h2_pyscf` (line 96) `def _h2_pyscf(bond_length)` - *Build H2 using PySCF.*
- `_h2_hardcoded` (line 128) `def _h2_hardcoded(bond_length)` - *Build H2 with hardcoded values - CORRECTED for 2-qubit active space.*
- `__init__` (line 163) `def __init__(self, mol, n_qubits)`
- `_build_hamiltonian` (line 170) `def _build_hamiltonian(self)` - *Build the molecular Hamiltonian in JW representation.*
- `_build_openfermion_hamiltonian` (line 178) `def _build_openfermion_hamiltonian(self)` - *Build Hamiltonian using OpenFermion - FIXED API.*
- `_build_hardcoded_hamiltonian` (line 231) `def _build_hardcoded_hamiltonian(self)` - *Build hardcoded H2 Hamiltonian - CORRECTED COEFFICIENTS.

The H2/STO-3G Hamiltonian in the minimal active space (2 qubits)
with Jordan-Wigner transformation.

Derived from reference energies:
- E_HF = -1.11675928 Ha
- E_FCI = -1.13728383 Ha
- E_nuc = 0.71996899 Ha

The Hamiltonian has the form:
H = E_nuc + h1*Z0 + h2*Z1 + h3*Z0*Z1 + h4*X0*X1 + h5*Y0*Y1

For the 2-qubit active space model where:
- |10> is the HF state (bonding orbital occupied)
- The ground state is a superposition of |10> and |01>*
- `_apply_pauli` (line 276) `def _apply_pauli(self, state, pauli)` - *Apply Pauli operator to state vector - CORRECTED.*
- `expectation_value` (line 302) `def expectation_value(self, state)` - *Compute ⟨ψ|H|ψ⟩ for the given state.*
- `evaluate` (line 318) `def evaluate(self, amps)` - *Evaluate energy from amplitudes (supports both numpy and torch).*
- `__call__` (line 330) `def __call__(self, amps)`
- `__init__` (line 345) `def __init__(self, n_qubits, n_electrons)`
- `apply_double_excitation_2q` (line 373) `def apply_double_excitation_2q(self, state, theta)` - *Apply double excitation for 2-qubit H2 model.

This rotates between |10> and |01>:
|10> -> cos(theta)*|10> - sin(theta)*|01>
|01> -> sin(theta)*|10> + cos(theta)*|01>*
- `apply_single_excitation` (line 395) `def apply_single_excitation(self, state, o, v, theta)` - *Apply single excitation as Givens rotation.*
- `apply_double_excitation` (line 416) `def apply_double_excitation(self, state, o1, o2, v1, v2, theta)` - *Apply double excitation for 4+ qubit systems.*
- `apply` (line 443) `def apply(self, state, thetas)` - *Apply UCCSD ansatz to state.*
- `__repr__` (line 490) `def __repr__(self)`
- `__init__` (line 517) `def __init__(self, mol)`
- `prepare_hf_state` (line 528) `def prepare_hf_state(self)` - *Prepare Hartree-Fock state.

For H2 with 2 electrons in 4 spin-orbitals:
|1100⟩ means electrons in orbitals 0 and 1.*
- `run` (line 546) `def run(self, max_iter, tol)` - *Run VQE optimization.*
- `cost` (line 566) `def cost(thetas)`

#### `quantum_framework_molecular_fixed.py`
**Path:** `quantum_framework_molecular_fixed.py`
**File Doc:** *Quantum Framework Molecular Module - FIXED VERSION ==================================================== Molecular simulation with VQE and UCCSD ansatz using MPS representation.  FIXES: - Corrected OpenFermion geometry (charge=0 for H2 neutral) - Fixed hardcoded Hamiltonian coefficients - Fixed _apply_pauli for correct amplitude indexing - Fixed HF state preparation (0s and 1s correctly) - Fixed PySCF atom string syntax - Fixed n_orbitals and n_qubits for H2/STO-3G  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `MoleculeData` (line 64) `class MoleculeData`
- `MoleculeBuilder` (line 77) `class MoleculeBuilder` - *Build molecule data for quantum chemistry calculations.
Uses PySCF when available, falls back to hardcoded values.*
- `ExactJWEnergy` (line 147) `class ExactJWEnergy` - *Exact Jordan-Wigner energy evaluator.
Computes molecular energy using JW transformation.

FIXED: Proper Hamiltonian construction and expectation values.*
- `UCCSDAnsatz` (line 343) `class UCCSDAnsatz` - *Unitary Coupled Cluster Singles and Doubles ansatz.

FIXED: Correct excitation operator implementation.*
- `VQEResult` (line 457) `class VQEResult`
- `VQESolver` (line 491) `class VQESolver` - *Variational Quantum Eigensolver.

FIXED: Correct HF state preparation and energy evaluation.*

**Functions:**
- `_make_logger` (line 50) `def _make_logger(name)`

**Methods:**
- `_get_sd_indices` (line 323) `def _get_sd_indices(n_electrons, n_qubits)` - *Get single and double excitation indices for UCCSD.*
- `run_vqe_demo` (line 614) `def run_vqe_demo()` - *Run a quick VQE demo to verify the fixes.*
- `h2_sto3g` (line 84) `def h2_sto3g(bond_length)` - *Build H2 molecule with STO-3G basis.*
- `_h2_pyscf` (line 91) `def _h2_pyscf(bond_length)` - *Build H2 using PySCF - FIXED atom string syntax.*
- `_h2_hardcoded` (line 126) `def _h2_hardcoded()` - *Build H2 with hardcoded values - FIXED coefficients.*
- `__init__` (line 155) `def __init__(self, mol, n_qubits)`
- `_build_hamiltonian` (line 162) `def _build_hamiltonian(self)` - *Build the molecular Hamiltonian in JW representation.*
- `_build_openfermion_hamiltonian` (line 169) `def _build_openfermion_hamiltonian(self)` - *Build Hamiltonian using OpenFermion - FIXED geometry.*
- `_build_hardcoded_hamiltonian` (line 215) `def _build_hardcoded_hamiltonian(self)` - *Build hardcoded H2 Hamiltonian - FIXED coefficients.

The H2/STO-3G Hamiltonian in JW form (standard reference):
H = g0*I + g1*Z0 + g2*Z1 + g3*Z0*Z1 + g4*X0*X1 + g5*Y0*Y1

With coefficients for bond length 0.735 Å:*
- `_apply_pauli` (line 247) `def _apply_pauli(self, state, pauli)` - *Apply Pauli operator to state vector.

FIXED: Correct amplitude indexing and phase handling.*
- `expectation_value` (line 281) `def expectation_value(self, state)` - *Compute ⟨ψ|H|ψ⟩ for the given state.

FIXED: Correct inner product calculation.*
- `evaluate` (line 305) `def evaluate(self, amps)` - *Evaluate energy from MPS amplitudes.*
- `__call__` (line 319) `def __call__(self, amps)`
- `__init__` (line 350) `def __init__(self, n_qubits, n_electrons, backend)`
- `apply_single_excitation` (line 359) `def apply_single_excitation(self, state, o, v, theta)` - *Apply single excitation operator exp(theta * (a_v† a_o - a_o† a_v)).

In JW, this is a Givens rotation between orbitals o and v.*
- `apply_double_excitation` (line 387) `def apply_double_excitation(self, state, o1, o2, v1, v2, theta)` - *Apply double excitation operator.

Simplified: applies pairwise excitation with rotation.*
- `apply` (line 419) `def apply(self, state, thetas)` - *Apply UCCSD ansatz to state.*
- `__repr__` (line 471) `def __repr__(self)`
- `__init__` (line 498) `def __init__(self, qc, config)`
- `prepare_hf_state` (line 502) `def prepare_hf_state(self, mol)` - *Prepare Hartree-Fock state.

FIXED: Correct bitstring with 0s and 1s.

For H2 with 2 electrons in 4 spin-orbitals:
|1100⟩ means electrons in orbitals 0 and 1 (occupied spin-orbitals)*
- `run` (line 525) `def run(self, mol, backend, max_iter, tol)` - *Run VQE optimization.*
- `cost` (line 556) `def cost(thetas)`

#### `quantum_framework_molecular_v2.py`
**Path:** `quantum_framework_molecular_v2.py`
**File Doc:** *Quantum Framework Molecular Module - Production Version ======================================================== Molecular VQE simulation with OpenFermion integration, MPS optimization, and multiple precision modes.  Features: - OpenFermion for all Hamiltonians (NO hardcoded values) - Precision mode flag: direct statevector vs MPS compression - Integration with Schrodinger, Dirac, Hamiltonian backends - Smart initialization with MP2 + parameter scan - Cached Pauli operations for x10-100 speedup - Particle-conserving MPS for stability - Adaptive bond dimension - TOML configuration  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `MolecularConfig` (line 95) `class MolecularConfig` - *Configuration for molecular VQE simulations.*
- `MoleculeData` (line 172) `class MoleculeData` - *Molecular data structure with all necessary information.*
- `MoleculeBuilder` (line 200) `class MoleculeBuilder` - *Build molecules using OpenFermion + PySCF.

NO hardcoded values - everything comes from quantum chemistry calculations.*
- `HamiltonianBuilder` (line 352) `class HamiltonianBuilder` - *Build molecular Hamiltonians using OpenFermion.

NO hardcoded coefficients - everything from first principles.*
- `CachedPauliOperation` (line 468) `class CachedPauliOperation` - *Precomputed Pauli operation for fast application.*
- `CachedHamiltonianEvaluator` (line 477) `class CachedHamiltonianEvaluator` - *Hamiltonian evaluator with cached Pauli operations.

Improvement: x10-100 speedup for repeated evaluations.*
- `SmartInitializer` (line 613) `class SmartInitializer` - *Smart parameter initialization for VQE.

Improvements:
- MP2 amplitude estimation
- Systematic parameter scan
- Parabolic refinement
- Sensitivity analysis*
- `UCCSDAnsatz` (line 746) `class UCCSDAnsatz` - *Unitary Coupled Cluster Singles and Doubles ansatz.

Features:
- Particle-conserving excitations
- Works with both direct and MPS representations
- Identity check (θ=0 → HF state)*
- `MPSState` (line 879) `class MPSState` - *Matrix Product State for scalable quantum simulation.

Features:
- Adaptive bond dimension
- Efficient gate application
- Entanglement tracking*
- `ParticleConservingState` (line 982) `class ParticleConservingState` - *State representation that preserves particle number symmetry.

Improvement: x4 reduction in Hilbert space, better convergence.*
- `VQEResult` (line 1064) `class VQEResult` - *VQE result container.*
- `VQESolver` (line 1104) `class VQESolver` - *Production VQE solver with all improvements.

Features:
- OpenFermion Hamiltonians (no hardcoded values)
- Precision mode flag (direct vs MPS)
- Smart initialization with MP2 + scan
- Cached Pauli operations
- Particle conservation option
- Backend integration*
- `BackendIntegrator` (line 1287) `class BackendIntegrator` - *Integration with existing backends (Hamiltonian, Schrodinger, Dirac).

Uses pre-trained models from QC repository.*
- `PseudoMolData` (line 301) `class PseudoMolData`

**Functions:**
- `_make_logger` (line 75) `def _make_logger(name)`

**Methods:**
- `run_vqe` (line 1332) `def run_vqe(molecule, precision_mode, config_path)` - *Convenience function to run VQE.

Args:
    molecule: Molecule name ("H2", "H2O", "LiH")
    precision_mode: Use direct statevector (True) or MPS (False)
    config_path: Path to TOML config file
    **kwargs: Additional config overrides

Returns:
    VQEResult*
- `from_toml` (line 131) `def from_toml(cls, toml_path)` - *Load configuration from TOML file.*
- `build` (line 208) `def build(name, geometry, basis, charge, multiplicity, description)` - *Build molecule using OpenFermion.

Args:
    name: Molecule name (e.g., "H2", "H2O")
    geometry: List of (atom_symbol, (x, y, z)) in Angstrom
    basis: Basis set (e.g., "sto-3g", "6-31g")
    charge: Molecular charge
    multiplicity: Spin multiplicity
    description: Optional description

Returns:
    MoleculeData with all properties computed*
- `_run_pyscf_direct` (line 289) `def _run_pyscf_direct(geometry, basis, charge, multiplicity)` - *Run PySCF directly if openfermionpyscf not available.*
- `h2` (line 324) `def h2(bond_length, basis)` - *Build H2 molecule.*
- `h2o` (line 330) `def h2o(bond_length_oh, angle_hoh, basis)` - *Build H2O molecule.*
- `lih` (line 342) `def lih(bond_length, basis)` - *Build LiH molecule.*
- `build_jw_hamiltonian` (line 360) `def build_jw_hamiltonian(mol)` - *Build Jordan-Wigner transformed Hamiltonian using OpenFermion.

Args:
    mol: MoleculeData with geometry, basis, etc.

Returns:
    (pauli_terms, nuclear_repulsion)*
- `build_hamiltonian_matrix` (line 417) `def build_hamiltonian_matrix(mol, n_qubits)` - *Build full Hamiltonian matrix for small systems.

Args:
    mol: MoleculeData
    n_qubits: Number of qubits

Returns:
    Hamiltonian matrix (2^n_qubits, 2^n_qubits)*
- `_pauli_matrix` (line 441) `def _pauli_matrix(pauli_list, n_qubits)` - *Build matrix for a Pauli string.*
- `__init__` (line 484) `def __init__(self, mol, config)`
- `_precompute_operations` (line 503) `def _precompute_operations(self)` - *Precompute all Pauli operations for fast evaluation.*
- `_compute_pauli_mapping` (line 522) `def _compute_pauli_mapping(self, pauli_list)` - *Compute index mapping and phases for a Pauli string.*
- `apply_pauli_fast` (line 561) `def apply_pauli_fast(self, state, op)` - *Apply cached Pauli operation.*
- `expectation_value` (line 568) `def expectation_value(self, state)` - *Compute energy expectation value.*
- `batch_expectation` (line 587) `def batch_expectation(self, states)` - *Compute expectation for batch of states.*
- `__init__` (line 624) `def __init__(self, mol, ansatz, evaluator, config)`
- `estimate_mp2_amplitude` (line 630) `def estimate_mp2_amplitude(self)` - *Estimate doubles amplitude from MP2 theory.

θ_MP2 ≈ t_2^(1) / 2
where t_2^(1) = <ij||ab> / (ε_i + ε_j - ε_a - ε_b)*
- `scan_parameter_space` (line 657) `def scan_parameter_space(self, hf_state, n_samples, param_range)` - *Systematic scan of parameter space.

Returns:
    (best_thetas, best_energy)*
- `_parabolic_refinement` (line 712) `def _parabolic_refinement(self, x_vals, y_vals, best_idx, n_params)` - *Refine minimum using parabolic interpolation.*
- `initialize` (line 735) `def initialize(self, hf_state)` - *Complete initialization with all techniques.*
- `__init__` (line 756) `def __init__(self, n_qubits, n_electrons, config)`
- `_generate_excitations` (line 764) `def _generate_excitations(self)` - *Generate all single and double excitations.*
- `apply` (line 785) `def apply(self, state, thetas)` - *Apply UCCSD ansatz to state.

Args:
    state: State vector (2^n_qubits,)
    thetas: Parameters (n_params,)

Returns:
    Transformed state vector*
- `_apply_single` (line 817) `def _apply_single(self, state, i, a, theta)` - *Apply single excitation as Givens rotation.*
- `_apply_double` (line 838) `def _apply_double(self, state, i, j, a, b, theta)` - *Apply double excitation.*
- `verify_identity` (line 862) `def verify_identity(self, hf_state, evaluator, hf_energy)` - *Verify that θ=0 gives HF state.*
- `__init__` (line 889) `def __init__(self, n_qubits, config)`
- `to_statevector` (line 910) `def to_statevector(self)` - *Convert MPS to full statevector.*
- `from_statevector` (line 918) `def from_statevector(cls, state, n_qubits, config)` - *Create MPS from statevector.*
- `compute_entanglement` (line 953) `def compute_entanglement(self, bond_idx)` - *Compute entanglement entropy at bond.*
- `__init__` (line 989) `def __init__(self, n_qubits, n_particles)`
- `_generate_fock_states` (line 1000) `def _generate_fock_states(self)` - *Generate all states with fixed particle number.*
- `hf_state` (line 1013) `def hf_state(self)` - *Create HF state in subspace.*
- `apply_excitation` (line 1027) `def apply_excitation(self, state, occ, vir, theta)` - *Apply excitation preserving particle number.*
- `to_full_statevector` (line 1051) `def to_full_statevector(self, state)` - *Convert subspace state to full statevector.*
- `__repr__` (line 1081) `def __repr__(self)`
- `__init__` (line 1117) `def __init__(self, mol, config)`
- `prepare_hf_state` (line 1137) `def prepare_hf_state(self)` - *Prepare Hartree-Fock state.*
- `evaluate` (line 1152) `def evaluate(self, state)` - *Evaluate energy.*
- `apply_ansatz` (line 1159) `def apply_ansatz(self, state, thetas)` - *Apply UCCSD ansatz.*
- `run` (line 1184) `def run(self)` - *Run VQE optimization.*
- `__init__` (line 1294) `def __init__(self, config)`
- `_load_models` (line 1301) `def _load_models(self)` - *Load pre-trained backend models.*
- `get_backend_energy` (line 1318) `def get_backend_energy(self, state, backend_name)` - *Get energy estimate from backend model.*
- `cost` (line 1212) `def cost(thetas)`

#### `quantum_framework_physics.py`
**Path:** `quantum_framework_physics.py`
**File Doc:** *Quantum Framework Physics Module ================================ Neural network backends for quantum physics simulations. Provides spectral layers, Hamiltonian networks, and Dirac operators.  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `SpectralLayer` (line 26) `class SpectralLayer(Module)` - *Spectral convolution in frequency domain.
Learns complex kernels that modulate Fourier coefficients.*
- `HamiltonianBackboneNet` (line 66) `class HamiltonianBackboneNet(Module)` - *Hamiltonian backbone: single-channel field -> H|psi>.
Shared by all physics backends as the H operator.*
- `SchrodingerSpectralNet` (line 92) `class SchrodingerSpectralNet(Module)` - *Schrodinger network: 2-channel [psi_real, psi_imag] -> evolved wavefunction.
Uses spectral convolution for physics-informed evolution.*
- `DiracSpectralNet` (line 120) `class DiracSpectralNet(Module)` - *Dirac network: 8-channel spinor [4 components x (real,imag)] -> evolved spinor.
Handles 4-component spinor evolution for relativistic quantum mechanics.*
- `GammaMatrices` (line 150) `class GammaMatrices` - *Dirac gamma matrices in Dirac (standard) or Weyl representation.
gamma^0 = beta, gamma^i = beta * alpha_i*
- `PotentialGenerator` (line 250) `class PotentialGenerator` - *Spatial potentials for eigenstate initialization.*
- `DiracHamiltonianOperator` (line 294) `class DiracHamiltonianOperator` - *Dirac Hamiltonian operator for relativistic quantum mechanics.
H_Dirac = c * alpha . p + beta * m * c^2 + V(r)

In atomic units (c = 1/alpha ~ 137):
H = c * alpha . p + beta * m * c^2 + V*
- `LambShiftCalculator` (line 382) `class LambShiftCalculator` - *Calculates the Lamb shift using Bethe's formula.
The Lamb shift is the small difference between 2s_{1/2} and 2p_{1/2} energy levels
due to QED effects (vacuum fluctuations and self-energy).*
- `AnomalousMagneticMoment` (line 439) `class AnomalousMagneticMoment` - *Calculates the electron's anomalous magnetic moment (g-2).
The electron g-factor is slightly different from 2 due to QED effects:
g = 2(1 + a_e) where a_e = alpha/(2*pi) + higher-order terms*
- `DiracHydrogenAtom` (line 489) `class DiracHydrogenAtom` - *Relativistic hydrogen atom with Dirac equation.
Computes energy levels including fine structure.*
- `ZitterbewegungSimulator` (line 571) `class ZitterbewegungSimulator` - *Simulates the Zitterbewegung (trembling motion) of a relativistic electron.
In Dirac theory, the position operator has a term oscillating with frequency
~ 2mc^2/hbar, which is the interference between positive and negative energy states.*

**Methods:**
- `__init__` (line 32) `def __init__(self, channels, grid_size)`
- `forward` (line 43) `def forward(self, x)`
- `__init__` (line 72) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 81) `def forward(self, x)`
- `__init__` (line 98) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 109) `def forward(self, x)`
- `__init__` (line 126) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 139) `def forward(self, x)`
- `__init__` (line 156) `def __init__(self, representation, device)`
- `_init_matrices` (line 161) `def _init_matrices(self)`
- `to` (line 244) `def to(self, device)`
- `__init__` (line 253) `def __init__(self, grid_size, potential_depth, potential_width)`
- `_grid` (line 258) `def _grid(self)`
- `harmonic` (line 263) `def harmonic(self)`
- `double_well` (line 268) `def double_well(self)`
- `coulomb` (line 274) `def coulomb(self)`
- `periodic_lattice` (line 280) `def periodic_lattice(self)`
- `mixed` (line 284) `def mixed(self, seed)`
- `__init__` (line 303) `def __init__(self, grid_size, electron_mass, c_light, device)`
- `_precompute_operators` (line 311) `def _precompute_operators(self)`
- `apply_dirac_hamiltonian` (line 318) `def apply_dirac_hamiltonian(self, spinor, potential)` - *Apply Dirac Hamiltonian to 4-component spinor.

Args:
    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor
    potential: Optional scalar potential V(r)

Returns:
    H * psi with same shape as input*
- `time_evolution` (line 362) `def time_evolution(self, spinor, dt, potential, normalization_eps)` - *Time evolution of Dirac spinor using first-order split-step.
psi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi*
- `__init__` (line 389) `def __init__(self, alpha_fs, c_light, electron_mass)`
- `bethe_formula` (line 394) `def bethe_formula(self, n, l, Z)` - *Bethe's non-relativistic formula for Lamb shift.
Delta E_Lamb = (8*alpha^3 / 3*pi*n^3) * |psi_n(0)|^2 * ln(E_avg / E_n)*
- `_higher_l_shift` (line 407) `def _higher_l_shift(self, n, l, Z)`
- `full_lamb_shift` (line 412) `def full_lamb_shift(self, n, l, j, Z)` - *Calculate full Lamb shift including radiative corrections.
Delta E = Delta E_SE + Delta E_Uehling + Delta E_rel*
- `__init__` (line 446) `def __init__(self, alpha_fs)`
- `schwinger_term` (line 449) `def schwinger_term(self)`
- `second_order` (line 452) `def second_order(self)`
- `third_order` (line 456) `def third_order(self)`
- `fourth_order` (line 460) `def fourth_order(self)`
- `fifth_order` (line 464) `def fifth_order(self)`
- `calculate_a_e` (line 468) `def calculate_a_e(self, order)`
- `__init__` (line 495) `def __init__(self, c_light, alpha_fs)`
- `energy_level_dirac` (line 499) `def energy_level_dirac(self, n, kappa)` - *Exact Dirac energy level for hydrogen-like atom.
E = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)*
- `fine_structure_splitting` (line 512) `def fine_structure_splitting(self, n, l)` - *Calculate fine structure splitting for given n, l.
Returns energies for j = l+1/2 and j = l-1/2*
- `energy_spectrum` (line 535) `def energy_spectrum(self, n_max)`
- `__init__` (line 578) `def __init__(self, grid_size, c_light, electron_mass, device)`
- `create_gaussian_wave_packet` (line 585) `def create_gaussian_wave_packet(self, sigma, momentum)`
- `compute_position_expectation` (line 603) `def compute_position_expectation(self, spinor)`
- `compute_velocity_expectation` (line 615) `def compute_velocity_expectation(self, spinor)`

#### `quantum_framework_visualization.py`
**Path:** `quantum_framework_visualization.py`
**File Doc:** *Quantum Framework Visualization Module ======================================== Orbital visualization, Monte Carlo sampling, and entangled state visualization.  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `WavefunctionCalculator` (line 59) `class WavefunctionCalculator` - *Calculates hydrogen atom wavefunctions for visualization.
Uses analytical formulas for radial and angular parts.*
- `MonteCarloSampler` (line 115) `class MonteCarloSampler` - *Monte Carlo sampling for orbital visualization.
Uses rejection sampling to generate 3D point clouds.*
- `OrbitalVisualizer` (line 202) `class OrbitalVisualizer` - *High-resolution visualization of hydrogen orbitals.
Creates 2D projections and 3D scatter plots.*
- `EntangledHydrogenSampler` (line 290) `class EntangledHydrogenSampler` - *Monte Carlo sampler for entangled hydrogen states.
Samples from joint probability distribution of entangled orbitals.*
- `EntangledHydrogenVisualizer` (line 325) `class EntangledHydrogenVisualizer` - *Visualizer for entangled hydrogen states.
Creates high-resolution visualizations with multiple orbitals.*

**Functions:**
- `_make_logger` (line 46) `def _make_logger(name)`

**Methods:**
- `__init__` (line 65) `def __init__(self, config)`
- `radial_wavefunction` (line 69) `def radial_wavefunction(n, l, r)`
- `spherical_harmonic_real` (line 81) `def spherical_harmonic_real(l, m, theta, phi)`
- `psi_3d` (line 92) `def psi_3d(self, n, l, m, r, theta, phi)`
- `psi_on_grid` (line 97) `def psi_on_grid(self, n, l, m)`
- `__init__` (line 121) `def __init__(self, config, wavefunction_calc)`
- `find_max_probability` (line 125) `def find_max_probability(self, n, l, m)`
- `sample` (line 149) `def sample(self, n, l, m, num_samples)`
- `__init__` (line 208) `def __init__(self, config)`
- `visualize` (line 211) `def visualize(self, data, save_path)`
- `__init__` (line 296) `def __init__(self, config, wavefunction_calc)`
- `sample_entangled_state` (line 301) `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples, entanglement_weight)`
- `__init__` (line 331) `def __init__(self, config)`
- `visualize` (line 334) `def visualize(self, data, quantum_result, save_path)`

#### `quantum_lab.py`
**Path:** `quantum_lab.py`
**File Doc:** *Q2C Quantum Lab - Interactive Educational TUI ============================================= A guided, bilingual (English/Spanish) terminal experience for learning quantum computing and quantum chemistry with the Q2C simulator.  It is built on top of the real simulation engine (quantum_framework_core): every probability bar, entropy value and orbital picture you see is computed live, not pre-recorded.  Usage: python quantum_lab.py                # interactive, asks for language python quantum_lab.py --lang es      # start in Spanish python quantum_lab.py --lesson 3     # jump straight into lesson 3  Sections: 1. Lessons     - guided course: qubits, measurement, entanglement, Grover search and quantum chemistry 2. Playground  - build circuits gate by gate, watch the state evolve 3. Chemistry   - molecule cards and an ASCII hydrogen-orbital viewer 4. Glossary    - quick reference of the vocabulary used everywhere  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `OrbitalSpec` (line 431) `class OrbitalSpec` - *Definition of a real hydrogen orbital for the ASCII viewer.*
- `H2VQEEngine` (line 604) `class H2VQEEngine` - *Minimal live VQE for H2 in the 2-qubit active space.

The trial state cos(theta/2)|01> + sin(theta/2)|10> is prepared with a
real circuit (Ry, CNOT, X) on the MPS engine, and the energy is read
from the Jordan-Wigner H2 Hamiltonian of quantum_framework_molecular.*
- `Quiz` (line 659) `class Quiz`
- `Lesson` (line 667) `class Lesson`
- `QuantumLab` (line 672) `class QuantumLab` - *Interactive educational TUI driven by the real Q2C engine.*

**Functions:**
- `probability_bars` (line 325) `def probability_bars(probs, n_qubits, lang, max_rows)` - *Build a table of probability bars for a state's distribution.*
- `counts_bars` (line 353) `def counts_bars(counts, total, lang)` - *Build a table of measurement-count bars.*
- `draw_circuit` (line 369) `def draw_circuit(n_qubits, instructions)` - *Render an ASCII timeline of the circuit, one line per qubit.*
- `parse_angle` (line 394) `def parse_angle(token)` - *Parse an angle like '1.57', 'pi', '-pi/2' or '3*pi/4' into radians.*
- `sample_measurements` (line 414) `def sample_measurements(probs, n_qubits, n_samples, rng)` - *Draw measurement outcomes from a probability distribution.*

**Methods:**
- `_radial` (line 440) `def _radial(n, l, r)` - *Hydrogen radial wavefunction R_nl(r) in atomic units.*
- `_ang_s` (line 452) `def _ang_s(x, y, z, r)`
- `_ang_pz` (line 456) `def _ang_pz(x, y, z, r)`
- `_ang_px` (line 460) `def _ang_px(x, y, z, r)`
- `_ang_dz2` (line 464) `def _ang_dz2(x, y, z, r)`
- `_ang_dxz` (line 469) `def _ang_dxz(x, y, z, r)`
- `_ang_dxy` (line 473) `def _ang_dxy(x, y, z, r)`
- `field_to_text` (line 490) `def field_to_text(psi)` - *Render a real scalar field as colored ASCII: brightness = |psi|^2, color = sign.*
- `render_orbital` (line 516) `def render_orbital(spec, rows, cols)` - *Render |psi|^2 of a hydrogen orbital on a plane slice as colored ASCII.*
- `render_h2_molecular_orbital` (line 536) `def render_h2_molecular_orbital(kind, rows, cols)` - *Render the bonding or antibonding LCAO molecular orbital of H2.*
- `landscape_plot` (line 556) `def landscape_plot(energies, e_hf, e_fci, marker, rows)` - *Draw an ASCII plot of E(theta) over one full period with HF and FCI
reference lines and an optional optimizer marker.*
- `launch_quantum_lab` (line 1474) `def launch_quantum_lab(config, loader, lang, lesson)` - *Entry point used both standalone and from quantum_framework_main.*
- `main` (line 1500) `def main()`
- `row_of` (line 567) `def row_of(e)`
- `__init__` (line 613) `def __init__(self, qc)`
- `ansatz_instructions` (line 621) `def ansatz_instructions(self, theta)`
- `energy` (line 624) `def energy(self, theta)`
- `correlation_pct` (line 633) `def correlation_pct(self, e)`
- `landscape` (line 636) `def landscape(self, cols)`
- `optimize` (line 640) `def optimize(self, theta0, lr, max_iters, tol)` - *Gradient descent; yields (iteration, theta, energy) live.*
- `__init__` (line 675) `def __init__(self, config, loader, lang)`
- `t` (line 683) `def t(self, key)`
- `pause` (line 690) `def pause(self)`
- `panel` (line 697) `def panel(self, body, title, style)`
- `show_state` (line 701) `def show_state(self, probs, n_qubits, title)`
- `run_quiz` (line 707) `def run_quiz(self, quiz)`
- `lessons` (line 728) `def lessons(self)`
- `_demo_superposition` (line 733) `def _demo_superposition(self)`
- `_demo_rotation` (line 744) `def _demo_rotation(self)`
- `_demo_measurement` (line 766) `def _demo_measurement(self)`
- `_demo_bell` (line 779) `def _demo_bell(self)`
- `_demo_ghz_w` (line 792) `def _demo_ghz_w(self)`
- `_demo_grover` (line 799) `def _demo_grover(self)`
- `_demo_molecule` (line 815) `def _demo_molecule(self)`
- `_vqe` (line 821) `def _vqe(self)`
- `_demo_h2_clouds` (line 826) `def _demo_h2_clouds(self)`
- `_demo_vqe_ansatz` (line 835) `def _demo_vqe_ansatz(self)`
- `_demo_vqe_landscape` (line 840) `def _demo_vqe_landscape(self)`
- `_demo_vqe_live` (line 847) `def _demo_vqe_live(self)`
- `_lessons_en` (line 875) `def _lessons_en(self)`
- `_lessons_es` (line 1015) `def _lessons_es(self)`
- `run_lesson` (line 1159) `def run_lesson(self, lesson)`
- `lessons_menu` (line 1175) `def lessons_menu(self)`
- `_rebuild_state` (line 1198) `def _rebuild_state(self, n_qubits, instructions)`
- `_playground_dashboard` (line 1216) `def _playground_dashboard(self, n_qubits, state, instructions)`
- `_show_amplitudes` (line 1228) `def _show_amplitudes(self, state, n_qubits)`
- `playground` (line 1249) `def playground(self)`
- `_molecule_card` (line 1346) `def _molecule_card(self, mol)`
- `molecule_explorer` (line 1365) `def molecule_explorer(self)`
- `orbital_viewer` (line 1383) `def orbital_viewer(self)`
- `chemistry_menu` (line 1403) `def chemistry_menu(self)`
- `glossary` (line 1422) `def glossary(self)`
- `banner` (line 1437) `def banner(self)`
- `main_menu` (line 1443) `def main_menu(self)`

#### `quantum_simulator.py`
**Path:** `quantum_simulator.py`
**File Doc:** *Quantum Simulation Framework - Complete Edition ================================================ Self-contained quantum simulation framework with: - Quantum circuit simulation (Hamiltonian, Schrodinger, Dirac backends) - Atomic orbital visualization with Monte Carlo sampling - Molecular VQE/UCCSD simulation - Relativistic hydrogen with fine structure and Zitterbewegung - Entangled state visualization - Multi-atom orbital support  All configuration via TOML file - no hardcoded values.*

**Classes:**
- `FrameworkConfig` (line 68) `class FrameworkConfig`
- `AtomData` (line 164) `class AtomData`
- `MoleculeData` (line 174) `class MoleculeData`
- `OrbitalData` (line 192) `class OrbitalData`
- `ConfigLoader` (line 200) `class ConfigLoader`
- `SpectralLayer` (line 288) `class SpectralLayer(Module)`
- `HamiltonianBackboneNet` (line 303) `class HamiltonianBackboneNet(Module)`
- `SchrodingerSpectralNet` (line 321) `class SchrodingerSpectralNet(Module)`
- `DiracSpectralNet` (line 341) `class DiracSpectralNet(Module)`
- `GammaMatrices` (line 361) `class GammaMatrices`
- `JointHilbertState` (line 380) `class JointHilbertState`
- `IPhysicsBackend` (line 410) `class IPhysicsBackend(ABC)`
- `HamiltonianBackend` (line 420) `class HamiltonianBackend(IPhysicsBackend)`
- `SchrodingerBackend` (line 470) `class SchrodingerBackend(IPhysicsBackend)`
- `DiracBackend` (line 505) `class DiracBackend(IPhysicsBackend)`
- `IQuantumGate` (line 638) `class IQuantumGate(ABC)`
- `HadamardGate` (line 649) `class HadamardGate(IQuantumGate)`
- `PauliXGate` (line 662) `class PauliXGate(IQuantumGate)`
- `PauliYGate` (line 674) `class PauliYGate(IQuantumGate)`
- `PauliZGate` (line 686) `class PauliZGate(IQuantumGate)`
- `SGate` (line 698) `class SGate(IQuantumGate)`
- `TGate` (line 710) `class TGate(IQuantumGate)`
- `RxGate` (line 723) `class RxGate(IQuantumGate)`
- `RyGate` (line 737) `class RyGate(IQuantumGate)`
- `RzGate` (line 751) `class RzGate(IQuantumGate)`
- `CNOTGate` (line 766) `class CNOTGate(IQuantumGate)`
- `CZGate` (line 778) `class CZGate(IQuantumGate)`
- `SWAPGate` (line 790) `class SWAPGate(IQuantumGate)`
- `ToffoliGate` (line 802) `class ToffoliGate(IQuantumGate)`
- `CircuitInstruction` (line 832) `class CircuitInstruction`
- `QuantumCircuit` (line 838) `class QuantumCircuit`
- `QuantumResult` (line 894) `class QuantumResult`
- `PotentialGenerator` (line 908) `class PotentialGenerator`
- `JointStateFactory` (line 972) `class JointStateFactory`
- `QuantumComputer` (line 999) `class QuantumComputer`
- `WavefunctionCalculator` (line 1043) `class WavefunctionCalculator`
- `MonteCarloSampler` (line 1080) `class MonteCarloSampler`
- `DiracHydrogenAtom` (line 1159) `class DiracHydrogenAtom`
- `ZitterbewegungSimulator` (line 1200) `class ZitterbewegungSimulator`
- `OrbitalVisualizer` (line 1282) `class OrbitalVisualizer`
- `EntangledVisualizer` (line 1343) `class EntangledVisualizer`
- `QuantumSimulationFramework` (line 1409) `class QuantumSimulationFramework`
- `InteractiveMenu` (line 1535) `class InteractiveMenu`

**Functions:**
- `_make_logger` (line 54) `def _make_logger(name, level)`

**Methods:**
- `_single_qubit_unitary` (line 587) `def _single_qubit_unitary(state, qubit, u, backend)`
- `_two_qubit_unitary` (line 611) `def _two_qubit_unitary(state, ctrl, tgt, u4)`
- `_solve_eigenstate` (line 949) `def _solve_eigenstate(config, potential, n)`
- `_build_basis_amplitude` (line 965) `def _build_basis_amplitude(config, basis_idx)`
- `main` (line 1868) `def main()`
- `from_toml` (line 112) `def from_toml(cls, toml_path)`
- `__init__` (line 201) `def __init__(self, config_path)`
- `_find_config` (line 209) `def _find_config(self)`
- `_load` (line 219) `def _load(self)`
- `_load_defaults` (line 229) `def _load_defaults(self)`
- `_parse_atoms` (line 236) `def _parse_atoms(self)`
- `_parse_molecules` (line 241) `def _parse_molecules(self)`
- `_parse_orbitals` (line 246) `def _parse_orbitals(self)`
- `get_atom` (line 251) `def get_atom(self, symbol)`
- `get_molecule` (line 259) `def get_molecule(self, name)`
- `get_orbital` (line 267) `def get_orbital(self, name)`
- `atoms` (line 276) `def atoms(self)`
- `molecules` (line 280) `def molecules(self)`
- `orbitals` (line 284) `def orbitals(self)`
- `__init__` (line 289) `def __init__(self, channels, grid_size)`
- `forward` (line 295) `def forward(self, x)`
- `__init__` (line 304) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 310) `def forward(self, x)`
- `__init__` (line 322) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 330) `def forward(self, x)`
- `__init__` (line 342) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers)`
- `forward` (line 350) `def forward(self, x)`
- `__init__` (line 362) `def __init__(self, representation, device)`
- `__init__` (line 381) `def __init__(self, amplitudes, n_qubits)`
- `normalize_` (line 388) `def normalize_(self)`
- `probabilities` (line 393) `def probabilities(self)`
- `entropy` (line 397) `def entropy(self)`
- `most_probable_bitstring` (line 402) `def most_probable_bitstring(self)`
- `clone` (line 406) `def clone(self)`
- `evolve_amplitude` (line 412) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 416) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 421) `def __init__(self, config)`
- `_load` (line 429) `def _load(self)`
- `_precompute_laplacian` (line 443) `def _precompute_laplacian(self)`
- `_apply_h` (line 450) `def _apply_h(self, field)`
- `evolve_amplitude` (line 458) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 465) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 471) `def __init__(self, config, hamiltonian)`
- `_load` (line 478) `def _load(self)`
- `evolve_amplitude` (line 493) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 501) `def apply_phase(self, amp, phase_angle)`
- `__init__` (line 506) `def __init__(self, config, hamiltonian)`
- `_load` (line 515) `def _load(self)`
- `_precompute_dirac` (line 530) `def _precompute_dirac(self)`
- `_pack` (line 537) `def _pack(self, amp)`
- `_unpack` (line 547) `def _unpack(self, spinor)`
- `_analytical_dirac` (line 553) `def _analytical_dirac(self, spinor)`
- `evolve_amplitude` (line 564) `def evolve_amplitude(self, amp, dt)`
- `apply_phase` (line 577) `def apply_phase(self, amp, phase_angle)`
- `evolve_spinor` (line 580) `def evolve_spinor(self, spinor, dt)`
- `name` (line 641) `def name(self)`
- `apply` (line 645) `def apply(self, state, backend, targets, params)`
- `name` (line 651) `def name(self)`
- `apply` (line 654) `def apply(self, state, backend, targets, params)`
- `name` (line 664) `def name(self)`
- `apply` (line 667) `def apply(self, state, backend, targets, params)`
- `name` (line 676) `def name(self)`
- `apply` (line 679) `def apply(self, state, backend, targets, params)`
- `name` (line 688) `def name(self)`
- `apply` (line 691) `def apply(self, state, backend, targets, params)`
- `name` (line 700) `def name(self)`
- `apply` (line 703) `def apply(self, state, backend, targets, params)`
- `name` (line 712) `def name(self)`
- `apply` (line 715) `def apply(self, state, backend, targets, params)`
- `name` (line 725) `def name(self)`
- `apply` (line 728) `def apply(self, state, backend, targets, params)`
- `name` (line 739) `def name(self)`
- `apply` (line 742) `def apply(self, state, backend, targets, params)`
- `name` (line 753) `def name(self)`
- `apply` (line 756) `def apply(self, state, backend, targets, params)`
- `name` (line 768) `def name(self)`
- `apply` (line 771) `def apply(self, state, backend, targets, params)`
- `name` (line 780) `def name(self)`
- `apply` (line 783) `def apply(self, state, backend, targets, params)`
- `name` (line 792) `def name(self)`
- `apply` (line 795) `def apply(self, state, backend, targets, params)`
- `name` (line 804) `def name(self)`
- `apply` (line 807) `def apply(self, state, backend, targets, params)`
- `__init__` (line 839) `def __init__(self, n_qubits)`
- `_append` (line 843) `def _append(self, gate_name, targets, params)`
- `h` (line 846) `def h(self, qubit)`
- `x` (line 849) `def x(self, qubit)`
- `y` (line 852) `def y(self, qubit)`
- `z` (line 855) `def z(self, qubit)`
- `s` (line 858) `def s(self, qubit)`
- `t` (line 861) `def t(self, qubit)`
- `rx` (line 864) `def rx(self, qubit, theta)`
- `ry` (line 867) `def ry(self, qubit, theta)`
- `rz` (line 870) `def rz(self, qubit, theta)`
- `cnot` (line 873) `def cnot(self, control, target)`
- `cz` (line 876) `def cz(self, control, target)`
- `swap` (line 879) `def swap(self, qubit1, qubit2)`
- `ccx` (line 882) `def ccx(self, ctrl0, ctrl1, target)`
- `run` (line 885) `def run(self, state, backend)`
- `__init__` (line 895) `def __init__(self, state)`
- `entropy` (line 898) `def entropy(self)`
- `most_probable_bitstring` (line 901) `def most_probable_bitstring(self)`
- `probabilities` (line 904) `def probabilities(self)`
- `__init__` (line 909) `def __init__(self, config)`
- `_grid` (line 913) `def _grid(self)`
- `harmonic` (line 918) `def harmonic(self)`
- `double_well` (line 923) `def double_well(self)`
- `coulomb` (line 929) `def coulomb(self)`
- `periodic_lattice` (line 935) `def periodic_lattice(self)`
- `mixed` (line 939) `def mixed(self, seed)`
- `__init__` (line 973) `def __init__(self, config)`
- `_empty` (line 976) `def _empty(self, n_qubits)`
- `all_zeros` (line 979) `def all_zeros(self, n_qubits)`
- `basis_state` (line 986) `def basis_state(self, n_qubits, k)`
- `from_bitstring` (line 995) `def from_bitstring(self, bitstring)`
- `__init__` (line 1000) `def __init__(self, config)`
- `create_circuit` (line 1011) `def create_circuit(self, n_qubits)`
- `run_circuit` (line 1014) `def run_circuit(self, circuit, initial_state, backend)`
- `bell_state` (line 1021) `def bell_state(self, backend)`
- `ghz_state` (line 1027) `def ghz_state(self, n_qubits, backend)`
- `factory` (line 1035) `def factory(self)`
- `backends` (line 1039) `def backends(self)`
- `__init__` (line 1044) `def __init__(self, config)`
- `radial_wavefunction` (line 1048) `def radial_wavefunction(n, l, r)`
- `spherical_harmonic_real` (line 1060) `def spherical_harmonic_real(l, m, theta, phi)`
- `psi_3d` (line 1071) `def psi_3d(self, n, l, m, r, theta, phi)`
- `energy_analytical` (line 1076) `def energy_analytical(self, n)`
- `__init__` (line 1081) `def __init__(self, config, wavefunction_calc)`
- `find_max_probability` (line 1085) `def find_max_probability(self, n, l, m)`
- `sample_orbital` (line 1109) `def sample_orbital(self, n, l, m, num_samples, Z)`
- `sample_entangled_state` (line 1146) `def sample_entangled_state(self, n1, l1, m1, n2, l2, m2, num_samples)`
- `__init__` (line 1160) `def __init__(self, config)`
- `energy_level_dirac` (line 1165) `def energy_level_dirac(self, n, kappa, Z)`
- `energy_schrodinger` (line 1173) `def energy_schrodinger(self, n, Z)`
- `fine_structure_splitting` (line 1176) `def fine_structure_splitting(self, n, l, Z)`
- `energy_spectrum` (line 1184) `def energy_spectrum(self, n_max, Z)`
- `__init__` (line 1201) `def __init__(self, config, dirac_backend)`
- `create_gaussian_wave_packet` (line 1206) `def create_gaussian_wave_packet(self, sigma, momentum)`
- `compute_position_expectation` (line 1224) `def compute_position_expectation(self, spinor)`
- `compute_velocity_expectation` (line 1235) `def compute_velocity_expectation(self, spinor)`
- `simulate` (line 1246) `def simulate(self, duration, dt, sigma)`
- `__init__` (line 1283) `def __init__(self, config)`
- `visualize` (line 1286) `def visualize(self, data, save_path, title_suffix)`
- `__init__` (line 1344) `def __init__(self, config)`
- `visualize` (line 1347) `def visualize(self, data, quantum_result, save_path)`
- `__init__` (line 1410) `def __init__(self, config_path)`
- `_ensure_output_dir` (line 1422) `def _ensure_output_dir(self)`
- `list_available_atoms` (line 1425) `def list_available_atoms(self)`
- `list_available_molecules` (line 1428) `def list_available_molecules(self)`
- `list_available_orbitals` (line 1431) `def list_available_orbitals(self)`
- `get_atom` (line 1434) `def get_atom(self, symbol)`
- `get_molecule` (line 1437) `def get_molecule(self, name)`
- `get_orbital` (line 1440) `def get_orbital(self, name)`
- `run_quantum_circuit` (line 1443) `def run_quantum_circuit(self, circuit, backend)`
- `visualize_orbital` (line 1446) `def visualize_orbital(self, orbital_name, num_samples, save, Z, title_suffix)`
- `visualize_atom_orbitals` (line 1458) `def visualize_atom_orbitals(self, atom_symbol, num_samples, save)`
- `visualize_entangled_state` (line 1482) `def visualize_entangled_state(self, orbital1, orbital2, num_samples, save)`
- `compute_relativistic_energy` (line 1496) `def compute_relativistic_energy(self, n, l, Z)`
- `compute_energy_spectrum` (line 1499) `def compute_energy_spectrum(self, n_max, Z)`
- `run_zitterbewegung_simulation` (line 1502) `def run_zitterbewegung_simulation(self, duration, dt, sigma)`
- `run_all_demonstrations` (line 1505) `def run_all_demonstrations(self, num_samples)`
- `__init__` (line 1536) `def __init__(self, framework)`
- `display_header` (line 1540) `def display_header(self)`
- `display_main_menu` (line 1548) `def display_main_menu(self)`
- `get_user_choice` (line 1564) `def get_user_choice(self, prompt)`
- `_display_quantum_result` (line 1570) `def _display_quantum_result(self, result, n_qubits)`
- `orbital_menu` (line 1578) `def orbital_menu(self)`
- `atom_orbital_menu` (line 1602) `def atom_orbital_menu(self)`
- `entangled_menu` (line 1622) `def entangled_menu(self)`
- `quantum_circuit_menu` (line 1651) `def quantum_circuit_menu(self)`
- `relativistic_menu` (line 1737) `def relativistic_menu(self)`
- `zitterbewegung_menu` (line 1777) `def zitterbewegung_menu(self)`
- `molecular_menu` (line 1796) `def molecular_menu(self)`
- `atomic_menu` (line 1820) `def atomic_menu(self)`
- `run` (line 1840) `def run(self)`

#### `quantum_visualizer.py`
**Path:** `quantum_visualizer.py`
**File Doc:** *quantum_visualizer.py - Production Quantum State Visualizer ============================================================ Brutal real-time visualization of quantum state evolution using trained neural network backends (Hamiltonian, Schrodinger, Dirac).  Imports and uses existing quantum_computer.py, molecular_sim.py, and advanced_experiments.py infrastructure.  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `VisualizerConfig` (line 109) `class VisualizerConfig`
- `QuantumStateSnapshot` (line 157) `class QuantumStateSnapshot`
- `BackendComparisonResult` (line 170) `class BackendComparisonResult`
- `VisualizationResult` (line 181) `class VisualizationResult`
- `IVisualizationComponent` (line 190) `class IVisualizationComponent(ABC)`
- `ProbabilityBarRenderer` (line 196) `class ProbabilityBarRenderer(IVisualizationComponent)`
- `BlochSphereRenderer` (line 227) `class BlochSphereRenderer(IVisualizationComponent)`
- `PhasePlotRenderer` (line 258) `class PhasePlotRenderer(IVisualizationComponent)`
- `EntropyPlotRenderer` (line 291) `class EntropyPlotRenderer(IVisualizationComponent)`
- `BackendComparisonRenderer` (line 313) `class BackendComparisonRenderer(IVisualizationComponent)`
- `QuantumStateAnalyzer` (line 341) `class QuantumStateAnalyzer`
- `CircuitExecutor` (line 397) `class CircuitExecutor`
- `StandardCircuits` (line 457) `class StandardCircuits`
- `FigureBuilder` (line 528) `class FigureBuilder`
- `QuantumVisualizer` (line 612) `class QuantumVisualizer`

**Functions:**
- `_make_logger` (line 93) `def _make_logger(name)`

**Methods:**
- `main` (line 856) `def main()`
- `render` (line 192) `def render(self, data, axes, config)`
- `render` (line 197) `def render(self, data, axes, config)`
- `_get_colors` (line 218) `def _get_colors(self, probs, config)`
- `render` (line 228) `def render(self, data, axes, config)`
- `render` (line 259) `def render(self, data, axes, config)`
- `render` (line 292) `def render(self, snapshots, axes, config)`
- `render` (line 314) `def render(self, results, axes, config)`
- `__init__` (line 342) `def __init__(self, config)`
- `compute_probabilities` (line 345) `def compute_probabilities(self, state)`
- `compute_phases` (line 349) `def compute_phases(self, state)`
- `compute_entropy` (line 360) `def compute_entropy(self, probs)`
- `compute_bloch_vectors` (line 367) `def compute_bloch_vectors(self, state)`
- `create_snapshot` (line 374) `def create_snapshot(self, state, step, gate_name)`
- `__init__` (line 398) `def __init__(self, qc, config)`
- `execute_sequence` (line 403) `def execute_sequence(self, gates, n_qubits, backend_name)`
- `compare_backends` (line 423) `def compare_backends(self, gates, n_qubits, reference_backend)`
- `bell_state` (line 459) `def bell_state()`
- `ghz_state` (line 466) `def ghz_state(n_qubits)`
- `qft` (line 473) `def qft(n_qubits)`
- `grover_oracle` (line 488) `def grover_oracle(n_qubits, marked)`
- `grover_diffusion` (line 501) `def grover_diffusion(n_qubits)`
- `custom_sequence` (line 514) `def custom_sequence(sequence)`
- `__init__` (line 529) `def __init__(self, config)`
- `build_evolution_figure` (line 537) `def build_evolution_figure(self, snapshots, backend_results)`
- `build_summary_figure` (line 563) `def build_summary_figure(self, snapshots, backend_results)`
- `_render_backend_fidelity` (line 594) `def _render_backend_fidelity(self, results, axes)`
- `__init__` (line 613) `def __init__(self, config)`
- `_initialize` (line 620) `def _initialize(self)`
- `_init_quantum_computer` (line 629) `def _init_quantum_computer(self)`
- `visualize_bell_state` (line 654) `def visualize_bell_state(self)`
- `visualize_ghz_state` (line 683) `def visualize_ghz_state(self, n_qubits)`
- `visualize_qft` (line 711) `def visualize_qft(self, n_qubits)`
- `visualize_grover` (line 739) `def visualize_grover(self, n_qubits, marked_state)`
- `visualize_custom_circuit` (line 778) `def visualize_custom_circuit(self, gates, n_qubits, name)`
- `run_all_visualizations` (line 810) `def run_all_visualizations(self)`
- `_save_figure` (line 825) `def _save_figure(self, fig, name)`
- `_print_summary` (line 842) `def _print_summary(self, results)`

#### `relativistic_hydrogen.py`
**Path:** `relativistic_hydrogen.py`
**File Doc:** *Dirac Relativistic Hydrogen Visualizer ====================================== Validation suite for Dirac equation grokking via Hamiltonian Topological Crystallization. Extends the Schrodinger visualization architecture to handle relativistic quantum mechanics.  Features: - Relativistic Hydrogen Atom energy levels (Fine Structure) - Zitterbewegung (Trembling Motion) reconstruction - Spin-orbit coupling visualization - 4-component spinor evolution*

**Classes:**
- `Config` (line 41) `class Config`
- `LoggerFactory` (line 95) `class LoggerFactory`
- `GammaMatrices` (line 113) `class GammaMatrices` - *Dirac gamma matrices in Dirac (standard) representation.
gamma^0 = beta, gamma^i = beta * alpha_i*
- `DiracHamiltonianOperator` (line 199) `class DiracHamiltonianOperator` - *Dirac Hamiltonian operator for relativistic quantum mechanics.
H_Dirac = c * alpha . p + beta * m * c^2 + V(r)

In atomic units (c = 1/alpha ~ 137):
H = c * alpha . p + beta * m * c^2 + V*
- `SpectralLayer` (line 310) `class SpectralLayer(Module)`
- `DiracSpectralNetwork` (line 351) `class DiracSpectralNetwork(Module)` - *Neural network for learning Dirac equation dynamics.
Handles 4-component spinors with real and imaginary parts (8 channels total).*
- `DiracModelWrapper` (line 393) `class DiracModelWrapper` - *Wrapper to load and use the trained Dirac model.*
- `DiracHydrogenAtom` (line 516) `class DiracHydrogenAtom` - *Relativistic hydrogen atom with Dirac equation.
Computes energy levels including fine structure.*
- `ZitterbewegungSimulator` (line 640) `class ZitterbewegungSimulator` - *Simulates the Zitterbewegung (trembling motion) of a relativistic electron.

In Dirac theory, the position operator has a term oscillating with frequency
~ 2mc^2/hbar, which is the interference between positive and negative energy states.

<x(t)> = <x(0)> + (p/m) * t + oscillating term
The oscillating term has amplitude ~ hbar/(2mc) ~ 10^-12 m*
- `DiracWavefunctionCalculator` (line 833) `class DiracWavefunctionCalculator` - *Calculate relativistic hydrogen wavefunctions.*
- `DiracMonteCarloSampler` (line 956) `class DiracMonteCarloSampler` - *Monte Carlo sampling for relativistic orbital visualization.*
- `DiracVisualizer` (line 1075) `class DiracVisualizer` - *Visualization suite for Dirac equation results.*
- `DiracValidationSuite` (line 1370) `class DiracValidationSuite` - *Complete validation suite for Dirac equation grokking.*

**Methods:**
- `main` (line 1613) `def main()`
- `create_logger` (line 97) `def create_logger(name, level)`
- `__init__` (line 118) `def __init__(self, device)`
- `_init_matrices` (line 122) `def _init_matrices(self)`
- `__init__` (line 207) `def __init__(self, config)`
- `_precompute_operators` (line 215) `def _precompute_operators(self)`
- `apply_dirac_hamiltonian` (line 223) `def apply_dirac_hamiltonian(self, spinor, potential)` - *Apply Dirac Hamiltonian to 4-component spinor.

Args:
    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor
    potential: Optional scalar potential V(r)

Returns:
    H * psi with same shape as input*
- `time_evolution` (line 282) `def time_evolution(self, spinor, dt, potential)` - *Time evolution of Dirac spinor using first-order split-step.
psi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi*
- `__init__` (line 311) `def __init__(self, channels, grid_size)`
- `forward` (line 322) `def forward(self, x)`
- `__init__` (line 356) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, spinor_components)`
- `forward` (line 379) `def forward(self, x)`
- `__init__` (line 397) `def __init__(self, config)`
- `_find_best_checkpoint` (line 406) `def _find_best_checkpoint(self)`
- `_load_model` (line 452) `def _load_model(self)`
- `apply_hamiltonian` (line 498) `def apply_hamiltonian(self, spinor, potential)` - *Apply Hamiltonian using analytical operator.
The NN model learns spinor evolution, but the Hamiltonian operator
is applied analytically for physical validation.*
- `evolve_spinor` (line 506) `def evolve_spinor(self, spinor, dt, potential)` - *Evolve spinor in time using the analytical Dirac operator.*
- `__init__` (line 521) `def __init__(self, config)`
- `energy_level_dirac` (line 526) `def energy_level_dirac(self, n, kappa)` - *Exact Dirac energy level for hydrogen-like atom.

E = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)

For hydrogen (Z=1):
E = mc^2 * [1 + (alpha^2 / (n - |kappa| + sqrt(kappa^2 - alpha^2)))^2]^(-1/2)

Args:
    n: Principal quantum number
    kappa: Relativistic quantum number (kappa = -(l+1) for j=l+1/2, kappa = l for j=l-1/2)

Returns:
    Energy in atomic units (relative to m*c^2)*
- `fine_structure_splitting` (line 556) `def fine_structure_splitting(self, n, l)` - *Calculate fine structure splitting for given n, l.

Fine structure includes:
1. Relativistic correction to kinetic energy
2. Spin-orbit coupling
3. Darwin term (for l=0)

Returns energies for j = l+1/2 and j = l-1/2*
- `energy_spectrum` (line 597) `def energy_spectrum(self, n_max)` - *Generate relativistic energy spectrum up to n_max.*
- `__init__` (line 650) `def __init__(self, config, model_wrapper)`
- `create_gaussian_wave_packet` (line 657) `def create_gaussian_wave_packet(self, sigma, momentum)` - *Create a Gaussian wave packet for a free particle.

For Dirac, we need a 4-component spinor that's a superposition
of positive energy states.*
- `compute_position_expectation` (line 702) `def compute_position_expectation(self, spinor)` - *Compute expectation value of position operator.
<x> = <psi| x |psi>*
- `compute_velocity_expectation` (line 724) `def compute_velocity_expectation(self, spinor)` - *Compute expectation value of velocity operator.
In Dirac theory, v = c * alpha

<v_x> = c * <psi| alpha_x |psi>*
- `simulate` (line 750) `def simulate(self, duration, dt, sigma)` - *Run Zitterbewegung simulation.

Returns time evolution of position and velocity showing the
oscillatory ZBW term.*
- `__init__` (line 837) `def __init__(self, config)`
- `radial_wavefunction_schrodinger` (line 843) `def radial_wavefunction_schrodinger(n, l, r)` - *Non-relativistic radial wavefunction for comparison.*
- `radial_wavefunction_dirac` (line 853) `def radial_wavefunction_dirac(self, n, kappa, r, Z)` - *Relativistic radial wavefunctions for hydrogen.

Returns (f, g) - small and large components.
For bound states, the Dirac radial functions are:
f(r) = sqrt((E+mc^2)/(2E)) * G(r)
g(r) = sqrt((E-mc^2)/(2E)) * F(r)

Simplified version using Sommerfeld fine-structure formula.*
- `spherical_harmonic_real` (line 900) `def spherical_harmonic_real(self, l, m, theta, phi)` - *Real spherical harmonics.*
- `spin_angular_function` (line 910) `def spin_angular_function(self, kappa, m_j, theta, phi)` - *Spin-angular functions Omega_{kappa,m_j}(theta, phi).

These couple the orbital and spin degrees of freedom.*
- `__init__` (line 960) `def __init__(self, config, model_wrapper)`
- `sample_orbital` (line 966) `def sample_orbital(self, n, l, j, num_samples)` - *Sample points from a relativistic hydrogen orbital.*
- `__init__` (line 1079) `def __init__(self, config)`
- `visualize_orbital` (line 1082) `def visualize_orbital(self, data, save_path)` - *Visualize relativistic orbital.*
- `visualize_energy_spectrum` (line 1213) `def visualize_energy_spectrum(self, spectrum, save_path)` - *Visualize relativistic energy spectrum with fine structure.*
- `visualize_zitterbewegung` (line 1296) `def visualize_zitterbewegung(self, zbw_data, save_path)` - *Visualize Zitterbewegung oscillation.*
- `__init__` (line 1374) `def __init__(self, config)`
- `print_header` (line 1398) `def print_header(self)`
- `validate_fine_structure` (line 1419) `def validate_fine_structure(self)` - *Validate fine structure energy corrections.*
- `validate_zitterbewegung` (line 1477) `def validate_zitterbewegung(self)` - *Validate Zitterbewegung simulation.*
- `validate_energy_spectrum` (line 1509) `def validate_energy_spectrum(self)` - *Validate complete energy spectrum.*
- `validate_orbital` (line 1524) `def validate_orbital(self, orbital_name, num_samples)` - *Validate single orbital visualization.*
- `run_full_validation` (line 1541) `def run_full_validation(self)` - *Run complete validation suite.*
- `interactive_mode` (line 1575) `def interactive_mode(self)` - *Run in interactive mode.*

#### `test_qc_integration.py`
**Path:** `test_qc_integration.py`
**File Doc:** *BDD-style integration tests for qc_integration.py and qc_dashboard.py.  Uses pytest with descriptive test names following the pattern: test_<scenario>_<expected_behavior>  Coverage: - qc_integration: IntegrationConfig, GateInstruction, CircuitIR, OpenQasmAdapter, StandardCircuitFactory, IntegrationBridge - qc_dashboard: DashboardConfig, GateItem, VisualisationEngine, SimulatorBackend, DashboardApp  Run: pytest test_qc_integration.py -v pytest test_qc_integration.py -v -k "qasm or bridge"*

**Classes:**
- `TestIntegrationConfig` (line 70) `class TestIntegrationConfig` - *IntegrationConfig: centralised configuration with no hardcoded values.*
- `TestGateInstruction` (line 100) `class TestGateInstruction` - *GateInstruction: lightweight quantum gate descriptor.*
- `TestCircuitIR` (line 128) `class TestCircuitIR` - *CircuitIR: intermediate representation of a quantum circuit.*
- `TestOpenQasmAdapter` (line 165) `class TestOpenQasmAdapter` - *OpenQasmAdapter: OpenQASM 2.0 string <-> CircuitIR conversion.*
- `TestStandardCircuitFactory` (line 260) `class TestStandardCircuitFactory` - *StandardCircuitFactory: builds CircuitIR for common algorithms.*
- `TestIntegrationBridge` (line 292) `class TestIntegrationBridge` - *IntegrationBridge: facade for all format conversions.*
- `TestGateItem` (line 346) `class TestGateItem` - *GateItem: circuit builder gate representation.*
- `TestDashboardConfig` (line 372) `class TestDashboardConfig` - *DashboardConfig: centralised configuration for dashboard.*
- `TestVisualisationEngine` (line 400) `class TestVisualisationEngine` - *VisualisationEngine: renders figures from quantum state snapshots.*
- `TestSimulatorBackend` (line 441) `class TestSimulatorBackend` - *SimulatorBackend: lightweight wrapper around QC framework.*
- `TestSnapshotData` (line 473) `class TestSnapshotData` - *SnapshotData: quantum state snapshot for dashboard visualisation.*
- `TestSadPaths` (line 517) `class TestSadPaths` - *Edge cases and error conditions across all modules.*
- `TestEndToEnd` (line 585) `class TestEndToEnd` - *End-to-end scenarios combining multiple modules.*

**Functions:**
- `config` (line 42) `def config()`
- `bridge` (line 47) `def bridge(config)`
- `qasm_adapter` (line 52) `def qasm_adapter(config)`
- `bell_circuit` (line 57) `def bell_circuit()`
- `ghz_circuit` (line 62) `def ghz_circuit()`

**Methods:**
- `test_default_config_has_supported_gates` (line 73) `def test_default_config_has_supported_gates(self, config)`
- `test_gate_name_map_is_complete` (line 79) `def test_gate_name_map_is_complete(self, config)`
- `test_reverse_gate_name_map_is_consistent` (line 83) `def test_reverse_gate_name_map_is_consistent(self, config)`
- `test_qasm_version_default` (line 87) `def test_qasm_version_default(self, config)`
- `test_max_qubits_defaults_are_positive` (line 90) `def test_max_qubits_defaults_are_positive(self, config)`
- `test_create_single_qubit_gate` (line 103) `def test_create_single_qubit_gate(self)`
- `test_create_two_qubit_gate` (line 109) `def test_create_two_qubit_gate(self)`
- `test_create_gate_with_params` (line 114) `def test_create_gate_with_params(self)`
- `test_targets_are_immutable` (line 118) `def test_targets_are_immutable(self)`
- `test_create_empty_circuit` (line 131) `def test_create_empty_circuit(self)`
- `test_append_gate` (line 136) `def test_append_gate(self)`
- `test_append_gate_out_of_range_raises` (line 141) `def test_append_gate_out_of_range_raises(self)`
- `test_multiple_gates` (line 146) `def test_multiple_gates(self)`
- `test_repr_includes_qubits_and_gates` (line 152) `def test_repr_includes_qubits_and_gates(self)`
- `test_export_bell_state_contains_header` (line 168) `def test_export_bell_state_contains_header(self, qasm_adapter, bell_circuit)`
- `test_export_bell_state_has_qreg_and_creg` (line 173) `def test_export_bell_state_has_qreg_and_creg(self, qasm_adapter, bell_circuit)`
- `test_export_bell_state_has_gates` (line 178) `def test_export_bell_state_has_gates(self, qasm_adapter, bell_circuit)`
- `test_export_ghz_state` (line 183) `def test_export_ghz_state(self, qasm_adapter, ghz_circuit)`
- `test_export_qft_has_swap` (line 189) `def test_export_qft_has_swap(self, qasm_adapter, config)`
- `test_export_parametric_gate` (line 194) `def test_export_parametric_gate(self, qasm_adapter)`
- `test_export_exceeds_max_qubits_raises` (line 201) `def test_export_exceeds_max_qubits_raises(self, qasm_adapter)`
- `test_import_bell_state_roundtrip` (line 206) `def test_import_bell_state_roundtrip(self, qasm_adapter, bell_circuit)`
- `test_import_ghz_roundtrip` (line 214) `def test_import_ghz_roundtrip(self, qasm_adapter, ghz_circuit)`
- `test_import_from_standard_qasm_string` (line 220) `def test_import_from_standard_qasm_string(self, qasm_adapter)`
- `test_import_with_parametric_gates` (line 234) `def test_import_with_parametric_gates(self, qasm_adapter)`
- `test_import_empty_qasm_returns_zero_qubit_circuit` (line 247) `def test_import_empty_qasm_returns_zero_qubit_circuit(self, qasm_adapter)`
- `test_bell_state_has_two_gates` (line 263) `def test_bell_state_has_two_gates(self)`
- `test_bell_state_has_two_qubits` (line 269) `def test_bell_state_has_two_qubits(self)`
- `test_ghz_state` (line 273) `def test_ghz_state(self)`
- `test_qft_three_qubits` (line 278) `def test_qft_three_qubits(self)`
- `test_grover_iterations` (line 283) `def test_grover_iterations(self)`
- `test_export_qasm_returns_string` (line 295) `def test_export_qasm_returns_string(self, bridge, bell_circuit)`
- `test_import_qasm_roundtrip` (line 300) `def test_import_qasm_roundtrip(self, bridge, bell_circuit)`
- `test_full_openqasm_roundtrip_bell` (line 306) `def test_full_openqasm_roundtrip_bell(self, bridge)`
- `test_full_openqasm_roundtrip_ghz` (line 313) `def test_full_openqasm_roundtrip_ghz(self, bridge)`
- `test_full_openqasm_roundtrip_qft` (line 320) `def test_full_openqasm_roundtrip_qft(self, bridge)`
- `test_qiskit_not_available_by_default` (line 327) `def test_qiskit_not_available_by_default(self, bridge)`
- `test_pennylane_not_available_by_default` (line 332) `def test_pennylane_not_available_by_default(self, bridge)`
- `test_export_qasm_custom_qreg_name` (line 337) `def test_export_qasm_custom_qreg_name(self, bridge, bell_circuit)`
- `test_create_single_qubit_gate` (line 349) `def test_create_single_qubit_gate(self)`
- `test_create_two_qubit_gate` (line 357) `def test_create_two_qubit_gate(self)`
- `test_create_parametric_gate` (line 362) `def test_create_parametric_gate(self)`
- `test_default_values` (line 375) `def test_default_values(self)`
- `test_gate_list_includes_standard_gates` (line 382) `def test_gate_list_includes_standard_gates(self)`
- `test_qasm_initial_contains_header` (line 389) `def test_qasm_initial_contains_header(self)`
- `test_engine_available_with_matplotlib` (line 403) `def test_engine_available_with_matplotlib(self)`
- `test_render_full_dashboard_returns_bytes` (line 415) `def test_render_full_dashboard_returns_bytes(self)`
- `test_synthetic_execute_returns_snapshots` (line 444) `def test_synthetic_execute_returns_snapshots(self)`
- `test_empty_circuit_returns_init_snapshot` (line 457) `def test_empty_circuit_returns_init_snapshot(self)`
- `test_create_snapshot` (line 476) `def test_create_snapshot(self)`
- `test_entropy_updates` (line 489) `def test_entropy_updates(self)`
- `test_probabilities_normalized` (line 500) `def test_probabilities_normalized(self)`
- `test_gate_instruction_empty_targets` (line 520) `def test_gate_instruction_empty_targets(self)`
- `test_circuit_ir_append_negative_qubit_raises` (line 524) `def test_circuit_ir_append_negative_qubit_raises(self)`
- `test_openqasm_import_empty_string` (line 529) `def test_openqasm_import_empty_string(self, qasm_adapter)`
- `test_openqasm_import_garbage_string` (line 534) `def test_openqasm_import_garbage_string(self, qasm_adapter)`
- `test_openqasm_export_zero_qubit_circuit` (line 538) `def test_openqasm_export_zero_qubit_circuit(self, qasm_adapter)`
- `test_standard_circuit_factory_qft_one_qubit` (line 543) `def test_standard_circuit_factory_qft_one_qubit(self)`
- `test_standard_circuit_factory_grover_minimal` (line 548) `def test_standard_circuit_factory_grover_minimal(self)`
- `test_circuit_ir_repr_no_gates` (line 552) `def test_circuit_ir_repr_no_gates(self)`
- `test_synthetic_snapshot_probabilities_sum_to_one` (line 557) `def test_synthetic_snapshot_probabilities_sum_to_one(self)`
- `test_visualisation_engine_handles_no_snapshots` (line 570) `def test_visualisation_engine_handles_no_snapshots(self)`
- `test_build_export_import_qasm_roundtrip` (line 588) `def test_build_export_import_qasm_roundtrip(self, bridge)`
- `test_qasm_to_circuitir_to_framework_mps` (line 598) `def test_qasm_to_circuitir_to_framework_mps(self, bridge)`
- `test_ghz_export_qasm_and_reimport_matches` (line 609) `def test_ghz_export_qasm_and_reimport_matches(self, bridge)`
- `test_qft_circuit_qasm_roundtrip` (line 616) `def test_qft_circuit_qasm_roundtrip(self, bridge)`
- `test_export_qasm_with_custom_names` (line 622) `def test_export_qasm_with_custom_names(self, bridge)`
- `test_import_qasm_preserves_gate_order` (line 628) `def test_import_qasm_preserves_gate_order(self, bridge)`
- `test_full_pipeline_build_export_import_to_mps` (line 644) `def test_full_pipeline_build_export_import_to_mps(self, bridge)`

#### `test_quantum_framework.py`
**Path:** `test_quantum_framework.py`
**File Doc:** *Quantum Framework Test Suite ============================ Comprehensive pytest tests for the MPS quantum simulation framework.  Run with: pytest test_quantum_framework.py -v pytest test_quantum_framework.py -v -k "bell"  # Run only bell state tests pytest test_quantum_framework.py -v --tb=short  # Short traceback  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `TestFrameworkConfig` (line 72) `class TestFrameworkConfig` - *Tests for FrameworkConfig.*
- `TestMPSState` (line 95) `class TestMPSState` - *Tests for MPSState.*
- `TestBellState` (line 130) `class TestBellState` - *Tests for Bell state preparation.*
- `TestGHZState` (line 167) `class TestGHZState` - *Tests for GHZ state preparation.*
- `TestWState` (line 209) `class TestWState` - *Tests for W state preparation.*
- `TestQuantumGates` (line 261) `class TestQuantumGates` - *Tests for quantum gates.*
- `TestPhaseCoherence` (line 338) `class TestPhaseCoherence` - *Tests for phase coherence and unitarity.*
- `TestGroverAlgorithm` (line 403) `class TestGroverAlgorithm` - *Tests for Grover search algorithm.*
- `TestQFT` (line 431) `class TestQFT` - *Tests for Quantum Fourier Transform.*
- `TestMemoryScaling` (line 458) `class TestMemoryScaling` - *Tests for memory scaling.*
- `TestPrecisionMode` (line 501) `class TestPrecisionMode` - *Tests for precision mode.*
- `TestEdgeCases` (line 525) `class TestEdgeCases` - *Tests for edge cases.*

**Functions:**
- `config` (line 43) `def config()` - *Create default framework configuration.*
- `qc` (line 53) `def qc(config)` - *Create quantum computer instance.*
- `config_precision` (line 59) `def config_precision()` - *Create precision mode configuration.*

**Methods:**
- `test_default_config` (line 75) `def test_default_config(self)` - *Test default configuration values.*
- `test_custom_config` (line 83) `def test_custom_config(self)` - *Test custom configuration values.*
- `test_create_state` (line 98) `def test_create_state(self, config)` - *Test state creation.*
- `test_initial_state` (line 104) `def test_initial_state(self, qc)` - *Test initial |00...0> state.*
- `test_clone_state` (line 113) `def test_clone_state(self, qc)` - *Test state cloning.*
- `test_bell_entropy` (line 133) `def test_bell_entropy(self, qc)` - *Test Bell state entropy.*
- `test_bell_probabilities` (line 141) `def test_bell_probabilities(self, qc)` - *Test Bell state probabilities.*
- `test_bell_entanglement` (line 154) `def test_bell_entanglement(self, qc)` - *Test Bell state entanglement.*
- `test_ghz_entropy` (line 170) `def test_ghz_entropy(self, qc)` - *Test GHZ state entropy.*
- `test_ghz_probabilities` (line 179) `def test_ghz_probabilities(self, qc)` - *Test GHZ state probabilities.*
- `test_ghz_scaling` (line 192) `def test_ghz_scaling(self, qc)` - *Test GHZ state memory scaling.*
- `test_w_state_probabilities` (line 212) `def test_w_state_probabilities(self, qc)` - *Test W state probabilities.*
- `test_w_state_entropy` (line 227) `def test_w_state_entropy(self, qc)` - *Test W state entropy.*
- `test_w_state_no_zero_probabilities` (line 247) `def test_w_state_no_zero_probabilities(self, qc)` - *Test that W state has non-zero probabilities for single-excitation states.*
- `test_hadamard_gate` (line 264) `def test_hadamard_gate(self, qc)` - *Test Hadamard gate.*
- `test_pauli_x_gate` (line 275) `def test_pauli_x_gate(self, qc)` - *Test Pauli-X gate.*
- `test_pauli_z_gate` (line 285) `def test_pauli_z_gate(self, qc)` - *Test Pauli-Z gate on |+> state.*
- `test_cnot_gate` (line 297) `def test_cnot_gate(self, qc)` - *Test CNOT gate.*
- `test_swap_gate` (line 309) `def test_swap_gate(self, qc)` - *Test SWAP gate.*
- `test_rotation_gates` (line 321) `def test_rotation_gates(self, qc)` - *Test rotation gates.*
- `test_hzh_equals_x` (line 341) `def test_hzh_equals_x(self, qc)` - *Test HZH = X identity.*
- `test_xx_equals_identity` (line 354) `def test_xx_equals_identity(self, qc)` - *Test XX = I identity.*
- `test_cnot_cnot_equals_identity` (line 366) `def test_cnot_cnot_equals_identity(self, qc)` - *Test CNOT CNOT = I identity.*
- `test_norm_preservation` (line 381) `def test_norm_preservation(self, qc)` - *Test that norm is preserved after gates.*
- `test_grover_3_qubits` (line 406) `def test_grover_3_qubits(self, qc)` - *Test Grover search on 3 qubits.*
- `test_grover_speedup` (line 417) `def test_grover_speedup(self, qc)` - *Test Grover speedup.*
- `test_qft_entropy` (line 434) `def test_qft_entropy(self, qc)` - *Test QFT entropy.*
- `test_memory_linear_scaling` (line 461) `def test_memory_linear_scaling(self)` - *Test that memory scales sub-exponentially with qubits (MPS property).*
- `test_compression_ratio` (line 478) `def test_compression_ratio(self)` - *Test MPS compression ratio vs full statevector for large n.*
- `test_precision_mode_config` (line 504) `def test_precision_mode_config(self)` - *Test precision mode configuration.*
- `test_precision_mode_bell_state` (line 510) `def test_precision_mode_bell_state(self, config_precision)` - *Test Bell state in precision mode.*
- `test_single_qubit` (line 528) `def test_single_qubit(self, qc)` - *Test single qubit operations.*
- `test_large_bond_dimension` (line 539) `def test_large_bond_dimension(self)` - *Test with large bond dimension.*
- `test_empty_circuit` (line 549) `def test_empty_circuit(self, qc)` - *Test empty circuit.*

#### `topological_hilbert_compression2.py`
**Path:** `topological_hilbert_compression2.py`
**File Doc:** *Topological Hilbert Space Compression ====================================== Implements Matrix Product State (MPS) factorization with vacuum-core architecture for scaling quantum simulations beyond 20 qubits.  Architecture: - MPS/TTN factorization reduces O(2^n) to O(n * chi^2) - Vacuum Core projects irrelevant Hilbert subspace to zero - Hybrid backend routes subcircuits to optimal processors - Topological protection via winding numbers and Berry phases  Author: Gris Iscomeback License: AGPL v3*

**Classes:**
- `HilbertPhase` (line 47) `class HilbertPhase(Enum)`
- `TopologicalCompressionConfig` (line 56) `class TopologicalCompressionConfig`
- `ITensorNetwork` (line 85) `class ITensorNetwork(ABC)`
- `MPSCore` (line 115) `class MPSCore` - *Matrix Product State core tensor A^{[k]}_{i_k} with left and right bond indices.
Shape: (chi_left, d, chi_right) where d=2 for qubits.*
- `MPSState` (line 162) `class MPSState(ITensorNetwork)` - *Matrix Product State representation of n-qubit quantum state.

|psi> = sum_{i_1...i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} ... A^{[n]}_{i_n} |i_1...i_n>

Memory: O(n * chi^2 * d) vs O(d^n) for full statevector.
For n=30, chi=16: ~30KB vs 8GB for statevector.*
- `VacuumCore` (line 351) `class VacuumCore` - *Vacuum Core architecture that projects irrelevant Hilbert subspace to zero.

Inspired by the HPU-Core achieving 99.996% sparsity:
- Active core: small subspace carrying quantum information
- Vacuum: 99%+ of Hilbert space forced to zero by regularization
- Protection: topological invariants prevent core destruction*
- `TopologicalProtector` (line 411) `class TopologicalProtector` - *Provides topological protection for quantum states via:
- Winding number monitoring
- Berry phase calculation
- Edge state preservation*
- `HybridBackend` (line 452) `class HybridBackend(ABC)`
- `DirectBackend` (line 466) `class DirectBackend(HybridBackend)` - *Direct tensor backend using JointHilbertState representation.
Limited to config.max_qubits_direct qubits due to exponential memory.*
- `MPSBackend` (line 515) `class MPSBackend(HybridBackend)` - *Matrix Product State backend for scalable quantum simulation.
Handles up to config.max_qubits_mps qubits with sub-exponential memory.*
- `TopologicalHilbertSimulator` (line 572) `class TopologicalHilbertSimulator` - *Main simulator implementing hybrid architecture for scalable quantum simulation.

Architecture:
    - n <= max_qubits_direct: HPU-Core direct tensor (exact wavefunction evolution)
    - n > max_qubits_direct: MPS with vacuum core compression*
- `Schrodinger20Experiment` (line 723) `class Schrodinger20Experiment` - *Validates topological compression on 20-qubit molecular ground state.

Target: H2O (10 electrons, ~20 qubits)
Metrics:
    - Energy error < 1 mHa vs Qiskit VQE
    - Inference time < 100ms
    - Memory < 50MB (vs 500MB for direct tensor)*

**Methods:**
- `main` (line 956) `def main()`
- `__post_init__` (line 80) `def __post_init__(self)`
- `amplitude` (line 87) `def amplitude(self, basis_index)`
- `apply_single_qubit_gate` (line 91) `def apply_single_qubit_gate(self, qubit, gate)`
- `apply_two_qubit_gate` (line 95) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)`
- `norm` (line 99) `def norm(self)`
- `probabilities` (line 103) `def probabilities(self)`
- `entropy` (line 107) `def entropy(self)`
- `memory_bytes` (line 111) `def memory_bytes(self)`
- `__init__` (line 121) `def __init__(self, chi_left, chi_right, d, device, dtype)`
- `_initialize` (line 130) `def _initialize(self)`
- `tensor` (line 136) `def tensor(self)`
- `tensor` (line 142) `def tensor(self, value)`
- `left_canonicalize` (line 147) `def left_canonicalize(self)`
- `right_canonicalize` (line 154) `def right_canonicalize(self)`
- `__init__` (line 172) `def __init__(self, n_qubits, config)`
- `_initialize` (line 181) `def _initialize(self)`
- `_bond_dimension` (line 195) `def _bond_dimension(self, site)`
- `amplitude` (line 198) `def amplitude(self, basis_index)`
- `apply_single_qubit_gate` (line 209) `def apply_single_qubit_gate(self, qubit, gate)`
- `apply_two_qubit_gate` (line 219) `def apply_two_qubit_gate(self, qubit_a, qubit_b, gate)`
- `_swap_qubits_in_gate` (line 236) `def _swap_qubits_in_gate(self, gate)`
- `_apply_adjacent_gate` (line 245) `def _apply_adjacent_gate(self, qubit, gate)`
- `_apply_nonadjacent_gate` (line 285) `def _apply_nonadjacent_gate(self, qubit_a, qubit_b, gate)`
- `norm` (line 294) `def norm(self)`
- `_canonicalize` (line 301) `def _canonicalize(self)`
- `probabilities` (line 308) `def probabilities(self)`
- `entropy` (line 318) `def entropy(self)`
- `memory_bytes` (line 329) `def memory_bytes(self)`
- `entanglement_entropy` (line 335) `def entanglement_entropy(self, cut)`
- `__init__` (line 361) `def __init__(self, n_qubits, config)`
- `_initialize` (line 370) `def _initialize(self)`
- `_compute_berry_phases` (line 375) `def _compute_berry_phases(self)`
- `add_active_state` (line 381) `def add_active_state(self, basis_index, winding_number)`
- `_compute_winding_number` (line 389) `def _compute_winding_number(self, basis_index)`
- `is_topologically_protected` (line 394) `def is_topologically_protected(self, basis_index)`
- `sparsity` (line 398) `def sparsity(self)`
- `project_to_active` (line 403) `def project_to_active(self, state)`
- `__init__` (line 419) `def __init__(self, config)`
- `compute_winding_number` (line 424) `def compute_winding_number(self, state, qubit)`
- `compute_berry_phase` (line 433) `def compute_berry_phase(self, state, qubit_a, qubit_b)`
- `is_protected` (line 444) `def is_protected(self, state, vacuum_core)`
- `can_handle` (line 454) `def can_handle(self, n_qubits)`
- `create_state` (line 458) `def create_state(self, n_qubits)`
- `apply_gate` (line 462) `def apply_gate(self, state, gate_name, targets, params)`
- `__init__` (line 472) `def __init__(self, config)`
- `_load_quantum_computer` (line 479) `def _load_quantum_computer(self)`
- `can_handle` (line 497) `def can_handle(self, n_qubits)`
- `create_state` (line 500) `def create_state(self, n_qubits)`
- `apply_gate` (line 507) `def apply_gate(self, state, gate_name, targets, params)`
- `__init__` (line 521) `def __init__(self, config)`
- `_initialize_gate_cache` (line 526) `def _initialize_gate_cache(self)`
- `can_handle` (line 537) `def can_handle(self, n_qubits)`
- `create_state` (line 540) `def create_state(self, n_qubits)`
- `apply_gate` (line 543) `def apply_gate(self, state, gate_name, targets, params)`
- `__init__` (line 581) `def __init__(self, config)`
- `_select_backend` (line 590) `def _select_backend(self, n_qubits, force_mps)`
- `create_circuit` (line 602) `def create_circuit(self, n_qubits, force_mps)`
- `h` (line 610) `def h(self, qubit)`
- `x` (line 613) `def x(self, qubit)`
- `y` (line 616) `def y(self, qubit)`
- `z` (line 619) `def z(self, qubit)`
- `rx` (line 622) `def rx(self, qubit, theta)`
- `ry` (line 625) `def ry(self, qubit, theta)`
- `rz` (line 628) `def rz(self, qubit, theta)`
- `cnot` (line 631) `def cnot(self, control, target)`
- `cz` (line 634) `def cz(self, control, target)`
- `swap` (line 637) `def swap(self, qubit_a, qubit_b)`
- `run` (line 640) `def run(self)`
- `probabilities` (line 650) `def probabilities(self)`
- `entropy` (line 655) `def entropy(self)`
- `memory_usage` (line 666) `def memory_usage(self)`
- `compression_ratio` (line 682) `def compression_ratio(self)`
- `detect_phase` (line 691) `def detect_phase(self)`
- `_compute_average_bond_dimension` (line 714) `def _compute_average_bond_dimension(self)`
- `__init__` (line 734) `def __init__(self, config)`
- `run_bell_state` (line 739) `def run_bell_state(self, n_qubits, use_mps)`
- `run_ghz_state` (line 758) `def run_ghz_state(self, n_qubits, use_mps)`
- `run_w_state` (line 777) `def run_w_state(self, n_qubits, use_mps)`
- `prepare_ghz_state` (line 798) `def prepare_ghz_state(self, n_qubits, force_mps)` - *Prepare GHZ state directly without SWAP overhead.

GHZ: |00...0> + |11...1> normalized.
MPS representation:
- A^{[0]}_{i_0,α_1}: A[0,0,0]=1/√2, A[0,1,1]=1/√2, shape (1,2,2)
- A^{[k]}_{α_k,i_k,α_{k+1}}: A[0,0,0]=A[1,1,1]=1, shape (2,2,2)
- A^{[n-1]}_{α_{n-1},i_{n-1},0}: A[0,0,0]=A[1,1,0]=1, shape (2,2,1)*
- `run_scaling_benchmark` (line 844) `def run_scaling_benchmark(self, max_qubits, use_mps)`
- `run_all` (line 897) `def run_all(self)`
- `_print_summary` (line 909) `def _print_summary(self)`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
