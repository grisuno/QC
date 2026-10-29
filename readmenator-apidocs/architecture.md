# QC Architecture and Execution Flows

[Back to index](index.md)

## Architecture Overview

6 components stacked in 3 dependency tiers (weighted Eades-Lin-Smyth ordering: arrows point from dependent to dependency, top to bottom). Labels are import counts, thick arrows the top quartile; 1 transitively implied edges are omitted. Dashed red arrows are upward dependencies: the smallest set of imports that closes every cycle and breaks the layering. Cyan components hold entry points. Interactive version: `readmenator-apidocs/index.html#/flows`.

Upward dependencies to fix first: `root: quantum_simulator` -> `root: quantum_framework_core` (2 imports)

```mermaid
flowchart TB
    classDef cycle fill:#1f1530,stroke:#f472b6,color:#fce7f3,stroke-width:2px;
    classDef default fill:#111827,stroke:#38bdf8,color:#e5e7eb;
    classDef entry fill:#0e7490,stroke:#22d3ee,color:#ecfeff,stroke-width:2px;
    subgraph lane_0["Tier 1 - entry / orchestration"]
        direction LR
        n_c1["immersive<br/>11 files, 246 symbols<br/>Core / utility"]:::entry
        n_d_immersive["immersive<br/>1 files, 19 symbols<br/>Core / utility"]:::default
        n_d__root_["(root)<br/>2 files, 13 symbols<br/>Core / utility"]:::default
        n_c3["root: quantum_framework_molecular_v2<br/>2 files, 73 symbols<br/>Core / utility"]:::default
    end
    subgraph lane_1["Tier 2"]
        direction LR
        n_c2["root: quantum_framework_core<br/>8 files, 447 symbols<br/>Core / utility"]:::cycle
    end
    subgraph lane_2["Tier 3 - foundation"]
        direction LR
        n_c0["root: quantum_simulator<br/>16 files, 1029 symbols<br/>Core / utility"]:::entry
    end
    n_c1 ==>|"3"| n_c2
    n_c2 ==>|"9"| n_c0
    n_c2 <-.-|"2 upward"| n_c0
    linkStyle 2 stroke:#f87171,stroke-width:2px;
    n_c1 ~~~ n_c2
    n_c2 ~~~ n_c0
```

---

## Execution Flows

Entry points are scored from `__main__` guards, `main` functions, entry-style names and files, and call-graph sources. Each flow walks resolved calls in call-site order (depth <= 4, <= 40 steps); numbers give the call order, dashed arrows revisit an already expanded callee, red arrows are recursion. Interactive version with zoom and per-node docs: `readmenator-apidocs/index.html#/flows`.

### `app.main`

`immersive/app.py:382`, score 13: entry-style name `main`; defined in entry file `app.py`; called from the `__main__` guard; called from the `__main__` guard.

Call tree:

```mermaid
flowchart LR
    classDef default fill:#111827,stroke:#38bdf8,color:#e5e7eb;
    classDef entry fill:#0e7490,stroke:#22d3ee,color:#ecfeff,stroke-width:2px;
    classDef external fill:#1a1530,stroke:#a78bfa,color:#ede9fe;
    n_immersive_app_py__main["app.main<br/>app.py:382"]:::entry
    n_immersive_app_py__App["app.App<br/>app.py:92"]:::default
    n_immersive_app_py__parse_args["app.parse_args<br/>app.py:369"]:::default
    n_immersive_app_py__run["app.App.run<br/>app.py:344"]:::default
    n_immersive_app_py__frame["app.App.frame<br/>app.py:329"]:::default
    n_immersive_hud_py__resize["hud.Hud.resize<br/>hud.py:51"]:::external
    n_immersive_glkit_py__make_texture["glkit.make_texture<br/>glkit.py:97"]:::external
    n_immersive_postfx_py__resize["postfx.PostProcess.resize<br/>postfx.py:133"]:::external
    n_immersive_postfx_py__Target["postfx.Target<br/>Colour-only framebuffer with a half-float tex...<br/>postfx.py:87"]:::external
    n_immersive_app_py__update["app.App.update<br/>app.py:264"]:::default
    n_immersive_app_py__update_tour["app.App.update_tour<br/>app.py:197"]:::default
    n_immersive_postfx_py__begin["postfx.PostProcess.begin<br/>postfx.py:142"]:::external
    n_immersive_postfx_py__bind["postfx.Target.bind<br/>postfx.py:107"]:::external
    n_immersive_postfx_py__present["postfx.PostProcess.present<br/>postfx.py:154"]:::external
    n_immersive_glkit_py__use["glkit.Program.use<br/>glkit.py:49"]:::external
    n_immersive_postfx_py__source_size["postfx.source_size<br/>Texel grid of the texture each downsample lev...<br/>postfx.py:193"]:::external
    n_immersive_glkit_py__v2["glkit.Program.v2<br/>glkit.py:63"]:::external
    n_immersive_glkit_py__f1["glkit.Program.f1<br/>glkit.py:57"]:::external
    n_immersive_postfx_py___pass["postfx.PostProcess._pass<br/>postfx.py:146"]:::external
    n_immersive_glkit_py__i1["glkit.Program.i1<br/>glkit.py:60"]:::external
    n_immersive_glkit_py__draw["glkit.FullscreenPass.draw<br/>glkit.py:92"]:::external
    n_immersive_hud_py__begin["hud.Hud.begin<br/>hud.py:65"]:::external
    n_immersive_app_py__draw_chrome["app.App.draw_chrome<br/>app.py:286"]:::default
    n_immersive_hud_py__end["hud.Hud.end<br/>hud.py:156"]:::external
    n_immersive_app_py__screenshot["app.App.screenshot<br/>app.py:214"]:::default
    n_immersive_app_py__handle_event["app.App.handle_event<br/>app.py:224"]:::default
    n_immersive_app_py__stop_tour["app.App.stop_tour<br/>app.py:176"]:::default
    n_immersive_app_py__go["app.App.go<br/>app.py:147"]:::default
    n_immersive_hud_py__toggle["hud.Lang.toggle<br/>hud.py:31"]:::external
    n_immersive_app_py__run_command["app.App.run_command<br/>app.py:157"]:::default
    n_immersive_app_py__start_tour["app.App.start_tour<br/>app.py:169"]:::default
    n_immersive_hud_py__hit["hud.Hud.hit<br/>hud.py:122"]:::external
    n_immersive_app_py__main -->|"1"| n_immersive_app_py__App
    n_immersive_app_py__main -->|"2"| n_immersive_app_py__parse_args
    n_immersive_app_py__main -->|"3"| n_immersive_app_py__run
    n_immersive_app_py__run -->|"1"| n_immersive_app_py__frame
    n_immersive_app_py__frame -->|"1"| n_immersive_hud_py__resize
    n_immersive_hud_py__resize -->|"1"| n_immersive_glkit_py__make_texture
    n_immersive_app_py__frame -->|"2"| n_immersive_postfx_py__resize
    n_immersive_postfx_py__resize -->|"1"| n_immersive_postfx_py__Target
    n_immersive_app_py__frame -->|"3"| n_immersive_app_py__update
    n_immersive_app_py__update -->|"1"| n_immersive_app_py__update_tour
    n_immersive_app_py__frame -->|"4"| n_immersive_postfx_py__begin
    n_immersive_postfx_py__begin -->|"1"| n_immersive_postfx_py__bind
    n_immersive_app_py__frame -->|"5"| n_immersive_postfx_py__present
    n_immersive_postfx_py__present -->|"1"| n_immersive_glkit_py__use
    n_immersive_postfx_py__present -->|"2"| n_immersive_postfx_py__source_size
    n_immersive_postfx_py__present -->|"3"| n_immersive_glkit_py__v2
    n_immersive_postfx_py__present -->|"4"| n_immersive_glkit_py__f1
    n_immersive_postfx_py__present -->|"5"| n_immersive_postfx_py___pass
    n_immersive_postfx_py__present -->|"6"| n_immersive_glkit_py__i1
    n_immersive_postfx_py__present -->|"7"| n_immersive_glkit_py__draw
    n_immersive_app_py__frame -->|"6"| n_immersive_hud_py__begin
    n_immersive_app_py__frame -->|"7"| n_immersive_app_py__draw_chrome
    n_immersive_app_py__frame -->|"8"| n_immersive_hud_py__end
    n_immersive_hud_py__end -->|"1"| n_immersive_glkit_py__use
    n_immersive_hud_py__end -->|"2"| n_immersive_glkit_py__i1
    n_immersive_hud_py__end -->|"3"| n_immersive_glkit_py__v2
    n_immersive_hud_py__end -->|"4"| n_immersive_glkit_py__draw
    n_immersive_app_py__run -->|"2"| n_immersive_app_py__screenshot
    n_immersive_app_py__run -->|"3"| n_immersive_app_py__handle_event
    n_immersive_app_py__handle_event -->|"1"| n_immersive_app_py__stop_tour
    n_immersive_app_py__handle_event -->|"2"| n_immersive_app_py__go
    n_immersive_app_py__handle_event -->|"3"| n_immersive_hud_py__toggle
    n_immersive_app_py__handle_event -.-> n_immersive_app_py__screenshot
    n_immersive_app_py__handle_event -->|"4"| n_immersive_app_py__run_command
    n_immersive_app_py__run_command -.-> n_immersive_app_py__go
    n_immersive_app_py__run_command -->|"1"| n_immersive_app_py__start_tour
    n_immersive_app_py__run_command -.-> n_immersive_app_py__stop_tour
    n_immersive_app_py__run_command -.-> n_immersive_hud_py__toggle
    n_immersive_app_py__handle_event -->|"5"| n_immersive_hud_py__hit
```

Cross-file sequence:

```mermaid
sequenceDiagram
    autonumber
    participant p_immersive_app_py as app.py
    participant p_immersive_hud_py as hud.py
    participant p_immersive_glkit_py as glkit.py
    participant p_immersive_postfx_py as postfx.py
    p_immersive_app_py->>p_immersive_hud_py: Hud.resize()
    p_immersive_hud_py->>p_immersive_glkit_py: make_texture()
    p_immersive_app_py->>p_immersive_postfx_py: PostProcess.resize()
    p_immersive_app_py->>p_immersive_postfx_py: PostProcess.begin()
    p_immersive_app_py->>p_immersive_postfx_py: PostProcess.present()
    p_immersive_postfx_py->>p_immersive_glkit_py: Program.use()
    p_immersive_postfx_py->>p_immersive_glkit_py: Program.v2()
    p_immersive_postfx_py->>p_immersive_glkit_py: Program.f1()
    p_immersive_postfx_py->>p_immersive_glkit_py: Program.i1()
    p_immersive_postfx_py->>p_immersive_glkit_py: FullscreenPass.draw()
    p_immersive_app_py->>p_immersive_hud_py: Hud.begin()
    p_immersive_app_py->>p_immersive_hud_py: Hud.end()
    p_immersive_hud_py->>p_immersive_glkit_py: Program.use()
    p_immersive_hud_py->>p_immersive_glkit_py: Program.i1()
    p_immersive_hud_py->>p_immersive_glkit_py: Program.v2()
    p_immersive_hud_py->>p_immersive_glkit_py: FullscreenPass.draw()
    p_immersive_app_py->>p_immersive_hud_py: Lang.toggle()
    p_immersive_app_py->>p_immersive_hud_py: Hud.hit()
```

### `advanced_experiments (__main__)`

`advanced_experiments.py:1337`, score 8: `if __name__ == "__main__"` guard.

Call tree:

```mermaid
flowchart LR
    classDef default fill:#111827,stroke:#38bdf8,color:#e5e7eb;
    classDef entry fill:#0e7490,stroke:#22d3ee,color:#ecfeff,stroke-width:2px;
    n_advanced_experiments_py___main_["advanced_experiments (__main__)<br/>advanced_experiments.py"]:::entry
    n_advanced_experiments_py___make_logger["advanced_experiments._make_logger<br/>advanced_experiments.py:121"]:::default
    n_advanced_experiments_py__main["advanced_experiments.main<br/>Main entry point.<br/>advanced_experiments.py:1304"]:::default
    n_advanced_experiments_py__AdvancedExperimentRunner["advanced_experiments.AdvancedExperimentRunner<br/>Runs all three advanced experiments.<br/>advanced_experiments.py:1223"]:::default
    n_advanced_experiments_py__run_grover["advanced_experiments.AdvancedExperimentRunner.run_grover<br/>Run Grover's algorithm experiment.<br/>advanced_experiments.py:1235"]:::default
    n_advanced_experiments_py__GroverConfig["advanced_experiments.GroverConfig<br/>Configuration for Grover's algorithm experime...<br/>advanced_experiments.py:141"]:::default
    n_advanced_experiments_py__GroverSearch["advanced_experiments.GroverSearch<br/>Complete Grover's algorithm implementation us...<br/>advanced_experiments.py:240"]:::default
    n_advanced_experiments_py__run["advanced_experiments.GroverSearch.run<br/>Run Grover's search algorithm.<br/>advanced_experiments.py:300"]:::default
    n_advanced_experiments_py___calculate_entropy["advanced_experiments.GroverSearch._calculate_entropy<br/>Calculate Shannon entropy from probability di...<br/>advanced_experiments.py:266"]:::default
    n_advanced_experiments_py__apply["advanced_experiments.GroverOracle.apply<br/>Apply oracle: flip phase of marked state.<br/>advanced_experiments.py:176"]:::default
    n_advanced_experiments_py__apply_206["advanced_experiments.GroverDiffusionOperator.apply<br/>Apply diffusion operator using Hadamard + Ora...<br/>advanced_experiments.py:206"]:::default
    n_advanced_experiments_py__run_qed["advanced_experiments.AdvancedExperimentRunner.run_qed<br/>Run QED effects experiment.<br/>advanced_experiments.py:1252"]:::default
    n_advanced_experiments_py__QEDConfig["advanced_experiments.QEDConfig<br/>Configuration for QED effects calculations.<br/>advanced_experiments.py:404"]:::default
    n_advanced_experiments_py__QEDEffectsExperiment["advanced_experiments.QEDEffectsExperiment<br/>Complete QED effects experiment combining Lam...<br/>advanced_experiments.py:722"]:::default
    n_advanced_experiments_py__run_full_analysis["advanced_experiments.QEDEffectsExperiment.run_full_analysis<br/>Run complete QED analysis.<br/>advanced_experiments.py:736"]:::default
    n_advanced_experiments_py__compare_2s_2p["advanced_experiments.LambShiftCalculator.compare_2s_2p<br/>Compare Lamb shift for 2s_{1/2} and 2p_{1/2}...<br/>advanced_experiments.py:558"]:::default
    n_advanced_experiments_py__full_report["advanced_experiments.AnomalousMagneticMoment.full_report<br/>Generate a full report on g-2 calculations.<br/>advanced_experiments.py:686"]:::default
    n_advanced_experiments_py___calculate_energy_levels["advanced_experiments.QEDEffectsExperiment._calculate_ener...<br/>Calculate hydrogen energy levels including QE...<br/>advanced_experiments.py:761"]:::default
    n_advanced_experiments_py__run_polyatomic["advanced_experiments.AdvancedExperimentRunner.run_polyatomic<br/>Run polyatomic molecule experiment.<br/>advanced_experiments.py:1266"]:::default
    n_advanced_experiments_py__PolyatomicExperiment["advanced_experiments.PolyatomicExperiment<br/>Complete polyatomic molecule experiment.<br/>advanced_experiments.py:1105"]:::default
    n_advanced_experiments_py__run_analysis["advanced_experiments.PolyatomicExperiment.run_analysis<br/>Run complete analysis for a molecule.<br/>advanced_experiments.py:1123"]:::default
    n_advanced_experiments_py__run_pyscf["advanced_experiments.PolyatomicVQE.run_pyscf<br/>Run PySCF calculation for the molecule.<br/>advanced_experiments.py:999"]:::default
    n_advanced_experiments_py__run_all_1278["advanced_experiments.AdvancedExperimentRunner.run_all<br/>Run all experiments.<br/>advanced_experiments.py:1278"]:::default
    n_advanced_experiments_py___main_ -->|"1"| n_advanced_experiments_py___make_logger
    n_advanced_experiments_py___main_ -->|"2"| n_advanced_experiments_py__main
    n_advanced_experiments_py__main -->|"1"| n_advanced_experiments_py__AdvancedExperimentRunner
    n_advanced_experiments_py__main -->|"2"| n_advanced_experiments_py__run_grover
    n_advanced_experiments_py__run_grover -->|"1"| n_advanced_experiments_py__GroverConfig
    n_advanced_experiments_py__run_grover -->|"2"| n_advanced_experiments_py__GroverSearch
    n_advanced_experiments_py__run_grover -->|"3"| n_advanced_experiments_py__run
    n_advanced_experiments_py__run -->|"1"| n_advanced_experiments_py___calculate_entropy
    n_advanced_experiments_py__run -->|"2"| n_advanced_experiments_py__apply
    n_advanced_experiments_py__run -->|"3"| n_advanced_experiments_py__apply_206
    n_advanced_experiments_py__main -->|"3"| n_advanced_experiments_py__run_qed
    n_advanced_experiments_py__run_qed -->|"1"| n_advanced_experiments_py__QEDConfig
    n_advanced_experiments_py__run_qed -->|"2"| n_advanced_experiments_py__QEDEffectsExperiment
    n_advanced_experiments_py__run_qed -->|"3"| n_advanced_experiments_py__run_full_analysis
    n_advanced_experiments_py__run_full_analysis -->|"1"| n_advanced_experiments_py__compare_2s_2p
    n_advanced_experiments_py__run_full_analysis -->|"2"| n_advanced_experiments_py__full_report
    n_advanced_experiments_py__run_full_analysis -->|"3"| n_advanced_experiments_py___calculate_energy_levels
    n_advanced_experiments_py__main -->|"4"| n_advanced_experiments_py__run_polyatomic
    n_advanced_experiments_py__run_polyatomic -->|"1"| n_advanced_experiments_py__PolyatomicExperiment
    n_advanced_experiments_py__run_polyatomic -->|"2"| n_advanced_experiments_py__run_analysis
    n_advanced_experiments_py__run_analysis -->|"1"| n_advanced_experiments_py__run_pyscf
    n_advanced_experiments_py__main -->|"5"| n_advanced_experiments_py__run_all_1278
    n_advanced_experiments_py__run_all_1278 -.-> n_advanced_experiments_py__run_grover
    n_advanced_experiments_py__run_all_1278 -.-> n_advanced_experiments_py__run_qed
    n_advanced_experiments_py__run_all_1278 -.-> n_advanced_experiments_py__run_polyatomic
```

### `app (__main__)`

`app.py:406`, score 8: `if __name__ == "__main__"` guard.

Call tree:

```mermaid
flowchart LR
    classDef default fill:#111827,stroke:#38bdf8,color:#e5e7eb;
    classDef entry fill:#0e7490,stroke:#22d3ee,color:#ecfeff,stroke-width:2px;
    n_app_py___main_["app (__main__)<br/>app.py"]:::entry
    n_app_py__PolarizabilityCalculator["app.PolarizabilityCalculator<br/>app.py:237"]:::default
    n_app_py___main_ -->|"1"| n_app_py__PolarizabilityCalculator
```

---
