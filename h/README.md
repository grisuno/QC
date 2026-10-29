# h: a Hamiltonian network that computes the Hamiltonian

Proof of concept. Same network as the project's `HamiltonianBackboneNet` (1x1 projection, two spectral layers acting on the
rFFT2 of the 2D torus, 1x1 projection, GELU in between) and the same `state_dict` layout, so a checkpoint from here loads
strict as `hamiltonian.pth` in `quantum_computer.QuantumComputer`.

## What was wrong with the old checkpoint

`experiment2.py` trained against the evolved field with `MSELoss` and counted a sample as correct when its MSE was below 0.05.
For unit-norm 16x16 fields the MSE of the zero map is 1/256 = 0.0039, so the zero map passes. The lambda pressure
(`lambda * distance to the nearest integer`) then pulled every weight to zero, and `hamiltonian.pth` stored exactly
`val_loss = 1/256`, `val_acc = 1.0`, 99.996 % near-zero weights. The Maxwell checkpoint shows the same fingerprint
(`val_loss = 1/1536`).

## Corrections

| Point | Old | Here |
|-------|-----|------|
| Target | evolved field, norm restored | `H psi`, H = -(hbar^2/2m) laplacian on the torus (the operator the backend applies) |
| Loss / metric | MSE, accuracy at MSE < 0.05 | relative error `|out - H psi| / |H psi|` (the zero map scores exactly 1) |
| Inputs | unit-norm white noise | band-limited fields of random amplitude plus line fields like the quantum computer's basis amplitudes, random torus shifts |
| Lambda pressure | grows by decades regardless of accuracy | grows only while relative error < 1 %, backs off and restores the last good weights when it exceeds 3 %, float64 |
| Initial kernels | fixed | imaginary ratio chosen by the GOE/GUE level-statistics scan (phase 0) |
| Batch size, seed | fixed | batch prospecting (phase 1) and seed mining (phase 2), as in the Maxwell protocol |

## Protocol

```bash
python3 -m h.train --out h/runs/run1                    # phases 0-3, long run (hours on CPU)
python3 -m h.train --out h/runs/quick --quick            # minutes
python3 -m h.train --out h/runs/x --skip-mining --seed 3 --batch 32 --ratio 0.1 --epochs 1200
python3 -m h.train --out h/runs/run1 --target-ensemble goe --export h/weights/hamiltonian_h.pth
```

Phase 0 scans the GOE/GUE imaginary ratio (gap-ratio and Wigner-surmise statistics), phase 1 trains briefly at each batch size,
phase 2 probes seeds and keeps the cooling one, phase 3 trains long with the adaptive lambda. Checkpoints use the project's
format (`epoch`, `model_state_dict`, `optimizer_state_dict`, `metrics`, `config`, `lambda_pressure`); `best.pth` is the lowest
relative error, `latest.pth` the final pressure state.

## Monitoring a run

Phase 3 appends one JSON line per report to `<run>/progress.jsonl`; phases 0-2 print to the terminal (unbuffered).

```bash
python3 -m h.monitor h/runs/run1 --watch     # status, ETA, last rows, alarm if the output collapses to zero
python3 -m h.train --out h/runs/run1 2>&1 | tee h/runs/run1.log
```

## Validation and use

```bash
python3 -m h.validate h/weights/hamiltonian_h.pth       # loads through the project's own HamiltonianBackend
python3 -m h.research list
python3 -m h.research run hamiltonian [checkpoint]
python3 -m h.research run maxwell [checkpoint]          # audits the Maxwell network against the exact TM operator
Q2C_HAMILTONIAN=$PWD/h/weights/hamiltonian_h.pth .venv/bin/python qc_immersive.py --scene entangle
```

`Q2C_HAMILTONIAN` makes the viewer load another Hamiltonian checkpoint; the default stays `hamiltonian.pth`.

## Topology

`h/topology.py` measures the boundary winding of the channel-averaged complex kernel (as in `check_fase_berry.py`) and the
discrete Berry phase along a checkpoint sequence (as in `berry_phase_calculator.py`). These are the definitions of the
repositories; the paper's values (Berry +-10.26 rad, winding +-2) are not reproduced or claimed here.
