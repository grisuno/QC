"""Load a checkpoint through the project's own HamiltonianBackend / QuantumComputer and check it computes H.

    python3 -m h.validate h/runs/run1/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .config import HConfig  # noqa: E402
from .data import TorusHamiltonian, validation_set  # noqa: E402
from .metrics import evaluate, propagation_fidelity  # noqa: E402
from .topology import kernel_report  # noqa: E402


def validate(checkpoint: str) -> dict:
    logging.disable(logging.CRITICAL)
    import quantum_computer as qcm

    config = HConfig()
    cfg = qcm.SimulatorConfig(device="cpu", hamiltonian_checkpoint=str(checkpoint))
    computer = qcm.QuantumComputer(cfg)
    backend = computer._h_backend
    if backend.net is None:
        return {"loaded": False}
    val_x, val_y = validation_set(config)
    via_backend = torch.stack([backend._apply_h(x).reshape(config.grid_size, config.grid_size) for x in val_x[:128]])
    exact = TorusHamiltonian(config).apply(val_x[:128])
    operator = (via_backend - exact).flatten(1).norm(dim=1) / exact.flatten(1).norm(dim=1)
    state0 = computer._factory.all_zeros(1)
    project_fields = state0.amplitudes[0]
    project_error = []
    for channel in range(2):
        field = project_fields[channel]
        if float(field.norm()) < 1e-9:
            continue
        reference = TorusHamiltonian(config).apply(field.unsqueeze(0))[0]
        project_error.append(float((backend._apply_h(field).reshape_as(reference) - reference).norm() / reference.norm()))
    circuit = qcm.QuantumCircuit(2)
    circuit.h(0).cnot(0, 1).evolve([0], steps=20)
    state = computer._factory.all_zeros(2)
    for inst in circuit._instructions:
        state = qcm._GATE_REGISTRY[inst.gate_name].apply(state, backend, inst.targets, inst.params)
    probs = state.probabilities().tolist()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "loaded": True,
        "operator_rel_error_via_backend": float(operator.mean()),
        "operator_rel_error_on_project_fields": project_error,
        "metrics": evaluate(backend.net, val_x, val_y, config),
        "propagation": propagation_fidelity(backend.net, config),
        "bell_probabilities_after_evolve": probs,
        "kernels": kernel_report(raw["model_state_dict"]),
        "lambda_pressure": raw.get("lambda_pressure"),
        "epoch": raw.get("epoch"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    result = validate(parser.parse_args(argv).checkpoint)
    print(json.dumps(result, indent=1, default=float))
    return 0 if result.get("loaded") else 1


if __name__ == "__main__":
    sys.exit(main())
