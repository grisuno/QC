"""All hyper-parameters of the H experiment in one dataclass (no magic numbers elsewhere)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class HConfig:
    grid_size: int = 16
    hidden_dim: int = 32
    num_spectral_layers: int = 2
    mass: float = 1.0
    hbar: float = 1.0

    kinetic_prefactor: float = 0.5
    cutoff_min: float = 1.0
    cutoff_max: float = 3.14
    scale_min: float = 0.2
    scale_max: float = 1.0
    line_fraction: float = 0.3
    line_modes: int = 3
    train_samples: int = 4096
    val_samples: int = 512
    data_seed: int = 999

    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 0.0
    epochs: int = 20000
    grad_clip: float = 1.0
    cosine_floor: float = 0.01

    accuracy_target: float = 0.01
    accuracy_tolerance: float = 0.03
    norm_ratio_tolerance: float = 0.02

    batch_candidates: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    batch_prospect_epochs: int = 600
    batch_prospect_seed: int = 42

    mining_attempts: int = 24
    mining_start_seed: int = 1
    mining_prospect_epochs: int = 300
    mining_report_interval: int = 100

    ratio_candidates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.18, 0.3, 0.5, 0.75, 1.0])
    ensemble_size: int = 128
    ensemble_matrices: int = 32
    spacing_weight: float = 0.5
    correlation_weight: float = 0.5
    init_std: float = 0.1

    lambda_initial: float = 1e-8
    lambda_floor: float = 1e-12
    lambda_max: float = 1e35
    lambda_growth: float = 10.0
    lambda_backoff: float = 100.0
    lambda_interval_epochs: int = 250
    lattice_step: float = 1.0

    checkpoint_every_epochs: int = 500
    max_checkpoints: int = 10
    log_every_epochs: int = 100
    device: str = "cpu"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "HConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
