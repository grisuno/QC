"""The H experiment: architecture parity with the project, anti-trivial metrics, mining pieces, checkpoint round trip."""

import math
from dataclasses import replace

import pytest
import torch

from h.config import HConfig
from h.data import HamiltonianDataset, TorusHamiltonian, validation_set
from h.ensembles import scan_imaginary_ratio
from h.lambda_pressure import AdaptiveLambda
from h.metrics import relative_error
from h.model import HamiltonianTorusNet
from h.topology import berry_trajectory, boundary_winding
from h.trainer import fit


@pytest.fixture(scope="module")
def tiny():
    return replace(HConfig(), train_samples=256, val_samples=64, log_every_epochs=5, ensemble_size=64, ensemble_matrices=3)


def test_same_state_dict_and_forward_as_the_projects_network():
    physics = pytest.importorskip("quantum_framework_physics")
    cfg = HConfig()
    mine = HamiltonianTorusNet(cfg)
    theirs = physics.HamiltonianBackboneNet(cfg.grid_size, cfg.hidden_dim, cfg.num_spectral_layers)
    assert {k: v.shape for k, v in mine.state_dict().items()} == {k: v.shape for k, v in theirs.state_dict().items()}
    theirs.load_state_dict(mine.state_dict(), strict=True)
    x = torch.randn(3, cfg.grid_size, cfg.grid_size)
    assert torch.allclose(mine(x), theirs(x), atol=1e-6)


def test_zero_output_scores_a_relative_error_of_one(tiny):
    x, y = validation_set(tiny)
    assert float(relative_error(torch.zeros_like(y), y).mean()) == pytest.approx(1.0)


def test_exact_operator_is_the_kinetic_energy_on_the_torus(tiny):
    op = TorusHamiltonian(tiny)
    g = tiny.grid_size
    k = 3 * 2 * math.pi / g
    field = torch.cos(k * torch.arange(g))[:, None].expand(g, g).contiguous()
    assert torch.allclose(op.apply(field), 0.5 * k**2 * field, atol=1e-4)


def test_short_training_beats_the_zero_solution(tiny):
    result = fit(tiny, seed=1, batch_size=32, imaginary_ratio=0.3, epochs=40, report_every=40)
    assert result["final"]["rel_error"] < 0.85
    assert result["final"]["norm_ratio"] > 0.3


def test_checkpoint_loads_in_the_project_backend_and_applies_h(tiny, tmp_path):
    qcm = pytest.importorskip("quantum_computer")
    fit(tiny, seed=1, batch_size=32, imaginary_ratio=0.3, epochs=40, out_dir=tmp_path, report_every=40)
    cfg = qcm.SimulatorConfig(device="cpu", hamiltonian_checkpoint=str(tmp_path / "latest.pth"))
    backend = qcm.QuantumComputer(cfg)._h_backend
    assert backend.net is not None
    x, y = validation_set(tiny)
    out = torch.stack([backend._apply_h(v).reshape(tiny.grid_size, tiny.grid_size) for v in x[:16]])
    assert float(relative_error(out, y[:16]).mean()) < 0.85


def test_level_statistics_separate_goe_from_gue():
    cfg = HConfig()
    gue, goe = scan_imaginary_ratio(cfg, "gue", seed=0), scan_imaginary_ratio(cfg, "goe", seed=0)
    ratios = {row["ratio"]: row["gap_ratio"] for row in gue["rows"]}
    assert goe["best_ratio"] == 0.0 and gue["best_ratio"] >= 0.1
    assert ratios[0.0] == pytest.approx(0.5307, abs=0.02) and ratios[1.0] == pytest.approx(0.5996, abs=0.02)


def test_winding_of_an_analytic_vortex_kernel_is_one():
    g = 16
    ky = torch.arange(g // 2 + 1)[:, None].float() - (g // 4)
    kx = torch.arange(g)[None, :].float() - g / 2
    kernel = torch.exp(1j * torch.atan2(ky, kx)).expand(2, 2, g // 2 + 1, g).clone()
    assert abs(boundary_winding(kernel)) == pytest.approx(1.0, abs=0.05)


def test_berry_trajectory_is_zero_for_a_static_sequence(tiny):
    state = HamiltonianTorusNet(tiny).state_dict()
    assert berry_trajectory([state, state, state])["total_berry_phase"] == pytest.approx(0.0, abs=1e-5)


def test_lambda_grows_only_while_accurate_and_backs_off_otherwise(tiny):
    cfg = replace(tiny, lambda_interval_epochs=10)
    pressure = AdaptiveLambda(cfg)
    start = float(pressure.value)
    assert pressure.update(10, cfg.accuracy_target / 2) == "grow" and float(pressure.value) > start
    grown = float(pressure.value)
    assert pressure.update(11, cfg.accuracy_tolerance * 2) == "backoff" and float(pressure.value) < grown
    assert pressure.update(12, (cfg.accuracy_target + cfg.accuracy_tolerance) / 2) == "hold"


def test_research_registry_lists_both_experiments():
    from h.research import EXPERIMENTS

    assert set(EXPERIMENTS) == {"hamiltonian", "maxwell"}


def test_maxwell_audit_flags_a_network_that_does_not_compute_maxwell(tmp_path):
    from h.experiments.maxwell import MaxwellTorusNet, audit

    torch.manual_seed(0)
    path = tmp_path / "random.pth"
    torch.save({"model_state_dict": MaxwellTorusNet().state_dict(), "epoch": 0, "metrics": {}}, path)
    result = audit(str(path))
    assert result["computes_maxwell"] is False and result["relative_error"] > 0.5


def test_exact_maxwell_operator_preserves_the_norm_it_is_given():
    from h.experiments.maxwell import make_batch

    x, y = make_batch(8, seed=3)
    real_in = x.reshape(8, 3, 2, 16, 16)[:, :, 0].flatten(1).norm(dim=1)
    real_out = y.reshape(8, 3, 2, 16, 16)[:, :, 0].flatten(1).norm(dim=1)
    assert torch.allclose(real_in, real_out, atol=1e-5)
