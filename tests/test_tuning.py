"""Fine-tuning engine of the 3D Fine-tune scene: exact targets tied to the backend, seen/unseen split, convergence, apply/reset."""

import math
import time

import pytest
import torch

pytest.importorskip("torch")
tuning = pytest.importorskip("immersive.tuning")


@pytest.fixture(scope="module")
def hub():
    pytest.importorskip("quantum_computer")
    from immersive.backend import BackendHub

    return BackendHub.shared()


def wait_for(session, predicate, timeout=90.0):
    start = time.time()
    while time.time() - start < timeout:
        snap = session.read()
        if snap.ready and snap.seen is not None and predicate(snap):
            return snap
        time.sleep(0.2)
    raise AssertionError(f"timeout: {session.read().status}")


def test_families_are_unit_scaled_and_deterministic(hub):
    task = tuning.OperatorTask(hub, "hamiltonian")
    step = lambda psi, t: task.free_step(psi, t)
    for family in tuning.DATA_MODES:
        a = tuning.make_fields(family, 12, task.grid, task.base_field, step, torch.Generator().manual_seed(5))
        b = tuning.make_fields(family, 12, task.grid, task.base_field, step, torch.Generator().manual_seed(5))
        norms = tuning.complex_norm(a)
        assert torch.equal(a, b)
        assert float(norms.min()) >= tuning.SCALE_RANGE[0] - 1e-5 and float(norms.max()) <= tuning.SCALE_RANGE[1] + 1e-5


def test_hamiltonian_exact_target_is_half_k_squared(hub):
    task = tuning.OperatorTask(hub, "hamiltonian")
    g = task.grid
    k = 3 * 2 * math.pi / g
    field = torch.cos(k * torch.arange(g, dtype=torch.float32))[:, None].expand(g, g).contiguous()[None]
    assert torch.allclose(task.exact(field), 0.5 * k**2 * field, atol=1e-4)


def test_schrodinger_exact_target_is_the_backends_own_analytical_propagator(hub):
    import quantum_computer as qcm

    task = tuning.OperatorTask(hub, "schrodinger")
    analytic = qcm.HamiltonianBackend(qcm.SimulatorConfig(device="cpu", hamiltonian_checkpoint=""))
    amp = hub.new_state(1).amplitudes[0].detach().cpu().clone()
    x = amp[None].float()
    expected = analytic.evolve_amplitude(amp, task.dt)
    assert analytic.net is None
    assert torch.allclose(task.exact(x)[0], expected, atol=1e-5)


def test_dirac_exact_target_preserves_the_spinor_norm(hub):
    task = tuning.OperatorTask(hub, "dirac")
    step = lambda psi, t: task.free_step(psi, t)
    psi = tuning.make_fields("smooth", 4, task.grid, task.base_field, step, torch.Generator().manual_seed(1))
    x = task.encode(psi)
    y = task.exact(x)
    assert x.shape == (4, 8, task.grid, task.grid) and torch.isfinite(y).all()
    assert torch.allclose(y.flatten(1).norm(dim=1), x.flatten(1).norm(dim=1), rtol=1e-4)


def test_seen_and_unseen_sets_never_share_a_sample(hub):
    session = tuning.TuneSession("hamiltonian", seed=11)
    try:
        wait_for(session, lambda s: True)
        train = session.train_x.flatten(1)
        for family, (x, _) in session.val.items():
            gram = (x.flatten(1) @ train.T).abs().max()
            norms = x.flatten(1).norm(dim=1).max() * train.norm(dim=1).max()
            assert float(gram / norms) < 0.999
    finally:
        session.stop()


def test_hamiltonian_session_improves_on_seen_and_unseen_data(hub):
    session = tuning.TuneSession("hamiltonian", seed=3)
    try:
        before = wait_for(session, lambda s: True)
        assert before.seen["err_baseline"] == pytest.approx(1.0, abs=0.05) or before.seen["err_baseline"] < 1.0
        session.request("toggle")
        after = wait_for(session, lambda s: s.step >= 400)
        assert after.seen["err_baseline"] < 0.5 * before.seen["err_baseline"]
        assert all(row["err_baseline"] < 0.6 * before.unseen[f]["err_baseline"] for f, row in after.unseen.items())
        assert after.plates.shape == (4, tuning.PLATE_RES, tuning.PLATE_RES, 4) and bool((after.plates == after.plates).all())
        assert set(["delta", "alpha", "kappa", "t_eff_book", "berry_total", "winding_total"]) <= set(after.weights)
    finally:
        session.stop()


def test_apply_pushes_weights_to_the_backend_and_reset_restores_them(hub):
    backend = hub.backend("hamiltonian")
    original = {k: v.clone() for k, v in backend.net.state_dict().items()}
    session = tuning.TuneSession("hamiltonian", seed=4)
    try:
        wait_for(session, lambda s: True)
        session.request("toggle")
        wait_for(session, lambda s: s.step >= 60)
        session.request("toggle")
        session.request("apply")
        wait_for(session, lambda s: s.applied)
        changed = any(not torch.equal(original[k], v) for k, v in backend.net.state_dict().items())
        assert changed
        session.request("reset")
        wait_for(session, lambda s: not s.applied)
        assert all(torch.equal(original[k], v) for k, v in backend.net.state_dict().items())
    finally:
        session.stop()
        backend.net.load_state_dict(original)


def test_mse_loss_keeps_the_zero_solution_alive(hub):
    session = tuning.TuneSession("hamiltonian", loss_kind="mse", seed=5)
    try:
        wait_for(session, lambda s: True)
        session.request("toggle")
        snap = wait_for(session, lambda s: s.step >= 150)
        assert snap.seen["err_baseline"] > 0.9
    finally:
        session.stop()
