"""The viewer must show the project's own computation: gates, checkpoints, Dirac energies, Lamb shift."""

import pytest

from conftest import run_to_idle


def test_all_three_checkpoints_load(bridge):
    assert all(bridge.BackendHub.shared().loaded.values())


def test_bell_pair_through_project_quantum_computer(bridge):
    reg = bridge.BackendRegister(2)
    reg.queue(reg.single("H", 0))
    reg.queue(reg.cnot(0, 1))
    run_to_idle(reg)
    probs = reg.probabilities()
    assert probs[0] == pytest.approx(0.5, abs=1e-4) and probs[3] == pytest.approx(0.5, abs=1e-4)
    assert reg.entropy(0) == pytest.approx(1.0, abs=1e-4)
    assert reg.deviation() < 1e-5


@pytest.mark.parametrize("gate", ["H", "X", "Y", "Z", "S", "T", "Rx", "Ry"])
def test_single_gates_match_the_ideal_unitary(bridge, gate):
    reg = bridge.BackendRegister(1)
    reg.queue(reg.single("H", 0))
    reg.queue(reg.single(gate, 0))
    run_to_idle(reg)
    assert reg.deviation() < 1e-5


def test_project_bloch_vector_uses_the_standard_y_axis(bridge):
    reg = bridge.BackendRegister(1)
    reg.queue(reg.single("Rx", 0))
    run_to_idle(reg)
    assert reg.backend_bloch(0)[1] == pytest.approx(reg.bloch(0)[1], abs=1e-5)
    assert reg.backend_bloch(0)[1] < -0.99


def test_dirac_backend_accepts_float64_input(bridge):
    import torch

    hub = bridge.BackendHub.shared()
    amp = torch.rand(2, hub.grid, hub.grid, dtype=torch.float64)
    out = hub.backend("dirac").evolve_amplitude(amp, 0.01)
    assert out.shape == amp.shape and torch.isfinite(out).all()


def test_dirac_fine_structure_matches_literature():
    from immersive import spin

    link = spin.DiracBackendLink()
    if not link.ok:
        pytest.skip(link.error)
    lower, upper = spin.SpinorComponent(2, 1, -1, 1, 1.0), spin.SpinorComponent(2, 1, 1, 1, 1.0)
    assert (link.energy(upper) - link.energy(lower)) * spin.HARTREE_HZ / 1e9 == pytest.approx(10.97, rel=0.005)


@pytest.mark.parametrize("n,l,j,expected_mhz,tolerance", [
    (1, 0, 0.5, 8172.0, 0.01),
    (2, 0, 0.5, 1045.0, 0.01),
    (2, 1, 0.5, -12.8, 0.03),
])
def test_lamb_shift_matches_literature(n, l, j, expected_mhz, tolerance):
    pytest.importorskip("torch")
    from quantum_framework_physics import LambShiftCalculator

    shift = LambShiftCalculator().full_lamb_shift(n, l, j)["total_MHz"]
    assert shift == pytest.approx(expected_mhz, rel=tolerance)


def test_lamb_2s_2p_splitting_is_the_famous_1057_mhz():
    pytest.importorskip("torch")
    from quantum_framework_physics import LambShiftCalculator

    calc = LambShiftCalculator()
    gap = calc.full_lamb_shift(2, 0, 0.5)["total_MHz"] - calc.full_lamb_shift(2, 1, 0.5)["total_MHz"]
    assert gap == pytest.approx(1057.8, rel=0.01)


def test_hamiltonian_audit_reports_the_networks_real_quality(bridge):
    result = bridge.hamiltonian_check()
    assert {"hermiticity", "kinetic_gap", "fidelity", "idle_fidelity"} <= set(result)
    assert 0.0 <= result["fidelity"] <= 1.0 + 1e-6


def test_register_applies_the_project_hamiltonian_operator(bridge):
    reg = bridge.BackendRegister(2)
    reg.queue(reg.single("H", 0))
    run_to_idle(reg)
    energy = reg.hamiltonian_expectation()
    assert set(energy) == {"network", "reference"}
    assert all(abs(v) < 1e6 for v in energy.values())
