"""Tests for the immersive viewer: physics kernels, backend bridge, spin, PySCF fields, launcher, GL smoke test."""

import math
import os

import numpy as np
import pytest
from scipy.special import sph_harm_y

from immersive.physics import (PRESETS, QubitRegister, angular_constants, density_peak, energy_ev, overlap_1s,
                               radial_function, view_radius, ylm_from_table)

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("l", range(4))
def test_angular_table_matches_scipy(l):
    rng = np.random.default_rng(1)
    theta, phi = rng.uniform(0, np.pi, 40), rng.uniform(0, 2 * np.pi, 40)
    consts = angular_constants()
    for m in range(-l, l + 1):
        assert np.allclose(ylm_from_table(l, m, theta, phi, consts), sph_harm_y(l, m, theta, phi))


@pytest.mark.parametrize("n,l", [(1, 0), (2, 1), (3, 2), (4, 3)])
def test_radial_functions_are_normalised(n, l):
    r = np.linspace(0, view_radius(n) * 2, 60001)
    assert np.trapezoid(radial_function(n, l, r) ** 2 * r**2, r) == pytest.approx(1.0, abs=1e-4)


def test_energy_levels():
    assert energy_ev(1) == pytest.approx(-13.6057, abs=1e-3)
    assert energy_ev(2) == pytest.approx(-3.4014, abs=1e-3)


def test_presets_are_normalised_and_fit_shader():
    for preset in PRESETS:
        assert 1 <= len(preset.states) <= 4
        assert sum(abs(c) ** 2 for *_, c in preset.states) == pytest.approx(1.0)
        qn = [(n, l, m) for n, l, m, _ in preset.states]
        coefs = np.array([c for *_, c in preset.states])
        assert density_peak(qn, coefs, view_radius(max(n for n, _, _ in qn))) > 0


def test_overlap_limits():
    assert overlap_1s(0.0) == pytest.approx(1.0)
    assert overlap_1s(10.0) == pytest.approx(0.0020127, abs=1e-6)


def test_ideal_register_bell_pair():
    reg = QubitRegister(2)
    reg.queue(reg.single("H", 0))
    reg.queue(reg.cnot(0, 1))
    for _ in range(400):
        reg.update(0.01)
    assert reg.entropy(0) == pytest.approx(1.0, abs=1e-6)
    counts = reg.measure(500, np.random.default_rng(3))
    assert counts["01"] == 0 and counts["10"] == 0


# ---------------------------------------------------------------- project backend bridge

@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("quantum_computer")
    from immersive import backend as bridge

    return bridge


def run_to_idle(reg, dt=0.02):
    for _ in range(2000):
        if not reg.busy:
            return
        reg.update(dt)
    raise AssertionError("register never became idle")


def test_backend_checkpoints_are_loaded(backend):
    hub = backend.BackendHub.shared()
    assert all(hub.loaded.values()), hub.loaded


def test_backend_bell_pair_matches_ideal_unitary(backend):
    reg = backend.BackendRegister(2)
    reg.queue(reg.single("H", 0))
    reg.queue(reg.cnot(0, 1))
    run_to_idle(reg)
    probs = reg.probabilities()
    assert probs[0] == pytest.approx(0.5, abs=1e-4) and probs[3] == pytest.approx(0.5, abs=1e-4)
    assert reg.entropy(0) == pytest.approx(1.0, abs=1e-4)
    assert reg.deviation() < 1e-5


@pytest.mark.parametrize("gate", ["H", "X", "Y", "Z", "S", "T", "Rx", "Ry"])
def test_backend_single_gates_match_ideal(backend, gate):
    reg = backend.BackendRegister(1)
    reg.queue(reg.single("H", 0))
    reg.queue(reg.single(gate, 0))
    run_to_idle(reg)
    assert reg.deviation() < 1e-5


@pytest.mark.xfail(strict=False, reason="JointHilbertState.bloch_vector mirrors the y axis (Rx(pi/2)|0> gives +y, standard is -y)")
def test_project_bloch_vector_uses_standard_y_convention(backend):
    reg = backend.BackendRegister(1)
    reg.queue(reg.single("Rx", 0))
    run_to_idle(reg)
    assert reg.backend_bloch(0)[1] == pytest.approx(reg.bloch(0)[1], abs=1e-5)


def test_backend_spatial_evolution_preserves_probabilities(backend):
    reg = backend.BackendRegister(2)
    reg.queue(reg.single("H", 0))
    run_to_idle(reg)
    before = reg.probabilities()
    reg.backend_name = "hamiltonian"
    reg.evolve(3)
    assert reg.error == ""
    assert reg.probabilities() == pytest.approx(before, abs=1e-4)


# ---------------------------------------------------------------- spin physics

@pytest.fixture(scope="module")
def spin_mod():
    from immersive import spin

    return spin


@pytest.mark.parametrize("l,mj2,sign", [(0, 1, 1), (1, 1, 1), (1, 3, 1), (1, 1, -1), (2, -3, 1), (2, 1, -1)])
def test_clebsch_coefficients_are_normalised(spin_mod, l, mj2, sign):
    up, dn = spin_mod.clebsch(l, mj2, sign)
    assert up**2 + dn**2 == pytest.approx(1.0)


@pytest.mark.parametrize("key,expected", [("1", 1.0), ("4", 1.0), ("5", 1 / 3), ("6", -1 / 3)])
def test_spin_expectation_values(spin_mod, key, expected):
    preset = next(p for p in spin_mod.SPIN_PRESETS if p.key == key)
    comps = preset.components
    link = spin_mod.DiracBackendLink()
    energies = [spin_mod.component_energy(c, link, preset.field_tesla) for c in comps]
    reach = view_radius(max(c.n for c in comps)) * 0.9
    result = spin_mod.measure(comps, spin_mod.evolved_coefficients(comps, energies, 0.0), reach)
    assert result.norm == pytest.approx(1.0, abs=0.01)
    assert result.sigma[2] == pytest.approx(expected, abs=0.01)


def test_larmor_precession_rotates_spin_about_field(spin_mod):
    preset = next(p for p in spin_mod.SPIN_PRESETS if p.key == "2")
    comps = preset.components
    link = spin_mod.DiracBackendLink()
    energies = [spin_mod.component_energy(c, link, preset.field_tesla) for c in comps]
    period = 2 * math.pi / abs(energies[0] - energies[1])
    sigma = [spin_mod.measure(comps, spin_mod.evolved_coefficients(comps, energies, period * f), 7.0).sigma for f in (0.0, 0.25, 0.5)]
    assert sigma[0] == pytest.approx([1, 0, 0], abs=0.01)
    assert sigma[1] == pytest.approx([0, 1, 0], abs=0.01)
    assert sigma[2] == pytest.approx([-1, 0, 0], abs=0.01)
    ghz = abs(energies[0] - energies[1]) * spin_mod.HARTREE_HZ / 1e9
    assert ghz == pytest.approx(2.0 * 13.9962 * 5.0, rel=0.01)


def test_dirac_backend_gives_literature_fine_structure(spin_mod):
    link = spin_mod.DiracBackendLink()
    if not link.ok:
        pytest.skip(link.error)
    lower = spin_mod.SpinorComponent(2, 1, -1, 1, 1.0)
    upper = spin_mod.SpinorComponent(2, 1, 1, 1, 1.0)
    ghz = (link.energy(upper) - link.energy(lower)) * spin_mod.HARTREE_HZ / 1e9
    assert ghz == pytest.approx(10.97, rel=0.005)


# ---------------------------------------------------------------- PySCF fields

def test_matter_fields_conserve_electrons_and_spin():
    pytest.importorskip("pyscf")
    from immersive import matter

    water = next(s for s in matter.LIBRARY if s.key == "H2O")
    solved = matter.solve(water)
    dv = (2 * solved.half / (matter.GRID - 1)) ** 3
    amp = matter.field_amplitude(solved, "density")
    assert (amp**2).sum() * dv == pytest.approx(10.0, rel=0.02)
    assert solved.e_total == pytest.approx(-75.984, abs=0.01)
    oxygen = next(s for s in matter.LIBRARY if s.key == "O2")
    triplet = matter.solve(oxygen)
    spin_amp = matter.field_amplitude(triplet, "spin")
    dv = (2 * triplet.half / (matter.GRID - 1)) ** 3
    assert (np.sign(spin_amp) * spin_amp**2).sum() * dv == pytest.approx(2.0, abs=0.02)


def test_project_vqe_reaches_fci():
    pytest.importorskip("quantum_lab")
    from immersive.backend import run_project_vqe

    result = run_project_vqe()
    assert abs(result["error"]) < 1e-6


def test_launcher_lists_every_scene():
    qc_launcher = pytest.importorskip("qc_launcher")
    keys = {e.key for e in qc_launcher.EXPERIENCES}
    assert {"tour", "atom", "spin", "matter", "bloch", "entangle", "lab"} <= keys


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display and an OpenGL 3.3 driver")
@pytest.mark.parametrize("scene", ["atom", "spin", "bloch", "entangle", "matter"])
def test_gl_scenes_render(scene, tmp_path):
    if scene == "matter":
        pytest.importorskip("pyscf")
    from immersive.app import App, parse_args

    out = tmp_path / f"{scene}.png"
    app = App(parse_args(["--screenshot", str(out), "--scene", scene, "--frames", "5", "--size", "640x360"]))
    app.run()
    assert out.exists() and out.stat().st_size > 5000
    import pygame

    pygame.quit()
