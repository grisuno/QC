"""Exact-physics checks that need no GPU, no checkpoints and no PySCF."""

import math

import numpy as np
import pytest

from immersive import extras, spin
from immersive.physics import energy_ev, overlap_1s, radial_function, view_radius


@pytest.mark.parametrize("n,l", [(1, 0), (2, 1), (3, 2), (4, 3), (12, 11)])
def test_radial_functions_are_normalised(n, l):
    r = np.linspace(0, view_radius(n) * 2, 200001)
    assert np.trapezoid(radial_function(n, l, r) ** 2 * r**2, r) == pytest.approx(1.0, abs=1e-4)


def test_energies_and_overlap():
    assert energy_ev(1) == pytest.approx(-13.6057, abs=1e-3)
    assert overlap_1s(0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("n,l,m", [(1, 0, 0), (2, 1, 0), (3, 2, 1), (4, 3, -2)])
def test_born_samples_follow_exact_mean_radius(n, l, m):
    rmax = view_radius(n)
    sampler = extras.BornSampler.build([(n, l, m, 1.0)], [1.0], rmax)
    points = sampler.draw(40000, np.random.default_rng(5))
    expected = (3 * n * n - l * (l + 1)) / 2
    assert sampler.mean_radius_exact == pytest.approx(expected, rel=0.01)
    assert np.linalg.norm(points, axis=1).mean() == pytest.approx(expected, rel=0.03)


def test_rydberg_packet_is_normalised_and_orbits_at_the_kepler_rate():
    packet = extras.RydbergPacket(n0=10, grid=64)
    assert sum(c * c for _, c in packet.weights) == pytest.approx(1.0)
    angles = []
    for fraction in (0.0, 0.1, 0.2):
        centre = packet.centroid(fraction * packet.period)
        angles.append(math.atan2(centre[1], centre[0]))
        assert packet.norm(fraction * packet.period) == pytest.approx(1.0, abs=0.02)
    assert angles[0] == pytest.approx(0.0, abs=0.05)
    assert angles[2] == pytest.approx(0.2 * 2 * math.pi, abs=0.12)


def test_spin_expectations_and_zeeman():
    link = spin.DiracBackendLink()
    for key, expected in (("1", 1.0), ("5", 1 / 3), ("6", -1 / 3)):
        preset = next(p for p in spin.SPIN_PRESETS if p.key == key)
        comps = preset.components
        energies = [spin.component_energy(c, link, preset.field_tesla) for c in comps]
        result = spin.measure(comps, spin.evolved_coefficients(comps, energies, 0.0), view_radius(2) * 0.9)
        assert result.sigma[2] == pytest.approx(expected, abs=0.01)


def test_clebsch_normalisation():
    for l, mj2, sign in ((0, 1, 1), (1, 3, 1), (1, 1, -1), (2, -3, 1)):
        up, down = spin.clebsch(l, mj2, sign)
        assert up**2 + down**2 == pytest.approx(1.0)
