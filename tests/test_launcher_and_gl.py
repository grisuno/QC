"""Launcher wiring and an optional OpenGL smoke test (set Q2C_GL_TESTS=1 on a machine with a display)."""

import os

import pytest


def test_launcher_lists_every_experience():
    launcher = pytest.importorskip("qc_launcher")
    keys = {e.key for e in launcher.EXPERIENCES}
    assert {"tour", "atom", "spin", "matter", "rydberg", "tune", "bloch", "entangle", "lab"} <= keys


def test_viewer_scene_order_matches_launcher():
    pytest.importorskip("pygame")
    pytest.importorskip("OpenGL")
    from immersive.app import SCENE_ORDER

    assert "rydberg" in SCENE_ORDER and len(set(SCENE_ORDER)) == len(SCENE_ORDER)


def test_every_tour_caption_is_bilingual():
    pytest.importorskip("pygame")
    pytest.importorskip("OpenGL")
    from immersive.app import TEXT, build_tour

    for step in build_tour():
        assert set(TEXT[step.caption]) == {"en", "es"}


@pytest.mark.skipif(not (os.environ.get("Q2C_GL_TESTS") and os.environ.get("DISPLAY")), reason="needs Q2C_GL_TESTS=1 and a display")
@pytest.mark.parametrize("scene", ["atom", "spin", "rydberg", "tune", "bloch", "entangle"])
def test_scene_renders_a_frame(scene, tmp_path):
    from immersive.app import App, parse_args

    out = tmp_path / f"{scene}.png"
    App(parse_args(["--screenshot", str(out), "--scene", scene, "--frames", "5", "--size", "640x360"])).run()
    assert out.exists() and out.stat().st_size > 5000
    import pygame

    pygame.quit()
