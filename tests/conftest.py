"""Shared setup for the deploy gate: repository root on sys.path, headless defaults, shared fixtures."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pytest


@pytest.fixture(scope="session")
def bridge():
    """The viewer's bridge to the project simulator, with the three network checkpoints loaded."""
    pytest.importorskip("torch")
    pytest.importorskip("quantum_computer")
    from immersive import backend

    return backend


def run_to_idle(register, dt: float = 0.02, limit: int = 2000):
    for _ in range(limit):
        if not register.busy:
            return
        register.update(dt)
    raise AssertionError("register never became idle")
