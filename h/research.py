"""Research-suite runner for the H experiments.

    python3 -m h.research list
    python3 -m h.research run hamiltonian [checkpoint]
    python3 -m h.research run maxwell [checkpoint]
    python3 -m h.research run all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict

HERE = Path(__file__).resolve().parent
DEFAULT_HAMILTONIAN = HERE / "weights" / "hamiltonian_h.pth"


def run_hamiltonian(checkpoint: str | None = None) -> dict:
    from .validate import validate

    return validate(checkpoint or str(DEFAULT_HAMILTONIAN))


def run_maxwell(checkpoint: str | None = None) -> dict:
    from .experiments.maxwell import DEFAULT_CHECKPOINT, audit

    return audit(checkpoint or str(DEFAULT_CHECKPOINT))


EXPERIMENTS: Dict[str, tuple[Callable[..., dict], str]] = {
    "hamiltonian": (run_hamiltonian, "H network loaded through the project's HamiltonianBackend: operator error, Hermiticity, propagation, kernel topology"),
    "maxwell": (run_maxwell, "Maxwell network checkpoint versus the exact TM Maxwell operator on the torus"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("name", choices=[*EXPERIMENTS, "all"])
    run.add_argument("checkpoint", nargs="?")
    args = parser.parse_args(argv)
    if args.command == "list":
        for name, (_, doc) in EXPERIMENTS.items():
            print(f"{name:12} {doc}")
        return 0
    names = list(EXPERIMENTS) if args.name == "all" else [args.name]
    for name in names:
        print(f"== {name}")
        print(json.dumps(EXPERIMENTS[name][0](args.checkpoint if args.name != "all" else None), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
