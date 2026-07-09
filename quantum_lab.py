#!/usr/bin/env python3
"""
Q2C Quantum Lab - Interactive Educational TUI
=============================================
A guided, bilingual (English/Spanish) terminal experience for learning
quantum computing and quantum chemistry with the Q2C simulator.

It is built on top of the real simulation engine (quantum_framework_core):
every probability bar, entropy value and orbital picture you see is
computed live, not pre-recorded.

Usage:
    python quantum_lab.py                # interactive, asks for language
    python quantum_lab.py --lang es      # start in Spanish
    python quantum_lab.py --lesson 3     # jump straight into lesson 3

Sections:
    1. Lessons     - guided course: qubits, measurement, entanglement,
                     Grover search and quantum chemistry
    2. Playground  - build circuits gate by gate, watch the state evolve
    3. Chemistry   - molecule cards and an ASCII hydrogen-orbital viewer
    4. Glossary    - quick reference of the vocabulary used everywhere

Author: Gris Iscomeback
License: AGPL v3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    print("This program requires PyTorch:  pip install torch")
    sys.exit(1)

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    from rich.prompt import Prompt
    from rich.align import Align
    from rich.columns import Columns
    from rich.live import Live
except ImportError:
    print("This program requires 'rich':  pip install rich")
    sys.exit(1)

from quantum_framework_core import (
    ConfigLoader,
    FrameworkConfig,
    MPSQuantumComputer,
    run_grover_search,
)

console = Console()

BAR_WIDTH = 40
MAX_PLAYGROUND_QUBITS = 8
STATEVECTOR_DISPLAY_LIMIT = 5
DENSITY_RAMP = " ·:-=+*#%@"

# =============================================================================
# Bilingual text catalog
# =============================================================================

TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "Q2C QUANTUM LAB",
        "subtitle": "Learn quantum computing & chemistry by doing",
        "menu_lessons": "Lessons (guided course)",
        "menu_playground": "Playground (build your own circuits)",
        "menu_chemistry": "Chemistry lab (molecules & orbitals)",
        "menu_glossary": "Glossary",
        "menu_lang": "Cambiar a espanol",
        "menu_quit": "Quit",
        "choose": "Choose an option",
        "press_enter": "[dim]Press Enter to continue...[/dim]",
        "lessons_title": "GUIDED COURSE",
        "lesson_done": "Lesson complete!",
        "back": "Back",
        "quiz_title": "Quick check",
        "quiz_correct": "Correct!",
        "quiz_wrong": "Not quite. The right answer was",
        "your_answer": "Your answer",
        "playground_title": "CIRCUIT PLAYGROUND",
        "playground_qubits": "How many qubits? (1-8)",
        "playground_help": (
            "Commands:\n"
            "  h N / x N / y N / z N / s N / t N   single-qubit gates on qubit N\n"
            "  rx N THETA / ry N THETA / rz N THETA  rotations (THETA in radians)\n"
            "  cnot C T / cz C T / swap A B          two-qubit gates\n"
            "  sample N      simulate N measurements\n"
            "  amps          show amplitudes (phase + magnitude)\n"
            "  undo          remove last gate\n"
            "  reset         start over\n"
            "  bell / ghz    load a preset entangled state\n"
            "  help          show this message\n"
            "  q             leave the playground"
        ),
        "playground_prompt": "playground",
        "invalid": "Invalid command. Type 'help' for the command list.",
        "state_panel": "Quantum state",
        "circuit_panel": "Circuit",
        "entropy_label": "Entanglement entropy (middle cut)",
        "memory_label": "MPS memory",
        "samples_title": "Measurement results",
        "amplitudes_title": "Amplitudes",
        "amp_too_big": "Amplitude table available for up to {n} qubits.",
        "basis": "Basis state",
        "probability": "Probability",
        "amplitude": "Amplitude",
        "phase": "Phase",
        "counts": "Counts",
        "chem_title": "CHEMISTRY LAB",
        "chem_molecules": "Molecule explorer",
        "chem_orbitals": "Hydrogen orbital viewer (ASCII art)",
        "molecule_pick": "Pick a molecule",
        "orbital_pick": "Pick an orbital",
        "mol_field_formula": "Formula",
        "mol_field_desc": "Description",
        "mol_field_electrons": "Electrons",
        "mol_field_orbitals": "Spatial orbitals",
        "mol_field_qubits": "Qubits needed (Jordan-Wigner)",
        "mol_field_bond": "Bond length",
        "mol_field_hf": "Hartree-Fock energy",
        "mol_field_fci": "Exact (FCI) energy",
        "mol_field_corr": "Correlation energy",
        "mol_field_basis": "Basis set",
        "mol_explainer": (
            "The [bold]correlation energy[/bold] is the part that simple\n"
            "mean-field theory (Hartree-Fock) misses. Capturing it is the\n"
            "whole point of quantum algorithms like VQE: electrons dance\n"
            "together (they are correlated), and a quantum computer can\n"
            "represent that dance natively."
        ),
        "orbital_legend": (
            "[cyan]cyan[/cyan] = wavefunction positive   "
            "[magenta]magenta[/magenta] = negative   "
            "brightness = probability density |psi|^2\n"
            "White gaps between colored lobes are [bold]nodes[/bold]: "
            "places where the electron is never found."
        ),
        "orbital_info": "n={n}  l={l}  plane={plane}  box=±{extent:.0f} a0",
        "glossary_title": "GLOSSARY",
        "lang_set": "Language set to English.",
        "goodbye": "Thanks for visiting the Quantum Lab. Keep exploring!",
        "run_demo": "[bold yellow]>>> Running live simulation...[/bold yellow]",
        "theory": "theory",
        "simulated": "simulated",
        "clouds_bonding": "Bonding cloud (qubit 0)",
        "clouds_antibonding": "Antibonding cloud (qubit 1)",
        "clouds_note": (
            "Two protons sit on the horizontal axis, 0.74 A apart.\n"
            "[cyan]Bonding[/cyan]: the electron cloud piles up BETWEEN the nuclei and\n"
            "glues the molecule together. In the [bold]antibonding[/bold] cloud the\n"
            "wavefunction changes sign ([cyan]cyan[/cyan] -> [magenta]magenta[/magenta]) halfway: that empty\n"
            "vertical gap is a node, and an electron living there pushes the\n"
            "atoms apart instead of binding them."
        ),
        "vqe_ansatz_title": "The ansatz circuit (one knob: theta)",
        "vqe_landscape_title": "Energy landscape E(theta) - every dot is a live circuit run",
        "vqe_live_title": "VQE live - gradient descent on the landscape",
        "vqe_corr_label": "correlation captured",
        "vqe_final": ("Converged in {it} iterations:  E = {e:+.6f} Ha   "
                      "(FCI {fci:+.6f} Ha, error {err:.1e} Ha)"),
    },
    "es": {
        "title": "Q2C LABORATORIO CUANTICO",
        "subtitle": "Aprende computacion y quimica cuantica haciendo",
        "menu_lessons": "Lecciones (curso guiado)",
        "menu_playground": "Playground (arma tus propios circuitos)",
        "menu_chemistry": "Laboratorio de quimica (moleculas y orbitales)",
        "menu_glossary": "Glosario",
        "menu_lang": "Switch to English",
        "menu_quit": "Salir",
        "choose": "Elige una opcion",
        "press_enter": "[dim]Presiona Enter para continuar...[/dim]",
        "lessons_title": "CURSO GUIADO",
        "lesson_done": "Leccion completada!",
        "back": "Volver",
        "quiz_title": "Mini examen",
        "quiz_correct": "Correcto!",
        "quiz_wrong": "No exactamente. La respuesta correcta era",
        "your_answer": "Tu respuesta",
        "playground_title": "PLAYGROUND DE CIRCUITOS",
        "playground_qubits": "Cuantos qubits? (1-8)",
        "playground_help": (
            "Comandos:\n"
            "  h N / x N / y N / z N / s N / t N   compuertas de 1 qubit sobre el qubit N\n"
            "  rx N THETA / ry N THETA / rz N THETA  rotaciones (THETA en radianes)\n"
            "  cnot C T / cz C T / swap A B          compuertas de 2 qubits\n"
            "  sample N      simula N mediciones\n"
            "  amps          muestra amplitudes (fase + magnitud)\n"
            "  undo          quita la ultima compuerta\n"
            "  reset         empieza de nuevo\n"
            "  bell / ghz    carga un estado entrelazado de ejemplo\n"
            "  help          muestra este mensaje\n"
            "  q             salir del playground"
        ),
        "playground_prompt": "playground",
        "invalid": "Comando invalido. Escribe 'help' para ver la lista.",
        "state_panel": "Estado cuantico",
        "circuit_panel": "Circuito",
        "entropy_label": "Entropia de entrelazamiento (corte central)",
        "memory_label": "Memoria MPS",
        "samples_title": "Resultados de medicion",
        "amplitudes_title": "Amplitudes",
        "amp_too_big": "La tabla de amplitudes esta disponible hasta {n} qubits.",
        "basis": "Estado base",
        "probability": "Probabilidad",
        "amplitude": "Amplitud",
        "phase": "Fase",
        "counts": "Conteos",
        "chem_title": "LABORATORIO DE QUIMICA",
        "chem_molecules": "Explorador de moleculas",
        "chem_orbitals": "Visor de orbitales de hidrogeno (arte ASCII)",
        "molecule_pick": "Elige una molecula",
        "orbital_pick": "Elige un orbital",
        "mol_field_formula": "Formula",
        "mol_field_desc": "Descripcion",
        "mol_field_electrons": "Electrones",
        "mol_field_orbitals": "Orbitales espaciales",
        "mol_field_qubits": "Qubits necesarios (Jordan-Wigner)",
        "mol_field_bond": "Longitud de enlace",
        "mol_field_hf": "Energia Hartree-Fock",
        "mol_field_fci": "Energia exacta (FCI)",
        "mol_field_corr": "Energia de correlacion",
        "mol_field_basis": "Base",
        "mol_explainer": (
            "La [bold]energia de correlacion[/bold] es la parte que la teoria\n"
            "de campo medio (Hartree-Fock) no captura. Capturarla es el\n"
            "objetivo de algoritmos cuanticos como VQE: los electrones se\n"
            "mueven coordinados (estan correlacionados) y una computadora\n"
            "cuantica puede representar ese baile de forma nativa."
        ),
        "orbital_legend": (
            "[cyan]cian[/cyan] = funcion de onda positiva   "
            "[magenta]magenta[/magenta] = negativa   "
            "brillo = densidad de probabilidad |psi|^2\n"
            "Los huecos blancos entre lobulos de color son [bold]nodos[/bold]: "
            "lugares donde el electron nunca aparece."
        ),
        "orbital_info": "n={n}  l={l}  plano={plane}  caja=±{extent:.0f} a0",
        "glossary_title": "GLOSARIO",
        "lang_set": "Idioma cambiado a espanol.",
        "goodbye": "Gracias por visitar el Laboratorio Cuantico. Sigue explorando!",
        "run_demo": "[bold yellow]>>> Ejecutando simulacion en vivo...[/bold yellow]",
        "theory": "teoria",
        "simulated": "simulado",
        "clouds_bonding": "Nube enlazante (qubit 0)",
        "clouds_antibonding": "Nube antienlazante (qubit 1)",
        "clouds_note": (
            "Dos protones estan sobre el eje horizontal, a 0.74 A.\n"
            "[cyan]Enlazante[/cyan]: la nube electronica se acumula ENTRE los nucleos y\n"
            "pega la molecula. En la nube [bold]antienlazante[/bold] la funcion de onda\n"
            "cambia de signo ([cyan]cian[/cyan] -> [magenta]magenta[/magenta]) a mitad de camino: ese hueco\n"
            "vertical vacio es un nodo, y un electron que viva ahi separa los\n"
            "atomos en vez de unirlos."
        ),
        "vqe_ansatz_title": "El circuito ansatz (una perilla: theta)",
        "vqe_landscape_title": "Paisaje de energia E(theta) - cada punto es un circuito en vivo",
        "vqe_live_title": "VQE en vivo - descenso por gradiente sobre el paisaje",
        "vqe_corr_label": "correlacion capturada",
        "vqe_final": ("Convergio en {it} iteraciones:  E = {e:+.6f} Ha   "
                      "(FCI {fci:+.6f} Ha, error {err:.1e} Ha)"),
    },
}

GLOSSARY: Dict[str, List[Tuple[str, str]]] = {
    "en": [
        ("Qubit", "The quantum bit. Unlike a classical bit it can be in a superposition a|0> + b|1>."),
        ("Superposition", "A state that is a weighted combination of basis states. Weights are complex amplitudes."),
        ("Amplitude", "Complex number attached to each basis state. Its squared magnitude is the probability."),
        ("Measurement", "Asking the qubit '0 or 1?'. The state collapses; outcomes follow the probabilities."),
        ("Entanglement", "Correlation with no classical equivalent: measuring one qubit instantly constrains the other."),
        ("Bell state", "The simplest maximally entangled 2-qubit state: (|00> + |11>)/sqrt(2)."),
        ("Gate", "A reversible operation on qubits, represented by a unitary matrix (H, X, CNOT, ...)."),
        ("Entropy (entanglement)", "Measures how entangled two halves of a system are. 0 = none, 1 bit = Bell pair."),
        ("Grover's algorithm", "Quantum search: finds a marked item among N with ~sqrt(N) steps instead of ~N."),
        ("Orbital", "Region of space where an electron in an atom is likely to be found; solution of the Schrodinger equation."),
        ("Hartree-Fock (HF)", "Mean-field approximation: each electron feels the average of the others."),
        ("FCI", "Full Configuration Interaction: the exact answer within a basis set. Exponentially expensive."),
        ("Correlation energy", "E_FCI - E_HF. The part that requires treating electrons collectively."),
        ("VQE", "Variational Quantum Eigensolver: a hybrid algorithm that finds molecular ground-state energies."),
        ("Jordan-Wigner", "Recipe that maps electrons in orbitals onto qubits (1 spin-orbital = 1 qubit)."),
        ("MPS", "Matrix Product State: compressed representation that lets this simulator reach 33+ qubits."),
    ],
    "es": [
        ("Qubit", "El bit cuantico. A diferencia del bit clasico puede estar en superposicion a|0> + b|1>."),
        ("Superposicion", "Un estado que es combinacion ponderada de estados base. Los pesos son amplitudes complejas."),
        ("Amplitud", "Numero complejo asociado a cada estado base. Su magnitud al cuadrado es la probabilidad."),
        ("Medicion", "Preguntarle al qubit '0 o 1?'. El estado colapsa; los resultados siguen las probabilidades."),
        ("Entrelazamiento", "Correlacion sin equivalente clasico: medir un qubit restringe al instante el otro."),
        ("Estado de Bell", "El estado maximamente entrelazado mas simple de 2 qubits: (|00> + |11>)/sqrt(2)."),
        ("Compuerta", "Operacion reversible sobre qubits, representada por una matriz unitaria (H, X, CNOT, ...)."),
        ("Entropia (entrelazamiento)", "Mide cuan entrelazadas estan dos mitades de un sistema. 0 = nada, 1 bit = par de Bell."),
        ("Algoritmo de Grover", "Busqueda cuantica: encuentra un item marcado entre N en ~sqrt(N) pasos en vez de ~N."),
        ("Orbital", "Region del espacio donde es probable encontrar un electron; solucion de la ecuacion de Schrodinger."),
        ("Hartree-Fock (HF)", "Aproximacion de campo medio: cada electron siente el promedio de los demas."),
        ("FCI", "Full Configuration Interaction: la respuesta exacta dentro de una base. Costo exponencial."),
        ("Energia de correlacion", "E_FCI - E_HF. La parte que exige tratar a los electrones colectivamente."),
        ("VQE", "Variational Quantum Eigensolver: algoritmo hibrido que halla energias de estado base moleculares."),
        ("Jordan-Wigner", "Receta que mapea electrones en orbitales a qubits (1 espin-orbital = 1 qubit)."),
        ("MPS", "Matrix Product State: representacion comprimida que permite a este simulador llegar a 33+ qubits."),
    ],
}


# =============================================================================
# Rendering helpers
# =============================================================================

def probability_bars(probs: np.ndarray, n_qubits: int, lang: str,
                     max_rows: int = 16) -> Table:
    """Build a table of probability bars for a state's distribution."""
    t = TEXT[lang]
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column(t["basis"], justify="right", style="bold cyan")
    table.add_column("", justify="left")
    table.add_column(t["probability"], justify="right")

    order = np.argsort(probs)[::-1]
    shown = [i for i in order if probs[i] > 1e-6][:max_rows]
    if not shown:
        shown = list(order[:max_rows])
    shown.sort()

    for i in shown:
        p = float(probs[i])
        filled = int(round(p * BAR_WIDTH))
        bar = Text("█" * filled, style="green")
        bar.append("░" * (BAR_WIDTH - filled), style="grey30")
        table.add_row(f"|{format(i, f'0{n_qubits}b')}>", bar, f"{p:6.1%}")

    hidden = int((probs > 1e-6).sum()) - len(shown)
    if hidden > 0:
        table.add_row("...", Text(f"(+{hidden})", style="dim"), "")
    return table


def counts_bars(counts: Dict[str, int], total: int, lang: str) -> Table:
    """Build a table of measurement-count bars."""
    t = TEXT[lang]
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column(t["basis"], justify="right", style="bold cyan")
    table.add_column("", justify="left")
    table.add_column(t["counts"], justify="right")
    for key in sorted(counts):
        c = counts[key]
        filled = int(round(c / total * BAR_WIDTH))
        bar = Text("█" * filled, style="yellow")
        bar.append("░" * (BAR_WIDTH - filled), style="grey30")
        table.add_row(f"|{key}>", bar, f"{c}")
    return table


def draw_circuit(n_qubits: int, instructions: List[Tuple[str, List[int]]]) -> Text:
    """Render an ASCII timeline of the circuit, one line per qubit."""
    lanes = [[f"q{q}: ─"] for q in range(n_qubits)]
    for name, targets in instructions:
        width = max(len(name) + 2, 5)
        for q in range(n_qubits):
            if len(targets) == 1 and q == targets[0]:
                cell = f"[{name}]".center(width, "─")
            elif len(targets) == 2 and q == targets[0]:
                symbol = "●" if name in ("CNOT", "CZ", "CRZ") else "x" if name == "SWAP" else "●"
                cell = symbol.center(width, "─")
            elif len(targets) == 2 and q == targets[1]:
                symbol = {"CNOT": "⊕", "CZ": "●", "SWAP": "x"}.get(name, "□")
                cell = symbol.center(width, "─")
            elif len(targets) == 2 and min(targets) < q < max(targets):
                cell = "┼".center(width, "─")
            else:
                cell = "─" * width
            lanes[q].append(cell)
    text = Text()
    for q in range(n_qubits):
        text.append("".join(lanes[q]) + "─\n", style="white")
    return text


def parse_angle(token: str) -> float:
    """Parse an angle like '1.57', 'pi', '-pi/2' or '3*pi/4' into radians."""
    expr = token.strip().lower().replace(" ", "")
    sign = 1.0
    if expr.startswith("-"):
        sign = -1.0
        expr = expr[1:]
    numerator, _, denominator = expr.partition("/")
    den = float(denominator) if denominator else 1.0
    if "pi" in numerator:
        factor_str = numerator.replace("*", "").replace("pi", "")
        factor = float(factor_str) if factor_str else 1.0
        num = factor * math.pi
    else:
        num = float(numerator)
    if den == 0:
        raise ValueError("division by zero in angle")
    return sign * num / den


def sample_measurements(probs: np.ndarray, n_qubits: int, n_samples: int,
                        rng: np.random.Generator) -> Dict[str, int]:
    """Draw measurement outcomes from a probability distribution."""
    p = probs / probs.sum()
    outcomes = rng.choice(len(p), size=n_samples, p=p)
    counts: Dict[str, int] = {}
    for o in outcomes:
        key = format(int(o), f"0{n_qubits}b")
        counts[key] = counts.get(key, 0) + 1
    return counts


# =============================================================================
# Hydrogen orbital ASCII renderer
# =============================================================================

@dataclass
class OrbitalSpec:
    """Definition of a real hydrogen orbital for the ASCII viewer."""
    name: str
    n: int
    l: int
    angular: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    plane: str = "xz"


def _radial(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """Hydrogen radial wavefunction R_nl(r) in atomic units."""
    from scipy.special import genlaguerre
    from math import factorial
    rho = 2.0 * r / n
    norm = math.sqrt((2.0 / n) ** 3 * factorial(n - l - 1) / (2.0 * n * factorial(n + l)))
    return norm * np.exp(-rho / 2.0) * rho ** l * genlaguerre(n - l - 1, 2 * l + 1)(rho)


_SQ = math.sqrt


def _ang_s(x, y, z, r):
    return np.full_like(r, 0.5 * _SQ(1.0 / math.pi))


def _ang_pz(x, y, z, r):
    return _SQ(3.0 / (4.0 * math.pi)) * np.divide(z, r, out=np.zeros_like(r), where=r > 0)


def _ang_px(x, y, z, r):
    return _SQ(3.0 / (4.0 * math.pi)) * np.divide(x, r, out=np.zeros_like(r), where=r > 0)


def _ang_dz2(x, y, z, r):
    cos2 = np.divide(z * z, r * r, out=np.zeros_like(r), where=r > 0)
    return _SQ(5.0 / (16.0 * math.pi)) * (3.0 * cos2 - 1.0)


def _ang_dxz(x, y, z, r):
    return _SQ(15.0 / (4.0 * math.pi)) * np.divide(x * z, r * r, out=np.zeros_like(r), where=r > 0)


def _ang_dxy(x, y, z, r):
    return _SQ(15.0 / (4.0 * math.pi)) * np.divide(x * y, r * r, out=np.zeros_like(r), where=r > 0)


ORBITAL_SPECS: List[OrbitalSpec] = [
    OrbitalSpec("1s", 1, 0, _ang_s),
    OrbitalSpec("2s", 2, 0, _ang_s),
    OrbitalSpec("2p_z", 2, 1, _ang_pz),
    OrbitalSpec("2p_x", 2, 1, _ang_px),
    OrbitalSpec("3s", 3, 0, _ang_s),
    OrbitalSpec("3p_z", 3, 1, _ang_pz),
    OrbitalSpec("3d_z2", 3, 2, _ang_dz2),
    OrbitalSpec("3d_xz", 3, 2, _ang_dxz),
    OrbitalSpec("3d_xy", 3, 2, _ang_dxy, plane="xy"),
]


def field_to_text(psi: np.ndarray) -> Text:
    """Render a real scalar field as colored ASCII: brightness = |psi|^2, color = sign."""
    rows, cols = psi.shape
    density = psi * psi
    peak = density.max()
    if peak <= 0:
        peak = 1.0
    intensity = np.sqrt(density / peak)

    text = Text()
    levels = len(DENSITY_RAMP) - 1
    for i in range(rows):
        for j in range(cols):
            level = int(round(intensity[i, j] * levels))
            ch = DENSITY_RAMP[level]
            if level == 0:
                text.append(" ")
            else:
                style = "cyan" if psi[i, j] >= 0 else "magenta"
                if level >= levels - 2:
                    style = "bold bright_" + style
                text.append(ch, style=style)
        text.append("\n")
    return text


def render_orbital(spec: OrbitalSpec, rows: int = 27, cols: int = 64) -> Tuple[Text, float]:
    """Render |psi|^2 of a hydrogen orbital on a plane slice as colored ASCII."""
    extent = 1.8 * spec.n * spec.n + 3.0
    horizontal = np.linspace(-extent, extent, cols)
    vertical = np.linspace(extent, -extent, rows)
    hh, vv = np.meshgrid(horizontal, vertical)

    if spec.plane == "xz":
        x, y, z = hh, np.zeros_like(hh), vv
    else:
        x, y, z = hh, vv, np.zeros_like(hh)

    r = np.sqrt(x * x + y * y + z * z)
    psi = _radial(spec.n, spec.l, r) * spec.angular(x, y, z, r)
    return field_to_text(psi), extent


H2_BOND_BOHR = 1.4


def render_h2_molecular_orbital(kind: str, rows: int = 19, cols: int = 38) -> Text:
    """Render the bonding or antibonding LCAO molecular orbital of H2."""
    extent = 5.0
    horizontal = np.linspace(-extent, extent, cols)
    vertical = np.linspace(extent, -extent, rows)
    zz, xx = np.meshgrid(horizontal, vertical)
    half = H2_BOND_BOHR / 2.0
    r1 = np.sqrt(xx * xx + (zz - half) ** 2)
    r2 = np.sqrt(xx * xx + (zz + half) ** 2)
    if kind == "bonding":
        psi = np.exp(-r1) + np.exp(-r2)
    else:
        psi = np.exp(-r1) - np.exp(-r2)
    return field_to_text(psi)


# =============================================================================
# VQE energy landscape plot
# =============================================================================

def landscape_plot(energies: List[float], e_hf: float, e_fci: float,
                   marker: Optional[int] = None, rows: int = 14) -> Text:
    """
    Draw an ASCII plot of E(theta) over one full period with HF and FCI
    reference lines and an optional optimizer marker.
    """
    cols = len(energies)
    e_top = max(max(energies), e_hf) + 0.003
    e_bot = min(min(energies), e_fci) - 0.003
    span = e_top - e_bot

    def row_of(e: float) -> int:
        frac = (e - e_bot) / span
        return min(rows - 1, max(0, int(round((1.0 - frac) * (rows - 1)))))

    grid = [[(" ", None) for _ in range(cols)] for _ in range(rows)]
    hf_row, fci_row = row_of(e_hf), row_of(e_fci)
    for j in range(0, cols, 2):
        grid[hf_row][j] = ("╌", "yellow")
        grid[fci_row][j] = ("╌", "green")
    for j, e in enumerate(energies):
        grid[row_of(e)][j] = ("·", "bold cyan")
    if marker is not None:
        j = min(cols - 1, max(0, marker))
        grid[row_of(energies[j])][j] = ("◆", "bold red")

    label_width = 11
    text = Text()
    for i in range(rows):
        if i == hf_row:
            text.append(f"{e_hf:>9.4f} ─", style="yellow")
        elif i == fci_row:
            text.append(f"{e_fci:>9.4f} ─", style="green")
        else:
            text.append(" " * (label_width - 1) + "│", style="dim")
        for ch, style in grid[i]:
            text.append(ch, style=style)
        if i == hf_row:
            text.append("  HF", style="yellow")
        elif i == fci_row:
            text.append("  FCI", style="green")
        text.append("\n")
    text.append(" " * (label_width - 1) + "└" + "─" * cols + "\n", style="dim")
    axis = " " * label_width + "0" + "π".center(cols // 2 - 1) + "  " + "2π".rjust(cols // 2 - 2)
    text.append(axis + "\n", style="dim")
    return text


class H2VQEEngine:
    """
    Minimal live VQE for H2 in the 2-qubit active space.

    The trial state cos(theta/2)|01> + sin(theta/2)|10> is prepared with a
    real circuit (Ry, CNOT, X) on the MPS engine, and the energy is read
    from the Jordan-Wigner H2 Hamiltonian of quantum_framework_molecular.
    """

    def __init__(self, qc: MPSQuantumComputer) -> None:
        from quantum_framework_molecular import MoleculeBuilder, ExactJWEnergy
        self.qc = qc
        self.mol = MoleculeBuilder.h2_sto3g()
        self.evaluator = ExactJWEnergy(self.mol, self.mol.n_qubits)
        self.e_hf = self.mol.hf_energy
        self.e_fci = self.mol.fci_energy

    def ansatz_instructions(self, theta: float) -> List[Tuple[str, List[int]]]:
        return [("RY", [0]), ("CNOT", [0, 1]), ("X", [1])]

    def energy(self, theta: float) -> float:
        state = self.qc.create_state(2)
        circuit = self.qc.create_circuit(2)
        circuit.ry(0, theta)
        circuit.cnot(0, 1)
        circuit.x(1)
        state = circuit.run(state)
        return float(self.evaluator.evaluate(state.to_statevector()))

    def correlation_pct(self, e: float) -> float:
        return (self.e_hf - e) / (self.e_hf - self.e_fci) * 100.0

    def landscape(self, cols: int = 58) -> Tuple[List[float], List[float]]:
        thetas = [2.0 * math.pi * j / (cols - 1) for j in range(cols)]
        return thetas, [self.energy(t) for t in thetas]

    def optimize(self, theta0: float = math.pi, lr: float = 25.0,
                 max_iters: int = 40, tol: float = 1e-7):
        """Gradient descent; yields (iteration, theta, energy) live."""
        theta = theta0
        eps = 1e-4
        for it in range(1, max_iters + 1):
            e = self.energy(theta)
            yield it, theta % (2.0 * math.pi), e
            grad = (self.energy(theta + eps) - self.energy(theta - eps)) / (2.0 * eps)
            if abs(grad) < tol:
                return
            theta -= lr * grad


# =============================================================================
# Lesson engine
# =============================================================================

@dataclass
class Quiz:
    question: str
    options: List[str]
    correct: int
    explanation: str


@dataclass
class Lesson:
    title: str
    steps: List[object] = field(default_factory=list)


class QuantumLab:
    """Interactive educational TUI driven by the real Q2C engine."""

    def __init__(self, config: FrameworkConfig, loader: ConfigLoader,
                 lang: str = "en") -> None:
        self.config = config
        self.loader = loader
        self.lang = lang
        self.qc = MPSQuantumComputer(config)
        self.rng = np.random.default_rng(config.random_seed if hasattr(config, "random_seed") else 42)

    def t(self, key: str) -> str:
        return TEXT[self.lang][key]

    # ------------------------------------------------------------------
    # Small building blocks
    # ------------------------------------------------------------------

    def pause(self) -> None:
        console.print()
        try:
            Prompt.ask(self.t("press_enter"), default="", show_default=False)
        except (EOFError, KeyboardInterrupt):
            pass

    def panel(self, body: str, title: str = "", style: str = "cyan") -> None:
        console.print(Panel(body, title=f"[bold]{title}[/bold]" if title else None,
                            border_style=style, padding=(1, 2)))

    def show_state(self, probs: np.ndarray, n_qubits: int,
                   title: Optional[str] = None) -> None:
        console.print(Panel(probability_bars(probs, n_qubits, self.lang),
                            title=title or self.t("state_panel"),
                            border_style="green"))

    def run_quiz(self, quiz: Quiz) -> None:
        console.print(Rule(f"[bold yellow]{self.t('quiz_title')}[/bold yellow]"))
        console.print(f"\n[bold]{quiz.question}[/bold]\n")
        for i, opt in enumerate(quiz.options, start=1):
            console.print(f"  [cyan]{i}[/cyan]. {opt}")
        choices = [str(i) for i in range(1, len(quiz.options) + 1)]
        try:
            answer = Prompt.ask(f"\n{self.t('your_answer')}", choices=choices, default="1")
        except (EOFError, KeyboardInterrupt):
            return
        if int(answer) - 1 == quiz.correct:
            console.print(f"\n[bold green]{self.t('quiz_correct')}[/bold green] {quiz.explanation}\n")
        else:
            correct_text = quiz.options[quiz.correct]
            console.print(f"\n[bold red]{self.t('quiz_wrong')}[/bold red] "
                          f"[bold]{quiz.correct + 1}. {correct_text}[/bold]\n{quiz.explanation}\n")

    # ------------------------------------------------------------------
    # Lessons
    # ------------------------------------------------------------------

    def lessons(self) -> List[Lesson]:
        if self.lang == "es":
            return self._lessons_es()
        return self._lessons_en()

    def _demo_superposition(self) -> None:
        console.print(self.t("run_demo"))
        state = self.qc.create_state(1)
        probs = state.probabilities().numpy()
        self.show_state(probs, 1, "|0>")
        circuit = self.qc.create_circuit(1)
        circuit.h(0)
        state = circuit.run(state)
        probs = state.probabilities().numpy()
        self.show_state(probs, 1, "H|0>")

    def _demo_rotation(self) -> None:
        console.print(self.t("run_demo"))
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("theta", justify="right")
        table.add_column("P(|0>)", justify="left")
        table.add_column("P(|1>)", justify="left")
        for frac in range(0, 9):
            theta = frac * math.pi / 8
            state = self.qc.create_state(1)
            circuit = self.qc.create_circuit(1)
            circuit.ry(0, theta)
            state = circuit.run(state)
            p = state.probabilities().numpy()
            w0 = int(round(p[0] * 24))
            w1 = int(round(p[1] * 24))
            table.add_row(
                f"{frac}pi/8",
                Text("█" * w0 + "░" * (24 - w0), style="green"),
                Text("█" * w1 + "░" * (24 - w1), style="blue"),
            )
        console.print(Panel(table, title="Ry(theta)|0>", border_style="green"))

    def _demo_measurement(self) -> None:
        console.print(self.t("run_demo"))
        state = self.qc.create_state(1)
        circuit = self.qc.create_circuit(1)
        circuit.h(0)
        state = circuit.run(state)
        probs = state.probabilities().numpy()
        for n in (10, 100, 10000):
            counts = sample_measurements(probs, 1, n, self.rng)
            console.print(Panel(counts_bars(counts, n, self.lang),
                                title=f"{self.t('samples_title')}: {n}",
                                border_style="yellow"))

    def _demo_bell(self) -> None:
        console.print(self.t("run_demo"))
        state = self.qc.bell_state(2)
        probs = state.probabilities().numpy()
        self.show_state(probs, 2, "(|00> + |11>)/sqrt(2)")
        entropy = state.entropy()
        console.print(f"  {self.t('entropy_label')}: [bold magenta]{entropy:.4f}[/bold magenta] "
                      f"(max = 1.0)\n")
        counts = sample_measurements(probs, 2, 1000, self.rng)
        console.print(Panel(counts_bars(counts, 1000, self.lang),
                            title=f"{self.t('samples_title')}: 1000",
                            border_style="yellow"))

    def _demo_ghz_w(self) -> None:
        console.print(self.t("run_demo"))
        ghz = self.qc.ghz_state(3)
        self.show_state(ghz.probabilities().numpy(), 3, "GHZ(3)")
        w = self.qc.w_state(3)
        self.show_state(w.probabilities().numpy(), 3, "W(3)")

    def _demo_grover(self) -> None:
        console.print(self.t("run_demo"))
        n_qubits = 4
        marked = 11
        result = run_grover_search(self.qc, n_qubits, [marked])
        table = Table(show_header=False, box=None)
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")
        table.add_row("N", f"2^{n_qubits} = {2 ** n_qubits}")
        table.add_row("marked", f"|{format(marked, f'0{n_qubits}b')}>")
        table.add_row("iterations", str(result["iterations"]))
        table.add_row("P(success)", f"[bold green]{result['probability']:.2%}[/bold green]")
        table.add_row("classical", f"{1 / 2 ** n_qubits:.2%}")
        table.add_row("speedup", f"{result['speedup']:.1f}x")
        console.print(Panel(table, title="Grover", border_style="green"))

    def _demo_molecule(self) -> None:
        console.print(self.t("run_demo"))
        mol = self.loader.get_molecule("H2")
        if mol is not None:
            self._molecule_card(mol)

    def _vqe(self) -> H2VQEEngine:
        if not hasattr(self, "_vqe_engine"):
            self._vqe_engine = H2VQEEngine(self.qc)
        return self._vqe_engine

    def _demo_h2_clouds(self) -> None:
        console.print(self.t("run_demo"))
        bonding = Panel(render_h2_molecular_orbital("bonding"),
                        title=self.t("clouds_bonding"), border_style="green")
        antibonding = Panel(render_h2_molecular_orbital("antibonding"),
                            title=self.t("clouds_antibonding"), border_style="red")
        console.print(Columns([bonding, antibonding]))
        console.print(Panel(self.t("clouds_note"), border_style="dim"))

    def _demo_vqe_ansatz(self) -> None:
        drawn = [("RY(θ)", [0]), ("CNOT", [0, 1]), ("X", [1])]
        console.print(Panel(draw_circuit(2, drawn),
                            title=self.t("vqe_ansatz_title"), border_style="blue"))

    def _demo_vqe_landscape(self) -> None:
        console.print(self.t("run_demo"))
        engine = self._vqe()
        _, energies = engine.landscape()
        console.print(Panel(landscape_plot(energies, engine.e_hf, engine.e_fci),
                            title=self.t("vqe_landscape_title"), border_style="blue"))

    def _demo_vqe_live(self) -> None:
        console.print(self.t("run_demo"))
        engine = self._vqe()
        _, energies = engine.landscape()
        cols = len(energies)
        final_it, final_e = 0, engine.e_hf
        with Live(console=console, refresh_per_second=12) as live:
            for it, theta, e in engine.optimize():
                marker = int(round(theta / (2.0 * math.pi) * (cols - 1))) % cols
                pct = max(0.0, min(100.0, engine.correlation_pct(e)))
                filled = int(round(pct / 100.0 * BAR_WIDTH))
                gauge = Text("█" * filled, style="green")
                gauge.append("░" * (BAR_WIDTH - filled), style="grey30")
                footer = Text.assemble(
                    (f" iter {it:>2}   θ = {theta:4.2f} rad   E = {e:+.6f} Ha\n", "bold"),
                    (f" {self.t('vqe_corr_label')}: ", ""),
                )
                footer.append_text(gauge)
                footer.append(f" {pct:5.1f}%", style="bold green")
                plot = landscape_plot(energies, engine.e_hf, engine.e_fci, marker=marker)
                live.update(Panel(Group(plot, footer),
                                  title=self.t("vqe_live_title"), border_style="blue"))
                final_it, final_e = it, e
                time.sleep(0.15)
        err = abs(final_e - engine.e_fci)
        console.print("\n[bold green]✓[/bold green] " + self.t("vqe_final").format(
            it=final_it, e=final_e, fci=engine.e_fci, err=err))

    def _lessons_en(self) -> List[Lesson]:
        return [
            Lesson("1. The qubit and superposition", [
                ("What is a qubit?",
                 "A classical bit is either [bold]0[/bold] or [bold]1[/bold].\n"
                 "A qubit can be [bold]both at once[/bold]:\n\n"
                 "    |psi> = a|0> + b|1>\n\n"
                 "where a and b are complex numbers called [bold]amplitudes[/bold],\n"
                 "with |a|^2 + |b|^2 = 1. When you measure, you get 0 with\n"
                 "probability |a|^2 and 1 with probability |b|^2."),
                ("The Hadamard gate",
                 "The most famous gate is [bold]H[/bold] (Hadamard). Applied to |0> it\n"
                 "creates an equal superposition: H|0> = (|0> + |1>)/sqrt(2).\n"
                 "Watch the real simulator do it:"),
                self._demo_superposition,
                ("Continuous control",
                 "Superposition is not all-or-nothing. The rotation gate Ry(theta)\n"
                 "lets you dial any mixture between |0> and |1>:"),
                self._demo_rotation,
                Quiz("If a qubit is in state (|0> + |1>)/sqrt(2), what is the probability of measuring 1?",
                     ["0%", "25%", "50%", "100%"], 2,
                     "The amplitude of |1> is 1/sqrt(2), so P = (1/sqrt(2))^2 = 1/2."),
            ]),
            Lesson("2. Measurement and randomness", [
                ("Collapse",
                 "Before measurement a qubit holds amplitudes; after measurement\n"
                 "it is definitely 0 or 1. Quantum randomness is [bold]fundamental[/bold]:\n"
                 "it is not lack of knowledge, it is how nature works.\n\n"
                 "Let's measure H|0> many times and watch the statistics converge\n"
                 "to 50/50 - few samples are noisy, many samples are exact."),
                self._demo_measurement,
                Quiz("Why do 10 measurements rarely give exactly 5 zeros and 5 ones?",
                     ["The simulator is broken",
                      "Each measurement is an independent random event; small samples fluctuate",
                      "The qubit remembers previous results",
                      "Hadamard is imprecise"], 1,
                     "Same reason 10 coin flips rarely give exactly 5 heads. The law of large numbers needs large numbers."),
            ]),
            Lesson("3. Entanglement and Bell states", [
                ("Two qubits, one destiny",
                 "Apply H to qubit 0, then CNOT(0,1). The result is the Bell state:\n\n"
                 "    (|00> + |11>)/sqrt(2)\n\n"
                 "Notice what is missing: there is [bold]no[/bold] |01> and no |10>.\n"
                 "If you measure qubit 0 and get 0, qubit 1 is instantly 0 too -\n"
                 "even if it is on the other side of the galaxy."),
                self._demo_bell,
                ("Entropy as a ruler",
                 "Entanglement entropy measures the connection between the two\n"
                 "halves: 0.0 means independent, 1.0 bit means maximally\n"
                 "entangled. The Bell state hits exactly 1.0 - the simulator\n"
                 "computed that from the actual MPS tensors."),
                ("Bigger families: GHZ and W",
                 "With 3+ qubits there are different [bold]kinds[/bold] of entanglement.\n"
                 "GHZ = (|000> + |111>)/sqrt(2): all or nothing.\n"
                 "W = (|001> + |010> + |100>)/sqrt(3): exactly one excitation, shared."),
                self._demo_ghz_w,
                Quiz("In the Bell state (|00> + |11>)/sqrt(2), you measure qubit 0 and get 1. What is qubit 1?",
                     ["50/50 random", "Definitely 0", "Definitely 1", "Unknowable"], 2,
                     "Only |00> and |11> exist in the superposition, so the outcomes are perfectly correlated."),
            ]),
            Lesson("4. Grover's quantum search", [
                ("Finding a needle",
                 "Classically, finding 1 marked item among N unsorted items takes\n"
                 "on average N/2 looks. Grover's algorithm uses interference to\n"
                 "amplify the amplitude of the marked item, and needs only about\n"
                 "sqrt(N) steps.\n\n"
                 "With 4 qubits, N = 16: classical guessing succeeds 6.25% of the\n"
                 "time on the first try. Watch Grover:"),
                self._demo_grover,
                Quiz("For N = 1,000,000 items, roughly how many Grover iterations are needed?",
                     ["1,000,000", "500,000", "~1,000", "10"], 2,
                     "sqrt(1,000,000) = 1,000. That quadratic speedup is the whole magic."),
            ]),
            Lesson("5. Quantum chemistry: molecules as qubits", [
                ("Why chemistry is hard",
                 "Electrons repel each other and move in a correlated way. The\n"
                 "exact wavefunction of N electrons lives in a space that grows\n"
                 "[bold]exponentially[/bold] with N - classical computers choke on it.\n\n"
                 "Quantum computers represent that space natively. The\n"
                 "[bold]Jordan-Wigner[/bold] mapping assigns one qubit per spin-orbital:\n"
                 "qubit = 1 means 'electron here', qubit = 0 means 'empty'."),
                ("The energy ladder",
                 "Hartree-Fock (HF) treats each electron in the average field of\n"
                 "the others - good but not exact. The difference between HF and\n"
                 "the exact (FCI) energy is the [bold]correlation energy[/bold]. VQE uses\n"
                 "a quantum circuit as a flexible guess and tunes it until the\n"
                 "energy is minimal. For H2 this framework recovers [bold]100%[/bold] of\n"
                 "the correlation energy:"),
                self._demo_molecule,
                Quiz("In the Jordan-Wigner mapping, what does one qubit represent?",
                     ["One atom", "One molecule", "One spin-orbital (occupied or empty)", "One electron pair"], 2,
                     "Each spin-orbital becomes a qubit: |1> = occupied, |0> = empty. H2 in a minimal basis needs 4."),
            ]),
            Lesson("6. VQE live: hunting the ground state of H2", [
                ("The variational principle",
                 "Nature is lazy: a molecule settles into its state of [bold]lowest\n"
                 "energy[/bold]. Quantum mechanics adds a guarantee: ANY trial\n"
                 "wavefunction you can dream up has an energy [bold]>=[/bold] the true\n"
                 "ground-state energy. You can approach the floor, never cross it.\n\n"
                 "The [bold]VQE[/bold] recipe follows directly: prepare a trial state with\n"
                 "a quantum circuit that has tunable knobs, measure its energy,\n"
                 "and let a classical optimizer turn the knobs until the energy\n"
                 "stops dropping. Where it stops is (approximately) the molecule."),
                ("The electron clouds of H2",
                 "When two hydrogen atoms meet, their 1s clouds merge into two\n"
                 "[bold]molecular orbitals[/bold]: the bonding combination (1s + 1s) piles\n"
                 "electron density between the protons and glues them together;\n"
                 "the antibonding one (1s - 1s) has a node between them.\n\n"
                 "Our 2-qubit model uses exactly these clouds:\n"
                 "qubit 0 = bonding, qubit 1 = antibonding. The Hartree-Fock\n"
                 "state is |10>: bonding occupied, antibonding empty."),
                self._demo_h2_clouds,
                ("A circuit with one knob",
                 "The trial state is  cos(θ/2)|01> + sin(θ/2)|10> :  a single\n"
                 "continuous knob θ mixes 'electron in bonding' with 'electron\n"
                 "in antibonding'. At θ = π it is exactly the Hartree-Fock state.\n"
                 "Ry(θ) creates the mixture, CNOT entangles, X flips:"),
                self._demo_vqe_ansatz,
                ("The energy landscape",
                 "Sweep θ around a full turn, run the circuit for each value on\n"
                 "the MPS engine, and ask the real H2 Hamiltonian for the energy.\n"
                 "The [yellow]yellow dashed line[/yellow] is Hartree-Fock; the [green]green one[/green] is the\n"
                 "exact (FCI) answer. The valley dipping below HF [bold]is[/bold] the\n"
                 "correlation energy:"),
                self._demo_vqe_landscape,
                ("Watch the optimizer descend",
                 "Now the real thing. Gradient descent starts at the Hartree-Fock\n"
                 "state (θ = π) and follows the slope downhill. The [red]red diamond[/red]\n"
                 "is the optimizer walking on the landscape; the bar fills up as\n"
                 "correlation energy is captured:"),
                self._demo_vqe_live,
                Quiz("Why can the VQE energy never drop below the FCI (exact) value?",
                     ["The optimizer is too slow",
                      "The variational principle: any trial state has E >= the true ground energy",
                      "Rounding errors prevent it",
                      "Because the circuit has only one parameter"], 1,
                     "That is the safety net of VQE: the exact ground state is a hard floor. The whole game is approaching it from above."),
            ]),
        ]

    def _lessons_es(self) -> List[Lesson]:
        return [
            Lesson("1. El qubit y la superposicion", [
                ("Que es un qubit?",
                 "Un bit clasico es [bold]0[/bold] o [bold]1[/bold].\n"
                 "Un qubit puede ser [bold]ambos a la vez[/bold]:\n\n"
                 "    |psi> = a|0> + b|1>\n\n"
                 "donde a y b son numeros complejos llamados [bold]amplitudes[/bold],\n"
                 "con |a|^2 + |b|^2 = 1. Al medir, obtienes 0 con probabilidad\n"
                 "|a|^2 y 1 con probabilidad |b|^2."),
                ("La compuerta Hadamard",
                 "La compuerta mas famosa es [bold]H[/bold] (Hadamard). Aplicada a |0>\n"
                 "crea una superposicion igualitaria: H|0> = (|0> + |1>)/sqrt(2).\n"
                 "Mira al simulador real hacerlo:"),
                self._demo_superposition,
                ("Control continuo",
                 "La superposicion no es todo-o-nada. La rotacion Ry(theta)\n"
                 "permite elegir cualquier mezcla entre |0> y |1>:"),
                self._demo_rotation,
                Quiz("Si un qubit esta en (|0> + |1>)/sqrt(2), cual es la probabilidad de medir 1?",
                     ["0%", "25%", "50%", "100%"], 2,
                     "La amplitud de |1> es 1/sqrt(2), asi que P = (1/sqrt(2))^2 = 1/2."),
            ]),
            Lesson("2. Medicion y azar", [
                ("Colapso",
                 "Antes de medir, el qubit guarda amplitudes; despues de medir,\n"
                 "es definitivamente 0 o 1. El azar cuantico es [bold]fundamental[/bold]:\n"
                 "no es falta de informacion, es como funciona la naturaleza.\n\n"
                 "Midamos H|0> muchas veces y veamos las estadisticas converger\n"
                 "a 50/50: pocas muestras son ruidosas, muchas son exactas."),
                self._demo_measurement,
                Quiz("Por que 10 mediciones rara vez dan exactamente 5 ceros y 5 unos?",
                     ["El simulador esta roto",
                      "Cada medicion es un evento aleatorio independiente; las muestras chicas fluctuan",
                      "El qubit recuerda resultados anteriores",
                      "Hadamard es imprecisa"], 1,
                     "Por lo mismo que 10 monedas rara vez dan exactamente 5 caras. La ley de los grandes numeros necesita numeros grandes."),
            ]),
            Lesson("3. Entrelazamiento y estados de Bell", [
                ("Dos qubits, un destino",
                 "Aplica H al qubit 0 y luego CNOT(0,1). El resultado es el\n"
                 "estado de Bell:\n\n"
                 "    (|00> + |11>)/sqrt(2)\n\n"
                 "Fijate en lo que falta: [bold]no[/bold] hay |01> ni |10>.\n"
                 "Si mides el qubit 0 y sale 0, el qubit 1 es instantaneamente 0,\n"
                 "aunque este del otro lado de la galaxia."),
                self._demo_bell,
                ("La entropia como regla",
                 "La entropia de entrelazamiento mide la conexion entre las dos\n"
                 "mitades: 0.0 = independientes, 1.0 bit = maximamente\n"
                 "entrelazadas. El estado de Bell da exactamente 1.0, calculado\n"
                 "desde los tensores MPS reales del simulador."),
                ("Familias mas grandes: GHZ y W",
                 "Con 3+ qubits hay distintos [bold]tipos[/bold] de entrelazamiento.\n"
                 "GHZ = (|000> + |111>)/sqrt(2): todo o nada.\n"
                 "W = (|001> + |010> + |100>)/sqrt(3): una sola excitacion, compartida."),
                self._demo_ghz_w,
                Quiz("En el estado de Bell (|00> + |11>)/sqrt(2), mides el qubit 0 y sale 1. Que vale el qubit 1?",
                     ["50/50 aleatorio", "Definitivamente 0", "Definitivamente 1", "Imposible saber"], 2,
                     "Solo |00> y |11> existen en la superposicion: los resultados estan perfectamente correlacionados."),
            ]),
            Lesson("4. Busqueda cuantica de Grover", [
                ("Encontrar la aguja",
                 "Clasicamente, hallar 1 item marcado entre N sin ordenar toma en\n"
                 "promedio N/2 intentos. Grover usa interferencia para amplificar\n"
                 "la amplitud del item marcado y necesita solo ~sqrt(N) pasos.\n\n"
                 "Con 4 qubits, N = 16: adivinar a la primera funciona 6.25% de\n"
                 "las veces. Mira a Grover:"),
                self._demo_grover,
                Quiz("Para N = 1,000,000 de items, cuantas iteraciones de Grover hacen falta aprox.?",
                     ["1,000,000", "500,000", "~1,000", "10"], 2,
                     "sqrt(1,000,000) = 1,000. Ese speedup cuadratico es toda la magia."),
            ]),
            Lesson("5. Quimica cuantica: moleculas como qubits", [
                ("Por que la quimica es dificil",
                 "Los electrones se repelen y se mueven de forma correlacionada.\n"
                 "La funcion de onda exacta de N electrones vive en un espacio que\n"
                 "crece [bold]exponencialmente[/bold] con N: las computadoras clasicas se\n"
                 "ahogan.\n\n"
                 "Las computadoras cuanticas representan ese espacio de forma\n"
                 "nativa. El mapeo [bold]Jordan-Wigner[/bold] asigna un qubit por\n"
                 "espin-orbital: qubit = 1 significa 'electron aqui', 0 = 'vacio'."),
                ("La escalera de energias",
                 "Hartree-Fock (HF) trata cada electron en el campo promedio de\n"
                 "los demas: bueno pero no exacto. La diferencia entre HF y la\n"
                 "energia exacta (FCI) es la [bold]energia de correlacion[/bold]. VQE usa\n"
                 "un circuito cuantico como propuesta flexible y lo ajusta hasta\n"
                 "minimizar la energia. Para H2 este framework recupera el\n"
                 "[bold]100%[/bold] de la energia de correlacion:"),
                self._demo_molecule,
                Quiz("En el mapeo Jordan-Wigner, que representa un qubit?",
                     ["Un atomo", "Una molecula", "Un espin-orbital (ocupado o vacio)", "Un par de electrones"], 2,
                     "Cada espin-orbital se vuelve un qubit: |1> = ocupado, |0> = vacio. H2 en base minima necesita 4."),
            ]),
            Lesson("6. VQE en vivo: cazando el estado base de H2", [
                ("El principio variacional",
                 "La naturaleza es perezosa: una molecula se acomoda en su estado\n"
                 "de [bold]minima energia[/bold]. La mecanica cuantica agrega una garantia:\n"
                 "CUALQUIER funcion de onda de prueba que imagines tiene energia\n"
                 "[bold]>=[/bold] la energia real del estado base. Puedes acercarte al piso,\n"
                 "nunca atravesarlo.\n\n"
                 "La receta de [bold]VQE[/bold] sale directo de ahi: prepara un estado de\n"
                 "prueba con un circuito cuantico con perillas ajustables, mide su\n"
                 "energia, y deja que un optimizador clasico gire las perillas\n"
                 "hasta que la energia deje de bajar. Donde se detiene esta\n"
                 "(aproximadamente) la molecula."),
                ("Las nubes de electrones de H2",
                 "Cuando dos atomos de hidrogeno se encuentran, sus nubes 1s se\n"
                 "fusionan en dos [bold]orbitales moleculares[/bold]: la combinacion\n"
                 "enlazante (1s + 1s) acumula densidad electronica entre los\n"
                 "protones y los pega; la antienlazante (1s - 1s) tiene un nodo\n"
                 "entre ellos.\n\n"
                 "Nuestro modelo de 2 qubits usa exactamente estas nubes:\n"
                 "qubit 0 = enlazante, qubit 1 = antienlazante. El estado\n"
                 "Hartree-Fock es |10>: enlazante ocupado, antienlazante vacio."),
                self._demo_h2_clouds,
                ("Un circuito con una perilla",
                 "El estado de prueba es  cos(θ/2)|01> + sin(θ/2)|10> :  una sola\n"
                 "perilla continua θ mezcla 'electron en enlazante' con 'electron\n"
                 "en antienlazante'. En θ = π es exactamente el estado\n"
                 "Hartree-Fock. Ry(θ) crea la mezcla, CNOT entrelaza, X voltea:"),
                self._demo_vqe_ansatz,
                ("El paisaje de energia",
                 "Barre θ en una vuelta completa, corre el circuito para cada\n"
                 "valor en el motor MPS, y preguntale al Hamiltoniano real de H2\n"
                 "por la energia. La [yellow]linea amarilla punteada[/yellow] es Hartree-Fock; la\n"
                 "[green]verde[/green] es la respuesta exacta (FCI). El valle que baja de HF\n"
                 "[bold]es[/bold] la energia de correlacion:"),
                self._demo_vqe_landscape,
                ("Mira al optimizador descender",
                 "Ahora lo real. El descenso por gradiente arranca en el estado\n"
                 "Hartree-Fock (θ = π) y sigue la pendiente cuesta abajo. El\n"
                 "[red]diamante rojo[/red] es el optimizador caminando sobre el paisaje;\n"
                 "la barra se llena a medida que captura energia de correlacion:"),
                self._demo_vqe_live,
                Quiz("Por que la energia de VQE nunca puede bajar del valor FCI (exacto)?",
                     ["El optimizador es muy lento",
                      "El principio variacional: todo estado de prueba tiene E >= la energia base real",
                      "Los errores de redondeo lo impiden",
                      "Porque el circuito tiene un solo parametro"], 1,
                     "Esa es la red de seguridad de VQE: el estado base exacto es un piso duro. Todo el juego es acercarse desde arriba."),
            ]),
        ]

    def run_lesson(self, lesson: Lesson) -> None:
        console.clear()
        console.print(Rule(f"[bold cyan]{lesson.title}[/bold cyan]"))
        for step in lesson.steps:
            if isinstance(step, tuple):
                title, body = step
                self.panel(body, title)
                self.pause()
            elif isinstance(step, Quiz):
                self.run_quiz(step)
                self.pause()
            elif callable(step):
                step()
                self.pause()
        console.print(f"[bold green]✓ {self.t('lesson_done')}[/bold green]\n")

    def lessons_menu(self) -> None:
        while True:
            console.clear()
            console.print(Rule(f"[bold cyan]{self.t('lessons_title')}[/bold cyan]"))
            lessons = self.lessons()
            for i, lesson in enumerate(lessons, start=1):
                console.print(f"  [cyan]{i}[/cyan]. {lesson.title}")
            console.print(f"  [cyan]b[/cyan]. {self.t('back')}")
            choices = [str(i) for i in range(1, len(lessons) + 1)] + ["b"]
            choice = Prompt.ask(self.t("choose"), choices=choices, default="1")
            if choice == "b":
                return
            self.run_lesson(lessons[int(choice) - 1])
            self.pause()

    # ------------------------------------------------------------------
    # Playground
    # ------------------------------------------------------------------

    SINGLE_GATES = {"h", "x", "y", "z", "s", "t"}
    ROTATION_GATES = {"rx", "ry", "rz"}
    DOUBLE_GATES = {"cnot", "cz", "swap"}

    def _rebuild_state(self, n_qubits: int,
                       instructions: List[Tuple[str, List[int], Optional[float]]]):
        state = self.qc.create_state(n_qubits)
        circuit = self.qc.create_circuit(n_qubits)
        for name, targets, param in instructions:
            lname = name.lower()
            if lname in self.SINGLE_GATES:
                getattr(circuit, lname)(targets[0])
            elif lname in self.ROTATION_GATES:
                getattr(circuit, lname)(targets[0], param)
            elif lname == "cnot":
                circuit.cnot(targets[0], targets[1])
            elif lname == "cz":
                circuit.cz(targets[0], targets[1])
            elif lname == "swap":
                circuit.swap(targets[0], targets[1])
        return circuit.run(state)

    def _playground_dashboard(self, n_qubits: int, state,
                              instructions: List[Tuple[str, List[int], Optional[float]]]) -> None:
        drawn = [(name.upper(), targets) for name, targets, _ in instructions]
        console.print(Panel(draw_circuit(n_qubits, drawn),
                            title=self.t("circuit_panel"), border_style="blue"))
        probs = state.probabilities().numpy()
        self.show_state(probs, n_qubits)
        if n_qubits >= 2:
            entropy = state.entropy()
            console.print(f"  {self.t('entropy_label')}: [bold magenta]{entropy:.4f}[/bold magenta]"
                          f"   {self.t('memory_label')}: {state.memory_bytes() / 1024:.1f} KB\n")

    def _show_amplitudes(self, state, n_qubits: int) -> None:
        if n_qubits > STATEVECTOR_DISPLAY_LIMIT:
            console.print(self.t("amp_too_big").format(n=STATEVECTOR_DISPLAY_LIMIT))
            return
        vec = state.to_statevector()
        vec_np = vec.numpy() if hasattr(vec, "numpy") else np.asarray(vec)
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column(self.t("basis"), justify="right", style="bold cyan")
        table.add_column(self.t("amplitude"), justify="right")
        table.add_column(self.t("phase"), justify="right")
        table.add_column(self.t("probability"), justify="right")
        for i, amp in enumerate(vec_np):
            mag = abs(amp)
            if mag < 1e-9:
                continue
            phase = math.degrees(math.atan2(amp.imag if np.iscomplexobj(vec_np) else 0.0,
                                            amp.real if np.iscomplexobj(vec_np) else float(amp)))
            table.add_row(f"|{format(i, f'0{n_qubits}b')}>",
                          f"{mag:.4f}", f"{phase:+7.1f}°", f"{mag * mag:6.1%}")
        console.print(Panel(table, title=self.t("amplitudes_title"), border_style="magenta"))

    def playground(self) -> None:
        console.clear()
        console.print(Rule(f"[bold cyan]{self.t('playground_title')}[/bold cyan]"))
        raw = Prompt.ask(self.t("playground_qubits"), default="2")
        try:
            n_qubits = max(1, min(MAX_PLAYGROUND_QUBITS, int(raw)))
        except ValueError:
            n_qubits = 2

        instructions: List[Tuple[str, List[int], Optional[float]]] = []
        state = self._rebuild_state(n_qubits, instructions)
        console.print(Panel(self.t("playground_help"), border_style="cyan"))
        self._playground_dashboard(n_qubits, state, instructions)

        while True:
            try:
                line = Prompt.ask(f"[bold]{self.t('playground_prompt')}[/bold]")
            except (EOFError, KeyboardInterrupt):
                return
            parts = line.strip().lower().split()
            if not parts:
                continue
            cmd = parts[0]

            if cmd in ("q", "quit", "exit"):
                return
            if cmd == "help":
                console.print(Panel(self.t("playground_help"), border_style="cyan"))
                continue
            if cmd == "reset":
                instructions = []
            elif cmd == "undo":
                if instructions:
                    instructions.pop()
            elif cmd == "bell":
                if n_qubits < 2:
                    console.print(self.t("invalid"))
                    continue
                instructions = [("h", [0], None), ("cnot", [0, 1], None)]
            elif cmd == "ghz":
                if n_qubits < 2:
                    console.print(self.t("invalid"))
                    continue
                instructions = [("h", [0], None)]
                instructions += [("cnot", [i, i + 1], None) for i in range(n_qubits - 1)]
            elif cmd == "amps":
                self._show_amplitudes(state, n_qubits)
                continue
            elif cmd == "sample":
                try:
                    n_samples = int(parts[1]) if len(parts) > 1 else 1000
                    n_samples = max(1, min(n_samples, 1_000_000))
                except ValueError:
                    console.print(self.t("invalid"))
                    continue
                probs = state.probabilities().numpy()
                counts = sample_measurements(probs, n_qubits, n_samples, self.rng)
                console.print(Panel(counts_bars(counts, n_samples, self.lang),
                                    title=f"{self.t('samples_title')}: {n_samples}",
                                    border_style="yellow"))
                continue
            elif cmd in self.SINGLE_GATES:
                try:
                    qubit = int(parts[1])
                    assert 0 <= qubit < n_qubits
                except (IndexError, ValueError, AssertionError):
                    console.print(self.t("invalid"))
                    continue
                instructions.append((cmd, [qubit], None))
            elif cmd in self.ROTATION_GATES:
                try:
                    qubit = int(parts[1])
                    theta = parse_angle(parts[2])
                    assert 0 <= qubit < n_qubits
                except (IndexError, ValueError, AssertionError):
                    console.print(self.t("invalid"))
                    continue
                instructions.append((cmd, [qubit], theta))
            elif cmd in self.DOUBLE_GATES:
                try:
                    a, b = int(parts[1]), int(parts[2])
                    assert 0 <= a < n_qubits and 0 <= b < n_qubits and a != b
                except (IndexError, ValueError, AssertionError):
                    console.print(self.t("invalid"))
                    continue
                instructions.append((cmd, [a, b], None))
            else:
                console.print(self.t("invalid"))
                continue

            state = self._rebuild_state(n_qubits, instructions)
            self._playground_dashboard(n_qubits, state, instructions)

    # ------------------------------------------------------------------
    # Chemistry lab
    # ------------------------------------------------------------------

    def _molecule_card(self, mol) -> None:
        t = TEXT[self.lang]
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")
        table.add_row(t["mol_field_formula"], mol.formula)
        table.add_row(t["mol_field_desc"], mol.description)
        table.add_row(t["mol_field_electrons"], str(mol.n_electrons))
        table.add_row(t["mol_field_orbitals"], str(mol.n_orbitals))
        table.add_row(t["mol_field_qubits"], str(mol.n_qubits))
        table.add_row(t["mol_field_bond"], f"{mol.bond_length_angstrom:.4f} A")
        table.add_row(t["mol_field_basis"], mol.basis)
        table.add_row(t["mol_field_hf"], f"{mol.hf_energy_hartree:.6f} Ha")
        table.add_row(t["mol_field_fci"], f"{mol.fci_energy_hartree:.6f} Ha")
        corr = mol.fci_energy_hartree - mol.hf_energy_hartree
        table.add_row(t["mol_field_corr"], f"[bold magenta]{corr:.6f} Ha[/bold magenta]")
        console.print(Panel(table, title=f"[bold]{mol.name}[/bold]", border_style="green"))
        console.print(Panel(t["mol_explainer"], border_style="dim"))

    def molecule_explorer(self) -> None:
        molecules = list(self.loader.molecules.values())
        if not molecules:
            console.print("[red]No molecules in configuration.[/red]")
            return
        while True:
            console.clear()
            console.print(Rule(f"[bold cyan]{self.t('chem_molecules')}[/bold cyan]"))
            for i, mol in enumerate(molecules, start=1):
                console.print(f"  [cyan]{i}[/cyan]. {mol.name} ({mol.formula}) - {mol.description}")
            console.print(f"  [cyan]b[/cyan]. {self.t('back')}")
            choices = [str(i) for i in range(1, len(molecules) + 1)] + ["b"]
            choice = Prompt.ask(self.t("molecule_pick"), choices=choices, default="1")
            if choice == "b":
                return
            self._molecule_card(molecules[int(choice) - 1])
            self.pause()

    def orbital_viewer(self) -> None:
        while True:
            console.clear()
            console.print(Rule(f"[bold cyan]{self.t('chem_orbitals')}[/bold cyan]"))
            for i, spec in enumerate(ORBITAL_SPECS, start=1):
                console.print(f"  [cyan]{i}[/cyan]. {spec.name}")
            console.print(f"  [cyan]b[/cyan]. {self.t('back')}")
            choices = [str(i) for i in range(1, len(ORBITAL_SPECS) + 1)] + ["b"]
            choice = Prompt.ask(self.t("orbital_pick"), choices=choices, default="1")
            if choice == "b":
                return
            spec = ORBITAL_SPECS[int(choice) - 1]
            art, extent = render_orbital(spec)
            console.print(Panel(Align.center(art), title=f"[bold]{spec.name}[/bold]",
                                border_style="blue"))
            console.print(self.t("orbital_info").format(
                n=spec.n, l=spec.l, plane=spec.plane, extent=extent))
            console.print(Panel(self.t("orbital_legend"), border_style="dim"))
            self.pause()

    def chemistry_menu(self) -> None:
        while True:
            console.clear()
            console.print(Rule(f"[bold cyan]{self.t('chem_title')}[/bold cyan]"))
            console.print(f"  [cyan]1[/cyan]. {self.t('chem_molecules')}")
            console.print(f"  [cyan]2[/cyan]. {self.t('chem_orbitals')}")
            console.print(f"  [cyan]b[/cyan]. {self.t('back')}")
            choice = Prompt.ask(self.t("choose"), choices=["1", "2", "b"], default="1")
            if choice == "1":
                self.molecule_explorer()
            elif choice == "2":
                self.orbital_viewer()
            else:
                return

    # ------------------------------------------------------------------
    # Glossary
    # ------------------------------------------------------------------

    def glossary(self) -> None:
        console.clear()
        console.print(Rule(f"[bold cyan]{self.t('glossary_title')}[/bold cyan]"))
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan", justify="right", no_wrap=True)
        table.add_column(justify="left")
        for term, definition in GLOSSARY[self.lang]:
            table.add_row(term, definition)
        console.print(table)
        self.pause()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def banner(self) -> None:
        title = Text(self.t("title"), style="bold cyan")
        subtitle = Text(self.t("subtitle"), style="dim")
        console.print(Panel(Align.center(Text.assemble(title, "\n", subtitle)),
                            border_style="cyan", padding=(1, 4)))

    def main_menu(self) -> None:
        while True:
            console.clear()
            self.banner()
            console.print(f"  [cyan]1[/cyan]. {self.t('menu_lessons')}")
            console.print(f"  [cyan]2[/cyan]. {self.t('menu_playground')}")
            console.print(f"  [cyan]3[/cyan]. {self.t('menu_chemistry')}")
            console.print(f"  [cyan]4[/cyan]. {self.t('menu_glossary')}")
            console.print(f"  [cyan]5[/cyan]. {self.t('menu_lang')}")
            console.print(f"  [cyan]q[/cyan]. {self.t('menu_quit')}")
            try:
                choice = Prompt.ask(self.t("choose"),
                                    choices=["1", "2", "3", "4", "5", "q"], default="1")
            except (EOFError, KeyboardInterrupt):
                choice = "q"
            if choice == "1":
                self.lessons_menu()
            elif choice == "2":
                self.playground()
            elif choice == "3":
                self.chemistry_menu()
            elif choice == "4":
                self.glossary()
            elif choice == "5":
                self.lang = "es" if self.lang == "en" else "en"
                console.print(f"[green]{self.t('lang_set')}[/green]")
            else:
                console.print(f"\n[bold cyan]{self.t('goodbye')}[/bold cyan]\n")
                return


def launch_quantum_lab(config: Optional[FrameworkConfig] = None,
                       loader: Optional[ConfigLoader] = None,
                       lang: Optional[str] = None,
                       lesson: Optional[int] = None) -> None:
    """Entry point used both standalone and from quantum_framework_main."""
    if config is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "quantum_framework_config.toml")
        config = (FrameworkConfig.from_toml(config_path)
                  if os.path.exists(config_path) else FrameworkConfig())
    if loader is None:
        loader = ConfigLoader()

    if lang not in ("en", "es"):
        console.print()
        lang = Prompt.ask("Language / Idioma", choices=["en", "es"], default="en")

    lab = QuantumLab(config, loader, lang=lang)
    if lesson is not None:
        lessons = lab.lessons()
        if 1 <= lesson <= len(lessons):
            lab.run_lesson(lessons[lesson - 1])
            return
    lab.main_menu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2C Quantum Lab - educational TUI")
    parser.add_argument("--lang", choices=["en", "es"], default=None,
                        help="Interface language (en or es)")
    parser.add_argument("--lesson", type=int, default=None,
                        help="Jump straight into lesson N (1-6)")
    args = parser.parse_args()
    try:
        launch_quantum_lab(lang=args.lang, lesson=args.lesson)
    except KeyboardInterrupt:
        console.print("\n")


if __name__ == "__main__":
    main()
