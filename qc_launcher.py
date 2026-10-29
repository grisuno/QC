#!/usr/bin/env python3
"""Q2C launcher: a Textual front door to the immersive GPU viewer and the rich Quantum Lab."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, OptionList, Select, Static
from textual.widgets.option_list import Option

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable
VIEWER = [PYTHON, "-W", "ignore", str(ROOT / "qc_immersive.py")]
LAB = [sys.executable, "-W", "ignore", str(ROOT / "quantum_lab.py")]

ORBITALS = [
    ("1s  sphere", "1"), ("2s  hollow shell", "2"), ("2p  dumbbell", "3"), ("2p x  real orbital", "4"),
    ("3d z2  lobes + ring", "5"), ("3d xz  four lobes", "6"), ("3d x2-y2", "7"), ("4f  three nodal planes", "8"),
    ("1s + 2p  breathing superposition", "9"), ("2p +1  spinning phase", "0"),
]
RESOLUTIONS = [("1280 x 720", "1280x720"), ("1600 x 900", "1600x900"), ("1920 x 1080", "1920x1080")]

LOGO = r"""[b #78c8ff]   ____  ____   ____
  / __ \|___ \ / ___|
 | |  | | __) | |
 | |__| |/ __/| |___
  \___\_\_____|\____|[/]  [dim]quantum lab[/]"""


@dataclass
class Experience:
    key: str
    title: dict
    blurb: dict
    command: List[str]
    uses_orbital: bool = False


EXPERIENCES = [
    Experience("tour", {"es": "Tour cinematografico", "en": "Cinematic tour"},
               {"es": "Una pelicula interactiva de 80 segundos: del atomo de hidrogeno al entrelazamiento, con camara y subtitulos. Ideal para empezar.",
                "en": "An 80 second interactive film: from the hydrogen atom to entanglement, with camera work and subtitles. Best place to start."},
               VIEWER + ["--tour"]),
    Experience("atom", {"es": "Atomo: nubes de electrones", "en": "Atom: electron clouds"},
               {"es": "Nubes volumetricas calculadas en tu GPU, pixel a pixel. Corta la nube en tajadas, mira los nodos, la fase en color y superposiciones que laten.",
                "en": "Volumetric clouds computed on your GPU, pixel by pixel. Slice the cloud open, see nodes, phase as colour and beating superpositions."},
               VIEWER + ["--scene", "atom"], True),
    Experience("spin", {"es": "Spin: ver el spin del electron", "en": "Spin: see the electron spin"},
               {"es": "Espinores de Pauli exactos: flechas de spin dentro de la nube, precesion de Larmor en un campo magnetico y batido spin-orbita con las energias de tu backend de Dirac.",
                "en": "Exact Pauli spinors: spin arrows inside the cloud, Larmor precession in a magnetic field and spin-orbit beating using your Dirac backend energies."},
               VIEWER + ["--scene", "spin"]),
    Experience("matter", {"es": "Materia: atomos y moleculas reales", "en": "Matter: real atoms and molecules"},
               {"es": "H a Ne y 10 moleculas (H2O, CH4, N2, O2, etileno...) calculadas ab initio con PySCF: densidad, orbitales, densidad de spin y FCI. Incluye el VQE real del proyecto para H2.",
                "en": "H to Ne and 10 molecules (H2O, CH4, N2, O2, ethylene...) computed ab initio with PySCF: density, orbitals, spin density and FCI. Includes the project's real VQE for H2."},
               VIEWER + ["--scene", "matter"]),
    Experience("bloch", {"es": "Qubit: esfera de Bloch", "en": "Qubit: Bloch sphere"},
               {"es": "Corre sobre tu QuantumComputer. Un qubit es un punto en una esfera. Aplica H, X, Y, Z, S, T y mira cada compuerta girar la flecha con su estela.",
                "en": "Runs on your QuantumComputer. A qubit is a point on a sphere. Apply H, X, Y, Z, S, T and watch each gate rotate the arrow, leaving a trail."},
               VIEWER + ["--scene", "bloch"]),
    Experience("entangle", {"es": "Entrelazamiento", "en": "Entanglement"},
               {"es": "Corre sobre tu QuantumComputer. Dos esferas de Bloch y un haz entre ellas. Crea un par de Bell: las flechas se encogen a cero mientras la correlacion se vuelve perfecta.",
                "en": "Runs on your QuantumComputer. Two Bloch spheres and a beam between them. Build a Bell pair: the arrows shrink to zero while the correlation becomes perfect."},
               VIEWER + ["--scene", "entangle"]),
    Experience("lab", {"es": "Laboratorio en terminal (rich)", "en": "Terminal lab (rich)"},
               {"es": "Las 6 lecciones interactivas, el VQE de H2 en vivo y el glosario, dentro de la terminal.",
                "en": "The 6 interactive lessons, live H2 VQE and glossary, right inside the terminal."},
               LAB),
]

CHROME = {
    "title": {"es": "Q2C  Laboratorio Cuantico", "en": "Q2C  Quantum Lab"},
    "pick": {"es": "Elige una experiencia", "en": "Pick an experience"},
    "orbital": {"es": "Orbital inicial", "en": "Starting orbital"},
    "res": {"es": "Resolucion", "en": "Resolution"},
    "launch": {"es": "Enter: lanzar   L: idioma   Q: salir", "en": "Enter: launch   L: language   Q: quit"},
    "running": {"es": "Visor abierto (pid {pid}). Puedes lanzar otro.", "en": "Viewer open (pid {pid}). You can launch another."},
}


class Launcher(App):
    CSS = """
    Screen { background: #070b18; }
    #logo { height: 6; padding: 1 2; }
    #menu { width: 38; border: round #3a5a99; padding: 0 1; }
    #detail { border: round #3a5a99; padding: 1 2; }
    #blurb { height: auto; margin-bottom: 1; }
    #options { height: auto; }
    Select { margin-bottom: 1; }
    #status { color: #ffbe6e; margin-top: 1; }
    Label { color: #96a8c8; }
    """
    BINDINGS = [
        Binding("l", "toggle_lang", "Lang"),
        Binding("q", "quit", "Quit"),
        Binding("enter", "launch", "Launch", priority=True),
    ]

    def __init__(self):
        super().__init__()
        self.lang = "en"
        self.index = 0
        self.processes: List[subprocess.Popen] = []

    def t(self, key: str, **kw) -> str:
        return CHROME[key][self.lang].format(**kw)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(LOGO, id="logo")
        with Horizontal():
            with Vertical(id="menu"):
                yield OptionList(*self._options(), id="list")
            with Vertical(id="detail"):
                yield Static("", id="blurb")
                with Vertical(id="options"):
                    yield Select(ORBITALS, value="5", allow_blank=False, id="orbital")
                    yield Select(RESOLUTIONS, value="1280x720", allow_blank=False, id="res")
                yield Static("", id="status")
        yield Footer()

    def _options(self):
        return [Option(e.title[self.lang], id=e.key) for e in EXPERIENCES]

    def on_mount(self):
        self.title = self.t("title")
        self.query_one("#list", OptionList).highlighted = 0
        self.refresh_detail()

    def refresh_detail(self):
        exp = EXPERIENCES[self.index]
        self.query_one("#blurb", Static).update(f"[b]{exp.title[self.lang]}[/b]\n\n{exp.blurb[self.lang]}\n\n[dim]{self.t('launch')}[/dim]")
        self.query_one("#orbital", Select).display = exp.uses_orbital
        self.query_one("#res", Select).display = exp.command[:len(VIEWER)] == VIEWER
        self.title = self.t("title")

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted):
        self.index = event.option_index
        self.refresh_detail()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        self.index = event.option_index
        self.action_launch()

    def action_toggle_lang(self):
        self.lang = "en" if self.lang == "es" else "es"
        listing = self.query_one("#list", OptionList)
        listing.clear_options()
        listing.add_options(self._options())
        listing.highlighted = self.index
        self.refresh_detail()

    def build_command(self, exp: Experience) -> List[str]:
        command = list(exp.command)
        if command[:len(VIEWER)] == VIEWER:
            command += ["--size", str(self.query_one("#res", Select).value), "--lang", self.lang]
            if exp.uses_orbital:
                command += ["--do", f"preset:{self.query_one('#orbital', Select).value}"]
        return command

    def action_launch(self):
        exp = EXPERIENCES[self.index]
        command = self.build_command(exp)
        if exp.key == "lab":
            with self.suspend():
                subprocess.run(command, cwd=ROOT)
            return
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.processes.append(proc)
        self.query_one("#status", Static).update(self.t("running", pid=proc.pid))


if __name__ == "__main__":
    Launcher().run()
