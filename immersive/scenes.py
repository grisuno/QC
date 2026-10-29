"""Interactive scenes: hydrogen orbital clouds, H2 molecular orbitals, Bloch sphere, entanglement."""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import pygame
from OpenGL import GL as gl

from . import matter, spin
from .backend import BackendRegister, run_project_vqe
from .glkit import OrbitCamera, make_texture, upload_radial, upload_volume
from .hud import ACCENT, DIM, INK, WARM, Hud, Lang
from .physics import (MAX_STATES, PRESETS, SINGLE_GATES, HARTREE_EV, QubitRegister, density_peak, energy_ev,
                      orbital_label, overlap_1s, radial_table, view_radius)

TEXT: Dict[str, Dict[str, str]] = {
    "tab_atom": {"en": "Atom", "es": "Atomo"},
    "tab_spin": {"en": "Spin", "es": "Spin"},
    "tab_matter": {"en": "Matter", "es": "Materia"},
    "hint_spin": {"en": "drag: orbit | wheel: zoom | 1-6: spin states | A: arrows | up/down: magnetic field | X: slice | S: sweep | C: colour",
                  "es": "arrastra: orbitar | rueda: zoom | 1-6: estados de spin | A: flechas | arriba/abajo: campo magnetico | X: corte | S: barrido | C: color"},
    "energy_backend": {"en": "ENERGIES - project backend", "es": "ENERGIAS - backend del proyecto"},
    "dirac_source": {"en": "Energies from DiracHydrogenAtom", "es": "Energias de DiracHydrogenAtom"},
    "dirac_ok": {"en": "backend OK", "es": "backend OK"},
    "dirac_missing": {"en": "backend unavailable", "es": "backend no disponible"},
    "splitting": {"en": "Level splitting", "es": "Desdoblamiento"},
    "field_keys": {"en": "up/down to change", "es": "arriba/abajo para cambiar"},
    "time_scale": {"en": "Time scale", "es": "Escala de tiempo"},
    "spin_legend": {"en": "colour = spin direction  (blue up, olive down)", "es": "color = direccion del spin  (azul arriba, oliva abajo)"},
    "tab_bloch": {"en": "Qubit", "es": "Qubit"},
    "tab_entangle": {"en": "Entanglement", "es": "Entrelazamiento"},
    "energy": {"en": "Energy", "es": "Energia"},
    "radial_nodes": {"en": "Radial nodes", "es": "Nodos radiales"},
    "angular_nodes": {"en": "Angular nodes", "es": "Nodos angulares"},
    "phase_legend": {"en": "Wave phase", "es": "Fase de la onda"},
    "positive": {"en": "+", "es": "+"},
    "negative": {"en": "-", "es": "-"},
    "l0": {"en": "s: a sphere. No nodal plane, the electron can sit on the nucleus.",
           "es": "s: una esfera. Sin plano nodal, el electron puede estar sobre el nucleo."},
    "l1": {"en": "p: two lobes, one nodal plane. Opposite sign on each side.",
           "es": "p: dos lobulos, un plano nodal. Signo opuesto a cada lado."},
    "l2": {"en": "d: four lobes (or a ring), two nodal surfaces.",
           "es": "d: cuatro lobulos (o un anillo), dos superficies nodales."},
    "l3": {"en": "f: eight lobes, three nodal surfaces.", "es": "f: ocho lobulos, tres superficies nodales."},
    "cut_on": {"en": "Slice ON", "es": "Corte ACTIVO"},
    "cut_off": {"en": "Slice off", "es": "Corte apagado"},
    "sweep": {"en": "Sweep", "es": "Barrido"},
    "color_mode": {"en": "Colour", "es": "Color"},
    "mode_phase": {"en": "phase", "es": "fase"},
    "mode_heat": {"en": "heat", "es": "calor"},
    "mode_spin": {"en": "spin direction", "es": "direccion de spin"},
    "mode_ice": {"en": "ice", "es": "hielo"},
    "hint_atom": {"en": "drag: orbit | wheel: zoom | 1-9,0: orbitals | N L M: quantum numbers | X: slice | Z: slice axis | S: sweep | C: colour | -/=: density",
                  "es": "arrastra: orbitar | rueda: zoom | 1-9,0: orbitales | N L M: numeros cuanticos | X: corte | Z: eje del corte | S: barrido | C: color | -/=: densidad"},
    "hint_molecule": {"en": "drag: orbit | wheel: zoom | left/right: bond length | B: bonding/antibonding | X: slice | S: sweep | C: colour",
                      "es": "arrastra: orbitar | rueda: zoom | izq/der: longitud de enlace | B: enlazante/antienlazante | X: corte | S: barrido | C: color"},
    "hint_bloch": {"en": "drag: orbit | wheel: zoom | H X Y Z S T: gates | R: reset | 0 1 +: prepare state | K: backend | V: evolve",
                   "es": "arrastra: orbitar | rueda: zoom | H X Y Z S T: compuertas | R: reiniciar | 0 1 +: preparar estado | K: backend | V: evolucion"},
    "hint_entangle": {"en": "drag: orbit | H / J: Hadamard on qubit 0 / 1 | C: CNOT | B: Bell pair | M: measure x200 | R: reset | K: backend | V: evolve",
                      "es": "arrastra: orbitar | H / J: Hadamard en qubit 0 / 1 | C: CNOT | B: par de Bell | M: medir x200 | R: reiniciar | K: backend | V: evolucion"},
    "hint_matter": {"en": "drag: orbit | wheel: zoom | left/right: system | up/down: orbital | D: density | P: spin density | F: FCI (H2) | O: orbital | G: atoms/molecules | X: slice | S: sweep",
                    "es": "arrastra: orbitar | rueda: zoom | izq/der: sistema | arriba/abajo: orbital | D: densidad | P: densidad de spin | F: FCI (H2) | O: orbital | G: atomos/moleculas | X: corte | S: barrido"},
    "display_note": {"en": "shown as rho^0.6", "es": "mostrado como rho^0.6"},
    "kind_density": {"en": "Density", "es": "Densidad"},
    "kind_spin": {"en": "Spin", "es": "Spin"},
    "kind_mo": {"en": "Orbital", "es": "Orbital"},
    "kind_fci": {"en": "FCI", "es": "FCI"},
    "orbitals": {"en": "Molecular orbitals", "es": "Orbitales moleculares"},
    "groups": {"en": "atoms / molecules", "es": "atomos / moleculas"},
    "computing": {"en": "Computing SCF with PySCF...", "es": "Calculando SCF con PySCF..."},
    "need_pyscf": {"en": "PySCF is not installed", "es": "PySCF no esta instalado"},
    "need_pyscf_how": {"en": "Create the local environment:  python3 -m venv --system-site-packages .venv && .venv/bin/pip install pyscf  then start the viewer with .venv/bin/python qc_immersive.py",
                       "es": "Crea el entorno local:  python3 -m venv --system-site-packages .venv && .venv/bin/pip install pyscf  y lanza el visor con .venv/bin/python qc_immersive.py"},
    "vqe_title": {"en": "PROJECT VQE (backend)", "es": "VQE DEL PROYECTO (backend)"},
    "iterations": {"en": "iterations", "es": "iteraciones"},
    "bond": {"en": "Bond length", "es": "Longitud de enlace"},
    "overlap": {"en": "Overlap S", "es": "Solapamiento S"},
    "bonding": {"en": "Bonding  sigma 1s", "es": "Enlazante  sigma 1s"},
    "antibonding": {"en": "Antibonding  sigma* 1s", "es": "Antienlazante  sigma* 1s"},
    "bonding_note": {"en": "Same phase: the clouds add up between the nuclei and glue them together.",
                     "es": "Misma fase: las nubes se suman entre los nucleos y los pegan."},
    "anti_note": {"en": "Opposite phase: a nodal plane splits the molecule. No glue, the nuclei repel.",
                  "es": "Fase opuesta: un plano nodal parte la molecula. Sin pegamento, los nucleos se repelen."},
    "eq_note": {"en": "H2 equilibrium: 1.40 a0 (0.74 A). The VQE lesson finds this minimum.",
                "es": "Equilibrio de H2: 1.40 a0 (0.74 A). La leccion de VQE encuentra este minimo."},
    "state": {"en": "Qubit state", "es": "Estado del qubit"},
    "theta": {"en": "Polar angle", "es": "Angulo polar"},
    "phi": {"en": "Azimuth", "es": "Azimut"},
    "length": {"en": "Arrow length", "es": "Largo de la flecha"},
    "gate_note": {"en": "Every gate is a rotation of the arrow on the sphere.",
                  "es": "Cada compuerta es una rotacion de la flecha sobre la esfera."},
    "gate_H": {"en": "Hadamard: rotates 180 degrees about the X+Z diagonal. |0> becomes an equal superposition.",
               "es": "Hadamard: gira 180 grados sobre la diagonal X+Z. |0> pasa a una superposicion pareja."},
    "gate_X": {"en": "Pauli-X: quantum NOT. 180 degrees about X.", "es": "Pauli-X: NOT cuantico. 180 grados sobre X."},
    "gate_Y": {"en": "Pauli-Y: 180 degrees about Y.", "es": "Pauli-Y: 180 grados sobre Y."},
    "gate_Z": {"en": "Pauli-Z: flips the phase. 180 degrees about Z.", "es": "Pauli-Z: invierte la fase. 180 grados sobre Z."},
    "gate_S": {"en": "S gate: a quarter turn about Z (90 degrees of phase).", "es": "Compuerta S: un cuarto de vuelta sobre Z (90 grados de fase)."},
    "gate_T": {"en": "T gate: an eighth turn about Z. Key to universal quantum computing.",
               "es": "Compuerta T: un octavo de vuelta sobre Z. Clave de la computacion cuantica universal."},
    "gate_Rx": {"en": "Rx(45): a small rotation about X.", "es": "Rx(45): una rotacion pequena sobre X."},
    "gate_Ry": {"en": "Ry(45): a small rotation about Y.", "es": "Ry(45): una rotacion pequena sobre Y."},
    "reset": {"en": "Reset", "es": "Reiniciar"},
    "entropy": {"en": "Entanglement entropy", "es": "Entropia de entrelazamiento"},
    "bits": {"en": "bit", "es": "bit"},
    "ent_none": {"en": "Product state: each qubit has its own pure arrow on the surface.",
                 "es": "Estado producto: cada qubit tiene su propia flecha pura sobre la superficie."},
    "ent_some": {"en": "Partial entanglement: the arrows are shrinking, information moves into the correlation.",
                 "es": "Entrelazamiento parcial: las flechas se encogen, la informacion pasa a la correlacion."},
    "ent_full": {"en": "Maximally entangled! Each qubit alone is completely random (arrow length 0), yet the pair is perfectly correlated.",
                 "es": "Maximamente entrelazados! Cada qubit solo es totalmente aleatorio (flecha de largo 0), pero el par esta perfectamente correlacionado."},
    "measure": {"en": "Measure x200", "es": "Medir x200"},
    "results": {"en": "Measurement results", "es": "Resultados de la medicion"},
    "bell": {"en": "Bell pair", "es": "Par de Bell"},
    "probabilities": {"en": "Probabilities (colour = phase)", "es": "Probabilidades (color = fase)"},
    "qubit": {"en": "qubit", "es": "qubit"},
    "backend_title": {"en": "BACKEND - your simulator", "es": "BACKEND - tu simulador"},
    "ckpt_loaded": {"en": "checkpoint loaded", "es": "checkpoint cargado"},
    "ckpt_missing": {"en": "analytic fallback", "es": "respaldo analitico"},
    "physics": {"en": "Physics", "es": "Fisica"},
    "vs_ideal": {"en": "gap vs ideal unitary", "es": "brecha vs unitaria ideal"},
    "evolve_on": {"en": "V: spatial evolution ON", "es": "V: evolucion espacial ACTIVA"},
    "evolve_off": {"en": "V: spatial evolution off", "es": "V: evolucion espacial apagada"},
    "steps": {"en": "steps", "es": "pasos"},
}

PHASE_BAR_STOPS = 24


class Scene:
    key = ""
    tab_key = ""
    hint_key = ""

    def __init__(self, app):
        self.app = app
        self.camera = OrbitCamera()
        self.keymap: Dict[int, str] = {}

    def enter(self):
        pass

    def loading(self) -> bool:
        return False

    def update(self, dt: float):
        self.camera.update(dt)

    def render(self):
        raise NotImplementedError

    def draw_hud(self, hud: Hud, tr: Lang):
        pass

    def command(self, cmd: str):
        pass

    def on_key(self, ev: pygame.event.Event) -> bool:
        cmd = self.keymap.get(ev.key)
        if cmd:
            self.command(cmd)
            return True
        return False

    def panel_rect(self, hud: Hud, x: float, y: float, w: float, h: float) -> pygame.Rect:
        rect = pygame.Rect(hud.s(x), hud.s(y), hud.s(w), hud.s(h))
        hud.panel(rect)
        return rect


class VolumeScene(Scene):
    """Shared raymarching state for atom and molecule scenes: slicing, colour, density."""

    def __init__(self, app):
        super().__init__(app)
        self.prog = app.programs["orbital"]
        self.radial_tex = make_texture(8, 1, gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT, np.zeros((1, 8), np.float32))
        self.rmax = 8.0
        self.norm = 1.0
        self.reveal = 0.0
        self.cut = False
        self.cut_axis = 2
        self.cut_offset = 0.0
        self.sweep = False
        self.color_mode = 0
        self.density = 1.5
        self.gamma = float(os.environ.get('Q2C_GAMMA', '0.85'))
        self.fade_density = 1.0
        self._sweep_clock = 0.0
        self.keymap.update({
            pygame.K_x: "cut", pygame.K_z: "axis", pygame.K_s: "sweep", pygame.K_c: "color",
            pygame.K_MINUS: "dens-", pygame.K_EQUALS: "dens+", pygame.K_LEFTBRACKET: "cut-",
            pygame.K_RIGHTBRACKET: "cut+",
        })

    @property
    def cloud_radius(self) -> float:
        return self.rmax * 0.72

    def cut_normal(self) -> Tuple[np.ndarray, float]:
        reach = self.cloud_radius
        if self.cut_axis == 3:
            eye = self.camera.eye()
            return eye / np.linalg.norm(eye), self.cut_offset * reach
        normal = np.zeros(3)
        normal[self.cut_axis] = 1.0
        return normal, self.cut_offset * reach

    def command(self, cmd: str):
        if cmd == "cut":
            self.cut = not self.cut
        elif cmd == "axis":
            self.cut_axis = (self.cut_axis + 1) % 4
            self.cut = True
        elif cmd == "sweep":
            self.sweep = not self.sweep
            if self.sweep:
                self.cut = True
        elif cmd == "color":
            self.color_mode = (self.color_mode + 1) % 3
        elif cmd == "dens+":
            self.density = min(12.0, self.density * 1.25)
        elif cmd == "dens-":
            self.density = max(0.1, self.density / 1.25)
        elif cmd == "cut+":
            self.cut_offset = min(1.0, self.cut_offset + 0.05)
            self.sweep = False
        elif cmd == "cut-":
            self.cut_offset = max(-1.0, self.cut_offset - 0.05)
            self.sweep = False

    def update(self, dt: float):
        super().update(dt)
        self.reveal = min(self.rmax + 3.0, self.reveal + dt * (self.rmax + 3.0) / 1.7)
        if self.sweep:
            self._sweep_clock += dt
            self.cut_offset = 0.85 * math.sin(self._sweep_clock * 0.55)

    def set_volume_uniforms(self):
        p = self.prog
        p.use()
        self.camera.set_uniforms(p, *self.app.size)
        p.f1("uTime", self.app.time)
        p.f1("uFade", 1.0)
        p.f1("uRMax", self.rmax)
        p.f1("uNorm", self.norm)
        p.f1("uDensity", self.density * self.fade_density * 14.0 / self.rmax)
        p.f1("uGamma", self.gamma)
        p.i1("uColorMode", self.color_mode)
        p.f1("uReveal", self.reveal)
        normal, offset = self.cut_normal()
        p.i1("uCut", 1 if self.cut else 0)
        p.v3("uCutN", normal)
        p.f1("uCutD", offset)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.radial_tex)
        p.i1("uRadial", 0)

    def phase_legend(self, hud: Hud, tr: Lang, x: int, y: int):
        w, h = hud.s(210), hud.s(12)
        hud.text(tr("phase_legend"), x, y - hud.s(22), 13, DIM)
        for i in range(PHASE_BAR_STOPS):
            phase = -math.pi + 2 * math.pi * i / (PHASE_BAR_STOPS - 1)
            pygame.draw.rect(hud.surface, hud.hue_color(phase), pygame.Rect(x + i * w // PHASE_BAR_STOPS, y, w // PHASE_BAR_STOPS + 1, h))
        hud.text("-pi", x, y + h + hud.s(3), 12, DIM)
        hud.text("0", x + w // 2, y + h + hud.s(3), 12, DIM, anchor="tc")
        hud.text("+pi", x + w, y + h + hud.s(3), 12, DIM, anchor="tr")

    def cut_hud(self, hud: Hud, tr: Lang, x: float, y: float):
        label = tr("cut_on") if self.cut else tr("cut_off")
        axis = ["X", "Y", "Z", "cam"][self.cut_axis]
        extra = f"  [{axis}]  {self.cut_offset:+.2f}" if self.cut else ""
        first = tr("mode_spin") if getattr(self, "key", "") == "spin" else tr("mode_phase")
        mode = [first, tr("mode_heat"), tr("mode_ice")][self.color_mode]
        hud.text(f"{label}{extra}   {tr('color_mode')}: {mode}" + (f"   {tr('sweep')}" if self.sweep else ""), x, y, 13, DIM)


class AtomScene(VolumeScene):
    key = "atom"
    tab_key = "tab_atom"
    hint_key = "hint_atom"

    def __init__(self, app):
        super().__init__(app)
        self.preset = 2
        self.qn = (2, 1, 0)
        self.manual = False
        self.states: List[Tuple[int, int, int, complex]] = []
        self.label = ""
        self.note_key = ""
        self.clock_au = 0.0
        self.spin = False
        self.keymap.update({
            pygame.K_n: "n+", pygame.K_l: "l+", pygame.K_m: "m+",
        })
        for p in PRESETS:
            self.keymap[getattr(pygame, f"K_{p.key}")] = f"preset:{p.key}"
        self.load_preset(PRESETS[self.preset])

    def enter(self):
        self.reveal = 0.0

    def load_preset(self, preset):
        self.manual = False
        self.preset = PRESETS.index(preset)
        self.spin = any(abs(c) > 0 and m != 0 for _, _, m, c in preset.states) and len(preset.states) == 1
        self.set_states(preset.states, preset.name_en)
        self.note_key = ""

    def set_states(self, states, label: str):
        states = list(states)[:MAX_STATES]
        self.states = states
        self.label = label
        qn = [(n, l, m) for n, l, m, _ in states]
        n_max = max(n for n, _, _ in qn)
        self.rmax = view_radius(n_max)
        coefs = np.array([c for *_, c in states], dtype=complex)
        self.norm = 1.0 / density_peak(qn, coefs, self.rmax)
        upload_radial(self.radial_tex, radial_table(qn, self.rmax))
        self.rows = len(qn)
        self.reveal = 0.0
        self.camera.fly_to(self.rmax * 2.15)

    def command(self, cmd: str):
        if cmd.startswith("preset:"):
            key = cmd.split(":")[1]
            self.load_preset(next(p for p in PRESETS if p.key == key))
            return
        if cmd in ("n+", "l+", "m+"):
            n, l, m = self.qn if self.manual else (self.states[0][0], self.states[0][1], self.states[0][2])
            if cmd == "n+":
                n = n % 4 + 1
                l = min(l, n - 1)
            elif cmd == "l+":
                l = (l + 1) % n
            else:
                m = m + 1 if m < l else -l
            m = max(-l, min(l, m))
            self.qn = (n, l, m)
            self.manual = True
            self.spin = m != 0
            self.set_states([(n, l, m, 1.0)], orbital_label(n, l, m))
            return
        super().command(cmd)

    def coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        qn = np.zeros((MAX_STATES, 3), np.int32)
        co = np.zeros((MAX_STATES, 2), np.float32)
        e0 = -0.5 / self.states[0][0] ** 2
        for k, (n, l, m, c) in enumerate(self.states):
            e = -0.5 / n**2
            phase = -(e - e0) * self.clock_au
            if self.spin:
                phase += -0.35 * self.clock_au
            z = complex(c) * complex(math.cos(phase), math.sin(phase))
            qn[k] = (n, l, m)
            co[k] = (z.real, z.imag)
        return qn, co

    def update(self, dt: float):
        super().update(dt)
        self.clock_au += dt * 3.0

    def render(self):
        self.set_volume_uniforms()
        p = self.prog
        qn, co = self.coefficients()
        p.i1("uMode", 0)
        p.i1("uCount", len(self.states))
        p.v3ia("uQN[0]", qn)
        p.v2a("uCoef[0]", co)
        p.f1("uRows", max(1, len(self.states)))
        self.app.fullscreen.draw()

    def draw_hud(self, hud: Hud, tr: Lang):
        preset = None if self.manual else PRESETS[self.preset]
        title = (preset.name_es if (preset and tr.code == "es") else (preset.name_en if preset else self.label))
        rect = self.panel_rect(hud, 18, 70, 330, 300)
        x, y = rect.x + hud.s(18), rect.y + hud.s(14)
        size = 30
        while size > 16 and hud.font(size, True).size(title)[0] > rect.width - hud.s(36):
            size -= 2
        hud.text(title, x, y, size, INK, True)
        y += hud.s(44)
        if len(self.states) == 1 or self.manual:
            n, l, m = self.states[0][:3]
            hud.text(f"n = {n}    l = {l}    m = {m:+d}", x, y, 17, ACCENT, True)
            y += hud.s(28)
            hud.text(f"{tr('energy')}: {energy_ev(n):.2f} eV", x, y, 15, INK)
            y += hud.s(22)
            hud.text(f"{tr('radial_nodes')}: {n - l - 1}     {tr('angular_nodes')}: {l}", x, y, 15, INK)
            y += hud.s(30)
            l_note = tr(f"l{l}")
            for line in hud.wrap(l_note, rect.width - hud.s(36), 14):
                hud.text(line, x, y, 14, DIM)
                y += hud.s(19)
        else:
            hud.text("psi = " + " + ".join(f"c{n}{'spdf'[l]}" for n, l, _, _ in self.states), x, y, 17, ACCENT, True)
            y += hud.s(28)
            es = sorted({energy_ev(n) for n, *_ in self.states})
            hud.text(f"E = {', '.join(f'{e:.2f}' for e in es)} eV", x, y, 15, INK)
            y += hud.s(30)
        if preset:
            note = preset.note_es if tr.code == "es" else preset.note_en
            for line in hud.wrap(note, rect.width - hud.s(36), 14):
                hud.text(line, x, y, 14, WARM)
                y += hud.s(19)
        self.cut_hud(hud, tr, rect.x + hud.s(18), rect.bottom - hud.s(34))
        self.phase_legend(hud, tr, hud.size[0] - hud.s(248), hud.s(78))
        self.draw_energy_panel(hud, tr, rect.bottom / hud.scale + 12)

    def draw_energy_panel(self, hud: Hud, tr: Lang, top: float):
        link = spin.shared_link()
        rect = self.panel_rect(hud, 18, top, 330, 150)
        x, y = rect.x + hud.s(14), rect.y + hud.s(10)
        hud.text(tr("energy_backend"), x, y, 14, ACCENT, True)
        y += hud.s(22)
        states = self.states
        if len(states) == 1 or self.manual:
            n, l = states[0][0], states[0][1]
            e_s = -0.5 / n**2
            hud.text(f"Schrodinger   {e_s * HARTREE_EV:+.6f} eV", x, y, 13, INK)
            y += hud.s(20)
            comps = [spin.SpinorComponent(n, l, sign, 1, 1.0) for sign in ((1,) if l == 0 else (-1, 1))]
            energies = []
            for comp in comps:
                e = link.energy(comp)
                energies.append(e)
                hud.text(f"Dirac {comp.label}  {e * HARTREE_EV:+.6f} eV   ({(e - e_s) * 1e6:+.2f} uHa)", x, y, 13, INK)
                y += hud.s(20)
            if len(energies) == 2:
                hud.text(f"{tr('splitting')} = {abs(energies[0] - energies[1]) * spin.HARTREE_HZ / 1e9:.2f} GHz", x, y, 13, WARM, True)
        else:
            energies = sorted({-0.5 / n**2 for n, *_ in states})
            delta = abs(energies[-1] - energies[0])
            wavelength_nm = 45.5633525 / delta
            hud.text(f"\u0394E = {delta * HARTREE_EV:.3f} eV   (Lyman-\u03b1)", x, y, 13, INK)
            y += hud.s(20)
            hud.text(f"f = {delta * spin.HARTREE_HZ / 1e15:.3f} PHz   \u03bb = {wavelength_nm:.1f} nm", x, y, 13, INK)
            y += hud.s(20)
            hud.text(f"{tr('time_scale')}: 1 s = {spin.real_time_label(3.0)}", x, y, 13, WARM)
        if not link.ok:
            hud.text(link.error or tr("dirac_missing"), x, rect.bottom - hud.s(22), 11, (255, 140, 120))


class SpinScene(VolumeScene):
    """Pauli-spinor hydrogen states: spin direction as colour and arrows, driven by the project's Dirac energies."""

    key = "spin"
    tab_key = "tab_spin"
    hint_key = "hint_spin"
    ARROWS = 40

    def __init__(self, app):
        super().__init__(app)
        self.link = spin.shared_link()
        self.preset = spin.SPIN_PRESETS[1]
        self.components = self.preset.components
        self.field_tesla = self.preset.field_tesla
        self.energies: List[float] = []
        self.clock_au = 0.0
        self.scale = 1.0
        self.measurement: spin.SpinMeasurement | None = None
        self.measure_clock = 0.0
        self.show_arrows = True
        self.density = 1.3
        for preset in spin.SPIN_PRESETS:
            self.keymap[getattr(pygame, f"K_{preset.key}")] = f"preset:{preset.key}"
        self.keymap.update({pygame.K_a: "arrows", pygame.K_UP: "b+", pygame.K_DOWN: "b-"})
        self.load(self.preset)

    def enter(self):
        self.reveal = 0.0

    def load(self, preset: spin.SpinPreset):
        self.preset, self.components, self.field_tesla = preset, preset.components, preset.field_tesla
        self.refresh()
        self.reveal = 0.0
        self.clock_au = 0.0
        self.camera.fly_to(self.rmax * 2.0)

    def refresh(self):
        comps = self.components
        self.rmax = view_radius(max(c.n for c in comps))
        self.energies = [spin.component_energy(c, self.link, self.field_tesla) for c in comps]
        self.scale = spin.time_scale(comps, self.link, self.field_tesla) if len(comps) > 1 else 1.0
        self.norm = spin.exposure(comps, self.energies, self.rmax)
        upload_radial(self.radial_tex, radial_table([(c.n, c.l, 0) for c in comps], self.rmax))
        self.measure_clock = 1.0

    def command(self, cmd: str):
        if cmd.startswith("preset:"):
            key = cmd.split(":")[1]
            self.load(next(p for p in spin.SPIN_PRESETS if p.key == key))
        elif cmd == "arrows":
            self.show_arrows = not self.show_arrows
        elif cmd in ("b+", "b-") and self.preset.field_tesla > 0:
            factor = 1.25 if cmd == "b+" else 0.8
            self.field_tesla = float(np.clip(self.field_tesla * factor, 0.5, 60.0))
            self.refresh()
        else:
            super().command(cmd)

    def coefficients(self) -> np.ndarray:
        return spin.evolved_coefficients(self.components, self.energies, self.clock_au)

    def update(self, dt: float):
        super().update(dt)
        self.clock_au += dt * self.scale
        self.measure_clock += dt
        if self.measure_clock > 0.07:
            self.measure_clock = 0.0
            self.measurement = spin.measure(self.components, self.coefficients(), self.rmax * 0.72, self.ARROWS)

    def render(self):
        self.set_volume_uniforms()
        p = self.prog
        coefs = self.coefficients()
        qn = np.zeros((MAX_STATES, 4), np.int32)
        co = np.zeros((MAX_STATES, 2), np.float32)
        for k, (c, z) in enumerate(zip(self.components, coefs)):
            qn[k] = (c.n, c.l, c.mj2, c.sign)
            co[k] = (z.real, z.imag)
        p.i1("uMode", 2)
        p.i1("uCount", len(self.components))
        p.v4ia("uSP[0]", qn)
        p.v2a("uCoef[0]", co)
        p.f1("uRows", max(1, len(self.components)))
        m = self.measurement
        if m is not None and self.show_arrows:
            n = len(m.arrows_pos)
            pos, dirs, w = np.zeros((40, 3)), np.zeros((40, 3)), np.zeros(40)
            pos[:n], dirs[:n], w[:n] = m.arrows_pos, m.arrows_dir, m.arrows_weight
            p.i1("uNAr", n)
            p.v3a("uArP[0]", pos)
            p.v3a("uArD[0]", dirs)
            gl.glUniform1fv(p.loc("uArW[0]"), 40, np.ascontiguousarray(w, dtype=np.float32))
            p.f1("uArL", self.rmax * 0.72 * 0.13)
        else:
            p.i1("uNAr", 0)
        p.i1("uNAt", 0)
        p.i1("uNBd", 0)
        self.app.fullscreen.draw()

    def draw_hud(self, hud: Hud, tr: Lang):
        pre = self.preset
        rect = self.panel_rect(hud, 18, 70, 372, 470)
        x, y = rect.x + hud.s(18), rect.y + hud.s(12)
        title = pre.name_es if tr.code == "es" else pre.name_en
        for line in hud.wrap(title, rect.width - hud.s(36), 17, True):
            hud.text(line, x, y, 17, INK, True)
            y += hud.s(24)
        y += hud.s(4)
        for line in hud.wrap(pre.note_es if tr.code == "es" else pre.note_en, rect.width - hud.s(36), 13):
            hud.text(line, x, y, 13, WARM)
            y += hud.s(18)
        y += hud.s(8)
        for comp, energy in zip(self.components, self.energies):
            hud.text(f"{comp.label}  m_j={comp.mj2:+d}/2   E = {energy:.8f} Ha", x, y, 13, INK)
            y += hud.s(19)
        link_ok = tr("dirac_ok") if self.link.ok else tr("dirac_missing")
        hud.text(f"{tr('dirac_source')}: {link_ok}", x, y, 12, (130, 235, 160) if self.link.ok else (255, 150, 120))
        y += hud.s(18)
        if len(self.components) == 2 and self.components[0].l == self.components[1].l and self.components[0].n == self.components[1].n:
            delta = abs(self.energies[0] - self.energies[1]) * spin.HARTREE_HZ / 1e9
            hud.text(f"{tr('splitting')} = {delta:.2f} GHz", x, y, 13, ACCENT, True)
            y += hud.s(20)
        elif len(self.components) == 2:
            delta = abs(self.energies[0] - self.energies[1]) * spin.HARTREE_HZ / 1e9
            hud.text(f"{tr('splitting')} = {delta:.1f} GHz  (B = {self.field_tesla:.1f} T)", x, y, 13, ACCENT, True)
            y += hud.s(20)
        if self.preset.field_tesla > 0:
            hud.text(f"B = {self.field_tesla:.1f} T   ({tr('field_keys')})", x, y, 12, DIM)
            y += hud.s(18)
        if len(self.components) > 1:
            hud.text(f"{tr('time_scale')}: 1 s = {spin.real_time_label(self.scale)}", x, y, 13, WARM)
            y += hud.s(22)
        m = self.measurement
        if m is not None:
            y += hud.s(4)
            hud.text("\u27e8\u03c3\u27e9", x, y, 15, INK, True)
            for k, (axis, value) in enumerate(zip("xyz", m.sigma)):
                row = y + hud.s(26) + k * hud.s(22)
                hud.text(f"{axis} {value:+.3f}", x, row, 13, DIM)
                track = pygame.Rect(x + hud.s(78), row + hud.s(5), hud.s(120), hud.s(9))
                hud.bar(track, 0.5 + 0.5 * float(value), (255, 190, 110) if value >= 0 else (120, 200, 255))
            self.spin_inset(hud, x + hud.s(240), y + hud.s(18), hud.s(54), m.sigma)
            y += hud.s(100)
            hud.text(f"\u222b|\u03c8|\u00b2 = {m.norm:.4f}", x, y, 12, DIM)
        self.cut_hud(hud, tr, x, rect.bottom - hud.s(30))
        hud.text(tr("spin_legend"), hud.size[0] - hud.s(18), hud.s(78), 13, DIM, anchor="tr")

    def spin_inset(self, hud: Hud, cx: int, cy: int, radius: int, sigma: np.ndarray):
        surf = hud.surface
        pygame.draw.circle(surf, (110, 170, 255, 120), (cx, cy), radius, 1)
        pygame.draw.ellipse(surf, (110, 170, 255, 80), pygame.Rect(cx - radius, cy - radius // 3, 2 * radius, 2 * radius // 3), 1)
        right, up, _ = self.camera.basis()
        screen = np.array([np.dot(sigma, right), np.dot(sigma, up)])
        tip = (cx + int(screen[0] * radius), cy - int(screen[1] * radius))
        color = tuple(int(255 * min(1.0, c)) for c in (0.5 + 0.5 * sigma / max(np.linalg.norm(sigma), 1e-9)))
        pygame.draw.line(surf, color, (cx, cy), tip, 3)
        pygame.draw.circle(surf, color, tip, hud.s(5))


class MatterScene(VolumeScene):
    """Atoms and molecules from PySCF (UHF/RHF, 6-31G) rendered from a 3D texture of the electron density."""

    key = "matter"
    tab_key = "tab_matter"
    hint_key = "hint_matter"
    DISPLAY_POWER = 0.6
    HEADLINE = {"density": "kind_density", "spin": "kind_spin", "mo": "kind_mo", "fci": "kind_fci"}

    def __init__(self, app):
        super().__init__(app)
        self.available = matter.pyscf_available()
        self.solver = matter.AsyncSolver()
        self.index = 3
        self.kind = "density"
        self.orbital = None
        self.loaded_token = None
        self.loaded: matter.Solved | None = None
        self.vol_tex = gl.glGenTextures(1)
        self.cut_axis = 0
        self.density = 0.9
        self.grid_charge = 0.0
        self.vqe = None
        self.geometry = None
        self.keymap.update({
            pygame.K_LEFT: "sys-", pygame.K_RIGHT: "sys+", pygame.K_UP: "orb+", pygame.K_DOWN: "orb-",
            pygame.K_d: "kind:density", pygame.K_p: "kind:spin", pygame.K_f: "kind:fci", pygame.K_o: "kind:mo",
            pygame.K_g: "group",
        })

    def enter(self):
        self.reveal = 0.0
        self.warm_up()

    def warm_up(self, keys=("H2O",)):
        if not self.available:
            return
        for key in keys:
            system = next(s for s in matter.LIBRARY if s.key == key)
            self.solver.prefetch(system, ("density", "spin"))

    @property
    def system(self) -> matter.System:
        return matter.LIBRARY[self.index]

    def loading(self) -> bool:
        return self.available and (self.loaded_token is None or self.loaded_token != (self.system.key, self.wanted_kind())) \
            and f"{self.system.key}" not in self.solver.errors

    def command(self, cmd: str):
        n = len(matter.LIBRARY)
        if cmd in ("sys+", "sys-"):
            self.index = (self.index + (1 if cmd == "sys+" else -1)) % n
            self.orbital = None
            self.kind = "density"
        elif cmd == "group":
            self.index = len(matter.MOLECULES) if self.index < len(matter.MOLECULES) else 0
            self.orbital = None
            self.kind = "density"
        elif cmd.startswith("sys:"):
            self.index = [s.key for s in matter.LIBRARY].index(cmd.split(":")[1])
            self.orbital = None
            self.kind = "density"
        elif cmd.startswith("kind:"):
            self.kind = cmd.split(":")[1]
            if self.kind == "mo" and self.orbital is None and self.loaded:
                self.orbital = self.loaded.homo
        elif cmd in ("orb+", "orb-") and self.loaded:
            top = len(self.loaded.orbitals) - 1
            current = self.loaded.homo if self.orbital is None else self.orbital
            self.orbital = max(0, min(top, current + (1 if cmd == "orb+" else -1)))
            self.kind = "mo"
        elif cmd == "homo" and self.loaded:
            self.orbital, self.kind = self.loaded.homo, "mo"
        elif cmd == "lumo" and self.loaded:
            self.orbital, self.kind = min(len(self.loaded.orbitals) - 1, self.loaded.homo + 1), "mo"
        else:
            super().command(cmd)

    def wanted_kind(self) -> str:
        solved = self.solver.cache.get(self.system.key)
        kind = self.kind
        if kind == "spin" and not (solved and solved.unrestricted):
            kind = "density"
        if kind == "fci" and self.system.key != "H2":
            kind = "density"
        if kind == "mo":
            return f"mo:{self.orbital if self.orbital is not None else (solved.homo if solved else 0)}"
        return kind

    def update(self, dt: float):
        super().update(dt)
        if not self.available:
            return
        system = self.system
        self.solver.request(system)
        solved = self.solver.cache.get(system.key)
        if solved is None:
            return
        kind = self.wanted_kind()
        self.solver.request_field(system, kind)
        token = (system.key, kind)
        if self.solver.fields_ready.get(token) and token != self.loaded_token:
            self.apply(solved, kind, token)
        if system.key == "H2" and self.vqe is None:
            try:
                self.vqe = run_project_vqe()
            except Exception as exc:
                self.vqe = {"error_text": f"{type(exc).__name__}: {str(exc)[:60]}"}

    def apply(self, solved: matter.Solved, kind: str, token):
        amp = matter.field_amplitude(solved, kind)
        shown = amp if kind.startswith("mo:") else np.sign(amp) * np.abs(amp) ** self.DISPLAY_POWER
        upload_volume(self.vol_tex, shown)
        new_system = self.loaded is None or self.loaded.system.key != solved.system.key
        self.loaded, self.loaded_token = solved, token
        self.rmax = solved.half
        self.norm = matter.exposure(shown, solved.half)
        dv = (2 * solved.half / (matter.GRID - 1)) ** 3
        self.grid_charge = float((np.sign(amp) * amp**2).sum() * dv)
        self.reveal = 0.0 if new_system else self.rmax + 3.0
        if new_system:
            self.camera.fly_to(solved.half * 2.0)
        symbols = solved.symbols
        coords = solved.atoms_bohr
        count = min(len(symbols), 12)
        pos, col, rad = np.zeros((12, 3)), np.zeros((12, 3)), np.zeros(12)
        for i in range(count):
            pos[i], col[i], rad[i] = coords[i], matter.CPK[symbols[i]], matter.DISPLAY_RADIUS[symbols[i]]
        pairs = matter.bonds(symbols, coords)[:16]
        a, b = np.zeros((16, 3)), np.zeros((16, 3))
        for k, (i, j) in enumerate(pairs):
            a[k], b[k] = coords[i], coords[j]
        self.geometry = (count, pos, col, rad, len(pairs), a, b)

    def render(self):
        self.set_volume_uniforms()
        p = self.prog
        p.i1("uMode", 3)
        p.i1("uCount", 0)
        p.f1("uRows", 1.0)
        if self.geometry is not None and self.loaded is not None:
            count, pos, col, rad, nb, a, b = self.geometry
            p.i1("uNAt", count)
            p.v3a("uAtP[0]", pos)
            p.v3a("uAtC[0]", col)
            gl.glUniform1fv(p.loc("uAtR[0]"), 12, np.ascontiguousarray(rad, dtype=np.float32))
            p.i1("uNBd", nb)
            p.v3a("uBdA[0]", a)
            p.v3a("uBdB[0]", b)
        else:
            p.i1("uNAt", 0)
            p.i1("uNBd", 0)
            p.f1("uReveal", 0.0)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_3D, self.vol_tex)
        p.i1("uVol", 2)
        self.app.fullscreen.draw()
        gl.glActiveTexture(gl.GL_TEXTURE0)

    def draw_hud(self, hud: Hud, tr: Lang):
        system = self.system
        rect = self.panel_rect(hud, 18, 70, 372, 560)
        x, y = rect.x + hud.s(18), rect.y + hud.s(12)
        if not self.available:
            hud.text(tr("need_pyscf"), x, y, 15, WARM, True)
            for line in hud.wrap(tr("need_pyscf_how"), rect.width - hud.s(36), 13):
                y += hud.s(20)
                hud.text(line, x, y, 13, DIM)
            return
        bw, bh = hud.s(36), hud.s(32)
        hud.button(pygame.Rect(x, y, bw, bh), "\u25c0", "sys-")
        hud.text(system.title, x + bw + hud.s(12), y + hud.s(3), 20, INK, True)
        hud.button(pygame.Rect(rect.right - hud.s(18) - bw, y, bw, bh), "\u25b6", "sys+")
        y += hud.s(42)
        hud.text(f"{self.index + 1}/{len(matter.LIBRARY)}   (G: {tr('groups')})", x, y, 12, DIM)
        y += hud.s(22)
        solved = self.loaded if (self.loaded and self.loaded.system.key == system.key) else None
        if solved is None:
            err = self.solver.errors.get(system.key)
            hud.text(err or tr("computing"), x, y, 14, WARM if not err else (255, 140, 120))
            return
        method = ("UHF" if solved.unrestricted else "RHF") + f"/{matter.BASIS}  (PySCF)"
        hud.text(f"{method}", x, y, 14, ACCENT, True)
        y += hud.s(22)
        hud.text(f"E = {solved.e_total:.6f} Ha" + ("" if solved.converged else "  (not converged)"), x, y, 14, INK)
        y += hud.s(20)
        if solved.fci_energy is not None:
            hud.text(f"FCI = {solved.fci_energy:.6f} Ha   (corr. {(solved.fci_energy - solved.e_total) * 1000:.1f} mHa)", x, y, 14, INK)
            y += hud.s(20)
        hud.text(f"\u222b\u03c1 dV = {self.grid_charge:.3f}   ({solved.mol.nelectron} e)   {tr('display_note')}", x, y, 12, DIM)
        y += hud.s(26)
        note = system.note_es if tr.code == "es" else system.note_en
        for line in hud.wrap(note, rect.width - hud.s(36), 13):
            hud.text(line, x, y, 13, WARM)
            y += hud.s(18)
        y += hud.s(8)
        kinds = [("density", tr("kind_density")), ("spin", tr("kind_spin")), ("mo", tr("kind_mo"))]
        if system.key == "H2":
            kinds.append(("fci", tr("kind_fci")))
        kw = (rect.width - hud.s(36) - hud.s(6) * (len(kinds) - 1)) // len(kinds)
        for i, (k, label) in enumerate(kinds):
            enabled = not (k == "spin" and not solved.unrestricted)
            hud.button(pygame.Rect(x + i * (kw + hud.s(6)), y, kw, hud.s(28)), label, f"kind:{k}", self.kind == k and enabled)
        y += hud.s(38)
        shown = self.orbital if self.orbital is not None else solved.homo
        lo = max(0, min(shown - 4, len(solved.orbitals) - 10))
        hud.text(tr("orbitals") + ("  (alpha)" if solved.unrestricted else ""), x, y, 13, INK, True)
        y += hud.s(22)
        for orb in solved.orbitals[lo:lo + 10]:
            sel = self.kind == "mo" and orb.index == shown
            tag = "HOMO" if orb.index == solved.homo else ("LUMO" if orb.index == solved.homo + 1 else "")
            color = ACCENT if sel else (INK if orb.occupation > 0 else DIM)
            dots = "\u25cf" * int(round(orb.occupation)) + "\u25cb" * (0 if orb.occupation > 0 else 1)
            hud.text(f"{'>' if sel else ' '} MO {orb.index + 1:2d}  {orb.energy_ev:8.2f} eV  {dots:<3} {tag}", x, y, 13, color, sel)
            y += hud.s(19)
        y += hud.s(6)
        self.cut_hud(hud, tr, x, y)
        if system.key == "H2" and self.vqe:
            self.draw_vqe(hud, tr)
        self.phase_legend(hud, tr, hud.size[0] - hud.s(248), hud.s(78))

    def draw_vqe(self, hud: Hud, tr: Lang):
        v = self.vqe
        rect = self.panel_rect(hud, hud.size[0] / hud.scale - 336, 140, 318, 190)
        x, y = rect.x + hud.s(14), rect.y + hud.s(10)
        hud.text(tr("vqe_title"), x, y, 15, ACCENT, True)
        y += hud.s(24)
        if "e_vqe" not in v:
            hud.text(v.get("error_text", "n/a"), x, y, 12, (255, 140, 120))
            return
        hud.text(f"{v['engine']}  -  {v['model']}", x, y, 11, DIM)
        y += hud.s(20)
        hud.text(f"E(VQE) = {v['e_vqe']:.9f} Ha", x, y, 13, INK)
        y += hud.s(20)
        hud.text(f"E(FCI) = {v['e_fci']:.9f} Ha", x, y, 13, INK)
        y += hud.s(20)
        hud.text(f"|error| = {abs(v['error']):.1e} Ha  ({v['iterations']} {tr('iterations')})", x, y, 13, (130, 235, 160))
        y += hud.s(26)
        hist = np.array(v["history"])
        top, bottom = hist.max(), hist.min()
        w, h = hud.s(290), hud.s(44)
        pts = [(x + int(i / max(1, len(hist) - 1) * w), y + h - int((e - bottom) / max(top - bottom, 1e-12) * h)) for i, e in enumerate(hist)]
        if len(pts) > 1:
            pygame.draw.lines(hud.surface, (120, 200, 255), False, pts, 2)


def fmt_c(z: complex) -> str:
    return f"{abs(z):.3f} @ {math.degrees(math.atan2(z.imag, z.real)):+4.0f} deg"


def backend_command(reg: BackendRegister, cmd: str):
    if cmd == "backend":
        reg.cycle_backend()
    elif cmd == "evolve":
        reg.live_evolve = not reg.live_evolve


def draw_backend_panel(hud: Hud, tr: Lang, reg: BackendRegister, x: float, y: float):
    """Show which engine produced the picture, plus the backend's spatial wavefunction fields."""
    info = reg.describe()
    fields = reg.fields()
    count = len(fields)
    height = 214 if count <= 2 else 262
    rect = pygame.Rect(hud.s(x), hud.s(y), hud.s(318), hud.s(height))
    hud.panel(rect)
    px, py = rect.x + hud.s(14), rect.y + hud.s(10)
    hud.text(tr("backend_title"), px, py, 15, ACCENT, True)
    py += hud.s(24)
    hud.text(f"quantum_computer.QuantumComputer  {info['grid']}x{info['grid']} grid", px, py, 12, DIM)
    py += hud.s(20)
    ok = tr("ckpt_loaded") if info["loaded"] else tr("ckpt_missing")
    hud.text(f"{tr('physics')}: {info['label']}  [{ok}]", px, py, 13, INK)
    py += hud.s(20)
    dev = info["deviation"]
    color = (130, 235, 160) if dev < 1e-3 else (255, 150, 120)
    hud.text(f"{tr('vs_ideal')}: {dev:.1e}", px, py, 13, color)
    py += hud.s(22)
    live = tr("evolve_on") if reg.live_evolve else tr("evolve_off")
    hud.text(f"{live}  ({info['evolve_steps']} {tr('steps')})", px, py, 12, WARM if reg.live_evolve else DIM)
    py += hud.s(20)
    if info["error"]:
        hud.text(info["error"], px, py, 11, (255, 140, 120))
    py += hud.s(16)
    size = hud.s(64)
    prob = np.abs(fields) ** 2
    peak = float(prob.max()) or 1.0
    for k in range(min(count, 4)):
        col, row = k % 4 if count <= 4 else k % 4, k // 4
        cell = pygame.Rect(px + col * (size + hud.s(8)), py + row * (size + hud.s(18)), size, size)
        hud.heatmap(prob[k], cell, peak)
        hud.text(f"|{k:0{reg.n}b}\u27e9", cell.centerx, cell.bottom + hud.s(2), 11, DIM, anchor="tc")


class BlochScene(Scene):
    key = "bloch"
    tab_key = "tab_bloch"
    hint_key = "hint_bloch"
    COLOR = (1.0, 0.62, 0.25)
    TRAIL = 48

    def __init__(self, app):
        super().__init__(app)
        self.prog = app.programs["bloch"]
        self.reg = None
        self.camera = OrbitCamera(dist=4.6, yaw=0.9, pitch=0.35)
        self.camera.fly_to(4.6)
        self.trail: deque = deque(maxlen=self.TRAIL)
        self.pop_clock = 0.0
        self.last_gate = ""
        self.keymap.update({
            pygame.K_h: "gate:H", pygame.K_x: "gate:X", pygame.K_y: "gate:Y", pygame.K_z: "gate:Z",
            pygame.K_s: "gate:S", pygame.K_t: "gate:T", pygame.K_r: "reset", pygame.K_0: "prep:0",
            pygame.K_1: "prep:1", pygame.K_PLUS: "prep:+", pygame.K_KP_PLUS: "prep:+", pygame.K_EQUALS: "prep:+",
            pygame.K_k: "backend", pygame.K_v: "evolve",
        })

    def enter(self):
        if self.reg is None:
            self.reg = BackendRegister(1)

    def command(self, cmd: str):
        if cmd.startswith("backend") or cmd == "evolve":
            backend_command(self.reg, cmd)
        elif cmd.startswith("gate:"):
            name = cmd.split(":")[1]
            self.reg.queue(self.reg.single(name, 0))
            self.last_gate = name
        elif cmd == "reset":
            self.reg.reset()
            self.trail.clear()
            self.last_gate = ""
        elif cmd.startswith("prep:"):
            self.reg.prepare(cmd.split(":")[1])
            self.trail.clear()

    def update(self, dt: float):
        super().update(dt)
        self.reg.update(dt)
        v = self.reg.bloch(0)
        if self.reg.busy:
            if not self.trail or np.linalg.norm(np.array(self.trail[-1]) - v) > 0.012:
                self.trail.append(tuple(v))
            self.pop_clock = 0.0
        else:
            self.pop_clock += dt
            if self.pop_clock > 0.9 and self.trail:
                self.trail.popleft()
                self.pop_clock = 0.82

    def render(self):
        p = self.prog
        p.use()
        self.camera.set_uniforms(p, *self.app.size)
        p.f1("uTime", self.app.time)
        p.f1("uFade", 1.0)
        p.i1("uNS", 1)
        p.v3a("uC[0]", np.zeros((2, 3)))
        vecs = np.zeros((2, 3))
        vecs[0] = self.reg.bloch(0)
        p.v3a("uVec[0]", vecs)
        p.v3a("uCol[0]", np.array([self.COLOR, self.COLOR]))
        p.f1("uLink", 0.0)
        p.f1("uPulse", self.reg.progress * (1.0 - self.reg.progress) * 4.0 if self.reg.busy else 0.0)
        trail = np.zeros((96, 3))
        pts = list(self.trail)
        for k, t in enumerate(pts):
            trail[k] = t
        p.v3a("uTrail[0]", trail)
        gl.glUniform1iv(p.loc("uTrailN[0]"), 2, np.array([len(pts), 0], dtype=np.int32))
        self.app.fullscreen.draw()

    def sphere_labels(self, hud: Hud, centre: np.ndarray, tag: str = ""):
        labels = [((0, 0, 1.25), "|0>"), ((0, 0, -1.25), "|1>"), ((1.28, 0, 0), "|+>"), ((-1.28, 0, 0), "|->"),
                  ((0, 1.28, 0), "|+i>"), ((0, -1.28, 0), "|-i>")]
        for offset, name in labels:
            x, y, ok = self.camera.project(centre + np.array(offset), *hud.size)
            if ok:
                hud.text(name.replace(">", "⟩").replace("|", "|"), x, y, 15, ACCENT, True, "c")

    def draw_hud(self, hud: Hud, tr: Lang):
        self.sphere_labels(hud, np.zeros(3))
        rect = self.panel_rect(hud, 18, 70, 330, 430)
        x, y = rect.x + hud.s(18), rect.y + hud.s(14)
        hud.text(tr("state"), x, y, 22, INK, True)
        y += hud.s(36)
        a, b = self.reg.psi
        hud.text(f"α |0⟩ : {fmt_c(a)}", x, y, 15, INK)
        y += hud.s(22)
        hud.text(f"β |1⟩ : {fmt_c(b)}", x, y, 15, INK)
        y += hud.s(28)
        probs = self.reg.probabilities()
        for k in range(2):
            hud.text(f"P(|{k}⟩) = {probs[k] * 100:5.1f}%", x, y, 14, DIM)
            hud.bar(pygame.Rect(x + hud.s(150), y + hud.s(5), hud.s(140), hud.s(9)), probs[k], (255, 170, 90) if k == 0 else (110, 200, 255))
            y += hud.s(22)
        v = self.reg.bloch(0)
        r = float(np.linalg.norm(v))
        theta = math.degrees(math.acos(np.clip(v[2] / r, -1, 1))) if r > 1e-6 else 0.0
        phi = math.degrees(math.atan2(v[1], v[0])) if r > 1e-6 else 0.0
        y += hud.s(6)
        hud.text(f"{tr('theta')}: {theta:6.1f} deg", x, y, 14, DIM)
        y += hud.s(20)
        hud.text(f"{tr('phi')}: {phi:6.1f} deg", x, y, 14, DIM)
        y += hud.s(20)
        hud.text(f"{tr('length')}: {r:.3f}", x, y, 14, DIM)
        y += hud.s(30)
        names = list(SINGLE_GATES.keys())
        bw, bh = hud.s(66), hud.s(32)
        for i, name in enumerate(names):
            col, row = i % 4, i // 4
            hot = self.reg.active_name.startswith(name + "(")
            hud.button(pygame.Rect(x + col * (bw + hud.s(6)), y + row * (bh + hud.s(6)), bw, bh), name, f"gate:{name}", hot)
        y += 2 * (bh + hud.s(6)) + hud.s(4)
        hud.button(pygame.Rect(x, y, hud.s(138), bh), tr("reset"), "reset")
        hud.button(pygame.Rect(x + hud.s(146), y, hud.s(66), bh), "|+⟩", "prep:+")
        hud.button(pygame.Rect(x + hud.s(218), y, hud.s(66), bh), "|1⟩", "prep:1")
        if self.last_gate:
            note = tr(f"gate_{self.last_gate}")
            ty = rect.bottom + hud.s(12)
            for line in hud.wrap(note, hud.s(330), 14):
                hud.text(line, rect.x + hud.s(6), ty, 14, WARM)
                ty += hud.s(19)
        else:
            hud.text(tr("gate_note"), rect.x + hud.s(6), rect.bottom + hud.s(12), 14, WARM)
        draw_backend_panel(hud, tr, self.reg, hud.size[0] / hud.scale - 336, 70)


class EntangleScene(Scene):
    key = "entangle"
    tab_key = "tab_entangle"
    hint_key = "hint_entangle"
    COLORS = ((1.0, 0.62, 0.25), (0.30, 0.85, 1.0))
    TRAIL = 48
    SEPARATION = 3.0
    HOME_YAW = -1.25

    def __init__(self, app):
        super().__init__(app)
        self.prog = app.programs["bloch"]
        self.reg = None
        self.camera = OrbitCamera(dist=8.4, yaw=self.HOME_YAW, pitch=0.28)
        self.camera.auto_speed = 0.0
        self.camera.fly_to(8.4)
        self.trails = [deque(maxlen=self.TRAIL), deque(maxlen=self.TRAIL)]
        self.pop_clock = 0.0
        self.results: Dict[str, int] | None = None
        self.results_anim = 0.0
        self.keymap.update({
            pygame.K_h: "h0", pygame.K_j: "h1", pygame.K_x: "x0", pygame.K_c: "cnot", pygame.K_b: "bell",
            pygame.K_m: "measure", pygame.K_r: "reset", pygame.K_k: "backend", pygame.K_v: "evolve",
        })

    def centres(self) -> np.ndarray:
        return np.array([[-self.SEPARATION / 2, 0, 0], [self.SEPARATION / 2, 0, 0]])

    def enter(self):
        if self.reg is None:
            self.reg = BackendRegister(2)

    def command(self, cmd: str):
        reg = self.reg
        if cmd.startswith("backend") or cmd == "evolve":
            backend_command(reg, cmd)
        elif cmd == "h0":
            reg.queue(reg.single("H", 0))
        elif cmd == "h1":
            reg.queue(reg.single("H", 1))
        elif cmd == "x0":
            reg.queue(reg.single("X", 0))
        elif cmd == "x1":
            reg.queue(reg.single("X", 1))
        elif cmd == "cnot":
            reg.queue(reg.cnot(0, 1))
        elif cmd == "bell":
            reg.reset()
            for t in self.trails:
                t.clear()
            self.results = None
            reg.queue(reg.single("H", 0))
            reg.queue(reg.cnot(0, 1))
        elif cmd == "measure":
            self.results = reg.measure(200)
            self.results_anim = 0.0
        elif cmd == "reset":
            reg.reset()
            for t in self.trails:
                t.clear()
            self.results = None

    def update(self, dt: float):
        super().update(dt)
        cam = self.camera
        if not cam.dragging and cam.idle > 2.5:
            sway = self.HOME_YAW + 0.2 * math.sin(self.app.time * 0.25)
            cam.yaw += (sway - cam.yaw) * (1 - math.exp(-dt * 0.8))
        self.reg.update(dt)
        self.results_anim = min(1.0, self.results_anim + dt * 1.5)
        moving = self.reg.busy
        for q in range(2):
            v = self.reg.bloch(q)
            if moving and (not self.trails[q] or np.linalg.norm(np.array(self.trails[q][-1]) - v) > 0.012):
                self.trails[q].append(tuple(v))
        if moving:
            self.pop_clock = 0.0
        else:
            self.pop_clock += dt
            if self.pop_clock > 0.9:
                for t in self.trails:
                    if t:
                        t.popleft()
                self.pop_clock = 0.82

    def render(self):
        p = self.prog
        p.use()
        self.camera.set_uniforms(p, *self.app.size)
        p.f1("uTime", self.app.time)
        p.f1("uFade", 1.0)
        p.i1("uNS", 2)
        p.v3a("uC[0]", self.centres())
        vecs = np.array([self.reg.bloch(0), self.reg.bloch(1)])
        p.v3a("uVec[0]", vecs)
        p.v3a("uCol[0]", np.array(self.COLORS))
        entropy = self.reg.entropy(0)
        p.f1("uLink", entropy)
        p.f1("uPulse", self.reg.progress * (1.0 - self.reg.progress) * 4.0 if self.reg.busy else 0.0)
        trail = np.zeros((96, 3))
        counts = []
        for q in range(2):
            pts = list(self.trails[q])
            for k, t in enumerate(pts):
                trail[q * 48 + k] = t
            counts.append(len(pts))
        p.v3a("uTrail[0]", trail)
        gl.glUniform1iv(p.loc("uTrailN[0]"), 2, np.array(counts, dtype=np.int32))
        self.app.fullscreen.draw()

    def draw_hud(self, hud: Hud, tr: Lang):
        centres = self.centres()
        for q in range(2):
            BlochScene.sphere_labels(self, hud, centres[q])
            x, y, ok = self.camera.project(centres[q] + np.array([0, 0, -1.65]), *hud.size)
            if ok:
                col = tuple(int(c * 255) for c in self.COLORS[q])
                hud.text(f"{tr('qubit')} {q}", x, y, 17, col, True, "c")
        rect = self.panel_rect(hud, 18, 70, 340, 500)
        x, y = rect.x + hud.s(18), rect.y + hud.s(14)
        hud.text(tr("probabilities"), x, y, 17, INK, True)
        y += hud.s(34)
        probs = self.reg.probabilities()
        for k in range(4):
            amp = self.reg.psi[k]
            phase = math.atan2(amp.imag, amp.real) if abs(amp) > 1e-6 else 0.0
            hud.text(f"|{k:02b}⟩", x, y, 15, INK, True)
            color = hud.hue_color(phase) if abs(amp) > 1e-3 else (90, 100, 120)
            hud.bar(pygame.Rect(x + hud.s(56), y + hud.s(5), hud.s(190), hud.s(11)), probs[k], color)
            hud.text(f"{probs[k] * 100:4.0f}%", x + hud.s(256), y, 14, DIM)
            y += hud.s(26)
        y += hud.s(10)
        entropy = self.reg.entropy(0)
        hud.text(f"{tr('entropy')}", x, y, 16, INK, True)
        y += hud.s(26)
        hud.bar(pygame.Rect(x, y, hud.s(250), hud.s(14)), entropy, (255, 120, 220))
        hud.text(f"{entropy:.2f} {tr('bits')}", x + hud.s(262), y - hud.s(3), 15, (255, 150, 230), True)
        y += hud.s(28)
        key = "ent_none" if entropy < 0.05 else ("ent_full" if entropy > 0.95 else "ent_some")
        for line in hud.wrap(tr(key), rect.width - hud.s(36), 14):
            hud.text(line, x, y, 14, WARM)
            y += hud.s(19)
        y = rect.bottom - hud.s(96)
        bw, bh = hud.s(96), hud.s(32)
        gap = hud.s(8)
        row = [("H₀", "h0"), ("H₁", "h1"), ("X₀", "x0")]
        for i, (label, action) in enumerate(row):
            hud.button(pygame.Rect(x + i * (bw + gap), y, bw, bh), label, action)
        y += bh + gap
        hud.button(pygame.Rect(x, y, bw, bh), "CNOT", "cnot", self.reg.active_name.startswith("CNOT"))
        hud.button(pygame.Rect(x + bw + gap, y, bw, bh), tr("bell"), "bell")
        hud.button(pygame.Rect(x + 2 * (bw + gap), y, bw, bh), tr("measure"), "measure")
        hud.button(pygame.Rect(x, y + bh + gap, hud.s(3 * bw + 2 * gap) // 1, hud.s(26)), tr("reset"), "reset")
        draw_backend_panel(hud, tr, self.reg, hud.size[0] / hud.scale - 336, 70)
        if self.results:
            panel = self.panel_rect(hud, hud.size[0] / hud.scale - 336, 350, 318, 190)
            px, py = panel.x + hud.s(16), panel.y + hud.s(12)
            hud.text(tr("results"), px, py, 16, INK, True)
            py += hud.s(30)
            total = sum(self.results.values())
            for bits, count in self.results.items():
                hud.text(f"|{bits}⟩", px, py, 14, INK, True)
                hud.bar(pygame.Rect(px + hud.s(52), py + hud.s(4), hud.s(170), hud.s(11)), count / total * self.results_anim, (255, 140, 220))
                hud.text(f"{count:3d}", px + hud.s(236), py, 14, DIM)
                py += hud.s(26)
