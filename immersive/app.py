"""Immersive viewer entry: window, event loop, cinematic tour director."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from OpenGL import GL as gl

from .backend import BackendHub
from .glkit import FullscreenPass, Program
from .postfx import PostProcess
from .hud import ACCENT, DIM, INK, WARM, Hud, Lang
from .scenes import TEXT, AtomScene, BlochScene, EntangleScene, MatterScene, RydbergScene, SpinScene, TuneScene
from .shaders import bloch_fragment, orbital_fragment, surface_fragment

SCENE_ORDER = ("atom", "spin", "matter", "rydberg", "tune", "bloch", "entangle")
FADE_SECONDS = 0.45
LETTERBOX = 0.085
SCREENSHOT_DT = 1.0 / 30.0

TOUR_TEXT = {
    "t1": {"en": "This is a hydrogen atom. Its electron is not a dot in orbit: it is a cloud of probability.",
           "es": "Este es un atomo de hidrogeno. Su electron no es un punto orbitando: es una nube de probabilidad."},
    "t2": {"en": "Excited states have shape. Colour is the phase of the wave: blue is positive, orange is negative.",
           "es": "Los estados excitados tienen forma. El color es la fase de la onda: azul positivo, naranja negativo."},
    "t3": {"en": "Let us slice the cloud open. Dark gaps are nodes: places the electron never visits.",
           "es": "Abramos la nube en tajadas. Los huecos oscuros son nodos: lugares que el electron jamas visita."},
    "t4": {"en": "Mix two states and the cloud breathes. This oscillating charge is how atoms emit light.",
           "es": "Mezcla dos estados y la nube respira. Esta carga oscilante es como los atomos emiten luz."},
    "t3b": {"en": "Every dot is one measurement. Measure the electron again and again and the cloud builds up: this is the Born rule.",
            "es": "Cada punto es una medicion. Mide el electron una y otra vez y la nube se va formando: esta es la regla de Born."},
    "t4c": {"en": "Mix many shells of a circular orbit and a wave packet emerges: a quantum electron that moves like a classical planet.",
            "es": "Mezcla muchas capas de una orbita circular y emerge un paquete de ondas: un electron cuantico que se mueve como un planeta clasico."},
    "t4b": {"en": "The electron also has spin. Colour and arrows show its direction; in a magnetic field it precesses.",
            "es": "El electron tambien tiene spin. Color y flechas muestran su direccion; en un campo magnetico precesa."},
    "t5": {"en": "Bring two atoms together and their clouds merge into a chemical bond.",
           "es": "Acerca dos atomos y sus nubes se funden en un enlace quimico."},
    "t6": {"en": "Flip the phase and a node cuts the molecule in half. The bond is gone.",
           "es": "Invierte la fase y un nodo parte la molecula en dos. El enlace desaparece."},
    "t7": {"en": "A qubit is a point on a sphere. Every quantum gate is simply a rotation.",
           "es": "Un qubit es un punto sobre una esfera. Cada compuerta cuantica es simplemente una rotacion."},
    "t8": {"en": "Entangle two qubits: each alone becomes pure noise, yet together they are perfectly correlated.",
           "es": "Entrelaza dos qubits: cada uno solo se vuelve puro ruido, pero juntos estan perfectamente correlacionados."},
    "t9": {"en": "Now it is your turn. Explore, slice, rotate, break things.",
           "es": "Ahora te toca a ti. Explora, corta, rota, rompe cosas."},
    "title": {"en": "Q2C  Quantum Lab", "es": "Q2C  Laboratorio Cuantico"},
    "subtitle": {"en": "See the invisible", "es": "Ve lo invisible"},
}
TEXT.update(TOUR_TEXT)
TEXT.update({
    "tour": {"en": "Cinematic tour", "es": "Tour cinematico"},
    "stop": {"en": "Stop tour", "es": "Detener tour"},
    "shot": {"en": "Screenshot saved", "es": "Captura guardada"},
})


@dataclass
class Step:
    duration: float
    caption: str
    scene: str
    acts: List[Tuple[float, Callable[["App"], None]]] = field(default_factory=list)


def build_tour() -> List[Step]:
    def cmd(*commands):
        return lambda app: [app.scene.command(c) for c in commands]

    def spin(speed):
        return lambda app: setattr(app.scene.camera, "auto_speed", speed)

    return [
        Step(7.0, "t1", "atom", [(0.0, cmd("preset:1")), (0.0, spin(0.30))]),
        Step(7.0, "t2", "atom", [(0.0, cmd("preset:3"))]),
        Step(10.0, "t3", "atom", [(0.0, cmd("preset:5")), (1.8, cmd("sweep"))]),
        Step(9.0, "t4", "atom", [(0.0, cmd("sweep", "preset:9", "cut")), (0.1, lambda app: setattr(app.scene, "cut", False))]),
        Step(9.0, "t3b", "atom", [(0.0, cmd("preset:3", "born"))]),
        Step(12.0, "t4c", "rydberg", [(0.0, spin(0.10)), (0.0, lambda app: setattr(app.scenes["atom"], "born", False))]),
        Step(10.0, "t4b", "spin", [(0.0, cmd("preset:2"))]),
        Step(9.0, "t5", "matter", [(0.0, spin(0.28)), (0.0, cmd("sys:H2O"))]),
        Step(9.0, "t6", "matter", [(0.0, cmd("sys:O2", "kind:spin")), (3.0, cmd("sys:C2H4", "kind:density", "homo")), (4.5, cmd("cut"))]),
        Step(11.0, "t7", "bloch", [(0.8, cmd("gate:H")), (3.6, cmd("gate:S")), (5.4, cmd("gate:T")), (7.2, cmd("gate:Y"))]),
        Step(12.0, "t8", "entangle", [(0.5, cmd("reset")), (1.0, cmd("h0")), (3.2, cmd("cnot")), (8.5, cmd("measure"))]),
        Step(6.0, "t9", "atom", [(0.0, cmd("preset:1")), (0.0, spin(0.2))]),
    ]


class App:
    def __init__(self, args):
        self.args = args
        pygame.init()
        pygame.font.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        width, height = (int(v) for v in args.size.lower().split("x"))
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        if args.screenshot:
            flags |= pygame.HIDDEN
        pygame.display.set_caption("Q2C Quantum Lab")
        pygame.display.set_mode((width, height), flags, vsync=0 if args.screenshot else 1)
        self.size = pygame.display.get_window_size()
        gl.glViewport(0, 0, *self.size)
        self.lang = Lang(TEXT, args.lang)
        self.programs = {"orbital": Program(orbital_fragment(), "orbital"), "bloch": Program(bloch_fragment(), "bloch"),
                         "surface": Program(surface_fragment(), "surface")}
        self.fullscreen = FullscreenPass()
        self.post = PostProcess(*self.size)
        BackendHub.preload()
        from . import spin as _spin

        import threading

        threading.Thread(target=_spin.shared_link, daemon=True).start()
        self.hud = Hud(*self.size)
        self.scenes: Dict[str, object] = {
            "atom": AtomScene(self), "spin": SpinScene(self), "matter": MatterScene(self), "bloch": BlochScene(self),
            "entangle": EntangleScene(self), "rydberg": RydbergScene(self), "tune": TuneScene(self),
        }
        self.scene = self.scenes[args.scene]
        self.scene.enter()
        self.time = 0.0
        self.veil = 0.0
        self.veil_goal = 0.0
        self.pending_scene: str | None = None
        self.running = True
        self.dragging = False
        self.tour: List[Step] = []
        self.tour_index = -1
        self.tour_clock = 0.0
        self.tour_fired: set = set()
        self.bars = 0.0
        self.toast = ("", 0.0)
        self.intro = 0.0 if (args.screenshot or args.no_intro) else 4.0
        self.fps = 0.0
        if self.intro > 0:
            self.scene.camera.dist = self.scene.camera.dist_goal * 2.4
            self.scene.camera.idle = 99.0
        for c in [c for c in args.do.split(",") if c]:
            self.run_command(c)
        if args.tour:
            self.start_tour()

    def go(self, name: str, instant: bool = False):
        if name == self.scene.key or name not in self.scenes:
            return
        if instant:
            self.scene = self.scenes[name]
            self.scene.enter()
            return
        self.pending_scene = name
        self.veil_goal = 1.0

    def run_command(self, cmd: str):
        if cmd.startswith("scene:"):
            self.go(cmd.split(":")[1], instant=True)
        elif cmd.startswith("go:"):
            self.go(cmd.split(":")[1])
        elif cmd == "tour":
            self.stop_tour() if self.tour else self.start_tour()
        elif cmd == "lang":
            self.lang.toggle()
        else:
            self.scene.command(cmd)

    def start_tour(self):
        self.scenes["matter"].warm_up(("H2O", "O2", "C2H4"))
        rydberg = self.scenes["rydberg"]
        if rydberg.packet is None:
            rydberg.build(rydberg.SHELLS[rydberg.shell_index])
        self.tour = build_tour()
        self.tour_index = -1
        self.tour_clock = 0.0
        self.advance_tour()

    def stop_tour(self):
        self.tour = []
        self.tour_index = -1
        for scene in self.scenes.values():
            scene.camera.auto_speed = 0.16

    def advance_tour(self):
        self.tour_index += 1
        if self.tour_index >= len(self.tour):
            self.stop_tour()
            return
        step = self.tour[self.tour_index]
        self.tour_clock = 0.0
        self.tour_fired = set()
        if step.scene != self.scene.key:
            self.scene = self.scenes[step.scene]
            self.scene.enter()
            self.veil = 1.0
            self.veil_goal = 0.0
        self.scene.camera.idle = 99.0

    def update_tour(self, dt: float):
        if not self.tour:
            return
        step = self.tour[self.tour_index]
        if self.scene.loading():
            return
        self.tour_clock += dt
        for i, (at, fn) in enumerate(step.acts):
            if i not in self.tour_fired and self.tour_clock >= at:
                self.tour_fired.add(i)
                fn(self)
        nxt = self.tour[self.tour_index + 1] if self.tour_index + 1 < len(self.tour) else None
        if nxt and nxt.scene != step.scene and self.tour_clock > step.duration - FADE_SECONDS:
            self.veil_goal = 1.0
        if self.tour_clock >= step.duration:
            self.advance_tour()

    def screenshot(self, path: str | None = None) -> str:
        w, h = self.size
        data = gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        image = pygame.image.fromstring(bytes(data), (w, h), "RGBA", True)
        if path is None:
            os.makedirs("screenshots", exist_ok=True)
            path = os.path.join("screenshots", time.strftime("q2c_%Y%m%d_%H%M%S.png"))
        pygame.image.save(image, path)
        return path

    def handle_event(self, ev):
        if ev.type == pygame.QUIT:
            self.running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                if self.tour:
                    self.stop_tour()
                else:
                    self.running = False
            elif ev.key == pygame.K_TAB:
                order = list(SCENE_ORDER)
                step = -1 if ev.mod & pygame.KMOD_SHIFT else 1
                self.go(order[(order.index(self.scene.key) + step) % len(order)])
            elif ev.key == pygame.K_F2:
                self.lang.toggle()
            elif ev.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()
            elif ev.key == pygame.K_F12:
                self.toast = (f"{self.lang('shot')}: {self.screenshot()}", 3.0)
            elif ev.key == pygame.K_t and not isinstance(self.scene, BlochScene):
                self.run_command("tour")
            elif ev.key == pygame.K_F5:
                self.run_command("tour")
            else:
                self.scene.on_key(ev)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            action = self.hud.hit(ev.pos)
            if action:
                self.run_command(action)
            else:
                self.dragging = True
                self.scene.camera.dragging = True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
            self.scene.camera.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self.scene.camera.drag(ev.rel[0], ev.rel[1], self.last_dt)
        elif ev.type == pygame.MOUSEWHEEL:
            self.scene.camera.zoom(ev.y)

    def update(self, dt: float):
        self.time += dt
        self.last_dt = dt
        step_v = dt / FADE_SECONDS
        if self.veil < self.veil_goal:
            self.veil = min(self.veil_goal, self.veil + step_v)
            if self.veil >= 1.0 and self.pending_scene:
                self.scene = self.scenes[self.pending_scene]
                self.scene.enter()
                self.pending_scene = None
                self.veil_goal = 0.0
        elif self.veil > self.veil_goal:
            self.veil = max(self.veil_goal, self.veil - step_v)
        goal = LETTERBOX if self.tour else 0.0
        self.bars += (goal - self.bars) * (1 - math.exp(-dt * 4.0))
        if self.intro > 0:
            self.intro = max(0.0, self.intro - dt)
        self.update_tour(dt)
        self.scene.update(dt)
        if self.toast[1] > 0:
            self.toast = (self.toast[0], self.toast[1] - dt)

    def draw_chrome(self):
        hud, tr = self.hud, self.lang
        w, h = self.size
        if not self.tour:
            hud.text("Q²C", hud.s(20), hud.s(14), 30, ACCENT, True)
            x = hud.s(92)
            for name in SCENE_ORDER:
                label = tr(self.scenes[name].tab_key)
                rect = pygame.Rect(x, hud.s(16), hud.s(len(label) * 9 + 34), hud.s(34))
                hud.button(rect, label, f"go:{name}", self.scene.key == name)
                x += rect.width + hud.s(8)
            right = w - hud.s(18)
            for label, action, width in ((tr("tour"), "tour", 150), (tr.code.upper(), "lang", 56)):
                rect = pygame.Rect(0, hud.s(16), hud.s(width), hud.s(34))
                rect.right = right
                hud.button(rect, label, action, False)
                right = rect.left - hud.s(8)
            self.scene.draw_hud(hud, tr)
            hint = tr(self.scene.hint_key)
            hud.text(hint, w / 2, h - hud.s(14), 13, DIM, anchor="bc")
            hud.text(f"{self.fps:3.0f} fps", w - hud.s(14), h - hud.s(14), 12, DIM, anchor="tr")
        else:
            bar = int(h * self.bars)
            pygame.draw.rect(hud.surface, (0, 0, 0, 255), pygame.Rect(0, 0, w, bar))
            pygame.draw.rect(hud.surface, (0, 0, 0, 255), pygame.Rect(0, h - bar, w, bar))
            step = self.tour[self.tour_index]
            fade_in = min(1.0, self.tour_clock / 0.8)
            fade_out = min(1.0, (step.duration - self.tour_clock) / 0.8)
            alpha = int(255 * max(0.0, min(fade_in, fade_out)))
            lines = hud.wrap(tr(step.caption), int(w * 0.62), 24)
            y = h - bar - hud.s(26) - len(lines) * hud.s(32)
            for line in lines:
                hud.text(line, w / 2, y, 24, INK, False, "tc", alpha=alpha)
                y += hud.s(32)
            hud.text(tr("stop") + "  [Esc]", w - hud.s(18), hud.s(8), 12, DIM, anchor="tr")
        if self.intro > 0:
            t = 1.0 - self.intro / 4.0
            alpha = int(255 * max(0.0, min(1.0, t * 3.0, (1.0 - t) * 2.5)))
            hud.text(tr("title"), w / 2, h * 0.40, 56, INK, True, "c", alpha=alpha)
            hud.text(tr("subtitle"), w / 2, h * 0.40 + hud.s(54), 22, ACCENT, False, "c", alpha=alpha)
        if self.toast[1] > 0:
            hud.text(self.toast[0], w / 2, h - hud.s(44), 14, WARM, anchor="bc")

    def frame(self, dt: float):
        w, h = pygame.display.get_window_size()
        if (w, h) != self.size:
            self.size = (w, h)
            gl.glViewport(0, 0, w, h)
            self.hud.resize(w, h)
            self.post.resize(w, h)
        self.update(dt)
        self.post.begin()
        self.scene.render()
        self.post.present(self.time, w, h)
        self.hud.begin()
        self.draw_chrome()
        self.hud.end(self.veil)

    def run(self):
        clock = pygame.time.Clock()
        if self.args.screenshot:
            deadline = time.time() + 60.0
            while self.scene.loading() and time.time() < deadline:
                self.frame(0.0)
                time.sleep(0.05)
            for _ in range(self.args.frames):
                for ev in pygame.event.get():
                    pass
                self.frame(SCREENSHOT_DT)
            pygame.display.flip()
            self.frame(0.0)
            print(self.screenshot(self.args.screenshot))
            return
        while self.running:
            dt = min(clock.tick(0) / 1000.0, 0.1)
            self.fps = self.fps * 0.9 + (1.0 / max(dt, 1e-4)) * 0.1
            for ev in pygame.event.get():
                self.handle_event(ev)
            self.frame(dt)
            pygame.display.flip()
        pygame.quit()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Q2C immersive viewer (GPU raymarched orbitals, Bloch sphere, entanglement)")
    p.add_argument("--scene", choices=SCENE_ORDER, default="atom")
    p.add_argument("--size", default="1280x720")
    p.add_argument("--lang", choices=("es", "en"), default="en")
    p.add_argument("--tour", action="store_true", help="start the cinematic tour")
    p.add_argument("--no-intro", action="store_true")
    p.add_argument("--do", default="", help="comma separated scene commands, e.g. preset:5,cut")
    p.add_argument("--screenshot", default="", help="render offscreen, save PNG and exit")
    p.add_argument("--frames", type=int, default=90, help="frames simulated before a screenshot")
    return p.parse_args(argv)


def main(argv=None):
    app = App(parse_args(argv))
    app.run()


if __name__ == "__main__":
    main()
