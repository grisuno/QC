"""2D heads-up display drawn with pygame into an RGBA surface and composited over the GL scene."""

from __future__ import annotations

import colorsys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pygame
from OpenGL import GL as gl

from .glkit import FullscreenPass, Program, make_texture
from .shaders import HUD_FRAGMENT

Color = Tuple[int, int, int]
INK: Color = (232, 240, 255)
DIM: Color = (150, 168, 200)
ACCENT: Color = (120, 200, 255)
WARM: Color = (255, 190, 110)
PANEL = (8, 14, 30, 150)
PANEL_EDGE = (110, 170, 255, 70)


class Lang:
    """Tiny bilingual string table. Keys missing in a language fall back to English."""

    def __init__(self, table: Dict[str, Dict[str, str]], code: str = "es"):
        self.table = table
        self.code = code

    def toggle(self):
        self.code = "en" if self.code == "es" else "es"

    def __call__(self, key: str, **kw) -> str:
        entry = self.table.get(key, {})
        text = entry.get(self.code) or entry.get("en") or key
        return text.format(**kw) if kw else text


class Hud:
    def __init__(self, width: int, height: int):
        self.fonts: Dict[Tuple[str, int], pygame.font.Font] = {}
        self.pass_ = FullscreenPass()
        self.prog = Program(HUD_FRAGMENT, "hud")
        self.tex = 0
        self.size = (0, 0)
        self.surface: pygame.Surface | None = None
        self.resize(width, height)
        self.regions: List[Tuple[pygame.Rect, str]] = []

    def resize(self, width: int, height: int):
        self.size = (width, height)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.tex = make_texture(width, height, gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        self.scale = height / 720.0

    def font(self, size: int, bold: bool = False) -> pygame.font.Font:
        px = max(9, int(size * self.scale))
        key = ("b" if bold else "r", px)
        if key not in self.fonts:
            path = pygame.font.match_font("dejavusans,notosans,liberationsans,arial", bold=bold)
            self.fonts[key] = pygame.font.Font(path, px)
        return self.fonts[key]

    def begin(self):
        self.surface.fill((0, 0, 0, 0))
        self.regions = []

    def s(self, v: float) -> int:
        return int(v * self.scale)

    def text(self, msg: str, x: float, y: float, size: int = 16, color: Color = INK, bold=False,
             anchor: str = "tl", alpha: int = 255, shadow: bool = True) -> pygame.Rect:
        font = self.font(size, bold)
        img = font.render(msg, True, color)
        rect = img.get_rect()
        px, py = int(x), int(y)
        if anchor == "tr":
            rect.topright = (px, py)
        elif anchor == "tc":
            rect.midtop = (px, py)
        elif anchor == "bl":
            rect.bottomleft = (px, py)
        elif anchor == "bc":
            rect.midbottom = (px, py)
        elif anchor == "c":
            rect.center = (px, py)
        else:
            rect.topleft = (px, py)
        if shadow:
            sh = font.render(msg, True, (0, 0, 0))
            sh.set_alpha(int(alpha * 0.7))
            self.surface.blit(sh, rect.move(1, 1))
        if alpha < 255:
            img.set_alpha(alpha)
        self.surface.blit(img, rect)
        return rect

    def panel(self, rect: pygame.Rect, radius: int = 12):
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, PANEL, layer.get_rect(), border_radius=self.s(radius))
        pygame.draw.rect(layer, PANEL_EDGE, layer.get_rect(), width=1, border_radius=self.s(radius))
        self.surface.blit(layer, rect.topleft)

    def bar(self, rect: pygame.Rect, frac: float, color: Color, back=(255, 255, 255, 28)):
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, back, layer.get_rect(), border_radius=rect.height // 2)
        w = max(0, int(rect.width * float(np.clip(frac, 0, 1))))
        if w > 0:
            pygame.draw.rect(layer, (*color, 255), pygame.Rect(0, 0, w, rect.height), border_radius=rect.height // 2)
        self.surface.blit(layer, rect.topleft)

    def button(self, rect: pygame.Rect, label: str, action: str, hot: bool = False):
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        fill = (60, 110, 190, 190) if hot else (20, 34, 66, 170)
        pygame.draw.rect(layer, fill, layer.get_rect(), border_radius=self.s(8))
        pygame.draw.rect(layer, PANEL_EDGE, layer.get_rect(), width=1, border_radius=self.s(8))
        self.surface.blit(layer, rect.topleft)
        self.text(label, rect.centerx, rect.centery, 14, INK, True, "c", shadow=False)
        self.regions.append((rect, action))

    def hit(self, pos) -> str | None:
        for rect, action in self.regions:
            if rect.collidepoint(pos):
                return action
        return None

    def heatmap(self, values: np.ndarray, rect: pygame.Rect, peak: float | None = None):
        peak = float(peak if peak is not None else values.max()) or 1.0
        x = np.clip(values / peak, 0.0, 1.0) ** 0.6
        rgb = np.stack([np.clip(1.6 * x, 0, 1) * 255, np.clip(2.2 * x - 0.55, 0, 1) * 255,
                        np.clip(0.55 + 1.4 * x - 2.0 * x * x, 0, 1) * 255 * (1 - 0.6 * x) + x**3 * 120], axis=-1)
        surf = pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8).swapaxes(0, 1))
        surf = pygame.transform.smoothscale(surf, rect.size)
        self.surface.blit(surf, rect.topleft)
        pygame.draw.rect(self.surface, (110, 170, 255, 120), rect, 1)

    def hue_color(self, phase: float) -> Color:
        r, g, b = colorsys.hsv_to_rgb((0.56 + phase / (2 * np.pi)) % 1.0, 0.7, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    def wrap(self, msg: str, width_px: int, size: int = 14, bold=False) -> List[str]:
        font = self.font(size, bold)
        lines, line = [], ""
        for word in msg.split():
            trial = (line + " " + word).strip()
            if font.size(trial)[0] <= width_px:
                line = trial
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    def end(self, fade: float = 0.0):
        if fade > 0.0:
            veil = pygame.Surface(self.size, pygame.SRCALPHA)
            veil.fill((0, 0, 0, int(255 * min(1.0, fade))))
            self.surface.blit(veil, (0, 0))
        data = pygame.image.tostring(self.surface, "RGBA", True)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.size[0], self.size[1], gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.prog.use()
        self.prog.i1("uTex", 1)
        self.prog.v2("uRes", self.size)
        self.pass_.draw()
        gl.glDisable(gl.GL_BLEND)
        gl.glActiveTexture(gl.GL_TEXTURE0)
