"""Glowing point sprites for Born-rule detections: each dot is one position measurement of the electron."""

from __future__ import annotations

import numpy as np
from OpenGL import GL as gl

from .glkit import Program

CAPACITY = 2400
LIFETIME_SECONDS = 7.0
BASE_SIZE_PX = 3.6
FLASH_RATE = 7.0

VERTEX = """#version 330 core
layout(location = 0) in vec4 aPosBirth;
uniform vec3 uEye;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uFwd;
uniform float uTanF;
uniform vec2 uRes;
uniform float uNow;
uniform float uLife;
uniform float uSize;
uniform float uDist;
out float vAge;
out float vFlash;
void main() {
    vec3 rel = aPosBirth.xyz - uEye;
    float z = dot(rel, uFwd);
    float age = uNow - aPosBirth.w;
    if (z < 0.01 || age < 0.0 || age > uLife) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        return;
    }
    vec2 ndc = vec2(dot(rel, uRight) / (z * uTanF * uRes.x / uRes.y), dot(rel, uUp) / (z * uTanF));
    gl_Position = vec4(ndc, 0.0, 1.0);
    vAge = age / uLife;
    vFlash = exp(-age * 7.0);
    float depth = clamp(uDist / z, 0.6, 1.6);
    gl_PointSize = uSize * (uRes.y / 720.0) * depth * (1.0 + 2.2 * vFlash);
}
"""

FRAGMENT = """#version 330 core
in float vAge;
in float vFlash;
uniform float uGain;
out vec4 fragColor;
void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    float core = exp(-d * d * 3.5);
    vec3 settled = vec3(0.30, 0.62, 1.0);
    vec3 hot = vec3(1.0, 0.93, 0.75);
    vec3 col = mix(settled, hot, vFlash) * core * (1.0 - vAge) * (0.55 + 3.0 * vFlash) * uGain;
    fragColor = vec4(col, 1.0);
}
"""


class PointCloud:
    """Ring buffer of timestamped positions drawn additively into the bound HDR framebuffer."""

    def __init__(self):
        self.program = Program(FRAGMENT, "points", VERTEX)
        self.data = np.zeros((CAPACITY, 4), np.float32)
        self.data[:, 3] = -1e6
        self.head = 0
        self.total = 0
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.data.nbytes, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, None)
        gl.glBindVertexArray(0)

    def clear(self):
        self.data[:, 3] = -1e6
        self.total = 0

    def add(self, positions: np.ndarray, now: float):
        for pos in positions:
            self.data[self.head, :3] = pos
            self.data[self.head, 3] = now
            self.head = (self.head + 1) % CAPACITY
        self.total += len(positions)

    def alive(self, now: float) -> np.ndarray:
        mask = (now - self.data[:, 3] >= 0) & (now - self.data[:, 3] < LIFETIME_SECONDS)
        return self.data[mask, :3]

    def draw(self, camera, size, now: float, gain: float = 1.0):
        prog = self.program
        prog.use()
        right, up, fwd = camera.basis()
        prog.v3("uEye", camera.eye())
        prog.v3("uRight", right)
        prog.v3("uUp", up)
        prog.v3("uFwd", fwd)
        prog.f1("uTanF", camera.fov_tan)
        prog.v2("uRes", size)
        prog.f1("uNow", now)
        prog.f1("uLife", LIFETIME_SECONDS)
        prog.f1("uSize", BASE_SIZE_PX)
        prog.f1("uDist", camera.dist)
        prog.f1("uGain", gain)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.data.nbytes, self.data)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, CAPACITY)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)
