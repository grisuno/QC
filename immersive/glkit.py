"""Thin OpenGL helpers: program compilation, uniforms, fullscreen pass, orbit camera."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from OpenGL import GL as gl

FULLSCREEN_VERTEX = """#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""


def _compile(kind: int, source: str, label: str) -> int:
    handle = gl.glCreateShader(kind)
    gl.glShaderSource(handle, source)
    gl.glCompileShader(handle)
    if not gl.glGetShaderiv(handle, gl.GL_COMPILE_STATUS):
        log = gl.glGetShaderInfoLog(handle)
        log = log.decode() if isinstance(log, bytes) else log
        raise RuntimeError(f"GLSL compile error in {label}:\n{log}")
    return handle


class Program:
    """Linked shader program with cached uniform locations and typed setters."""

    def __init__(self, fragment_source: str, label: str, vertex_source: str = FULLSCREEN_VERTEX):
        self.label = label
        vs = _compile(gl.GL_VERTEX_SHADER, vertex_source, label + ".vert")
        fs = _compile(gl.GL_FRAGMENT_SHADER, fragment_source, label + ".frag")
        self.handle = gl.glCreateProgram()
        gl.glAttachShader(self.handle, vs)
        gl.glAttachShader(self.handle, fs)
        gl.glLinkProgram(self.handle)
        if not gl.glGetProgramiv(self.handle, gl.GL_LINK_STATUS):
            raise RuntimeError(f"GLSL link error in {label}: {gl.glGetProgramInfoLog(self.handle)}")
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        self._loc: Dict[str, int] = {}

    def use(self):
        gl.glUseProgram(self.handle)

    def loc(self, name: str) -> int:
        if name not in self._loc:
            self._loc[name] = gl.glGetUniformLocation(self.handle, name)
        return self._loc[name]

    def f1(self, name, v):
        gl.glUniform1f(self.loc(name), float(v))

    def i1(self, name, v):
        gl.glUniform1i(self.loc(name), int(v))

    def v2(self, name, v):
        gl.glUniform2f(self.loc(name), *[float(x) for x in v])

    def v3(self, name, v):
        gl.glUniform3f(self.loc(name), *[float(x) for x in v])

    def v2a(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        gl.glUniform2fv(self.loc(name), len(arr), arr)

    def v3a(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        gl.glUniform3fv(self.loc(name), len(arr), arr)

    def v4ia(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.int32)
        gl.glUniform4iv(self.loc(name), len(arr), arr)

    def v3ia(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.int32)
        gl.glUniform3iv(self.loc(name), len(arr), arr)


class FullscreenPass:
    """Empty VAO + one oversized triangle; every effect is a fragment shader."""

    def __init__(self):
        self.vao = gl.glGenVertexArrays(1)

    def draw(self):
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)


def make_texture(width: int, height: int, internal, fmt, dtype, data=None, linear=True) -> int:
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal, width, height, 0, fmt, dtype, data)
    filt = gl.GL_LINEAR if linear else gl.GL_NEAREST
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, filt)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, filt)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    return tex


def upload_radial(tex: int, table: np.ndarray):
    rows, cols = table.shape
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, cols, rows, 0, gl.GL_RED, gl.GL_FLOAT, np.ascontiguousarray(table))


def upload_volume(tex: int, volume: np.ndarray):
    """Upload a (x, y, z) float volume as a linear R32F 3D texture (texture axes s,t,r = x,y,z)."""
    n0, n1, n2 = volume.shape
    data = np.ascontiguousarray(volume.transpose(2, 1, 0), dtype=np.float32)
    gl.glBindTexture(gl.GL_TEXTURE_3D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_R32F, n0, n1, n2, 0, gl.GL_RED, gl.GL_FLOAT, data)
    for axis in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T, gl.GL_TEXTURE_WRAP_R):
        gl.glTexParameteri(gl.GL_TEXTURE_3D, axis, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class OrbitCamera:
    """Z-up orbit camera with inertia, eased zoom and idle auto-rotation."""

    def __init__(self, dist: float = 14.0, yaw: float = 0.7, pitch: float = 0.35):
        self.dist = dist
        self.dist_goal = dist
        self.yaw = yaw
        self.pitch = pitch
        self.yaw_vel = 0.0
        self.pitch_vel = 0.0
        self.target = np.zeros(3)
        self.fov_tan = math.tan(math.radians(42) / 2)
        self.idle = 0.0
        self.auto_speed = 0.16
        self.dragging = False

    def drag(self, dx: float, dy: float, dt: float):
        self.yaw -= dx * 0.0055
        self.pitch = float(np.clip(self.pitch + dy * 0.0055, -1.45, 1.45))
        dt = max(dt, 1e-3)
        self.yaw_vel = -dx * 0.0055 / dt * 0.5
        self.idle = 0.0

    def zoom(self, steps: float):
        self.dist_goal = float(np.clip(self.dist_goal * (0.9**steps), 1.5, 400.0))
        self.idle = 0.0

    def fly_to(self, dist: float):
        self.dist_goal = dist

    def update(self, dt: float):
        self.dist += (self.dist_goal - self.dist) * (1 - math.exp(-dt * 4.0))
        if not self.dragging:
            self.idle += dt
            self.yaw += self.yaw_vel * dt
            self.yaw_vel *= math.exp(-dt * 2.2)
            if self.idle > 2.5:
                ramp = min(1.0, (self.idle - 2.5) / 2.0)
                self.yaw += self.auto_speed * ramp * dt

    def eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return self.target + self.dist * np.array([cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)])

    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fwd = _normalize(self.target - self.eye())
        right = _normalize(np.cross(fwd, np.array([0.0, 0.0, 1.0])))
        up = np.cross(right, fwd)
        return right, up, fwd

    def set_uniforms(self, prog: Program, width: int, height: int):
        right, up, fwd = self.basis()
        prog.v2("uRes", (width, height))
        prog.v3("uEye", self.eye())
        prog.v3("uRight", right)
        prog.v3("uUp", up)
        prog.v3("uFwd", fwd)
        prog.f1("uTanF", self.fov_tan)

    def project(self, world: np.ndarray, width: int, height: int):
        """Return (x, y, in_front) in pixel coordinates (origin top-left)."""
        right, up, fwd = self.basis()
        rel = np.asarray(world) - self.eye()
        z = float(np.dot(rel, fwd))
        if z < 1e-3:
            return 0.0, 0.0, False
        aspect = width / height
        x = float(np.dot(rel, right)) / (z * self.fov_tan * aspect)
        y = float(np.dot(rel, up)) / (z * self.fov_tan)
        return (x * 0.5 + 0.5) * width, (0.5 - y * 0.5) * height, True
