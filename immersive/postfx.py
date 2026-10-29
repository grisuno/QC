"""HDR post-processing: multi-level bloom, filmic tone mapping, grading, lens fringe, vignette and grain."""

from __future__ import annotations

from typing import List

import numpy as np
from OpenGL import GL as gl

from .glkit import FullscreenPass, Program

DOWNSAMPLE = """#version 330 core
out vec4 frag;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform vec2 uTarget;
uniform float uThreshold;
uniform float uKnee;
void main() {
    vec2 uv = gl_FragCoord.xy / uTarget;
    vec2 d = uTexel * 0.5;
    vec3 c = texture(uSrc, uv + vec2(-d.x, -d.y)).rgb + texture(uSrc, uv + vec2(d.x, -d.y)).rgb
           + texture(uSrc, uv + vec2(-d.x, d.y)).rgb + texture(uSrc, uv + vec2(d.x, d.y)).rgb;
    c *= 0.25;
    float luma = max(c.r, max(c.g, c.b));
    float soft = clamp(luma - uThreshold + uKnee, 0.0, 2.0 * uKnee);
    soft = soft * soft / (4.0 * uKnee + 1e-5);
    float gain = max(soft, luma - uThreshold) / max(luma, 1e-5);
    frag = vec4(c * gain, 1.0);
}
"""

BLUR = """#version 330 core
out vec4 frag;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform vec2 uDir;
void main() {
    vec2 uv = gl_FragCoord.xy * uTexel;
    vec3 c = texture(uSrc, uv).rgb * 0.2270270270;
    c += (texture(uSrc, uv + uDir * uTexel * 1.3846153846).rgb + texture(uSrc, uv - uDir * uTexel * 1.3846153846).rgb) * 0.3162162162;
    c += (texture(uSrc, uv + uDir * uTexel * 3.2307692308).rgb + texture(uSrc, uv - uDir * uTexel * 3.2307692308).rgb) * 0.0702702703;
    frag = vec4(c, 1.0);
}
"""

COMPOSITE = """#version 330 core
out vec4 frag;
uniform sampler2D uScene;
uniform sampler2D uBloom0;
uniform sampler2D uBloom1;
uniform sampler2D uBloom2;
uniform vec2 uRes;
uniform float uTime;
uniform float uExposure;
uniform float uBloomGain;
uniform float uFringe;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 q = uv - 0.5;
    vec2 off = q * dot(q, q) * uFringe;
    vec3 hdr = vec3(texture(uScene, uv + off).r, texture(uScene, uv).g, texture(uScene, uv - off).b);
    vec3 bloom = texture(uBloom0, uv).rgb * 0.55 + texture(uBloom1, uv).rgb * 0.75 + texture(uBloom2, uv).rgb * 0.95;
    vec3 col = (hdr + bloom * uBloomGain) * uExposure;
    col = aces(col);
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col * vec3(0.93, 1.0, 1.10), col * vec3(1.07, 1.0, 0.92), smoothstep(0.18, 0.85, luma));
    col = pow(col, vec3(1.0 / 2.2));
    col *= 1.0 - dot(q, q) * 1.05;
    col += (hash12(gl_FragCoord.xy + fract(uTime) * 91.0) - 0.5) * 0.016;
    frag = vec4(max(col, 0.0), 1.0);
}
"""


class Target:
    """Colour-only framebuffer with a half-float texture."""

    def __init__(self, width: int, height: int):
        self.size = (width, height)
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.tex, 0)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"framebuffer incomplete: {status}")

    def bind(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, *self.size)

    def destroy(self):
        gl.glDeleteFramebuffers(1, [self.fbo])
        gl.glDeleteTextures(1, [self.tex])


class PostProcess:
    LEVELS = (2, 4, 8)

    def __init__(self, width: int, height: int):
        self.full = FullscreenPass()
        self.down = Program(DOWNSAMPLE, "bloom-down")
        self.blur = Program(BLUR, "bloom-blur")
        self.comp = Program(COMPOSITE, "composite")
        self.scene: Target | None = None
        self.levels: List[Target] = []
        self.temps: List[Target] = []
        self.exposure = 0.82
        self.bloom_gain = 1.0
        self.fringe = 0.012
        self.threshold = 1.1
        self.resize(width, height)

    def resize(self, width: int, height: int):
        for target in [self.scene, *self.levels, *self.temps]:
            if target is not None:
                target.destroy()
        self.size = (width, height)
        self.scene = Target(width, height)
        self.levels = [Target(max(1, width // k), max(1, height // k)) for k in self.LEVELS]
        self.temps = [Target(max(1, width // k), max(1, height // k)) for k in self.LEVELS]

    def begin(self):
        self.scene.bind()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    def _pass(self, program: Program, target: Target, source: int):
        target.bind()
        program.use()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, source)
        program.i1("uSrc", 0)
        self.full.draw()

    def present(self, time: float, width: int, height: int):
        gl.glDisable(gl.GL_BLEND)
        source = self.scene.tex
        for i, level in enumerate(self.levels):
            self.down.use()
            src = source_size(self, i)
            self.down.v2("uTexel", (1.0 / src[0], 1.0 / src[1]))
            self.down.v2("uTarget", level.size)
            self.down.f1("uThreshold", self.threshold if i == 0 else 0.0)
            self.down.f1("uKnee", 0.5)
            self._pass(self.down, level, source)
            source = level.tex
        for level, temp in zip(self.levels, self.temps):
            texel = (1.0 / level.size[0], 1.0 / level.size[1])
            self.blur.use()
            self.blur.v2("uTexel", texel)
            self.blur.v2("uDir", (1.0, 0.0))
            self._pass(self.blur, temp, level.tex)
            self.blur.use()
            self.blur.v2("uTexel", texel)
            self.blur.v2("uDir", (0.0, 1.0))
            self._pass(self.blur, level, temp.tex)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, width, height)
        self.comp.use()
        for unit, tex, name in ((0, self.scene.tex, "uScene"), (1, self.levels[0].tex, "uBloom0"),
                                (2, self.levels[1].tex, "uBloom1"), (3, self.levels[2].tex, "uBloom2")):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            self.comp.i1(name, unit)
        self.comp.v2("uRes", (width, height))
        self.comp.f1("uTime", time)
        self.comp.f1("uExposure", self.exposure)
        self.comp.f1("uBloomGain", self.bloom_gain)
        self.comp.f1("uFringe", self.fringe)
        self.full.draw()
        gl.glActiveTexture(gl.GL_TEXTURE0)


def source_size(post: PostProcess, index: int):
    """Texel grid of the texture each downsample level reads from."""
    return post.scene.size if index == 0 else post.levels[index - 1].size
