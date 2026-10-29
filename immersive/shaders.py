"""GLSL sources. The orbital shader evaluates hydrogenic wavefunctions per pixel on the GPU."""

from __future__ import annotations

from .physics import angular_constants

COMMON = """#version 330 core
out vec4 frag;
uniform vec2 uRes;
uniform vec3 uEye;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uFwd;
uniform float uTanF;
uniform float uTime;
uniform float uFade;

const float PI = 3.14159265359;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x),
                  mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y);
    float b = mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x),
                  mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y);
    return mix(a, b, f.z);
}

vec3 rayDir(vec2 uv) {
    vec2 q = uv * 2.0 - 1.0;
    q.x *= uRes.x / uRes.y;
    return normalize(uFwd + (q.x * uRight + q.y * uUp) * uTanF);
}

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec3 background(vec3 rd) {
    float g = rd.z * 0.5 + 0.5;
    vec3 col = mix(vec3(0.008, 0.010, 0.024), vec3(0.018, 0.028, 0.066), g);
    float n = vnoise(rd * 2.2 + vec3(0.0, 0.0, uTime * 0.01)) * 0.6 + vnoise(rd * 5.0) * 0.4;
    col += vec3(0.11, 0.035, 0.17) * pow(n, 3.0) * 0.6;
    col += vec3(0.015, 0.07, 0.15) * pow(vnoise(rd * 3.0 + 7.0), 3.0) * 0.55;
    vec3 p = rd * 150.0;
    vec3 id = floor(p);
    vec3 f = fract(p) - 0.5;
    float h = hash13(id);
    float star = step(0.986, h) * smoothstep(0.33, 0.0, length(f));
    star *= 0.6 + 0.4 * sin(uTime * (1.0 + h * 3.0) + h * 50.0);
    col += vec3(0.8, 0.9, 1.0) * star * 1.1;
    return col;
}

vec3 finish(vec3 col, vec2 uv) {
    return max(col, 0.0) * uFade;
}

float glowAt(vec3 P, vec3 ro, vec3 rd, float scale) {
    float t = dot(P - ro, rd);
    if (t < 0.0) return 0.0;
    vec3 q = ro + rd * t - P;
    float w = scale * scale;
    return w / (dot(q, q) + w * 0.45);
}
"""

GEOM = """
float capIntersect(vec3 ro, vec3 rd, vec3 pa, vec3 pb, float ra) {
    vec3 ba = pb - pa;
    vec3 oa = ro - pa;
    float baba = dot(ba, ba);
    float bard = dot(ba, rd);
    float baoa = dot(ba, oa);
    float rdoa = dot(rd, oa);
    float oaoa = dot(oa, oa);
    float a = baba - bard * bard;
    float b = baba * rdoa - baoa * bard;
    float c = baba * oaoa - baoa * baoa - ra * ra * baba;
    float h = b * b - a * c;
    if (h >= 0.0) {
        float t = (-b - sqrt(h)) / a;
        float y = baoa + t * bard;
        if (y > 0.0 && y < baba) return t;
        vec3 oc = (y <= 0.0) ? oa : ro - pb;
        b = dot(rd, oc);
        c = dot(oc, oc) - ra * ra;
        h = b * b - c;
        if (h > 0.0) return -b - sqrt(h);
    }
    return -1.0;
}

vec3 capNormal(vec3 p, vec3 pa, vec3 pb) {
    vec3 ba = pb - pa;
    float k = clamp(dot(p - pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    return normalize(p - (pa + ba * k));
}

float sphHit(vec3 ro, vec3 rd, vec3 ce, float ra) {
    vec3 oc = ro - ce;
    float b = dot(oc, rd);
    float h = b * b - (dot(oc, oc) - ra * ra);
    if (h < 0.0) return -1.0;
    return -b - sqrt(h);
}

vec3 lit(vec3 n, vec3 rd, vec3 c) {
    float f = max(dot(n, -rd), 0.0);
    return c * (0.30 + 0.70 * f) + c * pow(1.0 - f, 2.0) * 0.45 + vec3(1.0) * pow(f, 24.0) * 0.25;
}

"""

ORBITAL_BODY = """
uniform int uMode;
uniform int uCount;
uniform ivec3 uQN[4];
uniform vec2 uCoef[4];
uniform sampler2D uRadial;
uniform float uRMax;
uniform float uNorm;
uniform float uDensity;
uniform int uColorMode;
uniform int uCut;
uniform vec3 uCutN;
uniform float uCutD;
uniform float uReveal;
uniform float uBond;
uniform float uSign;
uniform float uMolNorm;
uniform float uRows;
uniform float uGamma;
uniform sampler3D uVol;
uniform ivec4 uSP[4];
uniform int uNAr;
uniform vec3 uArP[40];
uniform vec3 uArD[40];
uniform float uArW[40];
uniform float uArL;
uniform int uNAt;
uniform vec3 uAtP[12];
uniform vec3 uAtC[12];
uniform float uAtR[12];
uniform int uNBd;
uniform vec3 uBdA[16];
uniform vec3 uBdB[16];
uniform float uBdR;

#define STEPS 200

vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

vec2 Ylm(int l, int m, vec3 u) {
    int am = abs(m);
    float c = u.z;
    vec2 w = u.xy;
    vec2 w2 = cmul(w, w);
    vec2 w3 = cmul(w2, w);
    vec2 y = vec2(0.0);
    if (l == 0) {
        y = vec2(@K_0_0@, 0.0);
    } else if (l == 1) {
        if (am == 0) y = vec2(@K_1_0@ * c, 0.0);
        else y = @K_1_1@ * w;
    } else if (l == 2) {
        if (am == 0) y = vec2(@K_2_0@ * (3.0 * c * c - 1.0), 0.0);
        else if (am == 1) y = @K_2_1@ * c * w;
        else y = @K_2_2@ * w2;
    } else {
        if (am == 0) y = vec2(@K_3_0@ * c * (5.0 * c * c - 3.0), 0.0);
        else if (am == 1) y = @K_3_1@ * (5.0 * c * c - 1.0) * w;
        else if (am == 2) y = @K_3_2@ * c * w2;
        else y = @K_3_3@ * w3;
    }
    if (m < 0) {
        y.y = -y.y;
        if ((am & 1) == 1) y = -y;
    }
    return y;
}

float radial(int k, float r) {
    float u = r / uRMax;
    if (u >= 1.0) return 0.0;
    return texture(uRadial, vec2(u, (float(k) + 0.5) / uRows)).r;
}

vec2 psiAtomic(vec3 p) {
    float r = length(p);
    vec3 u = p / max(r, 1e-4);
    vec2 s = vec2(0.0);
    for (int k = 0; k < uCount; k++) {
        s += cmul(uCoef[k], Ylm(uQN[k].y, uQN[k].z, u)) * radial(k, r);
    }
    return s;
}

vec2 psiMol(vec3 p) {
    vec3 A = vec3(0.0, 0.0, uBond * 0.5);
    float v = (exp(-length(p - A)) + uSign * exp(-length(p + A))) * uMolNorm;
    return vec2(v, 0.0);
}

vec4 psiSpin(vec3 p) {
    float r = length(p);
    vec3 u = p / max(r, 1e-4);
    vec2 up = vec2(0.0);
    vec2 dn = vec2(0.0);
    for (int k = 0; k < uCount; k++) {
        int l = uSP[k].y;
        int mj2 = uSP[k].z;
        int sg = uSP[k].w;
        float lf = float(l);
        float mj = 0.5 * float(mj2);
        float cu;
        float cd;
        if (sg > 0) {
            cu = sqrt(max((lf + mj + 0.5) / (2.0 * lf + 1.0), 0.0));
            cd = sqrt(max((lf - mj + 0.5) / (2.0 * lf + 1.0), 0.0));
        } else {
            cu = -sqrt(max((lf - mj + 0.5) / (2.0 * lf + 1.0), 0.0));
            cd = sqrt(max((lf + mj + 0.5) / (2.0 * lf + 1.0), 0.0));
        }
        int mu = (mj2 - 1) / 2;
        int md = (mj2 + 1) / 2;
        float R = radial(k, r);
        if (abs(mu) <= l) up += cmul(uCoef[k], Ylm(l, mu, u)) * (R * cu);
        if (abs(md) <= l) dn += cmul(uCoef[k], Ylm(l, md, u)) * (R * cd);
    }
    return vec4(up, dn);
}

vec3 spinRGB(vec3 dir) {
    return pow(0.5 + 0.5 * dir, vec3(1.15)) * 1.25;
}

vec2 psiVol(vec3 p) {
    return vec2(texture(uVol, p * (0.5 / uRMax) + 0.5).r, 0.0);
}

vec2 psiAt(vec3 p) {
    if (uMode == 3) return psiVol(p);
    return uMode == 0 ? psiAtomic(p) : psiMol(p);
}

vec4 psiFull(vec3 p) {
    if (uMode == 2) return psiSpin(p);
    return vec4(psiAt(p), 0.0, 0.0);
}

float geomHit(vec3 ro, vec3 rd, out vec3 gcol) {
    float best = 1e9;
    gcol = vec3(0.0);
    if (uMode == 2) {
        for (int i = 0; i < 40; i++) {
            if (i >= uNAr) break;
            vec3 D = uArD[i];
            vec3 a = uArP[i] - D * uArL * 0.5;
            vec3 b = uArP[i] + D * uArL * 0.5;
            vec3 ac = spinRGB(D) * (0.55 + 0.8 * uArW[i]);
            float t = capIntersect(ro, rd, a, b, 0.05 * uArL);
            if (t > 0.0 && t < best) {
                best = t;
                gcol = lit(capNormal(ro + rd * t, a, b), rd, ac);
            }
            t = sphHit(ro, rd, b, 0.15 * uArL);
            if (t > 0.0 && t < best) {
                best = t;
                gcol = lit(normalize(ro + rd * t - b), rd, ac);
            }
        }
        return best > 1e8 ? -1.0 : best;
    }
    for (int i = 0; i < 12; i++) {
        if (i >= uNAt) break;
        float t = sphHit(ro, rd, uAtP[i], uAtR[i]);
        if (t > 0.0 && t < best) {
            best = t;
            gcol = lit(normalize(ro + rd * t - uAtP[i]), rd, uAtC[i]);
        }
    }
    for (int i = 0; i < 16; i++) {
        if (i >= uNBd) break;
        float t = capIntersect(ro, rd, uBdA[i], uBdB[i], max(uBdR, 0.10));
        if (t > 0.0 && t < best) {
            best = t;
            gcol = lit(capNormal(ro + rd * t, uBdA[i], uBdB[i]), rd, vec3(0.62, 0.68, 0.80));
        }
    }
    return best > 1e8 ? -1.0 : best;
}

vec3 palette(float x) {
    x = clamp(x, 0.0, 1.0);
    vec3 a = vec3(0.0, 0.0, 0.03);
    vec3 b = vec3(0.45, 0.05, 0.48);
    vec3 c = vec3(0.95, 0.35, 0.15);
    vec3 d = vec3(1.0, 0.95, 0.70);
    if (x < 0.4) return mix(a, b, x / 0.4);
    if (x < 0.75) return mix(b, c, (x - 0.4) / 0.35);
    return mix(c, d, (x - 0.75) / 0.25);
}

vec3 colorizeSpin(vec4 v, float d) {
    float dens = max(dot(v, v), 1e-12);
    vec2 cr = vec2(v.x * v.z + v.y * v.w, v.x * v.w - v.y * v.z);
    vec3 dir = vec3(2.0 * cr.x, 2.0 * cr.y, v.x * v.x + v.y * v.y - v.z * v.z - v.w * v.w) / dens;
    float s = clamp(d, 0.0, 1.0);
    return spinRGB(dir) * (0.16 + 1.6 * pow(s, 0.5)) + vec3(1.0) * pow(s, 3.0) * 0.45;
}

vec3 colorize(vec2 psi, float d) {
    float s = clamp(d, 0.0, 1.0);
    if (uColorMode == 0) {
        float ph = atan(psi.y, psi.x) / (2.0 * PI);
        vec3 base = hsv2rgb(vec3(fract(0.56 + ph), 0.78, 1.0));
        return base * (0.18 + 1.7 * pow(s, 0.5)) + vec3(1.0) * pow(s, 3.0) * 0.5;
    }
    if (uColorMode == 1) return palette(pow(s, 0.5)) * 1.9;
    return vec3(0.22, 0.58, 1.0) * (0.15 + 1.6 * pow(s, 0.5)) + vec3(0.9, 0.95, 1.0) * pow(s, 2.0) * 0.9;
}

vec3 colorizeFull(vec4 v, float d) {
    if (uMode == 2 && uColorMode == 0) return colorizeSpin(v, d);
    return colorize(v.xy, d);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec3 rd = rayDir(uv);
    vec3 ro = uEye;
    vec3 col = vec3(0.0);
    vec3 acc = vec3(0.0);
    float T = 1.0;
    vec3 gcol = vec3(0.0);
    bool hitGeom = false;

    float b = dot(ro, rd);
    float cq = dot(ro, ro) - uRMax * uRMax;
    float h = b * b - cq;
    if (h > 0.0) {
        h = sqrt(h);
        float t0 = max(-b - h, 0.0);
        float t1 = -b + h;
        if ((uMode == 3 && uNAt > 0) || (uMode == 2 && uNAr > 0)) {
            float tg = geomHit(ro, rd, gcol);
            if (tg > 0.0) {
                t1 = min(t1, tg);
                hitGeom = true;
            }
        }
        float ds = (t1 - t0) / float(STEPS);
        float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        float t = t0 + ds * jitter;
        float prevSide = 0.0;
        vec3 pPrev = vec3(0.0);
        for (int i = 0; i < STEPS; i++) {
            vec3 p = ro + rd * t;
            t += ds;
            if (uCut == 1) {
                float side = dot(p, uCutN) - uCutD;
                if (side > 0.0) {
                    prevSide = side;
                    pPrev = p;
                    continue;
                }
                if (prevSide > 0.0) {
                    float f = prevSide / (prevSide - side);
                    vec3 pp = mix(pPrev, p, f);
                    prevSide = 0.0;
                    float rr = length(pp);
                    if (rr < uRMax * 0.985) {
                        vec4 psi = psiFull(pp);
                        float rv = 1.0 - smoothstep(uReveal - 2.0, uReveal, rr);
                        float d = dot(psi, psi) * uNorm * rv;
                        float lg = log2(d + 1e-7) * 1.4;
                        float line = smoothstep(0.07, 0.0, abs(fract(lg) - 0.5) - 0.40) * step(-9.0, lg);
                        vec3 sc = colorizeFull(psi, d) * (0.55 + 0.9 * pow(clamp(d, 0.0, 1.0), 0.4));
                        sc += vec3(0.02, 0.035, 0.07) + vec3(0.6, 0.8, 1.0) * line * 0.22;
                        float rim = smoothstep(uRMax * 0.985, uRMax * 0.93, rr);
                        float sa = (0.05 + 0.88 * smoothstep(0.0, 0.05, pow(d, 0.5))) * rim;
                        acc += T * sa * sc;
                        T *= 1.0 - sa;
                    }
                }
            }
            float rr = length(p);
            vec4 psi = psiFull(p);
            float rv = 1.0 - smoothstep(uReveal - 2.0, uReveal, rr);
            float d = dot(psi, psi) * uNorm * rv;
            float od = pow(clamp(d, 0.0, 1.5), uGamma) * uDensity;
            float a = 1.0 - exp(-od * ds * 0.9);
            acc += T * a * colorizeFull(psi, d);
            T *= 1.0 - a;
            if (T < 0.01) break;
        }
    }
    vec3 bg = background(rd);
    col = acc + T * (hitGeom ? gcol : bg);
    float nuc = 0.0;
    float sc0 = length(uEye) * 0.011;
    if (uMode == 3) {
        nuc = 0.0;
    } else if (uMode == 0 || uMode == 2) {
        nuc += glowAt(vec3(0.0), ro, rd, sc0);
    } else {
        nuc += glowAt(vec3(0.0, 0.0, uBond * 0.5), ro, rd, sc0);
        nuc += glowAt(vec3(0.0, 0.0, -uBond * 0.5), ro, rd, sc0);
    }
    col += vec3(1.0, 0.92, 0.75) * nuc * 0.9 * mix(1.0, T, 0.6) * smoothstep(0.0, 1.0, uReveal);
    frag = vec4(finish(col, uv), 1.0);
}
"""

BLOCH_BODY = """
uniform int uNS;
uniform vec3 uC[2];
uniform vec3 uVec[2];
uniform vec3 uCol[2];
uniform float uLink;
uniform float uPulse;
uniform vec3 uTrail[96];
uniform int uTrailN[2];

float gridLines(vec3 n) {
    float th = acos(clamp(n.z, -1.0, 1.0));
    float ph = atan(n.y, n.x);
    float step_ = PI / 6.0;
    float aa = max(fwidth(th), 0.002) * 1.3;
    float lat = abs(fract(th / step_ + 0.5) - 0.5) * step_;
    float lon = abs(fract(ph / step_ + 0.5) - 0.5) * step_ * sin(th);
    float g = smoothstep(aa, 0.0, lat) + smoothstep(aa, 0.0, lon) * smoothstep(0.02, 0.12, sin(th));
    float eq = smoothstep(aa * 1.8, 0.0, abs(th - PI * 0.5));
    return clamp(g * 0.55 + eq * 1.2, 0.0, 1.6);
}

void sphereLayer(int i, vec3 ro, vec3 rd, inout vec3 col) {
    vec3 C = uC[i];
    vec3 lro = ro - C;
    float bq = dot(lro, rd);
    float hq = bq * bq - (dot(lro, lro) - 1.0);
    vec3 V = uVec[i];
    vec3 aCol = uCol[i];
    float purity = length(V);
    float gl = glowAt(C + V, ro, rd, 0.07 + 0.03 * uPulse);
    if (hq > 0.0) {
        float hs = sqrt(hq);
        float tin = -bq - hs;
        float tout = -bq + hs;
        vec3 pb = lro + rd * tout;
        col += vec3(0.35, 0.55, 1.0) * gridLines(normalize(pb)) * 0.20;
        float chord = max(tout - max(tin, 0.0), 0.0);
        float mixed = 1.0 - purity;
        col += aCol * mixed * chord * (0.30 + 0.08 * sin(uTime * 2.5 + float(i) * 2.0));

        float best = 1e9;
        vec3 bc = vec3(0.0);
        float tt;
        vec3 axes[3] = vec3[3](vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1));
        vec3 axc[3] = vec3[3](vec3(1.0, 0.35, 0.35), vec3(0.4, 1.0, 0.5), vec3(0.45, 0.65, 1.0));
        for (int k = 0; k < 3; k++) {
            tt = capIntersect(lro, rd, -axes[k] * 1.14, axes[k] * 1.14, 0.007);
            if (tt > 0.0 && tt < best) {
                best = tt;
                bc = axc[k] * 0.75;
            }
        }
        if (purity > 0.02) {
            tt = capIntersect(lro, rd, vec3(0.0), V, 0.022);
            if (tt > 0.0 && tt < best) {
                best = tt;
                bc = lit(capNormal(lro + rd * tt, vec3(0.0), V), rd, aCol);
            }
            tt = sphHit(lro, rd, V, 0.062);
            if (tt > 0.0 && tt < best) {
                best = tt;
                bc = lit(normalize(lro + rd * tt - V), rd, mix(aCol, vec3(1.0), 0.35));
            }
            vec3 Vp = vec3(V.xy, 0.0);
            if (abs(V.z) > 0.03) {
                tt = capIntersect(lro, rd, V, Vp, 0.005);
                if (tt > 0.0 && tt < best) {
                    best = tt;
                    bc = aCol * 0.5;
                }
                tt = sphHit(lro, rd, Vp, 0.022);
                if (tt > 0.0 && tt < best) {
                    best = tt;
                    bc = aCol * 0.8;
                }
            }
        }
        int n = uTrailN[i];
        for (int k = 0; k < 48; k++) {
            if (k >= n) break;
            float age = float(k + 1) / float(n);
            vec3 tp = uTrail[i * 48 + k];
            tt = sphHit(lro, rd, tp, 0.006 + 0.018 * age);
            if (tt > 0.0 && tt < best) {
                best = tt;
                bc = aCol * (0.25 + 1.1 * age);
            }
        }
        if (best < 1e8) col = bc;
        if (tin > 0.0) {
            vec3 pf = lro + rd * tin;
            vec3 nf = normalize(pf);
            float fr = pow(1.0 - max(dot(nf, -rd), 0.0), 3.0);
            col += vec3(0.30, 0.50, 1.0) * (0.035 + 0.55 * fr);
            col += vec3(0.5, 0.75, 1.0) * gridLines(nf) * 0.34;
            col += vec3(1.0) * pow(max(dot(reflect(rd, nf), normalize(vec3(0.4, -0.3, 0.9))), 0.0), 40.0) * 0.5;
        }
    }
    col += aCol * gl * (0.55 + 0.7 * uPulse);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec3 rd = rayDir(uv);
    vec3 ro = uEye;
    vec3 col = background(rd) * 0.8;
    int first = 0;
    int second = 1;
    if (uNS == 2 && length(uC[1] - ro) > length(uC[0] - ro)) {
        first = 1;
        second = 0;
    }
    sphereLayer(first, ro, rd, col);
    if (uNS == 2) {
        sphereLayer(second, ro, rd, col);
        vec3 dirv = uC[1] - uC[0];
        float len = length(dirv);
        dirv /= len;
        float glow = 0.0;
        vec3 lc = vec3(0.0);
        for (int k = 0; k < 32; k++) {
            float s = (float(k) + 0.5) / 32.0;
            vec3 P = uC[0] + dirv * (1.0 + s * (len - 2.0));
            float t = dot(P - ro, rd);
            if (t < 0.0) continue;
            vec3 q = ro + rd * t - P;
            float w = 0.0030 + 0.0045 * sin(s * PI);
            float g = exp(-dot(q, q) / w) * (0.50 + 0.50 * sin(s * 20.0 - uTime * 6.0));
            glow += g;
            lc += mix(uCol[0], uCol[1], s) * g;
        }
        col += lc * uLink * 0.32 + vec3(1.0) * glow * uLink * 0.05;
    }
    frag = vec4(finish(col, uv), 1.0);
}
"""


SURFACE_BODY = """
uniform sampler2DArray uPlates;
uniform int uNP;
uniform vec3 uPlateC[4];
uniform int uPlateMode[4];
uniform float uHalf;
uniform float uHeight;
uniform float uPulse;

const int PLATE_STEPS = 96;
const float TEX_EPS = 1.0 / 64.0;

float solidH(int i, vec2 uv) { return texture(uPlates, vec3(uv, float(i))).r * uHeight; }
float ghostH(int i, vec2 uv) { return texture(uPlates, vec3(uv, float(i))).g * uHeight; }
float tint(int i, vec2 uv) { return texture(uPlates, vec3(uv, float(i))).b; }

bool boxHit(vec3 ro, vec3 rd, vec3 lo, vec3 hi, out float tin, out float tout) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (lo - ro) * inv;
    vec3 t1 = (hi - ro) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    tin = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    tout = min(min(tmax.x, tmax.y), tmax.z);
    return tout > tin;
}

vec2 plateUV(int i, vec3 p) {
    return (p.xy - uPlateC[i].xy) / (2.0 * uHalf) + 0.5;
}

float surfaceHit(int i, bool ghost, vec3 ro, vec3 rd, float tin, float tout) {
    float dt = (tout - tin) / float(PLATE_STEPS);
    float t = tin;
    float prevT = t;
    vec3 p = ro + rd * t;
    float prev = p.z - uPlateC[i].z - (ghost ? ghostH(i, plateUV(i, p)) : solidH(i, plateUV(i, p)));
    for (int k = 0; k < PLATE_STEPS; k++) {
        t += dt;
        p = ro + rd * t;
        vec2 uv = plateUV(i, p);
        float f = p.z - uPlateC[i].z - (ghost ? ghostH(i, uv) : solidH(i, uv));
        if (f * prev <= 0.0) {
            float a = prevT;
            float b = t;
            for (int r = 0; r < 6; r++) {
                float m = 0.5 * (a + b);
                vec3 q = ro + rd * m;
                vec2 quv = plateUV(i, q);
                float fm = q.z - uPlateC[i].z - (ghost ? ghostH(i, quv) : solidH(i, quv));
                if (fm * prev <= 0.0) b = m; else a = m;
            }
            return 0.5 * (a + b);
        }
        prev = f;
        prevT = t;
    }
    return -1.0;
}

vec3 surfaceNormal(int i, vec2 uv) {
    float hx = solidH(i, uv + vec2(TEX_EPS, 0.0)) - solidH(i, uv - vec2(TEX_EPS, 0.0));
    float hy = solidH(i, uv + vec2(0.0, TEX_EPS)) - solidH(i, uv - vec2(0.0, TEX_EPS));
    float span = 2.0 * TEX_EPS * 2.0 * uHalf;
    return normalize(vec3(-hx / span, -hy / span, 1.0));
}

vec3 plateColour(int i, vec2 uv) {
    float v = tint(i, uv);
    if (uPlateMode[i] == 1) return hsv2rgb(vec3(v * 0.5 + 0.5, 0.72, 0.96));
    return mix(vec3(0.16, 0.50, 1.0), vec3(1.0, 0.36, 0.14), smoothstep(0.0, 0.55, v));
}

float gridWire(vec2 uv, float cells) {
    vec2 g = uv * cells;
    vec2 d = abs(fract(g - 0.5) - 0.5);
    vec2 w = fwidth(g) * 1.2;
    vec2 l = smoothstep(w, vec2(0.0), d);
    return max(l.x, l.y);
}

void plateLayer(int i, vec3 ro, vec3 rd, inout vec3 col, inout float best) {
    vec3 c = uPlateC[i];
    float tin, tout;
    if (!boxHit(ro, rd, c - vec3(uHalf, uHalf, uHeight * 1.05), c + vec3(uHalf, uHalf, uHeight * 1.05), tin, tout)) return;
    float tb = (c.z - ro.z) / rd.z;
    if (tb > 0.0 && tb < best) {
        vec3 pb = ro + rd * tb;
        vec2 buv = plateUV(i, pb);
        if (all(greaterThan(buv, vec2(0.0))) && all(lessThan(buv, vec2(1.0)))) {
            vec2 e = min(buv, 1.0 - buv);
            float edge = smoothstep(0.012, 0.0, min(e.x, e.y));
            col += vec3(0.35, 0.55, 1.0) * (0.035 + 0.30 * edge + 0.08 * gridWire(buv, 16.0));
        }
    }
    float tS = surfaceHit(i, false, ro, rd, tin, tout);
    float tG = surfaceHit(i, true, ro, rd, tin, tout);
    float limit = tS > 0.0 ? tS : 1e9;
    if (tS > 0.0 && tS < best) {
        vec3 p = ro + rd * tS;
        vec2 uv = plateUV(i, p);
        vec3 n = surfaceNormal(i, uv);
        vec3 base = plateColour(i, uv);
        float diff = max(dot(n, normalize(vec3(0.4, -0.5, 0.8))), 0.0);
        float rim = pow(1.0 - max(dot(n, -rd), 0.0), 2.5);
        vec3 shaded = base * (0.26 + 0.74 * diff) + base * rim * 0.5 + vec3(1.0) * pow(max(dot(reflect(rd, n), normalize(vec3(0.3, -0.4, 0.85))), 0.0), 36.0) * 0.30;
        shaded *= 1.0 - 0.28 * gridWire(uv, 16.0);
        col = shaded * (1.0 + 0.4 * uPulse);
        best = tS;
    }
    if (tG > 0.0 && tG < best + 1e-4) {
        vec3 p = ro + rd * tG;
        vec2 uv = plateUV(i, p);
        float wire = gridWire(uv, 16.0);
        float visible = tG < limit ? 1.0 : 0.22;
        col += vec3(0.92, 0.96, 1.0) * (wire * 0.85 + 0.035) * visible;
    }
}

void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec3 rd = rayDir(uv);
    vec3 ro = uEye;
    vec3 col = background(rd) * 0.8;
    float best = 1e9;
    for (int i = 0; i < 4; i++) {
        if (i >= uNP) break;
        plateLayer(i, ro, rd, col, best);
    }
    frag = vec4(finish(col, uv), 1.0);
}
"""

HUD_FRAGMENT = """#version 330 core
out vec4 frag;
uniform sampler2D uTex;
uniform vec2 uRes;
void main() {
    frag = texture(uTex, gl_FragCoord.xy / uRes);
}
"""


def orbital_fragment() -> str:
    body = ORBITAL_BODY
    for (l, m), value in angular_constants().items():
        body = body.replace(f"@K_{l}_{m}@", repr(float(value)))
    return COMMON + GEOM + body


def bloch_fragment() -> str:
    return COMMON + GEOM + BLOCH_BODY


def surface_fragment() -> str:
    return COMMON + GEOM + SURFACE_BODY
