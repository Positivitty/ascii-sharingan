"""ASCII Sharingan: an eye opens, the tomoe spin up, it blinks into a
Mangekyo pinwheel, and the camera dives into the pupil: a figure crouched on a
utility pole, silhouetted against a huge full moon. Loops forever.

Run:   ./run.sh            (or: .venv/bin/python main.py)
Keys:  space = pause, q / Esc = quit
"""
import math
import os
import random
import sys
import time

import numpy as np

COLS, ROWS = 150, 60          # character grid
FONT_FAMILY, FONT_SIZE = "Menlo", 9
FPS = 30
TK_MS = 20.0                  # ms per frame we allow for canvas updates
LOOP = 36.0                   # seconds per full cycle

RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
NR = len(RAMP)
CH_DOT, CH_STAR, CH_PLUS = RAMP.index("."), RAMP.index("*"), RAMP.index("+")
CH_BAR, CH_DASH = RAMP.index("|"), RAMP.index("-")
CH_CARET, CH_V = RAMP.index("^"), RAMP.index("v")
CH_AT = RAMP.index("@")
HEX = ["#%03x" % k for k in range(4096)]

# Scene geometry (scene units: screen height = 2.0 at zoom 1)
EYE_W = 1.02      # half width of the eye opening
R_IRIS = 0.34
R_PUPIL = 0.12
R_PLANET = 0.036    # the moon
MOON_RED = False    # True = blood-red moon instead of a pale one
ZOOM_MAX = 25.0


# ---------------------------------------------------------------- helpers
def smooth(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _hash(ix, iy):
    n = ix * 374761393 + iy * 668265263
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & 0xFFFF) / 65535.0


def vnoise(x, y):
    """Smooth value noise in [0, 1]."""
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    ix, iy = x0.astype(np.int64), y0.astype(np.int64)
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a, b = _hash(ix, iy), _hash(ix + 1, iy)
    c, d = _hash(ix, iy + 1), _hash(ix + 1, iy + 1)
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy


T_SWAP = 28.9      # screen is fully black flame here; the camera cuts back out
T_AWAKEN = 8.7     # the blink that turns three tomoe into the Mangekyo


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def seg_dist(x, y, ax, ay, bx, by):
    """Distance from points (x, y) to the segment a-b."""
    vx, vy = bx - ax, by - ay
    u = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(x - ax - u * vx, y - ay - u * vy)


def taper(x, y, ax, ay, bx, by, w0, w1):
    """Mask of a stroke from a to b whose half-width goes from w0 to w1."""
    vx, vy = bx - ax, by - ay
    u = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(x - ax - u * vx, y - ay - u * vy) < w0 + (w1 - w0) * u


# "depth" fields: 1 at the middle of a shape, 0 on its edge, negative outside.
def capd(x, y, ax, ay, bx, by, w0, w1):
    vx, vy = bx - ax, by - ay
    u = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return 1 - np.hypot(x - ax - u * vx, y - ay - u * vy) / (w0 + (w1 - w0) * u + 1e-6)


def elld(x, y, cx, cy, rx, ry):
    return 1 - np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)


def boxd(x, y, x0, x1, y0, y1):
    hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    return np.minimum(np.minimum(x - x0, x1 - x) / hw, np.minimum(y - y0, y1 - y) / hh)


HAIR, SKIN, CLOTH = (60, 66, 105), (236, 198, 172), (78, 90, 130)
VEST, GUARD, DARK = (182, 188, 205), (150, 158, 182), (58, 60, 82)
STEEL, WOOD, CERAMIC, HILT = (215, 222, 240), (128, 106, 90), (196, 196, 180), (170, 44, 48)


def timeline(t):
    """-> (lid openness 0..1, zoom, gaze x, gaze y)"""
    blink = max(0.0, 1.0 - abs(t - T_AWAKEN) / 0.32)
    blink = blink * blink * (3 - 2 * blink)
    o = smooth(1.2, 4.5, t) * (1 - smooth(32.8, 35.2, t)) * (1 - blink)
    o = 0.03 + 0.97 * o
    k = smooth(10.0, 21.0, t) if t < T_SWAP else 0.0   # cut back to the eye behind the flames
    z = ZOOM_MAX ** k * (1 + 0.14 * smooth(20.0, 27.5, t) * k)
    look = smooth(4.3, 5.2, t) * (1 - smooth(6.5, 7.5, t))
    gx = 0.11 * math.sin((t - 4.3) * 1.7) * look
    gy = 0.02 * math.sin((t - 4.3) * 2.3) * look
    return o, z, gx, gy


# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self, cw, ch, seed=7):
        self.cell_x = cw / (ROWS * ch / 2.0)     # scene units per column
        self.cell_y = 2.0 / ROWS                 # scene units per row
        c = (np.arange(COLS) + 0.5 - COLS / 2.0) * self.cell_x
        r = (np.arange(ROWS) + 0.5 - ROWS / 2.0) * self.cell_y
        self.SX, self.SY = np.meshgrid(c, r)
        rng = random.Random(seed)
        self.dither = np.array(
            [[rng.uniform(-1, 1) for _ in range(COLS)] for _ in range(ROWS)])
        self.stars = []
        while len(self.stars) < 80:
            a, rad = rng.uniform(0, math.tau), R_PUPIL * 0.97 * math.sqrt(rng.random())
            if rad < R_PLANET * 1.9:
                continue
            self.stars.append((rad * math.cos(a), rad * math.sin(a),
                               rng.random() ** 1.8,          # magnitude
                               rng.uniform(0, math.tau), rng.uniform(1.5, 5.0)))
        # distant crows: (x0, y, speed, flap phase)
        self.flock = [(rng.uniform(-1, 1), rng.uniform(-0.8, 0.8) * R_PUPIL,
                       rng.choice((-1, 1)) * rng.uniform(0.02, 0.06),
                       rng.uniform(0, math.tau)) for _ in range(7)]

    def render(self, t):
        """-> (char index grid, 12-bit colour key grid), both ROWS x COLS."""
        o, z, gx, gy = timeline(t)
        X, Y = self.SX / z, self.SY / z

        # ---- eyelids
        s = np.clip(1 - (X / EYE_W) ** 2, 0.0, 1.0)
        sag = 0.16 * (1 - o) * s
        y_up = -o * 0.40 * s ** 0.8 + sag
        y_lo = o * 0.27 * s + sag
        inside = (np.abs(X) < EYE_W) & (Y > y_up) & (Y < y_lo)
        du, dl = y_up - Y, Y - y_lo          # distance above / below the lids

        # ---- skin
        n1 = vnoise(X * 2.5 + 3.0, Y * 7.0 + 1.0)
        n2 = vnoise(X * 9.0, Y * 22.0)
        bs = 0.52 + 0.28 * n1 + 0.14 * n2
        crease = 0.10 + 0.10 * o * s
        bs = bs * np.where(du > 0, 1 - 0.78 * np.exp(-((du - crease) / 0.045) ** 2), 1.0)
        bs = bs * np.where(du > 0, 1 - 0.95 * np.exp(-(du / 0.022) ** 2), 1.0)   # lash line
        bs = bs * np.where(dl > 0, 1 - 0.8 * np.exp(-(dl / 0.03) ** 2), 1.0)
        # the long crease running from the inner corner down across the cheek
        trough = seg_dist(X, Y, 0.97, 0.14, 0.30, 0.86)
        bs = bs * (1 - 0.9 * np.exp(-(trough / 0.022) ** 2))
        # ---- eyelashes: curved strokes fanning out from both lids
        lash = np.zeros_like(X)
        lift = min(1.0, max(-0.55, o * 1.6 - 0.6))    # they point down when the eye is shut
        for n in range(26):                           # upper lashes
            f = -0.94 + 1.88 * n / 25.0               # -1 = outer corner ... 1 = inner
            lx = f * EYE_W
            sh = max(0.0, 1 - f * f)
            ly = -o * 0.40 * sh ** 0.8 + 0.16 * (1 - o) * sh
            ang = -0.95 * f - 0.25                    # fan outward, sweeping to the outer side
            ln = 0.105 + 0.075 * max(0.0, -f) + 0.02 * math.sin(n * 2.4)
            mx, my = lx + 0.55 * ln * math.sin(ang), ly - 0.62 * ln * math.cos(ang) * lift
            tx, ty = lx + ln * math.sin(ang * 1.5), ly - ln * math.cos(ang * 1.5) * lift
            lash = np.maximum(lash, np.exp(-(seg_dist(X, Y, lx, ly, mx, my) / 0.016) ** 2))
            lash = np.maximum(lash, np.exp(-(seg_dist(X, Y, mx, my, tx, ty) / 0.011) ** 2))
        for n in range(12):                           # shorter lower lashes
            f = -0.90 + 1.5 * n / 11.0
            lx = f * EYE_W
            sh = max(0.0, 1 - f * f)
            ly = o * 0.27 * sh + 0.16 * (1 - o) * sh
            ang = -0.8 * f - 0.2
            ln = 0.05 + 0.03 * max(0.0, -f)
            lash = np.maximum(lash, np.exp(-(seg_dist(
                X, Y, lx, ly, lx + ln * math.sin(ang), ly + ln * math.cos(ang)) / 0.010) ** 2))
        bs = bs * (1 - 0.97 * lash)
        k = 0.34 + 0.60 * bs
        b = bs
        R, G, B = 236 * k, 196 * k, 172 * k

        # ---- blood: once the Mangekyo is open, tears of blood run down the cheek
        if t < T_AWAKEN + 0.3:
            bleed = 0.0
        else:
            bleed = smooth(9.0, 12.8, t) if t < T_SWAP else 1.0
        if bleed > 0:
            below = ~inside & (Y > y_lo - 0.005)
            for x0, reach, w0, lag in ((0.05, 0.82, 0.021, 0.0), (-0.13, 0.40, 0.013, 0.25)):
                amt = max(0.0, (bleed - lag) / (1 - lag))
                if amt <= 0:
                    continue
                tip = y_lo + reach * amt
                off = X - (x0 + 0.022 * np.sin(Y * 10 + x0 * 40) + 0.05 * (Y - 0.27))
                w = w0 * (1 + 0.25 * np.sin(Y * 23))
                e = np.maximum(1 - np.abs(off) / w,                       # the trail
                               1 - np.hypot(off, (Y - tip) * 0.75) / (w0 * 1.75))   # the drop at its tip
                run = below & (e > 0) & (Y < tip + w0 * 2.4)
                e = np.clip(e, 0, 1)
                b = np.where(run, 0.55 + 0.42 * e, b)
                R = np.where(run, 140 + 100 * e, R)
                G = np.where(run, 6 + 22 * e, G)
                B = np.where(run, 12 + 24 * e, B)

        # ---- sclera
        lid_shadow = np.exp(-np.clip(Y - y_up, 0, None) / 0.07)
        xe = np.abs(X) / EYE_W
        bw = np.clip(0.97 - 0.45 * xe ** 2 - 0.5 * lid_shadow, 0.05, 1.0)
        k = 0.45 + 0.55 * bw
        b = np.where(inside, bw, b)
        R = np.where(inside, 255 * k, R)
        G = np.where(inside, (250 - 45 * xe ** 3) * k, G)
        B = np.where(inside, (245 - 55 * xe ** 3) * k, B)

        # ---- iris
        dx, dy = X - gx, Y - gy
        d = np.hypot(dx, dy)
        rp = R_PUPIL * (1 + 0.05 * math.sin(t * 1.3))
        iris = inside & (d < R_IRIS)
        th = np.arctan2(dy, dx)
        ti = d / R_IRIS
        fib = 0.5 + 0.5 * np.sin(th * 46 + 5 * vnoise(dx * 30 + 7, dy * 30))
        fib2 = vnoise(np.cos(th) * 9 + 20, np.sin(th) * 9 + d * 14)
        bi = 0.50 + 0.20 * fib + 0.28 * fib2
        bi = bi * (1 - 0.45 * np.exp(-np.clip(Y - y_up, 0, None) / 0.08))
        mangekyo = t >= T_AWAKEN
        if not mangekyo:                       # three tomoe riding a thin ring
            rot = 0.5 * t + 5.0 * smooth(6.3, T_AWAKEN, t) ** 2
            mark = np.abs(ti - 0.58) < 0.02
            for n in range(3):
                a = rot + n * math.tau / 3
                hx, hy = 0.58 * R_IRIS * math.cos(a), 0.58 * R_IRIS * math.sin(a)
                mark |= np.hypot(dx - hx, dy - hy) < 0.135 * R_IRIS
                e = -wrap(th - a) / 0.95       # 0..1 along the trailing tail
                mark |= (e > 0) & (e < 1) & (np.abs(ti - 0.58 - 0.11 * e) < 0.125 * (1 - e))
        else:                                  # three curved blades
            rot = 0.35 * t
            mark = np.zeros_like(iris)
            for n in range(3):
                dlt = wrap(th - rot - n * math.tau / 3 - 1.3 * ti)
                mark |= (ti < 0.93) & (np.abs(dlt) < 0.62 * np.clip(1 - ti / 0.93, 0, 1) ** 0.7)
        mark |= ti > 0.90                      # heavy black rim
        bi = np.where(mark, 0.0, bi)
        k = 0.35 + 0.80 * bi
        mix = np.clip(ti, 0, 1) ** 1.5
        b = np.where(iris, bi, b)
        R = np.where(iris, (240 - 45 * mix) * k, R)
        G = np.where(iris, 24 * k, G)
        B = np.where(iris, 34 * k, B)

        if bleed > 0:                              # blood welling along the lower lid
            pool = inside & (y_lo - Y < 0.035 * min(1.0, bleed * 4)) & (np.abs(X + 0.03) < 0.26)
            b = np.where(pool, 0.85, b)
            R, G, B = np.where(pool, 205.0, R), np.where(pool, 14.0, G), np.where(pool, 22.0, B)

        # ---- pupil: a window onto a red moon
        pupil = inside & (d < rp)
        b = np.where(pupil, 0.0, b)

        u, v = dx / R_PLANET, dy / R_PLANET
        rr = u * u + v * v
        planet = pupil & (rr < 1.0)
        nz = np.sqrt(np.clip(1 - rr, 0, 1))
        lam = 0.55 + 0.45 * nz                      # full moon, lit head-on
        tex = vnoise(u * 2.2 + 10, v * 2.2 + 4)
        seas = sstep(0.45, 0.75, vnoise(u * 1.3 + 31, v * 1.3 + 17))   # dark maria
        bp = lam * (0.72 + 0.28 * tex) * (1 - 0.33 * seas)
        b = np.where(planet, 0.50 + 0.50 * bp, b)
        if MOON_RED:
            mr, mg, mb = 120 + 135 * bp ** 0.6, 8 + 40 * bp + 110 * bp ** 3, 18 + 40 * bp + 100 * bp ** 3
        else:
            mr, mg, mb = 150 + 105 * bp, 150 + 100 * bp, 140 + 95 * bp
        R, G, B = np.where(planet, mr, R), np.where(planet, mg, G), np.where(planet, mb, B)

        glow_amt = np.clip(1 - (np.sqrt(rr) - 1) / 0.38, 0, 1)
        glow = pupil & ~planet & (glow_amt > 0)
        b = np.where(glow, 0.24 * glow_amt ** 1.5, b)
        R = np.where(glow, (215 if MOON_RED else 120) * (0.4 + 0.6 * glow_amt), R)
        G = np.where(glow, (30 if MOON_RED else 135) * (0.4 + 0.6 * glow_amt), G)
        B = np.where(glow, (64 if MOON_RED else 190) * (0.4 + 0.6 * glow_amt), B)

        # ---- the pole, the wires, and the figure crouched on top (moon radii, v down)
        sway = math.sin(t * 1.7)
        TOP = 0.62                                   # top of the pole
        au = np.abs(u)
        fs = 1.8                                     # figure scale, anchored on the pole top
        fu, fv = u / fs, (v - TOP) / fs              # fv = 0 at his feet, negative = up
        afu = np.abs(fu)                             # the squat is symmetric
        grain = 0.75 + 0.25 * vnoise(u * 40, v * 3)
        # every part: (depth field, colour, brightness at its edge, brightness in its middle)
        # painted back to front; the dark edges are what separate one part from the next
        layers = [
            (np.where(v > TOP, 1 - au / 0.065, -1.0) * grain, WOOD, 0.14, 0.42),            # pole
            (boxd(u, v, -0.78, 0.78, TOP + 0.12, TOP + 0.20) * grain, WOOD, 0.14, 0.42),    # crossarms
            (boxd(u, v, -0.56, 0.56, TOP + 0.30, TOP + 0.37) * grain, WOOD, 0.14, 0.42),
            (boxd(np.abs(au - 0.70), v, -0.04, 0.04, TOP + 0.03, TOP + 0.12), CERAMIC, 0.35, 0.80),
            (boxd(np.abs(au - 0.36), v, -0.04, 0.04, TOP + 0.03, TOP + 0.12), CERAMIC, 0.35, 0.80),
            (capd(fu, fv, -0.17, -0.19, 0.25, -0.775, 0.022, 0.019), GUARD, 0.22, 0.60),    # sheath
            (capd(fu, fv, 0.25, -0.775, 0.31, -0.86, 0.02, 0.02), HILT, 0.30, 0.70),        # hilt
            (capd(fu, fv, 0.215, -0.735, 0.275, -0.775, 0.026, 0.026), DARK, 0.15, 0.35),   # guard
            (capd(fu, fv, 0.05, -0.68, 0.33, -0.56 + 0.035 * sway, 0.04, 0.004), HAIR, 0.14, 0.38),  # ponytail
            (capd(fu, fv, 0.0, -0.20, 0.0, -0.50, 0.105, 0.135), CLOTH, 0.16, 0.42),        # shirt
            (elld(fu, fv, 0.0, -0.525, 0.215, 0.06), CLOTH, 0.16, 0.42),                    # shoulders
            (capd(fu, fv, 0.0, -0.25, 0.0, -0.49, 0.072, 0.10), VEST, 0.30, 0.82),          # armour vest
            (capd(afu, fv, 0.075, -0.49, 0.105, -0.565, 0.028, 0.028), VEST, 0.30, 0.80),   # vest straps
            (elld(fu, fv, 0.0, -0.20, 0.12, 0.09), CLOTH, 0.16, 0.40),                      # hips
            (capd(afu, fv, 0.05, -0.21, 0.24, -0.30, 0.075, 0.062), CLOTH, 0.16, 0.48),     # thighs
            (capd(afu, fv, 0.24, -0.30, 0.06, -0.045, 0.058, 0.045), GUARD, 0.22, 0.68),    # shin guards
            (elld(afu, fv, 0.055, -0.02, 0.065, 0.03), DARK, 0.15, 0.36),                   # sandals
            (capd(afu, fv, 0.195, -0.52, 0.25, -0.36, 0.05, 0.042), SKIN, 0.28, 0.74),      # upper arms
            (capd(afu, fv, 0.25, -0.36, 0.16, -0.19, 0.042, 0.034), GUARD, 0.22, 0.70),     # arm guards
            (elld(afu, fv, 0.15, -0.165, 0.036, 0.036), DARK, 0.15, 0.38),                  # gloves
            (boxd(fu, fv, -0.04, 0.04, -0.635, -0.55), SKIN, 0.25, 0.55),                   # neck
            (capd(afu, fv, 0.0, -0.562, 0.085, -0.58, 0.024, 0.02), CLOTH, 0.16, 0.42),     # collar
            (elld(fu, fv, 0.0, -0.70, 0.085, 0.10), SKIN, 0.38, 0.86),                      # face
            (capd(afu, fv, 0.022, -0.672, 0.05, -0.636, 0.011, 0.011), DARK, 0.10, 0.20),   # lines under the eyes
            (capd(fu, fv, -0.018, -0.626, 0.018, -0.626, 0.008, 0.008), DARK, 0.10, 0.20),  # mouth
            (elld(fu, fv, 0.0, -0.775, 0.093, 0.05), HAIR, 0.14, 0.36),                     # hair
            (capd(afu, fv, 0.068, -0.77, 0.115, -0.585, 0.036, 0.004), HAIR, 0.14, 0.40),   # long bangs
        ]
        for hx, hy in ((-0.055, -0.84), (0.0, -0.855), (0.055, -0.84)):                     # spiky crown
            layers.append((capd(fu, fv, hx * 0.5, -0.77, hx, hy, 0.036, 0.004), HAIR, 0.14, 0.36))
        layers.append((boxd(fu, fv, -0.088, 0.088, -0.752, -0.722), CLOTH, 0.20, 0.40))     # headband
        layers.append((boxd(fu, fv, -0.045, 0.045, -0.757, -0.717), STEEL, 0.60, 0.98))     # its metal plate

        sil = np.zeros_like(pupil)
        for depth, (cr, cg, cb), lo, hi in layers:
            m = pupil & (depth > 0)
            e = np.clip(depth / 0.45, 0, 1) ** 0.8
            # dense characters everywhere; the colour (not the glyph) carries light and dark,
            # so each part reads as a solid block instead of scattered punctuation
            kc = 0.62 + 0.75 * (lo + (hi - lo) * e)
            b = np.where(m, 0.62 + 0.38 * e, b)
            R, G, B = np.where(m, cr * kc, R), np.where(m, cg * kc, G), np.where(m, cb * kc, B)
            sil |= m
        wire = (au > 0.70) & (np.abs(v - (TOP + 0.05 + 0.035 * (au - 0.70) ** 2)) < 0.03)
        wire |= (au > 0.50) & (np.abs(v - (TOP + 0.31 + 0.030 * (au - 0.50) ** 2)) < 0.03)
        wire &= pupil & ~sil
        b = np.where(wire, 0.17, b)
        R, G, B = np.where(wire, 70.0, R), np.where(wire, 80.0, G), np.where(wire, 125.0, B)
        sil |= wire
        # a bright rim where moonlight meets the silhouette, so the outline reads
        near = (np.roll(sil, 1, 0) | np.roll(sil, -1, 0) | np.roll(sil, 1, 1) | np.roll(sil, -1, 1))
        rim = planet & ~sil & near
        b = np.where(rim, 1.0, b)
        R, G, B = np.where(rim, 255.0, R), np.where(rim, 255.0, G), np.where(rim, 240.0, B)
        self._eyes = (TOP - 0.692 * fs, 0.036 * fs)  # (v of his eyes, +-u offset)

        # ---- corneal highlight
        hd = np.hypot(dx - 0.14, dy + 0.13)
        hl = sstep(0.07, 0.04, hd) * inside
        b = b + (1.0 - b) * hl
        R, G, B = R + (255 - R) * hl, G + (255 - G) * hl, B + (255 - B) * hl

        # ---- black flames: climb from the bottom until they swallow the screen,
        #      then lift away off the top to reveal the eye again
        rising = t < T_SWAP
        cov = smooth(24.8, 28.5, t) if rising else 1 - smooth(29.4, 32.2, t)
        flame = np.zeros_like(inside)
        if cov > 0:
            hw = self.SX.max()
            yb = (1 - self.SY) / 2.0                  # 0 at the bottom of the screen, 1 at the top
            if not rising:
                yb = 1 - yb
            yb = yb - 0.25 * (self.SX / hw) ** 2      # the sides run ahead of the middle
            up = t * 1.6                              # everything streams upward
            n1 = vnoise(self.SX * 2.2 + 5, self.SY * 2.0 + up)
            n2 = vnoise(self.SX * 6.0 + 9, self.SY * 3.0 + up * 1.7)
            n3 = vnoise(self.SX * 13.0, self.SY * 5.0 + up * 2.4)
            F = cov * 2.05 - 0.50 - yb + 0.55 * (n1 - 0.5) + 0.32 * (n2 - 0.5) + 0.10 * (n3 - 0.5)
            flame = F > 0
            edge = np.clip(1 - F / 0.11, 0, 1)        # 1 right on the flame's edge
            wisp = sstep(0.55, 0.85, 0.6 * n2 + 0.4 * n3)   # faint violet veins inside the black
            fb = np.maximum(edge ** 1.5, 0.30 * wisp)
            b = np.where(flame, fb, b)
            R = np.where(flame, 62 + 170 * edge ** 2 + 40 * wisp, R)
            G = np.where(flame, 30 + 150 * edge ** 3 + 12 * wisp, G)
            B = np.where(flame, 105 + 150 * edge ** 1.2 + 60 * wisp, B)

        # ---- to characters + colour keys
        lit = b > 0.035
        idx = np.clip(b * (NR - 1) + self.dither * 2.5, 1, NR - 1).astype(np.int16)
        idx = np.where(lit, idx, 0).astype(np.int16)
        q = lambda a: np.clip(a, 0, 255).astype(np.int16) >> 4
        key = (q(R) << 8) | (q(G) << 4) | q(B)
        key = np.where(lit, key, 0).astype(np.int16)

        self._stars(t, z, gx, gy, idx, key, pupil & ~planet & ~glow & ~sil & ~flame & (hl < 0.2))
        self._red_eyes(z, gx, gy, idx, key, pupil & ~flame)
        return idx, key

    def _red_eyes(self, z, gx, gy, idx, key, pupil):
        """Two glowing Sharingan dots in the silhouette's face."""
        if z < 7.0:
            return
        ev, eu = self._eyes
        row = int(ROWS / 2.0 + (ev * R_PLANET + gy) * z / self.cell_y)
        for sgn in (-1, 1):
            col = int(COLS / 2.0 + (sgn * eu * R_PLANET + gx) * z / self.cell_x)
            for cc_ in (col, col + sgn):             # each eye is two cells wide
                if 0 <= row < ROWS and 0 <= cc_ < COLS and pupil[row, cc_]:
                    idx[row, cc_], key[row, cc_] = CH_AT, 0xf12

    def _stars(self, t, z, gx, gy, idx, key, free):
        grow = min(1.0, (z - 1.0) / 6.0)
        for x, y, m, ph, sp in self.stars:
            if z < 1.25 + (1 - m) * 7.0:
                continue
            col = int(COLS / 2.0 + (x + gx) * z / self.cell_x)
            row = int(ROWS / 2.0 + (y + gy) * z / self.cell_y)
            if not (0 <= col < COLS and 0 <= row < ROWS) or not free[row, col]:
                continue
            tw = 0.7 + 0.3 * math.sin(t * sp + ph)
            size = int(round(m ** 2.2 * 2.6 * grow * tw))
            g = int(15 * min(1.0, 0.45 + 0.6 * m * tw))
            idx[row, col] = CH_PLUS if size else (CH_STAR if m > 0.45 else CH_DOT)
            key[row, col] = g * 0x111
            for n in range(1, size + 1):
                g = max(4, int(15 * (1 - n / (size + 1.0))))
                for rr_, cc_, ch in ((row - n, col, CH_BAR), (row + n, col, CH_BAR)):
                    if 0 <= rr_ < ROWS and free[rr_, cc_]:
                        idx[rr_, cc_], key[rr_, cc_] = ch, g * 0x111
            for n in range(1, 2 * size + 1):
                g = max(4, int(15 * (1 - n / (2 * size + 1.0))))
                for cc_ in (col - n, col + n):
                    if 0 <= cc_ < COLS and free[row, cc_]:
                        idx[row, cc_], key[row, cc_] = CH_DASH, g * 0x111
        if z < 3.0:
            return
        for x0, y, sp, ph in self.flock:       # distant crows drifting across
            x = ((x0 * R_PUPIL + sp * t + R_PUPIL) % (2 * R_PUPIL)) - R_PUPIL
            col = int(COLS / 2.0 + (x + gx) * z / self.cell_x)
            row = int(ROWS / 2.0 + (y + gy) * z / self.cell_y)
            if not (1 <= col < COLS - 1 and 0 <= row < ROWS):
                continue
            wing = CH_CARET if math.sin(t * 7 + ph) > 0 else CH_DASH
            for cc_, ch in ((col - 1, wing), (col, CH_V), (col + 1, wing)):
                if free[row, cc_]:
                    idx[row, cc_], key[row, cc_] = ch, 0x99b


# ---------------------------------------------------------------- app
def main():
    import tkinter as tk
    from tkinter import font as tkfont

    root = tk.Tk()
    root.title("sharingan")
    root.configure(bg="black")
    root.resizable(False, False)
    fnt = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE)
    cw, ch = fnt.measure("M"), fnt.metrics("linespace")
    cv = tk.Canvas(root, width=COLS * cw, height=ROWS * ch, bg="black",
                   highlightthickness=0)
    cv.pack()

    scene = Scene(cw, ch)
    items = [cv.create_text((c + 0.5) * cw, (r + 0.5) * ch, text=" ", font=fnt,
                            fill="#000", anchor="center")
             for r in range(ROWS) for c in range(COLS)]
    d_idx = np.zeros(ROWS * COLS, np.int16)      # what is on screen right now
    d_key = np.zeros(ROWS * COLS, np.int16)
    rng = np.random.default_rng(1)
    st = {"cost": 8e-6, "budget": 3000, "t": 0.0, "last": time.perf_counter(),
          "paused": False, "frames": 0, "began": time.perf_counter()}
    test_secs = float(os.environ.get("EYE_TEST_SECS", 0))
    configure, path = cv.tk.call, cv._w

    def tick():
        t0 = time.perf_counter()
        if not st["paused"]:
            st["t"] += t0 - st["last"]
        st["last"] = t0
        idx, key = scene.render(st["t"] % LOOP)
        idx, key = idx.ravel(), key.ravel()
        cells = np.nonzero((idx != d_idx) | (key != d_key))[0]
        if len(cells) > st["budget"]:            # too many: update a random subset
            cells = rng.choice(cells, st["budget"], replace=False)
        t1 = time.perf_counter()
        for k, i, c in zip(cells.tolist(), idx[cells].tolist(), key[cells].tolist()):
            configure(path, "itemconfigure", items[k], "-text", RAMP[i], "-fill", HEX[c])
        d_idx[cells], d_key[cells] = idx[cells], key[cells]
        if len(cells) > 50:                      # adapt the budget to this machine
            per = (time.perf_counter() - t1) / len(cells)
            st["cost"] = 0.7 * st["cost"] + 0.3 * per
            st["budget"] = int(min(12000, max(400, (TK_MS / 1000.0) / max(st["cost"], 1e-7))))
        st["frames"] += 1
        if test_secs and t0 - st["began"] > test_secs:
            print("frames/s %.1f  budget %d  cost %.1fus" % (
                st["frames"] / (t0 - st["began"]), st["budget"], st["cost"] * 1e6))
            root.destroy()
            return
        el = (time.perf_counter() - t0) * 1000
        root.after(max(1, int(1000 / FPS - el)), tick)

    root.bind("<space>", lambda e: st.update(paused=not st["paused"]))
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("q", lambda e: root.destroy())
    root.after(100, tick)
    root.mainloop()


def dump(t):
    """Print one frame as plain text (no window) - handy for debugging."""
    idx, _ = Scene(7, 13).render(t)
    print("\n".join("".join(RAMP[i] for i in row) for row in idx.tolist()))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dump":
        dump(float(sys.argv[2]))
    else:
        main()
