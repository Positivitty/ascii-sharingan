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
PHOTO = "ref/itachi_0.jpeg"   # picture to convert to characters for the scene in the pupil (None = drawn portrait)
PHOTO_CX, PHOTO_CY, PHOTO_HALF_H = 273.0, 170.0, 118.0   # centre of the final framing, and half its height, in picture pixels
PHOTO_FACE = (273.0, 190.0, 60.0, 96.0)                   # his face in the picture: centre x, y and half-width, half-height
PHOTO_EYE_ZOOM = 1.9                                      # how far the camera pushes in on his eyes
PHOTO_BG_DIM = 0.80                                       # 0 = the painting's full background, 1 = black
PHOTO_RAIN = 0.30                                         # share of columns with a raindrop (0 = no rain)
PHOTO_EYES_DRAW = False                                   # False = leave his eyes exactly as painted
PHOTO_EYES_OPEN = True                                    # False = his eyes stay shut and simply start to bleed
PHOTO_EYES = ((246.0, 186.0), (303.0, 186.0))             # his eyes, in picture pixels
PORTRAIT = True     # the scene in the pupil is a close-up of him, crows pouring from his mouth
INK = True          # the scene in the pupil is an ink-wash poster (red sun, cloaked figure, crow, pagoda)
FLAT = False        # poster look for the pole scene: flat red moon, black silhouette, white clouds, navy sky
ZOOM_MAX = 19.0 if PORTRAIT else 16.0 if INK else 20.0 if FLAT else 25.0


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


def cloudd(u, v, cx, cy, s):
    """A stylised cloud: flat base, three puffs, a long tail, and a curl cut into it."""
    x, y = (u - cx) / s, (v - cy) / abs(s)
    d = np.maximum.reduce([elld(x, y, 0.0, 0.0, 1.0, 0.20), elld(x, y, -0.45, -0.13, 0.30, 0.26),
                           elld(x, y, -0.05, -0.25, 0.36, 0.33), elld(x, y, 0.38, -0.15, 0.28, 0.24),
                           capd(x, y, 0.5, 0.06, 1.65, -0.03, 0.13, 0.01)])
    m = (d > 0) & (y < 0.13)
    rr, ang = np.hypot(x + 0.05, y + 0.18), np.arctan2(y + 0.18, x + 0.05)
    return m & ~((np.abs(rr - 0.16) < 0.04) & (ang > -2.7) & (ang < 1.3))


HAIR, SKIN, CLOTH = (60, 66, 105), (236, 198, 172), (78, 90, 130)
VEST, GUARD, DARK = (182, 188, 205), (150, 158, 182), (58, 60, 82)
STEEL, WOOD, CERAMIC, HILT = (215, 222, 240), (128, 106, 90), (196, 196, 180), (170, 44, 48)


def timeline(t):
    """-> (lid openness 0..1, zoom, gaze x, gaze y)"""
    blink = max(0.0, 1.0 - abs(t - T_AWAKEN) / 0.32)
    blink = blink * blink * (3 - 2 * blink)
    o = smooth(1.2, 4.5, t) * (1 - smooth(32.8, 35.2, t)) * (1 - blink)
    o = 0.03 + 0.97 * o
    if PHOTO:                                          # arrive sooner and then hold dead still
        k = smooth(10.0, 18.5, t) if t < T_SWAP else 0.0
        z = ZOOM_MAX ** k
    else:
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
        r2 = random.Random(seed + 11)
        r3 = random.Random(seed + 23)
        self.birds = [(r3.uniform(-1.9, 1.9), r3.gauss(-0.22, 0.2), r3.uniform(0.05, 0.14),
                       r3.uniform(0, math.tau), 0) for _ in range(22)]
        self.petals = [(r3.uniform(-1.9, 1.9), r3.uniform(-0.7, 0.7), r3.uniform(0.6, 1.4),
                        r3.uniform(0, math.tau), 0) for _ in range(16)]
        self.img = None
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PHOTO) if PHOTO else None
        if path and os.path.exists(path):
            from PIL import Image, ImageFilter, ImageOps
            im = Image.open(path).convert("RGB")
            im = im.filter(ImageFilter.UnsharpMask(radius=14, percent=70, threshold=0))   # local contrast
            im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=0))    # edges
            raw = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
            lum = raw @ np.array([0.30, 0.59, 0.11], np.float32)
            yy, xx = np.mgrid[0:raw.shape[0], 0:raw.shape[1]]
            in_face = np.hypot((xx - PHOTO_FACE[0]) / PHOTO_FACE[2], (yy - PHOTO_FACE[1]) / PHOTO_FACE[3]) < 1.0
            mist = (lum > 0.30) & (raw[..., 2] - raw[..., 0] > 0.012) & ~in_face     # the pale blue backdrop
            fg = Image.fromarray(((~mist) * 255).astype(np.uint8))
            fg = fg.filter(ImageFilter.MedianFilter(9)).filter(ImageFilter.MinFilter(7)).filter(ImageFilter.MaxFilter(7))
            fg = fg.filter(ImageFilter.GaussianBlur(1.2))
            self.img = np.dstack([np.asarray(im, np.float32) / 255.0, np.asarray(fg, np.float32) / 255.0])
        r4 = random.Random(seed + 31)
        self.bcrows = [(r4.uniform(-2.2, 2.2), r4.uniform(-0.85, 0.30), r4.uniform(0.10, 0.22),
                        r4.uniform(0, math.tau), r4.choice((-1, -1, 1))) for _ in range(6)]   # x, y, size, phase, heading
        self.mcrows = [(20.4 + 0.62 * i, r4.uniform(0, math.tau), r4.uniform(0, math.tau), r4.uniform(0.8, 1.3))
                       for i in range(12)]            # (birth, heading, flap phase, speed) - out of his mouth
        self.murder = []                         # (u0, v0, du, dv, phase, big) in moon radii
        for i in range(56):
            if i < 34:                           # the main swarm, streaming past the moon
                u0, v0 = r2.gauss(0.9, 0.75), r2.gauss(-0.45, 0.40)
            else:                                # stragglers everywhere
                u0, v0 = r2.uniform(-2.6, 2.6), r2.uniform(-1.15, 0.55)
            self.murder.append((u0, v0, r2.uniform(-0.10, -0.03), r2.uniform(-0.01, 0.01),
                                r2.uniform(0, math.tau), r2.random() < 0.65))
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
        if INK:
            b, R, G, B = (self._photo if self.img is not None else self._portrait if PORTRAIT else self._ink)(t, z, pupil, u, v, b, R, G, B)
            planet = glow = np.zeros_like(pupil)
            sil = pupil
        else:
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

            cloud_back = cloud_front = np.zeros_like(pupil)
            if FLAT:
                # flat poster colours: a solid red moon in a navy sky (the sky fades in with the zoom)
                b = np.where(planet, 0.80 + 0.18 * tex, b)
                R, G, B = (np.where(planet, 214 + 22 * tex, R), np.where(planet, 52 + 14 * tex, G),
                           np.where(planet, 54 + 12 * tex, B))
                zc = smooth(2.0, 7.0, z)
                sky = pupil & ~planet
                halo = np.clip(1 - (np.sqrt(rr) - 1) / 0.5, 0, 1) ** 2
                b = np.where(sky, zc * (0.25 + 0.14 * halo), b)
                R, G, B = (np.where(sky, 44 + 110 * halo, R), np.where(sky, 64 + 4 * halo, G),
                           np.where(sky, 96 - 10 * halo, B))
                glow = np.zeros_like(pupil)
                if z > 3.0:
                    dr = 0.055 * (t - 22.0)                  # the clouds drift, each at its own pace
                    for cx, cy, cs, sp in ((-1.05, -0.74, 0.62, 1.0), (1.75, -0.10, -0.50, -0.7),
                                           (-1.75, 0.38, 0.50, 1.3), (1.9, -0.95, 0.40, -1.1)):
                        cloud_back = cloud_back | cloudd(u, v, cx + sp * dr, cy, cs)
                    cloud_front = cloudd(u, v, 0.75 - 0.8 * dr, 1.17, -0.62) & pupil
                    cloud_back = cloud_back & pupil
                    b = np.where(cloud_back, 0.93, b)
                    R, G, B = (np.where(cloud_back, 222.0, R), np.where(cloud_back, 236.0, G),
                               np.where(cloud_back, 230.0, B))
            else:
                glow_amt = np.clip(1 - (np.sqrt(rr) - 1) / 0.38, 0, 1)
                glow = pupil & ~planet & (glow_amt > 0)
                b = np.where(glow, 0.24 * glow_amt ** 1.5, b)
                R = np.where(glow, (215 if MOON_RED else 120) * (0.4 + 0.6 * glow_amt), R)
                G = np.where(glow, (30 if MOON_RED else 135) * (0.4 + 0.6 * glow_amt), G)
                B = np.where(glow, (64 if MOON_RED else 190) * (0.4 + 0.6 * glow_amt), B)

            # ---- the pole, the wires, and the figure crouched on top (moon radii, v down)
            sway = math.sin(t * 1.7)
            TOP = 0.70 if FLAT else 0.62                 # top of the pole
            au = np.abs(u)
            fs = (1.4 if FLAT else 1.8)                                     # figure scale, anchored on the pole top
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

            if FLAT:
                arm = TOP + 0.15 + 0.035 * u                 # the crossarm leans a little
                flat = [np.where(v > TOP, 1 - au / 0.11, -1.0),
                        np.where((au < 1.05) & (np.abs(v - arm) < 0.05), 1.0, -1.0),
                        boxd(u, v - arm, 0.66, 0.74, -0.15, 0.0), elld(u, v - arm, 0.70, -0.16, 0.055, 0.045),
                        boxd(u, v - arm, 0.84, 0.92, -0.17, 0.0), elld(u, v - arm, 0.88, -0.18, 0.06, 0.05),
                        # a crow perched on the crossarm
                        capd(u, v - arm, -0.80, -0.20, -0.66, -0.10, 0.085, 0.045),
                        elld(u, v - arm, -0.84, -0.30, 0.055, 0.055),
                        capd(u, v - arm, -0.88, -0.30, -0.97, -0.275, 0.022, 0.004),
                        capd(u, v - arm, -0.68, -0.10, -0.55, -0.02, 0.04, 0.012),
                        boxd(u, v - arm, -0.775, -0.755, -0.12, 0.0)]
                sil = np.zeros_like(pupil)
                for depth in flat + [l[0] for l in layers[5:]]:      # skip the old pole hardware
                    sil |= pupil & (depth > 0)
                wire = (au > 0.2) & (np.abs(v - (arm + 0.05 + 0.55 * np.maximum(au - 0.2, 0) ** 1.6 * (u < 0) + 0.9 * np.maximum(u - 0.2, 0) ** 2 * (u > 0))) < 0.022)
                sil |= wire & pupil & ~planet
                sil &= ~cloud_front
                off = sil & ~planet                          # off the moon he would vanish into the night,
                b = np.where(sil, np.where(planet, 0.0, 0.78), b)     # so the pole is a solid dark slate there
                R, G, B = np.where(off, 50.0, R), np.where(off, 62.0, G), np.where(off, 92.0, B)
                b = np.where(cloud_front, 0.93, b)
                R, G, B = (np.where(cloud_front, 222.0, R), np.where(cloud_front, 236.0, G),
                           np.where(cloud_front, 230.0, B))
                sil |= cloud_front | cloud_back
            else:
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
            self._planet = planet
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
        if INK:
            (self._portrait_fx if PORTRAIT else self._ink_fx)(t, z, gx, gy, idx, key, self._paper & ~flame)
            return idx, key
        self._red_eyes(z, gx, gy, idx, key, pupil & ~flame)
        if FLAT:
            self._crows(t, z, gx, gy, idx, key, pupil & ~sil & ~flame)
        return idx, key

    def _photo(self, t, z, pupil, u, v, b, R, G, B):
        """The scene in the pupil is a picture, converted to characters cell by cell:
        how dense the glyph is follows the brightness, its colour follows the picture.
        Everything holds still except his hair; then the camera pushes in on his closed
        eyes and they open on the Mangekyo."""
        Hh = 1.0 / (R_PLANET * ZOOM_MAX)
        X, Y = u / Hh, v / Hh                          # screen-fitted: Y = -1..1 at full zoom
        zc = smooth(2.5, 8.0, z)
        img = self.img
        ih, iw = img.shape[:2]
        push = smooth(20.0, 23.2, t) if t < T_SWAP else 0.0                 # the second zoom, onto his eyes
        ecx = (PHOTO_EYES[0][0] + PHOTO_EYES[1][0]) / 2.0
        ecy = PHOTO_EYES[0][1] + 16.0
        half = PHOTO_HALF_H / (1 + (PHOTO_EYE_ZOOM - 1) * push)
        fx = PHOTO_CX + (ecx - PHOTO_CX) * push + X * half                  # picture pixels
        fy = PHOTO_CY + (ecy - PHOTO_CY) * push + Y * half
        px_, py_ = fx, fy                                                   # unwarped, for drawing the eyes

        # his hair stirs: a slow ripple, strongest at the loose ends, none on his face
        fd = np.hypot((fx - PHOTO_FACE[0]) / PHOTO_FACE[2], (fy - PHOTO_FACE[1]) / PHOTO_FACE[3])
        hd = np.hypot((fx - PHOTO_FACE[0]) / 125.0, (fy - 175.0) / 150.0)
        loose = sstep(0.85, 1.35, fd) * sstep(1.05, 0.75, hd) * sstep(60.0, 150.0, fy)
        fx = fx + 2.6 * loose * np.sin(fy * 0.060 - t * 1.5 + 0.02 * fx)
        fy = fy + 0.9 * loose * np.sin(fx * 0.045 + t * 1.1)

        fx, fy = np.clip(fx, -1, iw), np.clip(fy, -1, ih)
        inb = (fx >= 0) & (fx <= iw - 1) & (fy >= 0) & (fy <= ih - 1)
        x0 = np.clip(np.floor(fx).astype(np.int32), 0, iw - 2)
        y0 = np.clip(np.floor(fy).astype(np.int32), 0, ih - 2)
        ax, ay = np.clip(fx - x0, 0, 1)[..., None], np.clip(fy - y0, 0, 1)[..., None]
        smp = (img[y0, x0] * (1 - ax) + img[y0, x0 + 1] * ax) * (1 - ay) \
            + (img[y0 + 1, x0] * (1 - ax) + img[y0 + 1, x0 + 1] * ax) * ay
        c = smp[..., :3]
        lum = c @ np.array([0.30, 0.59, 0.11], np.float32)
        tone = np.clip((lum - 0.06) / 0.50, 0, 1) ** 0.85                   # stretch the contrast
        sat = np.clip(lum[..., None] + (c - lum[..., None]) * 1.9, 0, 1)    # and the colour
        colr = np.clip(sat * (0.60 + 0.70 * tone[..., None]) / np.maximum(lum[..., None], 0.16) * 0.60, 0, 1) * 255
        spot = sstep(1.0, 0.55, fd)
        haze = sstep(0.025, 0.085, c[..., 2] - c[..., 0])                   # the blue mist behind him...
        tone = tone * (1 - PHOTO_BG_DIM * haze * (1 - spot))                # ...drops back to a faint haze
        tone = np.clip(tone * (1 + 0.30 * spot), 0, 1)                      # and his face comes forward
        red = sstep(0.015, 0.07, c[..., 0] - np.maximum(c[..., 1], c[..., 2]))   # the collar's lining
        tone = np.maximum(tone, 0.92 * red)
        pv = np.where(inb, 0.05 + 0.95 * tone, 0.0)
        pr = colr[..., 0] + (236 - colr[..., 0]) * red
        pg = colr[..., 1] + (26 - colr[..., 1]) * red
        pb = colr[..., 2] + (38 - colr[..., 2]) * red

        # ---- rain: thin slanted streaks falling in front of everything but his face
        if PHOTO_RAIN:
            ci = np.floor((X + 0.12 * Y) * 46.0)
            h1 = (np.sin(ci * 127.1) * 43758.5453) % 1.0
            h2 = (np.sin(ci * 311.7) * 24634.6345) % 1.0
            drop = (((Y * 0.55 - t * (1.5 + 0.9 * h2) + h1 * 7.0) % 1.0) < 0.16) & (h2 < PHOTO_RAIN) \
                & (spot < 0.35) & inb
            pv = np.where(drop, 0.395, pv)
            pr, pg, pb = np.where(drop, 195.0, pr), np.where(drop, 212.0, pg), np.where(drop, 255.0, pb)

        # ---- his eyes: shut until the camera is close, then they open on the Mangekyo.
        #      They are only ever drawn on skin, so the bangs still hang over them.
        o = (smooth(22.9, 23.9, t) if t < T_SWAP else 1.0) if PHOTO_EYES_OPEN else 0.0
        spin = 0.5 * t
        skin = sstep(0.17, 0.25, lum)                                       # 0 on his hair, 1 on his face
        c2 = img[np.clip(y0 + 8, 0, ih - 1), x0, :3]                        # skin from just under each eye
        l2 = c2 @ np.array([0.30, 0.59, 0.11], np.float32)
        t2 = np.clip(np.clip((l2 - 0.06) / 0.50, 0, 1) ** 0.85 * 1.3, 0, 1)
        s2 = np.clip(l2[..., None] + (c2 - l2[..., None]) * 1.9, 0, 1)
        k2 = np.clip(s2 * (0.60 + 0.70 * t2[..., None]) / np.maximum(l2[..., None], 0.16) * 0.60, 0, 1) * 255
        IR = 2.9                                                            # iris radius, picture pixels
        for n, (ex0, ey0) in enumerate(PHOTO_EYES):
            dx, dy = px_ - ex0, py_ - ey0
            # the painting's own red eye is the mask, so the shape and the bangs over it are the artist's
            eye = (red > 0.25) & (np.abs(dx) < 13) & (np.abs(dy) < 7) & PHOTO_EYES_DRAW
            opened = eye & (np.abs(dy) < 4.5 * o) if o > 0 else np.zeros_like(eye)
            shut = eye & ~opened
            pv = np.where(shut, (0.05 + 0.90 * t2) * 0.85, pv)              # the closed lid: his own skin, in shade
            pr, pg, pb = np.where(shut, k2[..., 0], pr), np.where(shut, k2[..., 1], pg), np.where(shut, k2[..., 2], pb)
            lash = eye & (np.abs(dy - 1.6 * (1 - o) - 4.5 * o) < 0.75) & (o < 1)
            pv = np.where(lash, 0.0, pv)
            if o > 0:
                ir = np.hypot(dx, dy)
                pv = np.where(opened, 0.62, pv)                             # dark red around the iris
                pr, pg, pb = np.where(opened, 150.0, pr), np.where(opened, 14.0, pg), np.where(opened, 24.0, pb)
                iris = opened & (ir < IR)
                pv = np.where(iris, 1.0, pv)
                pr, pg, pb = np.where(iris, 255.0, pr), np.where(iris, 30.0, pg), np.where(iris, 40.0, pb)
                ang = np.arctan2(dy, dx) + spin
                blade = (ir > 0.9) & (ir < IR - 0.3) & (((ang * 3 / math.tau + 0.42 * ir / IR) % 1.0) < 0.33)
                pv = np.where(iris & (blade | (ir < 0.9)), 0.0, pv)         # three curved blades and the pupil
            if t > 22.4 and t < T_SWAP:                                     # blood runs from both eyes
                reach = (52.0, 40.0)[n] * smooth(22.4, 26.8, t)
                off = (-2.5, 3.0)[n]
                path = dx - off - 1.4 * np.sin(dy * 0.11 + n) - 0.03 * dy * (1, -1)[n]
                wid = 2.1 - 0.9 * np.clip(dy / 50.0, 0, 1)
                tear = (lum > 0.24) & (np.abs(path) < wid) & (dy > 2.0) & (dy < 2.0 + reach)
                bead = np.hypot(path, dy - 2.0 - reach) < 2.0 * (reach > 4)   # the drop at the head of the trail
                tear |= (lum > 0.24) & bead
                core = np.abs(path) < 0.6 * wid
                pv = np.where(tear, np.where(core, 1.0, 0.80), pv)
                pr = np.where(tear, np.where(core, 235.0, 165.0), pr)
                pg, pb = np.where(tear, 10.0, pg), np.where(tear, 20.0, pb)

        b = np.where(pupil, pv * zc, b)
        R, G, B = np.where(pupil, pr, R), np.where(pupil, pg, G), np.where(pupil, pb, B)
        self._paper = np.zeros_like(pupil)
        self._Hh = Hh
        return b, R, G, B

    def _portrait(self, t, z, pupil, u, v, b, R, G, B):
        """Close-up of him: hair, slashed headband, red eyes, high red-lined collar - and the
        crows that pour out of his mouth. X, Y are design units; the face is centred near (0, -0.15)."""
        Hh = 1.0 / (R_PLANET * ZOOM_MAX * 1.14)
        SC = 3.1                                      # how close the camera ends up
        X, Y = u / Hh / SC, v / Hh / SC - 0.075
        aX = np.abs(X)
        zc = smooth(2.5, 8.0, z)

        # misty blue-grey background, drifting
        mist = 0.6 * vnoise(X * 1.1, Y * 1.4 + 3) + 0.4 * vnoise(X * 3.0, Y * 3.5 + 8)
        trunks = sstep(0.72, 0.9, vnoise(X * 7 + 40, 0.5 + Y * 0.15)) * np.clip(0.5 - Y, 0, 1)
        pv = 0.08 + 0.22 * mist + 0.12 * trunks
        pr, pg, pb = 58 + 70 * mist + 50 * trunks, 64 + 70 * mist + 50 * trunks, 96 + 72 * mist + 44 * trunks
        bg = np.ones_like(pupil)

        def put(m, r, g, bl, dens):
            nonlocal pr, pg, pb, pv, bg
            pr, pg, pb = np.where(m, r, pr), np.where(m, g, pg), np.where(m, bl, pb)
            pv = np.where(m, dens, pv)
            bg = bg & ~m

        strands = vnoise(X * 36 + Y * 5, Y * 2.2 + 1)
        fray = vnoise(X * 22, Y * 7 + 4) - 0.5

        def hair(depth):
            m = depth + 0.22 * fray > 0
            sheen = sstep(0.5, 0.9, strands) * np.clip(1 - np.abs(Y + 0.47) / 0.20, 0.12, 1)
            put(m, 26 + 28 * strands + 120 * sheen, 28 + 32 * strands + 130 * sheen,
                46 + 52 * strands + 160 * sheen, 0.30 + 0.22 * strands + 0.40 * sheen)

        # hair behind the head
        hair(elld(X, Y, 0.0, -0.33, 0.36, 0.40))

        # cloak and shoulders
        fold = vnoise(X * 5 + 2, Y * 2.2)
        cloak = (Y > 0.40 + 0.13 * aX + 0.5 * np.clip(aX - 1.0, 0, 1) ** 2) & (aX < 1.28)
        put(cloak, 20 + 34 * fold, 22 + 38 * fold, 38 + 80 * fold, 0.28 + 0.30 * fold)
        cx_, cy_ = (X + 0.86) / 0.13, (Y - 0.63) / 0.13                       # a red cloud on the cloak
        cl = np.maximum.reduce([elld(cx_, cy_, 0, 0.15, 1.0, 0.42), elld(cx_, cy_, -0.45, -0.2, 0.5, 0.45),
                                elld(cx_, cy_, 0.2, -0.38, 0.55, 0.5), elld(cx_, cy_, 0.65, -0.05, 0.4, 0.38)])
        put(cloak & (cl > -0.14), 232, 228, 222, 0.9)
        put(cloak & (cl > 0.06), 206, 40, 44, 0.9)

        # neck, chest, mesh shirt, necklace
        inner = 0.13 + 0.27 * np.clip((0.75 - Y) / 0.75, 0, 1)                # the collar's inner edge
        nsh = 0.50 + 0.40 * np.clip((Y - 0.27) / 0.22, 0, 1)                  # shadow under the chin
        chest = (Y > 0.18) & (aX < np.where(Y < 0.50, 0.15 + 0.06 * np.clip((Y - 0.3) / 0.2, 0, 1), inner))
        put(chest, 214 * nsh, 192 * nsh, 186 * nsh, 0.42 + 0.45 * nsh)
        mesh = chest & (Y > 0.585 + 0.5 * X * X)
        hatch = (((X + Y) * 30) % 1.0 < 0.24) | (((X - Y) * 30) % 1.0 < 0.24)
        put(mesh, np.where(hatch, 26, 78), np.where(hatch, 32, 100), np.where(hatch, 40, 108),
            np.where(hatch, 0.25, 0.62))
        put(chest & (np.abs(Y - (0.475 + 1.1 * X * X)) < 0.011) & (aX < 0.19), 60, 60, 74, 0.5)
        for bx in (-0.075, 0.0, 0.075):
            put(np.hypot(X - bx, Y - 0.482 - 1.1 * bx * bx) < 0.022, 205, 212, 228, 0.95)

        # high collar: black outside, red lining inside
        collar = (aX > inner) & (aX < 0.68) & (Y > 0.0 + 0.9 * np.abs(aX - 0.50))
        cshade = 0.55 + 0.45 * vnoise(X * 6 + 9, Y * 3)
        put(collar, 24 + 30 * cshade, 26 + 32 * cshade, 44 + 60 * cshade, 0.35 + 0.25 * cshade)
        lining = collar & (aX < inner + 0.115)
        put(lining, 120 + 100 * cshade, 22 + 22 * cshade, 28 + 22 * cshade, 0.62 + 0.33 * cshade)

        # face: long and angular - flat cheeks, a straight jaw line, a narrow chin
        hw = np.interp(Y, [-0.60, -0.10, 0.13, 0.30], [0.235, 0.250, 0.165, 0.035])
        fd = 1 - aX / hw
        face = (fd > 0) & (Y > -0.60) & (Y < 0.30)
        sh = 0.52 + 0.48 * np.clip(fd * 3.0, 0, 1)
        sh = sh * (1 - 0.40 * sstep(-0.02, 0.20, X))                            # one side of the face in shadow
        sh = sh * (0.55 + 0.45 * sstep(-0.285, -0.17, Y))                       # the headband's shadow over the eyes
        sh = sh * (1 - 0.18 * sstep(0.10, 0.0, np.abs(Y - 0.02 - 0.9 * (aX - 0.12))) * (aX > 0.10))   # cheek hollow
        put(face, 214 * sh, 206 * sh, 212 * sh * (1 + 0.20 * (1 - sh)), 0.26 + 0.52 * sh)

        put(face & (capd(aX, Y, 0.035, -0.213, 0.215, -0.247, 0.010, 0.004) > 0), 0, 0, 0, 0.0)        # brows: low, stern
        ex = aX - 0.118
        ey = Y + 0.165 + 0.12 * ex                                              # the eyes slant down toward the nose
        almond = ((ex / 0.098) ** 2 + (ey / 0.040) ** 2 < 1) & (ey > -0.015)    # heavy, half-closed lids
        put(face & (ey > -0.038) & (ey <= -0.013) & (ex > -0.108) & (ex < 0.118), 0, 0, 0, 0.0)        # upper lash line
        put(face & (capd(ex, ey, 0.10, -0.025, 0.145, -0.045, 0.010, 0.003) > 0), 0, 0, 0, 0.0)        # its outer flick
        put(almond, 196, 200, 216, 0.80)
        ir = np.hypot(ex + 0.004, ey - 0.002)
        put(almond & (ir < 0.036), 232, 20, 34, 1.0)                            # the Sharingan, half under the lid
        put(almond & (np.abs(ir - 0.022) < 0.0045), 96, 0, 8, 0.55)
        put(almond & (ir < 0.011), 0, 0, 0, 0.0)
        put(face & (capd(aX, Y, 0.040, -0.118, 0.128, 0.012, 0.0075, 0.004) > 0), 62, 54, 66, 0.20)    # the lines under his eyes
        put(face & (capd(X, Y, 0.014, -0.10, 0.020, 0.022, 0.0065, 0.0055) > 0), 120, 114, 128, 0.28)  # nose: one thin shadow
        put(face & (capd(X, Y, -0.016, 0.038, 0.026, 0.038, 0.006, 0.005) > 0), 84, 78, 92, 0.22)
        op = smooth(19.8, 20.7, t)                                              # his lips part...
        put(face & (capd(X, Y, -0.058, 0.142, 0.058, 0.142, 0.0075, 0.005) > 0), 70, 44, 54, 0.22)
        put(face & (capd(X, Y, -0.022, 0.180, 0.022, 0.180, 0.005, 0.005) > 0), 150, 142, 152, 0.35)
        if op > 0:
            lens = np.abs(Y - 0.145) < 0.036 * op * (1 - (X / 0.066) ** 2)      # a slit with pointed corners
            put(face & (aX < 0.066) & lens, 0, 0, 0, 0.0)

        # headband with the slashed plate
        yc = -0.335 + 0.13 * X * X
        band = (np.abs(Y - yc) < 0.05) & (aX < 0.33) & ((fd > -0.25) | (aX < 0.3))
        weave = vnoise(X * 60, Y * 60)
        put(band, 40 + 26 * weave, 46 + 28 * weave, 80 + 40 * weave, 0.45 + 0.2 * weave)
        put((Y - yc > 0.05) & (Y - yc < 0.058) & face, 0, 0, 0, 0.0)   # its shadow on the brow
        plate = (np.abs(Y - yc) < 0.041) & (aX < 0.17)
        pl = 0.78 + 0.22 * vnoise(X * 9 + 3, Y * 30)
        put(plate, 205 * pl, 212 * pl, 230 * pl, 0.95)
        put(plate & (np.abs(np.hypot(X + 0.012, (Y - yc) * 1.0) - 0.020) < 0.006), 70, 76, 96, 0.5)    # leaf swirl
        put(plate & (np.abs(Y - yc - 0.004 - 0.05 * X) < 0.007) & (aX < 0.14), 30, 32, 44, 0.3)        # the slash

        # hair in front: side locks and the long bangs that frame his face
        hair(capd(aX, Y, 0.31, -0.30, 0.36, 0.36, 0.085, 0.008))
        hair(capd(aX, Y, 0.20, -0.52, 0.265, 0.16, 0.058, 0.006))
        hair(capd(X, Y, -0.10, -0.52, -0.185, -0.06, 0.032, 0.003))             # loose strands across the brow
        hair(capd(X, Y, 0.125, -0.52, 0.20, -0.12, 0.026, 0.003))

        # ---- the crows: out of his mouth and straight at us, growing as they come
        for birth, ang, ph, sp in sorted(self.mcrows, reverse=True):
            age = t - birth
            if not 0.0 < age < 3.4:
                continue
            d = sp * (0.06 * age + 0.13 * age * age)
            cx, cy = math.cos(ang) * d * 1.35, 0.142 + math.sin(ang) * d * 0.9
            sz = 0.016 + 0.058 * age * sp
            if abs(cx) > 0.60 + 3 * sz or abs(cy + 0.075) > 0.34 + 3 * sz:
                continue
            flap = math.sin(t * 11 + ph)
            wy = cy - sz * (0.2 + 1.5 * flap)
            el = cy - sz * (0.5 + 0.6 * flap)                                  # wings bend at the elbow
            dep = np.maximum.reduce([elld(X, Y, cx, cy, sz * 0.42, sz * 0.85),
                                     elld(X, Y, cx, cy + sz * 0.62, sz * 0.30, sz * 0.32),
                                     capd(X, Y, cx, cy - sz * 0.2, cx - sz * 1.5, el, sz * 0.50, sz * 0.34),
                                     capd(X, Y, cx, cy - sz * 0.2, cx + sz * 1.5, el, sz * 0.50, sz * 0.34),
                                     capd(X, Y, cx - sz * 1.5, el, cx - sz * 3.1, wy, sz * 0.36, sz * 0.03),
                                     capd(X, Y, cx + sz * 1.5, el, cx + sz * 3.1, wy, sz * 0.36, sz * 0.03),
                                     capd(X, Y, cx, cy - sz * 0.6, cx, cy - sz * 1.6, sz * 0.30, sz * 0.10)])
            put(dep > -0.30, 0, 0, 0, 0.0)                                     # a black gap, so they cut out of his face
            m = dep > 0
            edge = 1 - np.clip(dep * 1.3, 0, 1)                                # and a cold rim, so they read on black
            put(m, 62 + 140 * edge, 70 + 145 * edge, 104 + 140 * edge, 0.72 + 0.26 * edge)
            if sz > 0.05:
                put(np.hypot(aX * 0 + np.abs(X - cx) - sz * 0.15, Y - cy - sz * 0.50) < sz * 0.085, 240, 30, 36, 1.0)
            beak = capd(X, Y, cx, cy + sz * 0.75, cx, cy + sz * 1.15, sz * 0.12, sz * 0.02) > 0
            put(beak, 70, 74, 96, 0.6)

        b = np.where(pupil, pv * zc, b)
        R, G, B = np.where(pupil, pr, R), np.where(pupil, pg, G), np.where(pupil, pb, B)
        self._paper = pupil & bg
        self._Hh = Hh
        return b, R, G, B

    def _portrait_fx(self, t, z, gx, gy, idx, key, ok):
        """A few distant crows in the mist behind him (off: the scene reads better still)."""
        return
        if z < 6.0:
            return
        k = self._Hh * R_PLANET
        for X0, Y0, sp, ph, _ in self.birds[:12]:
            Xb = (X0 - sp * t + 1.9) % 3.8 - 1.9
            Yb = Y0 * 1.6 - 0.2 + 0.05 * math.sin(t * 0.8 + ph)
            col = int(COLS / 2.0 + (Xb * k + gx) * z / self.cell_x)
            row = int(ROWS / 2.0 + (Yb * k + gy) * z / self.cell_y)
            if not (1 <= col < COLS - 1 and 0 <= row < ROWS):
                continue
            wing = CH_CARET if math.sin(t * 8 + ph) > 0 else CH_DASH
            for cc_, ch in ((col - 1, wing), (col, CH_V), (col + 1, wing)):
                if ok[row, cc_]:
                    idx[row, cc_], key[row, cc_] = ch, 0x223

    def _ink(self, t, z, pupil, u, v, b, R, G, B):
        """The scene inside the pupil as an ink-wash poster. Black ink is simply empty cells;
        the paper is what is drawn. X, Y are screen-fitted: Y = -1..1 top to bottom at full zoom."""
        Hh = 1.0 / (R_PLANET * ZOOM_MAX * 1.14)
        X, Y = u / Hh, v / Hh
        zc = smooth(2.5, 8.0, z)                      # the picture develops as we fall in
        n_hi = vnoise(X * 13 + 3, Y * 13 + 8)         # edge raggedness for brush strokes
        n_lo = vnoise(X * 4 + 11, Y * 4 + 2)

        # paper, with grey mist drifting across it
        mist = sstep(0.50, 0.85, vnoise(X * 1.2 + t * 0.06 + 2, Y * 1.9 + 5))
        mist = np.maximum(mist, 0.8 * sstep(0.55, 0.9, vnoise(X * 2.1 - t * 0.04, Y * 3.0 + 9)))
        fib = vnoise(X * 10, Y * 16)
        tone = 0.93 + 0.07 * fib - 0.13 * mist
        pr, pg, pb = 250 * tone, 247 * tone, 238 * tone
        pv = 0.74 + 0.26 * tone                       # glyph density
        ink = np.zeros_like(X)                        # 1 = solid black

        def paint(mask, col, dens=0.9):
            nonlocal pr, pg, pb, pv, ink
            pr, pg, pb = np.where(mask, col[0], pr), np.where(mask, col[1], pg), np.where(mask, col[2], pb)
            pv = np.where(mask, dens, pv)
            ink = np.where(mask, 0.0, ink)

        def black(mask):
            nonlocal ink
            ink = np.where(mask, 1.0, ink)

        def wash(depth, grey, rough=0.5):
            """Grey ink with a ragged, dry-brush edge."""
            m = depth + rough * (n_hi - 0.5) > 0
            g = grey * (0.8 + 0.4 * n_lo)
            paint(m, (g, g, g * 1.05), 0.85)

        # red sun, brushed on: ragged rim and a few dry streaks
        sx, sy = X + 0.80, Y + 0.50
        sd = np.hypot(sx, sy)
        sun = sd < 0.31 + 0.03 * (vnoise(np.arctan2(sy, sx) * 2.5 + 4, sd * 3) - 0.5)
        sun &= ~((vnoise(sx * 5 + sy * 3, sy * 34) > 0.80) & (sd > 0.10) & (sx > -0.05))
        sv = 0.85 + 0.15 * n_lo
        paint(sun, (226 * sv, 40 * sv, 38 * sv), 0.97)

        # distant tree on the left
        black(capd(X, Y, -1.46, 0.80, -1.40, 0.22, 0.05, 0.028) > 0)
        black(capd(X, Y, -1.41, 0.34, -1.66, 0.04, 0.022, 0.01) > 0)
        black(capd(X, Y, -1.40, 0.28, -1.14, 0.10, 0.022, 0.01) > 0)
        for cx, cy, rx, ry in ((-1.66, -0.02, 0.24, 0.085), (-1.22, 0.05, 0.26, 0.08), (-1.45, -0.15, 0.22, 0.075),
                               (-1.02, 0.22, 0.16, 0.055), (-1.70, 0.20, 0.14, 0.05)):
            black(elld(X, Y, cx, cy, rx, ry) + 0.9 * (n_hi - 0.5) > 0)

        # pagoda on the right
        px = 1.34
        apx = np.abs(X - px)
        black((Y > -0.52) & (Y < -0.26) & (apx < 0.014))
        for i in range(5):
            y0, w = -0.27 + i * 0.185, 0.17 + i * 0.055
            f = (Y - y0) / 0.07                                           # 0 at the ridge, 1 at the eave
            lift = 0.05 * np.clip(apx / w - 0.75, 0, 1)                    # upturned tips
            f2 = (Y + lift - y0) / 0.07
            roof = (f2 > 0) & (f2 < 1) & (apx < w * (0.35 + 0.65 * f2))
            body = (Y >= y0 + 0.07) & (Y < y0 + 0.185) & (apx < w * 0.52)
            black(body & ((Y - y0 - 0.07) > 0.03))
            black(roof)

        # blossom branch reaching in from the top right, with a crow on it
        black(capd(X, Y, 1.85, -0.95, 0.50, -0.56, 0.05, 0.014) > 0)
        black(capd(X, Y, 0.95, -0.70, -0.45, -0.50, 0.022, 0.006) > 0)
        black(capd(X, Y, 1.25, -0.78, 1.02, -0.44, 0.016, 0.004) > 0)
        black(capd(X, Y, 1.50, -0.86, 1.62, -0.60, 0.014, 0.004) > 0)
        for i, (bx, by) in enumerate(((1.58, -0.84), (1.40, -0.72), (1.22, -0.83), (1.06, -0.62), (0.86, -0.74),
                                      (0.70, -0.58), (0.55, -0.66), (0.36, -0.50), (0.22, -0.60), (1.03, -0.45),
                                      (1.68, -0.97), (-0.22, -0.53), (1.63, -0.60), (1.30, -0.60), (-0.44, -0.50))):
            r = 0.034 + 0.012 * ((i * 7) % 3)
            petal = np.hypot(X - bx, Y - by) < r * (0.75 + 0.25 * np.cos(5 * np.arctan2(Y - by, X - bx) + i))
            paint(petal, (236, 150, 160) if i == 14 else (214, 48, 58), 0.9)
        black(capd(X, Y, 0.02, -0.80, 0.22, -0.655, 0.078, 0.048) > 0)       # crow: body
        black(elld(X, Y, -0.03, -0.85, 0.052, 0.052) > 0)                     # head
        black(capd(X, Y, -0.07, -0.85, -0.155, -0.82, 0.022, 0.004) > 0)      # beak
        black(capd(X, Y, 0.22, -0.655, 0.37, -0.585, 0.036, 0.014) > 0)       # tail
        black(boxd(X, Y, 0.085, 0.10, -0.68, -0.575) > 0)                     # legs
        black(boxd(X, Y, 0.135, 0.15, -0.68, -0.58) > 0)

        # the hill he stands on, with a fringe of grass
        hill = 0.70 + 0.05 * np.sin(X * 1.4 + 0.6) + 0.04 * (X / 1.78) ** 2
        black(Y > hill - 0.05 * vnoise(X * 45, t * 0.6 + 3) ** 2)

        # --- the figure: long cloak blowing left, arm out, head turned
        flut = vnoise(X * 9 + t * 1.6, Y * 9 - t * 0.5) - 0.5                # the wind in the cloth
        cloak = capd(X, Y, 0.13, -0.20, 0.07, 0.48, 0.135, 0.30) > 0
        cloak |= capd(X, Y, 0.0, 0.20, -0.58, 0.36 + 0.03 * math.sin(t * 1.9), 0.19, 0.015) + 0.5 * flut > 0
        cloak &= Y < 0.56 + 0.10 * vnoise(X * 16 + t * 0.9, 2.0) - 0.04      # ragged hem
        cloak |= elld(X, Y, 0.135, -0.265, 0.135, 0.10) > 0                   # high collar
        cloak |= capd(X, Y, 0.04, -0.17, -0.30, -0.20, 0.10, 0.085) > 0       # outstretched sleeve
        cloak |= capd(X, Y, -0.20, -0.16, -0.17, 0.03, 0.10, 0.04) > 0        # its drape
        black(boxd(X, Y, 0.035, 0.05, 0.50, 0.74) > 0)                        # legs
        black(boxd(X, Y, 0.13, 0.145, 0.50, 0.74) > 0)
        black(cloak)
        paint(elld(X, Y, -0.285, -0.10, 0.04, 0.095) > 0, (196, 44, 48), 0.9)             # red lining
        paint(capd(X, Y, -0.33, -0.215, -0.50, -0.20, 0.03, 0.013) > 0, (240, 222, 208), 0.95)   # hand
        for cx, cy, cs in ((0.16, 0.02, 0.085), (0.02, 0.30, 0.125)):        # the red clouds on the cloak
            x, y = (X - cx) / cs, (Y - cy) / cs
            cl = np.maximum.reduce([elld(x, y, 0, 0.15, 1.0, 0.42), elld(x, y, -0.45, -0.2, 0.5, 0.45),
                                    elld(x, y, 0.2, -0.38, 0.55, 0.5), elld(x, y, 0.65, -0.05, 0.4, 0.38)])
            paint(cloak & (cl > -0.12), (236, 232, 224), 0.9)                # pale outline
            paint(cloak & (cl > 0.08), (208, 44, 46), 0.92)
        black(elld(X, Y, 0.125, -0.40, 0.09, 0.105) > 0)                      # hair
        black(capd(X, Y, 0.20, -0.37, 0.27, -0.20, 0.032, 0.005) > 0)         # ponytail
        paint(elld(X, Y, 0.078, -0.385, 0.04, 0.062) > 0, (240, 226, 212), 0.95)          # face
        black(capd(X, Y, 0.105, -0.47, 0.04, -0.36, 0.03, 0.006) > 0)         # a bang across it
        paint(elld(X, Y, 0.066, -0.395, 0.016, 0.014) > 0, (230, 30, 40), 1.0)            # the eye

        # ink spatter around him
        spat = (vnoise(X * 31 + 7, Y * 31 + 1) > 0.90) & (np.hypot((X + 0.05) / 0.8, (Y - 0.1) / 0.6) < 1) \
            & (n_lo > 0.5)
        black(spat)

        vis = pupil
        lit = vis & (ink < 0.5)
        b = np.where(vis, np.where(ink < 0.5, pv * zc, 0.0), b)
        R, G, B = np.where(lit, pr, R), np.where(lit, pg, G), np.where(lit, pb, B)
        self._paper = lit & (pr > 150) & (pg > 150)   # open paper, where birds and petals may fly
        self._Hh = Hh
        return b, R, G, B

    def _ink_fx(self, t, z, gx, gy, idx, key, ok):
        """Small birds crossing the sky and blossom petals on the wind."""
        if z < 6.0:
            return
        k = self._Hh * R_PLANET
        for X0, Y0, sp, ph, _ in self.birds:
            Xb = (X0 - sp * t + 1.9) % 3.8 - 1.9
            Yb = Y0 + 0.04 * math.sin(t * 0.8 + ph)
            col = int(COLS / 2.0 + (Xb * k + gx) * z / self.cell_x)
            row = int(ROWS / 2.0 + (Yb * k + gy) * z / self.cell_y)
            if not (1 <= col < COLS - 1 and 1 <= row < ROWS):
                continue
            up = math.sin(t * 9 + ph) > 0                # wings up: a V; wings down: a dash
            for cc_, rr_ in ((col - 1, row - 1 if up else row), (col, row), (col + 1, row - 1 if up else row)):
                if ok[rr_, cc_]:
                    idx[rr_, cc_], key[rr_, cc_] = 0, 0
        for X0, Y0, sp, ph, _ in self.petals:
            Xp = (X0 - 0.10 * sp * t + 0.12 * math.sin(t * 1.3 + ph) + 1.9) % 3.8 - 1.9
            Yp = (Y0 + 0.09 * sp * t + 0.7) % 1.4 - 0.7
            col = int(COLS / 2.0 + (Xp * k + gx) * z / self.cell_x)
            row = int(ROWS / 2.0 + (Yp * k + gy) * z / self.cell_y)
            if 0 <= col < COLS and 0 <= row < ROWS and ok[row, col]:
                idx[row, col], key[row, col] = CH_STAR, 0xd34

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

    def _crows(self, t, z, gx, gy, idx, key, ok):
        """Crows wheeling across the sky behind him: pale on the night, black against the moon."""
        if z < 5.0:
            return
        planet = self._planet
        for u0, v0, du, dv, ph, big in self.murder:
            if z < 5.0 + 9.0 * (ph / math.tau):         # they arrive a few at a time
                continue
            u = (u0 + du * t + 2.6) % 5.2 - 2.6
            v = v0 + dv * t + 0.06 * math.sin(t * 0.9 + ph)
            col = int(COLS / 2.0 + (u * R_PLANET + gx) * z / self.cell_x)
            row = int(ROWS / 2.0 + (v * R_PLANET + gy) * z / self.cell_y)
            if not (1 <= col < COLS - 1 and 0 <= row < ROWS):
                continue
            flap = math.sin(t * 8 + ph) > 0
            wing = CH_CARET if flap else CH_DASH
            cells = ((col - 1, wing), (col, CH_V), (col + 1, wing)) if big else ((col, CH_V if flap else CH_CARET),)
            for cc_, ch in cells:
                if ok[row, cc_]:
                    if planet[row, cc_]:
                        idx[row, cc_], key[row, cc_] = 0, 0      # a black notch in the moon
                    else:
                        idx[row, cc_], key[row, cc_] = ch, 0xacd

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
