"""Resume-playback effects: cyan comet trail on the ball and the made-shot light wave.

Ported from the hackathon ``exam-b3-energy-finish/render_live_finish.py``; the hand-read ball points, rim
points and made frame are replaced by M5's ``ball_path_img`` / ``rim_img`` / ``made_t``. Effects are drawn
only from measured data: no ball path → no trail; not made → no wave; no rim → wave centred on the last
ball point.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

W, H = 1920, 1080
TRAIL_S = 0.25            # comet length in source seconds
WAVE_S = 1.22


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _interp(points: list[list[float]], t: float) -> Optional[tuple[float, float]]:
    if not points or t < points[0][0] or t > points[-1][0]:
        return None
    ts = [p[0] for p in points]
    return float(np.interp(t, ts, [p[1] for p in points])), float(np.interp(t, ts, [p[2] for p in points]))


class FinishEffects:
    """``apply(rgb, t_src, t_film)``; times: ball/rim in source seconds, wave clocked on film seconds."""

    def __init__(self, ball_path: list[list[float]], rim_path: list[list[float]], made: Optional[bool],
                 made_t: Optional[float], scale: tuple[float, float] = (1.0, 1.0), trail_end_t: Optional[float] = None):
        sx, sy = scale
        self.ball = [[p[0], p[1] * sx, p[2] * sy] for p in (ball_path or [])]
        self.rim = [[p[0], p[1] * sx, p[2] * sy] for p in (rim_path or [])]
        self.trail_end = trail_end_t if trail_end_t is not None else (self.ball[-1][0] if self.ball else None)
        self.made = bool(made) and made_t is not None
        self.made_t = made_t
        self.wave_film_t: Optional[float] = None          # set by caller: film time of made_t + 0.07 s
        center = None
        if self.made:
            center = _interp(self.rim, made_t) if self.rim else None
            if center is None and self.rim:
                center = tuple(min(self.rim, key=lambda p: abs(p[0] - made_t))[1:3])
            if center is None and self.ball:
                center = tuple(self.ball[-1][1:3])
        self.center = center
        self.stats = {"trail_frames": 0, "wave_frames": 0, "trail_points": len(self.ball), "wave_center": center}

    def apply(self, rgb: np.ndarray, t_src: float, t_film: float) -> np.ndarray:
        trail = self.ball and self.ball[0][0] < t_src <= (self.trail_end or 0)
        age = t_film - self.wave_film_t if self.wave_film_t is not None else -1
        wave = self.made and self.center is not None and 0 <= age < WAVE_S
        if not trail and not wave:
            return rgb
        image = Image.fromarray(rgb)
        overlay = Image.new("RGBA", (W, H))
        draw = ImageDraw.Draw(overlay)
        light = Image.new("RGB", (W // 2, H // 2))
        ld = ImageDraw.Draw(light)
        if trail:
            self.stats["trail_frames"] += 1
            pts = [_interp(self.ball, x) for x in np.linspace(max(self.ball[0][0], t_src - TRAIL_S), t_src, 40)]
            pts = [p for p in pts if p is not None]
            head = pts[-1] if pts else None
            pts = [p for p in pts if head is None or math.dist(p, head) >= 18]
            for j in range(1, len(pts)):
                s = (j / len(pts)) ** 0.72
                a, b = pts[j - 1], pts[j]
                ld.line([(a[0] / 2, a[1] / 2), (b[0] / 2, b[1] / 2)],
                        fill=(round(12 * s), round(100 * s), round(230 * s)), width=round(11 + 10 * s))
                draw.line([a, b], fill=(0, 133, 255, round(250 * s)), width=round(6 + 11 * s))
                draw.line([a, b], fill=(65, 222, 255, round(255 * s)), width=round(2 + 5 * s))
                draw.line([a, b], fill=(201, 248, 255, round(235 * s)), width=2)
        emission = ImageChops.screen(light.filter(ImageFilter.GaussianBlur(4)),
                                     light.filter(ImageFilter.GaussianBlur(12)).point(lambda x: round(x * .5))
                                     ).resize((W, H), Image.Resampling.BILINEAR)
        if wave:
            self.stats["wave_frames"] += 1
            cx, cy = self.center
            unit = 4
            yy, xx = np.mgrid[:H // unit, :W // unit].astype(np.float32)
            dx, dy = xx * unit - cx, yy * unit - cy
            attack = smooth(age / .065)
            release = 1 - smooth((age - .68) / .52)
            broad = np.exp(-.5 * ((dx / 245) ** 2 + (dy / 138) ** 2)) * attack * math.exp(-age * 1.9) * release * .62
            flash = np.exp(-.5 * ((dx / 72) ** 2 + (dy / 46) ** 2)) * attack * math.exp(-age * 9) * 1.15
            halo = np.zeros_like(dx)
            hot = np.zeros_like(dx)
            elliptic = np.sqrt(dx ** 2 + (dy / .43) ** 2)
            angle = np.arctan2(dy / .43, dx)
            for delay, weight, dur in ((0., 1., .98), (.14, .64, .94), (.29, .28, .83)):
                phase = (age - delay) / dur
                if not 0 <= phase < 1:
                    continue
                radius = 40 + 445 * (1 - (1 - phase) ** 1.65)
                env = smooth(phase / .065) * (1 - phase) ** 1.25 * weight
                wav = 1 + .009 * np.sin(angle * 6 + age * 9) + .006 * np.sin(angle * 11 - age * 7)
                dist = elliptic - radius * wav
                facing = .78 + .22 * np.sin(angle + .3)
                halo += np.exp(-.5 * (dist / 39) ** 2) * env * facing * .85
                hot += np.exp(-.5 * (dist / 8.0) ** 2) * env * facing * .75
                box = (cx - radius, cy - radius * .43, cx + radius, cy + radius * .43)
                draw.arc(box, 18, 145, fill=(111, 235, 255, round(205 * env)), width=4)
                draw.arc(box, 205, 295, fill=(65, 183, 255, round(205 * env * .7)), width=3)
            hoop = 1 - .48 * np.exp(-.5 * ((dx / 50) ** 2 + (dy / 25) ** 2))
            r = (broad * .05 + halo * .09 + hot * .48 + flash * .65) * hoop
            g = (broad * .33 + halo * .54 + hot * .93 + flash * .93) * hoop
            b = (broad * .92 + halo * 1.02 + hot + flash) * hoop
            wave_light = Image.fromarray(np.uint8(np.clip(np.dstack((r, g, b)) * 255, 0, 255))).resize((W, H), Image.Resampling.BILINEAR)
            emission = ImageChops.screen(emission, wave_light)
            pp = age / .72
            if 0 < pp < 1:
                st = smooth(age / .08) * (1 - pp) ** 1.8
                for k in range(10):
                    th = k * math.tau / 10 + .17
                    rad = 48 + (160 + (k % 3) * 37) * pp
                    x, y = cx + math.cos(th) * rad, cy + math.sin(th) * rad * .53
                    ln = 5 + 11 * (1 - pp)
                    draw.line([(x, y), (x + math.cos(th) * ln, y + math.sin(th) * ln * .53)],
                              fill=(115, 226, 255, round(145 * st)), width=3)
        out = Image.alpha_composite(ImageChops.screen(image, emission).convert("RGBA"), overlay)
        return np.asarray(out.convert("RGB"))
