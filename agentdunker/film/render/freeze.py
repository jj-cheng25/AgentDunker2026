"""Shot freeze: dimmed frame with a soft spotlight on the shooter, ball and rim, plus the B3 xFG card.

The card is the user-selected B3 design from the hackathon (``exam-freeze-b-font-options/B3``), drawn from
data: shooter short name, team name + jersey, and the official xFG. Without an official xFG only the
name strip is shown (no number is ever invented). Without a shooter box the frame is only dimmed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .style import CYAN, font, with_alpha

WHITE = (234, 247, 254, 255)
DIM = 0.42          # remaining brightness outside the spotlight (≈ 60 % darker)
CARD_W = 456


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class _Canvas:
    """3× supersampled drawing in native 1080p units (from the hackathon cards_ab.Canvas)."""

    def __init__(self, size):
        self.r = 3
        self.size = size
        self.image = Image.new("RGBA", (size[0] * 3, size[1] * 3))
        self.draw = ImageDraw.Draw(self.image)

    def font(self, size, latin=False, weight=500):
        return font(size * self.r, 700 if latin else weight, latin=latin)

    def measure(self, text, size, latin=False):
        l, t, r, b = self.font(size, latin).getbbox(text)
        return (r - l) / self.r, (b - t) / self.r

    def text(self, text, x, y, size, fill=WHITE, latin=False):
        f = self.font(size, latin)
        l, t, _, _ = f.getbbox(text)
        self.draw.text((round(x * self.r) - l, round(y * self.r) - t), text, font=f, fill=fill)

    def polygon(self, pts, fill):
        self.draw.polygon([(round(x * self.r), round(y * self.r)) for x, y in pts], fill=fill)

    def line(self, pts, fill, width=1.0):
        self.draw.line([(round(x * self.r), round(y * self.r)) for x, y in pts], fill=fill,
                       width=max(1, round(width * self.r)))

    def finish(self) -> Image.Image:
        return self.image.resize(self.size, Image.Resampling.LANCZOS)


def make_card(name: str, team: str, xfg: Optional[float]) -> tuple[Image.Image, tuple[int, int]]:
    """B3 card (456×288) or, without xFG, only its identity strip. Returns (card, leader anchor in card px)."""
    strip, height = 52, 288 if xfg is not None else 72
    c = _Canvas((CARD_W, height))
    delta = strip - 33
    w, h = c.image.size
    if xfg is not None:
        alpha = Image.new("L", (w, h))
        ImageDraw.Draw(alpha).rounded_rectangle((36, (28 + delta) * 3, 422 * 3, (height - 8) * 3), radius=150, fill=110)
        bg = Image.new("RGBA", (w, h), (2, 11, 20, 255))
        bg.putalpha(alpha.filter(ImageFilter.GaussianBlur(72)))
        c.image.alpha_composite(bg)
        c.draw = ImageDraw.Draw(c.image)
    top, bottom = 10, 10 + strip
    center = (top + bottom) / 2
    c.polygon([(25, top), (394, top), (380, bottom), (11, bottom)], (23, 109, 154, 232))
    c.polygon([(25, top), (31, top), (17, bottom), (11, bottom)], (104, 225, 255, 255))
    nw, nh = c.measure(name, 34)
    tw, th = c.measure(team, 28)
    if 39 + nw + 24 > 367 - tw:                       # long team label: drop it rather than overlap
        team, tw = "", 0
    if name:
        c.text(name, 39, center - nh / 2, 34)
    if team:
        c.text(team, 367 - tw, center - th / 2, 28, (209, 239, 249, 255))
    if xfg is None:
        return c.finish(), (17, bottom)
    number = f"{xfg * 100:.1f}"
    size = 171
    while size > 94 and c.measure(number, size, True)[0] > 335:
        size -= 0.5
    f = c.font(size, True)
    box = f.getbbox(number)
    text_w, text_h = box[2] - box[0], box[3] - box[1]
    lean = 0.105
    shift = math.ceil(text_h * lean)
    txt = Image.new("RGBA", (text_w + shift + 12, text_h + 6))
    ImageDraw.Draw(txt).text((-box[0], -box[1]), number, font=f, fill=WHITE)
    slanted = txt.transform(txt.size, Image.Transform.AFFINE, (1, lean, -shift, 0, 1, 0), Image.Resampling.BICUBIC)
    vx, vy = 20, 68 + delta
    c.image.alpha_composite(slanted, (vx * 3, vy * 3))
    c.draw = ImageDraw.Draw(c.image)
    pw, ph = c.measure("%", 51, True)
    c.text("%", vx + (text_w + shift) / 3 + 8, vy + text_h / 3 - ph - 0.5, 51, (*CYAN, 255), True)
    ly = 212 + delta
    c.line([(27, ly), (389, ly)], (139, 192, 211, 105), 0.8)
    c.line([(27, ly), (83, ly)], (78, 213, 254, 250), 2)
    mw, mh = c.measure("预期命中率", 32)
    sw, sh = c.measure("NBA 官方", 26)
    c.text("预期命中率", 27, 224 + delta, 32, (213, 233, 242, 255))
    c.text("NBA 官方", 389 - sw, 224 + delta + mh - sh, 26, (132, 174, 193, 255))
    return c.finish(), (27, ly)


def choose_card_position(card: Image.Image, protect: list[tuple], ball: Optional[tuple], rim: Optional[tuple]
                         ) -> tuple[int, int]:
    """Upper part of the frame, clear of the shooter/ball/rim boxes; prefer right of the rim (reference)."""
    w, h = card.size
    anchor = rim or ball or (960, 400)
    cands = []
    for y in (60, 90, 140, 200, 280):
        for dx in (190, -w - 190, 330, -w - 330, 60, -w - 60):
            cands.append((anchor[0] + dx, y))
    cands += [(1920 - w - 80, 80), (80, 80), (1920 - w - 80, 300), (80, 300)]

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    best = None
    for x, y in cands:
        x, y = int(min(max(40, x), 1920 - w - 40)), int(min(max(40, y), 1080 - h - 200))
        rect = (x, y, x + w, y + h)
        cost = sum(overlap(rect, p) for p in protect) * 10 + (math.dist((x, y + h), ball) if ball else 0) * 0.5
        if best is None or cost < best[0]:
            best = (cost, (x, y))
    return best[1]


@dataclass
class FreezeSpec:
    frame: np.ndarray                     # RGB 1080p source frame at release
    shooter_box: Optional[tuple]          # 1080p px
    ball: Optional[tuple]                 # 1080p px
    rim: Optional[tuple]                  # 1080p px
    name: str = ""
    team: str = ""
    xfg: Optional[float] = None


class FreezeRenderer:
    """``frame(t, duration)``: dim 0.2→1.5 s, card fades in 0.4→1.2 s, everything lifts in the last 0.4 s."""

    def __init__(self, spec: FreezeSpec):
        self.spec = spec
        src = Image.fromarray(spec.frame).convert("RGB")
        self.src = src
        mask = Image.new("L", (1920, 1080))
        d = ImageDraw.Draw(mask)
        protect = []
        if spec.shooter_box:
            l, t, r, b = spec.shooter_box
            bw, bh = r - l, b - t
            d.ellipse((l - bw * 0.45, t - bh * 0.25, r + bw * 0.45, b + bh * 0.12), fill=255)
            # Soft cone from above and a floor patch, like a stage light.
            cx = (l + r) / 2
            d.polygon([(cx - bw * 0.25, 0), (cx + bw * 0.25, 0), (r + bw * 0.5, b), (l - bw * 0.5, b)], fill=120)
            d.ellipse((l - bw * 0.7, b - bh * 0.08, r + bw * 0.7, b + bh * 0.16), fill=200)
            protect.append((l - 30, t - 30, r + 30, b + 30))
        for pt, rx, ry in ((spec.ball, 60, 60), (spec.rim, 90, 110)):
            if pt:
                d.ellipse((pt[0] - rx, pt[1] - ry, pt[0] + rx, pt[1] + ry), fill=255)
                protect.append((pt[0] - rx, pt[1] - ry, pt[0] + rx, pt[1] + ry))
        self.has_spot = bool(spec.shooter_box)
        self.mask = mask.filter(ImageFilter.GaussianBlur(28))
        dark = Image.blend(src, Image.new("RGB", src.size, (2, 10, 20)), 1 - DIM)
        self.graded = Image.composite(src, dark, self.mask) if self.has_spot else dark
        self.card = None
        if spec.name or spec.xfg is not None:
            self.card, self.card_anchor = make_card(spec.name, spec.team, spec.xfg)
            self.card_pos = choose_card_position(self.card, protect, spec.ball, spec.rim)
        self.layout = {"spotlight": self.has_spot, "card": self.card is not None,
                       "card_rect": None if self.card is None else [*self.card_pos, self.card_pos[0] + self.card.width,
                                                                       self.card_pos[1] + self.card.height]}

    def frame(self, t: float, duration: float) -> np.ndarray:
        lift = smooth((duration - t) / 0.4)
        grade = smooth((t - 0.2) / 1.3) * lift
        img = Image.blend(self.src, self.graded, grade).convert("RGBA")
        if self.card is not None:
            a = smooth((t - 0.4) / 0.8) * lift
            if a > 0.01:
                x, y = self.card_pos
                y += round((1 - a) * 16)
                over = Image.new("RGBA", img.size)
                d = ImageDraw.Draw(over)
                start = self.spec.ball or (self.spec.shooter_box and ((self.spec.shooter_box[0] + self.spec.shooter_box[2]) / 2,
                                                                      self.spec.shooter_box[1]))
                if start:
                    end = (x + self.card_anchor[0], y + self.card_anchor[1])
                    elbow = (end[0] - 60 if end[0] > start[0] else end[0] + 60, start[1])
                    d.line([start, elbow, end], fill=(108, 211, 236, 200), width=2, joint="curve")
                    d.ellipse((start[0] - 3, start[1] - 3, start[0] + 3, start[1] + 3), fill=(*CYAN, 230))
                over.alpha_composite(self.card, (x, y))
                img.alpha_composite(with_alpha(over, a))
        return np.asarray(img.convert("RGB"))
