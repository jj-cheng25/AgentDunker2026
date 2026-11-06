"""Live nameplates that follow M1 boxes on wide shots (REFERENCE_SPEC §2.2).

Tiering per player: number + Chinese short name (identity conf ≥ 0.6 and a name) → number only (jersey conf
≥ 0.5) → no plate. Boxes are smoothed; plates appear only on wide shots, only where the player is
observed (gaps ≤ 0.26 s bridged), runs shorter than 0.35 s are dropped, and every run fades in/out.
At most ``max_plates`` per frame; the hero (shooter) has priority and gets a two-row card when season
data exists, otherwise a normal plate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .style import name_chip, team_color, with_alpha

MIN_NAME_CONF = 0.6
MIN_JERSEY_CONF = 0.5
BRIDGE_S = 0.26
MIN_RUN_S = 0.35
FADE_S = 0.15
CUT_GUARD_S = 0.15


@dataclass
class Plate:
    track_id: int
    name: str
    jersey: str
    color: tuple
    tier: str                       # "name" | "number"
    hero: bool = False
    stat: tuple = ("", "")
    boxes: dict = field(default_factory=dict)      # source frame -> smoothed bbox (observed or bridged)


def short_name(info: dict) -> str:
    zh = info.get("name_zh_short") or ""
    if zh:
        return zh
    full = info.get("name_zh") or ""
    if full:
        return full.split("·")[-1]
    parts = [w for w in (info.get("name") or "").split() if w.rstrip(".").lower() not in ("jr", "sr", "ii", "iii", "iv")]
    return parts[-1] if parts else ""


def plate_specs(tracks: dict, identity: Optional[dict], hero_tid: Optional[int]) -> tuple[list[Plate], dict]:
    """Decide what each player track may show; returns plates and a summary for the report."""
    players = (identity or {}).get("players", {})
    teams = (identity or {}).get("teams", {})
    plates, counts = [], {"name": 0, "number": 0, "none": 0}
    for tr in tracks.get("tracks", []):
        if tr.get("role") != "player":
            continue
        info = players.get(str(tr["track_id"]), {})
        conf = float(info.get("conf") or 0)
        jersey = str(info.get("jersey") or "") if conf >= MIN_JERSEY_CONF else ""
        name = short_name(info) if conf >= MIN_NAME_CONF else ""
        if not jersey and not name:
            counts["none"] += 1
            continue
        abbr = info.get("team_abbr") or (teams.get(tr.get("team") or "", {}) or {}).get("abbr")
        tier = "name" if name else "number"
        counts[tier] += 1
        plates.append(Plate(tr["track_id"], name, jersey, team_color(abbr), tier, tr["track_id"] == hero_tid))
    return plates, counts


def _smooth_boxes(frames: list[dict], fps: float) -> dict[int, np.ndarray]:
    """Observed boxes, small gaps bridged by interpolation, then a centred moving average (±0.1 s)."""
    obs = [(fr["f"], np.array(fr["bbox"], float)) for fr in frames if fr.get("observed")]
    if not obs:
        return {}
    out: dict[int, np.ndarray] = {}
    max_gap = int(BRIDGE_S * fps)
    for (fa, ba), (fb, bb) in zip(obs, obs[1:] + [obs[-1]]):
        out[fa] = ba
        if 1 < fb - fa <= max_gap:
            for f in range(fa + 1, fb):
                w = (f - fa) / (fb - fa)
                out[f] = ba * (1 - w) + bb * w
    keys = sorted(out)
    arr = np.array([out[k] for k in keys])
    r = max(1, int(0.1 * fps))
    sm = {}
    for i, k in enumerate(keys):
        lo, hi = i, i
        while lo > 0 and keys[lo - 1] == keys[lo] - 1 and i - lo < r:
            lo -= 1
        while hi < len(keys) - 1 and keys[hi + 1] == keys[hi] + 1 and hi - i < r:
            hi += 1
        sm[k] = arr[lo:hi + 1].mean(axis=0)
    return sm


def _intersects(a, b, pad=4.0) -> bool:
    return a[0] < b[2] + pad and a[2] + pad > b[0] and a[1] < b[3] + pad and a[3] + pad > b[1]


class NameplateLayer:
    """``frame(t_src)`` → transparent 1920×1080 RGBA overlay for the live segment."""

    def __init__(self, tracks: dict, identity: Optional[dict], shots: list[dict], hero_tid: Optional[int] = None,
                 hero_stat: tuple = ("", ""), max_plates: int = 6, out_fps: float = 30.0):
        v = tracks["video"]
        self.fps = float(v["fps"])
        self.sx, self.sy = 1920 / float(v["width"]), 1080 / float(v["height"])
        self.max_plates = max_plates
        self.plates, self.counts = plate_specs(tracks, identity, hero_tid)
        by_id = {tr["track_id"]: tr for tr in tracks["tracks"]}
        wide = [(s["start_f"] + int(CUT_GUARD_S * self.fps), s["end_f"]) for s in shots if s.get("kind") == "wide"]
        self.wide = wide
        for p in self.plates:
            if p.hero and hero_stat[1]:
                p.stat = hero_stat
            boxes = _smooth_boxes(by_id[p.track_id]["frames"], self.fps)
            p.boxes = {f: b for f, b in boxes.items() if any(a <= f <= e for a, e in wide)}
        self.out_fps = out_fps
        self._opacity: dict[tuple[int, int], float] = {}
        self._pos: dict[int, np.ndarray] = {}
        self._last_t: Optional[float] = None
        self.stats = {"frames_with_plates": 0, "max_plates": 0, "plates": self.counts}

    def plan(self, duration: float) -> None:
        """Visibility runs on the output clock: drop short runs, fade in/out inside runs."""
        n = int(math.ceil(duration * self.out_fps))
        for p in self.plates:
            vis = [round(i / self.out_fps * self.fps) in p.boxes for i in range(n)]
            i = 0
            while i < n:
                if not vis[i]:
                    i += 1
                    continue
                j = i
                while j < n and vis[j]:
                    j += 1
                if (j - i) / self.out_fps >= MIN_RUN_S:
                    ramp = max(1, min(int(FADE_S * self.out_fps), (j - i) // 3))
                    for k in range(i, j):
                        self._opacity[(p.track_id, k)] = min(1.0, (k - i + 1) / ramp, (j - k) / ramp)
                i = j

    def frame(self, t_src: float) -> Image.Image:
        img = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
        if self._last_t is not None and abs(t_src - self._last_t) > 0.2:
            self._pos.clear()
        self._last_t = t_src
        k = round(t_src * self.out_fps)
        f = round(t_src * self.fps)
        live = []
        for p in self.plates:
            a = self._opacity.get((p.track_id, k), 0.0)
            b = p.boxes.get(f)
            if a > 0.01 and b is not None:
                box = (b[0] * self.sx, b[1] * self.sy, b[2] * self.sx, b[3] * self.sy)
                live.append((p, a, box))
        # Priority: hero, named over numbered, then larger (nearer) players.
        live.sort(key=lambda x: (not x[0].hero, x[0].tier != "name", -(x[2][3] - x[2][1])))
        live = live[:self.max_plates]
        faces = [(b[0], b[1], b[2], b[1] + (b[3] - b[1]) * 0.22) for _, _, b in live]
        occupied: list[tuple] = []
        draw = ImageDraw.Draw(img)
        shown = 0
        for p, a, box in live:
            chip = name_chip(p.name, p.jersey, p.color, *p.stat)
            w, h = chip.size
            l, t, r, btm = box
            cx = (l + r) / 2
            m = 10
            cands = [(cx - w / 2, t - h - m), (r + m, t - h * 0.3), (l - w - m, t - h * 0.3),
                     (cx - w / 2, t - 2 * h - 2 * m), (cx - w / 2, t - 3 * h - 3 * m)]
            prev = self._pos.get(p.track_id)
            best = None
            for ci, (x, y) in enumerate(cands):
                x, y = min(max(6, x), 1920 - w - 6), min(max(6, y), 1080 - h - 6)
                rect = (x, y, x + w, y + h)
                if any(_intersects(rect, o) for o in occupied) or any(_intersects(rect, fc, 2) for fc in faces):
                    continue
                cost = ci * 40 + (math.dist(prev, (x, y)) if prev is not None else 0)
                if best is None or cost < best[0]:
                    best = (cost, np.array([x, y], float))
            if best is None:
                continue
            pos = best[1]
            if prev is not None and math.dist(prev, pos) < 120:
                pos = prev * 0.6 + pos * 0.4                 # glide, do not jump
            self._pos[p.track_id] = pos
            x, y = int(pos[0]), int(pos[1])
            rect = (x, y, x + w, y + h)
            occupied.append(rect)
            # Thin leader from plate to the head.
            head = (cx, t - 2)
            anchor = (min(max(x + 8, cx), x + w - 8), y + h) if y + h <= t else ((x + w) if x + w <= l else x, y + h / 2)
            draw.line((*anchor, *head), fill=(10, 18, 28, int(110 * a)), width=3)
            draw.line((*anchor, *head), fill=(190, 216, 229, int(150 * a)), width=1)
            img.alpha_composite(with_alpha(chip, a), (x, y))
            shown += 1
        if shown:
            self.stats["frames_with_plates"] += 1
            self.stats["max_plates"] = max(self.stats["max_plates"], shown)
        return img
