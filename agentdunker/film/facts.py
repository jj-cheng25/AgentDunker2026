"""Measured tactical facts for the 3D phases, the commentary and the 3D annotations.

Positions: M1 observed feet → M2 ``H_img2court`` (divided by the third component) → metres, resampled at
10 Hz (gaps ≤ 0.5 s interpolated) and median-smoothed. Ball holder: M5 ``ball_state``. Every fact is a
geometric threshold crossing over a short window, with the numbers that triggered it; nothing is inferred
from appearance. Facts feed (1) phase boundaries and titles (each title must cite a fact id), (2) the only
action words the commentary may use, and (3) M4 annotations in the renderer's own format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FT = 0.3048
BASKET_X_FT = 41.75
STEP = 0.1


@dataclass
class Fact:
    id: str
    type: str                     # pass | screen | drive | cut | closeout | help
    t0: float
    t1: float
    actors: list
    text: str                     # Chinese, uses names from identity labels
    conf: float
    numbers: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "t0": round(float(self.t0), 2), "t1": round(float(self.t1), 2),
                "actors": [int(a) for a in self.actors], "text": self.text, "conf": round(float(self.conf), 2),
                "numbers": {k: float(v) for k, v in self.numbers.items()}}


def court_paths(tracks: dict, court: dict, fps: float) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Common 10 Hz time grid and per-player [n, 2] positions in metres (NaN where unknown)."""
    H = {fr["f"]: np.asarray(fr["H_img2court"], float) for fr in court.get("frames", [])
         if fr.get("ok") and fr.get("H_img2court")}
    t_end = max((fr["f"] for fr in court.get("frames", [])), default=0) / fps
    grid = np.arange(0.0, t_end + STEP / 2, STEP)
    out = {}
    for tr in tracks.get("tracks", []):
        if tr.get("role") != "player":
            continue
        ts, xs, ys = [], [], []
        for fr in tr["frames"]:
            h = H.get(fr["f"])
            if h is None or not fr.get("observed") or fr.get("foot_reliable") is False:
                continue
            v = h @ np.array([fr["foot"][0], fr["foot"][1], 1.0])
            if abs(v[2]) < 1e-9:
                continue
            ts.append(fr["f"] / fps)
            xs.append(v[0] / v[2] * FT)
            ys.append(v[1] / v[2] * FT)
        if len(ts) < 5:
            continue
        ts, xs, ys = map(np.asarray, (ts, xs, ys))
        p = np.full((len(grid), 2), np.nan)
        for i, g in enumerate(grid):
            j = np.searchsorted(ts, g)
            lo, hi = ts[max(0, j - 1)], ts[min(len(ts) - 1, j)]
            if lo - 0.05 <= g <= hi + 0.05 and hi - lo <= 0.5:
                p[i] = [np.interp(g, ts, xs), np.interp(g, ts, ys)]
        for k in range(2):                                  # 5-tap median against foot jitter
            col = p[:, k].copy()
            for i in range(2, len(col) - 2):
                w = col[i - 2:i + 3]
                if not np.isnan(w).any():
                    p[i, k] = np.median(w)
        out[tr["track_id"]] = p
    return grid, out


def _holder_at(events: dict, t: float) -> Optional[int]:
    for s in events.get("ball_state", []):
        if s["t0"] <= t < s["t1"] and s.get("holder") is not None and float(s.get("conf") or 0) >= 0.35:
            return s["holder"]
    return None


def derive(tracks: dict, court: dict, events: dict, fps: float, w0: float, w1: float,
           name, release_t: Optional[float] = None) -> list[Fact]:
    """Facts inside the replay window [w0, w1] (source seconds). ``name(tid, fallback)`` gives labels."""
    grid, P = court_paths(tracks, court, fps)
    team = {tr["track_id"]: tr.get("team") for tr in tracks.get("tracks", [])}
    shooter = (events.get("shot") or {}).get("shooter_track")
    off_team = team.get(shooter) or events.get("offense_team")
    sx = ((events.get("shot") or {}).get("shot_court_xy") or [None])[0]
    if sx is None:
        xs = [np.nanmean(p[:, 0]) for t_, p in P.items() if team.get(t_) == off_team]
        sx = np.nanmean(xs) if xs else 1.0
    basket = np.array([np.sign(sx or 1.0) * BASKET_X_FT * FT, 0.0])
    off = [t for t in P if team.get(t) == off_team]
    dfn = [t for t in P if team.get(t) not in (None, off_team)]
    idx = lambda t: int(round(t / STEP))                       # noqa: E731
    i0, i1 = max(0, idx(w0)), min(len(grid) - 1, idx(w1))
    facts: list[Fact] = []

    def add(typ, t0, t1, actors, text, conf, **nums):
        facts.append(Fact(f"F{len(facts) + 1}", typ, t0, t1, actors, text, conf, nums))

    # M5 events (pass / screen / drive) inside the window keep their own confidence.
    for e in events.get("events", []):
        if not (w0 - 0.3 <= e["t"] <= w1) or float(e.get("conf") or 0) < 0.35:
            continue
        a = e.get("actors") or []
        if e["type"] == "pass" and len(a) >= 2:
            add("pass", e["t"], e.get("t_end", e["t"] + 0.5), a[:2], f"{name(a[0], '持球人')}传给{name(a[1], '队友')}", e["conf"])
        elif e["type"] == "screen" and a:
            add("screen", e["t"], e.get("t_end", e["t"] + 1.0), a[:2], f"{name(a[0], '队友')}掩护", e["conf"])
    # Holder drives: ≥ 2 m closer to the basket within 2 s, ending ≤ 7 m from it.
    last_end = -1.0
    for i in range(i0, i1 - 5):
        t = grid[i]
        h = _holder_at(events, t)
        if h is None or h not in P or t < last_end:
            continue
        j = min(i1, i + 20)
        d0 = np.linalg.norm(P[h][i] - basket)
        d1s = np.linalg.norm(P[h][i:j + 1] - basket, axis=1)
        if np.isnan(d0) or np.all(np.isnan(d1s)):
            continue
        k = int(np.nanargmin(d1s))
        if d0 - d1s[k] >= 2.0 and d1s[k] <= 7.0 and all(_holder_at(events, grid[i + q]) == h for q in range(0, k + 1, 3)):
            add("drive", t, grid[i + k], [h], f"{name(h, '持球人')}突破", 0.7, gain_m=round(d0 - d1s[k], 1))
            last_end = grid[i + k] + 0.5
    # Off-ball cuts: a non-holder attacker ≥ 3 m closer to the basket within 2.5 s, ending ≤ 4.5 m.
    for p in off:
        i = i0
        while i < i1 - 5:
            j = min(i1, i + 25)
            d = np.linalg.norm(P[p][i:j + 1] - basket, axis=1)
            if np.isnan(d[0]) or np.all(np.isnan(d)):
                i += 3
                continue
            k = int(np.nanargmin(d))
            holds = any(_holder_at(events, grid[q]) == p for q in range(i, i + k + 1, 2)) or p == shooter
            if d[0] - d[k] >= 3.0 and d[k] <= 4.5 and not holds:
                add("cut", grid[i], grid[i + k], [p], f"{name(p, '队友')}切入", 0.6, gain_m=round(d[0] - d[k], 1))
                i += k + 10
            else:
                i += 3
    # Closeouts: from 0.3 s before the pass to 1 s after the catch a defender closes ≥ 1.2 m to ≤ 2.0 m.
    for f in [f for f in facts if f.type == "pass"]:
        r, tc = f.actors[1], f.t1
        if r not in P:
            continue
        a, b = max(0, idx(f.t0 - 0.3)), min(i1, idx(tc + 1.0))
        best = None
        for q in dfn:
            if a >= len(grid) or np.isnan(P[q][a]).any() or np.isnan(P[r][a]).any() or np.isnan(P[q][b]).any() or np.isnan(P[r][b]).any():
                continue
            da, db = np.linalg.norm(P[q][a] - P[r][a]), np.linalg.norm(P[q][b] - P[r][b])
            if da - db >= 1.2 and db <= 2.0 and (best is None or db < best[1]):
                best = (q, db, da)
        if best:
            add("closeout", f.t0, grid[b], [best[0], r], f"{name(best[0], '防守人')}扑防", 0.6,
                from_m=round(best[2], 1), to_m=round(best[1], 1))
    # Help: in the last 3 s before the window ends, a defender not guarding the holder moves ≥ 2 m toward
    # him and ends within 3 m.
    h = _holder_at(events, w1 - 0.3)
    if h in P:
        a, b = max(i0, idx(w1 - 3.0)), i1
        if not np.isnan(P[h][a]).any():
            near = min((q for q in dfn if not np.isnan(P[q][a]).any()),
                       key=lambda q: np.linalg.norm(P[q][a] - P[h][a]), default=None)
            for q in dfn:
                if q == near or np.isnan(P[q][a]).any() or np.isnan(P[q][b]).any() or np.isnan(P[h][b]).any():
                    continue
                da, db = np.linalg.norm(P[q][a] - P[h][a]), np.linalg.norm(P[q][b] - P[h][b])
                if da - db >= 2.0 and db <= 3.0:
                    add("help", grid[a], grid[b], [q, h], f"{name(q, '防守人')}协防", 0.6,
                        from_m=round(da, 1), to_m=round(db, 1))
    facts.sort(key=lambda f: f.t0)
    for k, f in enumerate(facts):
        f.id = f"F{k + 1}"
    return facts


ANNO_COLOR = {"pass": "ball", "drive": "off", "cut": "off", "screen": "gold", "closeout": "warn", "help": "def"}


def annotations(facts: list[Fact], w0: float) -> list[dict]:
    """M4/holo-renderer annotations (play time = source − w0, ids ``T<track>``)."""
    out = []
    for f in facts:
        t0, t1 = round(f.t0 - w0, 2), round(f.t1 - w0, 2)
        a = [f"T{x}" for x in f.actors]
        if f.type == "pass":
            out.append({"type": "arrow", "from": a[0], "to": a[1], "t": [t0, t1 + 0.6], "color": "ball", "width": 0.07,
                        "label": f.text})
        elif f.type in ("drive", "cut"):
            out.append({"type": "arrow", "path": {"who": a[0], "t": [t0, t1]}, "t": [t0, t1 + 0.4],
                        "color": "off", "dashed": f.type == "cut", "label": f.text})
        elif f.type == "help":
            out.append({"type": "arrow", "path": {"who": a[0], "t": [t0, t1]}, "t": [t0, t1 + 0.4],
                        "color": "def", "dashed": True, "label": f.text})
        elif f.type == "closeout":
            out.append({"type": "focus", "who": a[0], "t": [t0, t1 + 0.6], "color": "warn", "label": f.text})
        elif f.type == "screen":
            out.append({"type": "zone", "shape": "circle", "at": a[0], "r": 0.9, "t": [t0, t1 + 0.4],
                        "color": "gold", "label": f.text})
    return out


# Action words the commentary may only use when a fact (or the shot record) supports them.
ACTION_WORDS = {"虚晃": None, "假动作": None, "变向": None, "背身": None, "转身": None, "后撤步": None,
                "欧洲步": None, "胯下": None, "背后运球": None, "急停": None, "扣篮": None, "补扣": None,
                "空切": "cut", "切入": "cut", "切走": "cut", "掩护": "screen", "挡拆": "screen",
                "扑防": "closeout", "扑出": "closeout", "扑上": "closeout", "协防": "help", "包夹": "help",
                "突破": "drive", "回传": "pass", "传球": "pass", "分球": "pass", "传给": "pass"}


def unsupported_words(text: str, support: set[str]) -> list[str]:
    return [w for w, need in ACTION_WORDS.items() if w in text and (need is None or need not in support)]
