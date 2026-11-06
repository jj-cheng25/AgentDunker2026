"""Link per-frame ball candidates into one ball trajectory (Viterbi with a motion penalty)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import Det


@dataclass(frozen=True)
class BallObs:
    f: int
    xy: tuple[float, float]
    r: float  # ball radius in px
    score: float


@dataclass(frozen=True)
class Rim:
    f: int
    box: tuple[float, float, float, float]
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]


def link_ball(dets: dict[int, list[Det]], fps: float, max_speed: float = 1500.0, miss_cost: float = 0.35,
              min_score: float = 0.12, max_gap_s: float = 0.6) -> dict[int, BallObs]:
    """Choose at most one ball per frame: longest path over detections in a frame-ordered DAG.

    Path value = sum of detector scores - `miss_cost` per sampled frame skipped - a penalty when consecutive
    picks move faster than `max_speed` (px/s at 1080p, a fast pass plus a camera pan). Each node keeps its real
    position, so a gap never inherits a false positive's location.
    """
    frames = sorted(dets)
    if not frames:
        return {}
    idx = {f: i for i, f in enumerate(frames)}
    nodes = [(f, d) for f in frames for d in dets[f] if d.cls == "ball" and d.score >= min_score]
    if not nodes:
        return {}
    max_gap = max_gap_s * fps
    best = np.zeros(len(nodes))
    prev = np.full(len(nodes), -1)
    for j, (fj, dj) in enumerate(nodes):
        cj = dj.center
        rj = (dj.box[2] - dj.box[0] + dj.box[3] - dj.box[1]) / 4
        start = dj.score - miss_cost * idx[fj]
        bv, bp = start, -1
        for i in range(j - 1, -1, -1):
            fi, di = nodes[i]
            if fi == fj:
                continue
            if fj - fi > max_gap:
                break
            skipped = idx[fj] - idx[fi] - 1
            dist = np.hypot(cj[0] - di.center[0], cj[1] - di.center[1])
            excess = dist - max_speed * (fj - fi) / fps - 2.0 * rj
            v = best[i] + dj.score - miss_cost * skipped - max(0.0, excess) / 30.0
            if v > bv:
                bv, bp = v, i
        best[j], prev[j] = bv, bp
    # the path may end anywhere; frames after its end are misses
    final = best - miss_cost * np.array([len(frames) - 1 - idx[f] for f, _ in nodes])
    j = int(final.argmax())
    out: dict[int, BallObs] = {}
    while j >= 0:
        f, d = nodes[j]
        out[f] = BallObs(f, d.center, (d.box[2] - d.box[0] + d.box[3] - d.box[1]) / 4, d.score)
        j = int(prev[j])
    return out


def rims(dets: dict[int, list[Det]], min_score: float = 0.4, smooth_s: float = 0.2, fps: float = 30.0) -> dict[int, Rim]:
    """Best rim box per frame, median-smoothed over +-smooth_s (the box jitters when the ball or a hand
    covers part of the rim). Frames with no rim detection keep no entry; use rim_at() to borrow a neighbour."""
    raw = {}
    for f, ds in dets.items():
        rs = [d for d in ds if d.cls == "rim" and d.score >= min_score]
        if rs:
            d = max(rs, key=lambda d: d.score)
            raw[f] = (np.asarray(d.box), d.score)
    frames = sorted(raw)
    half = max(1, int(smooth_s * fps))
    out = {}
    for i, f in enumerate(frames):
        lo, hi = np.searchsorted(frames, f - half), np.searchsorted(frames, f + half, side="right")
        boxes = np.stack([raw[g][0] for g in frames[lo:hi]])
        out[f] = Rim(f, tuple(float(v) for v in np.median(boxes, axis=0)), raw[f][1])
    return out


def rim_at(rim_by_f: dict[int, Rim], f: int, max_gap: int) -> Rim | None:
    """Nearest rim observation within max_gap frames."""
    best = None
    for g in range(0, max_gap + 1):
        for h in (f - g, f + g):
            if h in rim_by_f:
                return rim_by_f[h]
    return best


def drop_static_near_rim(dets: dict[int, list[Det]], rim_by_f: dict[int, Rim], fps: float, window_s: float = 0.6,
                         tol: float = 0.12, min_frac: float = 0.5) -> dict[int, list[Det]]:
    """Remove ball candidates that stay fixed relative to the rim (net, logo, backboard parts).

    A real ball never rests at the same rim-relative spot for over a second; fixed structures do.
    """
    max_gap = int(fps * 0.3)
    rel: dict[int, list[tuple[float, float]]] = {}
    for f, ds in dets.items():
        r = rim_at(rim_by_f, f, max_gap)
        if r is None or r.width < 8:
            continue
        rel[f] = [((d.center[0] - r.center[0]) / r.width, (d.center[1] - r.center[1]) / r.width)
                  for d in ds if d.cls == "ball"]
    frames = sorted(rel)
    half = int(window_s * fps)
    out = {}
    for f, ds in dets.items():
        if f not in rel:
            out[f] = ds
            continue
        nb = [g for g in frames if g != f and abs(g - f) <= half]
        keep, k = [], 0
        for d in ds:
            if d.cls != "ball":
                keep.append(d)
                continue
            x, y = rel[f][k]
            k += 1
            if abs(x) < 2.5 and abs(y) < 3 and nb:
                hits = sum(any(abs(x - u) < tol and abs(y - v) < tol for u, v in rel[g]) for g in nb)
                if hits >= min_frac * len(nb):
                    continue
            keep.append(d)
        out[f] = keep
    return out
