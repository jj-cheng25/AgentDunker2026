"""Pick the crops of each player track that are best for reading a jersey number.

A candidate is a detector box of the track (M1 `detected`, else `observed`). Geometry ranks candidates
first (box height, not cut by the frame edge, not overlapped by another person); the best are decoded
and re-ranked by sharpness (variance of the Laplacian on the torso). The final K are spread in time so
the reader sees the player from several angles.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Crop:
    track_id: int
    f: int
    t: float
    bbox: tuple
    score: float
    sharp: float = 0.0
    img: np.ndarray | None = field(default=None, repr=False)


def _overlap(box, others) -> float:
    """Largest fraction of `box` covered by any other box."""
    x1, y1, x2, y2 = box
    a = max(1.0, (x2 - x1) * (y2 - y1))
    best = 0.0
    for o in others:
        iw = min(x2, o[2]) - max(x1, o[0])
        ih = min(y2, o[3]) - max(y1, o[1])
        if iw > 0 and ih > 0:
            best = max(best, iw * ih / a)
    return best


def _geometry_score(box, others, w, h) -> float:
    x1, y1, x2, y2 = box
    bh, bw = y2 - y1, x2 - x1
    if bh < 60 or bw < 20:
        return 0.0
    size = min(1.0, bh / 260.0)
    edge = 0.3 if (x1 < 4 or y1 < 4 or x2 > w - 4 or y2 > h - 4) else 1.0
    iso = max(0.0, 1.0 - 2.5 * _overlap(box, others))
    shape = 1.0 if 1.4 <= bh / max(bw, 1) <= 4.5 else 0.6
    return size * edge * iso * shape


def _crop(frame: np.ndarray, box, pad: float = 0.12) -> np.ndarray:
    """Box plus margin; the area outside the box is dimmed so the tracked person stands out."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh * 0.5))
    X2, Y2 = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh * 0.5))
    c = frame[Y1:Y2, X1:X2].astype(np.float32)
    mask = np.full(c.shape[:2], 0.45, np.float32)
    mask[int(y1) - Y1:int(y2) - Y1, int(x1) - X1:int(x2) - X1] = 1.0
    return (c * mask[..., None]).astype(np.uint8)


def _sharpness(frame: np.ndarray, box) -> float:
    x1, y1, x2, y2 = (int(v) for v in box)
    torso = frame[y1 + (y2 - y1) // 8: y1 + (y2 - y1) * 2 // 3, x1:x2]
    if torso.size == 0:
        return 0.0
    g = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (96, int(96 * g.shape[0] / max(g.shape[1], 1)) or 1))
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def candidates(tracks: list[dict], width: int, height: int, step: int) -> dict[int, list[Crop]]:
    """Geometry-scored candidates for every player track, every `step`-th frame."""
    by_frame = defaultdict(list)
    for t in tracks:
        for fr in t["frames"]:
            if fr.get("observed", True):
                by_frame[fr["f"]].append((t["track_id"], fr["bbox"]))
    out: dict[int, list[Crop]] = defaultdict(list)
    for t in tracks:
        if t.get("role") != "player":
            continue
        for fr in t["frames"]:
            if fr["f"] % step or not fr.get("detected", fr.get("observed", False)):
                continue
            others = [b for tid, b in by_frame[fr["f"]] if tid != t["track_id"]]
            s = _geometry_score(fr["bbox"], others, width, height)
            if s > 0.15:
                out[t["track_id"]].append(Crop(t["track_id"], fr["f"], fr["t"], tuple(fr["bbox"]), s))
    return out


def _spread(cands: list[Crop], k: int, min_dt: float, key) -> list[Crop]:
    picked: list[Crop] = []
    for c in sorted(cands, key=key, reverse=True):
        if all(abs(c.t - p.t) >= min_dt for p in picked):
            picked.append(c)
        if len(picked) == k:
            break
    return picked


def select_crops(video: str, tracks: list[dict], width: int, height: int, fps: float,
                 k: int = 8, pre: int = 30) -> dict[int, list[Crop]]:
    """K crops per player track: geometry pre-selection, then decode and re-rank by sharpness."""
    step = max(1, round(fps / 10))
    cands = candidates(tracks, width, height, step)
    pre_sel = {tid: _spread(cs, pre, 0.1, key=lambda c: c.score) for tid, cs in cands.items()}
    need = defaultdict(list)
    for cs in pre_sel.values():
        for c in cs:
            need[c.f].append(c)
    cap = cv2.VideoCapture(video)
    f = 0
    last = max(need) if need else -1
    while f <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for c in need.get(f, ()):
            c.sharp = _sharpness(frame, c.bbox)
            c.img = _crop(frame, c.bbox)
        f += 1
    cap.release()
    out = {}
    for tid, cs in pre_sel.items():
        cs = [c for c in cs if c.img is not None]
        if not cs:
            continue
        ref = np.median([c.sharp for c in cs]) or 1.0
        final = _spread(cs, k, 0.4, key=lambda c: c.score * min(1.5, c.sharp / ref))
        out[tid] = sorted(final, key=lambda c: c.f)
    return out

def sheet(crops: list[np.ndarray], tile_h: int = 340, cols: int = 4, labels: list[str] | None = None) -> np.ndarray:
    """Numbered grid of crops (labels default to 1..n)."""
    cells = []
    for n, c in enumerate(crops):
        c = cv2.resize(c, (max(8, int(c.shape[1] * tile_h / c.shape[0])), tile_h), interpolation=cv2.INTER_CUBIC)
        c = cv2.copyMakeBorder(c, 36, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(c, labels[n] if labels else str(n + 1), (6, 29), 0, 0.95, (0, 0, 200), 2)
        cells.append(c)
    w = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in cells]
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
