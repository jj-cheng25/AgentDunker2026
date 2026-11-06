"""Release frame and shooter: walk back from the rim along the ball's flight to the hand it left."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .balltrack import BallObs, Rim, rim_at
from .detector import Det
from .tracks import Tracks, contact

HOLD = 0.03      # ball-to-box gap (player heights) that counts as "in hand"
SHOT_POSES = ("player-jump-shot", "player-layup-dunk")


@dataclass
class Release:
    f: float                    # release frame (fractional: midpoint between last contact and first free frame)
    shooter_track: int | None
    flight_start: int           # first frame of the fitted flight
    conf: float
    evidence: dict


def _flight_start(ball: dict[int, BallObs], rim_by_f: dict[int, Rim], f_rim: int, fps: float) -> tuple[int, float]:
    """Earliest frame such that the ball path from it to f_rim fits a camera-compensated parabola.

    Coordinates are rim-relative (rim widths), which removes camera pan; x is fitted linear and y quadratic in t.
    Returns (frame, rms in ball diameters of the last accepted fit).
    """
    pts = []
    for f in sorted(ball):
        if f > f_rim:
            break
        r = rim_at(rim_by_f, f, int(fps * 0.3))
        if r is None or r.width < 8:
            continue
        b = ball[f]
        pts.append((f, (b.xy[0] - r.center[0]) / r.width, (b.xy[1] - r.center[1]) / r.width, b.r / r.width))
    if len(pts) < 4:
        return (pts[0][0] if pts else f_rim), 0.0
    pts = pts[::-1]  # backwards from the rim
    start, rms_ok = pts[min(3, len(pts) - 1)][0], 0.0
    for k in range(4, len(pts) + 1):
        seg = np.array(pts[:k])
        t = (seg[:, 0] - seg[0, 0]) / fps
        diam = 2 * np.median(seg[:, 3])
        rx = seg[:, 1] - np.polyval(np.polyfit(t, seg[:, 1], 1), t)
        ry = seg[:, 2] - np.polyval(np.polyfit(t, seg[:, 2], 2), t)
        res = np.hypot(rx, ry) / diam
        # a new point must fit, and the whole segment must stay smooth
        if res[-1] > 1.2 or np.sqrt(np.mean(res ** 2)) > 0.6:
            break
        start, rms_ok = int(seg[-1, 0]), float(np.sqrt(np.mean(res ** 2)))
    return start, rms_ok


def find_release(ball: dict[int, BallObs], rim_by_f: dict[int, Rim], tracks: Tracks, dets: dict[int, list[Det]],
                 f_rim: int, fps: float) -> Release | None:
    """f_rim: first frame the ball is near the rim on this attempt."""
    fs, rms = _flight_start(ball, rim_by_f, f_rim, fps)
    frames = sorted(f for f in ball if f <= f_rim)
    if not frames:
        return None
    # who holds the ball just before (and at) the flight start; look back up to 1 s
    window = [f for f in frames if fs - fps * 1.0 <= f <= fs + fps * 0.15]
    votes: dict[int, float] = {}
    for f in window:
        b = ball[f]
        poses = [d for d in dets.get(f, []) if d.cls in SHOT_POSES and d.score > 0.3 and contact(b.xy, b.r, d.box) <= HOLD]
        for p in tracks.players(f):
            c = contact(b.xy, b.r, p.box)
            if c <= HOLD:
                w = 1.0 + (f - window[0]) / max(1, fs - window[0] + 1)  # later contact weighs more
                if any(_iou(p.box, d.box) > 0.5 for d in poses):
                    w += 1.0
                votes[p.track_id] = votes.get(p.track_id, 0.0) + w
    shooter = max(votes, key=votes.get) if votes else None
    # release = last frame the shooter still touches the ball at or before the flight start + small slack,
    # followed by a frame where the ball is free of the shooter
    f_rel, last_touch = float(fs), None
    if shooter is not None:
        for f in frames:
            if f > fs + fps * 0.3:
                break
            p = tracks.box(shooter, f)
            if p is not None and contact(ball[f].xy, ball[f].r, p.box) <= HOLD:
                last_touch = f
        if last_touch is not None:
            nxt = [f for f in frames if f > last_touch]
            f_rel = (last_touch + nxt[0]) / 2 if nxt else float(last_touch)
    total = sum(votes.values()) or 1.0
    share = votes.get(shooter, 0.0) / total if shooter is not None else 0.0
    conf = round(min(1.0, 0.4 + 0.4 * share + (0.2 if last_touch is not None else 0.0)) * (1.0 if rms < 0.5 else 0.8), 3)
    return Release(f_rel, shooter, fs, conf, {"flight_rms_diam": round(rms, 3), "votes": votes,
                                               "last_touch": last_touch})


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
