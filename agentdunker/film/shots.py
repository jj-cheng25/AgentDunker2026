"""Camera shots: deterministic cut detection + wide/close classification.

Cuts come from a thumbnail-difference detector that ignores one-frame photo flashes; M1's shot list is
merged in when available. Each shot is then labelled ``wide`` (game view: nameplates and 3D allowed),
``close`` (close-up/other: no nameplates) by a rule on court calibration and person-box size. An
optional MOSS-VL multiple-choice pass (``moss.classify_shots``) can override the rule.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _thumbs(video: str) -> tuple[np.ndarray, float]:
    """Per-frame 64x36 grey thumbnails, normalised to zero mean / unit variance (flash-invariant)."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        out.append(((g - g.mean()) / (g.std() + 1e-3)).ravel())
    cap.release()
    return np.array(out), fps


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b)) / len(a)


def detect_cuts(video: str, threshold: float = 0.3, min_shot_s: float = 0.4) -> tuple[list[int], int, float]:
    """Frame indices where a new shot starts (excluding 0), total frames, fps.

    A cut is a structural jump (1 − correlation of normalised thumbnails ≥ ``threshold``; c01's real cuts
    score 0.57/0.72, whip pans ≤ 0.14, photo flashes vanish under the normalisation). A jump whose
    picture returns within 0.1 s is a flash or a one-frame glitch, not a cut.
    """
    g, fps = _thumbs(video)
    n = len(g)
    if n < 3:
        return [], n, fps
    d = np.array([0.0] + [_dist(g[i], g[i - 1]) for i in range(1, n)])
    cuts: list[int] = []
    min_gap = max(2, int(min_shot_s * fps))
    back = max(2, int(0.1 * fps))
    for i in np.flatnonzero(d >= threshold):
        if i < min_gap or i > n - 2:
            continue
        j = min(n - 1, i + back)
        if _dist(g[i - 1], g[j]) < threshold * 0.5:
            continue
        if cuts and i - cuts[-1] < min_gap:
            if d[i] > d[cuts[-1]]:
                cuts[-1] = int(i)
            continue
        cuts.append(int(i))
    return cuts, n, fps


def build_shots(video: str, n_frames: int, fps: float, tracks: Optional[dict]) -> list[dict]:
    """Shot list [{start_f, end_f, t0, t1}] from own cuts, merged with M1's shots when given."""
    try:
        cuts, n, fps2 = detect_cuts(video)
        n_frames = n or n_frames
        fps = fps2 or fps
    except Exception as e:                       # never fatal: one shot spanning the clip
        logger.warning("cut detection failed: %s", e)
        cuts = []
    m1 = [s["start_f"] for s in (tracks or {}).get("shots", [])[1:]]
    merged: list[int] = []
    for c in sorted(set(cuts) | set(m1)):
        if merged and c - merged[-1] < int(0.4 * fps):
            continue
        merged.append(c)
    bounds = [0, *merged, n_frames]
    return [{"start_f": a, "end_f": b - 1, "t0": round(a / fps, 3), "t1": round(b / fps, 3)}
            for a, b in zip(bounds, bounds[1:]) if b > a]


def classify_rule(shots: list[dict], tracks: Optional[dict], court: Optional[dict], height: int) -> list[dict]:
    """``wide`` iff the court is calibrated on most frames and players look small enough.

    Rule: court ok on ≥ 60 % of the shot's frames, and median player box height ≤ 0.42 × frame height with
    ≥ 4 players visible on a typical frame. Without court/tracks the shot is ``wide`` only if no data says
    otherwise (conf 0.3), so the film still shows a plain cut.
    """
    ok_by_f = {}
    if court:
        ok_by_f = {fr["f"]: bool(fr.get("ok")) for fr in court.get("frames", [])}
    per_f: dict[int, list[float]] = {}
    for tr in (tracks or {}).get("tracks", []):
        if tr.get("role") != "player":
            continue
        for fr in tr["frames"]:
            if fr.get("observed"):
                b = fr["bbox"]
                per_f.setdefault(fr["f"], []).append(b[3] - b[1])
    out = []
    for s in shots:
        fs = range(s["start_f"], s["end_f"] + 1)
        ok_frac = np.mean([ok_by_f.get(f, False) for f in fs]) if ok_by_f else None
        heights = [h for f in fs for h in per_f.get(f, [])]
        counts = [len(per_f[f]) for f in fs if f in per_f]
        med_h = float(np.median(heights)) / height if heights else None
        n_players = float(np.median(counts)) if counts else 0.0
        if ok_frac is None and med_h is None:
            kind, conf = "wide", 0.3
        else:
            court_ok = ok_frac is None or ok_frac >= 0.6
            small = med_h is not None and med_h <= 0.42 and n_players >= 4
            kind = "wide" if court_ok and (small or med_h is None) else "close"
            conf = 0.9 if (court_ok == small) else 0.6
        out.append({**s, "kind": kind, "conf": conf, "source": "rule",
                    "court_ok_frac": None if ok_frac is None else round(float(ok_frac), 3),
                    "player_h_frac": None if med_h is None else round(med_h, 3), "n_players": n_players})
    return out
