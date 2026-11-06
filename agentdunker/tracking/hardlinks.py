"""Optional identity pass: confident M3 jersey numbers as hard identity links.

M3 (``agentdunker.identity``) reads numbers on a few well-separated crops of every track of a first
``tracks.json`` and lists, per player, the frames whose crops show the chosen number. Those frames are
mapped back to the exact detections (box IoU against the first tracks), so a number labels
detections, not slots. Downstream (identities / recover):

- a tracklet holding detections of two numbers is cut between them (snapped to a tracker hand-over),
- tracklets / slots of one team with different numbers are never linked,
- equal numbers are linked whatever the gap, slots of one shot with the same number are merged and the
  freed slots refilled, and cross-cut linking follows the numbers.

Without an identity file nothing here runs and the default pipeline is unchanged.
"""
from __future__ import annotations

import logging
import numpy as np

from .identities import Tracklet, _snap_to_handover, _subset, hard_label
from .tracklets import Observations

logger = logging.getLogger(__name__)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def apply_identity(o: Observations, identity: dict, base_tracks: dict, min_conf: float = 0.7,
                   min_iou: float = 0.7) -> dict:
    """Label detections with ``"<team>#<jersey>"`` from M3's confident players (``o.det_jersey``).

    ``base_tracks`` is the tracks.json M3 read; returns a summary for ``meta.identity``.
    """
    if base_tracks["video"].get("sha256") != o.info.sha256 or identity.get("meta", {}).get("video", {}).get(
            "sha256", o.info.sha256) != o.info.sha256:
        logger.warning("identity pass: video hash mismatch, ignoring the identity file")
        return {"used": False, "reason": "video sha256 mismatch"}
    step = {f: k for k, f in enumerate(o.det_frames)}
    tracks = {int(t["track_id"]): t for t in base_tracks["tracks"]}
    o.det_jersey = {}
    sources: dict[str, list[int]] = {}
    n_frames = n_hit = 0
    for tid, p in identity.get("players", {}).items():
        t = tracks.get(int(tid))
        if t is None or t.get("team") not in ("A", "B") or not p.get("jersey") or (p.get("conf") or 0) < min_conf:
            continue
        label = f"{t['team']}#{p['jersey']}"
        frames = {fr["f"]: fr for fr in t["frames"]}
        for f in p.get("evidence_frames", []):
            fr, k = frames.get(f), step.get(f)
            n_frames += 1
            if fr is None or k is None or not fr.get("detected", True) or not len(o.boxes[k]):
                continue
            ious = _iou(np.array(fr["bbox"], float), o.boxes[k])
            i = int(np.argmax(ious))
            if ious[i] >= min_iou:
                o.det_jersey[(k, i)] = label
                n_hit += 1
                sources.setdefault(label, []).append(int(tid))
    sources = {lab: sorted(set(v)) for lab, v in sources.items()}
    logger.info("identity pass: %d numbers, %d/%d evidence frames on detections", len(sources), n_hit, n_frames)
    return {"used": True, "min_conf": min_conf, "numbers": len(sources), "evidence_frames": n_frames,
            "labelled_detections": n_hit, "sources": sources}


def split_numbers(o: Observations, tls: list[Tracklet], min_side_s: float = 0.3) -> list[Tracklet]:
    """Cut tracklets whose labelled detections show different numbers in clean temporal order."""
    if not o.det_jersey:
        return tls
    w = max(3, int(round(min_side_s * o.info.fps / o.timing["stride"])))
    win = int(round(1.0 * o.info.fps / o.timing["stride"]))
    out = []
    for t in tls:
        labs = [(j, o.det_jersey[(int(k), int(i))]) for j, (k, i) in enumerate(zip(t.ks, t.idx))
                if (int(k), int(i)) in o.det_jersey]
        runs: list[list] = []  # [label, first pos, last pos]
        for j, lab in labs:
            if runs and runs[-1][0] == lab:
                runs[-1][2] = j
            else:
                runs.append([lab, j, j])
        if len({r[0] for r in runs}) < 2 or len(runs) != len({r[0] for r in runs}):
            t.jersey = hard_label(o, t.ks, t.idx)  # one number, or interleaved reads (no clean cut)
            out.append(t)
            continue
        E = np.stack([o.embs[k][i] for k, i in zip(t.ks, t.idx)])
        cuts = []
        for a, b in zip(runs[:-1], runs[1:]):
            lo, hi = a[2] + 1, b[1]
            best, cut = -1.0, lo
            for c in range(lo, hi + 1):
                x, y = E[max(0, c - w):c].mean(0), E[c:c + w].mean(0)
                d = 1.0 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
                if d > best:
                    best, cut = d, c
            cuts.append(_snap_to_handover(t, tls, cut, lo, hi, win))
        bounds = [0, *cuts, len(t.ks)]
        for n, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            m = np.zeros(len(t.ks), bool)
            m[a:b] = True
            piece = _subset(o, t, m, t.tid * 1000 + 700 + n)
            piece.split_from, piece.split_kind = t.tid, "jersey"
            out.append(piece)
        logger.info("split tracklet %d by number %s at %s", t.tid, [r[0] for r in runs], [int(t.ks[c]) for c in cuts])
    return out
