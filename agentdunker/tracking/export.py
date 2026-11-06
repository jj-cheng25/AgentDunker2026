"""Chains -> contract tracks.json (every frame, observed flag, foot reliability)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .identities import Chain
from .tracklets import Observations
from .video import warp_box

ROLE = {"A": "player", "B": "player", "REF": "referee"}


def _jersey_rgb(o: Observations, groups: dict[str, list[Chain]]) -> dict[str, list[int]]:
    """Dominant torso colour per group: k-means (k=3) over central-torso pixels of up to 60 crops."""
    import cv2
    from sklearn.cluster import KMeans

    out = {}
    for g, chs in groups.items():
        keys = [(k, i) for ch in chs for k, (i, _) in ch.obs.items() if (k, i) in o.crops]
        px = []
        for k, i in keys[:: max(1, len(keys) // 60)]:
            c = cv2.imdecode(np.frombuffer(o.crops[(k, i)], np.uint8), cv2.IMREAD_COLOR)
            h, w = c.shape[:2]
            px.append(c[int(0.2 * h) : int(0.42 * h), int(0.33 * w) : int(0.67 * w)].reshape(-1, 3))
        if not px:
            out[g] = [128, 128, 128]
            continue
        X = np.concatenate(px).astype(np.float32)[::4]
        km = KMeans(3, n_init=3, random_state=0).fit(X)
        bgr = km.cluster_centers_[np.bincount(km.labels_).argmax()]
        out[g] = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
    return out


def _foot_reliable(stab: np.ndarray, img: np.ndarray, obs: np.ndarray, H: int, win: int) -> np.ndarray:
    """False where the foot point is cut off (frame bottom) or lifted off a smooth ground track (jump)."""
    h = stab[:, 3] - stab[:, 1]
    base = median_filter(stab[:, 3], size=max(3, win), mode="nearest")
    lifted = (base - stab[:, 3]) > 0.1 * np.maximum(h, 1)
    cut = img[:, 3] >= H - 3
    return obs & ~lifted & ~cut


def _segment_frames(o: Observations, det: dict, lo: int, hi: int, stride: int,
                    max_hold: float | None = None) -> list[dict]:
    """Frames lo..hi-1 of one shot from that shot's observations (camera-stabilised interpolation).

    ``max_hold`` (seconds): omit predicted frames held longer than that before the first / after the last
    observation, and predicted boxes mostly outside the image. None keeps every frame of the shot.
    """
    fs = np.array(sorted(det))
    S = np.stack([det[f][0] for f in fs])
    C = np.array([det[f][1] for f in fs])
    all_f = np.arange(lo, hi)
    stab = np.stack([np.interp(all_f, fs, S[:, j]) for j in range(4)], 1)
    conf = np.interp(all_f, fs, C)
    pos = np.searchsorted(fs, all_f)
    prev_f = fs[np.clip(pos - 1, 0, len(fs) - 1)]
    next_f = fs[np.clip(pos, 0, len(fs) - 1)]
    exact = np.isin(all_f, fs)
    inside = (all_f >= fs[0]) & (all_f <= fs[-1])
    observed = exact | (inside & ((next_f - prev_f) <= stride))
    gap_s = np.where(observed, 0.0, np.minimum(np.abs(all_f - prev_f), np.abs(next_f - all_f)) / o.info.fps)
    conf = np.where(observed, conf, conf * 0.5 * np.exp(-gap_s))
    img = np.stack([warp_box(np.linalg.inv(o.to_ref[f]), b) for f, b in zip(all_f, stab)])
    rel = _foot_reliable(stab, img, observed, o.info.height, int(o.info.fps * 0.8) | 1)
    keep = np.ones(len(all_f), bool)
    if max_hold is not None:
        W, H = o.info.width, o.info.height
        wh = np.maximum(img[:, 2:] - img[:, :2], 1)
        vis = (np.clip(img[:, 2], 0, W) - np.clip(img[:, 0], 0, W)) * (np.clip(img[:, 3], 0, H) - np.clip(img[:, 1], 0, H))
        off = vis < 0.5 * wh[:, 0] * wh[:, 1]
        keep = observed | (~off & (inside | (gap_s <= max_hold)))
    out = []
    for j, f in enumerate(all_f):
        if not keep[j]:
            continue
        b = img[j]
        out.append({
            "f": int(f), "t": round(float(o.times[f]), 4),
            "bbox": [round(float(v), 1) for v in b],
            "foot": [round(float((b[0] + b[2]) / 2), 1), round(float(b[3]), 1)],
            "conf": round(float(conf[j]), 3), "observed": bool(observed[j]),
            "detected": bool(exact[j]), "foot_reliable": bool(rel[j]),
        })
    return out


def chain_frames(o: Observations, ch: Chain, stride: int, max_hold: float | None = None) -> list[dict]:
    """All frames of every shot in which the identity was observed (shots without it are omitted)."""
    bounds = [*o.shots, len(o.times)]
    by_shot: dict[int, dict] = {}
    for k, (i, s) in ch.obs.items():
        f = o.det_frames[k]
        by_shot.setdefault(o.shot_of(f), {})[f] = (warp_box(o.to_ref[f], o.boxes[k][i]), s)
    out = []
    for shot in sorted(by_shot):
        out += _segment_frames(o, by_shot[shot], bounds[shot], bounds[shot + 1], stride, max_hold)
    return out


def to_contract(o: Observations, chains: list[Chain], meta: dict, max_hold: float | None = None) -> dict:
    stride = int(o.timing.get("stride", 1))
    order = sorted(chains, key=lambda c: ({"A": 0, "B": 1, "REF": 2}[c.group], min(c.obs)))
    groups: dict[str, list[Chain]] = {}
    for ch in order:
        groups.setdefault(ch.group, []).append(ch)
    jersey = _jersey_rgb(o, groups)
    ident = meta.get("identity") or {}
    tracks = []
    for i, ch in enumerate(order, 1):
        tracks.append({
            "track_id": i, "role": ROLE[ch.group], "team": ch.group if ch.group != "REF" else None,
            "team_color": jersey[ch.group],
            "group_conf": round(float(np.average([m.group_conf for m in ch.members],
                                                 weights=[m.weight for m in ch.members])), 3),
            "jersey_hint": ch.number(),
            "source_tracklets": [int(m.tid) for m in ch.members],
            "shot_links": ch.shot_links,
            "frames": chain_frames(o, ch, stride, max_hold),
        })
        if ident.get("used"):  # identity pass only: the hard number and the first-pass tracks it came from
            lab = ch.jersey()
            tracks[-1]["identity"] = None if lab is None else {
                "jersey": lab.split("#", 1)[1], "from_tracks": ident["sources"].get(lab, [])}
    i = o.info
    return {
        "video": {"path": i.path, "sha256": i.sha256, "fps": i.fps, "width": i.width, "height": i.height,
                  "n_frames": i.n_frames},
        "tracks": tracks,
        "shots": [{"start_f": int(a), "end_f": int(b) - 1} for a, b in zip(o.shots, [*o.shots[1:], len(o.times)])],
        "meta": meta,
    }
