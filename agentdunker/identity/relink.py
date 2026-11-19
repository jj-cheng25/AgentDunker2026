"""Re-link same-team tracks that swapped people at a crossing, with the jersey reads as evidence.

M1 can swap two team-mates when their boxes overlap (screens, the scrum under the rim after a shot): the
uniforms look alike to re-ID. Every read crop is a time-stamped observation of a number, so a track whose
reads change number after it crossed a team-mate is evidence of a swap. Crossings are visited in time
order; at each one the two tracks stay as they are unless exchanging their tails (every frame after the
crossing) removes a contradiction between the reads before and after it. A decision is applied at once,
so later crossings and reads see the corrected tracks. Without reads on both sides of a crossing nothing
changes: no evidence, no swap.
"""
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field

IOU_MIN = float(os.environ.get("ADK_ID_RELINK_IOU", 0.2))   # same-team boxes this close can be swapped
GAP_X = float(os.environ.get("ADK_ID_RELINK_GAP", 0.0))     # also: feet within GAP_X box widths (0 = off)
MIN_WEIGHT = 2       # a number needs this much read weight on one side to count (one agreed crop = 3)
MARGIN = 2           # a swap must remove at least this much contradicted weight


@dataclass
class Tile:
    """One read crop: the frame, every model's number for it, the view, and the crop itself."""
    f: int
    numbers: dict = field(default_factory=dict)     # model -> number
    view: str | None = None
    crop: object = None

    def weights(self) -> Counter:
        """Same weighting as Reading.support: two models agreeing weigh 3, one model 1, disagreement 0."""
        ns = set(self.numbers.values())
        if len(ns) != 1:
            return Counter()
        return Counter({ns.pop(): 3 if len(self.numbers) > 1 else 1})


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _near(a, b, k: float) -> bool:
    """Foot points (bottom centres) within k mean box widths: a hand-over through occlusion, no overlap."""
    w = ((a[2] - a[0]) + (b[2] - b[0])) / 2
    return math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, a[3] - b[3]) <= k * w


def crossings(players: list[dict], fps: float, iou_min: float = IOU_MIN) -> list[dict]:
    """Same-team box overlaps: one event per contiguous run per pair, at the frame of highest IoU."""
    by_f = defaultdict(list)
    for t in players:
        for fr in t["frames"]:
            by_f[fr["f"]].append((t["track_id"], t.get("team"), fr["bbox"]))
    gap = max(1, round(fps / 5))
    runs: dict[tuple, list] = defaultdict(list)
    for f in sorted(by_f):
        rows = by_f[f]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (ta, ga, ba), (tb, gb, bb) = rows[i], rows[j]
                if ga is None or ga != gb:
                    continue
                v = _iou(ba, bb)
                if v < iou_min and not (GAP_X and _near(ba, bb, GAP_X)):
                    continue
                r = runs[(min(ta, tb), max(ta, tb))]
                if r and f - r[-1][1] <= gap:
                    r[-1][1] = f
                    if v > r[-1][3]:
                        r[-1][2], r[-1][3] = f, v
                else:
                    r.append([f, f, f, v])
    out = [{"f": best, "a": a, "b": b, "iou": round(v, 3), "span": [f0, f1]}
           for (a, b), rs in runs.items() for f0, f1, best, v in rs]
    return sorted(out, key=lambda c: c["f"])


def _votes(tiles: list[Tile], lo: float, hi: float) -> Counter:
    v = Counter()
    for t in tiles:
        if lo < t.f <= hi:
            v.update(t.weights())
    return v


def _contra(x: Counter, y: Counter) -> int:
    """Read weight contradicted when x and y are the same person: >0 different numbers, <0 agreement."""
    if not x or not y:
        return 0
    (nx, wx), (ny, wy) = x.most_common(1)[0], y.most_common(1)[0]
    if wx < MIN_WEIGHT or wy < MIN_WEIGHT:
        return 0
    return min(wx, wy) if nx != ny else -min(wx, wy)


def relink(players: list[dict], tiles: dict[int, list[Tile]], fps: float) -> list[dict]:
    """Swap the tails of tracks in place (`players` frames and `tiles`); return the swaps made."""
    cross = crossings(players, fps)
    by_id = {t["track_id"]: t for t in players}
    swaps = []
    for k, c in enumerate(cross):
        a, b, f = c["a"], c["b"], c["f"]
        nxt = lambda tid: next((x["f"] for x in cross[k + 1:] if tid in (x["a"], x["b"])), math.inf)  # noqa: E731
        ta, tb = tiles.get(a, []), tiles.get(b, [])
        a0, a1 = _votes(ta, -math.inf, f), _votes(ta, f, nxt(a))
        b0, b1 = _votes(tb, -math.inf, f), _votes(tb, f, nxt(b))
        keep = _contra(a0, a1) + _contra(b0, b1)
        swap = _contra(a0, b1) + _contra(b0, a1)
        if keep - swap < MARGIN:
            continue
        fa, fb = by_id[a]["frames"], by_id[b]["frames"]
        by_id[a]["frames"] = [x for x in fa if x["f"] <= f] + [x for x in fb if x["f"] > f]
        by_id[b]["frames"] = [x for x in fb if x["f"] <= f] + [x for x in fa if x["f"] > f]
        tiles[a] = [t for t in ta if t.f <= f] + [t for t in tb if t.f > f]
        tiles[b] = [t for t in tb if t.f <= f] + [t for t in ta if t.f > f]
        for x in cross[k + 1:]:                 # boxes after f now belong to the other track
            x["a"], x["b"] = ({a: b, b: a}.get(x["a"], x["a"]), {a: b, b: a}.get(x["b"], x["b"]))
        swaps.append({"f": f, "t": round(f / fps, 3), "tracks": [a, b], "iou": c["iou"],
                      "evidence": {"keep": keep, "swap": swap,
                                   "before": [dict(a0), dict(b0)], "after": [dict(a1), dict(b1)]}})
    return swaps


def to_tiles(crops: dict, readings: dict) -> dict[int, list[Tile]]:
    """Per track, one Tile per read crop (crops are sorted by frame; reads are keyed by 1-based tile)."""
    out = {}
    for tid, cs in crops.items():
        rd = readings.get(tid)
        out[tid] = [Tile(c.f, {m: r[i + 1] for m, r in (rd.reads.items() if rd else ()) if i + 1 in r},
                         (rd.views.get(i + 1) if rd else None), c) for i, c in enumerate(cs)]
    return out


def from_tiles(tiles: dict[int, list[Tile]], readings: dict):
    """(crops, readings) rebuilt from relinked tiles; referee votes stay with the track."""
    from .jersey import Reading

    crops, out = {}, {}
    for tid, ts in tiles.items():
        ts = sorted(ts, key=lambda t: t.f)
        models = {m for t in ts for m in t.numbers}
        old = readings.get(tid)
        out[tid] = Reading(tid, reads={m: {i + 1: t.numbers[m] for i, t in enumerate(ts) if m in t.numbers}
                                       for m in models},
                           views={i + 1: t.view for i, t in enumerate(ts) if t.view},
                           referee_votes=old.referee_votes if old else 0, frames=[t.f for t in ts])
        crops[tid] = [t.crop for t in ts]
    return crops, out


def apply_relinks(tracks: dict, relinks: list[dict]) -> dict:
    """tracks.json with the swaps from identity.json applied (same order as they were decided)."""
    out = {**tracks, "tracks": [dict(t) for t in tracks.get("tracks", [])]}
    by_id = {t["track_id"]: t for t in out["tracks"]}
    for s in relinks:
        a, b = s["tracks"]
        f = s["f"]
        if a not in by_id or b not in by_id:
            continue
        fa, fb = by_id[a]["frames"], by_id[b]["frames"]
        by_id[a]["frames"] = [x for x in fa if x["f"] <= f] + [x for x in fb if x["f"] > f]
        by_id[b]["frames"] = [x for x in fb if x["f"] <= f] + [x for x in fa if x["f"] > f]
    if relinks:
        out["relinked"] = {"by": "identity", "swaps": [{"f": s["f"], "tracks": s["tracks"]} for s in relinks]}
    return out
