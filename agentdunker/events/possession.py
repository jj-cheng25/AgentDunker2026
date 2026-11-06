"""Ball holder per frame and the ball-state timeline (hold / dribble / pass / shot / loose)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .balltrack import BallObs
from .detector import Det
from .release import _iou
from .tracks import Tracks, contact

HOLD_GAP = 0.04          # ball-to-box gap (player heights) for a holder candidate
DRIBBLE_LOW = 0.55       # ball below this fraction of the holder's box height = on the bounce
BRIDGE_S = 0.6           # same holder on both sides of a gap shorter than this = one segment


@dataclass
class HolderObs:
    f: int
    track: int | None     # None = ball seen but nobody holds it
    low: float | None     # ball height in the holder box (0 top .. 1 feet)
    conf: float


@dataclass
class Segment:
    type: str             # hold | dribble | pass | shot | loose
    f0: int
    f1: int
    holder: int | None = None
    frm: int | None = None
    to: int | None = None
    conf: float = 0.0
    extra: dict = field(default_factory=dict)


def ambiguous_frames(ball: dict[int, BallObs], dets: dict[int, list[Det]], tracks: Tracks, frames: list[int]) -> list[int]:
    """Frames where the ball touches 2+ player boxes and no possession box settles it (worth a pose look)."""
    out = []
    for f in frames:
        b = ball.get(f)
        if b is None:
            continue
        cands = [p for p in tracks.players(f) if contact(b.xy, b.r, p.box) <= HOLD_GAP]
        poss = [d for d in dets.get(f, []) if d.cls == "player-in-possession" and d.score > 0.5]
        if len(cands) > 1 and not any(_iou(p.box, d.box) > 0.5 for p in cands for d in poss):
            out.append(f)
    return out


def holders(ball: dict[int, BallObs], dets: dict[int, list[Det]], tracks: Tracks, frames: list[int],
            people: dict | None = None) -> list[HolderObs]:
    """people: optional {frame: [pose.Person]} used to settle ambiguous frames by the nearest wrist."""
    from .release_pose import track_of, wrist_dist

    out = []
    for f in frames:
        b = ball.get(f)
        poss = [d for d in dets.get(f, []) if d.cls == "player-in-possession" and d.score > 0.5]
        players = tracks.players(f)
        if b is None:
            # ball hidden in hands: trust a confident possession box matched to a track
            best = max(((_iou(p.box, d.box), p, d) for p in players for d in poss), key=lambda x: x[0], default=None)
            if best and best[0] > 0.5:
                out.append(HolderObs(f, best[1].track_id, None, 0.5 * best[2].score))
            continue
        cands = [(contact(b.xy, b.r, p.box), p) for p in players]
        cands = [(c, p) for c, p in cands if c <= HOLD_GAP]
        if not cands:
            out.append(HolderObs(f, None, None, b.score))
            continue

        def key(cp):
            c, p = cp
            pbonus = max((_iou(p.box, d.box) for d in poss), default=0.0)
            cx = abs(b.xy[0] - (p.box[0] + p.box[2]) / 2) / max(1.0, p.box[2] - p.box[0])
            return c * 10 + cx - pbonus

        c, p = min(cands, key=key)
        ambiguous = len(cands) > 1 and not any(_iou(p.box, d.box) > 0.5 for d in poss)
        if ambiguous and people and f in people:
            d, who = wrist_dist(b, people[f])
            tid = track_of(who, tracks, f) if who is not None and d is not None and d <= 2.0 else None
            pick = next((q for _, q in cands if q.track_id == tid), None)
            if pick is not None:
                p, ambiguous = pick, False
        low = (b.xy[1] - p.box[1]) / max(1.0, p.box[3] - p.box[1])
        out.append(HolderObs(f, p.track_id, float(low), b.score * (0.6 if ambiguous else 1.0)))
    return out


def _runs(obs: list[HolderObs], fps: float, tracks: Tracks) -> list[list[HolderObs]]:
    """Group consecutive observations of the same holder, then clean the sequence.

    - a short run of X between two runs of the same holder H is H being tightly guarded (merged into H);
    - a short run of the other team between two teammates is the ball flying past a defender (dropped);
    - one-sample runs with no support are dropped.
    """
    runs: list[list[HolderObs]] = []
    for o in obs:
        if o.track is None:
            continue
        if runs and runs[-1][-1].track == o.track and (o.f - runs[-1][-1].f) / fps <= BRIDGE_S:
            runs[-1].append(o)
        else:
            runs.append([o])
    dur = lambda r: (r[-1].f - r[0].f) / fps
    team = tracks.team_of
    changed = True
    while changed:
        changed = False
        for i in range(1, len(runs) - 1):
            a, x, b = runs[i - 1], runs[i], runs[i + 1]
            h = a[-1].track
            if b[0].track == h and dur(x) < 0.7 and team(x[0].track) != team(h):
                runs[i] = [HolderObs(o.f, h, o.low, o.conf * 0.7) for o in x]
                changed = True
            elif (dur(x) < 0.5 and team(a[-1].track) == team(b[0].track) is not None
                  and team(x[0].track) != team(a[-1].track)):
                del runs[i]
                changed = True
            if changed:
                break
        if not changed:
            for i, r in enumerate(runs):
                if len(r) == 1 and len(runs) > 1:
                    del runs[i]
                    changed = True
                    break
        merged: list[list[HolderObs]] = []
        for r in runs:
            if merged and merged[-1][-1].track == r[0].track and (r[0].f - merged[-1][-1].f) / fps <= 2 * BRIDGE_S:
                merged[-1].extend(r)
            else:
                merged.append(r)
        runs = merged
    return runs


def _split_dribble(run: list[HolderObs], fps: float) -> list[Segment]:
    """Split one holder run into hold / dribble parts from the ball height in the holder's box."""
    win = 0.45 * fps
    labels = []
    for o in run:
        near = [q.low for q in run if q.low is not None and abs(q.f - o.f) <= win]
        labels.append("dribble" if near and max(near) > DRIBBLE_LOW else "hold")
    segs: list[Segment] = []
    for o, lab in zip(run, labels):
        if segs and segs[-1].type == lab:
            segs[-1].f1 = o.f
            segs[-1].extra["n"] += 1
            segs[-1].extra["s"] += o.conf
        else:
            segs.append(Segment(lab, o.f, o.f, holder=o.track, extra={"n": 1, "s": o.conf}))
    for s in segs:
        s.conf = round(s.extra.pop("s") / s.extra.pop("n"), 3)
    # tiny segments (< 0.25 s) take the label of the previous one
    out: list[Segment] = []
    for s in segs:
        if out and (s.f1 - s.f0) / fps < 0.25:
            out[-1].f1 = s.f1
        else:
            out.append(s)
    for a, b in zip(out, out[1:]):
        a.f1 = b.f0
    return out


def ball_states(obs: list[HolderObs], tracks: Tracks, fps: float, n_frames: int,
                shots: list[dict] | None = None) -> list[Segment]:
    """shots: [{"f_release": float, "f_end": int, "shooter": int|None, "conf": float}] split possessions."""
    shots = sorted(shots or [], key=lambda s: s["f_release"])
    # holder observations inside a shot's flight are not possession
    obs = [o for o in obs if not any(s["f_release"] < o.f < s["f_end"] for s in shots)]
    runs = _runs(obs, fps, tracks)
    segs: list[Segment] = []
    for r in runs:
        segs.extend(_split_dribble(r, fps))
    for s in shots:
        segs.append(Segment("shot", int(round(s["f_release"])), int(s["f_end"]), frm=s.get("shooter"),
                            conf=float(s.get("conf", 0.5))))
    segs.sort(key=lambda s: s.f0)
    # fill gaps between consecutive segments with pass / loose
    filled: list[Segment] = []
    for s in segs:
        if filled:
            p = filled[-1]
            if s.f0 > p.f1 + 1:
                a = p.holder if p.type in ("hold", "dribble") else None
                b = s.holder if s.type in ("hold", "dribble") else s.frm if s.type == "shot" else None
                ta, tb = tracks.team_of(a), tracks.team_of(b)
                if a is not None and b is not None and a != b and ta is not None and ta == tb and s.type != "shot":
                    dur = (s.f0 - p.f1) / fps
                    filled.append(Segment("pass", p.f1, s.f0, frm=a, to=b,
                                          conf=round(min(p.conf, s.conf) * (1.0 if dur < 1.5 else 0.6), 3)))
                elif not (a is not None and a == b):
                    filled.append(Segment("loose", p.f1, s.f0, conf=0.4))
                else:
                    p.f1 = s.f0          # same holder across an unobserved gap
            elif s.f0 < p.f1:
                p.f1 = s.f0
        filled.append(s)
    if filled and filled[-1].f1 < n_frames - 1 and filled[-1].type == "shot":
        filled.append(Segment("loose", filled[-1].f1, n_frames - 1, conf=0.3))
    # a hold/dribble straddling into a pass end is a catch handled by actions; merge adjacent same-holder segments
    out: list[Segment] = []
    for s in filled:
        if out and s.type == out[-1].type and s.type in ("hold", "dribble") and s.holder == out[-1].holder:
            out[-1].f1 = s.f1
            out[-1].conf = round((out[-1].conf + s.conf) / 2, 3)
        else:
            out.append(s)
    return out
