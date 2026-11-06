"""Release from wrist keypoints: the ball leaves the hand when its distance to the nearest wrist goes above
~2 ball diameters and stays there. Every such hand-off that is followed by the ball reaching the rim is an attempt
(this also separates a missed layup from the putback that follows it)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .balltrack import BallObs, Rim, rim_at
from .pose import Person, PoseNear
from .tracks import Tracks

HELD_D = 2.0        # wrist-to-ball-centre distance in ball diameters that still counts as "in hand"
MIN_HOLD_S = 0.15   # one person must keep the ball this long (a contesting hand or a ball flying past is shorter)


class PoseCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[int, list[Person]] = {}
        if path and path.exists():
            for f, ps in json.loads(path.read_text()).items():
                self.data[int(f)] = [Person(tuple(p["box"]), [tuple(w) for w in p["wrists"]],
                                            [tuple(e) for e in p["elbows"]], p["conf"]) for p in ps]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({str(f): [{"box": [round(v, 1) for v in p.box],
                                                       "wrists": [[round(v, 2) for v in w] for w in p.wrists],
                                                       "elbows": [[round(v, 2) for v in e] for e in p.elbows],
                                                       "conf": round(p.conf, 3)} for p in ps]
                                             for f, ps in sorted(self.data.items())}))


@dataclass
class HandOff:
    f: float                 # release frame (midpoint of last held / first free sample)
    f_held: int
    person: Person
    conf: float
    held: list[tuple[int, float, Person]]  # (frame, wrist distance, person) during the hold before release


def people_near_ball(video: str, ball: dict[int, BallObs], frames: list[int], cache: PoseCache,
                     pose: PoseNear | None) -> dict[int, list[Person]]:
    todo = [f for f in frames if f in ball and f not in cache.data]
    if todo:
        pose = pose or PoseNear()
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, todo[0])
        f, want = todo[0], set(todo)
        while f <= todo[-1]:
            if f in want:
                ok, im = cap.read()
                if not ok:
                    break
                cache.data[f] = pose.people(im, ball[f].xy)
            elif not cap.grab():
                break
            f += 1
        cap.release()
        cache.save()
    return {f: cache.data[f] for f in frames if f in cache.data}


def wrist_dist(b: BallObs, people: list[Person]) -> tuple[float | None, Person | None]:
    best, who = None, None
    for p in people:
        for x, y, c in p.wrists:
            if c > 0.3:
                d = float(np.hypot(x - b.xy[0], y - b.xy[1]) / max(1.0, 2 * b.r))
                if best is None or d < best:
                    best, who = d, p
    return best, who


def _person_dist(b: BallObs, p: Person) -> float | None:
    ds = [np.hypot(x - b.xy[0], y - b.xy[1]) / max(1.0, 2 * b.r) for x, y, c in p.wrists if c > 0.3]
    return float(min(ds)) if ds else None


def hand_offs(ball: dict[int, BallObs], people: dict[int, list[Person]], fps: float) -> list[HandOff]:
    """Hold runs: one person (followed by box IoU) keeps a wrist within HELD_D of the ball for >= MIN_HOLD_S.
    A run ends in a release when the ball then moves away from that person, or leaves the view (a high arc
    out of the top of the frame)."""
    from .release import _iou

    frames = [f for f in sorted(people) if f in ball]
    runs: list[list[tuple[int, float, Person]]] = []
    for f in frames:
        holders = [(d, p) for p in people[f] if (d := _person_dist(ball[f], p)) is not None and d <= HELD_D]
        cont = None
        if runs:
            lf, _, lp = runs[-1][-1]
            same = [(d, p) for d, p in holders if _iou(p.box, lp.box) > 0.4]
            if same and f - lf <= 0.2 * fps:
                cont = min(same, key=lambda x: x[0])
        if cont is not None:
            runs[-1].append((f, cont[0], cont[1]))
        elif holders:
            d, p = min(holders, key=lambda x: x[0])
            runs.append([(f, d, p)])
    out = []
    for r in runs:
        f_last, _, p = r[-1]
        if (f_last - r[0][0]) < MIN_HOLD_S * fps:
            continue
        nxt = [f for f in frames if f > f_last]
        # ball not seen again soon: it left the view (e.g. a high arc) -> release right after the last hold
        if not nxt or nxt[0] - f_last > 0.35 * fps:
            gap_end = f_last + max(1.0, fps / 30)
            out.append(HandOff(gap_end - 0.5 * max(1.0, fps / 30), f_last, p, 0.6, r))
            continue
        g = nxt[0]
        ds = []
        for h in [x for x in nxt if x <= f_last + 0.25 * fps and ball[x].score >= 0.3]:
            q = max(people.get(h, []), key=lambda q: _iou(q.box, p.box), default=None)
            dq = _person_dist(ball[h], q) if q is not None and _iou(q.box, p.box) > 0.3 else None
            if dq is not None:
                ds.append(dq)
        # gone: beyond HELD_D within 0.25 s and never back in the hand (< 0.7 HELD_D)
        if ds and (max(ds) <= HELD_D or min(ds) < 0.7 * HELD_D):
            continue
        out.append(HandOff((f_last + g) / 2, f_last, p, round(max(0.3, 1.0 - 2 * (g - f_last) / fps), 3), r))
    return out


def reaches_rim(ball: dict[int, BallObs], rim_by_f: dict[int, Rim], f0: float, fps: float, within_s: float = 2.0) -> int | None:
    """First frame after f0 when the ball is at/above the rim within ~1.2 rim widths sideways."""
    for f in sorted(ball):
        if f <= f0 or f > f0 + within_s * fps:
            continue
        r = rim_at(rim_by_f, f, int(fps * 0.3))
        if r is None or r.width < 8:
            continue
        x = (ball[f].xy[0] - r.center[0]) / r.width
        y = (ball[f].xy[1] - r.center[1]) / r.width
        if abs(x) < 1.2 and -3.0 < y < 0.3:
            return f
    return None


def shooter_of(h: HandOff, tracks: Tracks, dets: dict, fps: float) -> tuple[int | None, dict]:
    """Vote over the hold: whose wrist kept the ball, plus DEIM's pose class on that player's box
    (jump-shot / layup for the shooter, shot-block for a contesting defender)."""
    from .release import _iou
    votes: dict[int, float] = {}
    whole: dict[int, float] = {}
    for f, d, person in h.held:
        tid = track_of(person, tracks, f)
        if tid is None:
            continue
        recency = float(np.exp(-(h.f_held - f) / (0.1 * fps)))  # the last holder before release counts most
        votes[tid] = votes.get(tid, 0.0) + recency / (0.5 + d)
        whole[tid] = whole.get(tid, 0.0) + 1.0 / (0.5 + d)
    for f in range(int(h.f_held - 0.4 * fps), int(h.f + 0.2 * fps) + 1):
        for d in dets.get(f, []):
            if d.cls not in ("player-jump-shot", "player-layup-dunk", "player-shot-block") or d.score < 0.4:
                continue
            for p in tracks.players(f):
                if p.track_id in votes and _iou(p.box, d.box) > 0.5:
                    votes[p.track_id] += (-1.0 if d.cls == "player-shot-block" else 1.0) * d.score
    if not votes:
        # pose person matched no M1 box: fall back to the M1 player box the ball touches at the last hold frames
        from .tracks import contact
        for f, d, person in reversed(h.held):
            b = ball_at.get(f) if (ball_at := getattr(h, "ball", None)) else None
            wr = max(person.wrists, key=lambda w: w[2])
            pts = [(wr[0], wr[1])] + ([b.xy] if b else [])
            cands = [(min(contact(xy, 0.0, p.box) for xy in pts), p.track_id) for p in tracks.players(f)]
            cands = [c for c in cands if c[0] <= 0.05]
            if cands:
                tid = min(cands)[1]
                return tid, {tid: 0.0, "fallback": "m1_contact"}
        return None, votes
    best = max(votes, key=votes.get)
    held_most = max(whole, key=whole.get)
    if tracks.team_of(best) and tracks.team_of(held_most) and tracks.team_of(best) != tracks.team_of(held_most):
        best = held_most  # a defender's hand at the very end is a contest, not the shooter
    return best, {k: round(v, 2) for k, v in votes.items()}


def track_of(person: Person, tracks: Tracks, f: int, ball_xy: tuple[float, float] | None = None) -> int | None:
    """M1 track for a pose person: best box IoU (>= 0.3), else the single player box containing its best wrist."""
    from .release import _iou
    players = tracks.players(f)
    best = max(((_iou(p.box, person.box), p.track_id) for p in players), default=(0.0, None), key=lambda x: x[0])
    if best[0] >= 0.3:
        return best[1]
    wr = max(person.wrists, key=lambda w: w[2])
    inside = [p for p in players if p.box[0] <= wr[0] <= p.box[2] and p.box[1] <= wr[1] <= p.box[3]]
    if len(inside) == 1:
        return inside[0].track_id
    if best[0] >= 0.15:
        return best[1]
    # no overlapping M1 box: nearest player box centre within 0.6 body heights
    cx, cy = (person.box[0] + person.box[2]) / 2, (person.box[1] + person.box[3]) / 2
    ph = max(1.0, person.box[3] - person.box[1])
    near = min(((np.hypot((p.box[0] + p.box[2]) / 2 - cx, (p.box[1] + p.box[3]) / 2 - cy), p.track_id) for p in players),
               default=(np.inf, None), key=lambda x: x[0])
    return near[1] if near[0] < 0.6 * ph else None
