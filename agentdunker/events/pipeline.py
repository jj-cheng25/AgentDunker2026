"""End to end: video + M1 tracks (+ M2 court, M3 identity) -> events.json (PLAN contract + extras).

Always returns a document. When something is uncertain it degrades instead of failing: no rim visit -> `shot`
fields null; unclear made -> VLM check, then null; no court -> no screens/drives/points.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import made as made_mod
from .actions import action_windows, court_paths, drives, jump_windows, screens
from .balltrack import drop_static_near_rim, link_ball, rim_at, rims
from .court import Court, is_three, nearest_rim
from .official import match_shot
from .possession import Segment, ambiguous_frames, ball_states, holders
from .release import Release, find_release
from .release_pose import PoseCache, hand_offs, people_near_ball, reaches_rim, shooter_of
from .rimvisit import RimVisit, find_visits, rim_relative
from .scan import DetCache, detect_frames
from .tracks import Tracks
from .video import probe, sha256

logger = logging.getLogger(__name__)
SCHEMA = "adk-events/1"


def _uniform(cache: DetCache, frames: list[int]) -> dict:
    return {f: cache.data[f] for f in frames if f in cache.data}


def run(video: str, tracks_path: str | None, court_path: str | None = None, identity_path: str | None = None,
        cache_dir: str | None = None, use_vlm: bool = True, use_pose: bool = True, threads: int = 8) -> dict:
    info = probe(video)
    fps, n = info.fps, info.n_frames
    cache = DetCache(Path(cache_dir) / f"{Path(video).stem}.dets.json" if cache_dir else None)
    tracks = Tracks.load(tracks_path)
    court = Court.load(court_path)
    identity = json.loads(Path(identity_path).read_text()) if identity_path and Path(identity_path).exists() else None
    from .detector import BallDetector
    det = BallDetector(threads=threads)

    # 1. 10 Hz scan of the whole clip
    s10 = max(1, round(fps / 10))
    sparse_frames = list(range(0, n, s10))
    detect_frames(video, sparse_frames, cache, det)
    rim_by_f = rims(cache.data, fps=fps)
    sparse = _uniform(cache, sparse_frames)
    ball10 = link_ball(drop_static_near_rim(sparse, rim_by_f, fps), fps)

    # 2. shot attempts = the ball coming to the rim from above; dense scan around each (latest 4)
    visits = [v for v in find_visits(rim_relative(ball10, rim_by_f, int(fps * 0.3)), fps) if v.above and len(v.points) >= 2]
    visits = visits[-4:]
    s30 = max(1, round(fps / 30))
    attempts = []
    pose_cache = PoseCache(Path(cache_dir) / f"{Path(video).stem}.pose.json" if cache_dir else None)
    pose = None
    for v in visits:
        lo, hi = max(0, v.f_first - int(2.5 * fps)), min(n, v.f_last + int(0.8 * fps))
        fr = list(range(lo - lo % s30, hi, s30))
        detect_frames(video, fr, cache, det)
        dense = _uniform(cache, fr)
        ball = link_ball(drop_static_near_rim(dense, rim_by_f, fps), fps)
        judged = made_mod.judge(ball, rim_by_f, v.f_first - int(0.3 * fps), v.f_last + int(0.8 * fps), fps)
        f_end = int(judged.get("f_cross") or v.f_last)
        rels: list[Release] = []
        if use_pose:
            if pose is None:
                from .pose import PoseNear
                pose = PoseNear(threads=threads)
            win = [f for f in fr if v.f_first - 1.5 * fps <= f <= f_end]
            people = people_near_ball(video, ball, win, pose_cache, pose)
            hos = hand_offs(ball, people, fps)
            f_cross = judged.get("f_cross")
            for i, h in enumerate(hos):
                arrive = reaches_rim(ball, rim_by_f, h.f, fps)
                if arrive is None or any(h.f < h2.f < arrive for h2 in hos[i + 1:]):
                    continue  # never got to the rim, or was passed on first
                if judged.get("made") and f_cross is not None and h.f >= f_cross - 0.05 * fps:
                    continue  # the ball had already gone through: hands under the net, not a new attempt
                if _falls_after(ball, rim_by_f, h, fps):
                    continue  # a short touch on a falling ball (hands near the rim), not a release
                tid, votes = shooter_of(h, tracks, cache.data, fps)
                rels.append(Release(h.f, tid, int(h.f), h.conf, {"method": "wrist", "f_held": h.f_held,
                                                                 "arrive_f": arrive, "votes": votes}))
        if not rels:
            rel = find_release(ball, rim_by_f, tracks, dense, v.f_first, fps)
            if rel is None:
                continue
            rel.evidence["method"] = "box"
            rel.conf = round(rel.conf * 0.8, 3)
            rels = [rel]
        for k, rel in enumerate(rels):
            last = k == len(rels) - 1
            res = judged if last else {"made": False, "conf": 0.6, "rule": "ball released again before scoring"}
            attempts.append({"visit": v, "release": rel, "ball": ball, "made": res, "dets": cache.data})
    shot_doc, shot_att = _main_shot(attempts, info, video, rim_by_f, tracks, court, identity, use_vlm)

    # 3. ball state timeline + per-track actions (10 Hz)
    shots = [{"f_release": a["release"].f, "f_end": int(a["made"].get("f_cross") or a["visit"].f_last) + int(0.3 * fps),
              "shooter": a["release"].shooter_track, "conf": a["release"].conf} for a in attempts]
    people = None
    if use_pose and tracks.by_f:
        amb = ambiguous_frames(ball10, sparse, tracks, sparse_frames)
        if amb:
            if pose is None:
                from .pose import PoseNear
                pose = PoseNear(threads=threads)
            people = people_near_ball(video, ball10, amb, pose_cache, pose)
    obs = holders(ball10, sparse, tracks, sparse_frames, people)
    segs = ball_states(obs, tracks, fps, n, shots)
    paths = court_paths(tracks, court, sparse_frames) if court else {}
    jumps = jump_windows(cache.data, tracks, fps)
    scr = screens(segs, paths, tracks, fps, sparse_frames) if paths else []
    drv = drives(segs, paths, fps, sparse_frames) if paths else []
    acts = action_windows(segs, jumps, scr, {"shooter_track": shot_doc.get("shooter_track"),
                                             "release_f": info.f(shot_doc["release_t"]) if shot_doc.get("release_t") is not None else None,
                                             "conf": shot_doc.get("conf", 0.5)}, fps)
    events = _events(segs, scr, drv, attempts, shot_att, tracks, info)

    offense = _offense(segs, tracks, shot_doc)
    return {
        "schema": SCHEMA,
        "video": {"path": str(video), "sha256": sha256(video), "fps": fps, "width": info.width, "height": info.height,
                  "n_frames": n, "start_pts": info.start_pts},
        "events": events,
        "shot": shot_doc,
        "ball_state": [_seg_doc(s, info) for s in segs],
        "actions": [{**{k: v for k, v in a.items() if k not in ("f0", "f1")}, "t0": info.t(a["f0"]), "t1": info.t(a["f1"]),
                     "f0": int(a["f0"]), "f1": int(a["f1"])} for a in acts],
        "offense_team": offense,
        "attempts": [{"release_t": info.t(a["release"].f), "shooter_track": a["release"].shooter_track,
                      "made": a["made"].get("made"), "conf": a["release"].conf,
                      "rim_t": info.t(a["visit"].f_first)} for a in attempts],
        "meta": {"detector": "ortizeg/basketball-deim-m-640 (Apache-2.0)", "pose": "yolo11x-pose (AGPL-3.0)" if use_pose else None,
                 "vlm_model": made_mod.vlm.MODEL if use_vlm else None,
                 "inputs": {"tracks": tracks_path, "court": court_path, "identity": identity_path}},
    }


def _main_shot(attempts, info, video, rim_by_f, tracks, court, identity, use_vlm) -> tuple[dict, dict | None]:
    empty = {"release_t": None, "shooter_track": None, "made": None, "xfg": None, "ball_path_img": [], "conf": 0.0,
             "points": None, "shot_court_xy": None, "shot_distance_ft": None, "shot_type": None, "shot_type_conf": 0.0,
             "made_t": None, "bank_t": None, "rim_t": None, "rim_img": [], "release_f": None, "made_conf": 0.0, "made_source": None, "official": None}
    if not attempts:
        return empty, None
    # the clip's highlighted shot: the last attempt (a putback after a miss ends the possession)
    a = attempts[-1]
    rel, v, judged = a["release"], a["visit"], a["made"]
    fps = info.fps
    made, made_conf, source = judged.get("made"), judged.get("conf", 0.0), "ball_path"
    if made is None and use_vlm:
        f_from = v.f_first - int(0.2 * fps)
        ans = made_mod.vlm_check(video, rim_by_f, f_from, min(info.n_frames - 1, v.f_last + int(1.0 * fps)), fps)
        if ans and ans.get("made") is not None:
            made, made_conf, source = ans["made"], 0.7, f"vlm:{ans.get('_model')}"
            judged["vlm"] = {k: ans.get(k) for k in ("result", "reason", "evidence_frames", "_cache")}
    # bank: only on this attempt's own flight (release -> crossing), not on an earlier miss in the same visit
    if judged.get("f_cross"):
        judged = {**judged, "f_bank": made_mod.bank_frame(
            made_mod.rel_points(a["ball"], rim_by_f, rel.f, judged["f_cross"] + 1, fps), judged["f_cross"], fps)}
    # ball path: release -> rim and through it (observed detections only)
    f_end = int(judged.get("f_cross") or v.f_last) + int(0.5 * fps)
    path = [[info.t(f), round(b.xy[0], 1), round(b.xy[1], 1)] for f, b in sorted(a["ball"].items())
            if rel.f - 1 <= f <= f_end]
    points, shot_xy, shot_dist = None, None, None
    if court and rel.shooter_track is not None:
        p = tracks.box(rel.shooter_track, int(round(rel.f - 0.3 * fps))) or tracks.box(rel.shooter_track, int(rel.f))
        xy = court.to_court(int(rel.f), p.foot) if (p and p.foot) else None
        points = (3 if is_three(xy) else 2) if xy else None
        if xy:
            rx, ry = nearest_rim(xy)
            shot_xy = [round(xy[0], 2), round(xy[1], 2)]
            shot_dist = round(float(((xy[0] - rx) ** 2 + (xy[1] - ry) ** 2) ** 0.5), 1)
    doc = {**empty, "release_t": info.t(rel.f), "release_f": round(rel.f, 1), "shooter_track": rel.shooter_track,
           "made": made, "made_conf": made_conf, "made_source": source, "conf": rel.conf, "points": points,
           "shot_court_xy": shot_xy, "shot_distance_ft": shot_dist,
           "ball_path_img": path, "rim_t": info.t(v.f_first),
           "made_t": info.t(judged["f_cross"]) if made and judged.get("f_cross") else None,
           "bank_t": info.t(judged["f_bank"]) if judged.get("f_bank") else None,
           "rim_img": _rim_path(rim_by_f, rel.f - 0.5 * fps, f_end + 0.5 * fps, info),
           "evidence": {"release": rel.evidence, "made": {k: judged.get(k) for k in ("rule", "crossings", "vlm")}}}
    doc["shot_type"], doc["shot_type_conf"] = _shot_type(a["ball"], rel.f, fps, a.get("dets", {}), tracks, rel.shooter_track)
    off = match_shot(identity, rel.shooter_track, made)
    if off:
        doc["official"] = off
        doc["xfg"] = off.get("xfg")
        if off.get("conf", 0) >= 0.8 and off["match"].get("name"):
            if made is None or made_conf < 0.8:
                doc["made"], doc["made_source"], doc["made_conf"] = off["made"], "official_pbp", off["conf"]
            doc["points"] = off.get("points") or doc["points"]
    return doc, a


def _shot_type(ball, f_rel: float, fps: float, dets: dict, tracks: Tracks, shooter: int | None) -> tuple[str | None, float]:
    """jump_shot vs layup_dunk from DEIM's pose class on the shooter's box (or, without tracks, on the box that
    touches the ball) in the 0.5 s before release."""
    from .release import _iou
    from .tracks import contact
    score = {"jump_shot": 0.0, "layup_dunk": 0.0}
    for f in range(int(f_rel - 0.5 * fps), int(f_rel + 0.1 * fps) + 1):
        sb = tracks.box(shooter, f) if shooter is not None else None
        b = ball.get(f)
        for d in dets.get(f, []):
            if d.cls not in ("player-jump-shot", "player-layup-dunk") or d.score < 0.1:
                continue
            if sb is not None:
                if _iou(sb.box, d.box) < 0.4:
                    continue
            elif b is None or contact(b.xy, b.r, d.box) > 0.1:
                continue
            score["jump_shot" if d.cls == "player-jump-shot" else "layup_dunk"] += d.score
    tot = sum(score.values())
    if tot < 0.5:
        return None, 0.0
    k = max(score, key=score.get)
    return k, round(score[k] / tot, 3)


def _falls_after(ball, rim_by_f, h, fps: float) -> bool:
    """True when the hold was short (< 0.4 s) and the ball goes down right after: a real release sends the ball
    up (a dunk holds the ball much longer before pushing it down)."""
    if (h.f_held - h.held[0][0]) >= 0.4 * fps:
        return False
    after = [f for f in sorted(ball) if h.f < f <= h.f + 0.15 * fps]
    if len(after) < 2:
        return False
    r = rim_at(rim_by_f, after[0], int(0.3 * fps))
    if r is None:
        return False
    return (ball[after[-1]].xy[1] - ball[after[0]].xy[1]) / r.width > 0.15


def _rim_path(rim_by_f, f0: float, f1: float, info) -> list[list[float]]:
    """Observed rim boxes [[t, cx, cy, width]]: the rim is 10 ft up, so court.json cannot place it."""
    return [[info.t(f), round(r.center[0], 1), round(r.center[1], 1), round(r.width, 1)]
            for f, r in sorted(rim_by_f.items()) if f0 <= f <= f1]


def _seg_doc(s: Segment, info) -> dict:
    d = {"type": s.type, "t0": info.t(s.f0), "t1": info.t(s.f1), "f0": int(s.f0), "f1": int(s.f1), "conf": s.conf}
    if s.type in ("hold", "dribble"):
        d["holder"] = s.holder
    elif s.type in ("pass", "shot"):
        d["from"] = s.frm
        if s.type == "pass":
            d["to"] = s.to
    return d


def _offense(segs: list[Segment], tracks: Tracks, shot: dict) -> str | None:
    if shot.get("shooter_track") is not None:
        return tracks.team_of(shot["shooter_track"])
    tally: dict[str, float] = {}
    for s in segs:
        if s.type in ("hold", "dribble") and tracks.team_of(s.holder):
            tally[tracks.team_of(s.holder)] = tally.get(tracks.team_of(s.holder), 0) + (s.f1 - s.f0)
    return max(tally, key=tally.get) if tally else None


def _events(segs, scr, drv, attempts, shot_att, tracks, info) -> list[dict]:
    ev = []
    for s in segs:
        if s.type == "pass":
            ev.append({"t": info.t(s.f0), "type": "pass", "actors": [s.frm, s.to], "conf": s.conf,
                       "t_end": info.t(s.f1)})
    for s in scr:
        ev.append({"t": info.t(s["f0"]), "type": "screen", "actors": [s["track"], s["for"]], "conf": s["conf"],
                   "t_end": info.t(s["f1"])})
    for d in drv:
        ev.append({"t": info.t(d["f0"]), "type": "drive", "actors": [d["track"]], "conf": d["conf"], "t_end": info.t(d["f1"])})
    for a in attempts:
        rel = a["release"]
        ev.append({"t": info.t(rel.f), "type": "shot", "actors": [rel.shooter_track] if rel.shooter_track is not None else [],
                   "conf": rel.conf, "made": a["made"].get("made"), "main": a is shot_att})
        if a["made"].get("made") is False:
            # rebound: first holder after the miss
            f_after = a["made"].get("f_cross") or a["visit"].f_last
            nxt = next((s for s in segs if s.type in ("hold", "dribble") and s.f0 >= f_after), None)
            if nxt:
                off = tracks.team_of(rel.shooter_track) == tracks.team_of(nxt.holder)
                ev.append({"t": info.t(nxt.f0), "type": "rebound", "actors": [nxt.holder], "conf": round(0.8 * nxt.conf, 3),
                           "offensive": off})
        elif a["made"].get("made") and a["made"].get("f_cross"):
            ev.append({"t": info.t(a["made"]["f_cross"]), "type": "other", "subtype": "made_basket",
                       "actors": [rel.shooter_track] if rel.shooter_track is not None else [], "conf": a["made"]["conf"]})
    return sorted(ev, key=lambda e: e["t"])
