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
from .official import clock_secs, clock_to_t, defence_never_had_ball, match_shot, possession_changes
from .possession import Segment, ambiguous_frames, ball_states, holders
from .release import Release, find_release
from .release_pose import PoseCache, hand_offs, on_floor, people_near_ball, reaches_rim, shooter_of
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
            hos = hand_offs(ball, on_floor(people, tracks, court), fps)
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

    # 2b. the official record says who had the ball: with no possession change to the defence between the clip
    # start and the shot, defenders' "shots" and holds before the release are holder noise (matched games only)
    gate = _possession_gate(identity, shot_doc, tracks)
    if gate:
        de, gate_f = gate
        kept = [a for a in attempts if a is shot_att or a["release"].f >= gate_f
                or tracks.team_of(a["release"].shooter_track) != de]
        gate = {"defence": de, "until_t": shot_doc["release_t"], "dropped_attempts": len(attempts) - len(kept)}
        attempts = kept

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
    if gate:
        segs, gate["stripped_segments"] = _strip_defence(segs, tracks, gate["defence"], shot_doc["release_f"])
        logger.info("possession check (official PBP): %s", gate)
    paths = court_paths(tracks, court, sparse_frames) if court else {}
    jumps = jump_windows(cache.data, tracks, fps)
    scr = screens(segs, paths, tracks, fps, sparse_frames) if paths else []
    drv = drives(segs, paths, fps, sparse_frames) if paths else []
    acts = action_windows(segs, jumps, scr, {"shooter_track": shot_doc.get("shooter_track"),
                                             "release_f": info.f(shot_doc["release_t"]) if shot_doc.get("release_t") is not None else None,
                                             "conf": shot_doc.get("conf", 0.5)}, fps)
    events = _events(segs, scr, drv, attempts, shot_att, tracks, info)
    events = sorted(events + _official_changes(identity, shot_doc, segs, tracks, info), key=lambda e: e["t"])

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
        "possession_check": gate or None,
        "meta": {"detector": "ortizeg/basketball-deim-m-640 (Apache-2.0)", "pose": "yolo11x-pose (AGPL-3.0)" if use_pose else None,
                 "vlm_model": made_mod.vlm.MODEL if use_vlm else None,
                 "inputs": {"tracks": tracks_path, "court": court_path, "identity": identity_path}},
    }


def _main_shot(attempts, info, video, rim_by_f, tracks, court, identity, use_vlm) -> tuple[dict, dict | None]:
    empty = {"release_t": None, "shooter_track": None, "made": None, "xfg": None, "ball_path_img": [], "conf": 0.0,
             "points": None, "shot_court_xy": None, "shot_distance_ft": None, "shot_type": None, "shot_type_conf": 0.0,
             "made_t": None, "bank_t": None, "rim_t": None, "rim_img": [], "release_f": None, "made_conf": 0.0, "made_source": None, "official": None}
    if not attempts:
        return _official_only(identity, tracks, info) or empty, None
    a = pick_attempt(attempts)
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
    off = match_shot(identity, rel.shooter_track, made)
    shooter, override, kept = arbitrate(identity, off, rel.shooter_track, a["ball"], tracks, rel.f, fps)
    if override and override.get("to") is not None:
        # the record's match (name/team agreement, team candidates) now refers to the shooter it handed the shot
        # to, so M7 can name an unnamed one (2026-11-27 PL-NBA k03: Sengun's track had no name)
        off = match_shot(identity, shooter, made) or off
    rel.shooter_track = shooter      # the attempt list, shot events and actions follow
    points, shot_xy, shot_dist = None, None, None
    if court and shooter is not None:
        p = tracks.box(shooter, int(round(rel.f - 0.3 * fps))) or tracks.box(shooter, int(rel.f))
        xy = court.to_court(int(rel.f), p.foot) if (p and p.foot) else None
        points = (3 if is_three(xy) else 2) if xy else None
        if xy:
            rx, ry = nearest_rim(xy)
            shot_xy = [round(xy[0], 2), round(xy[1], 2)]
            shot_dist = round(float(((xy[0] - rx) ** 2 + (xy[1] - ry) ** 2) ** 0.5), 1)
    doc = {**empty, "release_t": info.t(rel.f), "release_f": round(rel.f, 1), "shooter_track": shooter,
           "made": made, "made_conf": made_conf, "made_source": source, "conf": rel.conf, "points": points,
           "shot_court_xy": shot_xy, "shot_distance_ft": shot_dist,
           "ball_path_img": path, "rim_t": info.t(v.f_first),
           "made_t": info.t(judged["f_cross"]) if made and judged.get("f_cross") else None,
           "bank_t": info.t(judged["f_bank"]) if judged.get("f_bank") else None,
           "rim_img": _rim_path(rim_by_f, rel.f - 0.5 * fps, f_end + 0.5 * fps, info),
           "evidence": {"release": rel.evidence, "made": {k: judged.get(k) for k in ("rule", "crossings", "vlm")}}}
    if override:
        doc["evidence"]["shooter_override"] = override
    if kept:
        doc["evidence"]["shooter_kept"] = kept
    doc["shot_type"], doc["shot_type_conf"] = _shot_type(a["ball"], rel.f, fps, a.get("dets", {}), tracks, shooter)
    if off:
        doc["official"] = off
        doc["xfg"] = off.get("xfg")
        named = bool((((identity or {}).get("players") or {}).get(str(shooter)) or {}).get("name"))
        pin = official_pins(off, override, named)
        if pin:
            # firm: the play-by-play outcome stands even against a confident ball path (h12 held-out: a
            # rim-bouncing three made, read as "came down outside the rim"; 2026-11-27 PL-NBA m05/m08/m12: misses
            # off the front rim read as "fell through" at 0.9, which the broadcast angle cannot tell apart)
            firm = pin == "team" or (off.get("conf", 0) >= 0.9 and off["match"].get("n_candidates") == 1)
            if made is None or made_conf < 0.8 or (firm and made is not off["made"]):
                _official_outcome(doc, a, judged, made, made_conf, off)
            doc["points"] = off.get("points") or doc["points"]
    return doc, a


AT_RIM_MAX_X = 3.0           # rim widths: a shot's ball comes down within this of the rim centre
OFFICIAL_UNNAMED_CONF = 0.6   # a matched game (every "matched" game in 66 test clips was the right one)


def pick_attempt(attempts: list[dict]) -> dict:
    """The clip's highlighted shot: the last attempt (a putback after a miss ends the possession), except
    - a missed touch after the ball already reached the rim in the same visit: a tip at the rebound (c04: Dunn's
      corner three at 7.6 s lost to a hand near the rim at 8.45 s with no shooter);
    - a box-fallback "release" (no hand seen letting go) whose ball never comes down within AT_RIM_MAX_X rim
      widths of the rim: a tap or an outlet after the rebound (2026-11-27 PL-NBA m16: Dort's three, then a tap
      6.5 rim widths off the rim; m17: Irving's three, then an unclear ball path 2 s later)."""
    solid = [x for x in attempts if x["release"].evidence.get("method") != "box"
             or any(abs(c[1]) < AT_RIM_MAX_X for c in x["made"].get("crossings") or [])] or attempts
    proper = [x for x in solid if x["release"].f <= x["visit"].f_first or x["made"].get("made")]
    return (proper or solid)[-1]


def official_pins(off: dict, override: dict | None, named: bool) -> str | None:
    """Whether the play-by-play field goal is this clip's shot, firmly enough for its outcome and points:
    "name" when the shooter's name agrees (or the record moved the shot), in a firm match; "team" when the
    shooter track has no name to check (older seasons have no boxscore; or the jersey went unread) but the
    clip window holds exactly one field goal of its team, in a matched game."""
    m = off.get("match") or {}
    if off.get("conf", 0) >= 0.8 and (m.get("name") or override):
        return "name"
    if not named and m.get("team") and m.get("n_team") == 1 and off.get("conf", 0) >= OFFICIAL_UNNAMED_CONF:
        return "team"
    return None


def _official_outcome(doc: dict, a: dict, judged: dict, made: bool | None, made_conf: float, off: dict) -> None:
    """The play-by-play's made/missed replaces the video's, in the shot doc and in the attempt (events follow)."""
    if made is not None and made is not off["made"]:
        doc["evidence"]["made_disagreement"] = {"ball_path": made, "conf": made_conf,
                                                "rule": judged.get("rule"), "official": off["made"]}
        a["made"] = {**judged, "made": off["made"], "rule": "official_pbp"}   # in place: `a is shot_att`
        if not off["made"]:
            doc["made_t"] = None
    doc["made"], doc["made_source"], doc["made_conf"] = off["made"], "official_pbp", off["conf"]


def _official_only(identity: dict | None, tracks: Tracks, info) -> dict | None:
    """The clip's shot from the play-by-play alone, when the video found no attempt (h21 held-out: a cutting
    dunk, too short a flight for the ball path). Only for a firm match whose window holds exactly one field goal
    and a single track named as its shooter; the release time maps the PBP clock onto the score-bug reads.
    In a game with no names for the scorer's team (an older season without a boxscore) the shot comes with no
    track, from any matched game (2026-11-27 PL-NBA k08: Anunoby's cutting dunk); M7 names the scorer from the
    record."""
    off = match_shot(identity, None, None)
    if not off or off["match"].get("n_candidates") != 1 or not off.get("person_id"):
        return None
    players = (identity or {}).get("players") or {}
    hits = [int(t) for t, p in players.items()
            if str(p.get("person_id")) == str(off["person_id"]) and p.get("team_abbr") == off.get("team")]
    rosterless = not any(p.get("name") for p in players.values() if p.get("team_abbr") == off.get("team"))
    if not ((len(hits) == 1 and off.get("conf", 0) >= 0.9)
            or (not hits and rosterless and off.get("conf", 0) >= OFFICIAL_UNNAMED_CONF)):
        return None
    release_t = clock_to_t(identity, clock_secs(off.get("clock")), info.t(info.n_frames - 1))
    if release_t is None:
        return None
    f = info.f(release_t)
    shooter = hits[0] if hits else None
    logger.info("shot from PBP only: %s at %.2f s (track %s)", off.get("description"), release_t, shooter)
    return {"release_t": release_t, "release_f": round(float(f), 1), "shooter_track": shooter, "made": off["made"],
            "made_conf": off["conf"], "made_source": "official_pbp", "conf": 0.5, "points": off.get("points"),
            "shot_court_xy": None, "shot_distance_ft": off.get("shot_distance_ft"), "ball_path_img": [], "rim_t": None,
            "made_t": None, "bank_t": None, "rim_img": [], "shot_type": None, "shot_type_conf": 0.0,
            "xfg": off.get("xfg"), "official": off,
            "evidence": {"release": {"method": "official_pbp_only", "clock": off.get("clock")}}}


def _possession_gate(identity: dict | None, shot: dict, tracks: Tracks) -> tuple[str, float] | None:
    """(defence team label, release frame) when the official record leaves the ball with the offence from the
    clip start to the shot; None without a firm match or when the defence had the ball in between."""
    off, shooter = shot.get("official"), shot.get("shooter_track")
    if not off or shot.get("release_f") is None:
        return None
    ev = shot.get("evidence") or {}
    pinned = (off["match"].get("name") or ev.get("shooter_override") or ev.get("shooter_kept")
              or (ev.get("release") or {}).get("method") == "official_pbp_only")
    if not pinned or (shooter is None and not ev.get("shooter_override")):
        # h21 held-out: the dunk came from the PBP alone, so the check was skipped and the Lakers defenders next
        # to the Rockets' ball handler were narrated as passing to each other
        return None
    teams = (identity or {}).get("teams") or {}
    tri = lambda g: (teams.get(g) or {}).get("abbr")  # noqa: E731
    # no shooter track (the PBP scorer's own track was lost, h07): the offence is the scorer's team label
    o = tracks.team_of(shooter) if shooter is not None else next((g for g in teams if tri(g) == off.get("team")), None)
    d = {"A": "B", "B": "A"}.get(o or "")
    if not d or tri(o) != off.get("team"):
        return None
    return (d, float(shot["release_f"])) if defence_never_had_ball(identity, off, tri(o), tri(d)) else None


def _official_changes(identity: dict | None, shot: dict, segs: list[Segment], tracks: Tracks, info) -> list[dict]:
    """Steals and turnovers of the official record as events (type "other", subtype steal|turnover).

    The PBP clock is mapped onto the clip through the score-bug reads, then moved to the nearest ball-state change
    of possession to the gaining team (within 3 s), where the video shows it. Without that, the commentary went
    from one team's drive straight to the other team's pass (c06: B. James lost the ball, B. Podziemski stole it,
    then Curry's transition three), as if the passer had handed it over."""
    off = shot.get("official")
    changes = possession_changes(identity, off)
    if not changes:
        return []
    players = (identity or {}).get("players") or {}
    teams = {g: (v or {}).get("abbr") for g, v in ((identity or {}).get("teams") or {}).items()}

    def track_of(pid, tri):
        hits = [int(t) for t, p in players.items()
                if pid and str(p.get("person_id")) == str(pid) and p.get("team_abbr") == tri]
        return hits[0] if len(hits) == 1 else None

    held = [s for s in segs if s.type in ("hold", "dribble") and s.holder is not None]
    out = []
    for pc in changes:
        t = clock_to_t(identity, pc["clock_s"], info.t(info.n_frames - 1))
        if t is None:
            continue
        gain = pc.get("stealer_team") or next((x for x in teams.values() if x and x != pc.get("loser_team")), None)
        best = None
        for prev, s in zip(held, held[1:]):
            a, b = teams.get(tracks.team_of(prev.holder)), teams.get(tracks.team_of(s.holder))
            ts = info.t(s.f0)
            if b == gain and a not in (None, gain) and abs(ts - t) <= 3.0 and (best is None or abs(ts - t) < abs(best - t)):
                best = ts
        st, lo = track_of(pc.get("stealer_id"), pc.get("stealer_team")), track_of(pc.get("loser_id"), pc.get("loser_team"))
        out.append({"t": round(best if best is not None else t, 3), "type": "other", "subtype": pc["kind"],
                    "actors": [x for x in (st if pc["kind"] == "steal" else lo,) if x is not None],
                    "stealer_track": st, "loser_track": lo, "conf": 0.85 if best is not None else 0.7,
                    "official": {k: pc.get(k) for k in ("clock_s", "stealer", "stealer_team", "loser", "loser_team",
                                                        "turnover")}})
        logger.info("official %s at %.2f s (%s): %s", pc["kind"], out[-1]["t"],
                    "ball state" if best is not None else "clock only", out[-1]["official"])
    return out


def _strip_defence(segs: list[Segment], tracks: Tracks, defence: str, f_release: float) -> tuple[list[Segment], int]:
    """Before the release, holds/dribbles of a defender and passes touching one become loose ball."""
    out, n = [], 0
    for s in segs:
        people = [s.holder] if s.type in ("hold", "dribble") else [s.frm, s.to] if s.type == "pass" else []
        if s.f0 < f_release and any(p is not None and tracks.team_of(p) == defence for p in people):
            out.append(Segment("loose", s.f0, s.f1, conf=0.3, extra={"was": s.type, "why": "official: no defence possession"}))
            n += 1
        else:
            out.append(s)
    return out, n


PBP_HOLD_S = 1.5     # the PBP scorer's track must have had the ball in this long before the release ...
PBP_MIN_HELD = 3     # ... in at least this many observed frames, before it takes the shot from the video's pick,
PBP_HELD_SHARE = 0.5  # ... and in at least half as many frames as the video's pick
PBP_CONTEST_GAP = 0.3  # or the video's pick, on the other team, is this close (body heights) to it at the release


def ball_contact_frames(ball: dict, tracks: Tracks, tid: int, f0: float, f1: float) -> int:
    """Frames in [f0, f1] where the track's observed box holds the ball (release.HOLD gap)."""
    from .release import HOLD
    from .tracks import contact
    n = 0
    for f, b in ball.items():
        if f0 <= f <= f1:
            p = tracks.box(tid, f)
            if p is not None and p.observed and contact(b.xy, b.r, p.box) <= HOLD:
                n += 1
    return n


def box_gap(tracks: Tracks, a: int, b: int, f: float, fps: float) -> float | None:
    """Smallest gap between two tracks' boxes (observed or bridged) within 0.2 s of frame f, in body heights."""
    best = None
    for g in range(int(f - 0.2 * fps), int(f + 0.2 * fps) + 1):
        pa, pb = tracks.box(a, g), tracks.box(b, g)
        if pa is None or pb is None:
            continue
        h = max(1.0, min(pa.box[3] - pa.box[1], pb.box[3] - pb.box[1]))
        dx = max(pa.box[0] - pb.box[2], 0.0, pb.box[0] - pa.box[2])
        dy = max(pa.box[1] - pb.box[3], 0.0, pb.box[1] - pa.box[3])
        d = (dx * dx + dy * dy) ** 0.5 / h
        best = d if best is None else min(best, d)
    return best


def _team_abbr(identity: dict | None, tracks: Tracks, tid: int | None) -> str | None:
    """A track's team as a tricode: identity's own read for the player, else its M1 team label's abbreviation."""
    p = ((identity or {}).get("players") or {}).get(str(tid)) or {}
    if p.get("team_abbr"):
        return p["team_abbr"]
    return (((identity or {}).get("teams") or {}).get(tracks.team_of(tid) or "") or {}).get("abbr")


def arbitrate(identity: dict | None, off: dict | None, shooter: int | None, ball: dict, tracks: Tracks,
              f_rel: float, fps: float) -> tuple[int | None, dict | None, dict | None]:
    """(shooter, override, kept) when the play-by-play names a different player than the video's shooter.

    The named track takes the shot only if it had the ball before the release, at least half as long as the
    video's pick (h03/h04/h22/h19 held-out, c08 bench). Otherwise a video shooter on the scorer's team is kept
    and the names are flagged as swapped (h16: McDaniels's pull-up, his name on Gobert's track); a video
    shooter on the other team is dropped, since neither track can be the scorer (h01/h07: the shooter's own
    track was lost or swapped in M1)."""
    alt = official_shooter_track(identity, off, shooter)
    if alt is None:
        return official_team_shooter(identity, off, shooter, ball, tracks, f_rel, fps)
    span = (f_rel - PBP_HOLD_S * fps, f_rel + 0.1 * fps)
    held = ball_contact_frames(ball, tracks, alt, *span)
    held_v = ball_contact_frames(ball, tracks, shooter, *span) if shooter is not None else 0
    info = {"player": off.get("player"), "held": held, "held_video_pick": held_v}
    if held >= PBP_MIN_HELD and held >= PBP_HELD_SHARE * held_v:
        logger.info("shooter: track %s -> %s (the only field goal in the clip window is %s's)", shooter, alt,
                    off.get("player"))
        return alt, {"from": shooter, "to": alt, "by": "official_pbp", **info}, None
    if shooter is not None and _team_abbr(identity, tracks, shooter) == off.get("team"):
        logger.info("shooter: keep track %s; %s's named track %s never had the ball", shooter, off.get("player"), alt)
        return shooter, None, {"track": shooter, "pbp_track": alt, "person_id": off.get("person_id"), **info}
    gap = box_gap(tracks, alt, shooter, f_rel, fps) if shooter is not None else None
    if gap is not None and gap <= PBP_CONTEST_GAP:
        # the video's pick is a defender right on the scorer at the release: he was contesting the scorer's shot,
        # whose own box M1 lost or the ball detections missed (n14 new held-out: LeBron James's corner three)
        logger.info("shooter: track %s -> %s (%s's defender at the release)", shooter, alt, off.get("player"))
        return alt, {"from": shooter, "to": alt, "by": "official_pbp_contested", "gap": round(gap, 2), **info}, None
    logger.info("shooter: none (track %s is not on %s; %s's track %s never had the ball)", shooter,
                off.get("team"), off.get("player"), alt)
    return None, {"from": shooter, "to": None, "by": "official_pbp", **info,
                  "why": "the video's shooter is on the other team and the scorer's track never had the ball"}, None


def official_team_shooter(identity: dict | None, off: dict | None, shooter: int | None, ball: dict, tracks: Tracks,
                          f_rel: float, fps: float) -> tuple[int | None, dict | None, dict | None]:
    """(shooter, override, kept) when the play-by-play pins the shot on the other team than the video's pick but
    no track carries the scorer's name (older seasons have no boxscore, so nobody is named): the scorer is the
    unnamed team-mate of his who had the ball before the release, or the one the pick was contesting at the
    release; with neither, the pick is dropped (2026-11-27 PL-NBA k03 Sengun's hook, m19 Booker's floater: the
    video took the defender at the rim)."""
    if (not off or shooter is None or off["match"].get("n_candidates") != 1
            or off.get("conf", 0) < OFFICIAL_UNNAMED_CONF or not off.get("team")):
        return shooter, None, None
    if _team_abbr(identity, tracks, shooter) in (None, off["team"]):
        return shooter, None, None
    players = (identity or {}).get("players") or {}
    if any(str(p.get("person_id")) == str(off.get("person_id")) for p in players.values()):
        return shooter, None, None       # a track carries the scorer's name: official_shooter_track's business
    mates = [t for t, m in tracks.meta.items() if m.get("role", "player") == "player" and t != shooter and _team_abbr(identity, tracks, t) == off["team"]
             and not (players.get(str(t)) or {}).get("name")]
    if not mates:
        return shooter, None, None
    span = (f_rel - PBP_HOLD_S * fps, f_rel + 0.1 * fps)
    held = {t: ball_contact_frames(ball, tracks, t, *span) for t in mates}
    held_v = ball_contact_frames(ball, tracks, shooter, *span)
    best = max(mates, key=lambda t: held[t])
    info = {"player": off.get("player"), "team": off["team"], "held": held[best], "held_video_pick": held_v}
    if held[best] >= PBP_MIN_HELD and held[best] >= PBP_HELD_SHARE * held_v:
        logger.info("shooter: track %s -> %s (%s's team took the only field goal; it had the ball)", shooter, best,
                    off["team"])
        return best, {"from": shooter, "to": best, "by": "official_pbp_team", **info}, None
    gaps = {t: box_gap(tracks, t, shooter, f_rel, fps) for t in mates}
    near = [t for t, g in gaps.items() if g is not None and g <= PBP_CONTEST_GAP]
    if len(near) == 1:
        t = near[0]
        logger.info("shooter: track %s -> %s (%s's player the pick was guarding at the release)", shooter, t, off["team"])
        return t, {"from": shooter, "to": t, "by": "official_pbp_team_contested", "gap": round(gaps[t], 2),
                   **{**info, "held": held[t]}}, None
    logger.info("shooter: none (track %s is not on %s, which took the only field goal)", shooter, off["team"])
    return None, {"from": shooter, "to": None, "by": "official_pbp_team", **info,
                  "why": f"the video's shooter is not on {off['team']}, and no {off['team']} track had the ball "
                         "or was guarded by it at the release"}, None


def official_shooter_track(identity: dict | None, off: dict | None, shooter: int | None) -> int | None:
    """The track of the PBP scorer when the official record pins the shot and the video picked someone else.

    Only when the clip window holds exactly one field goal of a firmly matched game, the ball-path shooter's
    name is not that player's, and exactly one track is named as that player on his team (2026-11-19 c08: a
    defender under the ball was taken for the shooter of LeBron James's pull-up three)."""
    if not off or off["match"].get("name") or off["match"].get("n_candidates") != 1 or off.get("conf", 0) < 0.9:
        return None
    if not off.get("person_id"):
        return None
    players = (identity or {}).get("players") or {}
    hits = [int(tid) for tid, p in players.items()
            if str(p.get("person_id")) == str(off["person_id"]) and p.get("team_abbr") == off.get("team")]
    if len(hits) != 1 or hits[0] == shooter:
        return None
    return hits[0]


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
