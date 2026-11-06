"""Per-track action windows (catch, dribble, shoot, jump, screen) and the play-level event list."""
from __future__ import annotations

import numpy as np

from .court import Court, nearest_rim
from .detector import Det
from .possession import Segment
from .release import _iou
from .tracks import Tracks

JUMP_POSES = ("player-jump-shot", "player-layup-dunk", "player-shot-block")


def court_paths(tracks: Tracks, court: Court, frames: list[int]) -> dict[int, dict[int, tuple[float, float]]]:
    """Court position per track per sampled frame (observed feet only)."""
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for f in frames:
        for p in tracks.players(f):
            if p.foot is None or not p.observed:
                continue
            xy = court.to_court(f, p.foot)
            if xy is not None:
                out.setdefault(p.track_id, {})[f] = xy
    return out


def _speed(path: dict[int, tuple[float, float]], f: int, fps: float, half_s: float = 0.25) -> float | None:
    fs = sorted(g for g in path if abs(g - f) <= half_s * fps)
    if len(fs) < 2 or fs[-1] == fs[0]:
        return None
    a, b = path[fs[0]], path[fs[-1]]
    return float(np.hypot(b[0] - a[0], b[1] - a[1]) / ((fs[-1] - fs[0]) / fps))


def jump_windows(dets: dict[int, list[Det]], tracks: Tracks, fps: float) -> list[dict]:
    """Contiguous frames where DEIM labels a track's box as a jump shot / layup / block pose."""
    hits: dict[int, list[tuple[int, float, str]]] = {}
    for f, ds in dets.items():
        for d in ds:
            if d.cls not in JUMP_POSES or d.score < 0.45:
                continue
            best = max(((_iou(p.box, d.box), p.track_id) for p in tracks.players(f)), default=(0, None))
            if best[0] > 0.5:
                hits.setdefault(best[1], []).append((f, d.score, d.cls))
    out = []
    for tid, hs in hits.items():
        hs.sort()
        cur = [hs[0]]
        for h in hs[1:] + [None]:
            if h is not None and (h[0] - cur[-1][0]) / fps <= 0.25:
                cur.append(h)
                continue
            if (cur[-1][0] - cur[0][0]) / fps >= 0.1 or len(cur) >= 3:
                kinds = [c[2] for c in cur]
                out.append({"track": tid, "f0": cur[0][0], "f1": cur[-1][0], "pose": max(set(kinds), key=kinds.count),
                            "conf": round(float(np.mean([c[1] for c in cur])), 3)})
            if h is not None:
                cur = [h]
    return out


def screens(segs: list[Segment], paths: dict[int, dict[int, tuple[float, float]]], tracks: Tracks, fps: float,
            frames: list[int]) -> list[dict]:
    """Screens: an offensive player P stands still (< 3 ft/s) with a defender within 5 ft while a moving teammate
    (> 4 ft/s; the ball handler or a cutter) passes within 6 ft of P. Offense = the team holding the ball."""
    out = []
    for s in segs:
        if s.type not in ("hold", "dribble") or s.holder is None:
            continue
        team = tracks.team_of(s.holder)
        mates = [t for t in paths if tracks.team_of(t) == team]
        opps = [t for t in paths if tracks.team_of(t) not in (None, team)]
        for m in mates:
            hit, partner = [], {}
            for f in frames:
                if not (s.f0 <= f <= s.f1) or f not in paths[m]:
                    continue
                pm = paths[m][f]
                vm = _speed(paths[m], f, fps)
                if vm is None or vm > 3.0:
                    continue
                if not any(f in paths[o] and np.hypot(paths[o][f][0] - pm[0], paths[o][f][1] - pm[1]) < 5.0 for o in opps):
                    continue
                for q in mates:
                    if q == m or f not in paths[q]:
                        continue
                    vq = _speed(paths[q], f, fps)
                    if vq is not None and vq > 4.0 and np.hypot(paths[q][f][0] - pm[0], paths[q][f][1] - pm[1]) < 6.0:
                        hit.append(f)
                        partner[q] = partner.get(q, 0) + 1
                        break
            if len(hit) >= 2 and (hit[-1] - hit[0]) / fps >= 0.15:
                q = max(partner, key=partner.get)
                base = 0.5 if q == s.holder else 0.35
                out.append({"track": m, "f0": hit[0], "f1": hit[-1], "for": q, "on_ball": q == s.holder,
                            "conf": round(min(0.8, base + 0.05 * len(hit)), 3)})
    # one window per screener: merge overlapping detections
    out.sort(key=lambda x: (x["track"], x["f0"]))
    merged: list[dict] = []
    for x in out:
        if merged and merged[-1]["track"] == x["track"] and x["f0"] <= merged[-1]["f1"] + int(0.5 * fps):
            merged[-1]["f1"] = max(merged[-1]["f1"], x["f1"])
            merged[-1]["conf"] = max(merged[-1]["conf"], x["conf"])
        else:
            merged.append(x)
    return merged


def action_windows(segs: list[Segment], jumps: list[dict], scr: list[dict], shot: dict | None,
                   fps: float) -> list[dict]:
    acts = []
    for s in segs:
        if s.type == "dribble":
            acts.append({"track": s.holder, "type": "dribble", "f0": s.f0, "f1": s.f1, "conf": s.conf})
        elif s.type == "pass" and s.to is not None:
            acts.append({"track": s.to, "type": "catch", "f0": max(s.f0, s.f1 - int(0.15 * fps)),
                         "f1": s.f1 + int(0.3 * fps), "conf": s.conf})
            acts.append({"track": s.frm, "type": "pass", "f0": max(0, s.f0 - int(0.2 * fps)), "f1": s.f0 + int(0.1 * fps),
                         "conf": s.conf})
    if shot and shot.get("shooter_track") is not None and shot.get("release_f") is not None:
        rf = shot["release_f"]
        acts.append({"track": shot["shooter_track"], "type": "shoot", "f0": int(rf - 0.45 * fps),
                     "f1": int(rf + 0.15 * fps), "conf": shot.get("conf", 0.5)})
    for j in jumps:
        acts.append({"track": j["track"], "type": "jump", "f0": j["f0"], "f1": j["f1"], "conf": j["conf"],
                     "pose": j["pose"]})
    for s in scr:
        acts.append({"track": s["track"], "type": "screen", "f0": s["f0"], "f1": s["f1"], "conf": s["conf"],
                     "for": s["for"], "on_ball": s.get("on_ball")})
    return sorted(acts, key=lambda a: (a["f0"], a["track"] or 0))


def drives(segs: list[Segment], paths: dict[int, dict[int, tuple[float, float]]], fps: float,
           frames: list[int]) -> list[dict]:
    """Dribble segments where the holder moves toward the nearest rim faster than 7 ft/s for >= 0.5 s."""
    out = []
    for s in segs:
        if s.type != "dribble" or s.holder not in paths:
            continue
        p = paths[s.holder]
        fs = [f for f in frames if s.f0 <= f <= s.f1 and f in p]
        run: list[int] = []
        for a, b in zip(fs, fs[1:] + [None]):
            ok = False
            if b is not None:
                rim = nearest_rim(p[a])
                da = np.hypot(p[a][0] - rim[0], p[a][1] - rim[1])
                db = np.hypot(p[b][0] - rim[0], p[b][1] - rim[1])
                ok = (da - db) / ((b - a) / fps) > 7.0
            if ok:
                run.append(a)
            else:
                if run and (run[-1] - run[0]) / fps >= 0.4:
                    out.append({"track": s.holder, "f0": run[0], "f1": a, "conf": round(min(0.8, s.conf + 0.1), 3)})
                run = []
    return out
