"""Build the holo renderer's play JSON from M1 tracks, M2 court, optional M3 identity and M5 events.

Decisions (each with a degraded fallback, never an exception for "uncertain"):
- window: longest calibrated stretch of one camera shot (or ``start``/``end``), ending ~1 s after release;
- players: up to 5 per team with the most court measurements in the window;
- attacked basket: shooter's release position, else the half where the players stand;
- offense: shooter's team, else the team standing farther from the basket;
- ball: only where possession is known (events), else hidden;
- names: identity (conf >= 0.5) -> name + number; else number hint (optional); else team only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .annotations import validate as validate_annotations
from .trajectory import COURT_X, CourtFrames, PlayerPath, SmoothConfig, build_paths

logger = logging.getLogger(__name__)
FT = 0.3048
TEAM_ZH = {
    "ATL": "老鹰", "BOS": "凯尔特人", "BKN": "篮网", "CHA": "黄蜂", "CHI": "公牛", "CLE": "骑士", "DAL": "独行侠",
    "DEN": "掘金", "DET": "活塞", "GSW": "勇士", "HOU": "火箭", "IND": "步行者", "LAC": "快船", "LAL": "湖人",
    "MEM": "灰熊", "MIA": "热火", "MIL": "雄鹿", "MIN": "森林狼", "NOP": "鹈鹕", "NYK": "尼克斯", "OKC": "雷霆",
    "ORL": "魔术", "PHI": "76人", "PHX": "太阳", "POR": "开拓者", "SAC": "国王", "SAS": "马刺", "TOR": "猛龙",
    "UTA": "爵士", "WAS": "奇才",
}
EVENT_ZH = {"pass": "传球", "screen": "掩护", "drive": "突破", "shot": "出手", "rebound": "篮板", "cut": "空切",
            "closeout": "扑防", "help": "协防"}


@dataclass(frozen=True)
class SceneConfig:
    lead_in: float = 0.5           # s skipped after a camera cut (camera settles)
    end_before_release: float = 0.65  # s; the shot itself is shown as a live freeze by M7 (reference: 16.07 vs 16.72)
    min_window: float = 3.0        # s; shorter -> no replay (M7 skips the 3D segment)
    target_output: float = 24.0    # s of replay the default speed aims for
    min_speed: float = 0.5
    max_speed: float = 1.0
    shot_slow: float = 0.5         # extra slow factor over the 1.5 s before release
    end_hold: float = 1.0          # s freeze on the last frame
    gap_bridge: float = 0.35       # s of non-calibrated frames bridged inside a window (flashes)
    jersey_hints: bool = True      # show M1 jersey hints when M3 identity is absent
    identity_min_conf: float = 0.5
    orbit_radians: float = 0.10
    push: float = 0.05
    fit_scale: float = 0.92        # camera distance vs the conservative all-actors fit (closer = reference-like)
    end_judge: float = 2.0         # s at the end of the window that decide basket and offense
    crossing_ft: float = 6.0       # centroid this far into the other half = possession still coming up
    crossing_lead: float = 1.0     # s kept before the final half-court crossing
    min_observed_at_end: int = 8   # players that must be observed at the last replay frame
    pre_release_hold: float = 0.8  # s the shooter is shown holding the ball when possession is otherwise unknown


def _frame_times(tracks: dict) -> dict[int, float]:
    out = {}
    for tr in tracks["tracks"]:
        for fr in tr["frames"]:
            out.setdefault(fr["f"], fr["t"])
    return out


def choose_window(tracks: dict, court: dict, events: Optional[dict], cfg: SceneConfig,
                  start: Optional[float] = None, end: Optional[float] = None) -> Optional[tuple[float, float]]:
    """Source-time window: longest calibrated run inside one camera shot (gaps <= gap_bridge bridged)."""
    fps = float(tracks["video"]["fps"])
    n = int(tracks["video"]["n_frames"])
    ok = np.zeros(n, bool)
    for fr in court.get("frames", []):
        if fr.get("ok") and 0 <= fr["f"] < n:
            ok[fr["f"]] = True
    shots = tracks.get("shots") or [{"start_f": 0, "end_f": n - 1}]
    ftime = _frame_times(tracks)
    t_of = lambda f: ftime.get(f, f / fps)  # noqa: E731
    release = (events or {}).get("shot", {}).get("release_t")
    best = None
    bridge = int(round(cfg.gap_bridge * fps))
    for sh in shots:
        a, b = sh["start_f"], sh["end_f"]
        runs, cur, gap = [], None, 0
        for f in range(a, b + 1):
            if ok[f]:
                if cur is None:
                    cur = [f, f]
                cur[1], gap = f, 0
            elif cur is not None:
                gap += 1
                if gap > bridge:
                    runs.append(cur); cur, gap = None, 0
        if cur is not None:
            runs.append(cur)
        for r0, r1 in runs:
            t0, t1 = t_of(r0) + (cfg.lead_in if r0 == a and a > 0 else 0.0), t_of(r1)
            if release is not None and t0 <= release <= t1:
                t1 = min(t1, release - cfg.end_before_release)
                score = 1e6 + t1 - t0            # the run containing the shot always wins
            else:
                score = t1 - t0
            if best is None or score > best[0]:
                best = (score, t0, t1)
    if best is None:
        return None
    t0, t1 = best[1], best[2]
    if start is not None:
        t0 = max(t0, start) if end is None or start < t1 else start
    if end is not None:
        t1 = min(t1, end) if end > t0 else end
    if start is not None and end is not None:   # explicit request wins; smoother holds where uncalibrated
        t0, t1 = start, end
    need = 1.0 if (start is not None and end is not None) else cfg.min_window
    return (t0, t1) if t1 - t0 >= need else None


def _pick_players(paths: list[PlayerPath]) -> tuple[list[PlayerPath], list[int]]:
    keep, dropped = [], []
    for team in ("A", "B"):
        ps = sorted([p for p in paths if p.team == team], key=lambda p: -p.n_meas)
        keep += ps[:5]
        dropped += [p.track_id for p in ps[5:]]
    return keep, dropped


def _track_xy_at(path: PlayerPath, t: float) -> np.ndarray:
    k = int(np.clip(np.searchsorted(path.t, t), 0, len(path.t) - 1))
    return path.xy[k]


def parse_timeline(knots) -> list[list[float]]:
    """Validate director knots [[out_s, src_pts], ...]: out strictly increasing from 0, src non-decreasing."""
    k = [[float(a), float(b)] for a, b in knots]
    if len(k) < 2 or not all(np.isfinite(v) for row in k for v in row):
        raise ValueError("timeline needs >= 2 finite [out_s, src_s] knots")
    if k[0][0] != 0.0:
        k.insert(0, [0.0, k[0][1]])          # hold the first source frame until the first knot
    if any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(k, k[1:])):
        raise ValueError("timeline: output times must increase and source times must not decrease")
    if k[-1][1] <= k[0][1]:
        raise ValueError("timeline covers no source time")
    return k


def build_play(tracks: dict, court: dict, identity: Optional[dict] = None, events: Optional[dict] = None,
               start: Optional[float] = None, end: Optional[float] = None, cfg: SceneConfig = SceneConfig(),
               smooth: SmoothConfig = SmoothConfig(), timeline: Optional[list] = None,
               annotations=None) -> tuple[Optional[dict], dict]:
    """Return (play JSON or None, report). ``None`` means no usable 3D window (caller skips the segment).

    ``timeline`` = director pacing [[out_s, src_pts], ...]; it sets the window and the export clock.
    ``annotations`` = caller tactical graphics (renderer format, see annotations.py); invalid entries are dropped.
    """
    report: dict = {"ok": False, "degraded": []}
    knots = None
    if timeline is not None:
        try:
            knots = parse_timeline(timeline)
            start, end = knots[0][1], knots[-1][1]
        except (TypeError, ValueError) as e:
            report["degraded"].append(f"timeline ignored ({e}); default pacing used")
    window = choose_window(tracks, court, events, cfg, start, end)
    if window is None:
        report["reason"] = "no calibrated wide-shot window >= %.1f s" % cfg.min_window
        return None, report
    t0, t1 = window
    paths, dropped = _pick_players(build_paths(tracks, court, t0, t1, smooth))
    if paths and end is None:
        # Do not end on a stretch where key players are only predicted (film: Davis undetected after 16.1 s).
        count = np.sum([p.observed for p in paths], axis=0)
        ok = np.nonzero(count >= min(len(paths), cfg.min_observed_at_end))[0]
        t_last = float(paths[0].t[ok[-1]]) if len(ok) else t0
        if t_last < t1 - 0.2 and t_last - t0 >= cfg.min_window:
            report["degraded"].append(f"window end trimmed {t1:.2f}->{t_last:.2f} s (players unobserved)")
            t1 = t_last
            paths, dropped = _pick_players(build_paths(tracks, court, t0, t1, smooth))
    if dropped:
        report["degraded"].append(f"extra tracks not shown: {dropped}")
    if len(paths) < 2:
        report["reason"] = "fewer than 2 players with court positions"
        return None, report
    for team in ("A", "B"):
        k = sum(p.team == team for p in paths)
        if k < 5:
            report["degraded"].append(f"team {team}: only {k} players")
    shot = (events or {}).get("shot") or {}
    shooter = shot.get("shooter_track")
    release = shot.get("release_t")
    by_id = {p.track_id: p for p in paths}
    if shooter is not None and shooter not in by_id:
        report["degraded"].append(f"shooter track {shooter} not among shown players")
        shooter = None
    # Attacked basket and offense, from where the play ends (possession can change inside a clip).
    tail = paths[0].t >= t1 - cfg.end_judge
    if shooter is not None and release is not None and t0 <= release <= t1 + 2.0:
        side = np.sign(_track_xy_at(by_id[shooter], min(release, t1))[0]) or -1.0
    else:
        side = np.sign(np.median(np.concatenate([p.xy[tail, 0] for p in paths]))) or -1.0
    if start is None:
        # Start the final possession ~1 s before the players' centroid last crossed into the attacked half.
        cx = np.mean([p.xy[:, 0] for p in paths], axis=0) * side
        back = np.nonzero(cx < -cfg.crossing_ft)[0]
        if len(back):
            t_cross = float(paths[0].t[back[-1]])
            new_t0 = max(t0, t_cross - cfg.crossing_lead)
            if new_t0 > t0 + 0.5 and t1 - new_t0 >= cfg.min_window:
                report["degraded"].append(f"window start moved {t0:.2f}->{new_t0:.2f} s (final possession)")
                t0 = new_t0
                paths, _ = _pick_players(build_paths(tracks, court, t0, t1, smooth))
                by_id = {p.track_id: p for p in paths}
                tail = paths[0].t >= t1 - cfg.end_judge
    hoop = np.array([side * (COURT_X - 5.25), 0.0])
    if shooter is not None:
        off_team = by_id[shooter].team
    else:
        d = {tm: np.mean([np.linalg.norm(p.xy[tail] - hoop, axis=1).mean() for p in paths if p.team == tm] or [0])
             for tm in ("A", "B")}
        off_team = max(d, key=d.get)
        report["degraded"].append("offense inferred from spacing (farther team from basket)")
    report.update(window=[round(t0, 3), round(t1, 3)], basket="left" if side < 0 else "right", offense_team=off_team)

    def to_m(xy: np.ndarray) -> np.ndarray:
        # renderer: X lateral (+ = far sideline when the basket is on the left), Z = metres from baseline
        if side < 0:
            return np.stack([-xy[:, 1] * FT, (xy[:, 0] + COURT_X) * FT], 1)
        return np.stack([xy[:, 1] * FT, (COURT_X - xy[:, 0]) * FT], 1)

    # Names.
    id_players = (identity or {}).get("players", {})
    abbr = {}
    track_meta = {tr["track_id"]: tr for tr in tracks["tracks"]}
    players, roster = [], {}
    rel = lambda t: round(float(t - t0), 3)  # noqa: E731  renderer clock starts at 0
    step = max(1, int(round(smooth.rate_hz / 10)))  # 10 Hz keyframes; renderer interpolates
    for p in paths:
        pid = f"T{p.track_id}"
        team = "off" if p.team == off_team else "def"
        m = to_m(p.xy)
        idx = list(range(0, len(p.t), step))
        if idx[-1] != len(p.t) - 1:
            idx.append(len(p.t) - 1)
        path = [[rel(p.t[i]), round(float(m[i, 0]), 3), round(float(m[i, 1]), 3)] for i in idx]
        info = id_players.get(str(p.track_id)) or {}
        num, name = "", ""
        if info and float(info.get("conf", 1.0)) >= cfg.identity_min_conf:
            num = str(info.get("jersey") or "")
            name = info.get("name_zh") or info.get("name") or ""
            if info.get("team_abbr"):
                abbr[team] = info["team_abbr"]
        elif cfg.jersey_hints and track_meta[p.track_id].get("jersey_hint"):
            num = str(track_meta[p.track_id]["jersey_hint"])
        roster[pid] = {"num": num, "name": name, "shortName": name}  # reference badges use full names
        players.append({"id": pid, "num": num, "team": team, "face": "ball", "path": path, "actions": [],
                        "observedFrac": round(p.observed_frac, 3),
                        "predictedIntervals": _intervals(p.t[~p.observed] - t0, 1.0 / smooth.rate_hz)})
    duration = rel(t1)
    _add_actions(players, paths, shooter, release, t0, t1, smooth.rate_hz)
    _add_marks(players)
    ball, ball_note = _ball_segments(events, shooter, release, t0, t1, {p.track_id for p in paths}, cfg)
    report["ball"] = ball_note
    teams = {k: TEAM_ZH.get(v, v) for k, v in abbr.items()}
    if len(teams) == 2:
        title = f"{teams['off']}对{teams['def']} · 3D 战术复盘"
    else:
        title = "3D 战术复盘"
    phases = _phases(events, t0, t1)
    play = {
        "title": title,
        "duration": duration,
        "camera": "sideline",
        "cameraPreset": {"pos": [-17, 9, 8.2], "target": [0.6, 0, 7.2]} if side < 0 else
                        {"pos": [17, 9, 8.2], "target": [-0.6, 0, 7.2]},
        "teams": {"off": teams.get("off", "进攻"), "def": teams.get("def", "防守")},
        "players": players,
        "fullCourt": bool(max(r[2] for pl in players for r in pl["path"]) > 13.5),
        "roster": roster,
        "ball": ball,
        "annotations": [],
        "phases": phases,
        "captions": [],
        "actorStyle": "anatomical-points",
        "exportFrameAllActors": True,
        "exportOrbitRadians": cfg.orbit_radians,
        "exportPush": cfg.push,
        "exportFitScale": cfg.fit_scale,
        "exportTimeline": ([[round(a, 3), round(b - t0, 3)] for a, b in knots] if knots else
                           _timeline(duration, None if release is None else release - t0, cfg)),
        "metadata": {"sourceTimeOffset": round(t0, 4), "sourceWindow": [round(t0, 4), round(t1, 4)],
                     "sourceCoverage": {"startSec": 0.0, "stopSec": duration},
                     "teamRolesVerified": True, "teamMembershipVerified": len(teams) == 2,
                     **({"ballObservation": "unknown"} if not ball else {})},
    }
    report["timeline_source"] = "director" if knots else "default"
    if annotations is not None:
        kept, dropped = validate_annotations(annotations, {p["id"] for p in players}, duration, t0)
        play["annotations"] = kept
        report["annotations"] = {"kept": len(kept), "dropped": dropped}
        if dropped:
            report["degraded"].append(f"{len(dropped)} annotations dropped")
    report.update(ok=True, n_players=len(players), duration_source=duration,
                  duration_output=play["exportTimeline"][-1][0],
                  observed_frac={p["id"]: p["observedFrac"] for p in players})
    return play, report


def _intervals(ts: np.ndarray, dt: float) -> list[list[float]]:
    out: list[list[float]] = []
    for t in ts:
        if out and t - out[-1][1] <= dt * 1.5:
            out[-1][1] = float(t)
        else:
            out.append([float(t), float(t)])
    return [[round(a, 2), round(b, 2)] for a, b in out]


def _add_actions(players: list[dict], paths: list[PlayerPath], shooter, release, t0, t1, rate) -> None:
    """Defenders 'defend' (stance) except while sprinting; shooter shoots/jumps at release."""
    for pl, p in zip(players, paths):
        if pl["team"] == "def":
            v = np.linalg.norm(np.gradient(p.xy, axis=0), axis=1) * rate * FT  # m/s
            slow = np.convolve(v < 3.5, np.ones(9) / 9, mode="same") > 0.5
            for a, b in _intervals(p.t[slow] - t0, 1.0 / rate):
                if b - a >= 0.3:
                    pl["actions"].append(["defend", a, b])
        if shooter is not None and p.track_id == shooter and release is not None and t0 <= release <= t1 + 0.5:
            r = release - t0
            pl["face"] = "hoop"
            pl["actions"] += [["shoot", round(r - 0.35, 3), round(r + 0.15, 3)],
                              ["jump", round(r - 0.25, 3), round(r + 0.4, 3)]]


def _add_marks(players: list[dict]) -> None:
    """Defender -> matchup by Hungarian on mean distance (used for facing when the ball is hidden)."""
    off = [p for p in players if p["team"] == "off"]
    de = [p for p in players if p["team"] == "def"]
    if not off or not de:
        return
    P = lambda pl: np.asarray(pl["path"])[:, 1:]  # noqa: E731
    C = np.array([[np.linalg.norm(P(d) - P(o), axis=1).mean() for o in off] for d in de])
    for i, j in zip(*linear_sum_assignment(C)):
        de[i]["mark"] = off[j]["id"]


def _ball_segments(events, shooter, release, t0, t1, shown: set, cfg: SceneConfig):
    """Ball only where possession is known (pass chain / shooter gather); hidden segments elsewhere."""
    passes = sorted([e for e in (events or {}).get("events", []) if e.get("type") == "pass"
                     and len(e.get("actors") or []) >= 2 and float(e.get("conf", 1.0)) >= 0.5], key=lambda e: e["t"])
    items = []  # (source_a, source_b, segment)
    holder, since = None, t0
    for e in passes:
        a, b = e["actors"][0], e["actors"][1]
        if a != holder:              # chain broken: possession before this pass is unknown
            since = max(since, e["t"] - 0.5)
        items.append((since, e["t"], {"hold": a}))
        flight = float(e.get("flight", 0.35))
        items.append((e["t"], e["t"] + flight, {"pass": [a, b], "arc": 0.4}))
        holder, since = b, e["t"] + flight
    if shooter is not None and release is not None:
        if holder != shooter:
            since = release - cfg.pre_release_hold
        items.append((since, release, {"hold": shooter}))
        items.append((release, release + 1.0, {"shot": shooter, "arc": 0.6}))
    segs, cursor = [], 0.0
    rel = lambda t: round(float(t - t0), 3)  # noqa: E731
    for a, b, seg in items:
        ids = [seg[k] for k in ("hold", "shot") if k in seg] + seg.get("pass", [])
        a, b = max(a, t0, t0 + cursor), min(b, t1)
        if b - a < (0.4 if "hold" in seg else 0.05) or any(i not in shown for i in ids):  # no ball flash-ins
            continue
        if rel(a) > cursor + 0.01:
            segs.append({"t": [cursor, rel(a)], "hidden": True})
        seg = {k: (f"T{v}" if k in ("hold", "shot") else [f"T{x}" for x in v] if k == "pass" else v)
               for k, v in seg.items()}
        if "hold" in seg:
            seg["mode"] = "dribble" if b - a > 1.0 else "hands"
        segs.append({"t": [rel(a), rel(b)], **seg})
        cursor = rel(b)
    if segs and cursor < rel(t1) - 0.01:
        segs.append({"t": [cursor, rel(t1)], "hidden": True})
    note = ("events" if passes else "shooter gather + shot") if segs else "hidden (possession unknown)"
    return segs, note


def _phases(events, t0, t1) -> list[dict]:
    """Minimap phase titles from events (name from label_zh or type; text from note)."""
    evs = sorted([e for e in (events or {}).get("events", []) if t0 - 1 <= e["t"] <= t1
                  and float(e.get("conf", 1.0)) >= 0.5], key=lambda e: e["t"])
    if any(e.get("label_zh") for e in evs):   # labelled phases given: unlabelled events do not split them
        evs = [e for e in evs if e.get("label_zh")]
    out = []
    for i, e in enumerate(evs):
        a = max(0.0, e["t"] - t0 - 0.3)
        b = (evs[i + 1]["t"] - t0 - 0.3) if i + 1 < len(evs) else t1 - t0
        name = e.get("label_zh") or EVENT_ZH.get(e.get("type"), "")
        if name and b > a:
            out.append({"t": [round(a, 2), round(b, 2)], "name": name, "intent": e.get("note", "")})
    return out


def _timeline(duration: float, release_rel: Optional[float], cfg: SceneConfig) -> list[list[float]]:
    """[[output_s, source_s], ...]: constant slow-down, extra slow 1.5 s before release, end hold."""
    speed = float(np.clip(duration / cfg.target_output, cfg.min_speed, cfg.max_speed))
    knots = [[0.0, 0.0]]
    if release_rel is not None and 1.5 < release_rel <= duration:
        a = release_rel - 1.5
        knots.append([a / speed, a])
        knots.append([knots[-1][0] + 1.5 / (speed * cfg.shot_slow), release_rel])
    last_out, last_src = knots[-1]
    knots.append([last_out + (duration - last_src) / speed, duration])
    knots.append([knots[-1][0] + cfg.end_hold, duration])
    return [[round(a, 3), round(b, 3)] for a, b in knots]
