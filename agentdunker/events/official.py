"""Official shot data: match the detected shot to a PBP field-goal action and attach NBA xFG.

Inputs come from M3's `identity.json` (`game_match` + per-track names). Data root: `ADK_NBA_DATA`
(same convention as agentdunker.identity). xFG is the NBA China per-shot "shot quality" in `gpshot/<game>/<player>.json`,
keyed by `eventId` == PBP `actionNumber` (e.g. 0022500491 #88 K. Durant = 0.213, the reference film's 21.3%).
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DATA = Path(os.environ.get("ADK_NBA_DATA",
                           Path.home() / ".cache" / "agentdunker" / "nba_data"))
CLOCK_PAD_S = 3.0


def clock_secs(clock: str | None) -> float | None:
    if not clock:
        return None
    m = re.match(r"PT(\d+)M([\d.]+)S", clock)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    # "4:54", and under a minute the broadcast ":53.6" (no minutes: a whole score-bug sequence was lost)
    m = re.match(r"^\s*(\d{0,2}):(\d{2}(?:\.\d+)?)\s*$", clock)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    m = re.match(r"^\s*(\d{1,2}(?:\.\d+)?)\s*$", clock)
    return float(m.group(1)) if m else None


def _pbp_path(gid: str) -> Path | None:
    p = DATA / "pbp" / f"{gid}.json"
    if p.exists():
        return p
    # the main mirror index first, then older-season mirrors (data/pbp_mirror_hist: 2022-23 .. 2024-25)
    extra = sorted(p for p in DATA.glob("pbp_mirror_*/pbp_index.json") if p.parent.name != "pbp_mirror")
    for idx in [DATA / "pbp_mirror" / "pbp_index.json", *extra]:
        if idx.exists():
            meta = json.loads(idx.read_text()).get(gid)
            if meta and (DATA / meta["path"]).exists():
                return DATA / meta["path"]
    return None


def _actions(gid: str) -> list[dict]:
    p = _pbp_path(gid)
    if p is None:
        return []
    d = json.loads(p.read_text())
    if isinstance(d, dict):
        d = d.get("game", d).get("actions", [])
    return d


GPSHOT_URL = "https://api.nba.cn/nba-web/api/web/inside/gpShot"
GPSHOT_CACHE = Path(os.environ.get("ADK_NBA_CN_CACHE", Path.home() / ".cache" / "agentdunker.events" / "nba_cn"))


def _gpshot(gid: str, person_id: int | str, timeout: float = 15.0) -> dict | None:
    """The player's per-shot NBA China data for one game: the local data root first, then one live request to
    api.nba.cn (the endpoint agentdunker.identity already uses for Chinese names), cached; never in offline mode.
    The local `gpshot/` copy covers ~350 games; the live lookup covers every game the endpoint serves."""
    p = DATA / "gpshot" / gid / f"{person_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    cached = GPSHOT_CACHE / gid / f"{person_id}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    if os.environ.get("ADK_OFFLINE") == "1":
        return None
    try:
        import requests

        r = requests.get(GPSHOT_URL, params={"gameId": gid, "playerId": person_id}, timeout=timeout)
        r.raise_for_status()
        body = r.json() or {}
    except Exception as e:  # noqa: BLE001  (no network, bad JSON: no xFG, never an error)
        logger.warning("nba.cn gpShot %s/%s: %s", gid, person_id, type(e).__name__)
        return None
    if not (body.get("data") or {}).get("shotList"):
        return None
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(body, ensure_ascii=False))
    return body


def xfg_for(gid: str, person_id: int | str, action_number: int) -> float | None:
    d = _gpshot(gid, person_id)
    for s in ((d or {}).get("data") or {}).get("shotList") or []:
        if int(s.get("eventId", -1)) == int(action_number):
            return float(s["value"]) if s.get("value") is not None else None
    return None


START_PAD_S = 2.0     # the first score-bug read comes a second or two after the clip starts


def _clip_window(identity: dict | None, off: dict | None) -> tuple[float, float] | None:
    """(shot clock, game clock at the clip start + pad) in seconds for a firmly matched shot, or None."""
    if not off or off.get("conf", 0) < 0.9:
        return None
    gm = (identity or {}).get("game_match") or {}
    shot_c = clock_secs(off.get("clock"))
    # game clock at the clip start: each score-bug read pushed back by its clip time (as if the clock ran; a
    # stopped clock only widens the window, i.e. makes the check stricter)
    starts = [r["clock_s"] + r["t"] for r in (identity or {}).get("scoreboard") or []
              if r.get("clock_s") is not None and r.get("t") is not None and r.get("period") in (None, gm.get("period"))]
    starts += [c for c in (clock_secs(x) for x in (gm.get("clock") or [])) if c is not None]
    if shot_c is None or not starts or gm.get("game_id") != off.get("game_id"):
        return None
    return shot_c, max(starts) + START_PAD_S


def clock_to_t(identity: dict | None, clock_s: float | None, t_max: float) -> float | None:
    """Clip time of a game-clock value, from the nearest score-bug read before it (the clock runs down)."""
    gm = (identity or {}).get("game_match") or {}
    reads = sorted((r["t"], r["clock_s"]) for r in (identity or {}).get("scoreboard") or []
                   if r.get("clock_s") is not None and r.get("t") is not None and r.get("period") in (None, gm.get("period")))
    if clock_s is None or not reads:
        return None
    before = [(t, c) for t, c in reads if c >= clock_s]
    t0, c0 = before[-1] if before else reads[0]
    return min(max(0.0, t0 + (c0 - clock_s)), t_max)


def possession_changes(identity: dict | None, off: dict | None) -> list[dict]:
    """Steals and turnovers in the official record between the clip start and the shot (firm match only), in
    game order, one entry per change of possession: {clock_s, kind: steal|turnover, stealer*, loser*}.
    A steal and the turnover it causes share a clock value and are merged (c06: B. James lost ball, B. Podziemski
    steal, then Curry's transition three)."""
    w = _clip_window(identity, off)
    if w is None:
        return []
    shot_c, hi = w
    gm = (identity or {}).get("game_match") or {}
    by_clock: dict[float, dict] = {}
    for a in _actions(off["game_id"]):
        c = clock_secs(a.get("clock"))
        if a.get("period") != gm.get("period") or c is None or not (shot_c <= c <= hi):
            continue
        kind = a.get("actionType")
        if kind not in ("steal", "turnover"):
            continue
        e = by_clock.setdefault(round(c, 1), {"clock_s": c})
        role = "stealer" if kind == "steal" else "loser"
        e.update({f"{role}_id": a.get("personId") or None, role: a.get("playerNameI"), f"{role}_team": a.get("teamTricode")})
        if kind == "turnover":
            e["turnover"] = a.get("subType")
    out = []
    for _, e in sorted(by_clock.items(), reverse=True):
        e["kind"] = "steal" if e.get("stealer_team") else "turnover"
        out.append(e)
    return out


def defence_never_had_ball(identity: dict | None, off: dict | None, offense_tri: str | None,
                           defence_tri: str | None) -> bool:
    """True when the play-by-play of a firmly matched game shows no change of possession to the defence
    between the clip's first score-bug clock and the shot: no field goal, free throw, rebound, steal or jump
    ball of the defence and no turnover of the offence. Ball contacts by defenders before the release are then
    tracker/holder noise, not possessions (c03 original: three DAL passes, a DAL drive and a DAL "shot" inside
    a HOU half-court set)."""
    if not offense_tri or not defence_tri:
        return False
    w = _clip_window(identity, off)
    if w is None:
        return False
    shot_c, hi = w
    gm = (identity or {}).get("game_match") or {}
    for a in _actions(off["game_id"]):
        if a.get("period") != gm.get("period") or a.get("actionNumber") == off.get("action_number"):
            continue
        c = clock_secs(a.get("clock"))
        if c is None or not (shot_c <= c <= hi):
            continue
        kind, tri = a.get("actionType"), a.get("teamTricode")
        # the defence had the ball: its own shot, rebound or turnover; the offence's steal or block; the
        # offence gave it away: its turnover; a jump ball: contested either way (c06: LAL turnover, GSW steal)
        if tri == defence_tri and kind in ("2pt", "3pt", "freethrow", "rebound", "turnover", "jumpball"):
            return False
        if tri == offense_tri and kind in ("turnover", "steal", "block", "jumpball"):
            return False
    return True


def _name_key(name: str | None) -> str:
    return re.sub(r"[^a-z]", "", (name or "").split()[-1].lower()) if name else ""


def match_shot(identity: dict | None, shooter_track: int | None, made: bool | None) -> dict | None:
    """Find the PBP field goal for this clip's shot. Returns official fields or None.

    Candidates are field goals in the matched period whose clock lies inside the clip's clock window (padded).
    Preference: shooter name/jersey team match, then agreement with the video `made`, then the latest attempt.
    """
    gm = (identity or {}).get("game_match") or {}
    gid, period = gm.get("game_id"), gm.get("period")
    clocks = [clock_secs(c) for c in (gm.get("clock") or [])]
    clocks = [c for c in clocks if c is not None]
    if not gid or not period or not clocks:
        return None
    hi, lo = max(clocks) + CLOCK_PAD_S, min(clocks) - CLOCK_PAD_S
    fgs = [a for a in _actions(gid) if a.get("isFieldGoal") == 1 and a.get("period") == period
           and lo <= (clock_secs(a.get("clock")) or -1) <= hi]
    if not fgs:
        return None
    player = ((identity or {}).get("players") or {}).get(str(shooter_track)) or {}
    want_name, want_team = _name_key(player.get("name")), player.get("team_abbr")

    def rank(a: dict) -> tuple:
        name_ok = bool(want_name) and _name_key(a.get("playerNameI") or a.get("playerName")) == want_name
        team_ok = bool(want_team) and a.get("teamTricode") == want_team
        made_ok = made is not None and (a.get("shotResult") == "Made") == made
        return (name_ok, team_ok, made_ok, -(clock_secs(a.get("clock")) or 0))

    a = max(fgs, key=rank)
    r = rank(a)
    out = {
        "game_id": gid, "action_number": a.get("actionNumber"), "person_id": a.get("personId"),
        "player": a.get("playerNameI"), "team": a.get("teamTricode"), "made": a.get("shotResult") == "Made",
        "points": 3 if a.get("actionType") == "3pt" else 2, "description": a.get("description"),
        "clock": a.get("clock"), "shot_distance_ft": a.get("shotDistance"),
        "match": {"name": r[0], "team": r[1], "made": r[2], "n_candidates": len(fgs)},
    }
    out["xfg"] = xfg_for(gid, a.get("personId"), a.get("actionNumber")) if a.get("personId") else None
    # Confidence: the clip window pins the play; name or team agreement and a single candidate make it firm.
    out["conf"] = round(min(1.0, float(gm.get("conf") or 0.5)) * (1.0 if (r[0] or len(fgs) == 1) else
                                                                  0.8 if r[1] else 0.6), 3)
    return out
