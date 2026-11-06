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
    m = re.match(r"^\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*$", clock)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r"^\s*(\d{1,2}(?:\.\d+)?)\s*$", clock)
    return float(m.group(1)) if m else None


def _pbp_path(gid: str) -> Path | None:
    p = DATA / "pbp" / f"{gid}.json"
    if p.exists():
        return p
    idx = DATA / "pbp_mirror" / "pbp_index.json"
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


def xfg_for(gid: str, person_id: int | str, action_number: int) -> float | None:
    p = DATA / "gpshot" / gid / f"{person_id}.json"
    if not p.exists():
        return None
    for s in json.loads(p.read_text()).get("data", {}).get("shotList", []):
        if int(s.get("eventId", -1)) == int(action_number):
            return float(s["value"]) if s.get("value") is not None else None
    return None


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
