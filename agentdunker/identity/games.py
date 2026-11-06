"""Official game data: find games consistent with scoreboard readings; rosters and on-court lineups.

Data root: `ADK_NBA_DATA` (default: the hackathon `data/` mirror). A compact score-timeline index of
every local play-by-play file is built once and cached (~/.cache/agentdunker.identity/pbp_compact.json.gz).
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DATA = Path(os.environ.get("ADK_NBA_DATA",
                           Path.home() / ".cache" / "agentdunker" / "nba_data"))
INDEX_CACHE = Path(os.environ.get("ADK_ID_INDEX", Path.home() / ".cache" / "agentdunker.identity" / "pbp_compact.json.gz"))
CLOCK_TOL = 8.0        # s; scoreboard score vs. PBP score state


def secs(clock: str | None) -> float | None:
    if not clock:
        return None
    m = re.match(r"PT(\d+)M([\d.]+)S", clock)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r"^\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*$", clock)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r"^\s*(\d{1,2}(?:\.\d+)?)\s*$", clock)     # "45.3" under a minute
    return float(m.group(1)) if m else None


def _actions(path: Path) -> list:
    d = json.loads(path.read_text())
    if isinstance(d, dict):
        d = d.get("game", d).get("actions", [])
    return d


def _pbp_files() -> dict[str, dict]:
    files = {p.stem: {"path": str(p), "date": None, "type": None} for p in (DATA / "pbp").glob("*.json")}
    idx = DATA / "pbp_mirror" / "pbp_index.json"
    if idx.exists():
        for gid, meta in json.loads(idx.read_text()).items():
            files[gid] = {"path": str(DATA / meta["path"]), "date": meta.get("date"), "type": meta.get("type")}
    return files


def _compact(acts: list) -> dict | None:
    """home/away tricodes and the score timeline [(period, clock_s, home, away)] at every score change."""
    home = away = None
    prev = (0, 0)
    timeline = []
    for x in acts:
        if x.get("scoreHome") in (None, "") or x.get("period") is None:
            continue
        s = (int(x["scoreHome"]), int(x["scoreAway"]))
        if s != prev:
            tri = x.get("teamTricode")
            if tri:
                if s[0] > prev[0] and s[1] == prev[1]:
                    home = home or tri
                elif s[1] > prev[1] and s[0] == prev[0]:
                    away = away or tri
            c = secs(x.get("clock"))
            timeline.append((int(x["period"]), c if c is not None else 0.0, s[0], s[1]))
            prev = s
    tris = {x.get("teamTricode") for x in acts if x.get("teamTricode")}
    if home and not away and len(tris) == 2:
        away = (tris - {home}).pop()
    if away and not home and len(tris) == 2:
        home = (tris - {away}).pop()
    if not (home and away):
        return None
    return {"home": home, "away": away, "timeline": timeline}


@lru_cache(maxsize=1)
def index() -> dict[str, dict]:
    if INDEX_CACHE.exists():
        return json.loads(gzip.decompress(INDEX_CACHE.read_bytes()))
    out = {}
    for gid, meta in _pbp_files().items():
        try:
            c = _compact(_actions(Path(meta["path"])))
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            continue
        if c:
            out[gid] = {**c, "date": meta["date"], "type": meta["type"], "path": meta["path"]}
    INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_CACHE.write_bytes(gzip.compress(json.dumps(out).encode()))
    logger.info("games: indexed %d play-by-play files", len(out))
    return out


def _score_states(timeline: list, period: int, clock: float, tol: float) -> set[tuple[int, int]]:
    """All (home, away) scores in effect at some time within ±tol s of (period, clock)."""
    states = set()
    cur = (0, 0)
    lo, hi = clock - tol, clock + tol
    for p, c, h, a in timeline:
        if p < period or (p == period and c > hi):
            cur = (h, a)
            continue
        if p == period and c >= lo:
            states.add(cur)
            cur = (h, a)
            continue
        break
    states.add(cur)
    return states


def consistent(game: dict, reading: dict, tol: float = CLOCK_TOL) -> bool | None:
    """Does a scoreboard reading {period, clock_s, scores: {tri: n}} fit this game? None = not checkable."""
    sc = reading.get("scores") or {}
    if reading.get("period") is None or reading.get("clock_s") is None or not {game["home"], game["away"]} <= set(sc):
        return None
    want = (sc[game["home"]], sc[game["away"]])
    return want in _score_states(game["timeline"], reading["period"], reading["clock_s"], tol)


def candidates(teams: set[str], readings: list[dict]) -> list[dict]:
    """Games whose two teams are `teams` and whose score timeline fits the readings.

    Returns [{game_id, date, type, n_fit, n_checked}] for games that fit every checkable reading
    (one miss is tolerated when ≥ 3 readings are checkable).
    """
    out = []
    for gid, g in index().items():
        if {g["home"], g["away"]} != teams:
            continue
        res = [consistent(g, r) for r in readings]
        checked = [r for r in res if r is not None]
        if not checked:
            continue
        fit = sum(checked)
        if fit == len(checked) or (len(checked) >= 3 and fit >= len(checked) - 1):
            out.append({"game_id": gid, "date": g["date"], "type": g["type"], "home": g["home"], "away": g["away"],
                        "n_fit": fit, "n_checked": len(checked)})
    return out


def rarity(readings: list[dict]) -> float | None:
    """Share of all local games (any teams) whose score timeline fits every reading; None if uncheckable.

    A low share means the score bug state is specific, so a single same-teams candidate is unlikely to be a
    coincidence with a game missing from the local data."""
    rs = [r for r in readings if r.get("period") is not None and r.get("clock_s") is not None
          and len(r.get("scores") or {}) == 2]
    if not rs:
        return None
    n = 0
    for g in index().values():
        for flip in (False, True):
            ok = True
            for r in rs:
                a, b = list(r["scores"].values())
                want = (b, a) if flip else (a, b)
                if want not in _score_states(g["timeline"], r["period"], r["clock_s"], CLOCK_TOL):
                    ok = False
                    break
            if ok:
                n += 1
                break
    return n / max(1, len(index()))


def _boxscore(gid: str) -> dict | None:
    for p in (DATA / "boxscore" / f"{gid}.json", DATA / "us_recovery" / "official" / "boxscore" / f"{gid}.json"):
        if p.exists():
            d = json.loads(p.read_text())
            return d.get("game", d)
    return None


@lru_cache(maxsize=64)
def roster(gid: str) -> dict[str, dict[str, dict]] | None:
    """{tricode: {jersey: {name, person_id, played, starter}}} from the official boxscore."""
    g = _boxscore(gid)
    if not g:
        return None
    out = {}
    for side in ("homeTeam", "awayTeam"):
        t = g[side]
        out[t["teamTricode"]] = {
            str(p.get("jerseyNum", "")).strip(): {
                "name": p.get("name") or f"{p.get('firstName')} {p.get('familyName')}",
                "person_id": p.get("personId"), "played": str(p.get("played")) == "1",
                "starter": str(p.get("starter")) == "1"}
            for p in t.get("players", []) if str(p.get("jerseyNum", "")).strip()}
    return out


def on_court(gid: str, period: int, clock: float) -> dict[str, set[str]] | None:
    """Jerseys on the floor per team at (period, clock), from starters + PBP substitutions.

    Between-period changes are logged as substitutions at the period start. Substitutions during a dead
    ball share a clock value, so players subbed out exactly at `clock` are kept as well (the set may
    hold 6–7 jerseys around a substitution).
    """
    ros = roster(gid)
    g = index().get(gid)
    if not ros or not g:
        return None
    pid2 = {(tri, v["person_id"]): j for tri, ps in ros.items() for j, v in ps.items()}
    court = {tri: {j for j, v in ps.items() if v["starter"]} for tri, ps in ros.items()}
    if any(len(s) != 5 for s in court.values()):
        return None
    extra: dict[str, set[str]] = {tri: set() for tri in court}
    for x in _actions(Path(g["path"])):
        p, c = x.get("period"), secs(x.get("clock"))
        if p is None or c is None:
            continue
        if p > period or (p == period and c < clock):
            break
        if x.get("actionType") != "substitution":
            continue
        tri = x.get("teamTricode")
        j = pid2.get((tri, x.get("personId")))
        if tri not in court or j is None:
            continue
        at_clock = p == period and abs(c - clock) < 0.05
        if x.get("subType") == "out":
            if at_clock:
                extra[tri].add(j)
            court[tri].discard(j)
        elif x.get("subType") == "in":
            court[tri].add(j)
    return {tri: court[tri] | extra[tri] for tri in court}

