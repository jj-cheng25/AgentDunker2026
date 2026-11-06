"""tracks.json + video -> identity.json: jersey numbers, team names, game match, player names.

Every decision degrades instead of failing: no credential / no score bug / no matching game give
team + number only (or nothing) with low confidence, never an exception for the caller.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import games, vlm
from .crops import select_crops
from .jersey import ACCEPT_SHARE, Reading, read_tracks, reread_choices
from .names_zh import TEAM_ZH, player_zh
from .scoreboard import read_scoreboard
from .teams import name_teams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentityConfig:
    crops_per_track: int = 8
    max_strong: int = 20          # gpt-6-astra reads per clip (brief: ≤ 20)
    workers: int = 8
    deduce: bool = True           # last unread on-court jersey of a fully read team goes to its last track
    min_fit: float = 0.75         # share of read jerseys that must be on the floor for a game match
    max_state_share: float = 0.05 # score-bug state must fit at most this share of all local games


def _scoreboard_teams(sb: list[dict]) -> list[str]:
    c = Counter(tuple(sorted(t for t in r["teams"] if t)) for r in sb)
    for pair, _ in c.most_common():
        if len(pair) == 2:
            return list(pair)
    return []


def _lineups(gid: str, sb: list[dict]) -> dict[str, set[str]] | None:
    """Union of on-court jerseys at every scoreboard reading of the clip."""
    out: dict[str, set[str]] = {}
    for r in sb:
        if r.get("period") is None or r.get("clock_s") is None:
            continue
        oc = games.on_court(gid, r["period"], r["clock_s"])
        for tri, js in (oc or {}).items():
            out.setdefault(tri, set()).update(js)
    return out or None


def _fit(nums: dict[str, list[str]], mapping: dict[str, str], lineup: dict[str, set[str]]) -> tuple[int, int]:
    hit = sum(n in lineup.get(mapping[g], set()) for g, ns in nums.items() for n in ns)
    return hit, sum(len(ns) for ns in nums.values())


def _match_game(sb: list[dict], teams: list[str], nums: dict[str, list[str]], cfg: IdentityConfig) -> dict:
    """Pick the game only when exactly one candidate fits both the score bug and the read jerseys."""
    res = {"game_id": "", "status": "none", "conf": 0.0, "candidates": []}
    if len(teams) != 2:
        res["reason"] = "score bug teams not read"
        return res
    cands = games.candidates(set(teams), sb)
    if not cands:
        res["reason"] = f"no local game of {teams[0]}-{teams[1]} fits the score bug"
        return res
    maps = [{"A": teams[0], "B": teams[1]}, {"A": teams[1], "B": teams[0]}]
    good, plausible = [], []
    for c in cands:
        lu = _lineups(c["game_id"], sb)
        c["lineups"] = {k: sorted(v, key=int) for k, v in (lu or {}).items()}
        if not lu:
            plausible.append(c)
            continue
        fits = [_fit(nums, m, lu) for m in maps]
        k = max(range(2), key=lambda i: fits[i][0])
        hit, n = fits[k]
        c.update(roster_fit=[hit, n], mapping=maps[k], mapping_margin=fits[k][0] - fits[1 - k][0])
        if n >= 3 and hit / n >= 0.5:
            good.append(c)
        elif n < 3:
            plausible.append(c)
    res["candidates"] = [{k: v for k, v in c.items() if k != "lineups"} for c in cands]
    p = games.rarity(sb)
    res["state_share"] = None if p is None else round(p, 4)
    if len(good) + len(plausible) == 1 and p is not None and p <= cfg.max_state_share:
        c = (good + plausible)[0]
        hit, n = c.get("roster_fit", (0, 0))
        share = hit / n if n >= 3 else None
        need = cfg.min_fit if p > 0.01 else 0.5      # a rare state needs less lineup confirmation
        if (share is None and p <= 0.01) or (share is not None and share >= need):
            res.update(status="matched", game_id=c["game_id"],
                       conf=round((1 - p) * (0.6 + 0.4 * (share if share is not None else 0.5) * min(1.0, n / 6)), 3))
    if res["status"] != "matched":
        res.update(status="uncertain", conf=0.0,
                   reason=f"{len(good) + len(plausible)} candidate(s) not rejected by the read jerseys of {len(cands)}; "
                          f"score-bug state shared by {res['state_share']} of local games")
    chosen = next((c for c in cands if c["game_id"] == res["game_id"]), None)
    if chosen:
        res["lineups"] = chosen["lineups"]
        res["mapping"] = chosen.get("mapping")
        res["mapping_margin"] = chosen.get("mapping_margin", 0)
        g = games.index()[chosen["game_id"]]
        res.update(home=g["home"], away=g["away"], date=g["date"], type=g["type"])
    return res


def _team_names(teams: list[str], vlm_map: dict | None, gm: dict) -> dict[str, dict]:
    """A/B -> {abbr, conf, source}. The on-court lineup decides when it clearly separates the two mappings."""
    vm = {g: (vlm_map or {}).get(g) for g in ("A", "B")}
    vconf = {"high": 0.85, "medium": 0.65, "low": 0.4}.get((vlm_map or {}).get("confidence"), 0.0)
    if len(teams) == 2 and not set(t for t in vm.values() if t) <= set(teams):
        vm, vconf = {"A": None, "B": None}, 0.0       # VLM named a team that is not on the score bug
    if gm.get("status") == "matched" and gm.get("mapping"):
        rm = gm["mapping"]
        if gm.get("mapping_margin", 0) >= 2:
            agree = vm == rm
            return {g: {"abbr": rm[g], "conf": 0.97 if agree else 0.85, "source": "lineup" + ("+vlm" if agree else "")}
                    for g in ("A", "B")}
    if vm["A"] and vm["B"]:
        return {g: {"abbr": vm[g], "conf": vconf, "source": "vlm"} for g in ("A", "B")}
    if len(teams) == 2 and (vm["A"] or vm["B"]):      # one side named: the other is the other score-bug team
        known = "A" if vm["A"] else "B"
        other = "B" if known == "A" else "A"
        vm[other] = next(t for t in teams if t != vm[known])
        return {g: {"abbr": vm[g], "conf": vconf * 0.8, "source": "vlm+scorebug"} for g in ("A", "B")}
    return {g: {"abbr": None, "conf": 0.0, "source": "none"} for g in ("A", "B")}


def _assign_numbers(players: list[dict], readings: dict[int, Reading], team_abbr: dict[str, str | None],
                    gm: dict, cfg: IdentityConfig) -> dict[int, dict]:
    """Per team, a max-weight one-to-one assignment of tracks to read numbers, then lineup deduction.

    Without a matched game a number needs ≥ 2 crops and ≥ 75 % of the track's read crops. With a matched
    game it must be on the floor and have weight ≥ 2 (two crops, or one crop read alike by two models);
    the lineup has already removed almost every wrong digit. A single read of a single crop never decides.
    """
    lineups = {tri: set(js) for tri, js in (gm.get("lineups") or {}).items()} if gm.get("status") == "matched" else {}
    out = {t["track_id"]: {"jersey": None, "conf": 0.0, "source": None, "evidence_frames": [],
                           "votes": dict(readings.get(t["track_id"], Reading(t["track_id"])).votes())} for t in players}
    for team in ("A", "B"):
        tids = [t["track_id"] for t in players if t["team"] == team]
        lu = lineups.get(team_abbr.get(team) or "")
        cand: dict[tuple[int, str], tuple[int, float, int]] = {}
        for tid in tids:
            sup, total = readings.get(tid, Reading(tid)).support(lu)
            rejected = []
            for n, (w, k) in sup.items():
                share = k / total
                if lu is not None:
                    if n not in lu:
                        rejected.append(n)
                        continue
                    ok = w >= 2
                else:
                    ok = k >= 2 and share >= ACCEPT_SHARE
                if ok:
                    conf = share * (1 - 0.5 ** w)
                    if lu is not None:      # lineup-checked: 0 wrong on the bench runs; keep the order, floor 0.6
                        conf = 0.6 + 0.4 * conf
                    cand[(tid, n)] = (w, round(conf, 3), k)
            if rejected:
                out[tid]["rejected"] = f"{','.join(rejected)} not on the floor"
        nums = sorted({n for _, n in cand})
        if nums:
            m = np.zeros((len(tids), len(nums)))
            for (tid, n), (w, _, _) in cand.items():
                m[tids.index(tid), nums.index(n)] = w
            for r, c in zip(*linear_sum_assignment(-m)):
                if m[r, c] > 0:
                    tid, n = tids[r], nums[c]
                    _, conf, k = cand[(tid, n)]
                    out[tid].update(jersey=n, conf=conf, source="vote" if k >= 2 else "vote+lineup",
                                    evidence_frames=readings[tid].evidence_frames(n))
        if cfg.deduce and lu and len(lu) == 5 and len(tids) == 5:
            free = lu - {out[tid]["jersey"] for tid in tids}
            open_ = [tid for tid in tids if out[tid]["jersey"] is None]
            if len(free) == 1 and len(open_) == 1:
                # right exactly when the lineup and the other four numbers are (25/25 on the bench runs)
                others = min(out[tid]["conf"] for tid in tids if tid != open_[0])
                out[open_[0]].update(jersey=free.pop(), conf=others, source="lineup-deduction")
    return out


def identify_video(video: str, tracks: dict, cfg: IdentityConfig = IdentityConfig()) -> dict:
    t0 = time.time()
    n_ledger = len(vlm.LEDGER)
    v = tracks["video"]
    fps, n_frames = float(v["fps"]), int(v["n_frames"])
    players = [t for t in tracks["tracks"] if t.get("role") == "player" and t.get("team") in ("A", "B")]
    groups = {g: [t["track_id"] for t in players if t["team"] == g] for g in ("A", "B")}
    timing = {}

    crops = select_crops(video, players, v["width"], v["height"], fps, k=cfg.crops_per_track)
    timing["crops"] = round(time.time() - t0, 1)
    with ThreadPoolExecutor(2) as ex:
        f_sb = ex.submit(read_scoreboard, video, n_frames, fps)
        readings = read_tracks(crops, max_strong=cfg.max_strong, workers=cfg.workers)
        sb = f_sb.result()
    if not sb:
        sb = read_scoreboard(video, n_frames, fps, frames=[int(n_frames * q) for q in (0.25, 0.4, 0.7)])
    timing["read"] = round(time.time() - t0, 1)

    teams = _scoreboard_teams(sb)
    views = {tid: rd.views for tid, rd in readings.items()}
    vlm_map = name_teams(groups, crops, views, teams) if players else None
    nums = {g: [n for tid in groups[g] if (n := readings.get(tid, Reading(tid)).best()[0])] for g in ("A", "B")}
    gm = _match_game(sb, teams, nums, cfg)
    team_names = _team_names(teams, vlm_map, gm)
    for g in ("A", "B"):
        team_names[g]["name_zh"] = TEAM_ZH.get(team_names[g]["abbr"] or "")
    abbr = {g: team_names[g]["abbr"] for g in ("A", "B")}
    numbers = _assign_numbers(players, readings, abbr, gm, cfg)
    if gm["status"] == "matched" and gm.get("lineups"):
        # Tracks still without a number: ask the strong model to choose among the team's free on-court jerseys.
        choices = {}
        for g in ("A", "B"):
            lu = set(gm["lineups"].get(abbr[g] or "", []))
            free = sorted(lu - {numbers[tid]["jersey"] for tid in groups[g]}, key=int)
            for tid in groups[g]:
                if numbers[tid]["jersey"] is None and free:
                    choices[tid] = free
        used = sum(1 for e in vlm.LEDGER[n_ledger:] if e["model"] == vlm.STRONG)
        for tid, r in reread_choices(crops, choices, cfg.max_strong - used, cfg.workers).items():
            readings[tid].reads[vlm.STRONG + ":choice"] = r
        if choices:
            numbers = _assign_numbers(players, readings, abbr, gm, cfg)

    ros = games.roster(gm["game_id"]) if gm["status"] == "matched" else None
    out_players = {}
    for t in players:
        n = numbers[t["track_id"]]
        tri = abbr[t["team"]]
        entry = ((ros or {}).get(tri) or {}).get(n["jersey"] or "", {}) if n["jersey"] else {}
        name = entry.get("name")
        zh = player_zh(gm["game_id"], entry.get("person_id")) if name else None
        out_players[str(t["track_id"])] = {
            "jersey": n["jersey"], "team": t["team"], "team_abbr": tri, "name": name,
            "person_id": entry.get("person_id"), "name_zh": (zh or {}).get("name_zh"),
            "name_zh_short": (zh or {}).get("name_zh_short"), "team_name_zh": TEAM_ZH.get(tri or ""),
            "conf": round(n["conf"] * (1.0 if not name else min(1.0, 0.5 + gm["conf"])), 3),
            "team_conf": team_names[t["team"]]["conf"], "evidence_frames": n["evidence_frames"],
            "source": n["source"], "votes": n["votes"], **({"rejected": n["rejected"]} if "rejected" in n else {})}
    clocks = [r["clock"] for r in sb if r.get("clock")]
    periods = [r["period"] for r in sb if r.get("period")]
    game_match = {
        "game_id": gm["game_id"], "status": gm["status"], "conf": gm["conf"],
        "period": Counter(periods).most_common(1)[0][0] if periods else 0,
        "clock": [clocks[0], clocks[-1]] if clocks else ["", ""],
        **{k: gm[k] for k in ("home", "away", "date", "type", "state_share", "reason") if k in gm},
        "candidates": gm.get("candidates", []),
    }
    timing["total"] = round(time.time() - t0, 1)
    return {
        "players": out_players,
        "teams": team_names,
        "game_match": game_match,
        "scoreboard": sb,
        "meta": {"video": {k: v.get(k) for k in ("path", "sha256", "fps", "n_frames")},
                 "vlm_teams": vlm_map, "api": vlm.ledger_summary(vlm.LEDGER[n_ledger:]), "timing_s": timing},
    }
