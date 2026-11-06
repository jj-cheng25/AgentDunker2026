"""Season metrics for the hero card (official NBA China tables in ``$ADK_SEASON_DATA``, default ``data/season``).

Files: ``<season>_<RegularSeason|Playoffs|Preseason>_<gravity|leverage|shotDifficulty>.json`` with ``rows``
keyed by ``PLAYERID``. The value is a *season* metric of the player, shown as e.g. "2025-26 赛季引力 +16.2",
never as a fact about the play. Missing player, season or empty table → None (plain plate).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

DATA = Path(os.environ.get("ADK_SEASON_DATA", Path(os.environ.get("ADK_NBA_DATA", Path.home() / ".cache" / "agentdunker" / "nba_data")) / "season"))
TYPES = {"regular": "RegularSeason", "playoff": "Playoffs", "playoffs": "Playoffs", "play-in": "Playoffs",
         "playin": "Playoffs", "preseason": "Preseason"}
LABEL = {"RegularSeason": "", "Playoffs": "季后赛", "Preseason": "季前赛"}


def season_of(date: str) -> Optional[str]:
    """'2026-01-04' → '2025-26' (seasons start in October)."""
    try:
        y, m = int(date[:4]), int(date[5:7])
    except (TypeError, ValueError):
        return None
    start = y if m >= 9 else y - 1
    return f"{start}-{(start + 1) % 100:02d}"


@lru_cache(maxsize=32)
def _table(season: str, kind: str, table: str) -> dict:
    p = DATA / f"{season}_{kind}_{table}.json"
    try:
        rows = json.loads(p.read_text()).get("rows") or []
    except (OSError, ValueError):
        return {}
    return {str(r.get("PLAYERID")): r for r in rows}


def gravity(identity: Optional[dict], track_id: Optional[int]) -> Optional[dict]:
    """{'caption': '2025-26 赛季引力', 'value': '+16.2', 'raw': 16.237, 'source': file} or None."""
    if not identity or track_id is None:
        return None
    p = (identity.get("players") or {}).get(str(track_id)) or {}
    gm = identity.get("game_match") or {}
    pid = p.get("person_id")
    if not pid or float(p.get("conf") or 0) < 0.85 or gm.get("status") != "matched":
        return None
    season, kind = season_of(gm.get("date") or ""), TYPES.get(str(gm.get("type") or "").lower())
    if not season or not kind:
        return None
    row = _table(season, kind, "gravity").get(str(pid))
    v = row.get("OVERALL_GRAV") if row else None
    if not isinstance(v, (int, float)):
        return None
    caption = f"{season} 赛季引力" if kind == "RegularSeason" else f"{season} {LABEL[kind]}引力"
    return {"caption": caption, "value": f"{v:+.1f}", "raw": v,
            "source": f"{season}_{kind}_gravity.json", "scope": "season"}
