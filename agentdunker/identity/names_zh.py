"""Chinese player and team names (NBA China). One request per player, cached per personId; failures give None.

Endpoint: api.nba.cn inside/gpShot (reachable from cpu-0; it carries playerFirstName/playerLastName in
Chinese for any gameId + playerId).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

URL = "https://api.nba.cn/nba-web/api/web/inside/gpShot"
CACHE = Path(os.environ.get("ADK_ID_ZH_CACHE", Path.home() / ".cache" / "agentdunker.identity" / "nba_cn"))

TEAM_ZH = {"ATL": "老鹰", "BOS": "凯尔特人", "BKN": "篮网", "CHA": "黄蜂", "CHI": "公牛", "CLE": "骑士", "DAL": "独行侠",
           "DEN": "掘金", "DET": "活塞", "GSW": "勇士", "HOU": "火箭", "IND": "步行者", "LAC": "快船", "LAL": "湖人",
           "MEM": "灰熊", "MIA": "热火", "MIL": "雄鹿", "MIN": "森林狼", "NOP": "鹈鹕", "NYK": "尼克斯", "OKC": "雷霆",
           "ORL": "魔术", "PHI": "76人", "PHX": "太阳", "POR": "开拓者", "SAC": "国王", "SAS": "马刺", "TOR": "猛龙",
           "UTA": "爵士", "WAS": "奇才"}


def player_zh(game_id: str, person_id: int | str | None, timeout: float = 15.0) -> dict | None:
    """{"name_zh": "贾巴里·史密斯", "name_zh_short": "史密斯"} or None."""
    if not person_id:
        return None
    cached = CACHE / f"{person_id}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    try:
        r = requests.get(URL, params={"gameId": game_id, "playerId": person_id}, timeout=timeout)
        r.raise_for_status()
        d = (r.json() or {}).get("data") or {}
    except (requests.RequestException, ValueError) as e:
        logger.warning("nba.cn name %s: %s", person_id, type(e).__name__)
        return None
    first, last = (d.get("playerFirstName") or "").strip(), (d.get("playerLastName") or "").strip()
    if not last:
        return None
    out = {"name_zh": f"{first}·{last}" if first else last, "name_zh_short": last}
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out, ensure_ascii=False))
    return out
