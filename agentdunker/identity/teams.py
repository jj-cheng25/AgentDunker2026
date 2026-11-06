"""Name the two jersey groups (M1 team A / B) as NBA teams: one VLM call on front-view crops of each group,
optionally constrained to the two tricodes read from the score bug."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from . import vlm
from .crops import Crop, sheet
from .scoreboard import TRICODES, norm_tricode

logger = logging.getLogger(__name__)

PROMPT = """Top row (tiles A1..): players of jersey group A. Bottom row (tiles B1..): players of jersey group B.
All are from one NBA game broadcast; the tracked player is the person inside the brighter rectangle.
Which NBA team does each group play for? Use the team name or city wordmark on the jersey first, then the
team's known colour schemes (home, away, City, Statement and Classic editions).
{hint}Answer with official 3-letter NBA tricodes, null if you cannot tell. The two groups are different teams."""

SCHEMA = {
    "type": "object",
    "properties": {
        "group_A": {"type": ["string", "null"]}, "group_B": {"type": ["string", "null"]},
        "wordmark_A": {"type": ["string", "null"]}, "wordmark_B": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["group_A", "group_B", "wordmark_A", "wordmark_B", "confidence"],
}


def _pick(team_tracks: list[int], crops: dict[int, list[Crop]], views: dict[int, dict], n: int = 6) -> list[np.ndarray]:
    """Up to n crops for one group, front views first, at most two per track."""
    scored = []
    for tid in team_tracks:
        for k, c in enumerate(crops.get(tid, []), 1):
            front = views.get(tid, {}).get(k) == "front"
            scored.append((front, c.score * (c.bbox[3] - c.bbox[1]), tid, c))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out, per = [], {}
    for _, _, tid, c in scored:
        if per.get(tid, 0) < 2:
            out.append(c.img)
            per[tid] = per.get(tid, 0) + 1
        if len(out) == n:
            break
    return out


def name_teams(groups: dict[str, list[int]], crops: dict[int, list[Crop]], views: dict[int, dict],
               candidates: list[str] | None) -> dict | None:
    """{A: tri|None, B: tri|None, confidence, wordmarks} or None when the call fails."""
    rows = {}
    for g in ("A", "B"):
        imgs = _pick(groups.get(g, []), crops, views)
        if not imgs:
            return None
        rows[g] = sheet(imgs, tile_h=300, cols=len(imgs), labels=[f"{g}{i + 1}" for i in range(len(imgs))])
    w = max(r.shape[1] for r in rows.values())
    img = np.vstack([cv2.copyMakeBorder(r, 0, 12, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                     for r in rows.values()])
    cands = [c for c in (candidates or []) if c in TRICODES]
    hint = (f"The score bug of this broadcast shows {cands[0]} and {cands[1]}, so the answer is one of them each.\n"
            if len(cands) == 2 else "")
    ans = vlm.ask([img], PROMPT.format(hint=hint), SCHEMA, model=vlm.CHEAP, name="team_naming", tag="teams")
    if ans is None:
        return None
    out = {"A": norm_tricode(None, ans.get("group_A")), "B": norm_tricode(None, ans.get("group_B")),
           "confidence": ans.get("confidence"), "wordmarks": [ans.get("wordmark_A"), ans.get("wordmark_B")]}
    if out["A"] == out["B"]:
        out["A"] = out["B"] = None
    logger.info("teams: %s", out)
    return out
