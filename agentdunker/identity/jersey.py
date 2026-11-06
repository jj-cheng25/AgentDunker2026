"""Read jersey numbers per track: one cheap VLM call per track (per-crop answers, "unreadable" allowed),
vote across crops, and send unclear tracks to the strong model for a second read."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import vlm
from .crops import Crop, sheet

logger = logging.getLogger(__name__)

PROMPT = """The numbered tiles are crops of ONE tracked basketball player from an NBA broadcast, taken at
different moments. In every tile the tracked player is the person inside the brighter (not dimmed) rectangle;
other people may also be visible - ignore them.
For EACH tile on its own, read the jersey number printed on the tracked player's jersey (chest, back or shorts).
Give a number only if every digit is clearly legible in that tile. If the number is partly hidden, folded,
blurred, too small, seen from the side, or you would have to guess any digit, answer null for that tile.
Do not copy a number from another tile. NBA numbers are 0-99 and may be "0" or "00".
Also say whether the tracked person looks like a referee instead of a player."""

SCHEMA = {
    "type": "object",
    "properties": {
        "tiles": {"type": "array", "items": {"type": "object", "properties": {
            "tile": {"type": "integer"},
            "number": {"type": ["string", "null"]},
            "view": {"type": "string", "enum": ["front", "back", "side", "unclear"]}},
            "required": ["tile", "number", "view"]}},
        "is_referee": {"type": "boolean"},
    },
    "required": ["tiles", "is_referee"],
}

ACCEPT_SHARE = 0.75
STRONG_BELOW = 0.95       # strong-model second read unless ≥ 5 crops agree with no conflict


@dataclass
class Reading:
    track_id: int
    reads: dict = field(default_factory=dict)       # model -> {tile: number}
    views: dict = field(default_factory=dict)       # tile -> view (cheap model)
    referee_votes: int = 0
    frames: list = field(default_factory=list)      # frame index per tile (1-based tiles)

    def votes(self) -> Counter:
        """Votes per crop (tile): a number read on a tile by every model that read it there; "?" when the
        models disagree on that tile. One crop is one vote however many models read it."""
        per = defaultdict(set)
        for r in self.reads.values():
            for t, n in r.items():
                per[t].add(n)
        return Counter(ns.pop() if len(ns) == 1 else "?" for ns in per.values())

    def best(self) -> tuple[str | None, float, int, int]:
        """(number, confidence, supporting crops, crops with any reading); number None unless the vote is clear."""
        v = self.votes()
        total = sum(v.values())
        nums = Counter({n: c for n, c in v.items() if n != "?"})
        if not nums:
            return None, 0.0, 0, total
        num, c = nums.most_common(1)[0]
        share = c / total
        conf = share * (1 - 0.5 ** c)
        if c >= 2 and share >= ACCEPT_SHARE:
            return num, round(conf, 3), c, total
        return None, round(conf, 3), c, total

    def support(self, allowed: set[str] | None = None) -> tuple[dict[str, tuple[int, int]], int]:
        """({number: (weight, crops)}, crops with any reading). A crop on which two reads agree weighs 3 (a
        single model also claims numbers it cannot see), a crop read once weighs 1; crops where the reads
        disagree only add to the total.
        With `allowed` (the on-court jerseys), readings outside it are ignored as known misreads."""
        per = defaultdict(list)
        for r in self.reads.values():
            for t, n in r.items():
                if allowed is None or n in allowed:
                    per[t].append(n)
        sup: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for ns in per.values():
            if len(set(ns)) == 1:
                sup[ns[0]][0] += 3 if len(ns) > 1 else 1
                sup[ns[0]][1] += 1
        return {n: (w, k) for n, (w, k) in sup.items()}, len(per)

    def evidence_frames(self, number: str | None) -> list[int]:
        return sorted({self.frames[t - 1] for per in self.reads.values() for t, n in per.items()
                       if n == number and 0 < t <= len(self.frames)})


def _norm(n) -> str | None:
    if n is None:
        return None
    s = str(n).strip().lstrip("#").strip()
    return s if s.isdigit() and len(s) <= 2 else None


def _read(track_id: int, crops: list[Crop], model: str) -> dict | None:
    img = sheet([c.img for c in crops])
    ans = vlm.ask([img], PROMPT + f"\nTiles: 1..{len(crops)}.", SCHEMA, model=model, name="jersey_read",
                  tag=f"jersey:{track_id}")
    if ans is None:
        return None
    nums, views = {}, {}
    for t in ans.get("tiles", []):
        try:
            k = int(t.get("tile"))
        except (TypeError, ValueError):
            continue
        if not 1 <= k <= len(crops):
            continue
        views[k] = t.get("view")
        n = _norm(t.get("number"))
        if n is not None:
            nums[k] = n
    return {"numbers": nums, "views": views, "referee": bool(ans.get("is_referee"))}


def read_tracks(crops: dict[int, list[Crop]], max_strong: int = 20, workers: int = 8) -> dict[int, Reading]:
    """Cheap read of every track, then a strong-model read of the tracks without a clear vote."""
    readings = {tid: Reading(tid, frames=[c.f for c in cs]) for tid, cs in crops.items()}
    with ThreadPoolExecutor(workers) as ex:
        res = dict(zip(crops, ex.map(lambda tid: _read(tid, crops[tid], vlm.CHEAP), crops)))
    for tid, r in res.items():
        if r is not None:
            readings[tid].reads[vlm.CHEAP] = r["numbers"]
            readings[tid].views = r["views"]
            readings[tid].referee_votes += r["referee"]
    unclear = sorted((tid for tid, rd in readings.items() if rd.best()[1] < STRONG_BELOW),
                     key=lambda tid: readings[tid].best()[1])[:max_strong]
    if unclear:
        with ThreadPoolExecutor(workers) as ex:
            res = dict(zip(unclear, ex.map(lambda tid: _read(tid, crops[tid], vlm.STRONG), unclear)))
        for tid, r in res.items():
            if r is not None:
                readings[tid].reads[vlm.STRONG] = r["numbers"]
                readings[tid].referee_votes += r["referee"]
    for tid, rd in readings.items():
        logger.info("jersey track %s: %s -> %s", tid, dict(rd.reads), rd.best())
    return readings


CHOICE_PROMPT = """The numbered tiles are crops of ONE tracked basketball player from an NBA broadcast; the tracked
player is the person inside the brighter (not dimmed) rectangle. According to the official lineup this player
wears one of these numbers: {choices}.
For EACH tile on its own, answer which of these numbers is printed on the tracked player's jersey (chest, back
or shorts), or null if the number is not visible clearly enough in that tile to tell. Do not guess."""


def reread_choices(crops: dict[int, list[Crop]], choices: dict[int, list[str]], budget: int,
                   workers: int = 8) -> dict[int, dict[int, str]]:
    """Strong-model re-read of tracks whose number must be one of `choices[tid]` (free on-court jerseys)."""
    tids = [tid for tid in choices if tid in crops and choices[tid]][:max(0, budget)]

    def one(tid: int) -> dict[int, str] | None:
        opts = choices[tid]
        schema = {"type": "object", "properties": {"tiles": {"type": "array", "items": {"type": "object", "properties": {
            "tile": {"type": "integer"}, "number": {"type": ["string", "null"], "enum": opts + [None]}},
            "required": ["tile", "number"]}}}, "required": ["tiles"]}
        ans = vlm.ask([sheet([c.img for c in crops[tid]])],
                      CHOICE_PROMPT.format(choices=", ".join(opts)) + f"\nTiles: 1..{len(crops[tid])}.",
                      schema, model=vlm.STRONG, name="jersey_choice", tag=f"choice:{tid}")
        if ans is None:
            return None
        out = {}
        for t in ans.get("tiles", []):
            n = _norm(t.get("number"))
            if n in opts and isinstance(t.get("tile"), int) and 1 <= t["tile"] <= len(crops[tid]):
                out[t["tile"]] = n
        return out

    with ThreadPoolExecutor(workers) as ex:
        res = dict(zip(tids, ex.map(one, tids)))
    logger.info("jersey choice re-read: %s", res)
    return {tid: r for tid, r in res.items() if r is not None}
