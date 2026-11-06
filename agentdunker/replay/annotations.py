"""Validate caller-supplied tactical annotations (renderer format, as in the hackathon play.json).

Input: {"annotations": [...], "clock": "replay" | "source_pts"} (or a bare list). "replay" times are seconds
from the replay window start (the play.json clock); "source_pts" times are converted with the window offset.
Entries with an unknown type, a bad time interval, or a reference to a player not in the replay are dropped
(with a reason); "ball" anchors are dropped too because the ball may be hidden.
"""
from __future__ import annotations

import math
from typing import Any

TYPES = {"arrow", "zone", "link", "focus", "label"}
MAX_LABEL = 40


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _interval(v: Any, shift: float) -> list[float] | None:
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(_num(x) for x in v)):
        return None
    a, b = float(v[0]) - shift, float(v[1]) - shift
    return [round(a, 3), round(b, 3)] if a < b else None


def _anchor(v: Any, ids: set[str]) -> str | None:
    """None if valid, else the reason."""
    if isinstance(v, str):
        return None if v in ids else f"unknown player {v!r}"
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(_num(x) for x in v):
        return None
    return f"bad anchor {v!r}"


def validate(data: Any, player_ids: set[str], duration: float, offset: float = 0.0) -> tuple[list[dict], list[str]]:
    """Return (kept annotations on the replay clock, dropped reasons)."""
    clock = "replay"
    if isinstance(data, dict):
        clock = data.get("clock", "replay")
        data = data.get("annotations", [])
    if not isinstance(data, list):
        return [], ["annotations must be a list"]
    shift = offset if clock == "source_pts" else 0.0
    kept, dropped = [], []
    for i, a in enumerate(data):
        why = _check(a, player_ids, duration, shift)
        if isinstance(why, str):
            dropped.append(f"#{i}: {why}")
        else:
            kept.append(why)
    return kept, dropped


def _check(a: Any, ids: set[str], duration: float, shift: float):
    if not isinstance(a, dict) or a.get("type") not in TYPES:
        return f"unknown type {a.get('type') if isinstance(a, dict) else a!r}"
    t = _interval(a.get("t"), shift)
    if t is None:
        return "t must be [t0, t1] with t0 < t1"
    t = [max(0.0, t[0]), min(duration, t[1])]
    if t[1] - t[0] < 0.05:
        return "outside the replay window"
    out = {k: v for k, v in a.items() if k in {"type", "color", "dashed", "closed", "shape", "preset"}}
    out["t"] = t
    for k in ("width", "height", "y", "bend", "r"):
        if k in a:
            if not _num(a[k]):
                return f"{k} must be a number"
            out[k] = float(a[k])
    for k in ("label", "text"):
        if k in a:
            if not isinstance(a[k], str):
                return f"{k} must be a string"
            out[k] = a[k][:MAX_LABEL]
    typ = a["type"]
    if typ == "arrow":
        if isinstance(a.get("path"), dict):
            p = a["path"]
            pt = _interval(p.get("t"), shift)
            if pt is None or not isinstance(p.get("who"), str):
                return "arrow path needs who and t"
            if (why := _anchor(p["who"], ids)):
                return why
            out["path"] = {"who": p["who"], "t": [max(0.0, pt[0]), min(duration, pt[1])]}
        else:
            for k in ("from", "to"):
                if (why := _anchor(a.get(k), ids)):
                    return f"arrow {k}: {why}"
                out[k] = a[k]
    elif typ == "zone":
        if a.get("shape") == "poly":
            pts = a.get("points")
            if not (isinstance(pts, list) and len(pts) >= 3 and all(_anchor(p, set()) is None for p in pts)):
                return "poly zone needs >= 3 [x, z] points"
            out["points"] = pts
        elif a.get("preset") == "paint":
            pass
        else:
            if (why := _anchor(a.get("at"), ids)):
                return f"zone at: {why}"
            if not _num(a.get("r")) or a["r"] <= 0:
                return "circle zone needs r > 0"
            out["at"] = a["at"]
    elif typ == "link":
        g = a.get("ids")
        if not (isinstance(g, list) and len(g) >= 2):
            return "link needs >= 2 ids"
        if (why := next((w for w in (_anchor(x, ids) for x in g) if w), None)):
            return why
        out["ids"] = g
    elif typ == "focus":
        if not isinstance(a.get("who"), str):
            return "focus needs who"
        if (why := _anchor(a["who"], ids)):
            return why
        out["who"] = a["who"]
    elif typ == "label":
        if (why := _anchor(a.get("at"), ids)):
            return f"label at: {why}"
        if not out.get("text"):
            return "label needs text"
        out["at"] = a["at"]
    return out
