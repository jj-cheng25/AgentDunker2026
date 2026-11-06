"""Commentary claim check: every action / tactical word must be backed by a measurement near the line.

A line's film time is mapped back to source time through the EDL; a word is supported only by a fact
(``facts.py``) or a confident M5 event that overlaps that source interval (± 1 s), or by the shot itself.
Words nothing in the chain can measure (虚晃, 牵动防守, 对抗 …) are never allowed.

When only part of a line is unsupported, the offending clause is cut out of the take: the take's internal
pauses must line up with the clause commas, and MOSS ASR must read back the shortened text. Otherwise the
whole line is dropped (audio and subtitle).
"""
from __future__ import annotations

import io
import logging
import re
from typing import Optional

import numpy as np

from . import models
from .media import SR

logger = logging.getLogger(__name__)

NEVER = ("虚晃", "假动作", "变向", "背身", "转身", "后撤步", "欧洲步", "胯下", "背后运球", "急停", "扣篮", "补扣",
         "牵动", "吸引", "注意力", "迎面", "对抗", "身体接触", "晃开", "甩开", "错位", "硬吃", "失位", "撕开",
         "拉扯", "牵制", "清空", "搅乱", "带走", "调动")    # effects on the defense: not measured
NEEDS = {
    ("空切", "切入", "切走", "内切", "再切", "溜底"): {"cut"},
    ("掩护", "挡拆", "顺下"): {"screen"},
    ("扑防", "扑出", "扑上"): {"closeout"},
    ("协防", "包夹", "补位", "收缩"): {"help"},
    ("突破", "硬突", "强突", "杀入", "强杀", "杀向", "冲击", "启动", "推进"): {"drive"},
    ("篮下", "禁区"): {"drive", "cut"},
    ("回传", "传球", "分球", "传给", "策应", "转移", "回敲", "回球"): {"pass"},
    ("拉开", "空间", "两侧", "单打"): {"space"},
}
TOL = 1.0
CLAUSE = re.compile(r"[^，,！!。；;？?]+[，,！!。；;？?]*")


def unsupported(text: str, support: set[str]) -> list[str]:
    bad = [w for w in NEVER if w in text]
    for words, need in NEEDS.items():
        bad += [w for w in words if w in text and not (need & support)]
    return bad


def source_interval(edl: dict, film_a: float, film_b: float) -> Optional[tuple[float, float]]:
    """Source-time span shown while the line plays (None for intro/bridge/outro)."""
    from .director import map_fwd
    ts = []
    for ft in np.linspace(film_a, film_b, 5):
        for s in edl["segments"]:
            a, b = s["film"]
            if not a <= ft <= b:
                continue
            if s.get("time_map"):
                ts.append(map_fwd(s["time_map"], ft - a))
            elif s["kind"] == "live":
                ts.append(s["source"][0] + (ft - a) * s.get("speed", 1.0))
            elif s["kind"] == "freeze":
                ts.append(float(s["source_t"]))
            break
    if not ts:
        return None
    lo, hi = min(ts), max(ts)
    rep = next((s for s in edl["segments"] if s["kind"] == "replay"), None)
    if rep and rep.get("phases") and rep["film"][0] <= film_a < rep["film"][1]:
        # A 3D line narrates its phase (M6 got that phase's facts): its support is the whole phase.
        u, at = film_a - rep["film"][0], 0.0
        for ph in rep["phases"]:
            if at <= u < at + ph["film_len"]:
                lo, hi = min(lo, ph["src"][0]), max(hi, ph["src"][1])
                break
            at += ph["film_len"]
    return lo, hi


def support_for(edl: dict, events: Optional[dict], span: Optional[tuple[float, float]]) -> set[str]:
    if span is None:
        return set()
    a, b = span[0] - TOL, span[1] + TOL
    out = {f["type"] for f in edl["decisions"].get("facts") or [] if f["t0"] <= b and f["t1"] >= a}
    out |= {e["type"] for e in (events or {}).get("events", [])
            if a <= e["t"] <= b and float(e.get("conf") or 0) >= 0.35}
    shot = edl.get("shot") or {}
    if shot.get("release_t") is not None and a <= shot["release_t"] <= b + 2.0:
        out.add("shot")
    return out


def _cer(a: str, b: str) -> float:
    a, b = re.sub(r"[^一-鿿0-9A-Za-z]", "", a), re.sub(r"[^一-鿿0-9A-Za-z]", "", b)
    if not a:
        return 1.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1] / len(a)


def _pauses(piece: np.ndarray) -> list[tuple[float, float]]:
    """Silent gaps (≥ 80 ms, 35 dB under the peak) strictly inside the speech, as (start_s, end_s)."""
    mono = np.abs(piece).mean(axis=1) if piece.ndim == 2 else np.abs(piece)
    hop = int(0.01 * SR)
    n = len(mono) // hop
    if n < 10:
        return []
    rms = np.sqrt((mono[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    loud = rms > max(rms.max() * 10 ** (-35 / 20), 1e-4)
    idx = np.flatnonzero(loud)
    if len(idx) < 2:
        return []
    gaps, i = [], idx[0]
    while i < idx[-1]:
        if loud[i]:
            i += 1
            continue
        j = i
        while j < idx[-1] and not loud[j]:
            j += 1
        if j - i >= 8:
            gaps.append((i * 0.01, j * 0.01))
        i = j
    return gaps


def _wav_bytes(x: np.ndarray) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1).astype(np.float32), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def trim(piece: np.ndarray, text: str, keep: list[bool], clauses: list[str]) -> Optional[tuple[np.ndarray, str]]:
    """Cut the unsupported clauses out of a take; None when pauses do not line up or ASR disagrees."""
    gaps = _pauses(piece)
    k = len(clauses) - 1
    if len(gaps) < k:
        return None
    cuts = sorted(sorted(gaps, key=lambda g: g[0] - g[1])[:k])          # the k longest pauses, in time order
    bounds = [0] + [int((g0 + g1) / 2 * SR) for g0, g1 in cuts] + [len(piece)]
    gap = np.zeros((int(0.15 * SR),) + piece.shape[1:], piece.dtype)
    parts, prev = [], None
    for i, ok in enumerate(keep):
        if not ok:
            continue
        if prev is not None and prev != i - 1:
            parts.append(gap)
        parts.append(piece[bounds[i]:bounds[i + 1]])
        prev = i
    new = np.concatenate(parts)
    kept = [c for c, ok in zip(clauses, keep) if ok]
    new_text = "".join(kept)
    end_mark = "！" if text.rstrip()[-1:] in "！!" else "。"
    new_text = re.sub(r"[，,；;]$", "", new_text.rstrip()) + ("" if new_text.rstrip()[-1:] in "！!。？?" else end_mark)
    heard = models.moss_asr(_wav_bytes(new), "claim-trim")
    if heard is None or _cer(new_text, heard) > 0.2:
        logger.info("clause trim rejected: want %s heard %s", new_text, heard)
        return None
    return new, new_text


def check(edl: dict, events: Optional[dict], pieces: list, lines: list[dict]
          ) -> tuple[list, list[dict], list[dict], list[dict]]:
    """→ (kept pieces, kept lines, dropped [{text, unsupported}], trimmed [{from, to, unsupported}])."""
    from .director import INTRO_S
    by_line = {id(p[0]): p for p in pieces}
    kept_p, kept_l, dropped, trimmed = [], [], [], []
    for l in lines:
        span = source_interval(edl, INTRO_S + l["start"], INTRO_S + l["end"])
        sup = support_for(edl, events, span)
        bad = unsupported(l["text"], sup)
        p = by_line.get(id(l))
        if not bad:
            kept_l.append(l)
            if p:
                kept_p.append(p)
            continue
        clauses = CLAUSE.findall(l["text"]) or [l["text"]]
        keep = [not unsupported(c, sup) for c in clauses]
        res = trim(p[1], l["text"], keep, clauses) if p is not None and any(keep) and len(clauses) > 1 else None
        if res is None:
            dropped.append({"text": l["text"], "unsupported": bad, "source_span": span})
            continue
        new, new_text = res
        trimmed.append({"from": l["text"], "to": new_text, "unsupported": bad})
        l["end"] = round(l["start"] + (l["end"] - l["start"]) * len(new) / max(1, len(p[1])), 3)
        l["text"] = new_text
        kept_l.append(l)
        kept_p.append((l, new, p[2]))
    return kept_p, kept_l, dropped, trimmed
