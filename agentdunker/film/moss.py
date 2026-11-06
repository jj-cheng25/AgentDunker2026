"""MOSS-VL used only for small multiple-choice checks (MOSS_VL_TASK_FIT: it reads and chooses well, it
invents when asked to describe). Each answer must be a single option letter; anything else is ignored and
the deterministic rule stands.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from . import models
from .media import frame_at

logger = logging.getLogger(__name__)

SHOT_PROMPT = ("这是篮球比赛转播里的一帧画面。它属于哪一类镜头？只回答一个字母，不要解释。\n"
               "A. 比赛全景：远景或中景机位拍球场，能看到多名球员在场上\n"
               "B. 近景特写：画面主要是一两名球员的近景\n"
               "C. 回放：带回放标识或慢动作特效的重播画面\n"
               "D. 观众或替补席\n"
               "E. 其他（广告、图文、过场）")
SHOT_KIND = {"A": "wide", "B": "close", "C": "replay", "D": "crowd", "E": "other"}


def ask_shot_type(img) -> Optional[str]:
    text = models.moss_vl(SHOT_PROMPT, [img], "shot-type")
    if text is None:
        return None
    m = re.match(r"\s*[（(]?([A-E])\b", text.strip().upper())
    return SHOT_KIND.get(m.group(1)) if m else None


def classify_shots(video: str, shots: list[dict], fps: float) -> list[dict]:
    """MOSS-VL as a veto: a rule-``wide`` shot stays wide unless MOSS-VL calls its middle frame something else.

    Measured on 53 hand-labelled shots (evidence/m7/shot_eval): person-box rule 47/53 (5 close-ups called
    wide), MOSS-VL alone 49/53, rule AND MOSS-VL 51/53. MOSS-VL never turns a close shot into a wide one.
    """
    out = []
    for s in shots:
        s = dict(s)
        if s.get("kind") == "wide" and s["t1"] - s["t0"] >= 0.4:
            kind = ask_shot_type(frame_at(video, (s["start_f"] + s["end_f"]) // 2))
            s["moss_vl"] = kind
            if kind is not None and kind != "wide":
                s["kind"], s["source"] = "close", "rule+moss-vl veto"
        out.append(s)
    return out


CARD_PROMPT = ("图中是一张篮球数据卡片。请读出卡片上的文字，按这个格式回答，看不清的项写“无”：\n"
               "名字：…\n数字：…")


def readback_card(film: str, t: float, rect: list[int], expect_name: str, expect_xfg: Optional[float]) -> dict:
    """Read the rendered xFG card back from the film and compare with what the EDL asked for."""
    from .media import probe as _probe
    info = _probe(film)
    img = frame_at(film, round(t * info.fps))
    x0, y0, x1, y1 = rect
    crop = img[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10]
    text = models.moss_vl(CARD_PROMPT, [crop], "readback-card")
    if text is None:
        return {"checked": False}
    # Format-free comparison: the model reads well but does not always follow the answer template.
    flat = re.sub(r"\s", "", text)
    nums = re.findall(r"\d+(?:\.\d+)?", flat)
    want_num = f"{expect_xfg * 100:.1f}" if expect_xfg is not None else ""
    return {"checked": True, "heard": text.strip()[:120],
            "name_ok": (not expect_name) or expect_name in flat,
            "number_ok": (not want_num) or want_num in nums}
