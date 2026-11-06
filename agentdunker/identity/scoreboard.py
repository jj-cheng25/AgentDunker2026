"""Read the broadcast score bug (teams, scores, period, game clock) on a few full frames.

One gemini call on all frames reads teams (text or logo), digits and banner. ADK_ID_SCOREBOARD_DIGITS=moss adds
MOSS-VL digit reads (one call per frame on the bottom 35 %, gemini as the fallback); it is off by default because
on the 14-clip bench MOSS-VL read 29/42 frames fully right against gemini's 42/42 (evidence/m3/scoreboard_h2h).
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import vlm
from .games import secs

logger = logging.getLogger(__name__)

TRICODES = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL",
            "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
            "UTA", "WAS"}
ALIASES = {"BRK": "BKN", "BKLYN": "BKN", "CHO": "CHA", "GS": "GSW", "GOS": "GSW", "NO": "NOP", "NOR": "NOP",
           "NY": "NYK", "PHO": "PHX", "SA": "SAS", "SAN": "SAS", "UTAH": "UTA", "UTH": "UTA", "WSH": "WAS",
           "WAS": "WAS", "LA": None}

PROMPT = """Each image is a full frame from the same NBA broadcast clip (image 1, 2, ... in order).
For each image, read the on-screen score bug (the score graphic, usually at the bottom or top).
For both teams in display order give: the team abbreviation text exactly as shown (null if only a logo),
the official NBA 3-letter tricode of that team (from the text, or from the logo if no text; null if unsure),
and the score. Also give the period (1-4, 5 for OT1, 6 for OT2...), the game clock exactly as shown
("6:35" or "45.3"), and any banner text in or next to the score bug (e.g. "PLAY-IN", "WEST FINALS GAME 6").
If there is no score bug in an image, set visible=false and leave the fields null. Never guess digits."""

TEAM = {"type": "object", "properties": {
    "text": {"type": ["string", "null"]}, "tricode": {"type": ["string", "null"]}, "score": {"type": ["integer", "null"]}},
    "required": ["text", "tricode", "score"]}
SCHEMA = {
    "type": "object",
    "properties": {"images": {"type": "array", "items": {"type": "object", "properties": {
        "image": {"type": "integer"}, "visible": {"type": "boolean"},
        "team1": TEAM, "team2": TEAM,
        "period": {"type": ["integer", "null"]}, "game_clock": {"type": ["string", "null"]},
        "banner": {"type": ["string", "null"]}},
        "required": ["image", "visible", "team1", "team2", "period", "game_clock", "banner"]}}},
    "required": ["images"],
}


# validated MOSS-VL protocol (agentdunker-robust-e2e courtvision.moss_numeric_qa, 28/28 digits); kept verbatim
MOSS_PROMPT = ('只读图中的电视转播比分牌，不把球衣、广告或现场计时器当比分牌。按屏幕从左到右抄两个比分'
               '（若竖排则从上到下），节次填整数。原样抄比赛时钟，保留小数秒。'
               '不输出队名、球员、比赛身份、进攻时钟或其他数字。看不清的数字填null、时钟填空串。'
               '没有可确认的电视比分牌时found=false、全部数字null、时钟空串，禁止推测。'
               '只输出一个JSON对象：{"found":布尔值,"leftScore":整数或null,"rightScore":整数或null,'
               '"period":整数或null,"gameClock":字符串}。')
DIGITS = os.environ.get("ADK_ID_SCOREBOARD_DIGITS", "gemini")


def norm_tricode(text: str | None, tricode: str | None) -> str | None:
    for v in (tricode, text):
        if not v:
            continue
        v = re.sub(r"[^A-Z]", "", str(v).upper())
        v = ALIASES.get(v, v)
        if v in TRICODES:
            return v
    return None


def grab(video: str, frames: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    out = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            out.append(img)
    cap.release()
    return out


def moss_digits(img: np.ndarray) -> dict | None:
    """MOSS-VL on the bottom 35 % of one frame: {scores: [left, right], period, clock, clock_s} or None."""
    h = img.shape[0]
    a = vlm.ask_moss(img[int(h * 0.65):], MOSS_PROMPT, tag="scoreboard-digits")
    if not a or not a.get("found"):
        return None
    sc = [a.get("leftScore"), a.get("rightScore")]
    clock = a.get("gameClock") if isinstance(a.get("gameClock"), str) else None
    per = a.get("period")
    if not all(isinstance(x, int) and 0 <= x < 250 for x in sc) or secs(clock) is None or secs(clock) > 720:
        return None
    return {"scores": sc, "period": per if isinstance(per, int) and 1 <= per <= 10 else None,
            "clock": clock.strip(), "clock_s": secs(clock)}


def _merge(r: dict, m: dict | None) -> dict:
    """Gemini reading r with MOSS-VL digits m (display order) where MOSS-VL read the score bug."""
    if not m or None in r["teams"]:
        return {**r, "digits": "gemini"}
    sc = m["scores"]
    g = [r["scores"].get(tri) for tri in r["teams"]]
    if g == sc[::-1] and sc[0] != sc[1]:
        sc = sc[::-1]                       # the two models disagree only on display order: keep gemini's
    return {**r, "scores": dict(zip(r["teams"], sc)), "period": m["period"] or r["period"], "clock": m["clock"],
            "clock_s": m["clock_s"], "digits": "moss-vl",
            "gemini": {"scores": r["scores"], "period": r["period"], "clock": r["clock"]}}


def read_scoreboard(video: str, n_frames: int, fps: float, frames: list[int] | None = None,
                    digits: str | None = None) -> list[dict]:
    """[{f, t, teams: [tri, tri], scores: {tri: n}, period, clock, clock_s, banner, digits}] for frames with a
    score bug. `digits`: "gemini" (default) or "moss"."""
    frames = frames or [int(n_frames * q) for q in (0.08, 0.5, 0.92)]
    imgs = grab(video, frames)
    if not imgs:
        return []
    with ThreadPoolExecutor(len(imgs) + 1) as ex:
        f_g = ex.submit(vlm.ask, imgs, PROMPT, SCHEMA, model=vlm.CHEAP, name="scorebug", tag="scoreboard")
        f_m = [ex.submit(moss_digits, im) for im in imgs] if (digits or DIGITS) == "moss" else []
        ans = f_g.result()
        moss = [f.result() for f in f_m]
    out = []
    for r in (ans or {}).get("images", []):
        i = r.get("image", 0) - 1
        if not r.get("visible") or not 0 <= i < len(frames):
            continue
        teams = [norm_tricode(r[k].get("text"), r[k].get("tricode")) for k in ("team1", "team2")]
        scores = {tri: r[k].get("score") for tri, k in zip(teams, ("team1", "team2"))
                  if tri and isinstance(r[k].get("score"), int)}
        out.append({"f": frames[i], "t": round(frames[i] / fps, 3), "teams": teams, "scores": scores,
                    "text": [r[k].get("text") for k in ("team1", "team2")],
                    "period": r.get("period"), "clock": r.get("game_clock"),
                    "clock_s": secs(r.get("game_clock")), "banner": r.get("banner")})
        if moss:
            out[-1] = _merge(out[-1], moss[i])
    logger.info("scoreboard: %s", out)
    return out
