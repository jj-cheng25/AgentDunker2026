"""Made / missed from the ball's passage through the rim, with a narrow-window VLM check when unclear."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .balltrack import BallObs, Rim, rim_at
from . import vlm

logger = logging.getLogger(__name__)

PROMPT = """These frames (in time order, labelled with frame numbers) are crops around the basket during one
NBA shot attempt. Did the ball go down through the hoop and net (made basket), or not (missed: off the rim,
off the backboard, air ball, blocked)? Judge only from what is visible. If the frames do not show it, say
"unclear"."""
SCHEMA = {"type": "object",
          "properties": {"result": {"type": "string", "enum": ["made", "missed", "unclear"]},
                         "evidence_frames": {"type": "array", "items": {"type": "integer"}},
                         "reason": {"type": "string"}},
          "required": ["result", "reason"]}


def rel_points(ball: dict[int, BallObs], rim_by_f: dict[int, Rim], f_from: float, f_to: float,
               fps: float) -> list[tuple[int, float, float]]:
    """(f, x, y) of the ball relative to the rim centre in rim widths (+y below the rim)."""
    pts = []
    for f in sorted(ball):
        if not f_from <= f <= f_to:
            continue
        r = rim_at(rim_by_f, f, int(fps * 0.3))
        if r is None or r.width < 8:
            continue
        b = ball[f]
        pts.append((f, (b.xy[0] - r.center[0]) / r.width, (b.xy[1] - r.center[1]) / r.width))
    return pts


def judge(ball: dict[int, BallObs], rim_by_f: dict[int, Rim], f_from: int, f_to: int, fps: float) -> dict:
    """Rule on the dense ball track: where (if at all) does the ball cross the rim plane going down, and does it
    keep falling under the rim (through the net)?"""
    pts = rel_points(ball, rim_by_f, f_from, f_to, fps)
    crossings = []
    for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
        if y0 < 0 <= y1 and (f1 - f0) <= 0.25 * fps:
            a = -y0 / (y1 - y0) if y1 != y0 else 0.0
            crossings.append((f0 + a * (f1 - f0), x0 + a * (x1 - x0)))
    out = {"crossings": [(round(f, 1), round(x, 2)) for f, x in crossings], "n_points": len(pts)}
    out["f_bank"] = bank_frame(pts, crossings[-1][0] if crossings else None, fps)
    # the last crossing decides: a rattle crosses, bounces up and crosses again
    if crossings:
        f_c, x_c = crossings[-1]
        out["f_cross"] = round(f_c, 1)
        below = [(x, y) for f, x, y in pts if f_c < f <= f_c + 0.6 * fps]
        through = [y for x, y in below if abs(x) < 0.7 and y > 0.6]
        bounced = [y for x, y in below if y < -0.3]
        if abs(x_c) <= 0.4 and through and not bounced:
            out.update(made=True, conf=0.9, rule="last crossing inside the rim, ball fell through")
        elif abs(x_c) >= 0.65:
            out.update(made=False, conf=0.8, rule="ball came down outside the rim")
        else:
            out.update(made=None, conf=0.0, rule="crossing near the rim edge or no clear fall-through")
    elif pts and min(y for _, _, y in pts) < -0.3 and len(pts) >= 6:
        out.update(made=None, conf=0.0, rule="no downward crossing seen")
    else:
        out.update(made=None, conf=0.0, rule="ball path at the rim unclear")
    return out


def bank_frame(pts: list[tuple[int, float, float]], f_cross: float | None, fps: float) -> int | None:
    """Backboard contact: before the ball first comes down to the rim, its horizontal motion reverses sharply
    while it is in the backboard area (above the rim, within ~1.3 rim widths sideways). Gravity alone never
    flips x, and later reversals are rim bounces (rattles), so only the first arrival is considered."""
    if f_cross is None:
        return None
    first_rim, was_above = f_cross, False
    for f, x, y in pts:
        was_above = was_above or y < -0.5
        if was_above and abs(x) < 0.9 and y > -0.35:
            first_rim = f
            break
    seq = [(f, x, y) for f, x, y in pts if f <= first_rim]
    win = 0.1 * fps
    best = None
    for f, x, y in seq:
        if not (-2.6 < y < -0.3 and abs(x) < 1.3):
            continue
        before = [(g, u, v) for g, u, v in seq if f - 2.5 * win <= g < f]
        after = [(g, u, v) for g, u, v in seq if f < g <= f + 2.5 * win]
        if len(before) < 2 or len(after) < 2:
            continue
        vx1 = (x - before[0][1]) / (f - before[0][0])
        vx2 = (after[-1][1] - x) / (after[-1][0] - f)
        vy1 = (y - before[0][2]) / (f - before[0][0])
        vy2 = (after[-1][2] - y) / (after[-1][0] - f)
        if np.sign(vx1) == np.sign(vx2) or min(abs(vx1), abs(vx2)) * fps < 0.8:
            continue  # need a clear reversal: > 0.8 rim widths per second on both sides
        strength = abs(vx1 - vx2) + 0.5 * abs(vy1 - vy2)
        if best is None or strength > best[1]:
            best = (f, strength)
    return int(best[0]) if best else None


def vlm_check(video: str, rim_by_f: dict[int, Rim], f_from: int, f_to: int, fps: float, n: int = 12) -> dict | None:
    """Ask the VLM on ~12 rim crops spread over [f_from, f_to]."""
    frames = np.linspace(f_from, f_to, n).round().astype(int).tolist()
    cap = cv2.VideoCapture(video)
    crops, labels = [], []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        r = rim_at(rim_by_f, f, int(fps * 0.5))
        if not ok or r is None:
            continue
        cx, cy, w = r.center[0], r.center[1], max(r.width, 40)
        x1, x2 = int(max(0, cx - 3.5 * w)), int(min(im.shape[1], cx + 3.5 * w))
        y1, y2 = int(max(0, cy - 3.0 * w)), int(min(im.shape[0], cy + 4.0 * w))
        crops.append(im[y1:y2, x1:x2])
        labels.append(str(f))
    cap.release()
    if len(crops) < 4:
        return None
    ans = vlm.ask_json(vlm.strip(crops, labels), PROMPT, SCHEMA)
    if not ans or ans.get("result") not in ("made", "missed"):
        return ans
    ans["made"] = ans["result"] == "made"
    return ans
