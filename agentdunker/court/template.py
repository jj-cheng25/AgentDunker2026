"""NBA court template in feet.

Court frame (shared contract): origin at centre court, x along the long side in [-47, 47],
y across in [-25, 25].  +y is the near sideline (the side of the main broadcast camera), so a
top-down map drawn with x to the right and y downward matches the usual broadcast view.
"""
from __future__ import annotations

import numpy as np

HALF_LEN = 47.0
HALF_WID = 25.0
LANE_HALF = 8.0           # 16 ft lane (outer edges)
FT_X = 47.0 - 19.0        # free-throw line, 19 ft from baseline
RIM_X = 47.0 - 5.25       # rim centre floor projection
THREE_R = 23.75
CORNER_Y = 22.0           # corner three, 3 ft from sideline
CORNER_X = RIM_X - float(np.sqrt(THREE_R ** 2 - CORNER_Y ** 2))  # ~32.8: arc meets straight section
CIRCLE_R = 6.0
HASH_X = 47.0 - 28.0      # sideline hash mark, 28 ft from baseline

# court-33 (Roboflow basketball) keypoint order -> court coordinates; None = not a painted ground point.
KEYPOINT_LABELS = ["01", "02", "04", "05", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17",
                   "19", "21", "23", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "37",
                   "38", "40", "41"]


def _left_half() -> list:
    b = -HALF_LEN
    return [(b, -HALF_WID), (b, -CORNER_Y), (b, -LANE_HALF), (b, LANE_HALF), (b, CORNER_Y), (b, HALF_WID),
            None,  # 09 rim projection
            (-CORNER_X, -CORNER_Y), (-CORNER_X, CORNER_Y),
            (-FT_X, -LANE_HALF), (-FT_X, 0.0), (-FT_X, LANE_HALF),
            (-HASH_X, -HALF_WID), (-(RIM_X - THREE_R), 0.0), (-HASH_X, HALF_WID)]


def _keypoints() -> list:
    left = _left_half()
    mid = [(0.0, -HALF_WID), (0.0, 0.0), (0.0, HALF_WID)]
    # right half in model order: 25 26 27 28 29 30 31 32 33 34 35 37 38 40 41
    r = HALF_LEN
    right = [(HASH_X, -HALF_WID), (RIM_X - THREE_R, 0.0), (HASH_X, HALF_WID),
             (FT_X, -LANE_HALF), (FT_X, 0.0), (FT_X, LANE_HALF),
             (CORNER_X, -CORNER_Y), (CORNER_X, CORNER_Y), None,
             (r, -HALF_WID), (r, -CORNER_Y), (r, -LANE_HALF), (r, LANE_HALF), (r, CORNER_Y), (r, HALF_WID)]
    return left + mid + right


KEYPOINTS_FT = _keypoints()
assert len(KEYPOINTS_FT) == 33
# Points on a straight line whose position *along* that line is weakly defined (line midpoints, arc tops,
# hash marks).  They get lower weight in the fit.
WEAK_KEYPOINTS = {"13", "29", "16", "26", "15", "17", "25", "27", "21"}


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int) -> np.ndarray:
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def _seg(p, q, step: float) -> np.ndarray:
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(2, int(np.ceil(np.linalg.norm(q - p) / step)) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return p + t * (q - p)


def court_lines(step: float = 0.5) -> dict[str, np.ndarray]:
    """Painted floor lines as dense polylines (feet), keyed by name."""
    L, W = HALF_LEN, HALF_WID
    lines = {
        "sideline_far": _seg((-L, -W), (L, -W), step),
        "sideline_near": _seg((-L, W), (L, W), step),
        "baseline_left": _seg((-L, -W), (-L, W), step),
        "baseline_right": _seg((L, -W), (L, W), step),
        "halfcourt": _seg((0, -W), (0, W), step),
        "center_circle": _arc(0, 0, CIRCLE_R, 0, 2 * np.pi, int(2 * np.pi * CIRCLE_R / step)),
    }
    alpha = float(np.arcsin(CORNER_Y / THREE_R))
    for side, s in (("left", -1.0), ("right", 1.0)):
        bx, ft, rim = s * L, s * FT_X, s * RIM_X
        lines[f"lane_{side}_far"] = _seg((bx, -LANE_HALF), (ft, -LANE_HALF), step)
        lines[f"lane_{side}_near"] = _seg((bx, LANE_HALF), (ft, LANE_HALF), step)
        lines[f"ftline_{side}"] = _seg((ft, -LANE_HALF), (ft, LANE_HALF), step)
        # outer (solid) half of the free-throw circle, facing centre court
        a_mid = 0.0 if s < 0 else np.pi
        lines[f"ftcircle_{side}"] = _arc(ft, 0, CIRCLE_R, a_mid - np.pi / 2, a_mid + np.pi / 2,
                                         int(np.pi * CIRCLE_R / step))
        lines[f"corner3_{side}_far"] = _seg((bx, -CORNER_Y), (s * CORNER_X, -CORNER_Y), step)
        lines[f"corner3_{side}_near"] = _seg((bx, CORNER_Y), (s * CORNER_X, CORNER_Y), step)
        # arc bulges toward centre court
        if s < 0:
            arc = _arc(rim, 0, THREE_R, -alpha, alpha, int(2 * alpha * THREE_R / step))
        else:
            arc = _arc(rim, 0, THREE_R, np.pi - alpha, np.pi + alpha, int(2 * alpha * THREE_R / step))
        lines[f"arc3_{side}"] = arc
    return lines


def court_outline() -> np.ndarray:
    return np.array([(-HALF_LEN, -HALF_WID), (HALF_LEN, -HALF_WID), (HALF_LEN, HALF_WID), (-HALF_LEN, HALF_WID)])
