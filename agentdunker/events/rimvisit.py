"""Find moments when the ball comes to the rim, and judge whether it went through."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .balltrack import BallObs, Rim, rim_at


@dataclass
class RimVisit:
    f_first: int            # first frame the ball is in the rim neighbourhood
    f_last: int
    f_cross: int | None     # frame the ball crosses the rim plane downward (interpolated)
    x_cross: float | None   # horizontal offset at the crossing, in rim widths (0 = centre)
    above: bool             # ball was seen above the rim (a real attempt, not a pass under the basket)
    points: list[tuple[int, float, float]] = field(default_factory=list)  # (f, x_rel, y_rel)


def rim_relative(ball: dict[int, BallObs], rim_by_f: dict[int, Rim], max_gap: int) -> dict[int, tuple[float, float]]:
    """Ball position relative to the rim centre, in rim widths (+y = below the rim)."""
    out = {}
    for f, b in ball.items():
        r = rim_at(rim_by_f, f, max_gap)
        if r is None or r.width < 8:
            continue
        out[f] = ((b.xy[0] - r.center[0]) / r.width, (b.xy[1] - r.center[1]) / r.width)
    return out


def find_visits(rel: dict[int, tuple[float, float]], fps: float, near_x: float = 1.6, top: float = -4.0,
                bottom: float = 1.2, gap_s: float = 0.7) -> list[RimVisit]:
    frames = [f for f in sorted(rel) if abs(rel[f][0]) < near_x and top < rel[f][1] < bottom]
    groups: list[list[int]] = []
    for f in frames:
        if groups and (f - groups[-1][-1]) / fps <= gap_s:
            groups[-1].append(f)
        else:
            groups.append([f])
    visits = []
    for g in groups:
        # extend with following frames (ball below the rim / dropping out) for the crossing test
        span = [f for f in sorted(rel) if g[0] <= f <= g[-1] + int(0.5 * fps)]
        pts = [(f, *rel[f]) for f in span]
        f_cross = x_cross = None
        for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
            if y0 < 0 <= y1 and abs(x0) < near_x:
                a = -y0 / (y1 - y0) if y1 != y0 else 0.0
                f_cross, x_cross = int(round(f0 + a * (f1 - f0))), float(x0 + a * (x1 - x0))
                break
        above = any(y < -0.3 for _, _, y in pts)
        visits.append(RimVisit(g[0], g[-1], f_cross, x_cross, above, pts))
    return visits
