"""Minimal reader for M2 court.json (image -> court feet; H is scale-free, divide by w)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RIMS_FT = ((-41.75, 0.0), (41.75, 0.0))


class Court:
    def __init__(self, data: dict | None):
        self.H = {}
        for fr in (data or {}).get("frames", []):
            if fr.get("ok") and fr.get("H_img2court"):
                self.H[int(fr["f"])] = np.asarray(fr["H_img2court"], dtype=float)

    @classmethod
    def load(cls, path: str | Path | None) -> "Court":
        return cls(json.loads(Path(path).read_text()) if path and Path(path).exists() else None)

    def __bool__(self) -> bool:
        return bool(self.H)

    def to_court(self, f: int, xy: tuple[float, float]) -> tuple[float, float] | None:
        H = self.H.get(f)
        if H is None:
            return None
        p = H @ np.array([xy[0], xy[1], 1.0])
        if abs(p[2]) < 1e-12:
            return None
        x, y = p[0] / p[2], p[1] / p[2]
        return (float(x), float(y)) if abs(x) < 60 and abs(y) < 35 else None


def nearest_rim(xy: tuple[float, float]) -> tuple[float, float]:
    return min(RIMS_FT, key=lambda r: (r[0] - xy[0]) ** 2 + (r[1] - xy[1]) ** 2)


def is_three(xy: tuple[float, float]) -> bool:
    """NBA 3-point line: 23.75 ft arc, 22 ft in the corners (|y| > 14 ft from the rim line)."""
    rx, ry = nearest_rim(xy)
    if abs(xy[1] - ry) >= 22.0 and abs(xy[0]) > 47 - 14:
        return True
    return float(np.hypot(xy[0] - rx, xy[1] - ry)) > 23.75
