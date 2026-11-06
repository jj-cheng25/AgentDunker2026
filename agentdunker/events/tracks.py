"""Read M1 tracks.json into per-frame player boxes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PBox:
    track_id: int
    team: str | None
    role: str
    box: tuple[float, float, float, float]
    foot: tuple[float, float] | None
    observed: bool


class Tracks:
    def __init__(self, data: dict):
        self.data = data
        self.by_f: dict[int, list[PBox]] = {}
        self.meta: dict[int, dict] = {}
        for tr in data.get("tracks", []):
            tid = int(tr["track_id"])
            self.meta[tid] = {k: v for k, v in tr.items() if k != "frames"}
            for fr in tr.get("frames", []):
                self.by_f.setdefault(int(fr["f"]), []).append(
                    PBox(tid, tr.get("team"), tr.get("role", "player"), tuple(fr["bbox"]),
                         tuple(fr["foot"]) if fr.get("foot") else None, bool(fr.get("observed", True))))

    @classmethod
    def load(cls, path: str | Path | None) -> "Tracks":
        return cls(json.loads(Path(path).read_text()) if path and Path(path).exists() else {})

    def players(self, f: int) -> list[PBox]:
        return [b for b in self.by_f.get(f, []) if b.role == "player"]

    def box(self, tid: int, f: int) -> PBox | None:
        return next((b for b in self.by_f.get(f, []) if b.track_id == tid), None)

    def team_of(self, tid: int | None) -> str | None:
        return self.meta.get(tid, {}).get("team") if tid is not None else None


def contact(ball_xy: tuple[float, float], ball_r: float, box: tuple[float, float, float, float]) -> float:
    """Gap between the ball and a player box, in player heights (0 = ball touches or is inside the box)."""
    x1, y1, x2, y2 = box
    h = max(1.0, y2 - y1)
    dx = max(x1 - ball_xy[0], 0.0, ball_xy[0] - x2)
    dy = max(y1 - ball_xy[1], 0.0, ball_xy[1] - y2)
    return max(0.0, (dx * dx + dy * dy) ** 0.5 - ball_r) / h
