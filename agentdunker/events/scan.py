"""Run the basketball detector over selected frames of a video, with an on-disk cache."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2

from .detector import BallDetector, Det

logger = logging.getLogger(__name__)
KEEP = {"ball", "ball-in-basket", "rim", "player-jump-shot", "player-layup-dunk", "player-in-possession",
        "player-shot-block"}


class DetCache:
    """Per-video cache {frame: [Det]} stored as JSON next to the outputs."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[int, list[Det]] = {}
        if path and path.exists():
            raw = json.loads(path.read_text())
            self.data = {int(f): [Det(d[0], tuple(d[1]), d[2]) for d in ds] for f, ds in raw.items()}

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {str(f): [[d.cls, [round(v, 1) for v in d.box], round(d.score, 3)] for d in ds]
               for f, ds in sorted(self.data.items())}
        self.path.write_text(json.dumps(raw))


def detect_frames(video: str, frames: list[int], cache: DetCache, detector: BallDetector | None = None,
                  min_score: float = 0.1) -> dict[int, list[Det]]:
    """Detect on `frames` (sorted, deduplicated); cached frames are not recomputed."""
    todo = sorted(set(f for f in frames if f not in cache.data))
    if todo:
        detector = detector or BallDetector()
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, todo[0])
        f = todo[0]
        want = set(todo)
        while f <= todo[-1]:
            if f in want:
                ok, frame = cap.read()
                if not ok:
                    break
                cache.data[f] = [d for d in detector.detect(frame, min_score=min_score) if d.cls in KEEP]
            elif not cap.grab():
                break
            f += 1
        cap.release()
        cache.save()
        logger.info("detected %d frames of %s", len(todo), video)
    return {f: cache.data.get(f, []) for f in frames}
