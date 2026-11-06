"""Video metadata: fps, frame count, size and the first frame's PTS (t = start_pts + f / fps)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int
    start_pts: float

    def t(self, f: float) -> float:
        return round(self.start_pts + f / self.fps, 4)

    def f(self, t: float) -> float:
        return (t - self.start_pts) * self.fps


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    start = 0.0
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=start_time", "-of", "json", path], capture_output=True, text=True, timeout=30)
        start = float(json.loads(r.stdout)["streams"][0].get("start_time") or 0.0)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        start = 0.0
    return VideoInfo(path, fps, n, w, h, start)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
