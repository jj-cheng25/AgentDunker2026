"""Wrist keypoints near the ball (YOLO11x-pose on a ball-centred crop; AGPL-3.0 like M1's YOLO11)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_POSE = Path(os.environ.get("ADK_MODEL_DIR", Path.home() / ".cache" / "agentdunker" / "models")) / "yolo11x-pose.pt"
WRISTS = (9, 10)
ELBOWS = (7, 8)


@dataclass(frozen=True)
class Person:
    box: tuple[float, float, float, float]
    wrists: list[tuple[float, float, float]]  # (x, y, conf) in frame pixels
    elbows: list[tuple[float, float, float]]
    conf: float


class PoseNear:
    def __init__(self, weights: Path | None = None, threads: int = 8):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(threads)
        w = weights or DEFAULT_POSE
        self.model = YOLO(str(w) if Path(w).exists() else str(__import__("agentdunker.weights", fromlist=["x"]).fetch_for_module("yolo11x-pose.pt", w)))

    def people(self, frame: np.ndarray, center: tuple[float, float], size: int = 640) -> list[Person]:
        h, w = frame.shape[:2]
        size = min(size, w, h)
        x1 = int(np.clip(center[0] - size / 2, 0, w - size))
        y1 = int(np.clip(center[1] - size / 2, 0, h - size))
        r = self.model(frame[y1:y1 + size, x1:x1 + size], imgsz=640, verbose=False, conf=0.2)[0]
        out = []
        if r.keypoints is None:
            return out
        for b, k, c in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            off = np.array([x1, y1])
            out.append(Person(tuple((b.reshape(2, 2) + off).ravel().tolist()),
                              [(*(k[i, :2] + off).tolist(), float(k[i, 2])) for i in WRISTS],
                              [(*(k[i, :2] + off).tolist(), float(k[i, 2])) for i in ELBOWS], float(c)))
        return out
