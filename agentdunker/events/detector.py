"""Basketball detector: DEIM-M fine-tuned on NBA broadcast frames (Apache-2.0).

Weights: ortizeg/basketball-deim-m-640 (trained on Roboflow basketball-player-detection-3, CC BY 4.0).
Classes include ball, ball-in-basket, rim and shot poses. The model was trained on 1920x1080 frames squashed to
640x640, so tiles keep the 16:9 aspect to stay in-distribution.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("ball", "ball-in-basket", "number", "player", "player-in-possession", "player-jump-shot",
           "player-layup-dunk", "player-shot-block", "referee", "rim")
WEIGHTS_URL = "https://huggingface.co/ortizeg/basketball-deim-m-640/resolve/main/deim_m_640.onnx"
WEIGHTS_SHA256 = "29f575c8127e5eadde6da60cd66c3d0a5873adc0cbfd4e2af9ee35fde339fac7"
DEFAULT_WEIGHTS = Path(os.environ.get("ADK_MODEL_DIR", Path.home() / ".cache" / "agentdunker" / "models")) / "deim_m_640.onnx"


@dataclass(frozen=True)
class Det:
    cls: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 source pixels
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


def ensure_weights(path: Path = DEFAULT_WEIGHTS) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {WEIGHTS_URL}: {digest}")
    tmp.rename(path)
    return path


class BallDetector:
    def __init__(self, weights: Path | None = None, threads: int = 8):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(ensure_weights(weights or DEFAULT_WEIGHTS)), opts,
                                         providers=__import__("agentdunker.device", fromlist=["x"]).ort_providers())

    def _run(self, crops: list[np.ndarray], min_score: float) -> list[list[Det]]:
        batch = np.stack([cv2.resize(c, (640, 640), interpolation=cv2.INTER_AREA)[:, :, ::-1] for c in crops])
        batch = batch.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        sizes = np.array([[c.shape[1], c.shape[0]] for c in crops], dtype=np.int64)
        labels, boxes, scores = self.sess.run(None, {"images": batch, "orig_target_sizes": sizes})
        out = []
        for lab, box, sc in zip(labels, boxes, scores):
            keep = (sc >= min_score) & (lab >= 1)  # labels are 1-based; 0 is the dataset super-category
            out.append([Det(CLASSES[int(l) - 1], tuple(float(v) for v in b), float(s))
                        for l, b, s in zip(lab[keep], box[keep], sc[keep])])
        return out

    def detect(self, frame: np.ndarray, rois: list[tuple[int, int, int, int]] | None = None,
               min_score: float = 0.15) -> list[Det]:
        """Detect on the full frame plus optional 16:9 ROIs (x1, y1, x2, y2); boxes in frame pixels."""
        h, w = frame.shape[:2]
        regions = [(0, 0, w, h)] + list(rois or [])
        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in regions]
        dets: list[Det] = []
        for (x1, y1, _, _), ds in zip(regions, self._run(crops, min_score)):
            dets += [Det(d.cls, (d.box[0] + x1, d.box[1] + y1, d.box[2] + x1, d.box[3] + y1), d.score) for d in ds]
        return nms(dets)


def roi_around(cx: float, cy: float, w: int, h: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x1 = int(np.clip(cx - w / 2, 0, frame_w - w))
    y1 = int(np.clip(cy - h / 2, 0, frame_h - h))
    return (x1, y1, x1 + w, y1 + h)


def tiles(frame_w: int, frame_h: int, n: int = 2) -> list[tuple[int, int, int, int]]:
    """n x n overlapping 16:9 tiles covering the frame (each 1/n·1.25 of the frame side)."""
    tw, th = int(frame_w / n * 1.25), int(frame_h / n * 1.25)
    xs = np.linspace(0, frame_w - tw, n).astype(int)
    ys = np.linspace(0, frame_h - th, n).astype(int)
    return [(int(x), int(y), int(x) + tw, int(y) + th) for y in ys for x in xs]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[Det], thr: float = 0.5) -> list[Det]:
    kept: list[Det] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(k.cls != d.cls or _iou(k.box, d.box) < thr for k in kept):
            kept.append(d)
    return kept
