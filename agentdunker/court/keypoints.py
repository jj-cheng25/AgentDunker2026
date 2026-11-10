"""court-33 keypoint detector (YOLO11m-pose, Roboflow basketball court labels) on OpenCV DNN CPU."""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_URL = ("https://huggingface.co/MTerryJack/Element-PitchCalib-Basketball-Court/resolve/"
             "81111962e7bfeb669cfbedad2ff455342dbb5529/weights.onnx")
MODEL_SHA256 = "9c00051bd207f5b76c6e492f3672be8dd7cce5a217ab271e971378fca057d26d"
MODEL_PATH = Path(os.environ.get("ADK_COURT_MODEL", Path(os.environ.get("ADK_MODEL_DIR", Path.home() / ".cache" / "agentdunker" / "models")) / "court-33-yolo11m-pose.onnx"))


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading court keypoint model to %s", path)
        __import__("agentdunker.weights", fromlist=["x"]).fetch_for_module("court-33-yolo11m-pose.onnx", path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            path.unlink()
            raise RuntimeError(f"court model sha256 mismatch: {digest}")
    return path


class CourtKeypointDetector:
    """Returns (33, 3) arrays of [x, y, conf] in source pixels."""

    def __init__(self, path: Path | None = None, size: int = 640, threads: int = 8):
        cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(str(ensure_model(path or MODEL_PATH)))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.size = size

    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        # The Roboflow training pipeline stretch-resizes to 640x640; letterboxing distorts keypoints by 40-120 px.
        h, w = frame_bgr.shape[:2]
        s = self.size
        img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_AREA)
        self.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (s, s), swapRB=True, crop=False))
        raw = self.net.forward()[0]  # (104, 8400)
        best = int(np.argmax(raw[4]))
        kp = raw[5:, best].reshape(33, 3).astype(np.float64)
        kp[:, 0] *= w / s
        kp[:, 1] *= h / s
        return kp, float(raw[4, best])
