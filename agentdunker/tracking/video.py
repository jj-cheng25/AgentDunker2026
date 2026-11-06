"""Video decoding with source PTS and frame-to-frame camera motion."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import av
import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    sha256: str
    fps: float
    width: int
    height: int
    n_frames: int


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_frames(path: str) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, pts_seconds, bgr) in presentation order."""
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            t = float(fr.pts * s.time_base) if fr.pts is not None else float(i / float(s.average_rate or 30))
            yield i, t, fr.to_ndarray(format="bgr24")


def probe(path: str) -> tuple[float, int, int]:
    with av.open(path) as c:
        s = c.streams.video[0]
        return float(s.average_rate or 30), s.codec_context.width, s.codec_context.height


class CameraMotion:
    """Estimates the similarity transform from frame f-1 to frame f, and flags cuts and flashes.

    Sparse LK flow on a downscaled grey image; person boxes from the latest detection are masked out
    so moving players do not drag the estimate. When fewer than ``min_inlier_frac`` of the tracked
    points fit one transform the frame is either a photographer flash (mean brightness jumps by
    > 20 %; the pre-flash frame stays the reference) or a shot cut.
    """

    def __init__(self, width: int, work_width: int = 640, min_inlier_frac: float = 0.25) -> None:
        self.scale = min(1.0, work_width / width)
        self.min_inlier_frac = min_inlier_frac
        self.prev: np.ndarray | None = None

    def _grey(self, bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def _estimate(self, prev: np.ndarray, g: np.ndarray, boxes: np.ndarray | None) -> tuple[np.ndarray | None, float]:
        mask = np.full(g.shape, 255, np.uint8)
        mask[int(g.shape[0] * 0.82):, :] = 0  # broadcast score bug lives at the bottom
        if boxes is not None:
            for x1, y1, x2, y2 in (np.asarray(boxes) * self.scale).astype(int):
                mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
        pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8, mask=mask)
        if pts is None or len(pts) < 12:
            return None, 1.0  # texture-less: cannot tell, assume static
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 12:
            return None, 0.0
        A, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if A is None or inl is None:
            return None, 0.0
        return np.vstack([A, [0, 0, 1]]), float(inl.sum()) / len(pts)

    def step(self, bgr: np.ndarray, boxes: np.ndarray | None) -> tuple[np.ndarray, str | None]:
        """Return (3x3 previous->current pixel transform, event) with event None | "cut" | "flash"."""
        g = self._grey(bgr)
        prev = self.prev
        if prev is None:
            self.prev = g
            return np.eye(3), None
        M, frac = self._estimate(prev, g, boxes)
        if frac < self.min_inlier_frac:
            if g.mean() > 1.2 * prev.mean() + 5:
                return np.eye(3), "flash"  # keep the pre-flash frame as reference
            self.prev = g
            return np.eye(3), "cut"
        self.prev = g
        if M is None:
            return np.eye(3), None
        S = np.diag([self.scale, self.scale, 1.0])
        return np.linalg.inv(S) @ M @ S, None


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(np.asarray(pts).reshape(-1, 2)))] @ H.T
    return p[:, :2] / p[:, 2:3]


def warp_box(H: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Warp an xyxy box by a similarity transform (corners → axis-aligned box)."""
    x1, y1, x2, y2 = box
    p = apply_h(H, np.array([[x1, y1], [x2, y2]]))
    return np.array([p[0, 0], p[0, 1], p[1, 0], p[1, 1]])
