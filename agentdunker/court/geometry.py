"""Per-frame homography: keypoint RANSAC seed + edge-based court-line refinement."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import template as T

LINES = T.court_lines(step=0.5)
KP_COURT = np.array([p if p is not None else (np.nan, np.nan) for p in T.KEYPOINTS_FT], float)
KP_WEIGHT = np.array([0.5 if lab in T.WEAK_KEYPOINTS else 1.0 for lab in T.KEYPOINT_LABELS])


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def _front(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """True where court points project in front of the camera (positive w)."""
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, 2] * np.sign(np.linalg.det(H)) > 0


@dataclass
class EdgeMap:
    """Painted-line evidence at working resolution (source px = work px / scale).

    resp: thin-stripe response (white top-hat or dark black-hat), so step edges between painted areas, which vary
    by arena, give no response.  hxx/hxy/hyy: Hessian of the smoothed response, used to check that a peak is a
    ridge running along the expected line direction.
    """
    resp: np.ndarray
    hxx: np.ndarray
    hxy: np.ndarray
    hyy: np.ndarray
    scale: float
    mask: np.ndarray         # uint8, 255 where evidence may be used
    dts: list | None = None  # distance transforms of ridge pixels per normal-orientation bin (0/45/90/135 deg)

    @property
    def edges(self) -> np.ndarray:   # shape helper kept for callers
        return self.resp

    @classmethod
    def from_frame(cls, frame_bgr: np.ndarray, work_width: int = 960, ignore_boxes=None, build: bool = True) -> "EdgeMap":
        h, w = frame_bgr.shape[:2]
        s = work_width / w
        small = frame_bgr if w == work_width else cv2.resize(frame_bgr, (work_width, round(h * s)),
                                                                interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        resp = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k),
                          cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)).astype(np.float32)
        resp = cv2.GaussianBlur(resp, (0, 0), 1.0)
        hxx = cv2.Sobel(resp, cv2.CV_32F, 2, 0, ksize=3)
        hyy = cv2.Sobel(resp, cv2.CV_32F, 0, 2, ksize=3)
        hxy = cv2.Sobel(resp, cv2.CV_32F, 1, 1, ksize=3)
        # crowd, players, benches and floor lettering are dense with stripes; random template lines that cross them
        # would collect chance matches, so evidence is only taken where stripes are sparse (open floor)
        density = cv2.boxFilter((resp > 40).astype(np.float32), -1, (25, 25))   # floor grain/gloss stays below 40
        mask = np.where(density < 0.3, 255, 0).astype(np.uint8)
        for x0, y0, x1, y1 in (ignore_boxes or []):
            mask[max(0, int(y0 * s)):int(y1 * s) + 1, max(0, int(x0 * s)):int(x1 * s) + 1] = 0
        em = cls(resp, hxx, hxy, hyy, s, mask)
        if build:
            em.build_dts()
        return em

    def build_dts(self, thr: float = 25.0) -> None:
        """Ridge pixels (strong, line-like, in usable area) binned by normal orientation -> distance transforms."""
        tr = self.hxx + self.hyy
        disc = np.sqrt((self.hxx - self.hyy) ** 2 + 4 * self.hxy ** 2)
        l1, l2 = 0.5 * (tr - disc), 0.5 * (tr + disc)          # l1: most negative curvature (across the ridge)
        ridge = (self.resp > thr) & (l1 < 0) & (np.abs(l1) > 2 * np.abs(l2)) & (self.mask > 0)
        theta = 0.5 * np.arctan2(2 * self.hxy, self.hxx - self.hyy) + np.pi / 2  # normal of the ridge, mod pi
        self.dts = []
        for k in range(4):
            d = np.abs(((theta - k * np.pi / 4) + np.pi / 2) % np.pi - np.pi / 2)
            b = ridge & (d < np.pi / 5)
            self.dts.append(cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5))


def keypoint_homography(kp: np.ndarray, conf_thr: float = 0.5, img_size=(1920, 1080), reproj_thr: float = 12.0):
    """RANSAC court->image H from court-33 keypoints. Returns (H or None, inlier index array)."""
    w, h = img_size
    valid = (kp[:, 2] >= conf_thr) & np.isfinite(KP_COURT[:, 0])
    valid &= (kp[:, 0] > -0.05 * w) & (kp[:, 0] < 1.05 * w) & (kp[:, 1] > -0.05 * h) & (kp[:, 1] < 1.05 * h)
    idx = np.flatnonzero(valid)
    if len(idx) < 4:
        return None, idx[:0]
    src = KP_COURT[idx].astype(np.float32)
    dst = kp[idx, :2].astype(np.float32)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, reproj_thr, maxIters=2000, confidence=0.999)
    if H is None:
        return None, idx[:0]
    inl = idx[inl.ravel() > 0]
    if len(inl) < 4 or _degenerate(KP_COURT[inl]):
        return None, inl
    H, _ = cv2.findHomography(KP_COURT[inl].astype(np.float32), kp[inl, :2].astype(np.float32), 0)
    return (H / H[2, 2] if H is not None else None), inl


def _degenerate(pts: np.ndarray) -> bool:
    c = pts - pts.mean(0)
    s = np.linalg.svd(c, compute_uv=False)
    return s[-1] < 2.0 or cv2.contourArea(cv2.convexHull(pts.astype(np.float32))) < 60.0


def plausible(H: np.ndarray, img_size=(1920, 1080)) -> bool:
    """Reject homographies that put the camera on the court or flip the court."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    w, h = img_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    Hinv = np.linalg.inv(H)
    q = np.c_[corners, np.ones(4)] @ Hinv.T
    # bottom image corners must map to finite court points in front of camera
    if np.any(q[2:, 2] * q[0, 2] <= 0) and np.any(q[2:, 2] * q[1, 2] <= 0):
        return False
    bottom = q[2:, :2] / q[2:, 2:3]
    return bool(np.all(np.abs(bottom) < 400))
