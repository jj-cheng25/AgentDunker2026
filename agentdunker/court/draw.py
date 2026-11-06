"""Overlay and minimap drawing."""
from __future__ import annotations

import cv2
import numpy as np

from . import template as T
from .geometry import LINES, _front, project


def draw_court(frame: np.ndarray, H_court2img: np.ndarray, color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    out = frame
    h, w = frame.shape[:2]
    for poly in LINES.values():
        ok = _front(H_court2img, poly)
        p = project(H_court2img, poly)
        ok &= np.all(np.abs(p) < 4 * max(w, h), axis=1)
        seg = []
        for good, q in zip(ok, p):
            if good:
                seg.append(q)
            elif len(seg) > 1:
                cv2.polylines(out, [np.round(np.array(seg)).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
                seg = []
            else:
                seg = []
        if len(seg) > 1:
            cv2.polylines(out, [np.round(np.array(seg)).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return out


def minimap(points_court=None, labels=None, colors=None, view_poly=None, ppf: float = 6.0, margin: int = 20) -> np.ndarray:
    """Top-down court (x right, y down = near sideline at bottom)."""
    W = int(2 * T.HALF_LEN * ppf + 2 * margin)
    Hh = int(2 * T.HALF_WID * ppf + 2 * margin)
    img = np.full((Hh, W, 3), (40, 70, 110), np.uint8)

    def to_px(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return np.round(np.c_[(p[:, 0] + T.HALF_LEN) * ppf + margin, (p[:, 1] + T.HALF_WID) * ppf + margin]).astype(np.int32)

    if view_poly is not None and len(view_poly) >= 3:
        ov = img.copy()
        cv2.fillPoly(ov, [to_px(view_poly)], (80, 120, 160))
        img = cv2.addWeighted(ov, 0.6, img, 0.4, 0)
    for poly in LINES.values():
        cv2.polylines(img, [to_px(poly)], False, (230, 230, 230), 1, cv2.LINE_AA)
    if points_court is not None:
        for i, p in enumerate(np.asarray(points_court, float).reshape(-1, 2)):
            if not np.all(np.isfinite(p)):
                continue
            c = tuple(int(v) for v in (colors[i] if colors is not None else (0, 0, 255)))
            q = tuple(to_px(p)[0])
            cv2.circle(img, q, 5, c, -1, cv2.LINE_AA)
            if labels is not None:
                cv2.putText(img, str(labels[i]), (q[0] + 5, q[1] - 5), 0, 0.4, c, 1, cv2.LINE_AA)
    return img
