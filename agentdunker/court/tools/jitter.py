"""Temporal stability of court.json: frame-to-frame disagreement between the court projection and image motion.

For consecutive frames i, i+1 of one shot (both with H), the court grid projected at i is carried to i+1 by the
KLT homography of the image itself and compared with the projection at i+1.  Camera pans/zooms cancel out; what
remains is line jitter (px, 1080p scale).

python -m agentdunker.court.tools.jitter <video> <court.json>
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from ..geometry import project
from ..video import WORK_W, interframe_motion, static_overlay_mask


def jitter(video: str, court: dict) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        frames.append(cv2.resize(fr, (WORK_W, round(h * WORK_W / w)), interpolation=cv2.INTER_AREA))
    scale = WORK_W / w
    mask = static_overlay_mask(frames)
    recs = court["frames"]
    gx, gy = np.meshgrid(np.linspace(-47, 47, 19), np.linspace(-25, 25, 11))
    grid = np.c_[gx.ravel(), gy.ravel()]
    out = []
    for i in range(len(recs) - 1):
        a, b = recs[i], recs[i + 1]
        if a["H_img2court"] is None or b["H_img2court"] is None or a.get("shot") != b.get("shot"):
            continue
        M, ratio = interframe_motion(frames[i], frames[i + 1], mask, scale)
        if M is None or ratio < 0.5:
            continue
        Ha, Hb = np.linalg.inv(np.asarray(a["H_img2court"])), np.linalg.inv(np.asarray(b["H_img2court"]))
        pa, pb = project(Ha, grid), project(Hb, grid)
        vis = np.isfinite(pa).all(1) & np.isfinite(pb).all(1) & (pb[:, 0] > 0) & (pb[:, 0] < w) & (pb[:, 1] > 0) & (pb[:, 1] < h)
        if vis.sum() < 4:
            continue
        out.append(float(np.median(np.linalg.norm(project(M, pa[vis]) - pb[vis], axis=1))) * 1080.0 / h)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("court")
    a = ap.parse_args()
    j = jitter(a.video, json.load(open(a.court)))
    print(f"pairs {len(j)}  jitter median {np.median(j):.2f}px  p95 {np.percentile(j, 95):.2f}px  max {j.max():.2f}px")


if __name__ == "__main__":
    main()
