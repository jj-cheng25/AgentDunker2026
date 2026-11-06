"""Evidence videos: court lines re-projected per frame, plus a top-down minimap."""
from __future__ import annotations

import subprocess

import cv2
import numpy as np

from .api import CourtCalibration
from .draw import draw_court, minimap


class H264Writer:
    def __init__(self, path: str, fps: float, size):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
             "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p",
             "-threads", "4", path], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def render_overlay(video_path: str, cal: CourtCalibration, out_path: str, width: int = 1280, feet=None):
    """feet: optional {frame: [(x, y, (b, g, r)), ...]} image foot points to show on the minimap."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = width / w
    S = np.diag([s, s, 1.0])
    mm_w = width
    mm_h = None
    writer = None
    f = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.resize(fr, (width, round(h * s)), interpolation=cv2.INTER_AREA)
        rec = cal.frames.get(f, {})
        H = cal.H(f)
        pts = cols = view = None
        if H is not None:
            Hc = S @ np.linalg.inv(H)
            color = (0, 255, 255) if rec.get("ok") else (0, 0, 255)
            draw_court(small, Hc, color, 2)
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
            view = cal.img_to_court(f, corners)
            if feet and f in feet:
                xy = np.array([[x, y] for x, y, _ in feet[f]], float)
                pts = cal.img_to_court(f, xy)
                cols = [c for *_, c in feet[f]]
                for (x, y, c) in feet[f]:
                    cv2.circle(small, (int(x * s), int(y * s)), 4, c, -1)
        label = f"f{f} {rec.get('src', '-')} conf {rec.get('conf', 0):.2f} {'OK' if rec.get('ok') else 'NOT OK'}"
        cv2.putText(small, label, (12, 30), 0, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(small, label, (12, 30), 0, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if view is not None and np.all(np.isfinite(view)):
            view = np.clip(view, -200, 200)
        mm = minimap(pts, None, cols, view if view is not None and np.all(np.isfinite(view)) else None)
        mm = cv2.resize(mm, (mm_w, round(mm.shape[0] * mm_w / mm.shape[1])))
        canvas = np.vstack([small, mm])
        if writer is None:
            writer = H264Writer(out_path, fps, (canvas.shape[1], canvas.shape[0] // 2 * 2))
        writer.write(canvas[: canvas.shape[0] // 2 * 2])
        f += 1
    if writer:
        writer.close()
