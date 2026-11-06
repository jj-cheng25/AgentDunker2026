"""Render a play to MP4: Three.js frames + source-video PIP, plus an optional side-by-side QA video.

The QA video puts the 3D frame next to the source frame with every replay position projected back
into the image (inverse homography), so position agreement can be checked by eye.
"""
from __future__ import annotations

import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .browser import render_frames
from .trajectory import CourtFrames, apply_h

logger = logging.getLogger(__name__)
FONT_PATHS = ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
COLORS = {"off": (53, 230, 255), "def": (255, 154, 46)}


def _font(size: int):
    from .browser import _cjk_font  # same bundled CJK font as the 3D page
    for p in [_cjk_font(), *FONT_PATHS]:
        if p and Path(p).exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def sample_timeline(knots: list, t: float) -> float:
    if t <= knots[0][0]:
        return knots[0][1]
    for (a0, a1), (b0, b1) in zip(knots, knots[1:]):
        if t <= b0:
            return a1 + (b1 - a1) * (t - a0) / (b0 - a0)
    return knots[-1][1]


class SourceReader:
    """Sequential reader returning the frame whose PTS is closest at or before ``t`` (t non-decreasing)."""

    def __init__(self, path: str, frame_times: dict[int, float], fps: float) -> None:
        self.cap = cv2.VideoCapture(path)
        self.times = frame_times
        self.fps = fps
        self.f = -1
        self.img = None

    def frame_index(self, t: float) -> int:
        f = max(0, int(round(t * self.fps)))
        while f > 0 and self.times.get(f, f / self.fps) > t + 0.5 / self.fps:
            f -= 1
        while self.times.get(f + 1, (f + 1) / self.fps) <= t + 0.5 / self.fps and f + 1 in self.times:
            f += 1
        return f

    def at(self, t: float) -> tuple[int, np.ndarray]:
        f = self.frame_index(t)
        if f < self.f:  # seek back (not expected)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            self.f = f - 1
        while self.f < f:
            ok, img = self.cap.read()
            if not ok:
                break
            self.f += 1
            self.img = img
        return self.f, cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)


def _pip(frame: np.ndarray, src: np.ndarray, label_font) -> np.ndarray:
    """Reference layout: 375x211 PIP bottom-right (x 1527-1902, y 752-963 at 1080p)."""
    H, W = frame.shape[:2]
    s = H / 1080
    w, h = int(375 * s), int(211 * s)
    x0, y0 = W - int(18 * s) - w, H - int(117 * s) - h
    out = frame.copy()
    out[y0:y0 + h, x0:x0 + w] = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([x0 - 1, y0 - 1, x0 + w, y0 + h], radius=int(6 * s), outline=(53, 230, 255, 110), width=max(1, int(2 * s)))
    d.rounded_rectangle([x0 + 6 * s, y0 + 5 * s, x0 + 62 * s, y0 + 25 * s], radius=3, fill=(0, 0, 0, 150))
    d.text((x0 + 12 * s, y0 + 7 * s), "原视频", font=label_font, fill=(230, 245, 255, 255))
    return np.asarray(img)


def _reproject(src: np.ndarray, f: int, src_t: float, play: dict, court: CourtFrames, font) -> np.ndarray:
    """Draw every replay position back on the source frame (dot = foot, ring = predicted)."""
    img = Image.fromarray(src)
    d = ImageDraw.Draw(img, "RGBA")
    H = court.get(f)
    side_left = play["cameraPreset"]["pos"][0] < 0
    rel = src_t - play["metadata"]["sourceTimeOffset"]
    if H is None:
        d.text((20, 20), "该帧无球场标定", font=font, fill=(255, 80, 80, 255))
        return np.asarray(img)
    Hinv = np.linalg.inv(H)
    for p in play["players"]:
        path = np.asarray(p["path"])
        X = np.interp(rel, path[:, 0], path[:, 1]) / 0.3048
        Z = np.interp(rel, path[:, 0], path[:, 2]) / 0.3048
        xy_ft = np.array([Z - 47, -X]) if side_left else np.array([47 - Z, X])
        u, v = apply_h(Hinv, xy_ft)
        if not (np.isfinite(u) and np.isfinite(v)) or not (-50 < u < src.shape[1] + 50 and -50 < v < src.shape[0] + 50):
            continue
        predicted = any(a <= rel <= b for a, b in p.get("predictedIntervals", []))
        c = COLORS[p["team"]]
        r = 11
        if predicted:
            d.ellipse([u - r, v - r, u + r, v + r], outline=c + (255,), width=3)
        else:
            d.ellipse([u - r, v - r, u + r, v + r], fill=c + (230,), outline=(0, 0, 0, 255), width=2)
        d.text((u + 12, v - 34), p["num"] or p["id"], font=font, fill=c + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    d.text((20, 20), f"源片 {src_t:5.2f}s · 圆点=3D 复盘位置反投影 · 空心=预测", font=font, fill=(255, 255, 255, 255),
           stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return np.asarray(img)


def _encoder(path: Path, w: int, h: int, fps: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-threads", "4", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def render_replay(play: dict, video: str, tracks: dict, court: dict, out: str, *, width: int = 1920, height: int = 1080,
                  fps: int = 30, workers: int = 3, pip: bool = True, compare_out: Optional[str] = None,
                  max_seconds: Optional[float] = None) -> dict:
    """Render ``play`` to ``out`` (silent H.264). Returns a receipt with timing."""
    knots = play["exportTimeline"]
    duration = knots[-1][0] if max_seconds is None else min(knots[-1][0], max_seconds)
    n = int(math.ceil(duration * fps))
    offset = play["metadata"]["sourceTimeOffset"]
    times = [(sample_timeline(knots, i / fps), i / fps) for i in range(n)]
    frame_times = {}
    for tr in tracks["tracks"]:
        for fr in tr["frames"]:
            frame_times.setdefault(fr["f"], fr["t"])
    reader = SourceReader(video, frame_times, float(tracks["video"]["fps"]))
    cf = CourtFrames(court)
    label_font, qa_font = _font(int(15 * height / 1080)), _font(22)
    enc = _encoder(Path(out), width, height, fps)
    qa = _encoder(Path(compare_out), 1920, 540, fps) if compare_out else None
    t_start = time.monotonic()

    def on_frame(i: int, img: np.ndarray) -> None:
        src_t = times[i][0] + offset
        f, src = reader.at(src_t)
        frame = _pip(img, src, label_font) if pip else img
        enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        if qa:
            left = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            right = cv2.resize(_reproject(src, f, src_t, play, cf, qa_font), (960, 540), interpolation=cv2.INTER_AREA)
            qa.stdin.write(np.ascontiguousarray(np.concatenate([left, right], 1)).tobytes())

    try:
        render_frames(play, times, width=width, height=height, workers=workers, on_frame=on_frame)
    finally:
        for e in (enc, qa):
            if e:
                e.stdin.close()
                e.wait()
    secs = time.monotonic() - t_start
    return {"output": str(out), "compare": compare_out, "frames": n, "fps": fps, "duration_s": round(n / fps, 3),
            "size": [width, height], "render_seconds": round(secs, 1), "sec_per_frame": round(secs / max(1, n), 3),
            "workers": workers, "source_window": play["metadata"]["sourceWindow"]}
