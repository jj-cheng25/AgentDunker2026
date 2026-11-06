"""Video/audio helpers: probing, frame access by source PTS, raw-frame encoding, atomic writes."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

cv2.setNumThreads(4)
W, H, FPS = 1920, 1080, 30
SR = 48000


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int
    duration: float
    start_pts: float
    has_audio: bool


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    fps = num / den if den else 30.0
    duration = float(v.get("duration") or d["format"]["duration"])
    n = int(v.get("nb_frames") or round(duration * fps))
    return VideoInfo(str(path), fps, int(v["width"]), int(v["height"]), n, duration,
                     float(v.get("start_time") or 0.0), any(s["codec_type"] == "audio" for s in d["streams"]))


def fit_1080(img: np.ndarray) -> np.ndarray:
    """Letterbox/resize any frame to 1920x1080 RGB."""
    h, w = img.shape[:2]
    if (w, h) == (W, H):
        return img
    s = min(W / w, H / h)
    nw, nh = round(w * s), round(h * s)
    out = np.zeros((H, W, 3), np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return out


class FrameReader:
    """Sequential-friendly frame access by source frame index (RGB, 1080p)."""

    def __init__(self, path: str | Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = -1
        self.img: Optional[np.ndarray] = None

    def get(self, index: int) -> np.ndarray:
        index = max(0, min(int(index), self.n - 1)) if self.n > 0 else max(0, int(index))
        if self.img is not None and index == self.pos:
            return self.img
        if index < self.pos or index > self.pos + 90:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.pos = index - 1
        while self.pos < index:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.pos += 1
            self.img = fit_1080(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.img is None:
            self.img = np.zeros((H, W, 3), np.uint8)
        return self.img

    def close(self) -> None:
        self.cap.release()


def frame_at(path: str | Path, index: int) -> np.ndarray:
    r = FrameReader(path)
    try:
        return r.get(index).copy()
    finally:
        r.close()


class Encoder:
    """Pipe RGB frames to an H.264 1080p30 file; written to a temp name and renamed on close."""

    def __init__(self, out: str | Path, fps: int = FPS, crf: int = 18):
        self.out = Path(out)
        self.tmp = self.out.with_name(self.out.stem + ".part" + self.out.suffix)
        self.n = 0
        self.proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-threads", "8", "-movflags", "+faststart", str(self.tmp)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self) -> Path:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed: {err[-800:]}")
        os.replace(self.tmp, self.out)
        return self.out


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-800:]}")


def read_wav(path: str | Path) -> np.ndarray:
    """Any audio file → float32 stereo 48 kHz array (n, 2)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def write_wav(path: str | Path, audio: np.ndarray) -> Path:
    import soundfile as sf
    path = Path(path)
    tmp = path.with_name(path.stem + ".part.wav")
    sf.write(str(tmp), np.clip(audio, -1, 1).astype(np.float32), SR, subtype="PCM_16")
    os.replace(tmp, path)
    return path
