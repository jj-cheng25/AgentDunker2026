"""Duck the source audio under the narration, mix, and mux a preview."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import audio
from .files import atomic_path


def duck_envelope(intervals: list[tuple[float, float]], n: int, bed_db: float, duck_db: float,
                  pre: float = 0.25, post: float = 0.35, ramp: float = 0.2) -> np.ndarray:
    """Gain curve: bed_db normally, duck_db from `pre` before each line to `post` after it."""
    env_db = np.full(n, bed_db, dtype=np.float32)
    for a, b in intervals:
        i, j = max(0, int((a - pre) * audio.SR)), min(n, int((b + post) * audio.SR))
        env_db[i:j] = min(bed_db, duck_db)
    k = max(1, int(ramp * audio.SR))
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(np.pad(env_db, (k // 2, k - k // 2 - 1), mode="edge"), kernel, mode="valid")
    return (10 ** (smooth / 20)).astype(np.float32)


def mix(vo: np.ndarray, bed: np.ndarray | None, intervals: list[tuple[float, float]],
        bed_db: float = -6.0, duck_db: float = -20.0) -> np.ndarray:
    out = vo.astype(np.float32).copy()
    if bed is not None and np.isfinite(bed_db):
        b = np.zeros_like(out)
        b[: min(len(b), len(bed))] = bed[: len(b)]
        out = out + b * duck_envelope(intervals, len(out), bed_db, duck_db)
    peak = np.abs(out).max()
    if peak > 0.95:
        out *= 0.95 / peak
    return out


def mux(video: str | Path, wav: str | Path, out: str | Path, start: float = 0.0, duration: float | None = None) -> Path:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-pix_fmt", "yuv420p", "-threads", "8", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    with atomic_path(out) as tmp:
        subprocess.run(cmd + [str(tmp)], check=True)
    return Path(out)
