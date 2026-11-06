"""Audio I/O and analysis (48 kHz mono float32), pitch-preserving tempo via ffmpeg rubberband."""
from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .files import atomic_path

SR = 48000
HOP = 0.01


def load(path: str | Path, start: float | None = None, dur: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def has_audio(path: str | Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False).stdout
    return bool(out.strip())


def save(path: str | Path, x: np.ndarray) -> Path:
    path = Path(path)
    with atomic_path(path) as tmp:
        wavfile.write(tmp, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return path


def wav_ok(path: str | Path) -> bool:
    """True when `path` is a complete RIFF/WAVE file that ffmpeg decodes to some samples.
    A truncated file still declares the full data size in its header, so that is checked
    first; ffmpeg alone would decode a truncated take without complaint."""
    path = Path(path)
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return False
            pos, has_fmt = 12, False
            while pos + 8 <= size:
                f.seek(pos)
                cid, n = struct.unpack("<4sI", f.read(8))
                if cid == b"fmt ":
                    has_fmt = True
                elif cid == b"data":
                    # 0xFFFFFFFF marks a streamed header without a final size.
                    if not has_fmt or n == 0 or (n != 0xFFFFFFFF and pos + 8 + n > size):
                        return False
                    break
                pos += 8 + n + (n & 1)
            else:
                return False
        return len(load(path)) > 0
    except (OSError, struct.error, subprocess.CalledProcessError):
        return False


def frame_db(x: np.ndarray) -> np.ndarray:
    hop = int(HOP * SR)
    n = max(1, len(x) // hop)
    frames = x[: n * hop].reshape(n, hop) if len(x) >= hop else np.pad(x, (0, hop - len(x)))[None]
    return 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)


def speech_edges(x: np.ndarray, on_db: float = -40, off_db: float = -50) -> tuple[float, float]:
    """First frame above on_db and last frame above off_db, in seconds."""
    db = frame_db(x)
    on, off = np.where(db > on_db)[0], np.where(db > off_db)[0]
    if len(on) == 0:
        return 0.0, 0.0
    return on[0] * HOP, (off[-1] + 1) * HOP


def lead_silence(x: np.ndarray, on: float, quiet_db: float = -55) -> float:
    """Seconds of quiet audio directly before the speech onset."""
    db = frame_db(x)[: int(round(on / HOP))]
    loud = np.where(db > quiet_db)[0]
    return (len(db) - (loud[-1] + 1 if len(loud) else 0)) * HOP


def tempo(x: np.ndarray, factor: float) -> np.ndarray:
    """factor > 1 makes speech shorter; pitch and formants are kept."""
    if abs(factor - 1) < 1e-3:
        return x
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.wav", Path(d) / "b.wav"
        save(a, x)
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(a), "-af",
                        f"rubberband=tempo={factor:.4f}:pitchq=quality:formant=preserved", str(b)], check=True)
        return load(b)


def fade(x: np.ndarray, fin: float = 0.005, fout: float = 0.03) -> np.ndarray:
    y = x.copy()
    a, b = int(fin * SR), int(fout * SR)
    if a and len(y) > a:
        y[:a] *= np.linspace(0, 1, a)
    if b and len(y) > b:
        y[-b:] *= np.linspace(1, 0, b)
    return y


def loudness_lufs(path: str | Path) -> float:
    err = subprocess.run(["ffmpeg", "-v", "info", "-nostdin", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True, check=False).stderr
    vals = [line for line in err.splitlines() if line.strip().startswith("I:")]
    return float(vals[-1].split()[1]) if vals else float("nan")


def tmp_wav(x: np.ndarray, pad_before: float = 0.1, pad_after: float = 0.15) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        name = f.name
    return save(name, np.concatenate([np.zeros(int(pad_before * SR)), x, np.zeros(int(pad_after * SR))]))
