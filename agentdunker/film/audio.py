"""Film audio: intro/outro music, transition SFX, narration lines (re-timed), loudness and limiting.

Live and replay segments carry only the commentary (no arena sound), as in the reference film. Without
narration the live segment keeps the original broadcast sound instead, so the film is never silent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .media import SR, read_wav

logger = logging.getLogger(__name__)
TARGET_LUFS = -21.0          # reference film, whole programme: −20.97 LUFS
VOICE_LUFS = -21.5           # reference body (voice only): −21.55 LUFS
SFX_DB = -9.0
PEAK = 0.95


def _place(track: np.ndarray, clip: np.ndarray, t: float, gain: float = 1.0) -> None:
    a = int(round(t * SR))
    if a >= len(track) or len(clip) == 0:
        return
    b = min(len(track), a + len(clip))
    src = clip[max(0, -a): b - a]
    track[max(0, a): max(0, a) + len(src)] += src * gain


def voice_pieces(vo: np.ndarray, lines: list[dict]) -> list[tuple[dict, np.ndarray, float]]:
    """Cut M6's placed voice track into one piece per spoken line (bounded by neighbour midpoints)."""
    spoken = [l for l in lines if l.get("text")]
    out = []
    for i, l in enumerate(spoken):
        lo = l["start"] - 0.35 if i == 0 else max(l["start"] - 0.35, (spoken[i - 1]["end"] + l["start"]) / 2)
        hi = l["end"] + 0.35 if i == len(spoken) - 1 else min(l["end"] + 0.35, (l["end"] + spoken[i + 1]["start"]) / 2)
        a, b = max(0, int(lo * SR)), min(len(vo), int(hi * SR))
        piece = vo[a:b].copy()
        ramp = min(len(piece) // 4, int(0.02 * SR))
        if ramp > 0:
            piece[:ramp] *= np.linspace(0, 1, ramp)[:, None]
            piece[-ramp:] *= np.linspace(1, 0, ramp)[:, None]
        out.append((l, piece, a / SR))
    return out


def loudness(x: np.ndarray) -> Optional[float]:
    try:
        import pyloudnorm as pyln
        v = pyln.Meter(SR).integrated_loudness(x)
        return float(v) if np.isfinite(v) else None
    except (ImportError, ValueError):
        return None


def build_mix(duration: float, segments: list[dict], *, intro: Optional[Path], outro: Optional[Path],
              sfx: Optional[np.ndarray], pieces: list[tuple[float, np.ndarray]], live_audio: Optional[np.ndarray],
              body_offset: float) -> tuple[np.ndarray, dict]:
    """``pieces`` = [(film_t, audio)] already re-timed; ``live_audio`` only when there is no narration."""
    track = np.zeros((int(duration * SR) + 1, 2), np.float32)
    seg = {s["kind"]: s for s in segments}
    voice = np.zeros_like(track)
    for t, clip in pieces:
        _place(voice, clip, t)
    stats: dict = {"voice_pieces": len(pieces)}
    if pieces:
        lv = loudness(voice)
        if lv is not None:
            voice *= 10 ** ((VOICE_LUFS - lv) / 20)
            stats["voice_lufs_in"] = round(lv, 2)
    track += voice
    if live_audio is not None and "live" in seg:
        _place(track, live_audio, seg["live"]["film"][0], 10 ** (-6 / 20))
        stats["live_original_audio"] = True
    if sfx is not None and "bridge" in seg:
        _place(track, sfx, seg["bridge"]["film"][0], 10 ** (SFX_DB / 20))
    for name, path in (("intro", intro), ("outro", outro)):
        if path is not None and name in seg:
            try:
                _place(track, read_wav(path), seg[name]["film"][0])
            except (OSError, RuntimeError) as e:
                logger.warning("%s audio missing: %s", name, e)
    peak = float(np.abs(track).max()) or 1.0
    if peak > PEAK:                                     # soft limit: scale only the over-peak samples
        over = np.abs(track) > PEAK
        track[over] = np.sign(track[over]) * (PEAK + (1 - PEAK) * np.tanh((np.abs(track[over]) - PEAK) / (1 - PEAK)))
        track = np.clip(track, -PEAK, PEAK)
    lf = loudness(track)
    if lf is not None and abs(lf - TARGET_LUFS) > 0.5:             # programme loudness ≈ reference (−21 LUFS)
        g = min(10 ** ((TARGET_LUFS - lf) / 20), PEAK / max(1e-6, float(np.abs(track).max())))
        track *= g
        stats["film_gain_db"] = round(20 * np.log10(g), 2)
    stats["film_lufs"] = loudness(track)
    stats["peak"] = round(float(np.abs(track).max()), 3)
    return track, stats
