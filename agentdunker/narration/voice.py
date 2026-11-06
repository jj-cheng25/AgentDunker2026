"""Multi-take Mossland synthesis with ASR read-back selection.

Rules learned from the hand-edited reference (demo03 edits.json):
- every request starts with "[pause 0.3s]" so the first consonant is not clipped ("队友" re-record),
  and ends with one so the last syllable is not cut at the file end (flash often stops mid-syllable);
- takes whose speech starts at the first audio frame are treated as onset-risky;
- the last character must be heard ("接着看他的终结" re-record), so first/last characters are checked;
- runaway generations (a 4-character line once came back as 36 s) are rejected by duration.
"""
from __future__ import annotations

import hashlib
import logging
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio, moss
from .textnorm import cer, edges_match, pinyin_cer, reading, spoken_len

logger = logging.getLogger(__name__)

LEAD_IN = "[pause 0.3s]"
TAIL_OUT = "[pause 0.3s]"
PRE_ROLL = 0.08
POST_ROLL = 0.15
MAX_CER = 0.05
SHORT = 5           # lines up to this many characters are read back doubled


@dataclass
class Take:
    path: str
    k: int
    on: float = 0.0
    off: float = 0.0
    lead: float = 0.0
    heard: str = ""
    cer: float = 1.0
    py_cer: float = 1.0
    onset_ok: bool = False
    edges_ok: bool = False
    score: float = math.inf
    ok: bool = False
    reasons: list = field(default_factory=list)

    @property
    def speech(self) -> float:
        return self.off - self.on


PAUSE_ALLOWANCE = 0.6   # the lead-in and tail pause tokens count toward expected_duration_sec
SPEECH_RATE = 5.4       # chars/s requested from TTS; reference delivery measures 5.75 c/s


def expected_duration(text: str, window: float) -> float:
    """Expected speech length (without the pause tokens). Requesting less than natural length
    squeezes short exclamations into fillers and garbled syllables (0/8 good takes vs 5/8)."""
    return round(max(0.6, min(window, spoken_len(text) / SPEECH_RATE)), 2)


def _analyze(take: Take, text: str, expected: float, window: float, keyterms: list[str], cache: Path) -> Take:
    x = audio.load(take.path)
    take.on, take.off = audio.speech_edges(x)
    take.lead = audio.lead_silence(x, take.on)
    if take.speech <= 0 or take.speech > 2.2 * expected + 0.8 or take.speech < 0.35 * expected:
        take.reasons.append(f"runaway duration {take.speech:.2f}s for expected {expected:.2f}s")
        return take
    seg = x[max(0, int((take.on - PRE_ROLL) * audio.SR)): int((take.off + POST_ROLL) * audio.SR)]
    if spoken_len(text) <= SHORT:
        # ASR often returns nothing for a 1-2 s clip; it hears the take twice in a row instead.
        seg = np.concatenate([seg, np.zeros(int(0.35 * audio.SR), dtype=seg.dtype), seg])
        text = text + "，" + text
    wav = audio.tmp_wav(seg)
    try:
        take.heard = moss.asr(wav, keyterms, cache)
    finally:
        wav.unlink(missing_ok=True)
    take.cer = cer(text, take.heard)
    take.py_cer = pinyin_cer(text, take.heard)
    # TTS output has a -45 dB floor, not digital silence: the onset counts as complete when
    # speech does not start in the first 30 ms and the first syllable is heard.
    take.onset_ok = take.on >= 0.03
    take.edges_ok = edges_match(text, take.heard)
    tail_room = len(x) / audio.SR - take.off
    if not take.onset_ok:
        take.reasons.append("speech starts at file start (onset risk)")
    if not take.edges_ok:
        take.reasons.append(f"first/last char mismatch: heard {take.heard!r}")
    if tail_room < 0.03:
        take.reasons.append("speech runs to file end")
    fit = take.speech / max(window, 0.3)
    take.score = (6 * min(take.cer, take.py_cer + 0.02) + (0 if take.onset_ok else 0.6) + (0 if take.edges_ok else 1.0)
                  + (0.5 if tail_room < 0.03 else 0) + 1.5 * max(0.0, fit - 1.0) + 0.3 * abs(math.log(take.speech / expected)))
    take.ok = min(take.cer, take.py_cer) <= MAX_CER and take.edges_ok and take.onset_ok and tail_room >= 0.03 and fit <= 1.2
    return take


def synth_line(text: str, window: float, workdir: Path, keyterms: list[str] | None = None,
               first: int = 3, most: int = 5, jobs: int = 4) -> tuple[Take | None, list[Take]]:
    """Synthesize takes until one passes or `most` takes are spent. Returns (best, all)."""
    keyterms = keyterms or []
    expected = expected_duration(text, window)
    tts_text = LEAD_IN + reading(text) + TAIL_OUT
    key = hashlib.sha1(f"{tts_text}|{expected + PAUSE_ALLOWANCE}".encode()).hexdigest()[:12]
    cache = workdir / "cache"
    takes: list[Take] = []

    def make(k: int) -> Take:
        path = workdir / "takes" / f"{key}_{k}.wav"
        for attempt in range(2):
            if path.exists() and not audio.wav_ok(path):
                # Left truncated by a killed run or a disk outage: drop it and synthesize again.
                logger.warning("discarding undecodable cached take %s", path.name)
                path.unlink(missing_ok=True)
            if not path.exists():
                moss.tts(tts_text, path, expected + PAUSE_ALLOWANCE)
            try:
                return _analyze(Take(str(path), k), text, expected, window, keyterms, cache)
            except subprocess.CalledProcessError:
                if attempt:
                    raise
                path.unlink(missing_ok=True)
        raise AssertionError("unreachable")

    k = 0
    with ThreadPoolExecutor(jobs) as pool:
        batch = first
        while k < most:
            n = min(batch, most - k)
            takes.extend(pool.map(make, range(k, k + n)))
            k += n
            if any(t.ok for t in takes):
                break
            batch = 2
        top = min(takes, key=lambda t: t.score)
        if not top.ok and top.cer <= MAX_CER and top.edges_ok and not top.onset_ok:
            # Only the onset rule failed: two more takes, as with the hand re-record of "队友".
            takes.extend(pool.map(make, range(k, k + 2)))
    ranked = sorted(takes, key=lambda t: (not t.ok, t.score))
    best: Take | None = ranked[0] if ranked and math.isfinite(ranked[0].score) else None
    return best, takes


def cut(take: Take) -> tuple[np.ndarray, float]:
    """Speech with up to PRE_ROLL before the onset and POST_ROLL after the offset.
    Returns (samples, onset offset inside the samples in seconds)."""
    x = audio.load(take.path)
    a = max(0, int((take.on - PRE_ROLL) * audio.SR))
    b = min(len(x), int((take.off + POST_ROLL) * audio.SR))
    return audio.fade(x[a:b], fin=0.004, fout=0.04), take.on - a / audio.SR
