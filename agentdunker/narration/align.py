"""Place chosen takes on the timeline.

Automatic versions of the hand edits in the reference (demo03 edits.json):
- shift: a line's speech onset goes to its event time minus a small lead;
- fit_before / place_after: lines never overlap; a breath gap (>= GAP) is kept, and a line
  may borrow up to SPILL seconds from the next window when the next line has slack;
- stretch 0.8 on a long drawl: very slow short phrases are tightened (MIN_RATE);
- speed 1.2: overflow is fixed by shorter wording first, then tempo <= MAX_TEMPO (pitch kept);
- delete: a line that still cannot fit after two rewrites is dropped instead of failing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio, moss, script, voice
from .textnorm import cer, pinyin_cer, spoken_len

logger = logging.getLogger(__name__)

GAP = 0.15
SPILL = 0.5
MAX_TEMPO = 1.2
MIN_RATE = 3.0      # chars/s; slower short phrases are tightened (reference "拉开" 0.8x)
MAX_DRAWL_FIX = 1.25


@dataclass
class Placed:
    slot: script.Slot
    text: str
    take: voice.Take | None = None
    start: float = 0.0           # speech onset on the timeline
    end: float = 0.0             # speech offset on the timeline
    tempo: float = 1.0
    in_window: bool = True
    final_heard: str = ""
    final_cer: float | None = None
    final_py_cer: float | None = None
    rewrites: list = field(default_factory=list)
    dropped: str = ""
    audio: str = ""
    samples: np.ndarray | None = None
    onset_in_samples: float = 0.0


def _rate(text: str, take: voice.Take) -> float:
    return spoken_len(text) / max(take.speech, 1e-3)


def _slack(p: Placed) -> float:
    if p.take is None:
        return p.slot.window
    return max(0.0, p.slot.window - p.take.speech / MAX_TEMPO)


def place_all(slots: list[script.Slot], facts: dict, roster: set[str], workdir: Path,
              keyterms: list[str], jobs: int = 6) -> list[Placed]:
    from concurrent.futures import ThreadPoolExecutor

    placed = [Placed(s, s.text) for s in slots if s.text]
    with ThreadPoolExecutor(max(1, jobs // 2)) as pool:
        results = list(pool.map(lambda p: voice.synth_line(p.text, p.slot.window, workdir, keyterms), placed))
    for p, (best, _takes) in zip(placed, results):
        p.take = best
        if best is not None and not best.ok and p.slot.kind != "bridge" and spoken_len(p.text) >= 4:
            # No take passed (misheard homophone such as 强侧/墙侧, clipped onset, runaway takes): reword once;
            # the new wording is used only if one of its takes passes every check.
            hint = (f"语音识别把“{p.text}”听成了“{best.heard}”，请换掉被听错的词" if best.cer > voice.MAX_CER
                    else "这句合成不稳定，请换一种说法，保持同样意思和长度")
            alt = script.shorten(p.text, max(p.slot.budget, spoken_len(p.text)), p.slot, facts, roster, hint=hint)
            p.rewrites.append({"from": p.text, "to": alt, "reason": "asr-mismatch"})
            if alt:
                alt_best, _ = voice.synth_line(alt, p.slot.window, workdir, keyterms)
                if alt_best is not None and alt_best.ok:
                    p.text, p.take = alt, alt_best
    prev_end = -1.0
    for i, p in enumerate(placed):
        nxt = placed[i + 1] if i + 1 < len(placed) else None
        for attempt in range(3):
            if p.take is None:
                break
            start = max(p.slot.start, prev_end + GAP)
            hard_section = p.slot.kind in ("bridge", "xfg")
            spill = 0.0 if hard_section or nxt is None else min(SPILL, _slack(nxt))
            avail = p.slot.end + spill - start
            speech = p.take.speech
            drawl = 1.0
            if _rate(p.text, p.take) < MIN_RATE and speech > 0.6:
                drawl = min(MAX_DRAWL_FIX, MIN_RATE / _rate(p.text, p.take))
            tempo = max(drawl, speech / max(avail, 1e-3))
            if tempo <= MAX_TEMPO + 1e-6 or attempt == 2:
                p.start, p.tempo = start, min(max(tempo, 1.0), MAX_TEMPO)
                p.end = start + speech / p.tempo
                p.in_window = p.end <= p.slot.end + spill + 1e-3
                break
            # Overflow: shorter wording at the measured speaking rate, then re-synthesize.
            max_chars = int(avail * MAX_TEMPO * _rate(p.text, p.take) * 0.95)
            short = script.shorten(p.text, max_chars, p.slot, facts, roster) if max_chars >= 2 else ""
            p.rewrites.append({"from": p.text, "to": short, "avail": round(avail, 2), "max_chars": max_chars})
            if not short:
                p.take, p.dropped = None, f"no wording fits {avail:.2f}s"
                break
            if spoken_len(short) >= spoken_len(p.text):
                p.start, p.tempo = start, MAX_TEMPO
                p.end = start + speech / MAX_TEMPO
                p.in_window = p.end <= p.slot.end + spill + 1e-3
                break
            p.text = short
            p.take, _ = voice.synth_line(short, avail, workdir, keyterms)
        if p.take is None:
            p.dropped = p.dropped or "no usable take"
            continue
        prev_end = p.end
    kept = [p for p in placed if p.take is not None]
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(lambda p: _render_line(p, workdir, keyterms), kept))
    return placed


def _render_line(p: Placed, workdir: Path, keyterms: list[str]) -> None:
    assert p.take is not None
    seg, onset = voice.cut(p.take)
    if p.tempo > 1.0 + 1e-3:
        seg = audio.tempo(seg, p.tempo)
        onset /= p.tempo
    p.samples, p.onset_in_samples = seg, onset
    out = audio.save(workdir / "lines" / f"line_{p.slot.idx:02d}.wav", seg)
    ref = p.text
    if spoken_len(ref) <= voice.SHORT:   # same doubled read-back as take selection
        seg = np.concatenate([seg, np.zeros(int(0.35 * audio.SR), dtype=seg.dtype), seg])
        ref = ref + "，" + ref
    wav = audio.tmp_wav(seg)
    try:
        p.final_heard = moss.asr(wav, keyterms, workdir / "cache")
    finally:
        wav.unlink(missing_ok=True)
    p.final_cer = cer(ref, p.final_heard)
    p.final_py_cer = pinyin_cer(ref, p.final_heard)
    p.audio = str(out)


def render_track(placed: list[Placed], duration: float, target_lufs: float = -20.0) -> np.ndarray:
    n = int(duration * audio.SR) + 1
    track = np.zeros(n, dtype=np.float32)
    for p in placed:
        if p.samples is None:
            continue
        a = int(round((p.start - p.onset_in_samples) * audio.SR))
        seg = p.samples[max(0, -a):][: max(0, n - max(a, 0))]
        a = max(a, 0)
        track[a: a + len(seg)] += seg
    # Per-track loudness: scale speech to roughly target_lufs using active-speech RMS.
    db = audio.frame_db(track)
    active = db > -45
    if active.any():
        rms_db = 10 * np.log10(np.mean(10 ** (db[active] / 10)))
        track *= 10 ** ((target_lufs + 3 - rms_db) / 20)
    peak = np.abs(track).max()
    if peak > 0.89:
        track *= 0.89 / peak
    return track


def retake(p: Placed, placed: list[Placed], facts: dict, roster: set[str], workdir: Path,
           keyterms: list[str], heard: str) -> bool:
    """One reword + retake for a line the whole-track read-back got wrong. Keeps the start time;
    the new line must end before the next line (minus GAP) at <= MAX_TEMPO."""
    hint = f"语音识别在整轨里把“{p.text}”听成了“{heard}”，请换掉被听错的词"
    alt = script.shorten(p.text, max(p.slot.budget, spoken_len(p.text)), p.slot, facts, roster, hint=hint)
    p.rewrites.append({"from": p.text, "to": alt, "reason": "track-asr-mismatch"})
    if not alt:
        return False
    later = [q.start for q in placed if q.samples is not None and q.start > p.start]
    limit = (min(later) - GAP) if later else p.slot.end
    best, _ = voice.synth_line(alt, limit - p.start, workdir, keyterms)
    if best is None or not best.ok:
        return False
    tempo = max(1.0, best.speech / max(limit - p.start, 1e-3))
    if tempo > MAX_TEMPO:
        return False
    p.text, p.take, p.tempo = alt, best, tempo
    p.end = p.start + best.speech / tempo
    p.in_window = p.end <= limit + 1e-3
    _render_line(p, workdir, keyterms)
    return True
