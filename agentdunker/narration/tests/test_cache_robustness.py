"""Truncated or corrupt cache files must be re-made, not crash the run (no API calls)."""
import json
from pathlib import Path

import numpy as np
import pytest

from agentdunker.narration import audio, files, moss, voice

TEXT = "关键时刻进攻方持球"


def _speech_like(seconds: float = 1.6) -> np.ndarray:
    t = np.arange(int(seconds * audio.SR)) / audio.SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    pad = np.zeros(int(0.3 * audio.SR), dtype=np.float32)
    return np.concatenate([pad, tone, pad])


@pytest.fixture
def fake_moss(monkeypatch):
    calls = []

    def tts(text, out, expected=None):
        calls.append(Path(out).name)
        audio.save(out, _speech_like())
        return {}

    monkeypatch.setattr(moss, "tts", tts)
    monkeypatch.setattr(moss, "asr", lambda path, keyterms=None, cache_dir=None: TEXT)
    return calls


def test_wav_ok_rejects_truncated_and_garbage(tmp_path):
    good = audio.save(tmp_path / "good.wav", _speech_like())
    assert audio.wav_ok(good)
    raw = good.read_bytes()
    for name, data in {"half": raw[: len(raw) // 2], "header": raw[:44], "empty": b"", "junk": b"RIFF" + b"\0" * 40}.items():
        (tmp_path / f"{name}.wav").write_bytes(data)
        assert not audio.wav_ok(tmp_path / f"{name}.wav"), name
    assert not audio.wav_ok(tmp_path / "missing.wav")


def test_corrupt_cached_takes_are_resynthesized(tmp_path, fake_moss):
    best, takes = voice.synth_line(TEXT, 3.0, tmp_path)
    assert best is not None and best.ok and len(fake_moss) == 3
    # The c01 case: takes _0 and _2 left truncated by a killed run / disk outage.
    by_k = {t.k: Path(t.path) for t in takes}
    by_k[0].write_bytes(by_k[0].read_bytes()[:1000])
    by_k[2].write_bytes(b"")
    fake_moss.clear()
    best, takes = voice.synth_line(TEXT, 3.0, tmp_path)
    assert best is not None and best.ok
    assert sorted(fake_moss) == sorted([by_k[0].name, by_k[2].name])   # only the bad ones are re-made
    assert all(audio.wav_ok(t.path) for t in takes)


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "out.wav"
    with pytest.raises(RuntimeError), files.atomic_path(target) as tmp:
        tmp.write_bytes(b"RIFF partial")
        raise RuntimeError("killed")
    assert not target.exists() and list(tmp_path.iterdir()) == []
    audio.save(target, _speech_like())
    assert audio.wav_ok(target) and [p.name for p in tmp_path.iterdir()] == ["out.wav"]


def test_tts_does_not_keep_an_undecodable_response(tmp_path, monkeypatch):
    class Resp:
        content = b"RIFF" + b"\0" * 100
        headers = None

    monkeypatch.setattr(moss, "_post", lambda *a, **k: Resp())
    monkeypatch.setattr(moss.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        moss.tts("测试", tmp_path / "t.wav")
    assert list(tmp_path.iterdir()) == []


def test_corrupt_json_cache_is_a_miss(tmp_path):
    p = tmp_path / "asr_x.json"
    p.write_text('{"text": "半')
    assert files.read_json(p) is None
    files.write_text(p, json.dumps({"text": "好"}, ensure_ascii=False))
    assert files.read_json(p) == {"text": "好"}


def test_moss_concurrency_capped_at_three():
    assert 1 <= moss.MOSS_CONCURRENCY <= 3
