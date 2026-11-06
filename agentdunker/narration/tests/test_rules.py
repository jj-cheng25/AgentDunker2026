"""Offline checks of the narration rules (no API calls)."""
import json
from pathlib import Path

import numpy as np

from agentdunker.narration import align, mix, script, voice
from agentdunker.narration.textnorm import bad_names, bad_numbers, per_line_errors, pinyin_cer, reading

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def test_reading_and_homophone_cer():
    assert reading("命中率只有21.3%") == "命中率只有百分之二十一点三"
    assert pinyin_cer("清空强侧", "清空墙侧") == 0.0


def test_fact_guards():
    assert bad_names("库里三分出手", set(), {"库里"}) == ["库里"]
    assert bad_names("杜兰特拉开单打", {"杜兰特"}, set()) == []
    assert bad_names("弗拉格突破", set(), set()) == ["弗拉格"]
    assert bad_numbers("砍下30分", set()) == ["30"]
    assert bad_numbers("预期命中率只有21.3%", {"21.3%"}) == []


def test_check_line_rejects_invented_details_and_wrong_result():
    doc = json.loads((FIX / "c06_lal_at_gsw.events.json").read_text())
    f = script.facts(doc)
    slot = script.Slot(0, 0.0, 2.0, [], "event", 10)
    assert any("空心" in e for e in script.check_line("空心入网！", slot, f, set()))
    assert script.check_line("三分命中！", slot, f, set()) == []
    miss = script.facts(json.loads((FIX / "c02_dal_at_hou.events.json").read_text()))
    assert script.check_line("打进了！", slot, miss, set())


def test_slots_reference_layout():
    doc = json.loads((FIX / "ref_film.events.json").read_text())
    slots = script.build_slots(doc)
    kinds = [s.kind for s in slots]
    assert kinds.count("bridge") == 1 and kinds.count("xfg") == 1
    assert all(s.window >= script.MIN_WINDOW for s in slots if s.kind == "event")
    assert all(a.end + script.GAP <= b.start + 1e-6 for a, b in zip(slots, slots[1:]))
    f = script.facts(doc)
    for s in slots:   # the no-model fallback always yields a line that passes the guards
        line = script.template_line(s, f)
        assert not line or not script.check_line(line, s, f, set())


def test_per_line_errors_assigns_to_lines():
    assert per_line_errors(["队友，拉开！", "回传杜兰特。"], "队友拉开回传杜兰特") == [(0, 4), (0, 5)]
    assert per_line_errors(["拔起三分！"], "拔起三分啊") == [(1, 4)]


def test_overflow_shortens_before_speeding(monkeypatch, tmp_path):
    """A take too long for its window is reworded shorter, then fitted at <= MAX_TEMPO."""
    slot = script.Slot(0, 1.0, 2.5, [{"t": 1.1}], "event", 6, "防守完全被撕开了！", "model")

    def fake_synth(text, window, workdir, keyterms):
        t = voice.Take("fake.wav", 0, on=0.1, off=0.1 + len(text) * 0.25, ok=True, score=0.0)
        return t, [t]

    monkeypatch.setattr(voice, "synth_line", fake_synth)
    monkeypatch.setattr(script, "shorten", lambda text, n, *a, **k: "撕开了！")
    monkeypatch.setattr(align, "_render_line", lambda p, w, k: None)
    placed = align.place_all([slot], {}, set(), tmp_path, [])
    p = placed[0]
    assert p.text == "撕开了！" and p.rewrites and p.tempo <= align.MAX_TEMPO and p.in_window


def test_duck_envelope_levels():
    env = mix.duck_envelope([(1.0, 2.0)], 48000 * 3, bed_db=-6.0, duck_db=-20.0)
    db = 20 * np.log10(env)
    assert abs(db[int(1.5 * 48000)] + 20) < 0.1 and abs(db[int(0.2 * 48000)] + 6) < 0.1
