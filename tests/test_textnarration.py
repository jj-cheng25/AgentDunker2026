import json

from agentdunker import config, textnarration


def test_template_lines_without_any_provider(tmp_path):
    config.override(llm={"provider": "none"}, voice={"backend": "none"})
    doc = {"duration": 12.0,
           "sections": [{"kind": "live", "start": 0, "end": 6}, {"kind": "bridge", "start": 6, "end": 8},
                        {"kind": "replay", "start": 8, "end": 12}],
           "events": [{"t": 1.0, "type": "pass", "actors": [1, 2], "conf": 0.9},
                      {"t": 4.0, "type": "shot", "actors": [2], "conf": 0.9, "made": True}],
           "shot": {"release_t": 4.0, "shooter_track": 2, "made": True, "xfg": None}}
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps(doc))
    out = tmp_path / "n" / "narration.json"
    res = textnarration.write_lines(ev, None, out)
    assert res["lines"] and all(l["text"] and l["audio"] == "" for l in res["lines"])
    assert {l["origin"] for l in res["lines"]} <= {"template", "fixed"}   # no model without a provider
    assert all(l["start"] < l["end"] for l in res["lines"])
    assert res["quality"]["voiced"] is False
    assert json.loads(out.read_text())["voice"]["backend"] == "none"
