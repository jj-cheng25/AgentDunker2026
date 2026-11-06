from pathlib import Path

from agentdunker import config, runner, weights


def test_stage_args_preset_and_threads():
    config.override(run={"preset": "fast", "threads": 6})
    a = runner.stage_args("m1", ["clip.mp4", "--threads", "16", "-o", "t.json"])
    assert a.count("--threads") == 1 and a[a.index("--threads") + 1] == "6"
    assert "yolo11m.pt" in a
    assert runner.stage_args("m5", ["v"])[-3:] == ["--no-pose", "--threads", "6"]
    config.override(run={"preset": "quality"})
    assert runner.stage_args("m2", ["v", "-o", "c.json"]) == ["v", "-o", "c.json"]


def test_registry_is_complete():
    names = [a.name for a in weights.ASSETS]
    assert len(names) == len(set(names))
    for a in weights.ASSETS:
        assert len(a.sha256) == 64 and a.size > 0 and a.license and a.url.startswith("https://")
    for p in ("fast", "quality"):                        # each preset has exactly one person detector
        assert sum(1 for a in weights.ASSETS if a.name.startswith("yolo11") and "pose" not in a.name and p in a.presets) == 1


def test_status_and_checksum(tmp_path):
    config.override(paths={"model_dir": str(tmp_path)})
    a = weights.Asset("x.bin", "https://example.invalid/x", "0" * 64, 3, "MIT", "test")
    assert weights.status(a) == "missing"
    (tmp_path / "x.bin").write_bytes(b"abc")
    assert weights.status(a) == "ok"
    assert weights.status(a, verify=True) == "checksum mismatch"


class _H:
    def __init__(self, path):
        self.path, self.sent = path, []

    def send_response(self, c): self.sent.append(c)
    def send_header(self, *a): pass
    def end_headers(self): pass

    class wfile:  # noqa: N801
        data = b""

        @classmethod
        def write(cls, b): cls.data += b


def test_serve_web_asset(tmp_path):
    config.override(paths={"model_dir": str(tmp_path)})
    web = tmp_path / "web"
    web.mkdir()
    assert not weights.serve_web_asset(_H("/models/Xbot.glb"), web)        # not downloaded
    (tmp_path / "Xbot.glb").write_bytes(b"glb")
    h = _H("/models/Xbot.glb")
    assert weights.serve_web_asset(h, web) and h.sent == [200]
    assert not weights.serve_web_asset(_H("/js/main.js"), web)


def test_sync_rules_are_wellformed():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync", Path(__file__).parents[1] / "tools" / "sync_modules.py")
    sync = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["sync"] = sync
    spec.loader.exec_module(sync)
    assert set(sync.EXCLUDE) == set(sync.MODULES)
    for r in sync.RULES:
        assert r.module in sync.MODULES and r.why and r.old != r.new
    assert sync.rewrite_imports("from adk_court import x\nimport adk_tracking") == \
        "from agentdunker.court import x\nimport agentdunker.tracking"
    assert sync.excluded("m7", "assets/fonts/NotoSansSC-VF.ttf") and not sync.excluded("m7", "assets/fonts/OFL-barlow.txt")
