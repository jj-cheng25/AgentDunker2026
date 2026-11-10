"""Fixes from the clean-install test of the public repo (2026-11-09)."""
import hashlib
import http.server
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from agentdunker import cli, config, weights


# ---------------------------------------------------------------- 4. downloads resume and retry
class _Flaky(http.server.BaseHTTPRequestHandler):
    """Serves DATA with Range support; the first full request is cut after half the bytes."""
    DATA = os.urandom(300_000)
    cuts = 1
    requests: list = []

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        body = self.DATA[start:]
        _Flaky.requests.append(start)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if _Flaky.cuts and not rng:
            _Flaky.cuts -= 1
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.connection.shutdown(2)          # drop the connection mid-file
            return
        self.wfile.write(body)

    def log_message(self, *_):
        pass


@pytest.fixture
def flaky_server():
    _Flaky.cuts, _Flaky.requests = 1, []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/w.bin"
    srv.shutdown()


def test_download_resumes_after_dropped_connection(tmp_path, flaky_server, monkeypatch):
    monkeypatch.setattr(weights.time, "sleep", lambda s: None)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    config.override(paths={"model_dir": str(tmp_path)})
    a = weights.Asset("w.bin", flaky_server, hashlib.sha256(_Flaky.DATA).hexdigest(), len(_Flaky.DATA), "MIT", "test")
    monkeypatch.setattr(weights, "BY_NAME", {"w.bin": a})
    logs = []
    assert weights.download(["w.bin"], log=logs.append) == []
    assert (tmp_path / "w.bin").read_bytes() == _Flaky.DATA
    assert _Flaky.requests[0] == 0 and _Flaky.requests[1] > 0       # second request resumed with Range
    assert not list(tmp_path.glob("*.part"))


def test_download_failure_says_run_again(tmp_path, monkeypatch, capsys):
    config.override(paths={"model_dir": str(tmp_path)})
    a = weights.Asset("x.bin", "http://127.0.0.1:9/x.bin", "0" * 64, 10, "MIT", "test")
    monkeypatch.setattr(weights, "ASSETS", [a])
    monkeypatch.setattr(weights, "BY_NAME", {"x.bin": a})
    monkeypatch.setattr(weights, "RETRIES", 1)
    monkeypatch.setattr(weights.time, "sleep", lambda s: None)
    rc = cli.main(["download-models", "--no-chromium", "--no-prerender"])
    out = capsys.readouterr().out
    assert rc == 1 and "x.bin" in out and "agentdunker download-models` again" in out


def test_checksum_mismatch_is_retried_once(tmp_path, flaky_server, monkeypatch):
    monkeypatch.setattr(weights.time, "sleep", lambda s: None)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    config.override(paths={"model_dir": str(tmp_path)})
    _Flaky.cuts = 0
    a = weights.Asset("w.bin", flaky_server, "f" * 64, len(_Flaky.DATA), "MIT", "test")
    logs = []
    assert weights._get(a, logs.append) == "checksum mismatch"
    assert len(_Flaky.requests) == 2 and any("downloading again" in l for l in logs)


# ---------------------------------------------------------------- 1. Chromium that cannot start
@pytest.fixture
def broken_chrome(tmp_path, monkeypatch):
    exe = tmp_path / "chrome"
    exe.write_text("#!/bin/sh\necho 'chrome: error while loading shared libraries: libnss3.so: cannot open shared "
                   "object file: No such file or directory' >&2\nexit 127\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("ADK_CHROMIUM", str(exe))
    weights._CHECK.clear()
    yield exe
    weights._CHECK.clear()


def test_chromium_check_reports_the_error(broken_chrome):
    ok, detail = weights.chromium_check()
    assert not ok and "127" in detail and "libnss3.so" in detail
    assert "playwright install-deps chromium" in weights.chromium_fix()


def test_doctor_fails_on_broken_chromium(broken_chrome, capsys):
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    line = next(l for l in out.splitlines() if "chromium" in l)
    assert line.startswith("[FAIL]") and "libnss3.so" in line
    assert "install-deps chromium" in out and "not ready" in out


def test_run_warns_loudly_but_still_makes_a_film(broken_chrome, tmp_path, capsys):
    from test_e2e_synthetic import synthetic_clip
    clip = synthetic_clip(tmp_path / "clip.mp4", seconds=4.0)
    rc = cli.main(["run", str(clip), "-o", str(tmp_path / "out"), "--offline",
                   "--skip", "tracks,court,identity,events,replay,narration"])
    err = capsys.readouterr().err
    head = "\n".join(err.splitlines()[:8])
    assert "WARNING: Chromium cannot start" in head and "libnss3.so" in head
    assert "3D replay, intro, outro and transition will be skipped" in head and "install-deps chromium" in head
    assert rc == 0 and (tmp_path / "out" / "film.mp4").exists()
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep["agentdunker"]["chromium"]["ok"] is False
    assert not (tmp_path / "out" / "work" / "tmp").exists()          # private TMPDIR removed


# ---------------------------------------------------------------- 2. no network in --offline
AUDIT = r'''
import sys, socket
log = open(sys.argv[1], "w")
def hook(ev, args):
    if ev in ("socket.connect", "socket.getaddrinfo"):
        log.write(f"{ev} {args[1] if ev == 'socket.connect' else args[0]}\n"); log.flush()
sys.addaudithook(hook)
import agentdunker                       # sets YOLO_OFFLINE before ultralytics is imported
import ultralytics.utils, ultralytics.utils.events
from ultralytics import YOLO
assert not ultralytics.utils.ONLINE and not ultralytics.utils.events.events.enabled
'''


def test_ultralytics_stays_offline(tmp_path):
    log = tmp_path / "net.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith("YOLO_")}
    env["HOME"] = str(tmp_path)
    r = subprocess.run([sys.executable, "-c", AUDIT, str(log)], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    remote = [l for l in log.read_text().splitlines() if "127.0.0.1" not in l and "localhost" not in l]
    assert remote == []
    assert not (tmp_path / ".config" / "Ultralytics").exists()      # settings live in our cache, not ~/.config


def test_offline_flag_blocks_downloads_and_services(monkeypatch):
    from agentdunker import providers
    monkeypatch.setenv("MOSSHUB_API_KEY", "k")
    monkeypatch.setenv("MOSS_API_KEY", "k")
    config.override(run={"offline": True})
    assert providers.offline() and providers.llm_key() is None and providers.moss_key("vl") is None
    with pytest.raises(RuntimeError, match="offline"):
        weights.fetch_for_module("yolo11x.pt", Path("/nonexistent/yolo11x.pt"))


def test_chromium_flags_disable_background_networking():
    for f in ("--disable-background-networking", "--disable-component-update", "--no-first-run",
              "--no-default-browser-check"):
        assert f in weights.CHROMIUM_QUIET_FLAGS
    src = (Path(weights.__file__).parent / "replay" / "browser.py").read_text()
    assert "CHROMIUM_QUIET_FLAGS" in src
    assert "CHROMIUM_QUIET_FLAGS" in (Path(weights.__file__).parent / "film" / "render" / "promo.py").read_text()
