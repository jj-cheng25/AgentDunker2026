"""Headless Chromium (SwiftShader, CPU WebGL) driver for the Three.js holo renderer in ``web/``.

``render_frames(play, times, width, height)`` returns one RGB array per requested frame. The page
exposes ``holo.renderAt(sourceTime, {outputTime})``, a pure function of time, so frames can be split
over several browser processes and rendered in any order.
"""
from __future__ import annotations

import base64
import functools
import http.server
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import websocket
from PIL import Image

logger = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).resolve().parent / "web"
# CJK font for labels, served to the page as /fonts/cjk.otf so the render never depends on system fonts.
CJK_FONT_CANDIDATES = (__import__("agentdunker.weights", fromlist=["x"]).fonts_dir() / "NotoSansSC-Regular.otf",)


def _cjk_font() -> Optional[Path]:
    for p in ([Path(os.environ["ADK_CJK_FONT"])] if os.environ.get("ADK_CJK_FONT") else []) + list(CJK_FONT_CANDIDATES):
        if p.is_file():
            return p
    logger.warning("no CJK font found (set ADK_CJK_FONT); Chinese labels fall back to system fonts")
    return None


class _CDP:
    def __init__(self, url: str) -> None:
        self.ws = websocket.create_connection(url, suppress_origin=True, timeout=300, http_proxy_host=None)
        self.seq = 0
        self.errors: list = []

    def call(self, method: str, params: Optional[dict] = None) -> dict:
        self.seq += 1
        self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("method") == "Runtime.exceptionThrown":
                self.errors.append(msg)
            if msg.get("id") == self.seq:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def evaluate(self, code: str, wait: bool = False):
        r = self.call("Runtime.evaluate", {"expression": code, "returnByValue": True, "awaitPromise": wait})
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r.get("result", {}).get("value")


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, play_bytes: bytes = b"", **kwargs):
        self.play_bytes = play_bytes
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802
        if urllib.parse.urlsplit(self.path).path == "/__play__.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(self.play_bytes)))
            self.end_headers()
            self.wfile.write(self.play_bytes)
            return
        if urllib.parse.urlsplit(self.path).path == "/fonts/cjk.otf":
            font = _cjk_font()
            if font is None:
                self.send_error(404)
                return
            body = font.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "font/otf")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        from agentdunker.weights import serve_web_asset  # downloaded assets (Xbot.glb)
        if not serve_web_asset(self, WEB_ROOT):
            super().do_GET()

    def log_message(self, *_):
        pass


def _chromium() -> str:
    override = os.environ.get("ADK_CHROMIUM")
    if override:
        return override
    for name in ("chromium", "chromium-browser", "google-chrome"):
        path = shutil.which(name)
        if path:
            return path
    from agentdunker.weights import chromium_path
    if chromium_path():
        return chromium_path()
    raise RuntimeError("chromium is required for the 3D replay renderer (agentdunker download-models)")


class _Page:
    """One headless browser with the replay page loaded."""

    def __init__(self, server_port: int, width: int, height: int) -> None:
        self.profile = tempfile.TemporaryDirectory(prefix="adk-replay-")
        self.logs = tempfile.TemporaryFile()
        self.browser = subprocess.Popen(
            [_chromium(), "--headless", "--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-swiftshader",
             "--use-gl=angle", "--use-angle=swiftshader", "--force-device-scale-factor=1",
             "--remote-debugging-port=0", "--user-data-dir=" + self.profile.name,
             f"--window-size={width},{height}", "about:blank"],
            stdout=self.logs, stderr=self.logs)
        port_file = Path(self.profile.name) / "DevToolsActivePort"
        deadline = time.monotonic() + 60
        while not port_file.exists():
            if self.browser.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Chromium did not start")
            time.sleep(0.1)
        port = int(port_file.read_text().splitlines()[0])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        targets = json.load(opener.open(f"http://127.0.0.1:{port}/json", timeout=15))
        self.cdp = _CDP(next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page"))
        self.cdp.call("Runtime.enable")
        self.cdp.call("Page.enable")
        self.cdp.call("Emulation.setDeviceMetricsOverride",
                      {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
        url = f"http://127.0.0.1:{server_port}/index.html?export=1&data=/__play__.json"
        self.cdp.call("Page.navigate", {"url": url})
        deadline = time.monotonic() + 90
        while not self.cdp.evaluate("!!window.holo?.renderAt"):
            if self.cdp.errors or time.monotonic() > deadline:
                raise RuntimeError(f"replay page failed to load: {self.cdp.errors[:2]}")
            time.sleep(0.1)
        self.cdp.evaluate("document.fonts.ready", True)
        self.size = (width, height)

    def frame(self, source_t: float, output_t: float) -> np.ndarray:
        self.cdp.evaluate(f"holo.renderAt({source_t!r}, {{outputTime:{output_t!r}}})", True)
        data = self.cdp.call("Page.captureScreenshot", {"format": "jpeg", "quality": 95})["data"]
        img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        if img.size != self.size:
            img = img.resize(self.size)
        return np.asarray(img)

    def close(self) -> None:
        try:
            self.cdp.ws.close()
        except Exception:  # noqa: BLE001 - best-effort shutdown
            pass
        self.browser.terminate()
        try:
            self.browser.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.browser.kill()
        self.profile.cleanup()


def render_frames(play: dict, times: Sequence[tuple[float, float]], *, width: int = 1920, height: int = 1080,
                  workers: int = 3, on_frame: Optional[Callable[[int, np.ndarray], None]] = None,
                  ) -> Optional[list[np.ndarray]]:
    """Render ``times`` = [(source_t, output_t), ...]. With ``on_frame`` frames are streamed in order."""
    body = json.dumps(play, ensure_ascii=False, allow_nan=False).encode()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Handler, directory=str(WEB_ROOT), play_bytes=body))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    n = len(times)
    workers = max(1, min(workers, n))
    # Strided split keeps the in-order streaming buffer small.
    chunks = [list(range(k, n, workers)) for k in range(workers)]
    results: dict[int, np.ndarray] = {}
    cond = threading.Condition()
    t0 = time.monotonic()

    def work(indices: list[int]) -> None:
        page = None
        try:
            for i in indices:
                for attempt in range(3):   # a stalled/crashed browser (e.g. under heavy load) is restarted
                    try:
                        page = page or _Page(server.server_port, width, height)
                        img = page.frame(*times[i])
                        break
                    except Exception as e:  # noqa: BLE001 - retried, re-raised after the last attempt
                        logger.warning("frame %d attempt %d failed (%s); restarting browser", i, attempt + 1, e)
                        if page:
                            page.close()
                        page = None
                        if attempt == 2:
                            raise
                with cond:
                    results[i] = img
                    cond.notify_all()
        finally:
            if page:
                page.close()

    out: list[np.ndarray] = []
    try:
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(work, c) for c in chunks if c]
            for i in range(n):
                with cond:
                    while i not in results:
                        failed = [f for f in futures if f.done() and f.exception()]
                        if failed:
                            raise failed[0].exception()
                        cond.wait(timeout=1.0)
                    img = results.pop(i)
                if on_frame:
                    on_frame(i, img)
                else:
                    out.append(img)
                if i % 60 == 0:
                    logger.info("rendered %d/%d frames (%.0fs)", i, n, time.monotonic() - t0)
    finally:
        server.shutdown()
        server.server_close()
    return None if on_frame else out
