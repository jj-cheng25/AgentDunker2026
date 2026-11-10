"""Intro / outro / cover and the data-driven "3D 战术解读" transition (Canvas + headless Chromium).

The transition page is the hackathon ``transition.html`` with the freeze frame and fonts injected; its
animation is 4.4 s and is time-stretched to the duration the director asks for.
"""
from __future__ import annotations

import base64
import glob
import hashlib
import io
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..media import Encoder, FPS, read_wav

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).resolve().parents[1] / "assets"
PROMO = ASSETS / "promo"
FONTS = __import__("agentdunker.weights", fromlist=["x"]).fonts_dir()
TRANSITION_NATIVE_S = 4.4


def chromium_executable() -> str | None:
    """A Chromium binary: $ADK_CHROMIUM, else the newest Playwright-managed build under
    $PLAYWRIGHT_BROWSERS_PATH, ~/.cache/ms-playwright or the shared ``tools/ms-playwright`` next to the repo."""
    from agentdunker.weights import chromium_path
    if chromium_path():
        return chromium_path()
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH"), Path.home() / ".cache/ms-playwright",
             Path(__file__).resolve().parents[3] / "tools/ms-playwright"]
    for root in filter(None, roots):
        found = sorted(glob.glob(str(Path(root) / "chromium-*/chrome-linux64/chrome")))
        if found:
            return found[-1]
    return None


def chromium_path_dir(work: Path) -> str | None:
    """A directory holding a ``chromium`` symlink (for modules that look it up on PATH), or None."""
    exe = chromium_executable()
    if not exe or shutil.which("chromium"):
        return None
    d = work / ".bin"
    d.mkdir(parents=True, exist_ok=True)
    link = d / "chromium"
    if not link.exists():
        link.symlink_to(exe)
    return str(d)


def _render_page(page: Path, times: list[float], enc: Optional[Encoder] = None, keep: tuple = ()) -> dict:
    """Drive a Canvas page (``ready()``, ``render(t)``) in headless Chromium; frames go to ``enc``.

    Returns {index: RGB array} for the frame indices listed in ``keep``.
    """
    from playwright.sync_api import sync_playwright

    kept = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_executable(),
                                    args=["--no-sandbox", "--allow-file-access-from-files",
                                          *__import__("agentdunker.weights", fromlist=["x"]).CHROMIUM_QUIET_FLAGS])
        pg = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        pg.goto(page.as_uri())
        pg.evaluate("ready()")
        for i, t in enumerate(times):
            pg.evaluate(f"render({t:.5f})")
            data = pg.evaluate("document.getElementById('c').toDataURL('image/png')")
            img = np.asarray(Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1]))).convert("RGB"))
            if enc is not None:
                enc.write(img)
            if i in keep:
                kept[i] = img
        browser.close()
    return kept


def _page(name: str, work: Path, **subs: str) -> Path:
    html = (PROMO / f"{name}.html").read_text().replace("__FONTS__", FONTS.as_uri())
    for k, v in subs.items():
        html = html.replace(k, v)
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{name}.html"
    out.write_text(html)
    return out


def render_transition(freeze_rgb: np.ndarray, out: Path, duration: float, work: Path) -> Path:
    """Silent 1080p30 transition clip of ``duration`` seconds starting on ``freeze_rgb``."""
    work.mkdir(parents=True, exist_ok=True)
    Image.fromarray(freeze_rgb).save(work / "freeze.png")
    page = _page("transition", work, __FREEZE__=(work / "freeze.png").as_uri())
    n = max(1, round(duration * FPS))
    enc = Encoder(out)
    try:
        _render_page(page, [i / FPS * TRANSITION_NATIVE_S / duration for i in range(n)], enc)
    except BaseException:
        enc.proc.kill()
        raise
    return enc.close()


def transition_sfx(work: Path, duration: float) -> np.ndarray:
    """Transition sound bed (NumPy synth from the hackathon), stretched to ``duration`` (+0.8 s tail)."""
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "tr_sfx.wav"
    if not wav.exists():
        shutil.copy(PROMO / "tr_sound.py", work / "tr_sound.py")
        subprocess.run([sys.executable, "tr_sound.py"], cwd=work, check=True, capture_output=True, timeout=300)
    audio = read_wav(wav)
    rate = TRANSITION_NATIVE_S / duration
    if abs(rate - 1) > 0.02:
        idx = np.clip(np.arange(int(len(audio) / rate)) * rate, 0, len(audio) - 1).astype(int)
        audio = audio[idx]
    return audio


COVER_FRAME = 102          # the cover is outro frame 102 (title, subtitle, slogan and logos all visible)
COVER_HOLD_S = 0.25        # the intro opens on the cover


def _cache_dir() -> Path:
    h = hashlib.sha1()
    for f in ("intro.html", "outro.html", "intro_sound.py", "outro_sound.py"):
        h.update((PROMO / f).read_bytes())
    for f in sorted(FONTS.glob("*.ttf")):
        h.update(f"{f.name}:{f.stat().st_size}".encode())
    root = Path(os.environ.get("ADK_CACHE", Path.home() / ".cache" / "agentdunker"))
    return root / "promo" / h.hexdigest()[:12]


def build_fixed() -> Path:
    """Render intro/outro (6 s, 1080p30) and the cover once from the project sources; cached by source hash."""
    d = _cache_dir()
    if (d / ".complete").exists() and all((d / f).exists() for f in ("intro.mp4", "outro.mp4", "cover.png",
                                                                     "intro_score.wav", "outro_score.wav")):
        return d
    final, d = d, d.with_name(f"{d.name}.part{os.getpid()}")    # render beside the cache, rename when complete
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    times = [i / FPS for i in range(int(6 * FPS))]
    enc = Encoder(d / "outro.mp4")
    cover = _render_page(_page("outro", d), times, enc, keep=(COVER_FRAME,))[COVER_FRAME]
    enc.close()
    Image.fromarray(cover).save(d / "cover.png")
    enc = Encoder(d / "intro.mp4")
    n_hold = round(COVER_HOLD_S * FPS)
    for _ in range(n_hold):
        enc.write(cover)
    _render_page(_page("intro", d), times[n_hold:], enc)
    enc.close()
    for name in ("intro", "outro"):
        shutil.copy(PROMO / f"{name}_sound.py", d / f"{name}_sound.py")
        subprocess.run([sys.executable, f"{name}_sound.py"], cwd=d, check=True, capture_output=True, timeout=600)
    (d / ".complete").write_text("ok\n")
    shutil.rmtree(final, ignore_errors=True)
    os.replace(d, final)
    return final


def fixed_clip(name: str) -> Path:
    """Intro/outro picture (6 s), rendered from ``assets/promo/<name>.html`` on first use."""
    return build_fixed() / f"{name}.mp4"


def fixed_audio(name: str) -> Path:
    """Intro score is 8 s (its last 2 s play under the start of the live segment); outro 6 s."""
    try:
        return build_fixed() / f"{name}_score.wav"
    except Exception as e:  # noqa: BLE001  (no Chromium / fonts: the film goes on without intro and outro)
        logger.warning("intro/outro unavailable: %s", str(e).splitlines()[0][:200])
        return None


def cover() -> Path:
    try:
        return build_fixed() / "cover.png"
    except Exception:  # noqa: BLE001  (compose() embeds the cover only if it exists)
        return Path("/nonexistent/cover.png")
