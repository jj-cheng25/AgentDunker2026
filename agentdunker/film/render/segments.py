"""Render each EDL segment to a silent 1080p30 H.264 clip."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from ..director import map_fwd
from ..media import Encoder, FrameReader, FPS, fit_1080
from .finish import FinishEffects
from .freeze import FreezeRenderer, FreezeSpec
from .nameplates import NameplateLayer
from .style import font

logger = logging.getLogger(__name__)


def _white(rgb: np.ndarray, a: float) -> np.ndarray:
    if a <= 0:
        return rgb
    return (rgb.astype(np.float32) * (1 - a) + 255 * a).astype(np.uint8)


def _black(rgb: np.ndarray, a: float) -> np.ndarray:
    if a <= 0:
        return rgb
    return (rgb.astype(np.float32) * (1 - min(1.0, a))).astype(np.uint8)


def render_live(video: str, seg: dict, out: Path, fps_src: float, plates: Optional[NameplateLayer]) -> dict:
    n = round((seg["film"][1] - seg["film"][0]) * FPS)
    s0 = seg["source"][0]
    reader = FrameReader(video)
    enc = Encoder(out)
    if plates is not None:
        plates.plan(n / FPS + s0)
    try:
        for k in range(n):
            t_src = s0 + k / FPS * seg.get("speed", 1.0)
            img = reader.get(round(t_src * fps_src))
            if plates is not None:
                base = Image.fromarray(img).convert("RGBA")
                base.alpha_composite(plates.frame(t_src))
                img = np.asarray(base.convert("RGB"))
            if k / FPS < 0.25:                                # the intro's white flash fades out
                img = _white(img, 1 - k / FPS / 0.25)
            enc.write(img)
    finally:
        reader.close()
    enc.close()
    return {"frames": n, "nameplates": plates.stats if plates else None}


def _subtitle_strip(text: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Pre-rendered subtitle band: (rgb, alpha 0–1, top y), composited only over its own rows."""
    f = font(28, 500)                      # reference: ≈ 26–28 px white on a translucent black bar
    y0, h = 1080 - 92, 46
    img = Image.new("RGBA", (1920, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = d.textlength(text, font=f)
    x0 = int((1920 - w) / 2 - 20)
    img.alpha_composite(Image.new("RGBA", (int(w + 40), h), (0, 0, 0, 140)), (max(0, x0), 0))
    ImageDraw.Draw(img).text((960, h // 2), text, font=f, fill=(255, 255, 255, 255), anchor="mm")
    a = np.asarray(img).astype(np.float32)
    return a[..., :3], a[..., 3:] / 255.0, y0


def render_replay(m4_video: str, m4_knots: list[list[float]], seg: dict, out: Path,
                  subtitles: list[tuple[float, float, str]]) -> dict:
    """Re-time M4's render onto the EDL time map, keyed on source PTS (its PIP stays in sync)."""
    n = round((seg["film"][1] - seg["film"][0]) * FPS)
    src_k, out_k = [k[1] for k in m4_knots], [k[0] for k in m4_knots]
    inc = [0] + [i for i in range(1, len(src_k)) if src_k[i] > src_k[i - 1] + 1e-6]
    hold_from = seg["time_map"][-2][0]
    strips = [(a, b, _subtitle_strip(text)) for a, b, text in subtitles]
    reader = FrameReader(m4_video)
    enc = Encoder(out)
    dup, last = 0, -1
    try:
        for k in range(n):
            u = k / FPS
            if u > hold_from:                                  # end hold: M4's own last frames
                t_out = out_k[inc[-1]] + (u - hold_from)
            else:
                src = map_fwd(seg["time_map"], u)
                t_out = float(np.interp(src, [src_k[i] for i in inc], [out_k[i] for i in inc]))
            idx = round(t_out * FPS)
            if reader.n:
                idx = min(reader.n - 1, idx)
            dup += idx == last
            last = idx
            rgb = reader.get(idx)
            for a, b, (srgb, salpha, y0) in strips:
                if a <= u < b:
                    k = min(1.0, (u - a) / 0.2, (b - u) / 0.2) * salpha
                    rgb = rgb.copy()
                    band = rgb[y0:y0 + srgb.shape[0]].astype(np.float32)
                    rgb[y0:y0 + srgb.shape[0]] = (band * (1 - k) + srgb * k).astype(np.uint8)
            if u < 0.35:
                rgb = _white(rgb, 1 - u / 0.35)
            enc.write(rgb)
    finally:
        reader.close()
    enc.close()
    return {"frames": n, "repeated_frames": dup}


def render_freeze(seg: dict, out: Path, spec: FreezeSpec) -> dict:
    n = round((seg["film"][1] - seg["film"][0]) * FPS)
    r = FreezeRenderer(spec)
    enc = Encoder(out)
    for k in range(n):
        enc.write(r.frame(k / FPS, n / FPS))
    enc.close()
    return {"frames": n, **r.layout}


def render_resume(video: str, seg: dict, out: Path, fps_src: float, fx: Optional[FinishEffects]) -> dict:
    n = round((seg["film"][1] - seg["film"][0]) * FPS)
    fade = seg.get("fade_out", 0.8)
    reader = FrameReader(video)
    enc = Encoder(out)
    try:
        for k in range(n):
            u = k / FPS
            t_src = map_fwd(seg["time_map"], u)
            img = reader.get(round(t_src * fps_src))
            if fx is not None:
                img = fx.apply(img, t_src, u)
            enc.write(_black(img, (u - (n / FPS - fade)) / fade))
    finally:
        reader.close()
    enc.close()
    return {"frames": n, "effects": fx.stats if fx else None}


def render_fixed(src: Path, out: Path) -> dict:
    """Re-encode a fixed asset with the common encoder so every segment concatenates cleanly."""
    reader = FrameReader(src)
    enc = Encoder(out)
    try:
        for k in range(reader.n):
            enc.write(fit_1080(reader.get(k)))
    finally:
        reader.close()
    enc.close()
    return {"frames": reader.n}
