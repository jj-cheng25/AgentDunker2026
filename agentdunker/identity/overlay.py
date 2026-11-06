"""Nameplate preview: a plate above every observed player with number + name (or team + number / team only),
and a header line with the game match result. Written as H.264 when ffmpeg is available."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

COLORS = {"A": (235, 200, 40), "B": (200, 60, 230)}      # BGR: cyan-ish / magenta, as in the M1 overlay


def _short(name: str) -> str:
    parts = name.split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else name


def _label(p: dict) -> tuple[str, bool]:
    """(text, confident). Low-confidence identities show team only."""
    tri = p.get("team_abbr") or p.get("team")
    if p.get("jersey") and p.get("name"):
        return f"#{p['jersey']} {_short(p['name'])}", p.get("conf", 0) >= 0.5
    if p.get("jersey"):
        return f"{tri} #{p['jersey']}", p.get("conf", 0) >= 0.5
    return f"{tri}", False


def _plate(img: np.ndarray, text: str, x: int, y: int, color, solid: bool) -> None:
    scale, th = 0.62, 1
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, th)
    x1, y1 = int(x - w / 2 - 6), int(y - h - 12)
    x1 = max(0, min(img.shape[1] - w - 12, x1))
    y1 = max(0, y1)
    over = img.copy()
    cv2.rectangle(over, (x1, y1), (x1 + w + 12, y1 + h + 10), (20, 20, 20), -1)
    cv2.addWeighted(over, 0.7 if solid else 0.45, img, 0.3 if solid else 0.55, 0, img)
    cv2.rectangle(img, (x1, y1), (x1 + w + 12, y1 + h + 10), color, 2 if solid else 1)
    cv2.putText(img, text, (x1 + 6, y1 + h + 4), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)


def _header(res: dict) -> str:
    gm = res.get("game_match", {})
    t = res.get("teams", {})
    teams = f"{t.get('A', {}).get('abbr') or 'A?'} vs {t.get('B', {}).get('abbr') or 'B?'}"
    if gm.get("status") == "matched":
        return (f"{teams} | game {gm['game_id']} {gm.get('date', '')} ({gm.get('away')} @ {gm.get('home')}) "
                f"P{gm.get('period')} {gm.get('clock', ['', ''])[0]} | conf {gm.get('conf')}")
    return f"{teams} | game: {gm.get('status', 'none')} ({len(gm.get('candidates', []))} candidates)"


def render_overlay(video: str, tracks: dict, res: dict, out: str, scale: float = 0.5) -> str:
    players = res.get("players", {})
    by_f: dict[int, list] = {}
    for t in tracks["tracks"]:
        p = players.get(str(t["track_id"]))
        if not p:
            continue
        text, solid = _label(p)
        for fr in t["frames"]:
            if fr.get("observed"):
                by_f.setdefault(fr["f"], []).append((fr["bbox"], text, solid, COLORS.get(p.get("team"), (200, 200, 200))))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    tmp = Path(tempfile.mkdtemp()) / "raw.mp4"
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    header = _header(res)
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        for bbox, text, solid, color in by_f.get(f, ()):
            x1, y1, x2, _ = (v * scale for v in bbox)
            _plate(img, text, (x1 + x2) / 2, y1 - 2, color, solid)
        cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(img, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        wr.write(img)
        f += 1
    wr.release()
    cap.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-threads", "8", out], check=False)
    if not Path(out).exists():
        shutil.move(str(tmp), out)
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return out
