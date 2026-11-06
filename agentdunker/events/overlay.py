"""Preview video: ball path, rim, holder, release / bank / make markers and the current ball state."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2

from .tracks import Tracks

COL = {"hold": (255, 200, 0), "dribble": (0, 200, 255), "pass": (0, 255, 0), "shot": (0, 0, 255),
       "loose": (180, 180, 180)}


def render(video: str, doc: dict, out: str, tracks_path: str | None = None, scale: float = 0.5) -> None:
    tracks = Tracks.load(tracks_path)
    info = doc["video"]
    fps, start = info["fps"], info.get("start_pts", 0.0)
    shot = doc.get("shot") or {}
    path = {round((t - start) * fps): (x, y) for t, x, y in shot.get("ball_path_img", [])}
    rim = {round((t - start) * fps): (cx, cy, w) for t, cx, cy, w in shot.get("rim_img", [])}
    segs = doc.get("ball_state", [])
    marks = {k: shot.get(k) for k in ("release_t", "bank_t", "made_t")}
    cap = cv2.VideoCapture(video)
    w, h = int(info["width"] * scale), int(info["height"] * scale)
    tmp = Path(tempfile.mkdtemp()) / "o.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail: list[tuple[float, float]] = []
    f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        t = start + f / fps
        seg = next((s for s in segs if s["f0"] <= f < s["f1"]), None)
        if seg and seg.get("holder") is not None:
            b = tracks.box(seg["holder"], f)
            if b:
                x1, y1, x2, y2 = map(int, b.box)
                cv2.rectangle(im, (x1, y1), (x2, y2), COL[seg["type"]], 3)
                cv2.putText(im, f"#{seg['holder']} {seg['type']}", (x1, y1 - 8), 0, 1.0, COL[seg["type"]], 2)
        if shot.get("shooter_track") is not None and marks["release_t"] and abs(t - marks["release_t"]) < 0.6:
            b = tracks.box(shot["shooter_track"], f)
            if b:
                x1, y1, x2, y2 = map(int, b.box)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(im, f"SHOOTER #{shot['shooter_track']}", (x1, y2 + 30), 0, 1.0, (0, 0, 255), 2)
        if f in rim:
            cx, cy, rw = rim[f]
            cv2.ellipse(im, (int(cx), int(cy)), (int(rw / 2), int(rw / 6)), 0, 0, 360, (0, 165, 255), 2)
        if f in path:
            trail.append(path[f])
        if marks["release_t"] and t < marks["release_t"]:
            trail = []
        for i, (x, y) in enumerate(trail[-40:]):
            cv2.circle(im, (int(x), int(y)), 4 + i // 10, (255, 255, 0), -1)
        lines = [f"t={t:6.2f}s f={f}"]
        if seg:
            lines.append(f"ball: {seg['type']}" + (f" {seg.get('from')}->{seg.get('to')}" if seg["type"] == "pass" else ""))
        for k, v in marks.items():
            if v is not None and 0 <= t - v < 0.8:
                lines.append({"release_t": "RELEASE", "bank_t": "BANK", "made_t": "MADE" if shot.get("made") else "RIM"}[k])
        if f == 0 or marks["release_t"] is None:
            lines.append("shot: none detected" if marks["release_t"] is None else "")
        lines = [s for s in lines if s]
        y0 = im.shape[0] - 70 * len(lines) - 20
        band = im[y0 - 10:, :720]
        band[:] = (band * 0.35).astype(band.dtype)
        for i, s in enumerate(lines):
            cv2.putText(im, s, (20, y0 + 50 + 70 * i), 0, 1.8, (255, 255, 255), 4)
        vw.write(cv2.resize(im, (w, h)))
        f += 1
    vw.release()
    cap.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", out], check=True)
