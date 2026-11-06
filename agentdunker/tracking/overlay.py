"""Overlay video: box + ID + team colour; thin box = predicted (observed=false)."""
from __future__ import annotations

import subprocess

import cv2
import numpy as np

from .video import iter_frames

DRAW = {"A": (255, 200, 0), "B": (200, 0, 255), None: (0, 215, 255)}  # BGR: team A cyan, B magenta, refs amber


def _bgr(rgb: list[int]) -> tuple[int, int, int]:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def render(video: str, res: dict, out: str, scale: float = 1.0) -> None:
    W, H = res["video"]["width"], res["video"]["height"]
    ow, oh = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
         "-r", str(res["video"]["fps"]), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out],
        stdin=subprocess.PIPE,
    )
    s = ow / W
    by_f = [{fr["f"]: fr for fr in tr["frames"]} for tr in res["tracks"]]
    for f, _, img in iter_frames(video):
        if scale != 1.0:
            img = cv2.resize(img, (ow, oh))
        n_obs = 0
        for tr, frames in zip(res["tracks"], by_f):
            fr = frames.get(f)
            if fr is None:
                continue
            x1, y1, x2, y2 = (np.array(fr["bbox"]) * s).astype(int)
            col = DRAW[tr["team"]]
            edge = (0, 0, 0)
            thick = 3 if fr["observed"] else 1
            n_obs += fr["observed"] and tr["role"] == "player"
            cv2.rectangle(img, (x1, y1), (x2, y2), edge, thick + 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
            lab = ("R" if tr["role"] == "referee" else tr["team"]) + str(tr["track_id"]) + ("" if fr["observed"] else "?")
            cv2.putText(img, lab, (x1, y1 - 6), 0, 0.8 * s + 0.3, edge, 4)
            cv2.putText(img, lab, (x1, y1 - 6), 0, 0.8 * s + 0.3, col, 2)
            cv2.rectangle(img, (x2 - 10, y1), (x2, y1 + 10), _bgr(tr["team_color"]), -1)  # jersey swatch
            if fr["observed"]:
                fx, fy = (np.array(fr["foot"]) * s).astype(int)
                cv2.circle(img, (fx, fy), 5, (0, 255, 0) if fr.get("foot_reliable", True) else (0, 0, 255), -1)
        n_pl = sum(tr["role"] == "player" for tr in res["tracks"])
        cv2.putText(img, f"f{f}  players {n_pl} (observed {n_obs})", (20, 40), 0, 1.0, (255, 255, 255), 3)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
