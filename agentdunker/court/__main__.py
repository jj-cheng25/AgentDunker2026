"""CLI: python -m agentdunker.court <video> -o court.json [--overlay overlay.mp4] [--tracks tracks.json]"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

from .api import CourtCalibration, to_contract  # noqa: E402
from .render import render_overlay  # noqa: E402
from .video import CourtTracker, read_clip  # noqa: E402

logger = logging.getLogger("agentdunker.court")


def _boxes_from_tracks(path: str) -> tuple[dict, dict]:
    data = json.loads(Path(path).read_text())
    boxes, feet = {}, {}
    for tr in data.get("tracks", []):
        col = tuple(int(c) for c in reversed(tr.get("team_color") or [0, 0, 255]))
        for fr in tr.get("frames", []):
            if fr.get("bbox"):
                boxes.setdefault(int(fr["f"]), []).append(fr["bbox"])
                x0, y0, x1, y1 = fr["bbox"]
                foot = fr.get("foot") or [(x0 + x1) / 2, y1]
                feet.setdefault(int(fr["f"]), []).append((foot[0], foot[1], col))
    return boxes, feet


def calibrate(video: str, tracks: str | None = None) -> dict:
    """Run the full court calibration on a video and return the court.json dict."""
    clip = read_clip(video)
    boxes = _boxes_from_tracks(tracks)[0] if tracks else None
    result = CourtTracker().run(clip, ignore_boxes=boxes)
    sha = hashlib.sha256()
    with open(video, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            sha.update(chunk)
    sha = sha.hexdigest()
    return to_contract(clip, result, sha)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Per-frame image->court homography for NBA broadcast video")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="court.json")
    ap.add_argument("--tracks", help="M1 tracks.json: masks players during line fitting, feet go on the minimap")
    ap.add_argument("--overlay", help="write an evidence video (court lines + minimap)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(asctime)s %(message)s")
    t0 = time.time()
    data = calibrate(a.video, a.tracks)
    Path(a.out).write_text(json.dumps(data))
    fr = data["frames"]
    ok = sum(f["ok"] for f in fr)
    print(f"{a.out}: {len(fr)} frames, ok {ok} ({ok / max(1, len(fr)):.1%}), {len(data['shots'])} shots, "
          f"{time.time() - t0:.0f}s")
    if a.overlay:
        feet = _boxes_from_tracks(a.tracks)[1] if a.tracks else None
        render_overlay(a.video, CourtCalibration(data), a.overlay, feet=feet)


if __name__ == "__main__":
    main()
