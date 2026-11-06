"""python -m agentdunker.identity <video> --tracks tracks.json -o identity.json [--overlay preview.mp4]"""
from __future__ import annotations

import argparse
import json
import logging
import os


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agentdunker.identity")
    ap.add_argument("video")
    ap.add_argument("--tracks", required=True, help="M1 tracks.json")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--overlay", help="write a nameplate preview video")
    ap.add_argument("--max-strong", type=int, default=20, help="gpt-6-astra reads per clip (0 = cheap model only)")
    ap.add_argument("--no-deduce", action="store_true", help="never assign the last on-court jersey by elimination")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import cv2

    cv2.setNumThreads(a.threads)
    from .pipeline import IdentityConfig, identify_video

    tracks = json.load(open(a.tracks))
    res = identify_video(a.video, tracks, IdentityConfig(max_strong=a.max_strong, deduce=not a.no_deduce))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    if a.overlay:
        from .overlay import render_overlay

        render_overlay(a.video, tracks, res, a.overlay)


if __name__ == "__main__":
    main()
