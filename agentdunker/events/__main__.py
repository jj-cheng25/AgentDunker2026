"""CLI: python -m agentdunker.events <video> --tracks tracks.json [--court court.json] [--identity identity.json] -o events.json"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentdunker.events", description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--tracks", required=True, help="M1 tracks.json")
    ap.add_argument("--court", help="M2 court.json (enables screens, drives, 2/3 points)")
    ap.add_argument("--identity", help="M3 identity.json (enables official PBP match and xFG)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--cache", help="detection/pose cache dir (default: next to the output)")
    ap.add_argument("--overlay", help="write a shot/ball-path preview mp4")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if not Path(a.video).is_file():
        print(f"agentdunker.events: video not found: {a.video}", file=sys.stderr)
        return 2
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from .pipeline import run

    out = Path(a.out)
    doc = run(a.video, a.tracks, a.court, a.identity, cache_dir=a.cache or str(out.parent / "m5_cache"),
              use_vlm=not a.no_vlm, use_pose=not a.no_pose, threads=a.threads)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    if a.overlay:
        from .overlay import render
        render(a.video, doc, a.overlay, a.tracks)
    s = doc["shot"]
    print(f"shot: release_t={s['release_t']} shooter={s['shooter_track']} made={s['made']} xfg={s['xfg']} "
          f"| {len(doc['events'])} events, {len(doc['ball_state'])} ball segments, {len(doc['actions'])} actions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
