"""CLI: python -m agentdunker.tracking <video> -o tracks.json [--overlay out.mp4] [--work-dir DIR]

Identity pass (optional, after M3): add --identity identity.json --identity-tracks <the tracks.json M3 read>.
"""
from __future__ import annotations

import argparse
import json
import logging

from .pipeline import track_video
from .tracklets import TrackConfig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agentdunker.tracking", description=__doc__)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="tracks.json")
    ap.add_argument("--overlay", help="also render an overlay MP4 here")
    ap.add_argument("--detector", default=TrackConfig.detector)
    ap.add_argument("--imgsz", type=int, default=TrackConfig.imgsz)
    ap.add_argument("--det-hz", type=float, default=TrackConfig.det_hz)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-vlm", action="store_true", help="local role/team only (no Mosshub call)")
    ap.add_argument("--identity", help="M3 identity.json: confident jersey numbers become hard identity links")
    ap.add_argument("--identity-tracks", help="the tracks.json that --identity was computed on")
    ap.add_argument("--max-hold", type=float, help="omit predicted frames held > S s before the first / after the last "
                    "sighting in a shot, and predicted boxes mostly off-screen (default: keep every frame)")
    ap.add_argument("--work-dir", help="keep / reuse pass 1 (detections, features) here; the identity pass then skips detection")
    a = ap.parse_args(argv)
    if a.identity and not a.identity_tracks:
        ap.error("--identity needs --identity-tracks (the tracks.json M3 read)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = TrackConfig(detector=a.detector, imgsz=a.imgsz, det_hz=a.det_hz)
    load = lambda p: json.load(open(p)) if p else None
    res = track_video(a.video, cfg, threads=min(16, a.threads), use_vlm=not a.no_vlm, identity=load(a.identity),
                      base_tracks=load(a.identity_tracks), work_dir=a.work_dir,
                      max_hold=a.max_hold)
    with open(a.out, "w") as f:
        json.dump(res, f)
    if a.overlay:
        from .overlay import render

        render(a.video, res, a.overlay)
    logging.getLogger("agentdunker.tracking").info("wrote %s (%d tracks)", a.out, len(res["tracks"]))


if __name__ == "__main__":
    main()
