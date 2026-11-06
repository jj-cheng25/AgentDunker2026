"""CLI: python -m agentdunker.replay <video> --tracks tracks.json --court court.json [-o replay.mp4] ..."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

from .api import make_replay  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agentdunker.replay", description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--court", required=True)
    ap.add_argument("--identity")
    ap.add_argument("--events")
    ap.add_argument("-o", "--out", default="replay.mp4")
    ap.add_argument("--start", type=float, help="replay window start, source PTS seconds")
    ap.add_argument("--end", type=float, help="replay window end, source PTS seconds")
    ap.add_argument("--timeline", help="director pacing: JSON file or inline JSON [[out_s, src_pts], ...] "
                    "(piecewise linear; sets the window; default = constant slow-down)")
    ap.add_argument("--annotations", help="tactical graphics JSON {\"annotations\": [...]} in renderer format "
                    "(ids T<track_id>, times on the replay clock or \"clock\": \"source_pts\")")
    ap.add_argument("--compare", help="also write a side-by-side QA video (3D | source with reprojected positions)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3, help="parallel headless browsers")
    ap.add_argument("--no-pip", action="store_true")
    ap.add_argument("--max-seconds", type=float, help="render only the first N output seconds (preview)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    timeline = None
    if a.timeline:
        try:
            src = Path(a.timeline)
            timeline = json.loads(src.read_text() if src.is_file() else a.timeline)
            if isinstance(timeline, dict):          # also accept {"timeline": [...]} / {"knots": [...]}
                timeline = timeline.get("timeline") or timeline.get("knots")
        except (OSError, ValueError) as e:
            logging.warning("unreadable --timeline (%s); default pacing", e)
            timeline = "invalid"
    try:
        rep = make_replay(a.video, a.tracks, a.court, a.out, identity=a.identity, events=a.events, start=a.start,
                          end=a.end, compare_out=a.compare, width=a.width, height=a.height, fps=a.fps,
                          workers=a.workers, pip=not a.no_pip, max_seconds=a.max_seconds, timeline=timeline,
                          annotations=a.annotations)
    except Exception as e:  # noqa: BLE001 - contract: always a replay.json, exit 0; caller skips the 3D segment
        logging.exception("agentdunker.replay failed")
        rep = {"ok": False, "reason": f"{type(e).__name__}: {e}"[:500], "degraded": []}
        out = Path(a.out).with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "inputs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
