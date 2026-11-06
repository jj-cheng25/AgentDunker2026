"""CLI: python -m agentdunker.narration --events events.json --video clip.mp4 -o narration.json"""
from __future__ import annotations

import argparse
import logging
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "8")

from .pipeline import run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Events -> Chinese commentary script -> Mossland voice -> aligned, ducked mix.")
    ap.add_argument("--events", required=True, help="events.json (PLAN events contract, optional sections/players)")
    ap.add_argument("--video", help="video whose audio is the bed and which the preview uses")
    ap.add_argument("-o", "--out", required=True, help="output narration.json; _vo.wav and _mix.wav are written beside it")
    ap.add_argument("--identity", help="identity.json (M3); confident names become allowed names")
    ap.add_argument("--bed", help="audio to duck under the narration (default: the video's audio)")
    ap.add_argument("--bed-db", type=float, default=-6.0, help="bed gain outside narration; -inf removes the bed")
    ap.add_argument("--duck-db", type=float, default=-20.0, help="bed gain under narration")
    ap.add_argument("--video-start", type=float, default=0.0, help="timeline 0 inside the video, seconds")
    ap.add_argument("--preview", help="write a preview mp4 (video + mix)")
    ap.add_argument("--workdir", help="cache for takes and ASR (default: <out>_work)")
    ap.add_argument("--no-llm", action="store_true", help="template lines only (no script model call)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = run(args.events, args.video, args.out, identity=args.identity, bed=args.bed, bed_db=args.bed_db,
              duck_db=args.duck_db, preview=args.preview, workdir=args.workdir, use_llm=not args.no_llm,
              video_start=args.video_start)
    q = res["quality"]
    print(f"lines={q['lines']} tier={q['tier']} in_window={q['all_in_window']} track_cer={q['track_cer']} "
          f"max_line_track_cer={q['max_line_track_cer']} max_isolated_cer={q['max_cer']} "
          f"onset_ok={q['all_onset_ok']} templates={q['template_lines']} dropped={len(q['dropped'])}")
    for line in res["lines"]:
        print(f"{line['start']:7.2f}-{line['end']:7.2f} x{line['tempo']:.2f} cer={line['track_cer']}/{line['cer']} {line['text']}")


if __name__ == "__main__":
    main()
