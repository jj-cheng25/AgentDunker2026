"""python -m agentdunker <clip.mp4> -o out/  →  out/film.mp4, out/report.json, out/work/"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentdunker", description=__doc__)
    ap.add_argument("clip", help="NBA broadcast offense clip (5–20 s)")
    ap.add_argument("-o", "--out", required=True, help="output directory")
    ap.add_argument("--reuse", help="directory with existing tracks/court/identity/events.json to reuse")
    ap.add_argument("--no-llm", action="store_true", help="template phase titles (no gemini call)")
    ap.add_argument("--no-moss-vl", action="store_true", help="skip the MOSS-VL checks (shot-type veto, card read-back)")
    ap.add_argument("--skip", default="", help="comma list of stages to skip: tracks,court,identity,events,replay,narration")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from .film import make_film
    rep = make_film(a.clip, a.out, reuse=a.reuse, use_llm=not a.no_llm, use_moss_vl=not a.no_moss_vl,
                    skip=tuple(s for s in a.skip.split(",") if s))
    print(json.dumps({k: rep.get(k) for k in ("ok", "film", "duration", "error")}, ensure_ascii=False))
    print(json.dumps({k: v["tier"] for k, v in rep.get("elements", {}).items()}, ensure_ascii=False))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
