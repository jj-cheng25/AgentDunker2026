"""Shot metrics on the M0 bench for a pred dir: shooter accuracy, release |err| (median), made accuracy.

Uses bench/score.py for the GT-id -> track assignment. `--fix clip=frame` overrides a GT release frame that was
verified wrong (reported to M0), and both numbers are printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CMD = Path(__file__).resolve().parents[3] / "agentdunker-command"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--fix", nargs="*", default=["c14_okc_at_sas=164"])
    a = ap.parse_args()
    sys.path.insert(0, str(CMD / "bench"))
    import score as S  # noqa: E402

    fixes = {k: int(v) for k, v in (x.split("=") for x in a.fix)}
    rows = []
    for g in sorted((CMD / "bench" / "gt").glob("c*.json")):
        gt = json.loads(g.read_text())
        play, clip = gt.get("play") or {}, gt["clip"]
        pd = Path(a.pred) / clip
        if not (pd / "events.json").exists() or play.get("release_t") is None:
            continue
        ev = json.loads((pd / "events.json").read_text())
        tr = json.loads((pd / "tracks.json").read_text())
        assign = S.score_tracks(gt, tr, 1)["_assign"]
        shot = ev.get("shot") or {}
        fps, start = gt["video"]["fps"], gt["video"].get("start_pts", 0.0)
        gt_t = play["release_t"]
        gt_fix = start + fixes[clip] / fps if clip in fixes else gt_t
        err = abs(shot["release_t"] - gt_t) if shot.get("release_t") is not None else None
        err_fix = abs(shot["release_t"] - gt_fix) if shot.get("release_t") is not None else None
        rows.append({"clip": clip, "shooter_gt": play.get("shooter"), "shooter_gt_track": assign.get(play.get("shooter")),
                     "shooter": shot.get("shooter_track"),
                     "shooter_ok": shot.get("shooter_track") is not None and str(shot.get("shooter_track")) == str(assign.get(play.get("shooter"))),
                     "release_err": err, "release_err_fixed_gt": err_fix, "made": shot.get("made"), "made_gt": play.get("made"),
                     "made_ok": shot.get("made") == play.get("made"), "method": (shot.get("evidence") or {}).get("release", {}).get("method"),
                     "bank_t": shot.get("bank_t"), "xfg": shot.get("xfg"), "shot_type": shot.get("shot_type"),
                     "points": shot.get("points"), "gt_type": play.get("shot_type")})
    n = len(rows)
    errs = [r["release_err"] for r in rows if r["release_err"] is not None]
    errs_f = [r["release_err_fixed_gt"] for r in rows if r["release_err_fixed_gt"] is not None]
    summ = {"n_clips": n,
            "shooter_acc": sum(r["shooter_ok"] for r in rows) / n,
            "release_median_err_s": float(np.median(errs)) if errs else None,
            "release_median_err_s_fixed_gt": float(np.median(errs_f)) if errs_f else None,
            "release_within_0.15s": sum(e <= 0.15 for e in errs_f) / n,
            "made_acc": sum(r["made_ok"] for r in rows) / n}
    out = {"summary": summ, "clips": rows}
    (Path(a.pred) / "m5_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1))
    for r in rows:
        print(f"{r['clip']:26s} shooter {r['shooter']!s:>4} gt {r['shooter_gt_track']!s:>4} {'ok' if r['shooter_ok'] else 'XX'} | "
              f"rel err {r['release_err_fixed_gt'] if r['release_err_fixed_gt'] is None else round(r['release_err_fixed_gt'], 3)!s:>6} ({r['method']}) | "
              f"made {r['made']} gt {r['made_gt']} | bank {r['bank_t']} xfg {r['xfg']} type {r['shot_type']} pts {r['points']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
