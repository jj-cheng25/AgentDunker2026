#!/usr/bin/env python3
"""Time the presets end to end and score their contracts on the private M0 benchmark.

    python tools/bench_presets.py --bench $CMD/bench --clips c01_hou_at_dal_kd_ref,c06_lal_at_gsw_curry \
        --presets quality,fast --out runs/presets [--threads 16] [--offline]

For every clip × preset: ``agentdunker run`` (fresh output dir; model-answer caches may be warm), then
``bench/score.py`` on the four contracts. Writes ``<out>/presets.json`` and ``<out>/presets.md``.
The benchmark clips are NBA footage and are not part of this repository.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONTRACTS = ("tracks", "court", "identity", "events")
STAGES = ("tracks", "court", "identity", "events", "replay", "narration")


def run(clip: Path, out: Path, preset: str, threads: int, offline: bool) -> dict:
    cmd = [sys.executable, "-m", "agentdunker", "run", str(clip), "-o", str(out), "--preset", preset,
           "--threads", str(threads)] + (["--offline"] if offline else [])
    t0 = time.monotonic()
    with open(out.parent / f"{out.name}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.monotonic() - t0
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    mods = rep.get("modules", {})
    timing = rep.get("timing_s", {})
    return {"rc": rc, "wall_s": round(wall, 1), "ok": rep.get("ok"), "duration": rep.get("duration"),
            "stages_s": {s: (mods.get(s) or {}).get("seconds") for s in STAGES},
            "render_s": round(sum(v for k, v in timing.items() if k.startswith("render_")), 1),
            "tiers": {k: v["tier"] for k, v in (rep.get("elements") or {}).items()}}


def score(bench: Path, pred: Path, clips: list[str], out: Path) -> dict:
    subprocess.run([sys.executable, str(bench / "score.py"), "--gt", str(bench / "gt"), "--pred", str(pred),
                    "--clips", ",".join(clips), "--out", str(out)], check=True, capture_output=True)
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--presets", default="quality,fast")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--offline", action="store_true", help="no model calls (fair timing, no paid calls)")
    a = ap.parse_args()
    clips = a.clips.split(",")
    res: dict = {"threads": a.threads, "offline": a.offline, "runs": {}, "scores": {}}
    for preset in a.presets.split(","):
        pred = a.out / preset / "pred"
        for c in clips:
            out = a.out / preset / c
            if out.exists():
                shutil.rmtree(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            r = run(a.bench / "clips" / f"{c}.mp4", out, preset, a.threads, a.offline)
            res["runs"].setdefault(preset, {})[c] = r
            print(preset, c, r["wall_s"], "s", r["ok"], flush=True)
            (pred / c).mkdir(parents=True, exist_ok=True)
            for k in CONTRACTS:
                if (out / "work" / f"{k}.json").exists():
                    shutil.copy(out / "work" / f"{k}.json", pred / c / f"{k}.json")
            (a.out / "presets.json").write_text(json.dumps(res, indent=1))
        res["scores"][preset] = score(a.bench, pred, clips, a.out / preset / "summary.json")
        (a.out / "presets.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({p: {c: r["wall_s"] for c, r in rs.items()} for p, rs in res["runs"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
