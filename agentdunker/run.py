"""``make_film`` with the package configuration applied: providers, preset, threads, runner, progress.

The film itself is made by ``agentdunker.film`` (the M7 assembler). This layer decides which services
the run can use, wires progress reporting, and adds a top-level summary of what was degraded and why
to ``report.json``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from . import __version__, config, providers, runner

logger = logging.getLogger(__name__)

STEPS = ["tracks", "court", "identity", "events", "shots", "director", "narration", "replay", "render_intro",
         "render_live", "render_bridge", "render_replay", "render_freeze", "render_resume", "render_outro",
         "audio", "compose"]
TIMED = {"module_narration": None, "module_replay": None, "modules_analysis": None, "retime": None}

SERVICE_EFFECTS = {
    "llm": "no LLM: local team/role split only, jersey numbers not read (no names), template phase titles, "
           "template commentary",
    "voice": "no voice: commentary shown as subtitles in the 3D replay; live play keeps the broadcast sound",
    "moss_vl": "no MOSS-VL: shot-type check and xFG-card read-back skipped (rules only)",
}


def _hook_timed(Run) -> None:
    """Report the assembler's own steps (shots, director, renders, audio) as progress events."""
    if getattr(Run.timed, "_adk_hooked", False):
        return
    orig = Run.timed

    def timed(self, name, fn, *a, **kw):
        quiet = name.startswith("module") or name in TIMED
        if not quiet:
            runner.emit(name, "start")
        t0 = time.monotonic()
        try:
            out = orig(self, name, fn, *a, **kw)
        except Exception:
            if not quiet:
                runner.emit(name, "failed", seconds=time.monotonic() - t0)
            raise
        if not quiet:
            runner.emit(name, "done", seconds=time.monotonic() - t0)
        return out

    timed._adk_hooked = True
    Run.timed = timed


def make_film(clip: str, out_dir: str, *, reuse: Optional[str] = None, skip: tuple = (),
              progress: Optional[Callable] = None) -> dict:
    config.apply()
    if progress:
        runner.on_progress(progress)
    from .film import film as film_mod
    from .film.pipeline import Run
    _hook_timed(Run)
    svc = providers.available()
    t0 = time.monotonic()
    logs = Path(out_dir) / "work" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs / "agentdunker.log")          # the whole run at INFO (console: warnings)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(fh)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    try:
        rep = film_mod.make_film(clip, out_dir, reuse=reuse, use_llm=svc["llm"]["ok"],
                                 use_moss_vl=svc["moss_vl"]["ok"], skip=skip)
    finally:
        root.removeHandler(fh)
        fh.close()
    return _finish(rep, out_dir, svc, t0)


def _finish(rep: dict, out_dir: str, svc: dict, t0: float) -> dict:
    degraded = [{"element": k, "tier": v["tier"], "reason": v.get("reason", "")}
                for k, v in (rep.get("elements") or {}).items() if v.get("tier") != "A"]
    rep["agentdunker"] = {
        "version": __version__, "preset": config.get()["run"]["preset"], "runner": config.get()["run"]["runner"],
        "threads": config.threads(), "seconds": round(time.monotonic() - t0, 1), "config": config.public(),
        "services": svc,
        "services_missing": {k: SERVICE_EFFECTS[k] for k, v in svc.items() if not v["ok"]},
        "degraded": degraded,
    }
    p = Path(out_dir) / "report.json"
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    tmp.replace(p)
    return rep
