"""``make_film`` with the package configuration applied: providers, preset, threads, runner, progress.

The film itself is made by ``agentdunker.film`` (the M7 assembler). This layer decides which services
the run can use, wires progress reporting, and adds a top-level summary of what was degraded and why
to ``report.json``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
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


PROMO_FILES = ("intro.mp4", "outro.mp4", "cover.png", "intro_score.wav", "outro_score.wav")
_promo_lock = threading.Lock()


def promo_ready() -> bool:
    """Intro / outro / cover already rendered for the current promo sources and fonts?"""
    from .film.render import promo
    d = promo._cache_dir()
    return all((d / f).exists() for f in PROMO_FILES)


def hook_promo() -> None:
    """One intro/outro render at a time, announced as its own stage (first run on a machine: ≈ 6 min, then cached)."""
    from .film.render import promo
    if getattr(promo.build_fixed, "_adk_hooked", False):
        return
    orig = promo.build_fixed

    def build_fixed():
        with _promo_lock:
            if promo_ready():
                return orig()
            runner.emit("promo", "start", mode="first run only, cached afterwards (≈ 6 min on 16 CPU threads)")
            t0 = time.monotonic()
            try:
                d = orig()
            except Exception as e:
                runner.emit("promo", "failed", seconds=time.monotonic() - t0, reason=str(e).splitlines()[0][:160])
                raise
            runner.emit("promo", "done", seconds=time.monotonic() - t0)
            return d

    build_fixed._adk_hooked = True
    promo.build_fixed = build_fixed


def prerender_promo(background: bool = False) -> Optional[threading.Thread]:
    """Render intro/outro now (download-models) or in a background thread (first film, overlaps the analysis)."""
    from .weights import chromium_check
    hook_promo()
    if promo_ready() or not chromium_check()[0]:
        return None
    from .film.render import promo

    def work():
        try:
            promo.build_fixed()
        except Exception as e:  # noqa: BLE001  (the film drops intro/outro and says why)
            logger.warning("intro/outro render failed: %s", e)

    if not background:
        work()
        return None
    th = threading.Thread(target=work, name="promo-render", daemon=True)
    th.start()
    return th


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
    # A private temp dir for the run (stages, Chromium's org.chromium.* dirs), removed afterwards.
    tmp = Path(out_dir).resolve() / "work" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    old_tmp = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(tmp)
    tempfile.tempdir = None
    try:
        prerender_promo(background=True)     # first run: intro/outro render overlaps the analysis
        rep = film_mod.make_film(clip, out_dir, reuse=reuse, use_llm=svc["llm"]["ok"],
                                 use_moss_vl=svc["moss_vl"]["ok"], skip=skip)
    finally:
        root.removeHandler(fh)
        fh.close()
        if old_tmp is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmp
        tempfile.tempdir = None
        shutil.rmtree(tmp, ignore_errors=True)
    return _finish(rep, out_dir, svc, t0)


def _chromium_state() -> dict:
    from .weights import chromium_check
    ok, detail = chromium_check()
    return {"ok": ok, "detail": detail}


def _finish(rep: dict, out_dir: str, svc: dict, t0: float) -> dict:
    degraded = [{"element": k, "tier": v["tier"], "reason": v.get("reason", "")}
                for k, v in (rep.get("elements") or {}).items() if v.get("tier") != "A"]
    rep["agentdunker"] = {
        "version": __version__, "preset": config.get()["run"]["preset"], "runner": config.get()["run"]["runner"],
        "threads": config.threads(), "seconds": round(time.monotonic() - t0, 1), "config": config.public(),
        "services": svc,
        "chromium": _chromium_state(),
        "services_missing": {k: SERVICE_EFFECTS[k] for k, v in svc.items() if not v["ok"]},
        "degraded": degraded,
    }
    p = Path(out_dir) / "report.json"
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    tmp.replace(p)
    return rep
