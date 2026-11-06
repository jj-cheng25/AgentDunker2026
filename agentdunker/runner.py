"""Runs the analysis stages for the film assembler (``agentdunker.film``).

Default: **in-process** — ``agentdunker.<stage>.__main__.main(argv)`` in this interpreter, one environment,
models loaded once per stage. ``runner = "subprocess"`` (``--isolate``) keeps the original M7 behaviour:
each stage is ``python -m agentdunker.<stage>`` in a child process (a crash or OOM cannot take the
assembler down, and timeouts are enforced). Both modes cache by input fingerprint in ``work/``, write
``work/logs/<stage>.log`` and never raise: a failed stage returns ``ok=False`` and the film degrades.

Presets add arguments per stage (``PRESETS``); the thread cap from the config replaces any ``--threads``.
"""
from __future__ import annotations

import importlib
import io
import json
import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from . import config, providers

logger = logging.getLogger(__name__)

STAGE = {"m1": "tracking", "m2": "court", "m3": "identity", "m4": "replay", "m5": "events", "m6": "narration"}
REPO = Path(__file__).resolve().parents[1]

# Stage arguments per preset. "quality" = the module defaults (what M7 runs).
PRESETS: dict[str, dict[str, list[str]]] = {
    "quality": {},
    "fast": {
        "m1": ["--detector", "yolo11m.pt", "--imgsz", "960", "--det-hz", "10"],
        "m5": ["--no-pose"],
    },
}
THREADED = {"m1", "m3", "m5"}          # stages with a --threads option

_listeners: list[Callable[[str, str, dict], None]] = []


def on_progress(fn: Callable[[str, str, dict], None]) -> None:
    """Register ``fn(stage, state, info)``; state is start | done | failed | cached."""
    _listeners.append(fn)


def emit(stage: str, state: str, **info) -> None:
    for fn in list(_listeners):
        try:
            fn(stage, state, info)
        except Exception:  # noqa: BLE001
            pass


def module_dir(mod: str) -> Path:
    return REPO


def module_python(mod: str) -> str:
    return sys.executable


def stage_args(mod: str, args: list) -> list[str]:
    args = [str(a) for a in args]
    preset = config.get()["run"]["preset"]
    extra = list(PRESETS.get(preset, {}).get(mod, []))
    if mod in THREADED:
        while "--threads" in args:
            i = args.index("--threads")
            del args[i:i + 2]
        extra += ["--threads", str(config.threads())]
    return args + extra


# ---------------------------------------------------------------- in-process execution
class _ThreadStdout(io.TextIOBase):
    """sys.stdout replacement that sends each thread's prints to that thread's stage log."""

    def __init__(self, real):
        self.real, self.local = real, threading.local()

    def write(self, s):  # noqa: D401
        f = getattr(self.local, "f", None)
        return (f or self.real).write(s)

    def flush(self):
        f = getattr(self.local, "f", None)
        (f or self.real).flush()


_stdout_lock = threading.Lock()


def _install_stdout() -> _ThreadStdout:
    with _stdout_lock:
        if not isinstance(sys.stdout, _ThreadStdout):
            sys.stdout = _ThreadStdout(sys.stdout)
        return sys.stdout


def _inprocess(mod: str, argv: list[str], log_path: Path) -> tuple[Optional[int], str]:
    pkg = f"agentdunker.{STAGE[mod]}"
    out = _install_stdout()
    with log_path.open("a") as log:
        handler = logging.StreamHandler(log)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(lambda r: r.name.startswith(pkg))
        root = logging.getLogger()
        root.addHandler(handler)
        out.local.f = log
        try:
            rc = importlib.import_module(f"{pkg}.__main__").main(argv)
            return (rc if isinstance(rc, int) else 0), ""
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            return code, "" if code == 0 else f"exit code {code}"
        except Exception as e:  # noqa: BLE001  — a stage never takes the film down
            log.write(traceback.format_exc())
            return 1, f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            out.local.f = None
            root.removeHandler(handler)


def run_module(name: str, mod: str, args: list, out: Path, work: Path, *, inputs: list = (),
               timeout: float = 3600, extra_outputs: Optional[dict] = None, env: Optional[dict] = None):
    from .film.modules import ModuleResult, _fingerprint, _subprocess_run_module

    argv = stage_args(mod, args)
    if mod == "m6" and providers.moss_key("speech") is None:
        from .textnarration import run_text_only
        return run_text_only(name, argv, out, work, inputs=list(inputs))
    emit(name, "start")
    if config.get()["run"]["runner"] == "subprocess":
        res = _subprocess_run_module(name, mod, argv, out, work, inputs=inputs, timeout=timeout,
                                     extra_outputs=extra_outputs, env=env)
        emit(name, "cached" if res.cached else ("done" if res.ok else "failed"), seconds=res.seconds, reason=res.reason)
        return res

    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = work / f".{name}.fingerprint"
    fp = _fingerprint(argv, [Path(p) for p in inputs])
    outputs = {"main": out, **(extra_outputs or {})}
    if out.exists() and stamp.exists() and stamp.read_text().strip() == fp:
        prev = json.loads((logs / f"{name}.json").read_text()) if (logs / f"{name}.json").exists() else {}
        emit(name, "cached", seconds=prev.get("seconds", 0.0))
        return ModuleResult(name, True, prev.get("seconds", 0.0), True, 0, "cached", outputs)
    if out.exists():
        out.replace(out.with_name(out.name + ".prev"))
    log_path = logs / f"{name}.log"
    log_path.write_text(f"$ agentdunker.{STAGE[mod]} {' '.join(argv)}  (in-process)\n")
    t0 = time.monotonic()
    rc, reason = _inprocess(mod, argv, log_path)
    if rc != 0 and any(k in reason for k in ("Stale file handle", "Input/output error", "Errno 116", "Errno 5")):
        logger.warning("[%s] disk error (%s); retrying once", name, reason)
        time.sleep(20)
        rc, reason = _inprocess(mod, argv, log_path)
        reason = (reason + "; " if reason else "") + "retried once after a disk error"
    seconds = time.monotonic() - t0
    ok = rc == 0 and out.exists()
    if rc == 0 and not out.exists():
        reason = "no output written"
    (logs / f"{name}.json").write_text(json.dumps({"seconds": round(seconds, 1), "rc": rc, "reason": reason}))
    if ok:
        stamp.write_text(fp)
    elif stamp.exists():
        stamp.unlink()
    emit(name, "done" if ok else "failed", seconds=seconds, reason=reason, log=str(log_path))
    return ModuleResult(name, ok, seconds, False, rc, reason, outputs)
