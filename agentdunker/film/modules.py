"""Run the M1–M6 modules as black-box subprocesses, each in its own interpreter.

Every call is logged to ``work/logs/<name>.log``, timed, and cached: if the output exists and the
inputs (paths, sizes, mtimes, arguments) are unchanged, the call is skipped. A failed or timed-out
module never raises; the caller gets ``ok=False`` and degrades.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
# Default layout: sibling worktrees adk-cmd-m1 … adk-cmd-m6 next to this repo. Override per module
# with ADK_M1_DIR … and ADK_M1_PYTHON …
MODULE_DIRS = {"m1": "adk-cmd-m1", "m2": "adk-cmd-m2", "m3": "adk-cmd-m3", "m4": "adk-cmd-m4",
               "m5": "adk-cmd-m5", "m6": "adk-cmd-m6"}
PACKAGES = {"m1": "agentdunker.tracking", "m2": "agentdunker.court", "m3": "agentdunker.identity", "m4": "agentdunker.replay",
            "m5": "agentdunker.events", "m6": "agentdunker.narration"}
THREADS = int(os.environ.get("ADK_THREADS", "16"))


def _m7_module_dir(mod: str) -> Path:
    env = os.environ.get(f"ADK_{mod.upper()}_DIR")
    if env:
        return Path(env)
    vendored = ROOT / "modules" / PACKAGES[mod]          # open-source layout: modules/<package>/
    if vendored.is_dir():
        return vendored.parent
    return ROOT.parent / MODULE_DIRS[mod]


def _m7_module_python(mod: str) -> str:
    env = os.environ.get(f"ADK_{mod.upper()}_PYTHON")
    if env:
        return env
    d = _m7_module_dir(mod)
    for venv in (".venv", f".venv-{mod}"):
        p = d / venv / "bin" / "python"
        if p.exists():
            return str(p)
    return "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable


@dataclass
class ModuleResult:
    name: str
    ok: bool
    seconds: float = 0.0
    cached: bool = False
    returncode: Optional[int] = None
    reason: str = ""
    outputs: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "seconds": round(self.seconds, 1), "cached": self.cached,
                "returncode": self.returncode, "reason": self.reason,
                "outputs": {k: str(v) for k, v in self.outputs.items()}}


def _tail(p: Path) -> str:
    try:
        return p.read_text(errors="replace")[-3000:]
    except OSError:
        return ""


def _fingerprint(args: list[str], inputs: list[Path]) -> str:
    h = hashlib.sha1(json.dumps([str(a) for a in args]).encode())
    for p in inputs:
        if p and Path(p).exists():
            st = Path(p).stat()
            h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _subprocess_run_module(name: str, mod: str, args: list[str], out: Path, work: Path, *, inputs: list[Path] = (),
               timeout: float = 3600, extra_outputs: Optional[dict] = None,
               env: Optional[dict] = None) -> ModuleResult:
    """``python -m <package> *args`` in the module's worktree; success = rc 0 and ``out`` exists."""
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = work / f".{name}.fingerprint"
    fp = _fingerprint(args, list(inputs))
    outputs = {"main": out, **(extra_outputs or {})}
    if out.exists() and stamp.exists() and stamp.read_text().strip() == fp:
        prev = json.loads((logs / f"{name}.json").read_text()) if (logs / f"{name}.json").exists() else {}
        return ModuleResult(name, True, prev.get("seconds", 0.0), True, 0, "cached", outputs)
    if out.exists():                      # a stale output must never pass as this run's result
        out.replace(out.with_name(out.name + ".prev"))
    cmd = [module_python(mod), "-m", PACKAGES[mod], *[str(a) for a in args]]
    full_env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
                "OPENBLAS_NUM_THREADS": str(THREADS), "PYTHONUNBUFFERED": "1", **(env or {})}
    full_env.pop("PYTHONPATH", None)
    t0 = time.monotonic()
    for attempt in (1, 2):                # one retry when killed (OOM) or hit by a network-disk error
        logger.info("[%s] %s (attempt %d)", name, " ".join(cmd[1:]), attempt)
        try:
            with (logs / f"{name}.log").open("w") as log:
                log.write("$ " + " ".join(cmd) + f"\n(cwd {module_dir(mod)}, attempt {attempt})\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=module_dir(mod), stdout=log, stderr=subprocess.STDOUT,
                                      env=full_env, timeout=timeout)
            rc, reason = proc.returncode, "" if proc.returncode == 0 else f"exit code {proc.returncode}"
        except subprocess.TimeoutExpired:
            rc, reason = None, f"timeout after {timeout:.0f}s"
            break
        except OSError as e:
            rc, reason = None, f"could not start: {e}"
        tail = _tail(logs / f"{name}.log")
        transient = (rc is not None and (rc < 0 or rc in (137, 143))) or rc is None or any(
            k in tail for k in ("Stale file handle", "Input/output error", "Errno 116", "Errno 5", "Bus error"))
        if rc == 0 or not transient or attempt == 2:
            break
        logger.warning("[%s] attempt %d failed (%s); retrying once", name, attempt, reason)
        time.sleep(20)
    if attempt == 2:
        reason = (reason + "; " if reason else "") + "retried once after kill/disk error"
    seconds = time.monotonic() - t0
    ok = rc == 0 and out.exists()
    if rc == 0 and not out.exists():
        reason = "no output written"
    if not ok:
        tail = (logs / f"{name}.log").read_text(errors="replace")[-600:] if (logs / f"{name}.log").exists() else ""
        logger.warning("[%s] failed (%s): %s", name, reason, tail.strip().splitlines()[-1:] if tail else "")
    (logs / f"{name}.json").write_text(json.dumps({"seconds": round(seconds, 1), "rc": rc, "reason": reason}))
    if ok:
        stamp.write_text(fp)
    elif stamp.exists():
        stamp.unlink()
    return ModuleResult(name, ok, seconds, False, rc, reason, outputs)


# --- agentdunker: one environment; modules run in-process by default (agentdunker.runner) ---
from agentdunker.runner import module_dir, module_python, run_module  # noqa: E402,F401
