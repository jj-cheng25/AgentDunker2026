"""``agentdunker doctor``: every requirement, one line each, with the fix."""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys

from . import config, providers, weights

PACKAGES = [  # (import name, pip name, needed for)
    ("numpy", "numpy", "everything"), ("cv2", "opencv-python", "video"), ("torch", "torch", "detection / ReID"),
    ("ultralytics", "ultralytics", "detector + BoT-SORT"), ("onnxruntime", "onnxruntime", "ball detector"),
    ("transformers", "transformers", "CLIP roles"), ("sklearn", "scikit-learn", "team clustering"),
    ("scipy", "scipy", "geometry / audio"), ("networkx", "networkx", "track linking"), ("av", "av", "video decode"),
    ("PIL", "pillow", "render"), ("requests", "requests", "model APIs"), ("websocket", "websocket-client", "3D renderer"),
    ("soundfile", "soundfile", "audio"), ("librosa", "librosa", "voice alignment"), ("pypinyin", "pypinyin", "voice QA"),
    ("pyloudnorm", "pyloudnorm", "loudness"), ("playwright", "playwright", "intro / transition pages"),
    ("lap", "lap", "BoT-SORT matching"), ("gdown", "gdown", "OSNet download"),
]


def _line(ok: bool | None, what: str, detail: str = "", fix: str = "") -> bool:
    mark = {True: " ok ", False: "FAIL", None: "warn"}[ok]
    print(f"[{mark}] {what:28s} {detail}" + (f"\n         → {fix}" if fix and ok is not True else ""))
    return ok is not False


def preflight() -> list[str]:
    """Cheap checks before a run; returned strings are warnings (the run still degrades gracefully)."""
    config.apply()
    out = []
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        out.append("ffmpeg/ffprobe not on PATH: the film cannot be encoded")
    missing = [a.name for a in weights.ASSETS if config.get()["run"]["preset"] in a.presets and weights.status(a) != "ok"]
    if missing:
        out.append(f"missing weights/assets: {', '.join(missing)} — run `agentdunker download-models`")
    if not weights.chromium_path():
        out.append("no Chromium: no 3D replay or transition — run `agentdunker download-models`")
    return out


def doctor(verify: bool = False, ping: bool = False) -> int:
    config.apply()
    ok = True
    print(f"agentdunker doctor · Python {platform.python_version()} · {platform.system()} {platform.machine()}")
    ok &= _line(sys.version_info >= (3, 11), "python ≥ 3.11", sys.version.split()[0])
    for mod, pip, why in PACKAGES:
        try:
            m = importlib.import_module(mod)
            _line(True, mod, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            ok &= _line(False, mod, f"{type(e).__name__} ({why})", f"pip install {pip}  (or reinstall: pip install -e .)")
    for tool in ("ffmpeg", "ffprobe"):
        p = shutil.which(tool)
        ver = subprocess.run([p, "-version"], capture_output=True, text=True).stdout.split("\n")[0][:60] if p else ""
        ok &= _line(bool(p), tool, ver, "install ffmpeg ≥ 6 with libx264 (apt install ffmpeg / conda install ffmpeg)")
    ch = weights.chromium_path()
    ok &= _line(bool(ch), "chromium", ch or "not found", "agentdunker download-models  (Playwright Chromium)")
    from .device import torch_device
    _line(True, "compute device", torch_device() + ("" if torch_device() == "cuda" else " (no CUDA GPU seen)"))
    _line(True, "threads", str(config.threads()))
    print(f"weights in {weights.model_dir()}")
    preset = config.get()["run"]["preset"]
    clip = [a for a in weights.ASSETS if a.kind == "clip"]
    for a in [a for a in weights.ASSETS if a.kind != "clip"] + clip[-1:]:
        if a.kind == "clip":
            bad = [c.name for c in clip if weights.status(c, verify=verify) != "ok"]
            ok &= _line(not bad, "CLIP ViT-L/14", "ok · MIT" if not bad else f"missing {', '.join(bad)}",
                        "agentdunker download-models")
            continue
        st = weights.status(a, verify=verify)
        need = preset in a.presets
        ok &= _line(True if st == "ok" else (False if need else None), a.name,
                    f"{st} · {a.license.split(' (')[0]}" + ("" if need else f" · only for {'/'.join(a.presets)}"),
                    "agentdunker download-models")
    print("services")
    svc = providers.available()
    l = config.llm()
    _line(svc["llm"]["ok"] or None, f"LLM ({l.provider})",
          f"{l.fast_model} / {l.strong_model}" if svc["llm"]["ok"] else "no key → local rules + template commentary",
          f"set {l.key_env} (or choose another provider in agentdunker.toml)" if l.key_env else "")
    m = config.get()["moss"]
    _line(svc["voice"]["ok"] or None, "voice (MOSS TTS/ASR)",
          f"voice {m['voice_id'][:8]}…" if svc["voice"]["ok"] else f"backend {config.get()['voice']['backend']} → subtitles only",
          f"set {m['key_env']} for the commentary voice")
    _line(svc["moss_vl"]["ok"] or None, "MOSS-VL checks", "on" if svc["moss_vl"]["ok"] else "off → rules only",
          f"set {m['key_env']}")
    if ping:
        for name, st in providers.ping().items():
            ok &= _line(True if st == "ok" else None, f"{name} key (live check)", st,
                        "replace the key; the run would fall back to rules/templates")
    data = config.path("nba_data")
    has = data.is_dir() and any(data.iterdir())
    _line(has or None, "official NBA data", str(data) if has else f"{data} (empty)",
          "optional: game match, names, xFG, season stats; set paths.nba_data / ADK_NBA_DATA")
    if config.get().get("_file"):
        print(f"config: {config.get()['_file']}")
    print("ready" if ok else "not ready: fix the FAIL lines above")
    return 0 if ok else 1
