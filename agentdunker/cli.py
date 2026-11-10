"""agentdunker — NBA clip → narrated 3D tactical breakdown film.

  agentdunker doctor                       check Python packages, ffmpeg, Chromium, weights, API keys
  agentdunker download-models              fetch all weights (sha256-checked), fonts, CLIP and Chromium
  agentdunker run clip.mp4 -o out/         make out/film.mp4 + out/report.json (re-run resumes from out/work)
  agentdunker ui                           local web page: upload a clip, watch progress, download the film
  agentdunker config                       print the effective configuration (no credentials)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__, config

LABELS = {"tracks": "players & referees (tracking)", "court": "court calibration", "identity": "jerseys & names",
          "events": "ball, shot & events", "shots": "camera shots", "director": "director (edit plan)",
          "narration": "commentary", "replay": "3D replay render", "render_intro": "intro", "render_live":
          "live play + nameplates", "render_bridge": "3D transition", "render_replay": "3D segment",
          "render_freeze": "shot freeze", "render_resume": "resume + light wave", "render_outro": "outro",
          "audio": "audio mix", "promo": "intro/outro render", "compose": "final mux", "moss_vl_shots": "MOSS-VL shot check",
          "moss_vl_readback": "MOSS-VL card read-back"}


class Progress:
    def __init__(self, stream=sys.stderr):
        self.t0 = time.monotonic()
        self.stream = stream

    def __call__(self, stage: str, state: str, info: dict) -> None:
        el = time.monotonic() - self.t0
        label = LABELS.get(stage, stage)
        sec = info.get("seconds")
        if state == "start":
            extra = f" — {info['mode']}" if info.get("mode") else ""
            msg = f"▶ {label}{extra}"
        elif state == "cached":
            msg = f"✓ {label}  (cached)"
        elif state == "done":
            msg = f"✓ {label}  {sec:.1f}s" if sec is not None else f"✓ {label}"
        else:
            msg = f"✗ {label}  {info.get('reason', '')}".rstrip()
            if info.get("log"):
                msg += f"\n            log: {info['log']}  (the film continues without it)"
        print(f"[{el:7.1f}s] {msg}", file=self.stream, flush=True)


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", help="config file (TOML); default ./agentdunker.toml or ~/.config/agentdunker/config.toml")
    ap.add_argument("-v", "--verbose", action="store_true", help="module logs on the console")


def _setup(a) -> None:
    import os
    if getattr(a, "config", None):
        os.environ["ADK_CONFIG"] = a.config
        config.load.cache_clear()
    over = {}
    run = {k: v for k, v in (("preset", getattr(a, "preset", None)), ("threads", getattr(a, "threads", None)),
                             ("runner", "subprocess" if getattr(a, "isolate", False) else None),
                             ("device", getattr(a, "device", None))) if v}
    if run:
        over["run"] = run
    if getattr(a, "provider", None):
        over["llm"] = {"provider": a.provider}
    if getattr(a, "no_voice", False):
        over["voice"] = {"backend": "none"}
    if getattr(a, "offline", False):
        over["llm"], over["voice"], over["moss"] = {"provider": "none"}, {"backend": "none"}, {"vl_checks": False}
        over.setdefault("run", {})["offline"] = True
    if over:
        config.override(**over)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in logging.getLogger().handlers:         # console: warnings only, unless -v; files get INFO
        h.setLevel(logging.INFO if getattr(a, "verbose", False) else logging.WARNING)


def cmd_run(a) -> int:
    from .doctor import preflight
    from .run import make_film
    clip = Path(a.clip)
    if not clip.is_file():
        print(f"error: clip not found: {clip}", file=sys.stderr)
        return 2
    problems = preflight()
    for p in problems:
        print(f"warning: {p}", file=sys.stderr)
    from .doctor import chromium_problem
    chrome = chromium_problem()
    if chrome:                       # always a film: warn loudly, then carry on without the Chromium parts
        bar = "!" * 78
        print(f"{bar}\nWARNING: {chrome}\n{bar}", file=sys.stderr)
    from .providers import available
    c, svc = config.get(), available()
    nokey = " (no key)"
    print(f"agentdunker {__version__} · preset {c['run']['preset']} · {config.threads()} threads · "
          f"LLM {config.llm().provider}{'' if svc['llm']['ok'] else nokey} · "
          f"voice {c['voice']['backend']}{'' if svc['voice']['ok'] else nokey} · runner {c['run']['runner']}", file=sys.stderr)
    out = Path(a.out)
    rep = make_film(str(clip), str(out), reuse=a.reuse, skip=tuple(s for s in (a.skip or "").split(",") if s),
                    progress=Progress())
    meta = rep.get("agentdunker", {})
    if rep.get("ok"):
        print(f"\nfilm   {rep['film']}  ({rep.get('duration')} s, {meta.get('seconds')} s to make)", file=sys.stderr)
    else:
        print(f"\nno film: {rep.get('error')}  (details: {out / 'report.json'})", file=sys.stderr)
    print(f"report {out / 'report.json'}", file=sys.stderr)
    for d in meta.get("degraded", []):
        print(f"  {d['tier']}  {d['element']:15s} {d['reason'][:110]}", file=sys.stderr)
    for v in meta.get("services_missing", {}).values():
        print(f"  note  {v}", file=sys.stderr)
    return 0 if rep.get("ok") else 1


def cmd_doctor(a) -> int:
    from .doctor import doctor
    return doctor(verify=a.verify, ping=a.ping)


def cmd_download(a) -> int:
    from . import weights
    config.apply()
    print(f"model dir: {weights.model_dir()}")
    failed = weights.download(names=a.only.split(",") if a.only else None, preset=a.preset, force=a.force)
    if not a.only:
        if not a.no_chromium and not weights.chromium_path():
            if not weights.install_chromium():
                failed.append("chromium")
        elif weights.chromium_path():
            print(f"  ok        Chromium {weights.chromium_path()}")
        ok, detail = weights.chromium_check()
        if not ok:
            print(f"  FAILED    Chromium cannot start: {detail}\n            fix: {weights.chromium_fix()}")
            failed.append("chromium (system libraries)")
        elif not failed and not a.no_prerender:
            from .run import prerender_promo, promo_ready
            if not promo_ready():
                print("  render    intro/outro once (≈ 6 min on 16 CPU threads; later films reuse it)", flush=True)
                t0 = time.monotonic()
                prerender_promo()
                print(f"  {'ok' if promo_ready() else 'FAILED':9s} intro/outro ({time.monotonic() - t0:.0f} s)")
            else:
                print("  ok        intro/outro (cached)")
    if failed:
        print(f"failed: {', '.join(failed)}\n"
              "→ run `agentdunker download-models` again: finished files are kept, partial ones resume.")
        return 1
    print("all assets present and verified")
    return 0


def cmd_ui(a) -> int:
    from .ui import launch
    config.apply()
    return launch(host=a.host, port=a.port, share=False)


def cmd_config(a) -> int:
    print(json.dumps(config.public(), indent=1, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentdunker", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"agentdunker {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="make a film from a clip")
    r.add_argument("clip", help="NBA broadcast offense clip, 5–20 s")
    r.add_argument("-o", "--out", required=True, help="output directory (film.mp4, report.json, work/)")
    r.add_argument("--preset", choices=["fast", "quality"], help="fast: YOLO11m at 10 Hz; quality: YOLO11x at 15 Hz (default)")
    r.add_argument("--threads", type=int, help="CPU threads for every stage (default 16)")
    r.add_argument("--device", choices=["auto", "cpu", "cuda"])
    r.add_argument("--provider", choices=["mosshub", "openai", "gemini", "none"], help="LLM/VLM provider")
    r.add_argument("--no-voice", action="store_true", help="subtitles instead of the MOSS voice")
    r.add_argument("--offline", action="store_true", help="no network model calls at all (rules + templates)")
    r.add_argument("--isolate", action="store_true", help="run each stage in its own process")
    r.add_argument("--reuse", help="directory with tracks/court/identity/events.json to reuse")
    r.add_argument("--skip", help="comma list: tracks,court,identity,events,replay,narration")
    _common(r)
    r.set_defaults(fn=cmd_run)

    d = sub.add_parser("doctor", help="check the environment")
    d.add_argument("--verify", action="store_true", help="also verify weight checksums (slow)")
    d.add_argument("--ping", action="store_true", help="check that the API keys are accepted (GET /models, free)")
    _common(d)
    d.set_defaults(fn=cmd_doctor)

    m = sub.add_parser("download-models", help="download weights, fonts, CLIP and Chromium")
    m.add_argument("--preset", choices=["fast", "quality"], help="only what this preset needs")
    m.add_argument("--only", help="comma list of asset names")
    m.add_argument("--force", action="store_true")
    m.add_argument("--no-chromium", action="store_true")
    m.add_argument("--no-prerender", action="store_true", help="do not render the intro/outro now (first film does it)")
    _common(m)
    m.set_defaults(fn=cmd_download)

    u = sub.add_parser("ui", help="local web UI (needs the [ui] extra)")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=7860)
    _common(u)
    u.set_defaults(fn=cmd_ui)

    c = sub.add_parser("config", help="print the effective configuration")
    _common(c)
    c.set_defaults(fn=cmd_config)

    a = ap.parse_args(argv)
    _setup(a)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
