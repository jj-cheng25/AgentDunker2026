"""Commentary without a voice: the narration stage's script, timed to its windows, no TTS.

Used when no MOSS voice is configured (``voice.backend = "none"`` or no ``MOSS_API_KEY``). The lines are
written exactly as the voiced path writes them (the configured LLM if there is one, else the
narration module's deterministic templates, which only state measured facts). The film then shows them
as subtitles in the 3D replay and keeps the broadcast sound under the live play.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from . import providers

logger = logging.getLogger(__name__)
RATE = 4.3            # chars per second (the narration module's budget rate)


def write_lines(events: Path, identity: Path | None, out: Path) -> dict:
    from .narration import pipeline, script
    from .narration.textnorm import spoken_len

    doc = json.loads(Path(events).read_text())
    doc = pipeline.merge_identity(doc, json.loads(Path(identity).read_text()) if identity else None)
    if not doc.get("duration"):
        doc["duration"] = max([e["t"] for e in doc.get("events", [])] or [0]) + 3
    use_llm = providers.llm_key() is not None
    slots, log = script.write_script(doc, pipeline.load_roster(), use_llm=use_llm)
    lines = []
    for s in slots:
        if not s.text:
            continue
        end = min(s.end, s.start + max(1.0, spoken_len(s.text) / RATE + 0.3))
        lines.append({"start": round(s.start, 3), "end": round(end, 3), "text": s.text, "audio": "",
                      "event_ref": s.kind, "event_t": round(s.events[0]["t"], 3) if s.events else round(s.start, 3),
                      "window": [round(s.start, 3), round(s.end, 3)], "origin": s.origin})
    n_tpl = sum(1 for l in lines if l["origin"] == "template")
    result = {"lines": lines, "mix": None, "vo": "narration_vo.wav", "duration": float(doc["duration"]),
              "voice": {"backend": "none", "writer": "llm" if use_llm else "template"},
              "quality": {"tier": "B", "lines": len(lines), "template_lines": n_tpl, "all_in_window": True,
                          "voiced": False, "reason": "no MOSS voice configured: subtitles only"},
              "script_log": log}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    tmp.replace(out)
    return result


def run_text_only(name: str, argv: list[str], out: Path, work: Path, inputs: list = ()):
    from .film.modules import ModuleResult
    from .runner import emit

    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--identity")
    ap.add_argument("-o", "--out", required=True)
    a, _ = ap.parse_known_args(argv)
    emit(name, "start", mode="subtitles only (no voice)")
    t0 = time.monotonic()
    try:
        write_lines(Path(a.events), Path(a.identity) if a.identity else None, Path(a.out))
        ok, reason = True, "no voice configured: subtitles only"
    except Exception as e:  # noqa: BLE001
        logger.exception("text-only narration failed")
        ok, reason = False, f"{type(e).__name__}: {e}"
    sec = time.monotonic() - t0
    emit(name, "done" if ok else "failed", seconds=sec, reason=reason)
    return ModuleResult(name, ok, sec, False, 0 if ok else 1, reason, {"main": out})
