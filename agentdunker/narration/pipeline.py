"""events.json -> narration.json + vo.wav + narration_mix.wav (+ optional preview mp4)."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from . import align, audio, files, mix, moss, script, voice
from .textnorm import per_line_errors

logger = logging.getLogger(__name__)
ROSTER_FILE = Path(__file__).with_name("roster_zh.json")


def load_roster(path: str | Path | None = None) -> set[str]:
    p = Path(path) if path else ROSTER_FILE
    return set(json.loads(p.read_text())) if p.exists() else set()


def video_duration(path: str | Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def merge_identity(doc: dict, identity: dict | None) -> dict:
    """identity.json (M3) players -> doc['players'] keyed by track id."""
    if not identity:
        return doc
    players = dict(doc.get("players") or {})
    for tid, p in (identity.get("players") or {}).items():
        players.setdefault(str(tid), p)
    return {**doc, "players": players}


def run(events: str | Path, video: str | Path | None, out_json: str | Path, identity: str | Path | None = None,
        bed: str | Path | None = None, bed_db: float = -6.0, duck_db: float = -20.0, preview: str | Path | None = None,
        workdir: str | Path | None = None, roster: str | Path | None = None, use_llm: bool = True,
        video_start: float = 0.0) -> dict:
    out_json = Path(out_json)
    workdir = Path(workdir) if workdir else out_json.parent / (out_json.stem + "_work")
    workdir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(events).read_text())
    doc = merge_identity(doc, json.loads(Path(identity).read_text()) if identity else None)
    if not doc.get("duration"):
        doc["duration"] = video_duration(video) - video_start if video else max(e["t"] for e in doc["events"]) + 3
    duration = float(doc["duration"])
    names = set(script.facts(doc)["allowed_names"])
    keyterms = sorted(names)
    roster_set = load_roster(roster)

    layout = [(round(s.start, 3), round(s.end, 3), s.kind) for s in script.build_slots(doc)]
    digest = hashlib.sha1(json.dumps([doc, layout, use_llm], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    cached = workdir / f"script_{digest}.json"
    saved: Any = files.read_json(cached)
    slots = script.build_slots(doc)
    try:
        for s in slots:
            s.text, s.origin = saved["lines"][str(s.idx)]
        script_log = {**saved["log"], "cached": cached.name}
    except (TypeError, KeyError, ValueError):   # no cache, or an unreadable one
        slots, script_log = script.write_script(doc, roster_set, use_llm=use_llm)
        files.write_text(cached, json.dumps({"lines": {str(s.idx): [s.text, s.origin] for s in slots}, "log": script_log},
                                            ensure_ascii=False, indent=1))
    f = script.facts(doc)
    placed = align.place_all(slots, f, roster_set, workdir, keyterms)
    kept = [p for p in placed if p.samples is not None]

    # Acceptance read-back: one ASR pass over the whole narration track, aligned to the script,
    # so short lines are judged in context (isolated 2-character clips are often misheard).
    # Lines the track read-back gets wrong get one reword + retake, then the track is checked again.
    for check in range(2):
        vo = align.render_track(kept, duration)
        vo_path = audio.save(out_json.with_name(out_json.stem + "_vo.wav"), vo)
        track_heard = moss.asr(vo_path, keyterms, workdir / "cache") if kept else ""
        track_errs = per_line_errors([p.text for p in kept], track_heard) if kept else []
        bad = [p for p, (e, n) in zip(kept, track_errs) if e / max(1, n) > voice.MAX_CER]
        if check == 1 or not bad:
            break
        # A list, not a generator: every bad line gets its retake.
        if not any([align.retake(p, placed, f, roster_set, workdir, keyterms, track_heard) for p in bad]):  # noqa: C419
            break
    track_cer = {id(p): e for p, e in zip(kept, track_errs)}

    bed_src = bed or (video if video and audio.has_audio(video) else None)
    bed_x = audio.load(bed_src, start=video_start or None, dur=duration) if bed_src else None
    intervals = [(p.start, p.end) for p in kept]
    mixed = mix.mix(vo, bed_x, intervals, bed_db=bed_db, duck_db=duck_db)
    mix_path = audio.save(out_json.with_name(out_json.stem + "_mix.wav"), mixed)

    lines_dir = out_json.with_name(out_json.stem + "_lines")
    lines_dir.mkdir(exist_ok=True)
    for p in kept:   # per-line audio lives next to the output, referenced relatively
        p.audio = str(files.copy(p.audio, lines_dir / Path(p.audio).name).relative_to(out_json.parent))

    events_list = doc.get("events", [])
    lines: list[dict[str, Any]] = []
    for p in placed:
        ev = p.slot.events[0] if p.slot.events else None
        ref = events_list.index(next(e for e in events_list if e["t"] == ev["t"])) if ev else p.slot.kind
        lines.append({
            "start": round(p.start, 3), "end": round(p.end, 3), "text": p.text if p.samples is not None else "",
            "audio": p.audio, "event_ref": ref, "event_t": round(ev["t"], 3) if ev else round(p.slot.start, 3),
            "window": [round(p.slot.start, 3), round(p.slot.end, 3)], "origin": p.slot.origin,
            "tempo": round(p.tempo, 3), "in_window": p.in_window if p.samples is not None else None,
            "cer": None if p.final_cer is None else round(p.final_cer, 4),
            "pinyin_cer": None if p.final_py_cer is None else round(p.final_py_cer, 4), "heard": p.final_heard,
            "track_cer": round(track_cer[id(p)][0] / max(1, track_cer[id(p)][1]), 4) if id(p) in track_cer else None,
            "onset_ok": p.take.onset_ok if p.take else None, "take": Path(p.take.path).name if p.take else None,
            "rewrites": p.rewrites, "dropped": p.dropped,
        })
    spoken = [l for l in lines if l["text"]]
    cers = [float(l["cer"]) for l in spoken if l["cer"] is not None]
    tot_err, tot_len = sum(e for e, _ in track_errs), sum(n for _, n in track_errs)
    quality: dict[str, Any] = {
        "track_cer": round(tot_err / max(1, tot_len), 4), "track_heard": track_heard,
        "max_line_track_cer": max((l["track_cer"] for l in spoken if l["track_cer"] is not None), default=None),
        "lines": len(spoken), "dropped": [l["window"] for l in lines if not l["text"]],
        "all_in_window": all(l["in_window"] for l in spoken),
        "max_cer": max(cers) if cers else None, "mean_cer": round(sum(cers) / len(cers), 4) if cers else None,
        "max_pinyin_cer": max((l["pinyin_cer"] for l in spoken if l["pinyin_cer"] is not None), default=None),
        "all_onset_ok": all(l["onset_ok"] for l in spoken),
        "template_lines": sum(1 for l in spoken if l["origin"] == "template"),
    }
    quality["tier"] = ("A" if quality["all_in_window"] and (quality["max_line_track_cer"] or 0) <= 0.05 and quality["all_onset_ok"]
                       else "B" if spoken else "C")
    result = {"lines": lines, "mix": mix_path.name, "vo": vo_path.name, "duration": duration,
              "voice": {"model": moss.TTS_MODEL, "voice_id": moss.VOICE_ID, "asr": moss.ASR_MODEL, "writer": moss.CHAT_MODEL},
              "mix_params": {"bed": str(bed_src) if bed_src else None, "bed_db": bed_db, "duck_db": duck_db},
              "quality": quality, "script_log": script_log}
    files.write_text(out_json, json.dumps(result, ensure_ascii=False, indent=1,
                                          default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    if preview and video:
        mix.mux(video, mix_path, preview, start=video_start, duration=duration)
    return result
