"""``make_film(clip, out_dir)``: the one-call entry point (CLI: ``python -m agentdunker``)."""
from __future__ import annotations

import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from . import director, models
from .audio import build_mix
from .media import probe, read_wav
from .pipeline import Run, _dump, _durations
from .render import promo
from .shots import build_shots, classify_rule

logger = logging.getLogger(__name__)


def _drop_segments(edl: dict, pieces: list, dropped: set[str]) -> list:
    """Remove failed segments from the timeline; shift later segments and narration pieces."""
    if not dropped:
        return pieces
    shift, kept, moves = 0.0, [], []
    for s in edl["segments"]:
        a, b = s["film"]
        if s["kind"] in dropped:
            moves.append((a, b, None))
            shift += b - a
            continue
        moves.append((a, b, -shift))
        s["film"] = [round(a - shift, 3), round(b - shift, 3)]
        kept.append(s)
    edl["segments"] = kept
    edl["duration"] = round(edl["duration"] - shift, 3)
    out = []
    for t, clip in pieces:
        for a, b, d in moves:
            if a <= t < b:
                if d is not None:
                    out.append((t + d, clip))
                break
        else:
            out.append((t - shift, clip))
    return out


def tiers(run: Run, edl: dict, data: dict, rj: Optional[dict], narration: Optional[dict]) -> None:
    lay = getattr(run, "layouts", {})
    have = {s["kind"] for s in edl["segments"]}
    t = run.tier
    t("intro", "A" if "intro" in have else "C")
    t("outro", "A" if "outro" in have else "C")
    t("live", "A" if "live" in have else "C")
    np_stats = (lay.get("live") or {}).get("nameplates") or {}
    plates = np_stats.get("plates") or {}
    if not data["tracks"]:
        t("nameplates", "C", "no M1 tracks")
    elif not np_stats.get("frames_with_plates"):
        t("nameplates", "C", "no confident identity on a wide shot" if data["identity"] else "no M3 identity: numbers not trusted")
    elif plates.get("number"):
        t("nameplates", "B", f"{plates.get('name', 0)} players named, {plates.get('number')} number only, "
          f"{plates.get('none', 0)} hidden", **np_stats)
    else:
        t("nameplates", "A", f"{plates.get('name', 0)} players named, {plates.get('none', 0)} hidden", **np_stats)
    hs = getattr(run, "hero_stat", None)
    if hs and np_stats.get("frames_with_plates"):
        t("hero_card", "A", f"{hs['caption']} {hs['value']} (season metric, {hs['source']})")
    else:
        t("hero_card", "B", "no season gravity for this shooter/season (or game not matched): normal plate")
    t("transition", "A" if "bridge" in have else "C")
    rep = director.seg(edl, "replay")
    if rep:
        deg = (rj or {}).get("degraded") or []
        t("replay_3d", "B" if deg or (rj or {}).get("n_players", 10) < 10 else "A",
          "; ".join(map(str, deg)) or "", window=rep["source"], n_players=(rj or {}).get("n_players"),
          repeated_frames=(lay.get("replay") or {}).get("repeated_frames"))
        want, got = edl["decisions"].get("replay_title") or "", (rj or {}).get("title") or ""
        t("title_3d", "A" if want and got == want else "B",
          "director title from measured facts" if want and got == want else
          ("M4 default title (no names / no matched game)" if not want else f"M4 shows {got!r}, wanted {want!r}"),
          title=got, title_source=(rj or {}).get("title_source"))
        srcs = {ph.get("text_source", "template") for ph in rep["phases"]}
        t("phase_titles", "A" if srcs == {"llm"} else "B", "gemini-3.8-flash from verified events" if srcs == {"llm"}
          else "template titles for some phases", phases=[[ph["title"], ph["desc"]] for ph in rep["phases"]])
        n_line = sum(bool(ph.get("line")) for ph in rep["phases"])
        t("subtitles_3d", "A" if n_line == len(rep["phases"]) else "B",
          "" if n_line == len(rep["phases"]) else f"{len(rep['phases']) - n_line} phase(s) show the fact description (no line)")
    else:
        why = edl["decisions"].get("replay_skip") or ("no calibrated wide window ≥ 3 s" if not edl["decisions"].get("replay_window")
                                                       else (run.mods.get("replay").reason if run.mods.get("replay") else "M4 failed"))
        t("replay_3d", "C", why)
    shot = edl.get("shot")
    fr = lay.get("freeze") or {}
    if "freeze" in have:
        t("freeze", "A" if fr.get("spotlight") else "B", "" if fr.get("spotlight") else "shooter box unknown: dim only")
        t("xfg_card", "A" if shot.get("xfg") is not None else "C",
          "official NBA xFG" if shot.get("xfg") is not None else "no official xFG for this shot (never invented)")
    else:
        t("freeze", "C", "no confident shot release" if not shot else "render failed")
        t("xfg_card", "C", "no shot")
    fx = ((lay.get("resume") or {}).get("effects")) or {}
    if "resume" in have:
        t("resume", "A")
        t("ball_trail", "A" if fx.get("trail_frames") else "C",
          "" if fx.get("trail_frames") else "no measured ball path after release")
        t("light_wave", "A" if fx.get("wave_frames") else "C",
          "" if fx.get("wave_frames") else ("shot missed" if shot and shot.get("made") is False else "make/rim not established"))
    else:
        for k in ("resume", "ball_trail", "light_wave"):
            t(k, "C", "no shot")
    if narration:
        q = narration.get("quality", {})
        dropped, trimmed = getattr(run, "dropped_lines", []), getattr(run, "trimmed_lines", [])
        tier = q.get("tier", "B") if not dropped else "B"
        t("narration", tier, ("" if tier == "A" else "see narration quality")
          + (f"; {len(dropped)} line(s) dropped for unsupported claims" if dropped else "")
          + (f"; {len(trimmed)} unsupported clause(s) cut" if trimmed else ""),
          lines=q.get("lines"), dropped_lines=dropped, trimmed_lines=trimmed, track_cer=q.get("track_cer"), all_in_window=q.get("all_in_window"))
    else:
        r = run.mods.get("narration")
        t("narration", "C", (r.reason if r else "not run") + "; live segment keeps the broadcast sound")
    rep_seg = director.seg(edl, "replay")
    n_ann = len(edl["decisions"].get("annotations") or [])
    if rep_seg:
        no_opt = any("no --annotations" in n for n in run.notes)
        t("annotations_3d", "A" if n_ann and not no_opt else "C",
          f"{n_ann} measured annotations" + ("; M4 cannot take them yet" if no_opt else ""))


def make_film(clip: str, out_dir: str, *, reuse: Optional[str] = None, use_llm: bool = True,
              use_moss_vl: bool = True, skip: tuple = ()) -> dict:
    t_start = time.monotonic()
    run = Run(clip, Path(out_dir), reuse=Path(reuse) if reuse else None, use_llm=use_llm,
              use_moss_vl=use_moss_vl, skip=skip)
    models.LEDGER.clear()
    report: dict = {"clip": run.video, "out": str(run.out), "ok": False}
    try:
        info = probe(run.video)
        video = {"path": run.video, "fps": info.fps, "n_frames": info.n_frames, "duration": info.duration,
                 "width": info.width, "height": info.height}
        data = run.timed("modules_analysis", run.analyse)
        shots = run.timed("shots", lambda: classify_rule(build_shots(run.video, info.n_frames, info.fps, data["tracks"]),
                                                         data["tracks"], data["court"], info.height))
        if run.use_moss_vl:
            from .moss import classify_shots
            shots = run.timed("moss_vl_shots", classify_shots, run.video, shots, info.fps)
        replay_ok = bool(data["tracks"] and data["court"]) and "replay" not in run.skip
        edl = run.timed("director", director.build_edl, video, shots, data["tracks"], data["court"],
                        data["identity"], data["events"], replay_available=replay_ok, use_llm=use_llm)
        _dump(run.work / "edl_planned.json", edl)
        run.data_events = data["events"]
        from .pipeline import _m4_supports
        if director.seg(edl, "replay") and _m4_supports("--timeline"):
            # Audio-first: narrate, fit the 3D pacing to the lines, then let M4 render exactly that pacing.
            narration = run.timed("module_narration", run.narrate, edl)
            pieces = run.timed("retime", run.retime, edl, narration)
            rj = run.timed("module_replay", run.replay3d, edl, data, True)
        else:
            with ThreadPoolExecutor(2) as ex:
                f_rep = ex.submit(run.timed, "module_replay", run.replay3d, edl, data)
                f_nar = ex.submit(run.timed, "module_narration", run.narrate, edl)
                rj, narration = f_rep.result(), f_nar.result()
            pieces = run.timed("retime", run.retime, edl, narration)
        if director.seg(edl, "replay") and rj is None:     # M4 unavailable: transition cuts to the freeze
            edl["decisions"]["replay_skip"] = run.mods["replay"].reason if "replay" in run.mods else "M4 not run"
            edl_pieces_drop = {"replay"}
        else:
            edl_pieces_drop = set()
        if edl_pieces_drop:
            pieces = _drop_segments(edl, pieces, edl_pieces_drop)
        paths = run.render(edl, data, rj, narration)
        failed = {k for k, v in run.layouts.items() if isinstance(v, dict) and "error" in v}
        pieces = _drop_segments(edl, pieces, failed)
        live_audio = None
        if not pieces:                                       # no commentary: keep the broadcast sound
            try:
                live_audio = read_wav(run.video) if info.has_audio else None
            except RuntimeError:
                live_audio = None
        mix, astats = run.timed("audio", build_mix, edl["duration"], edl["segments"],
                                intro=promo.fixed_audio("intro"), outro=promo.fixed_audio("outro"),
                                sfx=promo.transition_sfx(run.work / "transition", director.BRIDGE_S)
                                if director.seg(edl, "bridge") else None,
                                pieces=pieces, live_audio=live_audio, body_offset=director.INTRO_S)
        film = run.timed("compose", run.compose, paths, edl, mix)
        _dump(run.work / "edl.json", edl)
        tiers(run, edl, data, rj, narration)
        fz = director.seg(edl, "freeze")
        rect = (getattr(run, "layouts", {}).get("freeze") or {}).get("card_rect")
        if fz and rect and run.use_moss_vl and "readback" not in run.skip:
            from .moss import readback_card
            card = next(o for o in fz["overlays"] if o["type"] in ("xfg_card", "name_strip"))
            qa = run.timed("moss_vl_readback", readback_card, str(film), fz["film"][0] + 2.5, rect,
                           card["name"], card["xfg"])
            report["readback_qa"] = qa
            if qa.get("checked") and not (qa["name_ok"] and qa["number_ok"]):
                run.notes.append(f"MOSS-VL read-back mismatch on the xFG card: {qa.get('heard')}")
        fi = probe(film)
        report.update(ok=True, film=str(film), duration=round(fi.duration, 3),
                      structure=[[s["kind"], *s["film"]] for s in edl["segments"]],
                      audio=astats, video_segments_s=round(_durations(paths), 3))
    except Exception as e:                                   # last resort: never leave without a report
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()[-3000:]
        logger.exception("film failed")
    report["elements"] = run.elements
    report["modules"] = {k: v.as_dict() for k, v in run.mods.items()}
    report["timing_s"] = {**run.timing, "total": round(time.monotonic() - t_start, 1)}
    report["model_calls"] = {"director": models.ledger_summary(), "modules": _module_calls(run)}
    report["notes"] = run.notes
    _dump(run.out / "report.json", report)
    return report


def _module_calls(run: Run) -> dict:
    """Paid-call counts reported by the modules themselves (where they expose them)."""
    out = {}
    ident = _safe(run.work / "identity.json")
    if ident:
        out["m3"] = (ident.get("meta") or {}).get("api")
    tr = _safe(run.work / "tracks.json")
    if tr:
        out["m1"] = {k: v for k, v in (tr.get("meta") or {}).items() if "vlm" in k.lower() or "model" in k.lower()}
    ev = _safe(run.work / "events.json")
    if ev:
        out["m5"] = {k: v for k, v in (ev.get("meta") or {}).items() if "vlm" in k.lower()}
    nar = _safe(run.work / "narration" / "narration.json")
    if nar:
        out["m6"] = {"script": nar.get("script_log"), "lines": (nar.get("quality") or {}).get("lines")}
    return out


def _safe(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError):
        return None
