"""End-to-end: clip → modules → director → narration → audio-first re-timing → render → film.mp4 + report.json.

Nothing here raises for a bad input stage: every module and every element degrades (tier B) or is
omitted (tier C) with a reason in report.json. The worst case is intro + live + outro.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from . import director, models, season
from .audio import build_mix, voice_pieces
from .media import SR, frame_at, probe, read_wav, write_wav
from .modules import ModuleResult, run_module
from .render import promo, segments
from .render.finish import FinishEffects
from .render.freeze import FreezeSpec
from .render.nameplates import NameplateLayer
from .render.style import TEAM_ZH
from .shots import build_shots, classify_rule

logger = logging.getLogger(__name__)
CONTRACTS = ("tracks", "court", "identity", "events")


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError):
        return None


def _dump(p: Path, d: dict) -> Path:
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, p)
    return p


class Run:
    def __init__(self, video: str, out: Path, *, reuse: Optional[Path] = None, use_llm: bool = True,
                 use_moss_vl: bool = False, skip: tuple = ()):
        self.video = str(Path(video).resolve())
        self.out = Path(out).resolve()
        self.work = self.out / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.reuse, self.use_llm, self.use_moss_vl, self.skip = reuse, use_llm, use_moss_vl, set(skip)
        self.mods: dict[str, ModuleResult] = {}
        self.timing: dict[str, float] = {}
        self.elements: dict[str, dict] = {}
        self.notes: list[str] = []
        self.dropped_lines: list[dict] = []
        self.trimmed_lines: list[dict] = []
        self.data_events: Optional[dict] = None

    # ------------------------------------------------------------ helpers
    def tier(self, name: str, tier: str, reason: str = "", **extra) -> None:
        self.elements[name] = {"tier": tier, "reason": reason, **extra}

    def timed(self, name: str, fn, *a, **kw):
        t0 = time.monotonic()
        try:
            return fn(*a, **kw)
        finally:
            self.timing[name] = round(time.monotonic() - t0, 1)

    def _reused(self, name: str) -> Optional[Path]:
        """--reuse DIR: an existing contract JSON replaces running that module (always, and recorded)."""
        src = self.reuse / f"{name}.json" if self.reuse else None
        if not src or not src.exists():
            return None
        dst = self.work / f"{name}.json"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            tmp = dst.with_suffix(".json.part")
            shutil.copy(src, tmp)
            os.replace(tmp, dst)
        self.notes.append(f"{name}.json reused from {self.reuse}")
        return dst

    def module(self, name: str, mod: str, args: list, inputs: list, timeout: float = 3600) -> Optional[dict]:
        out = self.work / f"{name}.json"
        if name in self.skip:
            self.mods[name] = ModuleResult(name, False, reason="skipped by option")
            return None
        if self._reused(name):
            self.mods[name] = ModuleResult(name, True, cached=True, reason="reused", outputs={"main": out})
            return _load(out)
        res = run_module(name, mod, [*args, "-o", out], out, self.work, inputs=[Path(self.video), *inputs],
                         timeout=timeout)
        self.mods[name] = res
        doc = _load(out) if res.ok else None
        if res.ok and doc is None:
            res.ok, res.reason = False, "output is not valid JSON"
        return doc

    # ------------------------------------------------------------ stages
    def analyse(self) -> dict:
        v, w = self.video, self.work
        with ThreadPoolExecutor(2) as ex:
            ft = ex.submit(self.module, "tracks", "m1", [v, "--threads", "16"], [], 5400)
            fc = ex.submit(self.module, "court", "m2", [v], [], 3600)
            tracks, court = ft.result(), fc.result()
        identity = events = None
        if tracks:
            identity = self.module("identity", "m3", [v, "--tracks", w / "tracks.json"], [w / "tracks.json"], 2400)
            args = [v, "--tracks", w / "tracks.json", "--cache", w / "m5_cache"]
            ins = [w / "tracks.json"]
            for name, flag in (("court", "--court"), ("identity", "--identity")):
                if (w / f"{name}.json").exists() and (court if name == "court" else identity):
                    args += [flag, w / f"{name}.json"]
                    ins.append(w / f"{name}.json")
            events = self.module("events", "m5", args, ins, 2400)
        return {"tracks": tracks, "court": court, "identity": identity, "events": events}

    def narrate(self, edl: dict) -> Optional[dict]:
        if "narration" in self.skip:
            self.mods["narration"] = ModuleResult("narration", False, reason="skipped by option")
            return None
        doc_path = _dump(self.work / "narration_events.json", edl["narration_doc"])
        ndir = self.work / "narration"
        ndir.mkdir(exist_ok=True)
        args = ["--events", doc_path]
        if (self.work / "identity.json").exists():
            args += ["--identity", self.work / "identity.json"]
        out = ndir / "narration.json"
        res = run_module("narration", "m6", [*args, "-o", out], out, self.work, inputs=[doc_path], timeout=2400)
        if not res.ok:
            # A take truncated by an earlier kill/outage makes M6 crash on re-read: drop undecodable takes, retry.
            bad = [w for w in (ndir / "narration_work" / "takes").glob("*.wav")
                   if subprocess.run(["ffprobe", "-v", "error", str(w)], capture_output=True).returncode != 0]
            for j in (ndir / "narration_work").rglob("*.json"):       # truncated script/ASR caches
                try:
                    json.loads(j.read_text())
                except (OSError, ValueError):
                    bad.append(j)
            for w in bad:
                w.unlink()
            if bad:
                self.notes.append(f"removed {len(bad)} corrupt cached M6 file(s); narration retried")
                res = run_module("narration", "m6", [*args, "-o", out], out, self.work, inputs=[doc_path],
                                 timeout=2400)
        self.mods["narration"] = res
        doc = _load(out) if res.ok else None
        if doc is not None:
            vo = ndir / doc.get("vo", "narration_vo.wav")
            doc["_vo"] = str(vo) if vo.exists() else None
        return doc

    def replay3d(self, edl: dict, data: dict, use_timeline: bool = False) -> Optional[dict]:
        rep = director.seg(edl, "replay")
        if rep is None or "replay" in self.skip:
            if "replay" in self.skip:
                self.mods["replay"] = ModuleResult("replay", False, reason="skipped by option")
            return None
        ev = copy.deepcopy(data["events"] or {"events": []})
        for ph in rep["phases"]:                # phase labels drive M4's minimap panel
            ev.setdefault("events", []).append({"t": round(ph["src"][0] + 0.3, 3), "type": "other", "actors": [],
                                                "conf": 0.9, "label_zh": ph["title"], "note": ph["desc"]})
        ev_path = _dump(self.work / "events_m4.json", ev)
        out = self.work / "replay" / "replay.mp4"
        out.parent.mkdir(exist_ok=True)
        w0, w1 = rep["source"]
        args = [self.video, "--tracks", self.work / "tracks.json", "--court", self.work / "court.json",
                "--events", ev_path, "--workers", "3"]
        if use_timeline:      # M4 renders the director's audio-first pacing directly (no frame resampling)
            tl = _dump(self.work / "timeline_m4.json", {"timeline": rep["time_map"]})
            args += ["--timeline", json.dumps(rep["time_map"])]
            edl["decisions"]["m4_timeline"] = str(tl)
        else:
            args += ["--start", w0, "--end", w1]
        if (self.work / "identity.json").exists():
            args += ["--identity", self.work / "identity.json"]
        title = edl["decisions"].get("replay_title")
        if title and _m4_supports("--title"):
            args += ["--title", title]
        annos = edl["decisions"].get("annotations") or []
        if annos:
            ap = _dump(self.work / "annotations_m4.json", {"annotations": annos})
            if _m4_supports("--annotations"):
                args += ["--annotations", ap]
            else:
                self.notes.append(f"{len(annos)} 3D annotations prepared (work/annotations_m4.json); M4 has no --annotations option yet")
        bin_dir = promo.chromium_path_dir(self.work)      # M4 looks for `chromium` on PATH
        env = {"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"} if bin_dir else None
        res = run_module("replay", "m4", [*args, "-o", out], out, self.work, inputs=[ev_path, self.work / "tracks.json"],
                         timeout=3600, env=env)
        self.mods["replay"] = res
        rj = _load(out.with_name("replay.json")) if res.ok else None
        if rj is not None and not rj.get("ok"):
            res.ok, res.reason = False, f"M4 declined: {rj.get('reason')}"
            return None
        return rj

    def check_claims(self, edl: dict, pieces: list, lines: list[dict]) -> tuple[list, list[dict]]:
        """Cut unsupported clauses / drop unsupported lines (claims.py); recorded in report.json."""
        from . import claims
        pieces, lines, dropped, trimmed = claims.check(edl, self.data_events, pieces, lines)
        for d in dropped:
            self.dropped_lines.append(d)
            self.notes.append(f"commentary line dropped (unsupported {'/'.join(d['unsupported'])}): {d['text']}")
        for t in trimmed:
            self.trimmed_lines.append(t)
            self.notes.append(f"commentary clause cut (unsupported {'/'.join(t['unsupported'])}): {t['from']} → {t['to']}")
        return pieces, lines

    # ------------------------------------------------------------ audio-first 3D timing
    def retime(self, edl: dict, narration: Optional[dict]) -> list[tuple[float, np.ndarray]]:
        """Phase lengths follow their narration line (speed 0.5–1.0×); later segments and lines shift."""
        body = director.INTRO_S
        lines = [l for l in (narration or {}).get("lines", []) if l.get("text")]
        vo = read_wav(narration["_vo"]) if narration and narration.get("_vo") else None
        pieces = voice_pieces(vo, lines) if vo is not None else []
        pieces, lines = self.check_claims(edl, pieces, lines)
        rep = director.seg(edl, "replay")
        shift_of = {}
        if rep:
            r0 = rep["film"][0] - body
            new_knots, out_t, delta_acc = [], 0.0, 0.0
            for i, (ph, (ft, _)) in enumerate(zip(rep["phases"], rep["time_map"])):
                pref = director.REPLAY_FINAL_SPEED if i == len(rep["phases"]) - 1 else director.REPLAY_PREFERRED_SPEED
                p0, p1 = r0 + ft, r0 + ft + ph["film_len"]
                mine = [l for l in lines if p0 - 0.2 <= l["start"] < p1 - 0.2]
                src = ph["src"][1] - ph["src"][0]
                need = (max(l["end"] for l in mine) - p0 + 0.45) if mine else ph["film_len"]
                new_len = float(np.clip(max(need, src / pref),
                                        src / director.REPLAY_MAX_SPEED, src / director.REPLAY_MIN_SPEED))
                for l in mine:
                    shift_of[id(l)] = (r0 + out_t) - p0
                ph["film_len_planned"], ph["film_len"] = ph["film_len"], round(new_len, 3)
                ph["line"] = "".join(l["text"] for l in mine) if sum(len(l["text"]) for l in mine) <= 26 else (
                    mine[0]["text"] if mine else "")
                new_knots.append([round(out_t, 3), ph["src"][0]])
                out_t += new_len
            w1 = rep["source"][1]
            new_knots += [[round(out_t, 3), w1], [round(out_t + director.REPLAY_END_HOLD, 3), w1]]
            delta = (out_t + director.REPLAY_END_HOLD) - (rep["film"][1] - rep["film"][0])
            rep["time_map"] = new_knots
            rep["film"][1] = round(rep["film"][1] + delta, 3)
            after = False
            for s in edl["segments"]:
                if after:
                    s["film"] = [round(s["film"][0] + delta, 3), round(s["film"][1] + delta, 3)]
                after = after or s is rep
            edl["duration"] = round(edl["duration"] + delta, 3)
            rep_end_body = rep["film"][1] - body - delta
            for l in lines:
                if id(l) not in shift_of and l["start"] >= rep_end_body - 0.2:
                    shift_of[id(l)] = delta
            edl["decisions"]["replay_retime_delta_s"] = round(delta, 3)
        placed, prev_end = [], -1.0
        for l, piece, t0 in pieces:
            s = shift_of.get(id(l), 0.0)
            start = t0 + s
            if start < prev_end - 0.05:                   # never overlap two lines
                start = prev_end - 0.05
            placed.append((body + start, piece))
            prev_end = start + len(piece) / SR
            l["film_start"], l["film_end"] = round(body + l["start"] + s, 3), round(body + l["end"] + s, 3)
        return placed

    # ------------------------------------------------------------ render
    def render(self, edl: dict, data: dict, rj: Optional[dict], narration: Optional[dict]) -> list[Path]:
        segdir = self.work / "segments"
        segdir.mkdir(exist_ok=True)
        info = edl["source"]
        fps = float(info["fps"])
        sx, sy = 1920 / info["width"], 1080 / info["height"]
        tracks, identity, events = data["tracks"], data["identity"], data["events"]
        shot = edl.get("shot")
        paths, layouts = [], {}
        for i, s in enumerate(edl["segments"]):
            out = segdir / f"{i:02d}_{s['kind']}.mp4"
            k = s["kind"]
            try:
                if k in ("intro", "outro"):
                    layouts[k] = self.timed(f"render_{k}", segments.render_fixed, promo.fixed_clip(k), out)
                elif k == "live":
                    plates = None
                    if tracks:
                        hero = shot and shot["shooter_track"]
                        g = season.gravity(identity, hero)
                        self.hero_stat = g
                        plates = NameplateLayer(tracks, identity, edl["shots"], hero_tid=hero,
                                                hero_stat=(g["caption"], g["value"]) if g else ("", ""))
                    layouts[k] = self.timed("render_live", segments.render_live, self.video, s, out, fps, plates)
                elif k == "bridge":
                    frame = frame_at(self.video, max(0, int(info["n_frames"]) - 1))
                    self.timed("render_bridge", promo.render_transition, frame, out, s["film"][1] - s["film"][0],
                               self.work / "transition")
                elif k == "replay":
                    subs = []
                    u = 0.0
                    for ph in s["phases"]:
                        # Subtitles follow the phases (as in the reference); a phase whose line was dropped
                        # keeps its fact-derived description, silently.
                        text = ph.get("line") or ph.get("desc") or ""
                        if text:
                            subs.append((u + 0.1, u + ph["film_len"] - 0.07, text))
                        u += ph["film_len"]
                    layouts[k] = self.timed("render_replay", segments.render_replay, str(self.work / "replay/replay.mp4"),
                                            rj["timeline_source_pts"], s, out, subs)
                elif k == "freeze":
                    layouts[k] = self.timed("render_freeze", segments.render_freeze, s, out,
                                            self.freeze_spec(s, tracks, events, identity, fps, sx, sy))
                elif k == "resume":
                    es = (events or {}).get("shot") or {}
                    fx = FinishEffects(es.get("ball_path_img") or [], es.get("rim_img") or [], shot.get("made"),
                                       shot.get("made_t"), (sx, sy),
                                       trail_end_t=shot.get("made_t") or None)
                    if fx.made:
                        fx.wave_film_t = director._map_inv(s["time_map"], shot["made_t"]) + 0.07
                    layouts[k] = self.timed("render_resume", segments.render_resume, self.video, s, out, fps, fx)
                paths.append(out)
            except Exception as e:                       # one bad segment must not stop the film
                logger.exception("segment %s failed", k)
                self.notes.append(f"segment {k} dropped: {type(e).__name__}: {e}")
                layouts[k] = {"error": f"{type(e).__name__}: {e}"}
        self.layouts = layouts
        return paths

    def freeze_spec(self, s, tracks, events, identity, fps, sx, sy) -> FreezeSpec:
        es = (events or {}).get("shot") or {}
        # Freeze 0.1 s before M5's release (ball leaving the hand): the ball is still at the fingertips.
        rf = max(0, int(es.get("release_f") or round(s["source_t"] * fps)) - round(0.1 * fps))
        s["freeze_f"] = rf
        box = None
        tid = es.get("shooter_track")
        for tr in (tracks or {}).get("tracks", []):
            if tr["track_id"] == tid:
                near = min(tr["frames"], key=lambda fr: abs(fr["f"] - rf), default=None)
                if near and abs(near["f"] - rf) <= 3:
                    b = near["bbox"]
                    box = (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)

        def nearest(path, t, tol):
            p = min(path or [], key=lambda p: abs(p[0] - t), default=None)
            return (p[1] * sx, p[2] * sy) if p and abs(p[0] - t) <= tol else None
        rt = es.get("release_t") or s["source_t"]
        card = next(o for o in s["overlays"] if o["type"] in ("xfg_card", "name_strip"))
        return FreezeSpec(frame_at(self.video, rf), box, nearest(es.get("ball_path_img"), rt, 0.15),
                          nearest(es.get("rim_img"), rt, 1.0), card["name"], card["team"], card["xfg"])

    def compose(self, paths: list[Path], edl: dict, mix: np.ndarray) -> Path:
        listing = self.work / "concat.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in paths))
        silent = self.work / "picture.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy",
                        silent], check=True)
        wav = write_wav(self.work / "film_audio.wav", mix)
        film = self.out / "film.mp4"
        tmp = self.out / "film.part.mp4"
        cover = promo.cover()
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", wav]
        maps = ["-map", "0:v:0", "-map", "1:a:0"]
        if cover.exists():
            cmd += ["-i", cover]
            maps += ["-map", "2:v:0", "-c:v:1", "mjpeg", "-disposition:v:1", "attached_pic"]
        cmd += [*maps, "-c:v:0", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{edl['duration']:.3f}", "-movflags", "+faststart", tmp]
        subprocess.run([str(c) for c in cmd], check=True)
        os.replace(tmp, film)
        return film


def _m4_supports(flag: str) -> bool:
    from .modules import module_dir, module_python
    try:
        out = subprocess.run([module_python("m4"), "-m", "agentdunker.replay", "--help"], cwd=module_dir("m4"),
                             capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return flag in out


def _durations(paths: list[Path]) -> float:
    total = 0.0
    for p in paths:
        try:
            total += probe(p).duration
        except (subprocess.CalledProcessError, StopIteration, ValueError):
            pass
    return total
