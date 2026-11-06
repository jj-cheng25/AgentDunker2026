"""Director: module outputs → edit decision list (``edl.json``).

Deterministic rules follow REFERENCE_SPEC §1: intro 6 s → live 1.0× with nameplates → "3D 战术解读"
transition → 3D replay split into tactical phases → shot freeze (spotlight, official xFG card) → resume
(0.5× to the make, then ≈0.87×, light wave on a make) → outro 6 s. Every decision is conditioned on data
confidence; missing data removes an element instead of failing. Phase titles are the only text written by
a model (``gemini-3.8-flash``), from verified events only, with a template fallback.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from . import models
from .render.style import TEAM_ZH

logger = logging.getLogger(__name__)

INTRO_S = OUTRO_S = 6.0
BRIDGE_S = 4.8
FREEZE_S, FREEZE_NO_XFG_S = 5.2, 3.2
# 3D pacing. The reference runs its phases at 0.52/0.61/0.67/0.62× and slows the last (shot build-up) one
# further; the brief caps slow motion at 2×, so: 0.56× per phase, 0.5× for the last, longer only when the
# phase's commentary line needs it (never slower than 0.5×, never faster than 1.0×).
REPLAY_SPEED = REPLAY_PREFERRED_SPEED = 0.56
REPLAY_FINAL_SPEED = 0.5
REPLAY_MIN_SPEED, REPLAY_MAX_SPEED = 0.5, 1.0
REPLAY_END_HOLD = 1.0
REPLAY_MIN_S = 3.0
PRE_RELEASE = 0.65
RESUME_SLOW, RESUME_FAST = 0.5, 0.87
RESUME_TAIL = 4.5             # source seconds after the make kept in the resume segment
FADE_OUT_S = 0.8
MIN_PHASE_S, MAX_PHASE_S, MAX_PHASES = 1.8, 4.0, 5
EVENT_CONF = 0.35


# ---------------------------------------------------------------- facts from module outputs

def player_names(identity: Optional[dict]) -> dict[int, dict]:
    """track id → {short, full, jersey, team_abbr} for confident identities only."""
    out = {}
    for tid, p in ((identity or {}).get("players") or {}).items():
        conf = float(p.get("conf") or 0)
        if conf < 0.5:
            continue
        short = p.get("name_zh_short") or ((p.get("name_zh") or "").split("·")[-1]) if conf >= 0.6 else ""
        out[int(tid)] = {"short": short, "full": (p.get("name_zh") or short) if conf >= 0.6 else "",
                         "jersey": str(p.get("jersey") or ""), "team_abbr": p.get("team_abbr"),
                         "conf": conf}
    return out


def label(tid: Optional[int], names: dict, fallback: str = "持球人") -> str:
    p = names.get(tid) if tid is not None else None
    if p and p["short"]:
        return p["short"]
    if p and p["jersey"]:
        return f"{p['jersey']}号"
    return fallback


def shot_info(events: Optional[dict], n_dur: float) -> Optional[dict]:
    s = (events or {}).get("shot") or {}
    rt = s.get("release_t")
    if rt is None or not (0 < float(rt) < n_dur) or float(s.get("conf") or 0) < 0.3:
        return None
    return s


def replay_window(shots: list[dict], court: Optional[dict], fps: float, release_t: Optional[float],
                  n_frames: int) -> Optional[tuple[float, float, str]]:
    """Longest calibrated stretch of a wide shot (gaps ≤ 0.35 s bridged, 0.5 s after a cut skipped), ending
    0.65 s before release when the release is inside it. None when shorter than 3 s."""
    if not court:
        return None
    ok = np.zeros(n_frames, bool)
    for fr in court.get("frames", []):
        if 0 <= fr["f"] < n_frames and fr.get("ok"):
            ok[fr["f"]] = True
    best = None
    for s in shots:
        if s.get("kind") != "wide":
            continue
        a = s["start_f"] + (int(0.5 * fps) if s["start_f"] > 0 else 0)
        b = s["end_f"]
        if release_t is not None and a / fps < release_t <= (b + 1) / fps:
            b = min(b, int((release_t - PRE_RELEASE) * fps))
        f = a
        while f <= b:
            if not ok[f]:
                f += 1
                continue
            g, last = f, f
            while g <= b and (ok[g] or g - last <= int(0.35 * fps)):
                if ok[g]:
                    last = g
                g += 1
            t0, t1 = f / fps, last / fps
            # Prefer the run that ends nearest the shot, then the longest.
            score = (t1 - t0) + (5.0 if release_t is not None and abs(release_t - PRE_RELEASE - t1) < 0.3 else 0)
            if t1 - t0 >= REPLAY_MIN_S and (best is None or score > best[0]):
                best = (score, round(t0, 3), round(t1, 3))
            f = g
    if best is None:
        return None
    return best[1], best[2], "court-ok wide run"


# ---------------------------------------------------------------- phases

def _ball_holder(events: Optional[dict], t0: float, t1: float) -> Optional[int]:
    dur: dict[int, float] = {}
    for seg in (events or {}).get("ball_state", []):
        h = seg.get("holder")
        if h is None or float(seg.get("conf") or 0) < EVENT_CONF:
            continue
        ov = min(t1, seg["t1"]) - max(t0, seg["t0"])
        if ov > 0:
            dur[h] = dur.get(h, 0) + ov
    return max(dur, key=dur.get) if dur else None


def split_phases(w0: float, w1: float, events: Optional[dict], facts: list) -> list[dict]:
    """Phase boundaries at measured facts (pass/screen/cut/drive/closeout/help), each 1.8–4 s, at most 5."""
    order = {"drive": 0, "pass": 1, "screen": 2, "space": 3, "cut": 4, "closeout": 5, "help": 6}
    cands = sorted([f for f in facts if w0 + MIN_PHASE_S <= f.t0 <= w1 - MIN_PHASE_S * 0.6],
                   key=lambda f: (order.get(f.type, 9), -f.conf))
    bounds = [w0, w1]
    for f in cands:
        cand = sorted(bounds + [f.t0 - 0.3])
        if len(cand) - 1 > MAX_PHASES or any(y - x < MIN_PHASE_S for x, y in zip(cand, cand[1:])):
            continue
        bounds = cand
    while len(bounds) - 1 < MAX_PHASES:
        longest, i = max((y - x, i) for i, (x, y) in enumerate(zip(bounds, bounds[1:])))
        if longest <= MAX_PHASE_S or longest / 2 < MIN_PHASE_S:
            break
        bounds.insert(i + 1, (bounds[i] + bounds[i + 1]) / 2)
    phases = []
    for x, y in zip(bounds, bounds[1:]):
        mine = [f for f in facts if x - 0.35 <= f.t0 < y - 0.3 or (f.t0 < x - 0.35 and f.t1 > x + 0.8)]
        phases.append({"src": [round(float(x), 3), round(float(y), 3)], "facts": [f.as_dict() for f in mine],
                       "events": [e for e in (events or {}).get("events", []) if x - 0.3 <= e["t"] < y - 0.3
                                  and float(e.get("conf") or 0) >= EVENT_CONF and e.get("type") != "shot"],
                       "holder": _ball_holder(events, x, y)})
    return phases


FACT_TITLE = {"pass": "传球转移", "screen": "掩护配合", "cut": "无球切入", "drive": "持球突破",
              "closeout": "防守扑出", "help": "协防收缩", "space": "空间展开"}


def template_title(ph: dict, names: dict, last: bool, shooter: Optional[int] = None, seen_drive: bool = False
                   ) -> tuple[str, str, list]:
    fs = ph["facts"]
    if not fs:
        holder = label(ph["holder"], names) if ph["holder"] is not None else "持球人"
        return ("准备出手" if last else "持球观察"), f"{holder}持球，队友保持站位。", []
    key = next((f for t in ("drive", "pass", "screen", "space", "cut", "closeout", "help") for f in fs if f["type"] == t))
    title = FACT_TITLE[key["type"]]
    if key["type"] == "pass" and key["actors"][1:] == [shooter]:
        title = "回传组织"
    if key["type"] == "drive" and seen_drive:
        title = "再次启动"
    return title, "，".join(f["text"] for f in fs[:3]) + "。", [key["id"]]


def phase_titles(phases: list[dict], names: dict, use_llm: bool = True, shooter: Optional[int] = None) -> str:
    """Title + one sentence per phase. The model must cite the fact ids it used, from that phase only;
    a phase without facts or with an invalid answer keeps the template text (also fact-derived)."""
    seen = False
    for i, ph in enumerate(phases):
        ph["title"], ph["desc"], ph["evidence"] = template_title(ph, names, i == len(phases) - 1, shooter, seen)
        seen = seen or any(f["type"] == "drive" for f in ph["facts"])
        ph["text_source"] = "template"
    if not use_llm or not any(ph["facts"] for ph in phases):
        return "template"
    allowed = sorted({label(t, names) for t in names if label(t, names, "")})
    lines = []
    for i, ph in enumerate(phases):
        fs = "；".join(f"[{f['id']}] {f['text']}" for f in ph["facts"]) or "无"
        lines.append(f"阶段{i + 1}（源片 {ph['src'][0]:.1f}–{ph['src'][1]:.1f} 秒）：{fs}")
    prompt = ("你是NBA战术分析编辑。下面是一次进攻在3D复盘中的几个阶段，每个阶段列出了从球员位置测得的战术事实，"
              "方括号里是事实编号。为每个阶段写一个战术标题（2–6个汉字，例如 掩护接应、回传重新组织、空间展开、"
              "突破调整、再次启动）和一句说明（不超过26个汉字）。规则：\n"
              "1. 标题和说明只能描述该阶段列出的事实，并在 evidence 中写出用到的事实编号（必须来自该阶段）。\n"
              "2. 不要写事实里没有的动作（如虚晃、变向、转身、后撤步、扣篮）、位置、比分或数字。\n"
              "3. 没有事实的阶段，evidence 写空列表，标题写“持球观察”或“落位组织”。\n"
              "4. 标题只写战术名称，不要出现人名或号码；人名只能写在说明里。相邻阶段的标题不要重复同一个词。\n"
              f"5. 可用人名只有：{'、'.join(allowed) or '无（用持球人/队友/防守人）'}。\n\n" + "\n".join(lines))
    schema = {"type": "object", "properties": {"phases": {"type": "array", "items": {
        "type": "object", "properties": {"title": {"type": "string"}, "desc": {"type": "string"},
                                         "evidence": {"type": "array", "items": {"type": "string"}}},
        "required": ["title", "desc", "evidence"]}}}, "required": ["phases"]}
    ans = models.ask_json(prompt, schema, "phase-titles")
    rows = (ans or {}).get("phases") or []
    if len(rows) != len(phases):
        return "template"
    from .facts import unsupported_words
    for ph, row in zip(phases, rows):
        title, desc = str(row.get("title", "")).strip(), str(row.get("desc", "")).strip()
        ev = [str(x).strip("[] ") for x in row.get("evidence") or []]
        ids = {f["id"] for f in ph["facts"]}
        support = {f["type"] for f in ph["facts"] if f["id"] in ev}
        if not (2 <= len(title) <= 8 and 4 <= len(desc) <= 32):
            continue
        if ph["facts"] and (not ev or not set(ev) <= ids):
            continue                                  # must cite this phase's facts
        if not ph["facts"] and ev:
            continue
        if re.search(r"\d", re.sub(r"\d+号", "", title + desc)) or unsupported_words(title + desc, support):
            continue
        if any(m not in allowed for m in re.findall(r"\d+号", title + desc)):
            continue
        if re.search(r"\d|号", title) or any(n and n in title for n in allowed):
            continue                                  # titles name the tactic, not the player
        ph["title"], ph["desc"], ph["evidence"], ph["text_source"] = title, desc, ev, "llm"
    n_llm = sum(ph["text_source"] == "llm" for ph in phases)
    return f"llm {n_llm}/{len(phases)}"


TITLE_MAX = 24          # M4 shrinks/ellipsizes long titles; the reference title is 20 characters


def replay_title(identity: Optional[dict], names: dict, shooter: Optional[int], facts: list) -> str:
    """3D top-left title ``<away>客场对<home> · <possession summary>``, e.g. 火箭客场对独行侠 · 杜兰特单打前的空间组织.

    The head needs M3's matched game (otherwise the two teams without home/away, or nothing); the summary is
    built only from measured facts about the shooter's possession, ending in the shot.
    """
    gm = (identity or {}).get("game_match") or {}
    me = names.get(shooter) or {}
    off = me.get("team_abbr")
    head = ""
    if gm.get("status") == "matched" and TEAM_ZH.get(gm.get("away") or "") and TEAM_ZH.get(gm.get("home") or ""):
        head = f"{TEAM_ZH[gm['away']]}客场对{TEAM_ZH[gm['home']]}"
    elif off and TEAM_ZH.get(off):
        other = {p.get("team_abbr") for p in names.values()} - {off, None}
        if len(other) == 1 and TEAM_ZH.get(next(iter(other))):
            head = f"{TEAM_ZH[off]}对{TEAM_ZH[next(iter(other))]}"
    who = label(shooter, names, "") if shooter is not None else ""
    mine = lambda typ: [f for f in facts if f.type == typ and (                       # noqa: E731
        int(f.numbers.get("holder", -1)) == shooter if typ == "space" else (f.actors[-1] if typ == "pass" else f.actors[0]) == shooter)]
    last_pass = (mine("pass") or [None])[-1]
    passer = label(last_pass.actors[0], names, "") if last_pass else ""
    screens = [f for f in facts if f.type == "screen" and f.actors]
    if not who:
        return f"{head} · 3D 战术复盘" if head else ""        # "" → M4 keeps its own default title
    if mine("space"):
        tail = f"{who}单打前的空间组织"
    elif mine("cut") and last_pass and passer:
        tail = f"{passer}传球，{who}空切接应"
    elif mine("drive"):
        tail = f"{who}持球突破后出手"
    elif screens and label(screens[-1].actors[0], names, ""):
        tail = f"{label(screens[-1].actors[0], names, '')}掩护后{who}出手"
    elif last_pass and passer:
        tail = f"{passer}传球，{who}出手"
    else:
        tail = f"{who}出手前的进攻组织"
    title = f"{head} · {tail}" if head else tail
    if len(title) > TITLE_MAX and head:
        title = tail if len(tail) <= TITLE_MAX else head
    return title


# ---------------------------------------------------------------- resume

def resume_map(release_t: float, made_t: Optional[float], end_src: float) -> list[list[float]]:
    """[[segment_s, source_s], …]: 0.5× until just after the make, 0.5 s ramp, then ≈0.87×."""
    slow_end = min(end_src, (made_t if made_t is not None else release_t + 0.9) + 0.12)
    knots = [[0.0, release_t], [(slow_end - release_t) / RESUME_SLOW, slow_end]]
    ramp_src = min(end_src, slow_end + 0.5 * (RESUME_SLOW + RESUME_FAST) / 2)
    knots.append([knots[-1][0] + 0.5, ramp_src])
    if end_src > ramp_src:
        knots.append([knots[-1][0] + (end_src - ramp_src) / RESUME_FAST, end_src])
    return [[round(a, 3), round(b, 3)] for a, b in knots]


# ---------------------------------------------------------------- EDL

def build_edl(video: dict, shots: list[dict], tracks: Optional[dict], court: Optional[dict],
              identity: Optional[dict], events: Optional[dict], *, replay_available: bool = True,
              use_llm: bool = True) -> dict:
    fps, n, dur = float(video["fps"]), int(video["n_frames"]), float(video["duration"])
    names = player_names(identity)
    shot = shot_info(events, dur)
    release_t = float(shot["release_t"]) if shot else None
    made = bool(shot.get("made")) if shot and shot.get("made") is not None else None
    made_t = shot.get("made_t") if shot else None
    xfg = shot.get("xfg") if shot else None
    xfg_ok = xfg is not None and 0 < float(xfg) < 1 and (shot.get("official") or {}).get("xfg") is not None
    shooter = shot.get("shooter_track") if shot else None
    decisions: dict = {}

    segs = []
    t = 0.0

    def add(kind: str, length: float, **kw) -> dict:
        nonlocal t
        seg = {"kind": kind, "film": [round(t, 3), round(t + length, 3)], **kw}
        segs.append(seg)
        t += length
        return seg

    add("intro", INTRO_S, render="assets/promo/intro.mov")
    live_end = dur
    add("live", live_end, source=[0.0, round(live_end, 3)], speed=1.0,
        overlays=[{"type": "nameplates", "data": "tracks(M1)+identity(M3)",
                   "shots": [[s["t0"], s["t1"]] for s in shots if s.get("kind") == "wide"],
                   "hero_track": shooter}, {"type": "white_flash_in", "film": [0.0, 0.25]}])
    add("bridge", BRIDGE_S, freeze_source_t=round(max(0.0, live_end - 1.0 / fps), 3), render="transition.html",
        audio="tr_sound.py")

    window = replay_window(shots, court, fps, release_t, n) if replay_available else None
    decisions["replay_window"] = list(window) if window else None
    if window:
        w0, w1, why = window
        from . import facts as factmod
        try:
            facts = factmod.derive(tracks, court, events or {}, fps, w0, w1,
                                   lambda tid, fb: label(tid, names, fb), release_t)
        except Exception as e:                       # facts are optional; never fatal
            logger.warning("fact derivation failed: %s", e)
            facts = []
        decisions["facts"] = [f.as_dict() for f in facts]
        decisions["annotations"] = factmod.annotations(facts, w0)
        decisions["replay_title"] = replay_title(identity, names, shooter, facts)
        phases = split_phases(w0, w1, events, facts)
        decisions["phase_text"] = phase_titles(phases, names, use_llm, shooter)
        knots, out_t = [], 0.0
        for i, ph in enumerate(phases):
            a, b = ph["src"]
            ph["film_len"] = round((b - a) / (REPLAY_FINAL_SPEED if i == len(phases) - 1 else REPLAY_SPEED), 3)
            knots.append([round(out_t, 3), a])
            out_t += ph["film_len"]
        knots.append([round(out_t, 3), w1])
        knots.append([round(out_t + REPLAY_END_HOLD, 3), w1])
        add("replay", out_t + REPLAY_END_HOLD, source=[w0, w1], time_map=knots, phases=phases, window_reason=why,
            render="M4 agentdunker.replay (re-timed by source PTS)")
    if shot:
        freeze_len = FREEZE_S if xfg_ok else FREEZE_NO_XFG_S
        sh_name = (names.get(shooter) or {}).get("short", "")
        p = names.get(shooter, {})
        team = TEAM_ZH.get(p.get("team_abbr") or "", "")
        team_label = " · ".join(x for x in (team, f"{p['jersey']}号" if p.get("jersey") else "") if x)
        add("freeze", freeze_len, source_t=release_t, release_f=shot.get("release_f"),
            overlays=[{"type": "spotlight", "track": shooter},
                      {"type": "xfg_card" if xfg_ok else "name_strip", "name": sh_name, "team": team_label,
                       "xfg": float(xfg) if xfg_ok else None,
                       "xfg_source": "NBA official" if xfg_ok else None}])
        end_src = min(dur - 1.0 / fps, (made_t if made_t else release_t + 1.0) + RESUME_TAIL)
        rmap = resume_map(release_t, made_t if made else None, end_src)
        add("resume", rmap[-1][0], time_map=rmap, fade_out=FADE_OUT_S,
            overlays=[{"type": "ball_trail", "data": "events.shot.ball_path_img"},
                      {"type": "light_wave", "made": made, "made_t": made_t}])
    add("outro", OUTRO_S, render="assets/promo/outro.mov")

    edl = {"version": "agentdunker-edl/1", "fps": 30, "duration": round(t, 3),
           "source": {k: video[k] for k in ("path", "fps", "n_frames", "duration", "width", "height")},
           "shots": shots, "segments": segs, "decisions": decisions,
           "shot": None if not shot else {"release_t": release_t, "made": made, "made_t": made_t,
                                          "bank_t": shot.get("bank_t"), "shooter_track": shooter,
                                          "xfg": float(xfg) if xfg_ok else None,
                                          "xfg_source": "NBA official" if xfg_ok else None}}
    edl["narration_doc"] = narration_doc(edl, events, identity, names)
    return edl


def seg(edl: dict, kind: str) -> Optional[dict]:
    return next((s for s in edl["segments"] if s["kind"] == kind), None)


# ---------------------------------------------------------------- narration input for M6

EVENT_NOTE = {"pass": "pass", "drive": "drives toward the basket", "screen": "sets a screen",
              "cut": "cuts to the basket", "rebound": "rebound"}


def narration_doc(edl: dict, events: Optional[dict], identity: Optional[dict], names: dict) -> dict:
    """M6 input in body time (film − intro): sections + events + shot."""
    off = INTRO_S
    body = lambda ft: round(ft - off, 3)                                  # noqa: E731
    secs, evs = [], []
    for s in edl["segments"]:
        if s["kind"] in ("live", "bridge", "replay", "freeze", "resume"):
            secs.append({"start": body(s["film"][0]), "end": body(s["film"][1]), "kind": s["kind"]})
    live = seg(edl, "live")
    shot = edl.get("shot")
    tid = shot["shooter_track"] if shot else None
    # Live: opening + confident events at source time (live is 1.0× from body 0).
    for e in sorted((events or {}).get("events", []), key=lambda e: e["t"]):
        if float(e.get("conf") or 0) < EVENT_CONF or e["t"] >= live["source"][1]:
            continue
        if e["type"] == "other" and not e.get("note"):
            continue
        if shot and e["type"] != "shot" and e["t"] > shot["release_t"] + 0.2:
            continue
        acts = e.get("actors") or []
        if e["type"] == "shot":
            note = "shoots"
            if shot and shot.get("bank_t"):
                note += ", off the glass"
        elif e["type"] == "pass" and len(acts) >= 2:
            note = f"passes from {label(acts[0], names)} to {label(acts[1], names, 'teammate')}"
        else:
            note = EVENT_NOTE.get(e["type"], e["type"])
        evs.append({"t": round(e["t"], 3), "type": e["type"], "actors": acts, "conf": e.get("conf", 0.5), "note": note})
    # Measured facts (cut / drive / help / closeout) in the live part, unless an M5 event already covers them.
    for f in edl["decisions"].get("facts") or []:
        if f["type"] in ("pass", "screen") or (shot and f["t0"] > shot["release_t"]):
            continue
        if any(abs(e["t"] - f["t0"]) < 1.2 for e in evs):
            continue
        evs.append({"t": round(f["t0"], 3), "type": "drive" if f["type"] == "drive" else "other",
                    "actors": f["actors"][:1], "conf": f["conf"], "note": f"measured on court: {f['text']}"})
    if shot and shot.get("made") is not None and shot.get("made_t"):
        evs.append({"t": round(shot["made_t"], 3), "type": "other", "actors": [tid] if tid else [], "conf": 0.8,
                    "note": "the ball goes in" if shot["made"] else "the shot misses"})
    if not evs or evs[0]["t"] > 2.5:
        evs.insert(0, {"t": 0.2, "type": "other", "actors": [], "conf": 0.6, "note": "offense sets up in the half court"})
    rep = seg(edl, "replay")
    if rep:
        for ph, (ft, _) in zip(rep["phases"], rep["time_map"]):
            key = ph["facts"][0] if ph["facts"] else None
            what = "；".join(f["text"] for f in ph["facts"]) or "持球人观察，队友保持站位（无其他已测动作）"
            evs.append({"t": body(rep["film"][0] + ft) + 0.3, "type": key["type"] if key and key["type"] in
                        ("pass", "screen", "drive") else "other", "actors": (key or {}).get("actors", []),
                        "conf": 0.8, "note": f"3D replay, measured actions only: {what}. Effects on the defense (drawing, "
                                         "pulling or freezing defenders) were not measured: do not claim them",
                        "phase": ph["title"]})
    fr = seg(edl, "freeze")
    if fr:
        evs.append({"t": body(fr["film"][0]) + 0.2, "type": "shot", "actors": [tid] if tid else [], "conf": 0.9,
                    "note": "freeze frame at the release" + ("; official shot-quality card shown" if shot["xfg"] else "")})
    res = seg(edl, "resume")
    doc_shot = None
    if res and shot:
        doc_shot = {"release_t": body(res["film"][0]), "shooter_track": tid, "made": shot["made"],
                    "xfg": shot["xfg"], "points": ((events or {}).get("shot") or {}).get("points")}
        if shot.get("made") and shot.get("made_t"):
            mt = _map_inv(res["time_map"], shot["made_t"])
            evs.append({"t": body(res["film"][0] + mt) - 0.6, "type": "other", "actors": [tid] if tid else [],
                        "conf": 0.85, "note": "replay of the finish: " + ("banks it in" if shot.get("bank_t") else "it goes in")})
            if shot.get("xfg") is not None and shot["xfg"] <= 0.35 and res["film"][1] - res["film"][0] > 4.5:
                evs.append({"t": body(res["film"][0] + mt) + 1.6, "type": "other", "actors": [tid] if tid else [],
                            "conf": 0.8, "note": "slow-motion after the make: a low-probability shot was made "
                                                 "(official xFG below 35%), praise the finish without new numbers"})
    players = {}
    for t_id, p in ((identity or {}).get("players") or {}).items():
        if p.get("name_zh") and float(p.get("conf") or 0) >= 0.6:
            players[t_id] = {"jersey": p.get("jersey"), "team_abbr": p.get("team_abbr"), "name": p.get("name"),
                             "name_zh": p["name_zh"], "conf": p["conf"]}
    return {"duration": round(edl["duration"] - INTRO_S - OUTRO_S, 3), "sections": secs,
            "events": sorted(evs, key=lambda e: e["t"]), "shot": doc_shot, "players": players}


def _map_inv(knots: list[list[float]], src: float) -> float:
    """Segment time where a [[seg, src], …] map reaches ``src``."""
    xs, ys = [k[0] for k in knots], [k[1] for k in knots]
    return float(np.interp(src, ys, xs))


def map_fwd(knots: list[list[float]], seg_t: float) -> float:
    return float(np.interp(seg_t, [k[0] for k in knots], [k[1] for k in knots]))
