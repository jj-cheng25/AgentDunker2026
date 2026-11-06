"""Event timeline -> time-windowed commentary lines.

Each line is anchored to one event (or a burst of close events). Its character budget
comes from the window length, so the line can be spoken inside the window. Names and
numbers are only allowed when they come from the input data; violations are repaired by
the model once more, then replaced by a deterministic template line.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from . import moss
from .textnorm import bad_names, bad_numbers, int_zh, spoken_len

logger = logging.getLogger(__name__)

LEAD = 0.12        # line may start this much before its event
GAP = 0.18         # breath gap between lines
MIN_SLOT = 1.1     # events closer than this share one line
MAX_LINE = 4.6     # longest single line, seconds
MIN_WINDOW = 0.8   # shorter event windows get no line
RATE = 4.3         # budget speaking rate (chars/s); Mossland flash measured 4.3-6.7 c/s
MAX_CHARS = 22
# Shot details a line may only state when some event note supports them.
DETAIL_WORDS = {r"空心|唰": ("swish", "clean", "nothing but net"), r"打板|擦板|碰板": ("glass", "bank", "backboard"),
                r"碰框|涮框|弹框|滚框": ("rim",), r"扣篮|暴扣": ("dunk",), r"上篮": ("layup",)}
BRIDGE_TEXT = "接下来，让我们用三维视角，回看这次进攻战术！"

STYLE_ANCHOR = """杜兰特左侧面对单防。
队友，拉开！
球又传给了杜兰特，看他怎么处理。
左侧变向加速，突破第一道防线。
漂移抛投打板，打进！
史密斯绕出来接球。防守扑出，回传杜兰特，再次组织。
队友分到两侧，把上方空间留给杜兰特单打。
第一步没直接打穿，收回来，再找第二拍。
再次启动！压向篮下，防线跟着收缩，接着看他的终结！
这次出手，官方预期命中率只有百分之二十一点三。
顶着防守，杜兰特还是打板命中！难度摆在眼前，这就是他的终结能力！"""

SYSTEM = f"""你是NBA中文电视解说员，风格激情、短促、有专业视角。
风格范例（只学语气和节奏，不要照抄内容）：
{STYLE_ANCHOR}

写作规则：
1. 每个时间槽写一句口播，必须描述该槽的事件，不能提前说后面才发生的事。
2. 字数（不含标点）严格不超过该槽的 max_chars；时间短就用三五个字的短句。
3. 要讲“为什么”：空间被拉开、防守被吸引、出手难度、选择的好坏；但只能根据给出的事件和备注推断，不确定就只描述动作。
4. 只能使用 allowed_names 里的人名；未识别的球员用“持球人”“队友”“防守人”“外线射手”等角色称呼，绝不自创人名、号码或球队名。
5. 只能使用 allowed_numbers 里的数字（写成阿拉伯数字，如 21.3%；投篮类型可写“三分”）；没有数据就不要提任何数字、命中率、比分、时间。
6. 命中/未命中必须与 facts 一致；没有给出结果就不要说进没进；不要编造备注里没有的投篮细节（空心、打板、碰框、扣篮等）。
7. 关键时刻（出手、命中）用感叹号；铺垫用逗号句号。不要用破折号、省略号、英文、表情。
8. 相邻句子不要重复同一个词开头，整段读起来像一个连贯的回合。
只输出 JSON：{{"lines":[{{"slot":槽号,"text":"句子"}}]}}，每个槽都要有一项。"""


@dataclass
class Slot:
    idx: int
    start: float
    end: float
    events: list = field(default_factory=list)
    kind: str = "event"          # event | bridge | xfg
    budget: int = 0
    text: str = ""
    origin: str = ""
    notes: list = field(default_factory=list)

    @property
    def window(self) -> float:
        return self.end - self.start


def _sections(doc: dict) -> list[dict]:
    return sorted(doc.get("sections") or [], key=lambda s: s["start"])


def _section_at(sections: list[dict], t: float) -> dict | None:
    return next((s for s in sections if s["start"] <= t < s["end"]), None)


def build_slots(doc: dict) -> list[Slot]:
    duration = float(doc.get("duration") or doc.get("video", {}).get("duration") or 1e9)
    sections = _sections(doc)
    shot = doc.get("shot") or {}
    events = [dict(e) for e in doc.get("events", []) if float(e.get("conf", 1.0)) >= 0.35]
    anchors = []
    for s in sections:
        if s.get("kind") == "bridge":
            anchors.append({"t": s["start"] + 0.15, "kind": "bridge", "end": s["end"] - 0.1})
        elif s.get("kind") == "freeze" and shot.get("xfg") is not None:
            anchors.append({"t": s["start"] + 0.2, "kind": "xfg", "end": s["end"] - 0.2})
    for e in events:
        sec = _section_at(sections, float(e["t"]))
        if sec and sec.get("kind") == "bridge":
            continue
        if any(a["kind"] == "xfg" and abs(a["t"] - float(e["t"])) < 1.5 for a in anchors):
            continue
        anchors.append({"t": float(e["t"]), "kind": "event", "event": e})
    anchors.sort(key=lambda a: a["t"])

    slots: list[Slot] = []
    for a in anchors:
        start = max(0.0, a["t"] - (LEAD if a["kind"] == "event" else 0.0))
        sec = _section_at(sections, a["t"])
        if sec:
            start = max(start, sec["start"] + 0.05)
        if slots and slots[-1].kind == "event" and a["kind"] == "event" and start - slots[-1].start < MIN_SLOT:
            slots[-1].events.append(a["event"])
            continue
        slots.append(Slot(len(slots), start, a.get("end", start + MAX_LINE), [a["event"]] if "event" in a else [], a["kind"]))
        slots[-1].notes.append(a.get("end", start + MAX_LINE))   # the slot's own upper bound
    # A window too short for a phrase gets no line (reference: "迎着协防" was deleted by hand);
    # its time goes to the previous line. Ends are recomputed after every drop.
    while True:
        _set_ends(slots, sections, duration)
        short = next((s for s in slots if s.kind == "event" and s.window < MIN_WINDOW), None)
        if short is None:
            return slots
        slots.remove(short)


def _set_ends(slots: list[Slot], sections: list[dict], duration: float) -> None:
    for i, s in enumerate(slots):
        limit = slots[i + 1].start - GAP if i + 1 < len(slots) else duration - 0.1
        sec = _section_at(sections, s.start)
        if sec and s.kind == "event":
            limit = min(limit, sec["end"] - 0.05)
        s.end = min(s.notes[0], limit, s.start + MAX_LINE)
        s.idx = i
        s.budget = max(0, min(MAX_CHARS, int(s.window * RATE)))


def _names(doc: dict) -> dict[str, str]:
    """track_id -> short Chinese name, only for confident identities."""
    out = {}
    for tid, p in (doc.get("players") or {}).items():
        name = p.get("name_zh") or p.get("short_zh")
        if name and float(p.get("conf", 1.0)) >= 0.6:
            out[str(tid)] = p.get("short_zh") or name.split("·")[-1]
    return out


def facts(doc: dict) -> dict:
    shot = doc.get("shot") or {}
    names = _names(doc)
    allowed_numbers = set()
    f = {"allowed_names": sorted(set(names.values())), "players": names}
    if shot.get("xfg") is not None:
        pct = f"{round(float(shot['xfg']) * 100, 1):g}%"
        f["xfg"] = pct
        allowed_numbers.add(pct)
        allowed_numbers.add(pct.rstrip("%"))
    if shot.get("points") in (2, 3):
        f["shot_type"] = "三分球" if shot["points"] == 3 else "两分球"
        allowed_numbers.update({int_zh(shot["points"]), str(shot["points"])})
    if shot.get("made") is not None:
        f["shot_result"] = "命中" if shot["made"] else "未命中"
    if shot.get("shooter_track") is not None:
        f["shooter"] = names.get(str(shot["shooter_track"]), "持球人（未识别姓名）")
    for n in doc.get("allowed_numbers", []):
        allowed_numbers.add(str(n))
    f["allowed_numbers"] = sorted(allowed_numbers)
    f["all_events"] = doc.get("events", [])
    return f


def _event_view(e: dict, names: dict[str, str]) -> dict:
    actors = [names.get(str(a), "未识别球员") for a in e.get("actors", [])]
    view = {"t": round(float(e["t"]), 2), "type": e.get("type", "other"), "actors": actors}
    if e.get("note"):
        view["note"] = e["note"]
    return view


def check_line(text: str, slot: Slot, f: dict, roster: set[str]) -> list[str]:
    errs: list[str] = []
    n = spoken_len(text)
    if slot.kind == "bridge":
        return errs
    if n > slot.budget:
        errs.append(f"字数{n}超过上限{slot.budget}")
    if n == 0 and slot.budget >= 3:
        errs.append("空句")
    names = bad_names(text, set(f["allowed_names"]), roster)
    if names:
        errs.append("出现未授权的人名或外文：" + "、".join(names))
    nums = bad_numbers(text, set(f["allowed_numbers"]))
    if nums:
        errs.append("出现未授权的数字：" + "、".join(nums))
    if slot.kind == "xfg" and f.get("xfg") and f["xfg"].rstrip("%") not in text:
        errs.append(f"此槽必须说出官方预期命中率 {f['xfg']}")
    notes = " ".join(str(e.get("note", "")).lower() for e in f.get("all_events", []))
    for words, keys in DETAIL_WORDS.items():
        hit = re.search(words, text)
        if hit and not any(k in notes for k in keys):
            errs.append(f"“{hit.group(0)}”没有事件依据，不要编造细节")
    if re.search(r"[—…]", text):
        errs.append("不要用破折号或省略号")
    result = f.get("shot_result")
    made_words = re.search(r"打进|命中|得分|投进|进了|稳稳|入网", text)
    miss_words = re.search(r"没进|不进|打铁|偏出|弹框而出|未命中|失手", text)
    if result == "命中" and miss_words:
        errs.append("结果写反：这次出手命中")
    if result == "未命中" and made_words and not re.search(r"没|不", text):
        errs.append("结果写反：这次出手未命中")
    if result is None and (made_words or miss_words):
        errs.append("没有给出投篮结果，不要说进没进")
    return errs


def template_line(slot: Slot, f: dict) -> str:
    """Deterministic fallback, always within facts."""
    if slot.kind == "bridge":
        return BRIDGE_TEXT
    if slot.kind == "xfg" and f.get("xfg"):
        x = float(f["xfg"].rstrip("%"))
        word = "只有" if x < 35 else ("高达" if x > 55 else "是")
        for head in ("这次出手，官方预期命中率", "官方预期命中率", "预期命中率"):
            if spoken_len(head + word + f["xfg"]) <= slot.budget:
                return f"{head}{word}{f['xfg']}"
        return ""
    types = [e.get("type") for e in slot.events]
    who = next((f["players"].get(str(a)) for e in slot.events for a in e.get("actors", []) if f["players"].get(str(a))), "")
    if "shot" in types:
        line = f"{who}出手！" if who else "出手！"
        if f.get("shot_result") == "命中":
            line += "打进！"
        elif f.get("shot_result") == "未命中":
            line += "没进！"
    elif "drive" in types:
        line = f"{who}突破！" if who else "持球突破！"
    elif "screen" in types:
        line = "上来掩护！"
    elif "pass" in types:
        line = f"球交给{who}。" if who else "传球转移。"
    elif "rebound" in types:
        line = "篮板球！"
    else:
        line = ""
    while spoken_len(line) > slot.budget and "！" in line[:-1]:
        line = line.split("！")[0] + "！"
    return line if spoken_len(line) <= slot.budget else ""


def write_script(doc: dict, roster: set[str] | None = None, use_llm: bool = True) -> tuple[list[Slot], dict]:
    roster = roster or set()
    slots = build_slots(doc)
    f = facts(doc)
    for s in slots:
        if s.kind == "bridge":
            s.text, s.origin = BRIDGE_TEXT, "fixed"
    open_slots = [s for s in slots if s.kind != "bridge"]
    log: dict[str, Any] = {"calls": 0, "repairs": [], "fallbacks": []}
    feedback: dict[int, list[str]] = {}
    for attempt in range(3 if use_llm else 0):
        todo = [s for s in open_slots if not s.origin]
        if not todo:
            break
        payload = {
            "facts": {k: v for k, v in f.items() if k not in ("players", "all_events")},
            "previous_lines": {s.idx: s.text for s in slots if s.origin},
            "slots": [{"slot": s.idx, "time": f"{s.start:.1f}-{s.end:.1f}s", "kind": s.kind, "max_chars": s.budget,
                       "events": [_event_view(e, f["players"]) for e in s.events],
                       **({"fix": feedback[s.idx]} if s.idx in feedback else {})} for s in todo],
        }
        try:
            out = moss.chat_json(SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=0.8 if attempt == 0 else 0.5)
            log["calls"] += 1
        except (RuntimeError, ValueError, KeyError, IndexError) as exc:  # model/network failure -> template fallback
            logger.warning("script model call failed: %s", exc)
            break
        got = {int(item.get("slot", -1)): str(item.get("text", "")).strip() for item in out.get("lines", [])}
        for s in todo:
            text = got.get(s.idx, "")
            errs = check_line(text, s, f, roster)
            if errs:
                feedback[s.idx] = errs
                log["repairs"].append({"slot": s.idx, "text": text, "errors": errs})
            else:
                s.text, s.origin = text, "model"
    for s in open_slots:
        if not s.origin:
            s.text, s.origin = template_line(s, f), "template"
            log["fallbacks"].append(s.idx)
    return slots, log


def shorten(text: str, max_chars: int, slot: Slot, f: dict, roster: set[str], hint: str = "") -> str:
    """Shorter (or, with a hint, differently worded) line: model first, then clause dropping."""
    task = {"text": text, "max_chars": max_chars}
    if hint:
        task["hint"] = hint
    try:
        out = moss.chat_json(
            "你是篮球解说稿编辑。把句子改写到不超过给定字数（不含标点），保留最关键的动作和已有事实，不添加新事实、人名或数字，"
            "保持激情口播语气；如果给了 hint，按 hint 换掉容易被听错的词。只输出 JSON：{\"text\":\"改写后的句子\"}",
            json.dumps(task, ensure_ascii=False), temperature=0.3 if not hint else 0.6)
        cand = str(out.get("text", "")).strip()
        if cand and cand != text and spoken_len(cand) <= max_chars and not check_line(cand, Slot(slot.idx, slot.start, slot.end, slot.events, slot.kind, max_chars), f, roster):
            return cand
    except (RuntimeError, ValueError, KeyError, IndexError) as exc:   # fall back to clause dropping
        logger.warning("shorten call failed: %s", exc)
    if hint:
        return ""
    parts = [p for p in re.split(r"(?<=[，。！？])", text) if p]
    while len(parts) > 1 and spoken_len("".join(parts)) > max_chars:
        parts.pop()
    cand = "".join(parts)
    return cand if spoken_len(cand) <= max_chars else ""
