"""Changes of possession from the official record, and the possession check for PBP-only shots (2026-11-20 review:
c06 narrated a Lakers drive straight into a Warriors pass; h21 narrated Lakers defenders passing to each other)."""
import pytest

from agentdunker.events import official as O
from agentdunker.events import pipeline as P
from agentdunker.events.possession import Segment
from agentdunker.film import claims, director
from agentdunker.film.facts import Fact, annotations

ACTS = [
    {"period": 1, "clock": "PT10M25.00S", "actionType": "2pt", "teamTricode": "GSW"},
    {"period": 1, "clock": "PT10M15.00S", "actionType": "turnover", "subType": "lost ball", "teamTricode": "LAL",
     "personId": 1642355, "playerNameI": "B. James"},
    {"period": 1, "clock": "PT10M15.00S", "actionType": "steal", "teamTricode": "GSW", "personId": 1630228,
     "playerNameI": "B. Podziemski"},
    {"period": 1, "clock": "PT10M09.00S", "actionType": "3pt", "teamTricode": "GSW", "actionNumber": 30},
]
IDENT = {"game_match": {"game_id": "0012600010", "period": 1, "clock": ["10:20", "10:08"]},
         "scoreboard": [{"t": 1.167, "clock_s": 620.0, "period": 1}, {"t": 7.333, "clock_s": 614.0, "period": 1}],
         "teams": {"A": {"abbr": "LAL"}, "B": {"abbr": "GSW"}},
         "players": {"3": {"person_id": 1642355, "team_abbr": "LAL"}, "7": {"person_id": 1630228, "team_abbr": "GSW"}}}
OFF = {"game_id": "0012600010", "conf": 0.95, "clock": "PT10M09.00S", "action_number": 30,
       "match": {"name": True}, "team": "GSW"}


class _Tracks:
    def team_of(self, tid):
        return {5: "A", 3: "A", 7: "B", 6: "B", 9: "B"}.get(tid)


class _Info:
    n_frames = 480

    def t(self, f):
        return f / 30.0


def test_steal_and_its_turnover_are_one_change(monkeypatch):
    monkeypatch.setattr(O, "_actions", lambda gid: ACTS)
    ch = O.possession_changes(IDENT, OFF)
    assert len(ch) == 1 and ch[0]["kind"] == "steal"
    assert (ch[0]["stealer"], ch[0]["loser"], ch[0]["loser_team"]) == ("B. Podziemski", "B. James", "LAL")
    assert O.possession_changes(IDENT, {**OFF, "conf": 0.8}) == []          # not a firm match
    assert abs(O.clock_to_t(IDENT, 615.0, 16.0) - 6.167) < 1e-6


def test_official_steal_lands_on_the_video_change_of_possession(monkeypatch):
    monkeypatch.setattr(O, "_actions", lambda gid: ACTS)
    segs = [Segment("dribble", 12, 33, holder=5, conf=0.4), Segment("pass", 33, 45, frm=5, to=3, conf=0.4),
            Segment("dribble", 60, 138, holder=3, conf=0.56), Segment("loose", 138, 144, conf=0.4),
            Segment("dribble", 144, 168, holder=7, conf=0.71), Segment("pass", 222, 228, frm=7, to=6, conf=0.37),
            Segment("hold", 252, 266, holder=6, conf=0.74)]
    ev = P._official_changes(IDENT, {"official": OFF}, segs, _Tracks(), _Info())
    assert len(ev) == 1
    e = ev[0]
    assert (e["type"], e["subtype"], e["t"], e["conf"]) == ("other", "steal", 4.8, 0.85)   # PBP clock alone: 6.17 s
    assert (e["stealer_track"], e["loser_track"], e["actors"]) == (7, 3, [7])


def test_possession_check_covers_a_shot_from_the_pbp_alone(monkeypatch):
    monkeypatch.setattr(P, "defence_never_had_ball", lambda *a: True)
    shot = {"official": {"match": {"name": False}, "team": "GSW"}, "shooter_track": 9, "release_f": 200.0,
            "evidence": {"release": {"method": "official_pbp_only"}}}
    assert P._possession_gate(IDENT, shot, _Tracks()) == ("A", 200.0)
    shot["evidence"]["release"]["method"] = "wrist"                          # unnamed video shot: no check
    assert P._possession_gate(IDENT, shot, _Tracks()) is None


def test_steal_words_need_a_steal():
    assert claims.unsupported("波齐姆斯基抢断！", {"pass"}) == ["抢断"]
    assert claims.unsupported("波齐姆斯基抢断！", {"steal"}) == []
    assert claims.unsupported("詹姆斯失误", {"turnover"}) == []
    edl = {"decisions": {"facts": []}, "shot": {}}
    events = {"events": [{"t": 4.8, "type": "other", "subtype": "steal", "conf": 0.85}]}
    assert "steal" in claims.support_for(edl, events, (4.0, 5.0))


def test_steal_fact_title_and_annotation():
    f = Fact("F1", "steal", 4.8, 5.6, [7], "波齐姆斯基抢断", 0.85)
    (a,) = annotations([f], 2.0)
    assert (a["type"], a["who"], a["color"], a["label"]) == ("focus", "T7", "warn", "波齐姆斯基抢断")
    assert abs(a["t"][0] - 2.8) < 1e-6 and abs(a["t"][1] - 4.2) < 1e-6
    title, desc, ev = director.template_title({"facts": [f.as_dict()], "holder": 7}, {}, False)
    assert title == "抢断反击" and "抢断" in desc and ev == ["F1"]
    assert annotations([Fact("F2", "turnover", 4.8, 5.6, [], "持球人失误", 0.7)], 2.0) == []   # no track to mark


def test_narration_says_the_steal():
    names = {3: {"short": "詹姆斯", "jersey": "9"}, 7: {"short": "波齐姆斯基", "jersey": "2"}}
    edl = {"segments": [{"kind": "live", "film": [6.0, 20.0], "source": [0.0, 14.0]}], "decisions": {"facts": []},
           "shot": None, "duration": 26.0}
    events = {"events": [{"t": 4.8, "type": "other", "subtype": "steal", "actors": [7], "stealer_track": 7,
                          "loser_track": 3, "conf": 0.85, "official": {"stealer": "B. Podziemski"}}]}
    doc = director.narration_doc(edl, events, None, names)
    notes = [e["note"] for e in doc["events"]]
    assert any(n.startswith("steal: 波齐姆斯基 takes the ball away from 詹姆斯") for n in notes), notes


def test_latin_names_lose_their_diacritics():
    from agentdunker.film.render.nameplates import latin, short_name
    assert short_name({"name": "Jonas Valančiūnas"}) == "Valanciunas"
    assert short_name({"name_zh_short": "Biberović", "name": "Tarik Biberović"}) == "Biberovic"
    assert short_name({"name_zh_short": "邓恩"}) == "邓恩" and latin("克里斯·邓恩") == "克里斯·邓恩"


def test_chinese_names_from_the_season_tables(monkeypatch):
    from agentdunker.film import season
    monkeypatch.setattr(season, "zh_name", lambda pid: {"name_zh": "约纳斯·瓦兰丘纳斯", "name_zh_short": "瓦兰丘纳斯"}
                        if str(pid) == "202685" else None)
    ident = {"players": {"9": {"person_id": 202685, "name": "Jonas Valančiūnas", "name_zh": None},
                         "1": {"person_id": 1, "name": "A B", "name_zh": None},
                         "2": {"person_id": None, "name": None, "name_zh": None}}}
    out = director.chinese_names_fill(ident)
    assert out["players"]["9"]["name_zh_short"] == "瓦兰丘纳斯" and ident["players"]["9"]["name_zh"] is None
    assert out["players"]["1"]["name_zh"] is None
    assert director.chinese_names_fill({"players": {"1": {"person_id": 1}}}) is None


def test_long_team_label_shrinks_before_it_is_dropped():
    from agentdunker.film.render import freeze, style
    if not (style.FONTS / "NotoSansSC-VF.ttf").exists():
        pytest.skip("overlay fonts not downloaded (agentdunker download-models)")
    seen = []
    real = freeze._Canvas.text

    def spy(self, text, x, y, size, fill=freeze.WHITE, latin=False):
        seen.append((text, size))
        return real(self, text, x, y, size, fill, latin)

    freeze._Canvas.text = spy
    try:
        freeze.make_card("西亚卡姆", "步行者 · 43号", 0.259)
    finally:
        freeze._Canvas.text = real
    team = [s for t, s in seen if t == "步行者 · 43号"]
    assert team and 20 <= team[0] < 28, seen


def test_commentary_rejects_two_names_run_together():
    from agentdunker.narration.script import Slot, check_line
    f = {"allowed_names": ["克林根", "阿夫迪亚", "卡马拉"], "allowed_numbers": [], "all_events": []}
    slot = Slot(7, 21.7, 26.3, kind="event", budget=19)
    assert any("连在一起" in e for e in check_line("面对多人协防，克林根阿夫迪亚迅速出球！", slot, f, set()))
    assert not any("连在一起" in e for e in check_line("克林根传给阿夫迪亚，迅速出球！", slot, f, set()))
