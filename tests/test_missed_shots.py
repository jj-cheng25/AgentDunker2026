"""Missed shots (2026-11-27 review of 20 misses and 8 makes from PL-NBA possessions, 2022-25 games whose PBP
mirror has no boxscore, so no track is named): the record pins the shot by team, its made/missed stands against
the ball path, it moves the shot off a defender, and it names the shooter; a tap after the rebound is not the shot;
a dunk the video missed comes from the record even without a track."""
from agentdunker.events import official as O
from agentdunker.events import pipeline as P
from agentdunker.events.balltrack import BallObs
from agentdunker.events.release import Release
from agentdunker.events.rimvisit import RimVisit
from agentdunker.events.tracks import Tracks
from agentdunker.events.video import VideoInfo
from agentdunker.film import director

INFO = VideoInfo("clip.mp4", 30.0, 300, 1920, 1080, 0.0)
ACTS = [{"period": 3, "clock": "PT02M27.00S", "actionType": "2pt", "isFieldGoal": 1, "teamTricode": "MIN",
         "shotResult": "Missed", "personId": 1630162, "playerNameI": "A. Edwards", "actionNumber": 400,
         "description": "MISS A. Edwards driving reverse Layup"},
        {"period": 3, "clock": "PT02M25.00S", "actionType": "3pt", "isFieldGoal": 1, "teamTricode": "DAL",
         "shotResult": "Made", "personId": 1629029, "playerNameI": "L. Doncic", "actionNumber": 402}]
IDENT = {"game_match": {"game_id": "0042300314", "status": "matched", "conf": 0.894, "period": 3,
                        "clock": ["2:35", "2:27"]},
         "teams": {"A": {"abbr": "MIN"}, "B": {"abbr": "DAL"}},
         "players": {"5": {"team_abbr": "MIN", "jersey": "5", "name": None}, "7": {"team_abbr": "DAL", "name": None}}}


def _tracks(boxes_by_tid, teams, frames=range(0, 120)):
    return Tracks({"tracks": [{"track_id": t, "team": teams[t], "role": "player",
                               "frames": [{"f": f, "bbox": list(b), "observed": True} for f in frames]}
                              for t, b in boxes_by_tid.items()]})


def _attempt(f_rel, tid, method="wrist", made=True, crossings=((60.0, 0.0),), f_first=50):
    judged = {"made": made, "conf": 0.9 if made else 0.8, "rule": "test", "crossings": list(crossings),
              "f_cross": crossings[-1][0] if crossings else None}
    return {"visit": RimVisit(f_first, f_first + 20, None, None, True),
            "release": Release(f_rel, tid, int(f_rel), 0.93, {"method": method}), "ball": {}, "made": judged,
            "dets": {}}


def _pbp(monkeypatch, acts):
    monkeypatch.setattr(O, "_actions", lambda gid: acts)
    monkeypatch.setattr(O, "xfg_for", lambda *a: None)


def test_the_only_field_goal_of_the_shooters_team_pins_the_shot(monkeypatch):
    _pbp(monkeypatch, ACTS)
    off = O.match_shot(IDENT, 5, True)
    assert (off["player"], off["made"]) == ("A. Edwards", False)
    assert (off["match"]["n_candidates"], off["match"]["n_team"], off["conf"]) == (2, 1, 0.894)   # no 0.8 discount
    assert P.official_pins(off, None, named=False) == "team"
    assert P.official_pins(off, None, named=True) is None            # a name to check, and it does not match
    assert P.official_pins({**off, "conf": 0.5}, None, named=False) is None      # below any matched game (0.57)


def test_the_record_overrules_a_miss_read_as_made(monkeypatch):
    _pbp(monkeypatch, ACTS)
    tr = _tracks({5: (100, 100, 160, 260), 7: (700, 100, 760, 260)}, {5: "A", 7: "B"})
    doc, a = P._main_shot([_attempt(40, 5)], INFO, "clip.mp4", {}, tr, None, IDENT, False)
    assert (doc["made"], doc["made_source"], doc["made_t"]) == (False, "official_pbp", None)
    assert a["made"]["made"] is False and doc["evidence"]["made_disagreement"]["ball_path"] is True
    named = {**IDENT, "players": {**IDENT["players"], "5": {"team_abbr": "MIN", "name": "Mike Conley"}}}
    doc, _ = P._main_shot([_attempt(40, 5)], INFO, "clip.mp4", {}, tr, None, named, False)
    assert (doc["made"], doc["made_source"]) == (True, "ball_path")   # a named shooter the record does not name


def test_a_tap_far_from_the_rim_after_the_shot_is_not_the_shot():
    shot = _attempt(40, 8, made=False, crossings=((70.0, 0.9),))
    tap = _attempt(100, 5, method="box", made=False, crossings=((120.0, -6.5),), f_first=110)
    unclear = _attempt(100, None, method="box", made=None, crossings=(), f_first=110)
    putback = _attempt(100, 5, method="box", made=False, crossings=((120.0, -0.8),), f_first=110)
    assert P.pick_attempt([shot, tap]) is shot and P.pick_attempt([shot, unclear]) is shot
    assert P.pick_attempt([shot, putback]) is putback                # a box release at the rim is still a shot
    assert P.pick_attempt([tap]) is tap                              # and a lone attempt stays the shot


def test_the_records_team_takes_the_shot_from_the_defender_at_the_rim():
    off = {"conf": 0.894, "person_id": 1630578, "team": "HOU", "player": "A. Sengun",
           "match": {"name": False, "team": False, "n_candidates": 1, "n_team": 0}}
    ident = {"teams": {"A": {"abbr": "HOU"}, "B": {"abbr": "GSW"}}, "players": {}}
    ball = {f: BallObs(f, (130.0, 150.0), 10.0, 0.9) for f in range(0, 30)}
    holder, defender, far, farther = (100, 100, 160, 260), (165, 100, 225, 260), (700, 100, 760, 260), (900, 100, 960, 260)
    tr = _tracks({4: holder, 9: defender, 2: far}, {4: "A", 9: "B", 2: "A"})
    s, ov, kept = P.arbitrate(ident, off, 9, ball, tr, 30.0, 30.0)
    assert (s, ov["to"], ov["by"], kept) == (4, 4, "official_pbp_team", None)
    # no HOU track had the ball or stood next to the pick: the pick is dropped, the commentary names the scorer
    tr = _tracks({4: far, 9: holder, 2: farther}, {4: "A", 9: "B", 2: "A"})
    s, ov, _ = P.arbitrate(ident, off, 9, ball, tr, 30.0, 30.0)
    assert (s, ov["from"], ov["to"]) == (None, 9, None)
    # a pick on the scorer's team, or a game too loosely matched: nothing moves
    assert P.arbitrate(ident, off, 2, ball, tr, 30.0, 30.0) == (2, None, None)
    assert P.arbitrate(ident, {**off, "conf": 0.5}, 9, ball, tr, 30.0, 30.0) == (9, None, None)


def test_a_dunk_the_video_missed_comes_from_the_record_without_a_track(monkeypatch):
    _pbp(monkeypatch, [{"period": 1, "clock": "PT08M46.00S", "actionType": "2pt", "isFieldGoal": 1,
                        "teamTricode": "NYK", "shotResult": "Made", "personId": 1628384, "playerNameI": "O. Anunoby",
                        "actionNumber": 60, "description": "O. Anunoby cutting DUNK (2 PTS)"}])
    ident = {"game_match": {"game_id": "0042400174", "status": "matched", "conf": 0.883, "period": 1,
                            "clock": ["8:54", "8:46"]},
             "scoreboard": [{"t": 0.7, "clock_s": 534.0, "period": 1}, {"t": 8.3, "clock_s": 526.0, "period": 1}],
             "players": {"3": {"team_abbr": "NYK", "name": None}}}
    doc = P._official_only(ident, Tracks({}), INFO)
    assert (doc["shooter_track"], doc["made"], doc["points"], doc["release_t"]) == (None, True, 2, 8.3)
    ident["players"]["3"]["name"] = "Jalen Brunson"          # his team is named, he is not: no shot without a track
    assert P._official_only(ident, Tracks({}), INFO) is None


def test_the_record_names_an_unnamed_shooter_when_his_team_took_the_only_field_goal(monkeypatch):
    from agentdunker.film import season
    monkeypatch.setattr(season, "zh_name", lambda pid: {"name_zh": "格雷森·阿伦", "name_zh_short": "阿伦"})
    ident = {"players": {"4": {"team_abbr": "PHX", "jersey": "8", "name": None, "person_id": None, "conf": 0.97}}}
    off = {"conf": 0.843, "person_id": 1628960, "team": "PHX", "player": "G. Allen",
           "match": {"name": False, "team": True, "n_candidates": 2, "n_team": 1}}
    events = {"shot": {"shooter_track": 4, "official": off}}
    out = director.official_shooter_fill(ident, events)
    assert (out["players"]["4"]["name"], out["players"]["4"]["name_zh_short"]) == ("G. Allen", "阿伦")
    off["match"]["n_team"] = 2                                # two shots of his team in the window
    assert director.official_shooter_fill(ident, events) is None
    off["match"]["n_team"], off["conf"] = 1, 0.5              # below any matched game (0.57)
    assert director.official_shooter_fill(ident, events) is None
