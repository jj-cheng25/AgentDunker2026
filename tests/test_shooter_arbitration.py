"""Who shot: holds by people off the floor, the contest rule, and when the play-by-play may move the shooter
(2026-11-23 review of the 24 held-out clips against hand-checked shooters: h03/h04/h22 lost the release to a
spectator, h19 to the defender behind Towns, and the PBP override put the shot on the wrong player in h01, h07
and h16)."""
from agentdunker.events import pipeline as P
from agentdunker.events.balltrack import BallObs
from agentdunker.events.court import Court
from agentdunker.events.pose import Person
from agentdunker.events.release_pose import HandOff, on_floor, shooter_of
from agentdunker.events.tracks import Tracks
from agentdunker.film import director

# 10 px per foot, image (470, 250) = centre court
COURT = Court({"frames": [{"f": f, "ok": True, "H_img2court": [[0.1, 0, -47], [0, 0.1, -25], [0, 0, 1]]}
                          for f in range(0, 40)]})


def _person(box):
    return Person(tuple(box), [((box[0] + box[2]) / 2, box[1] + 20, 0.9)] * 2, [], 0.9)


def _tracks(boxes_by_tid, teams, frames=range(0, 40)):
    return Tracks({"tracks": [{"track_id": t, "team": teams[t], "role": "player",
                               "frames": [{"f": f, "bbox": list(b), "observed": True} for f in frames]}
                              for t, b in boxes_by_tid.items()]})


def test_people_off_the_floor_cannot_hold():
    on = _person((400, 300, 440, 420))           # feet (420, 420) -> (-5, 17) ft: on the floor
    fan = _person((400, 440, 440, 560))          # feet -> (-5, 31) ft: six feet past the sideline
    jumper = _person((600, 440, 640, 560))       # feet thrown past the line by the jump, but on an M1 box
    tracks = _tracks({7: (598, 440, 642, 562)}, {7: "A"})
    kept = on_floor({10: [on, fan, jumper]}, tracks, COURT)[10]
    assert on in kept and jumper in kept and fan not in kept
    assert on_floor({10: [fan]}, tracks, Court(None)) == {10: [fan]}      # no homography: nobody dropped


def test_a_clear_last_hold_is_not_flipped_to_the_defender():
    shooter_box, defender_box = (100, 100, 160, 260), (150, 100, 210, 260)
    tracks = _tracks({8: shooter_box, 4: defender_box}, {8: "B", 4: "A"})
    held = [(f, 0.5, _person(defender_box)) for f in range(0, 20)]       # the defender's hands on it early on
    held += [(f, 0.2, _person(shooter_box)) for f in range(20, 25)]     # the shooter's, right up to the release
    h = HandOff(25.5, 24, held[-1][2], 0.9, held)
    tid, votes = shooter_of(h, tracks, {}, 30.0)
    assert tid == 8, votes
    # a near tie at the end is still a contest: the hand that held it longest is the shooter's
    held = [(f, 0.3, _person(shooter_box)) for f in range(0, 20)] + [(24, 0.6, _person(defender_box))]
    tid, _ = shooter_of(HandOff(24.5, 24, held[-1][2], 0.9, held), tracks, {}, 30.0)
    assert tid == 8


def _off(team="MIN", pid=1630183):
    return {"conf": 0.95, "person_id": pid, "team": team, "player": "J. McDaniels",
            "match": {"name": False, "team": True, "n_candidates": 1}}


IDENT = {"players": {"3": {"person_id": 1629029, "team_abbr": "MIN", "name": "Rudy Gobert"},
                     "4": {"person_id": 1630183, "team_abbr": "MIN", "name": "Jaden McDaniels"},
                     "5": {"person_id": 1, "team_abbr": "DEN", "name": "X Y"}}}


def _ball(frames, xy=(130.0, 150.0)):
    return {f: BallObs(f, xy, 10.0, 0.9) for f in frames}


def test_the_named_track_takes_the_shot_only_if_it_had_the_ball():
    near, far = (100, 100, 160, 260), (700, 100, 760, 260)
    # the named track (4) dribbled it: it takes the shot from the defender the video picked
    tr = _tracks({4: near, 5: far}, {4: "A", 5: "B"})
    s, ov, kept = P.arbitrate(IDENT, _off(), 5, _ball(range(0, 30)), tr, 30.0, 30.0)
    assert (s, ov["to"], kept) == (4, 4, None)
    # the named track was across the floor while the video's pick, his team-mate, dribbled: keep the pick
    tr = _tracks({3: near, 4: far}, {3: "A", 4: "A"})
    s, ov, kept = P.arbitrate(IDENT, _off(), 3, _ball(range(0, 30)), tr, 30.0, 30.0)
    assert (s, ov) == (3, None) and (kept["track"], kept["pbp_track"]) == (3, 4)
    # ... and a pick on the other team leaves no shooter at all
    tr = _tracks({5: near, 4: far}, {5: "B", 4: "A"})
    s, ov, kept = P.arbitrate(IDENT, _off(), 5, _ball(range(0, 30)), tr, 30.0, 30.0)
    assert (s, ov["from"], ov["to"], kept) == (None, 5, None, None)


def test_a_kept_shooter_takes_the_scorers_name():
    events = {"shot": {"shooter_track": 3, "official": _off(),
                       "evidence": {"shooter_kept": {"track": 3, "pbp_track": 4, "person_id": 1630183,
                                                     "player": "J. McDaniels"}}}}
    out = director.official_shooter_swap(IDENT, events)
    assert out["players"]["3"]["name"] == "Jaden McDaniels" and out["players"]["4"]["name"] == "Rudy Gobert"
    assert IDENT["players"]["3"]["name"] == "Rudy Gobert"                          # the input is not touched
    events["shot"]["official"] = _off(team="DEN")
    assert director.official_shooter_swap(IDENT, events) is None                  # not the scorer's team
