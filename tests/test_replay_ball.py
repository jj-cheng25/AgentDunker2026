"""The 3D replay's ball follows M5's ball timeline (2026-11-27 review: with only confident pass events and the
shooter's gather, 22 of 62 test replays showed the ball for under half a second, though one player dribbled it
in plain view; the Durant reference showed it because its last confident pass went to the shooter)."""
from agentdunker.replay.scene import SceneConfig, _ball_segments

CFG = SceneConfig()
SHOWN = {1, 2, 4, 6, 9}
OFF = {1, 2, 4}


def _ev(states, events=()):
    return {"ball_state": states, "events": list(events), "shot": {"shooter_track": 4, "release_t": 8.0}}


def _visible(segs):
    return [(s["t"], s.get("hold") or s.get("pass"), s.get("mode")) for s in segs if not s.get("hidden")]


def test_the_ball_follows_the_dribbler_and_his_passes():
    ev = _ev([{"type": "dribble", "t0": 0.0, "t1": 0.6, "conf": 0.78, "holder": 1},
              {"type": "pass", "t0": 0.6, "t1": 0.7, "conf": 0.44, "from": 1, "to": 2},
              {"type": "hold", "t0": 0.7, "t1": 0.9, "conf": 0.44, "holder": 2},       # short, but between passes
              {"type": "pass", "t0": 0.9, "t1": 1.4, "conf": 0.44, "from": 2, "to": 4},
              {"type": "dribble", "t0": 1.4, "t1": 7.1, "conf": 0.58, "holder": 4}])
    segs, note = _ball_segments(ev, 4, 8.0, 0.0, 7.4, SHOWN, CFG, offense=OFF)
    assert note == "ball_state"
    assert _visible(segs) == [([0.0, 0.6], "T1", "dribble"), ([0.6, 0.7], ["T1", "T2"], None),
                              ([0.7, 0.9], "T2", "hands"), ([0.9, 1.4], ["T2", "T4"], None),
                              ([1.4, 7.1], "T4", "dribble")]


def test_defence_contact_noise_never_gets_the_ball_and_short_gaps_are_bridged():
    ev = _ev([{"type": "dribble", "t0": 0.7, "t1": 1.5, "conf": 0.77, "holder": 2},
              {"type": "hold", "t0": 2.0, "t1": 2.6, "conf": 0.38, "holder": 9},       # a defender's hands near it
              {"type": "dribble", "t0": 3.1, "t1": 3.4, "conf": 0.29, "holder": 1},    # too unsure
              {"type": "dribble", "t0": 3.6, "t1": 6.0, "conf": 0.53, "holder": 2}])
    segs, _ = _ball_segments(ev, 4, 8.0, 0.0, 6.0, SHOWN, CFG, offense=OFF)
    assert _visible(segs) == [([0.7, 6.0], "T2", "dribble")]                        # one possession, 2.1 s gap
    ev["ball_state"][-1].update(t0=4.2)                                             # 2.7 s apart: not bridged
    segs, _ = _ball_segments(ev, 4, 8.0, 0.0, 6.0, SHOWN, CFG, offense=OFF)
    assert [v[0] for v in _visible(segs)] == [[0.7, 1.5], [4.2, 6.0]]


def test_without_a_usable_ball_timeline_the_pass_chain_still_places_it():
    passes = [{"t": 5.0, "type": "pass", "actors": [2, 4], "conf": 0.6}]
    ev = _ev([{"type": "hold", "t0": 1.0, "t1": 2.0, "conf": 0.2, "holder": 2}], passes)
    segs, note = _ball_segments(ev, 4, 8.0, 0.0, 7.4, SHOWN, CFG, offense=OFF)
    assert note == "events" and _visible(segs)[0] == ([4.5, 5.0], "T2", "hands")
    segs, note = _ball_segments(_ev([]), 4, 8.0, 0.0, 7.4, SHOWN, CFG, offense=OFF)
    assert note == "hidden (possession unknown)" and segs == []                     # the gather alone is too short
