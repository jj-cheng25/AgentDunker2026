"""Director rules found on the held-out clips (2026-11-19)."""
from agentdunker.film import director


def _court(n, ok_from=0):
    return {"frames": [{"f": f, "ok": f >= ok_from} for f in range(n)]}


def test_short_lead_in_window_when_the_clip_starts_just_before_the_shot():
    fps, n = 30.0, 300
    shots = [{"kind": "wide", "start_f": 0, "end_f": n - 1}]
    # release at 2.66 s: the calibrated run before it is 2.0 s, under the 3 s minimum
    w = director.replay_window(shots, _court(n), fps, 2.66, n)
    assert w is not None and w[0] == 0.0 and abs(w[1] - 2.0) < 0.05 and "lead-in" in w[2]
    # no lead-in run: still None when everything is shorter than 3 s
    assert director.replay_window(shots, _court(n, ok_from=40), fps, 2.66, n) is None


def test_long_window_still_preferred_over_a_short_lead_in():
    fps, n = 30.0, 600
    shots = [{"kind": "wide", "start_f": 0, "end_f": 200}, {"kind": "wide", "start_f": 201, "end_f": n - 1}]
    # first wide shot fully calibrated (6.7 s), the second only 2 s before the release at 9.6 s
    court = {"frames": [{"f": f, "ok": f <= 200 or f >= 210} for f in range(n)]}
    w = director.replay_window(shots, court, fps, 9.6, n)
    assert w is not None and w[2] == "court-ok wide run" and w[1] <= 200 / fps + 0.01


def test_freeze_name_falls_back_to_the_english_surname():
    ident = {"players": {"6": {"conf": 0.99, "jersey": "9", "team_abbr": "BKN", "name": "E.J. Liddell",
                               "name_zh": None, "name_zh_short": None},
                         "7": {"conf": 0.99, "jersey": "24", "team_abbr": "SAS", "name": "Devin Vassell",
                               "name_zh": "德文·瓦塞尔", "name_zh_short": "瓦塞尔"}}}
    names = director.player_names(ident)
    assert names[6]["short"] == "Liddell" and director.label(6, names) == "Liddell"
    assert names[7]["short"] == "瓦塞尔"
