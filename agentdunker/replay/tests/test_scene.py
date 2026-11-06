"""Synthetic smoke test: image px == court ft (H = translation), 10 players, one runs into the backcourt."""
import numpy as np

from agentdunker.replay.scene import build_play

FPS, N = 30.0, 150
H = [[1, 0, -47.0 - 100], [0, 1, -25.0 - 100], [0, 0, 1]]   # pixel (u, v) -> court (u-147, v-125)


def _tracks():
    tracks = []
    for k in range(10):
        team = "A" if k < 5 else "B"
        x0, y0 = -30.0 + 3 * k, -15.0 + 3 * k
        frames = []
        for f in range(N):
            t = f / FPS
            x = x0 + (8.0 * t if k == 0 else 0.0)          # player 0 runs 40 ft, past half court
            u, v = x + 147, y0 + 125
            gap = k == 3 and 40 <= f < 90                  # player 3 unobserved for 1.7 s
            frames.append({"f": f, "t": t, "bbox": [u - 10, v - 60, u + 10, v], "foot": [u, v], "conf": 0.9,
                           "observed": not gap, "detected": not gap and f % 2 == 0, "foot_reliable": not gap})
        tracks.append({"track_id": k + 1, "role": "player", "team": team, "frames": frames})
    video = {"fps": FPS, "n_frames": N, "width": 1920, "height": 1080}
    court = {"frames": [{"f": f, "H_img2court": H, "ok": True} for f in range(N)]}
    return {"video": video, "tracks": tracks, "shots": [{"start_f": 0, "end_f": N - 1}]}, court


def test_ten_players_full_window_no_clamp():
    tracks, court = _tracks()
    play, rep = build_play(tracks, court)
    assert rep["ok"] and len(play["players"]) == 10
    for p in play["players"]:
        path = np.asarray(p["path"])
        assert path[0, 0] == 0.0 and abs(path[-1, 0] - play["duration"]) < 1e-6
        step = np.linalg.norm(np.diff(path[:, 1:], axis=0), axis=1).max()
        assert step < 1.0                                   # no teleport per 0.1 s keyframe
    runner = next(p for p in play["players"] if p["id"] == "T1")
    assert max(r[2] for r in runner["path"]) > 14.4 and play["fullCourt"]   # past half court, not clamped
    gap = next(p for p in play["players"] if p["id"] == "T4")
    assert gap["predictedIntervals"]                         # gap is filled but flagged
    assert play["metadata"].get("ballObservation") == "unknown" and play["ball"] == []


def test_director_timeline():
    tracks, court = _tracks()
    knots = [[0, 1.0], [6, 2.0], [8, 2.0], [10, 4.5]]          # slow phase, 2 s freeze, then faster
    play, rep = build_play(tracks, court, timeline=knots)
    assert rep["ok"] and rep["timeline_source"] == "director"
    assert play["metadata"]["sourceWindow"] == [1.0, 4.5]
    off = play["metadata"]["sourceTimeOffset"]
    assert [[a, round(b + off, 3)] for a, b in play["exportTimeline"]] == [[0, 1.0], [6, 2.0], [8, 2.0], [10, 4.5]]
    assert len(play["players"]) == 10


def test_bad_timeline_falls_back():
    tracks, court = _tracks()
    play, rep = build_play(tracks, court, timeline=[[0, 3.0], [2, 1.0]])   # source goes backwards
    assert rep["ok"] and rep["timeline_source"] == "default"
    assert any("timeline ignored" in d for d in rep["degraded"])


def test_annotations_validated():
    tracks, court = _tracks()
    ann = {"annotations": [
        {"type": "arrow", "path": {"who": "T1", "t": [0.5, 2.0]}, "t": [0.5, 2.5], "color": "off", "label": "突破"},
        {"type": "arrow", "from": "T1", "to": "T2", "t": [1.0, 1.5], "color": "ball"},
        {"type": "focus", "who": "T99", "t": [1, 2]},                   # unknown player -> dropped
        {"type": "arrow", "from": "ball", "to": "T2", "t": [1, 2]},     # ball anchor -> dropped
        {"type": "spiral", "t": [1, 2]},                                # unknown type -> dropped
        {"type": "link", "ids": ["T1", "T6"], "t": [3.0, 99.0]},        # clipped to the window
    ]}
    play, rep = build_play(tracks, court, annotations=ann)
    assert rep["annotations"]["kept"] == 3 and len(rep["annotations"]["dropped"]) == 3
    assert play["annotations"][-1]["t"][1] == play["duration"]
    play0, rep0 = build_play(tracks, court)
    assert play0["annotations"] == [] and "annotations" not in rep0


def test_title_option():
    from agentdunker.replay.scene import TITLE_MAX_CHARS, clean_title
    tracks, court = _tracks()
    play, rep = build_play(tracks, court)
    assert play["title"] == "3D 战术复盘" and rep["title_source"] == "default"      # unchanged without a title
    t = "火箭客场对独行侠 · 杜兰特单打前的空间组织"
    play, rep = build_play(tracks, court, title=f"  {t}\n")
    assert play["title"] == t and rep["title"] == t and rep["title_source"] == "caller"
    play, rep = build_play(tracks, court, events={"title": "事件标题"})
    assert play["title"] == "事件标题" and rep["title_source"] == "events"
    play, _ = build_play(tracks, court, events={"title": "事件标题"}, title="命令行标题")
    assert play["title"] == "命令行标题"                                             # caller wins
    play, rep = build_play(tracks, court, title="   ")
    assert play["title"] == "3D 战术复盘" and rep["title_source"] == "default"      # blank -> default
    long = clean_title("长" * 100)
    assert len(long) == TITLE_MAX_CHARS and long.endswith("…")
