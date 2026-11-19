"""Synthetic smoke test: image px == court ft (H = translation), 10 players, one runs into the backcourt."""
import copy
import os

import numpy as np
import pytest

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


def _pass_play(passer_team: str, receiver_team: str):
    """Play with one pass at 2.0 s (the passer holds the ball 1.5-2.0 s); team = "off" / "def" as labelled by scene."""
    tracks, court = _tracks()
    labels, _ = build_play(tracks, court)
    pick = lambda team, skip=(): next(int(p["id"][1:]) for p in labels["players"]   # noqa: E731
                                      if p["team"] == team and p["id"] not in ("T4", *skip))   # T4 has a gap
    a = pick(passer_team)
    b = pick(receiver_team, skip=(f"T{a}",))
    play, rep = build_play(tracks, court, events={"events": [{"type": "pass", "t": 2.0, "actors": [a, b], "conf": 0.9}]})
    return play, rep, f"T{a}"


def test_defence_holder_is_reported():
    play, rep, holder = _pass_play("def", "off")
    assert {"t": [1.5, 2.0], "hold": holder, "mode": "hands"} in play["ball"]
    assert any(d.startswith(f"ball with defence-labelled players ['{holder}']") for d in rep["degraded"])
    _, rep, _ = _pass_play("off", "off")
    assert not any("defence-labelled" in d for d in rep["degraded"])


def _chromium_ok() -> bool:
    from agentdunker.replay.browser import _chromium
    try:
        return bool(_chromium())
    except Exception:  # noqa: BLE001
        return False


def _stats_at(play: dict, t: float) -> str:
    """Render one export frame in headless Chromium; return the side panel's stats text (page errors raise)."""
    import functools
    import http.server
    import json
    import threading

    from agentdunker.replay.browser import WEB_ROOT, _Handler, _Page
    body = json.dumps(play, ensure_ascii=False, allow_nan=False).encode()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Handler, directory=str(WEB_ROOT), play_bytes=body))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    page = None
    try:
        page = _Page(server.server_port, 640, 360)
        page.frame(t, t)
        return page.cdp.evaluate("document.querySelector('[data-k=stats]').textContent")
    finally:
        if page:
            page.close()
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(bool(os.environ.get("CI")) or not _chromium_ok(), reason="needs Chromium; skipped in CI")
def test_render_any_holder():
    """2026-11-19: a defence-labelled holder crashed every frame (minimap r.k); now each case renders."""
    t = 1.75                                                   # inside the passer's hold
    off_hold, _, _ = _pass_play("off", "off")
    def_hold, _, def_id = _pass_play("def", "off")
    tracks, court = _tracks()
    no_ball, _ = build_play(tracks, court)                     # possession unknown everywhere
    unknown = copy.deepcopy(def_hold)
    for s in unknown["ball"]:
        if s.get("hold") == def_id:
            s["hold"] = "T99"
    no_def = copy.deepcopy(off_hold)
    no_def["players"] = [p for p in no_def["players"] if p["team"] == "off"]
    no_off = copy.deepcopy(def_hold)                           # the pass target disappears too -> that segment hidden
    no_off["players"] = [p for p in no_off["players"] if p["team"] == "def"]

    for play in (off_hold, def_hold):
        s = _stats_at(play, t)
        assert "持球" in s and "最近防守距离" in s and "3 m 内防守" in s, s
    for play in (unknown, no_ball):
        assert "持球" not in _stats_at(play, t)
    s = _stats_at(no_def, t)
    assert "持球" in s and "最近防守距离" not in s, s
    s = _stats_at(no_off, t)
    assert "持球" in s and "最近防守距离" not in s and "进攻平均间距" not in s, s


def test_timeline_samples_stay_inside_the_window():
    """2026-11-19 held-out h22: frame 192 of a 3.2 s window sampled 3.2000000000000006; the renderer rejected it."""
    from agentdunker.replay.compose import sample_timeline
    knots = [[0.0, 0.0], [6.4, 3.2], [7.4, 3.2]]
    ts = [sample_timeline(knots, i / 30) for i in range(int(7.4 * 30) + 1)]
    assert min(ts) >= 0.0 and max(ts) <= 3.2
