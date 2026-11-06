"""End-to-end smoke test on a generated clip (no NBA footage, no weights, no network).

The analysis stages are skipped, so this exercises the always-a-film path: director → render →
audio → mux, plus report.json. Needs ffmpeg; Chromium (intro/outro/transition) is optional.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def synthetic_clip(path, seconds=6.0, fps=30, w=1280, h=720):
    """A hardwood floor with court lines and ten moving 'players' (two jersey colours), with a tone track."""
    import cv2
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
                            stdin=subprocess.PIPE)
    rng = np.random.default_rng(0)
    start = rng.uniform([100, 250], [w - 100, h - 60], size=(10, 2))
    vel = rng.uniform(-3, 3, size=(10, 2))
    for i in range(int(seconds * fps)):
        img = np.full((h, w, 3), (60, 120, 190), np.uint8)
        cv2.rectangle(img, (40, 200), (w - 40, h - 20), (255, 255, 255), 3)
        cv2.line(img, (w // 2, 200), (w // 2, h - 20), (255, 255, 255), 3)
        cv2.circle(img, (w // 2, (200 + h - 20) // 2), 70, (255, 255, 255), 3)
        for k, (x, y) in enumerate(start + vel * i):
            color = (40, 40, 200) if k < 5 else (230, 230, 230)
            cv2.rectangle(img, (int(x) - 15, int(y) - 80), (int(x) + 15, int(y)), color, -1)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def test_film_without_analysis(tmp_path):
    from agentdunker import config
    from agentdunker.run import make_film
    config.override(llm={"provider": "none"}, voice={"backend": "none"}, moss={"vl_checks": False},
                    paths={"model_dir": str(tmp_path / "models"), "cache_dir": str(tmp_path / "cache")})
    clip = synthetic_clip(tmp_path / "clip.mp4")
    events = []
    rep = make_film(str(clip), str(tmp_path / "out"), skip=("tracks", "court", "identity", "events", "replay", "narration"),
                    progress=lambda s, st, i: events.append((s, st)))
    assert rep["ok"], rep.get("error")
    film = tmp_path / "out" / "film.mp4"
    assert film.exists() and rep["duration"] >= 5.5
    kinds = [s[0] for s in rep["structure"]]
    assert "live" in kinds
    saved = json.loads((tmp_path / "out" / "report.json").read_text())
    meta = saved["agentdunker"]
    assert set(meta["services_missing"]) == {"llm", "voice", "moss_vl"}
    assert any(d["element"] == "replay_3d" and d["tier"] == "C" for d in meta["degraded"])
    assert ("render_live", "done") in events
