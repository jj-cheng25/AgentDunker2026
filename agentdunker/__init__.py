"""AgentDunker: an NBA broadcast clip in, a narrated holographic 3D tactical breakdown film out.

    from agentdunker import make_film
    report = make_film("clip.mp4", "out/")      # out/film.mp4, out/report.json

Subpackages: tracking (players, referees, teams), court (homography), identity (jerseys, names, game),
events (ball, shot, xFG), replay (3D), narration (commentary + voice), film (director, render, assembly).
"""
import os as _os

__version__ = "0.1.0"

# Before anything can import ultralytics (also in stage subprocesses): no online check / telemetry / Sentry.
# Its settings file goes to <cache_dir>/ultralytics; config.apply() moves it if paths.cache_dir is configured,
# unless the user set YOLO_CONFIG_DIR themselves.
USER_YOLO_CONFIG_DIR = _os.environ.get("YOLO_CONFIG_DIR")
_os.environ["YOLO_OFFLINE"] = "true"
_os.environ.setdefault("YOLO_CONFIG_DIR", _os.path.join(_os.environ.get("XDG_CACHE_HOME") or _os.path.expanduser(
    "~/.cache"), "agentdunker", "cache", "ultralytics"))
