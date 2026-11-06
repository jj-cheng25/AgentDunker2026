"""AgentDunker: an NBA broadcast clip in, a narrated holographic 3D tactical breakdown film out.

    from agentdunker import make_film
    report = make_film("clip.mp4", "out/")      # out/film.mp4, out/report.json

Subpackages: tracking (players, referees, teams), court (homography), identity (jerseys, names, game),
events (ball, shot, xFG), replay (3D), narration (commentary + voice), film (director, render, assembly).
"""
__version__ = "0.1.0"


def make_film(clip: str, out_dir: str, **kw):
    from .run import make_film as _make_film
    return _make_film(clip, out_dir, **kw)
