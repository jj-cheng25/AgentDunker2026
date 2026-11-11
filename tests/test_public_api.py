"""The documented Python entry point: ``from agentdunker import make_film`` (README, package docstring)."""
import subprocess
import sys


def test_make_film_forwards_arguments(monkeypatch):
    import agentdunker
    from agentdunker import make_film, run

    seen = {}

    def fake(clip, out_dir, **kw):
        seen.update(clip=clip, out_dir=out_dir, kw=kw)
        return {"ok": True}

    monkeypatch.setattr(run, "make_film", fake)
    assert make_film is agentdunker.make_film
    assert make_film("clip.mp4", "out/", reuse="prev/", skip=("voice",)) == {"ok": True}
    assert seen == {"clip": "clip.mp4", "out_dir": "out/", "kw": {"reuse": "prev/", "skip": ("voice",)}}


def test_import_is_lazy():
    code = "import sys, agentdunker; from agentdunker import make_film; print('agentdunker.run' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
