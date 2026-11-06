"""``agentdunker ui``: a local Gradio page — upload a clip, watch the stages, preview and download the film.

Runs on 127.0.0.1 by default; nothing is uploaded anywhere except the model calls the config allows.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path

from . import config


def _job_dir(video: str) -> Path:
    h = hashlib.sha1(Path(video).read_bytes()).hexdigest()[:12]
    d = config.path("cache_dir").parent / "ui_runs" / f"{h}_{config.get()['run']['preset']}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run(video: str, preset: str, voice: bool):
    from . import runner
    from .cli import LABELS
    from .run import make_film

    config.override(run={"preset": preset}, voice={"backend": "moss" if voice else "none"})
    q: "queue.Queue" = queue.Queue()
    t0 = time.monotonic()

    def prog(stage, state, info):
        sec = info.get("seconds")
        tail = f" {sec:.1f}s" if sec is not None and state != "start" else ""
        q.put(f"[{time.monotonic() - t0:6.1f}s] {'▶✓✓✗'[['start', 'done', 'cached', 'failed'].index(state)]} "
              f"{LABELS.get(stage, stage)}{tail}{' (cached)' if state == 'cached' else ''}")

    runner._listeners.clear()
    out = _job_dir(video)
    box: dict = {}
    th = threading.Thread(target=lambda: box.update(rep=make_film(video, str(out), progress=prog)), daemon=True)
    th.start()
    log: list[str] = []
    while th.is_alive() or not q.empty():
        try:
            log.append(q.get(timeout=0.5))
            yield "\n".join(log), None, None
        except queue.Empty:
            pass
    rep = box.get("rep") or {}
    meta = rep.get("agentdunker", {})
    log.append("")
    log.append(f"film: {rep.get('duration')} s · made in {meta.get('seconds')} s" if rep.get("ok")
               else f"no film: {rep.get('error')}")
    log += [f"  {d['tier']} {d['element']}: {d['reason'][:100]}" for d in meta.get("degraded", [])]
    yield "\n".join(log), rep.get("film"), str(out / "report.json")


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> int:
    try:
        import gradio as gr
    except ImportError:
        print("the UI needs the [ui] extra:  pip install -e '.[ui]'")
        return 1
    with gr.Blocks(title="AgentDunker") as demo:
        gr.Markdown("## AgentDunker\nNBA 转播进攻片段（5–20 秒）→ 带解说的 3D 战术复盘短片")
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="clip", sources=["upload"])
                preset = gr.Radio(["fast", "quality"], value=config.get()["run"]["preset"], label="preset")
                voice = gr.Checkbox(value=config.get()["voice"]["backend"] == "moss", label="MOSS voice (needs MOSS_API_KEY)")
                go = gr.Button("Make film", variant="primary")
            with gr.Column():
                log = gr.Textbox(label="progress", lines=18)
                film = gr.Video(label="film")
                report = gr.File(label="report.json")
        go.click(_run, [video, preset, voice], [log, film, report])
    demo.queue().launch(server_name=host, server_port=port, share=share)
    return 0


if __name__ == "__main__":
    print(json.dumps(config.public(), indent=1))
