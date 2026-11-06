"""One-call entry point used by the CLI and by M7."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .compose import render_replay
from .scene import SceneConfig, build_play

logger = logging.getLogger(__name__)


def _load(path: Optional[str]) -> Optional[dict]:
    return json.loads(Path(path).read_text()) if path else None


def make_replay(video: str, tracks: str, court: str, out: str, *, identity: Optional[str] = None,
                events: Optional[str] = None, start: Optional[float] = None, end: Optional[float] = None,
                compare_out: Optional[str] = None, width: int = 1920, height: int = 1080, fps: int = 30,
                workers: int = 3, pip: bool = True, max_seconds: Optional[float] = None,
                cfg: SceneConfig = SceneConfig(), timeline: Optional[list] = None,
                annotations: Optional[str] = None, title: Optional[str] = None) -> dict:
    """Build the scene and render it. Always writes ``<out>.json``; ``report['ok']`` is False when
    there is no usable 3D window (the caller then skips the 3D segment instead of failing)."""
    t0 = time.monotonic()
    tr, co = _load(tracks), _load(court)
    ann = None
    if annotations:
        try:
            ann = _load(annotations)
        except (OSError, ValueError) as e:
            ann = []
            logger.warning("unreadable annotations %s (%s)", annotations, e)
    play, report = build_play(tr, co, _load(identity), _load(events), start, end, cfg, timeline=timeline,
                              annotations=ann, title=title)
    report["inputs"] = {"video": video, "tracks": tracks, "court": court, "identity": identity, "events": events}
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if play is not None:
        out_p.with_suffix(".play.json").write_text(json.dumps(play, ensure_ascii=False))
        report["play_json"] = str(out_p.with_suffix(".play.json"))
        report["scene_seconds"] = round(time.monotonic() - t0, 2)
        try:
            report["render"] = render_replay(play, video, tr, co, out, width=width, height=height, fps=fps,
                                             workers=workers, pip=pip, compare_out=compare_out, max_seconds=max_seconds)
        except Exception as e:  # noqa: BLE001 - the chain must go on without the 3D segment
            logger.exception("render failed")
            report.update(ok=False, reason=f"render failed: {type(e).__name__}: {e}"[:500])
        # [output_s, source_s] knots in absolute source PTS, for M6/M7 narration and cuts.
        off = play["metadata"]["sourceTimeOffset"]
        report["timeline_source_pts"] = [[a, round(b + off, 3)] for a, b in play["exportTimeline"]]
    out_p.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report
