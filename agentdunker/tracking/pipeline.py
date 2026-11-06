"""End-to-end: video -> tracks.json dict."""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .export import to_contract
from .hardlinks import apply_identity, split_numbers
from .identities import (Chain, Tracklet, adjudicate_vlm, assign_groups, build_chains, build_tracklets,
                         classify_roles, link_shots, select_tiles, split_mixed, split_team_switches)
from .models import RoleClassifier, set_threads
from .recover import Recovery
from .tracklets import Observations, TrackConfig, collect
from .video import sha256_file

logger = logging.getLogger(__name__)


def resolve_identities(o: Observations, tls: list[Tracklet]) -> list[Chain]:
    """Role-classified tracklets -> fixed identity slots with recovered detections."""
    tls = split_team_switches(o, split_numbers(o, split_mixed(o, tls)), assign_groups(tls))
    assign_groups(tls)
    chains = build_chains(o, tls, o.timing["stride"] / o.info.fps)
    Recovery(o, chains).run()
    return link_shots(o, chains)


def label_roles(o: Observations, tls: list[Tracklet], clf: RoleClassifier, use_vlm: bool = True) -> dict | None:
    """CLIP roles + one VLM adjudication; the VLM request overlaps the full CLIP pass.

    A quick 1-crop CLIP pass over the strongest tracklets ranks the VLM tiles, then the request runs in a
    thread while the full CLIP pass (<= 3 crops per tracklet) computes on the CPU.
    """
    if not use_vlm:
        classify_roles(o, tls, clf)
        return None
    classify_roles(o, tls, clf, budget=60, per=1)
    cands = select_tiles(o, tls)
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(adjudicate_vlm, o, tls, cands)
        classify_roles(o, tls, clf)
        return fut.result()


def _pass1(path: str, cfg: TrackConfig, work_dir: str | None) -> tuple[Observations, Path | None]:
    """Pass 1, reused from ``work_dir`` when an earlier run of the same video and config saved it."""
    if work_dir is None:
        return collect(path, cfg), None
    key = hashlib.sha256((sha256_file(path) + json.dumps(asdict(cfg), sort_keys=True)).encode()).hexdigest()[:16]
    p = Path(work_dir) / f"pass1_{key}.pkl"
    if p.exists():
        logger.info("pass1 reused from %s", p)
        with open(p, "rb") as f:
            return pickle.load(f), p  # our own cache file; never point work_dir at untrusted data
    return collect(path, cfg), p


def track_video(path: str, cfg: TrackConfig | None = None, threads: int = 16, use_vlm: bool = True,
                identity: dict | None = None, base_tracks: dict | None = None, work_dir: str | None = None,
                max_hold: float | None = None) -> dict:
    """Detect, track and team-assign all players and referees of a broadcast clip.

    Returns a dict following the PLAN.md ``tracks.json`` contract (plus ``meta``).
    Identity pass (optional): ``identity`` is M3's identity.json computed on ``base_tracks`` (an earlier
    tracks.json of this video); its confident numbers become hard links. ``work_dir`` keeps pass 1 so
    that second run skips detection. ``max_hold`` trims predicted frames (see ``export._segment_frames``).
    """
    cfg = cfg or TrackConfig()
    set_threads(threads)
    t0 = time.time()
    o, cache = _pass1(path, cfg, work_dir)
    t1 = time.time()
    tls = build_tracklets(o)
    vlm_res = label_roles(o, tls, RoleClassifier(), use_vlm)
    if cache is not None and not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(o, f)
    t2 = time.time()
    o.det_jersey = {}
    ident = None
    if identity is not None:
        if base_tracks is None:
            raise ValueError("the identity pass needs the tracks.json that identity.json was computed on")
        ident = apply_identity(o, identity, base_tracks)
    chains = resolve_identities(o, tls)
    meta = {
        "config": asdict(cfg), "n_tracklets": len(tls),
        **({"identity": ident} if ident is not None else {}),
        "vlm": None if vlm_res is None else {k: vlm_res[k] for k in ("model", "team1_jersey", "team2_jersey", "cache")},
        "timing_s": {**{k: round(v, 1) for k, v in o.timing.items()}, "roles_and_vlm": round(t2 - t1, 1),
                     "total": round(time.time() - t0, 1)},
    }
    if max_hold is not None:
        meta["max_hold_s"] = max_hold
    return to_contract(o, chains, meta, max_hold)
