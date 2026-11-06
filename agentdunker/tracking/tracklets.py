"""Pass 1: decode, camera motion, detection, appearance features, BoT-SORT tracklets."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.stracks import multi_gmc

from .models import ReID, load_detector, torso_color
from .video import CameraMotion, VideoInfo, probe, sha256_file, iter_frames

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackConfig:
    detector: str = "yolo11x.pt"
    imgsz: int = 1280
    det_hz: float = 15.0  # detection rate; frames in between are camera-compensated interpolation
    det_conf: float = 0.08
    min_height_frac: float = 0.04  # drop boxes shorter than this fraction of frame height
    crop_conf: float = 0.3  # keep a crop (for role classification) above this confidence


class _Dets:
    """Minimal Results-like container accepted by ultralytics trackers."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy.reshape(-1, 4).astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = np.zeros(len(conf), np.float32)
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        self.xywh = np.c_[self.xyxy[:, :2] + wh / 2, wh]

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, m):
        return _Dets(self.xyxy[m], self.conf[m])


class _Tracker(BOTSORT):
    """BoT-SORT with externally supplied camera warp and OSNet features."""

    warp = np.eye(2, 3)

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high) -> None:
        multi_gmc(strack_pool, self.warp)
        multi_gmc(unconfirmed, self.warp)


def _make_tracker(det_hz: float) -> _Tracker:
    args = SimpleNamespace(
        track_high_thresh=0.3, track_low_thresh=0.1, new_track_thresh=0.35,
        track_buffer=int(det_hz * 1.5), match_thresh=0.8, fuse_score=True, gmc_method="none",
        proximity_thresh=0.5, appearance_thresh=0.75, with_reid=False, model="auto",
    )
    tr = _Tracker(args)  # track_buffer counts detection steps
    tr.args.with_reid = True
    tr.encoder = lambda feats, boxes: feats
    return tr


def run_tracker(o: "Observations") -> list[dict[int, int]]:
    """BoT-SORT over the cached detections; returns per detection step {det index: tracklet id}."""
    out_all, prev, shot = [], None, -1
    for k, f in enumerate(o.det_frames):
        if o.shot_of(f) != shot:  # new shot: fresh tracker, ids offset so they stay unique
            shot, prev = o.shot_of(f), None
            tracker = _make_tracker(o.info.fps / o.timing["stride"])
            base = shot * 100000
        warp = np.eye(3) if prev is None else np.linalg.inv(o.to_ref[f]) @ o.to_ref[prev]
        tracker.warp = warp[:2].astype(np.float32)
        out = tracker.update(_Dets(o.boxes[k], o.confs[k]), None, o.embs[k])
        out_all.append({int(row[7]): base + int(row[4]) for row in out} if len(out) else {})
        prev = f
    return out_all


@dataclass
class Observations:
    info: VideoInfo
    times: np.ndarray  # (N,) PTS seconds
    to_ref: np.ndarray  # (N,3,3) maps frame-f pixels to frame-0 pixels (camera-stabilised plane)
    det_frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)  # per det frame (n,4)
    confs: list[np.ndarray] = field(default_factory=list)
    embs: list[np.ndarray] = field(default_factory=list)  # (n,512)
    colors: list[np.ndarray] = field(default_factory=list)  # (n,10)
    crops: dict[tuple[int, int], bytes] = field(default_factory=dict)  # (det_k, det_idx) -> jpeg
    track_of: list[dict[int, int]] = field(default_factory=list)  # per det frame: det_idx -> tracklet id
    role_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)  # CLIP role probs per crop
    det_vlm: dict[tuple[int, int], str] = field(default_factory=dict)  # VLM label per labelled crop
    det_number: dict[tuple[int, int], str] = field(default_factory=dict)  # VLM jersey number per labelled crop
    det_jersey: dict[tuple[int, int], str] = field(default_factory=dict)  # identity pass: "<team>#<number>" (M3)
    timing: dict[str, float] = field(default_factory=dict)
    shots: list[int] = field(default_factory=lambda: [0])  # first frame of every shot
    flashes: list[int] = field(default_factory=list)

    def shot_of(self, f: int) -> int:
        return int(np.searchsorted(self.shots, f, side="right") - 1)


def _crop(img: np.ndarray, b: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = np.clip(b, 0, [w - 1, h - 1, w - 1, h - 1]).astype(int)
    return img[y1 : max(y1 + 2, y2), x1 : max(x1 + 2, x2)]


def collect(path: str, cfg: TrackConfig) -> Observations:
    t_start = time.time()
    fps, W, Hh = probe(path)
    stride = max(1, int(round(fps / cfg.det_hz)))
    det = load_detector(cfg.detector)
    reid = ReID()
    cam = CameraMotion(W)
    times, to_ref = [], []
    obs_lists: dict[str, list] = {k: [] for k in ("det_frames", "boxes", "confs", "embs", "colors")}
    crops: dict[tuple[int, int], bytes] = {}
    cum = np.eye(3)  # frame-0 -> current
    last_boxes = None
    t_det = t_feat = 0.0
    n = 0
    shots, flashes = [0], []
    for f, t, img in iter_frames(path):
        A, event = cam.step(img, last_boxes)
        if event == "cut":
            shots.append(f)
        elif event == "flash":
            flashes.append(f)
        cum = A @ cum
        times.append(t)
        to_ref.append(np.linalg.inv(cum))
        n = f + 1
        if f % stride:
            continue
        t0 = time.time()
        r = det.predict(img, imgsz=cfg.imgsz, conf=cfg.det_conf, classes=[0], verbose=False)[0]
        xyxy, conf = r.boxes.xyxy.numpy(), r.boxes.conf.numpy()
        keep = (xyxy[:, 3] - xyxy[:, 1]) >= cfg.min_height_frac * Hh
        xyxy, conf = xyxy[keep], conf[keep]
        t1 = time.time()
        cr = [_crop(img, b) for b in xyxy]
        emb = reid(cr)
        col = np.stack([torso_color(c) for c in cr]) if cr else np.zeros((0, 10), np.float32)
        k = len(obs_lists["det_frames"])
        for i, (c, s) in enumerate(zip(cr, conf)):
            if s >= cfg.crop_conf:
                crops[(k, i)] = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        last_boxes = xyxy[conf > 0.3]
        t_det += t1 - t0
        t_feat += time.time() - t1
        for k_, v in zip(("det_frames", "boxes", "confs", "embs", "colors"), (f, xyxy, conf, emb, col)):
            obs_lists[k_].append(v)
    info = VideoInfo(path, sha256_file(path), fps, W, Hh, n)
    o = Observations(info, np.array(times), np.array(to_ref), crops=crops, shots=shots, flashes=flashes, **obs_lists)
    o.timing = {"pass1_s": time.time() - t_start, "detect_s": t_det, "features_s": t_feat, "stride": stride}
    o.track_of = run_tracker(o)
    logger.info("pass1 %d frames, %d det frames, shots at %s, flashes at %s, %.1fs", n, len(o.det_frames),
                shots, flashes, o.timing["pass1_s"])
    return o
