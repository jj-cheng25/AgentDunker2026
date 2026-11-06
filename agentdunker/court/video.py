"""Video-level court calibration: shots, inter-frame motion, per-shot camera position, tracking, smoothing."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import camera as cam
from .frame_solver import frame_score, match_lines, refine, solve_frame
from .geometry import EdgeMap, project
from .keypoints import CourtKeypointDetector

logger = logging.getLogger(__name__)
WORK_W = 960


@dataclass
class Clip:
    path: str
    fps: float
    size: tuple
    small: list = field(default_factory=list)      # BGR frames at WORK_W
    pts: list = field(default_factory=list)        # seconds


def read_clip(path: str, max_frames: int | None = None) -> Clip:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        raise ValueError(f"cannot read video size: {path}")
    clip = Clip(path, fps, (w, h))
    s = WORK_W / w
    while max_frames is None or len(clip.small) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        clip.pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        clip.small.append(cv2.resize(fr, (WORK_W, round(h * s)), interpolation=cv2.INTER_AREA))
    cap.release()
    if not clip.small:
        raise ValueError(f"no decodable frames: {path}")
    if not 10.0 <= clip.fps <= 120.0:          # broken container metadata: use the PTS spacing
        dt = np.diff(clip.pts)
        dt = dt[dt > 0]
        clip.fps = float(1.0 / np.median(dt)) if len(dt) else 30.0
        clip.fps = clip.fps if 10.0 <= clip.fps <= 120.0 else 30.0
    return clip


def frame_changes(frames: list) -> tuple[np.ndarray, np.ndarray]:
    """Per frame vs the previous one: HSV-histogram Bhattacharyya distance and 'is a repeated frame' flag.

    Frame-rate conversion leaves exact repeats (e.g. every third frame during some 60 fps replays); a repeat
    must keep the previous homography or the lines swim against the stepping image.
    """
    d = np.zeros(len(frames))
    dup = np.zeros(len(frames), bool)
    prev = prev_g = None
    for i, fr in enumerate(frames):
        small = cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            d[i] = cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)
            dup[i] = float(np.abs(g - prev_g).mean()) < 0.6
        prev, prev_g = hist, g
    return d, dup


def is_cut(hist_d: float, inlier_ratio: float) -> bool:
    """Hard cuts break feature tracking; dissolves show a colour change with weak tracking."""
    return inlier_ratio < 0.15 or (hist_d > 0.3 and inlier_ratio < 0.7) or hist_d > 0.5


def static_overlay_mask(frames: list, step: int = 5) -> np.ndarray:
    """255 where pixels may be used; 0 on screen-fixed graphics (score bug, logos) when the camera moves."""
    sample = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[::step]]).astype(np.float32)
    mask = np.full(sample.shape[1:], 255, np.uint8)
    if len(sample) < 6:
        return mask
    std = sample.std(0)
    static = (std < 6.0).astype(np.uint8)
    static = cv2.morphologyEx(static, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    static = cv2.dilate(static, np.ones((9, 9), np.uint8))
    if static.mean() < 0.2:   # camera barely moved -> do not mask the court itself
        mask[static > 0] = 0
    return mask


def interframe_motion(a: np.ndarray, b: np.ndarray, mask: np.ndarray, scale: float):
    """Homography mapping source-pixel coords of frame a to frame b (camera rotation/zoom), or None."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(ga, 600, 0.01, 8, mask=mask)
    if pts is None or len(pts) < 20:
        return None, 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, winSize=(21, 21), maxLevel=3)
    good = st.ravel() == 1
    if good.sum() < 20:
        return None, 0.0
    M, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 1.5)
    if M is None:
        return None, 0.0
    S = np.diag([scale, scale, 1.0])
    return np.linalg.inv(S) @ M @ S, float(inl.mean())


def fit_rotation_to_H(H_target: np.ndarray, p_init: np.ndarray, size, C: np.ndarray) -> np.ndarray:
    """4-DOF camera (fixed C) closest to H_target on a court grid visible in the image."""
    gx, gy = np.meshgrid(np.linspace(-47, 47, 13), np.linspace(-25, 25, 7))
    g = np.c_[gx.ravel(), gy.ravel()]
    q = project(H_target, g)
    w, h = size
    vis = np.isfinite(q).all(1) & (q[:, 0] > -0.5 * w) & (q[:, 0] < 1.5 * w) & (q[:, 1] > -0.5 * h) & (q[:, 1] < 1.5 * h)
    if vis.sum() < 4:
        vis[:] = True
    p = np.array(p_init, float)
    p[4:7] = C

    def r(x):
        pp = p.copy()
        pp[:4] = x
        return (project(cam.to_H(pp, size), g[vis]) - q[vis]).ravel()

    p[:4] = least_squares(r, p[:4], loss="soft_l1", f_scale=5.0, max_nfev=50).x
    return p


@dataclass
class FrameState:
    p: np.ndarray | None = None
    score: float = 0.0
    src: str = "none"            # measured | propagated | interpolated | none
    reproj: float = float("nan")


class CourtTracker:
    def __init__(self, detector: CourtKeypointDetector | None = None, kp_hz: float = 2.0):
        self.det = detector or CourtKeypointDetector()
        self.kp_hz = kp_hz

    def run(self, clip: Clip, ignore_boxes: dict | None = None) -> dict:
        n = len(clip.small)
        size = clip.size
        scale = WORK_W / size[0]
        ov_mask = static_overlay_mask(clip.small)
        hist_d, dup = frame_changes(clip.small)
        motion = [None] * n
        starts = [0]
        for i in range(n - 1):
            M, ratio = interframe_motion(clip.small[i], clip.small[i + 1], ov_mask, scale)
            if is_cut(hist_d[i + 1], ratio):
                starts.append(i + 1)            # dissolves give runs of 1-2 frame shots, which are skipped
            else:
                motion[i] = M
        starts.append(n)
        shots = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]
        logger.info("%d frames, shots %s", n, shots)

        def edge_map(i):
            boxes = [np.asarray(b, float) * scale for b in (ignore_boxes or {}).get(i, [])]
            em = EdgeMap.from_frame(clip.small[i], WORK_W, boxes, build=False)   # boxes already in work px
            em.scale = scale                                        # source px -> work px
            em.mask = cv2.bitwise_and(em.mask, ov_mask)
            em.build_dts()
            return em

        states = [FrameState() for _ in range(n)]
        kp_step = max(1, int(round(clip.fps / self.kp_hz)))
        for s0, s1 in shots:
            if s1 - s0 < 3:
                continue
            try:
                self._run_shot(clip, s0, s1, kp_step, motion, edge_map, states, dup)
                smooth_shot(states, s0, s1, size, clip.fps, motion=motion)
                for i in range(s0, s1):
                    finalize(states[i], edge_map(i), size)
            except Exception:                   # degrade this shot to ok=false rather than lose the clip
                logger.exception("shot %d-%d failed; frames marked not ok", s0, s1)
                for i in range(s0, s1):
                    states[i] = FrameState()
        return {"states": states, "shots": shots, "motion": motion}

    def _run_shot(self, clip, s0, s1, kp_step, motion, edge_map, states, dup):
        size = clip.size
        ems = {}

        def em_of(i):
            if i not in ems:
                ems[i] = edge_map(i)
            return ems[i]

        # 1) independent measurements on keyframes
        meas = {}
        for i in range(s0, s1, kp_step):
            full = cv2.resize(clip.small[i], size, interpolation=cv2.INTER_LINEAR)
            kp, _ = self.det(clip.small[i])
            kp[:, :2] /= WORK_W / size[0]
            fit = solve_frame(full, kp, em_of(i))
            if fit is not None:
                meas[i] = (fit, kp)
        good = {i: m for i, m in meas.items() if m[0].score >= 0.3}
        if not good:
            good = {i: m for i, m in meas.items() if m[0].score >= 0.2}
        if not good:
            logger.info("shot %d-%d: no court", s0, s1)
            return
        # 2) shared camera position for the shot
        Cs = np.array([m[0].p[4:7] for m in good.values()])
        ws = np.array([m[0].score for m in good.values()])
        C = np.array([_wmedian(Cs[:, k], ws) for k in range(3)])
        # 3) re-fit every keyframe measurement with fixed C, pick the anchor
        refit = {}
        for i, (fit, kp) in meas.items():
            p0 = fit_rotation_to_H(fit.H, fit.p, size, C)
            p, _ = refine(p0, em_of(i), size, fixed_C=C, radii=(16.0, 8.0))
            refit[i] = (p, _score(p, em_of(i), size))
        anchor = max(refit, key=lambda i: refit[i][1])
        states[anchor] = FrameState(refit[anchor][0], refit[anchor][1], "measured")
        sig = np.r_[0.01, 0.01, 0.01, 0.02, 1, 1, 1]
        # 4) track both directions from the anchor; keyframe measurements and re-detections compete
        for direction in (1, -1):
            prev = anchor
            last_det = anchor
            rng = range(anchor + 1, s1) if direction > 0 else range(anchor - 1, s0 - 1, -1)
            for i in rng:
                if dup[i] if direction > 0 else dup[i + 1]:      # repeated frame: same camera as its twin
                    states[i] = FrameState(states[prev].p.copy(), states[prev].score, states[prev].src)
                    prev = i
                    continue
                p_prev = states[prev].p
                M = motion[prev] if direction > 0 else (None if motion[i] is None else np.linalg.inv(motion[i]))
                H_init = cam.to_H(p_prev, size) if M is None else M @ cam.to_H(p_prev, size)
                p0 = fit_rotation_to_H(H_init, p_prev, size, C)
                p, _ = refine(p0, em_of(i), size, fixed_C=C, radii=(8.0,), prior=(p0, sig), max_nfev=25)
                best = (p, _score(p, em_of(i), size), "propagated")
                if i in refit and refit[i][1] > best[1] + 0.03:
                    best = (refit[i][0], refit[i][1], "measured")
                if best[1] < 0.2 and i not in refit and abs(i - last_det) >= 3:
                    last_det = i
                    kp, _ = self.det(clip.small[i])
                    kp[:, :2] /= WORK_W / size[0]
                    full = cv2.resize(clip.small[i], size, interpolation=cv2.INTER_LINEAR)
                    fit = solve_frame(full, kp, em_of(i), init=[p], fixed_C=C)
                    if fit is not None and fit.score > best[1] + 0.03:
                        best = (fit.p, fit.score, "measured")
                states[i] = FrameState(best[0], best[1], best[2])
                prev = i
                for k in [k for k in ems if abs(k - i) > 2 and k != anchor]:
                    del ems[k]


def _score(p: np.ndarray, em: EdgeMap, size) -> float:
    return frame_score(cam.to_H(p, size), em, 3.0)[0]


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


def _visible_grid(p: np.ndarray, size) -> np.ndarray:
    gx, gy = np.meshgrid(np.linspace(-47, 47, 19), np.linspace(-25, 25, 11))
    g = np.c_[gx.ravel(), gy.ravel()]
    q = project(cam.to_H(p, size), g)
    w, h = size
    vis = np.isfinite(q).all(1) & (q[:, 0] > 0) & (q[:, 0] < w) & (q[:, 1] > 0) & (q[:, 1] < h)
    return g[vis] if vis.sum() >= 4 else g


def _dead_reckon(states: list, s0: int, s1: int, size, motion: list) -> list:
    """Camera trajectory from KLT image motion alone, restarted from the line fit wherever motion is missing."""
    q = [None] * (s1 - s0)
    for i in range(s0, s1):
        p = states[i].p
        if p is None:
            continue
        prev = q[i - 1 - s0] if i > s0 else None
        M = motion[i - 1] if i > s0 else None
        if prev is None or M is None:
            q[i - s0] = p.copy()
        else:
            q[i - s0] = fit_rotation_to_H(M @ cam.to_H(prev, size), prev, size, prev[4:7])
    return q


def smooth_shot(states: list, s0: int, s1: int, size, fps: float, motion: list | None = None,
                window_s: float = 0.5, outlier_px: float = 12.0):
    """Complementary filter: KLT dead-reckoning carries fast camera motion, the line fits correct its drift.

    The residual (line-fitted camera - dead-reckoned camera) changes slowly, so it is smoothed robustly with a
    local quadratic; frames whose fit disagrees by > outlier_px become 'interpolated' and take the smoothed value.
    Repeated frames have identity motion, so they keep their twin's camera automatically.
    """
    idx = [i for i in range(s0, s1) if states[i].p is not None]
    if len(idx) < 3:
        return
    q = _dead_reckon(states, s0, s1, size, motion or [None] * len(states))
    t = np.array(idx, float)
    R = np.array([states[i].p[:4] - q[i - s0][:4] for i in idx])
    w0 = np.array([max(states[i].score, 1e-3) for i in idx])
    half = max(2.0, window_s * fps)
    w = w0.copy()
    for _ in range(3):
        S = _local_quadratic(t, R, w, t, half)
        dev = []
        for k, i in enumerate(idx):
            ps = np.r_[q[i - s0][:4] + S[k], states[i].p[4:]]
            g = _visible_grid(states[i].p, size)
            dev.append(np.median(np.linalg.norm(project(cam.to_H(ps, size), g) - project(cam.to_H(states[i].p, size), g), axis=1)))
        w = w0 * (np.array(dev) < outlier_px)
        if w.sum() == 0:
            w = w0
    S = _local_quadratic(t, R, w, np.arange(s0, s1, dtype=float), half)
    good = set(np.array(idx)[w > 0].tolist())
    for k, i in enumerate(range(s0, s1)):
        st = states[i]
        if st.p is None or q[k] is None or not np.all(np.isfinite(S[k])):
            continue
        st.p = np.r_[q[k][:4] + S[k], st.p[4:]]
        if i not in good:
            st.src = "interpolated"


def _local_quadratic(t, Y, w, tq, half):
    out = np.full((len(tq), Y.shape[1]), np.nan)
    for k, x in enumerate(tq):
        kw = w * np.exp(-0.5 * ((t - x) / half) ** 2)
        m = kw > 1e-4 * max(kw.max(), 1e-12)
        if kw[m].sum() < 1e-6:
            continue
        if m.sum() >= 5:
            A = np.c_[np.ones(m.sum()), t[m] - x, (t[m] - x) ** 2]
        elif m.sum() >= 2:
            A = np.c_[np.ones(m.sum()), t[m] - x]
        else:
            A = np.ones((1, 1))
        sw = np.sqrt(kw[m])[:, None]
        coef, *_ = np.linalg.lstsq(A * sw, Y[m] * sw, rcond=None)
        out[k] = coef[0]
    return out


def finalize(st: FrameState, em: EdgeMap, size):
    """Line support and residual of the final (smoothed) camera."""
    if st.p is None:
        return
    H = cam.to_H(st.p, size)
    st.score = frame_score(H, em, 3.0)[0]
    c, q, n, _, _ = match_lines(H, em, radius=8.0, step=2)
    st.reproj = float(np.median(np.abs(((project(H, c) - q) * n).sum(1)))) if len(c) else float("nan")
