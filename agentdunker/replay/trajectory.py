"""Court-space player trajectories: foot point x homography -> robust Kalman + RTS smoothing.

Units are feet in the PLAN court frame (origin centre court, x along the length, +y near sideline).
Every player gets a position at every sample time: gaps are interpolated by the smoother, and
before the first / after the last measurement the player is held (velocity decays to zero).
``observed`` marks samples that are near an accepted measurement; the rest are predictions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)
COURT_X, COURT_Y = 47.0, 25.0


@dataclass(frozen=True)
class SmoothConfig:
    """Motion model and gating. Defaults are tuned on c01/c14 (see status/m4.md)."""
    rate_hz: float = 30.0          # smoother sample rate
    foot_px_std: float = 8.0       # foot localisation noise (px) ...
    foot_px_rel: float = 0.03      # ... plus this fraction of the box height
    accel_std: float = 8.0         # white-acceleration process noise (ft/s^2 per sqrt(s))
    vel_tau: float = 1.2           # s, velocity decay; makes unobserved players settle instead of drift
    gate_chi2: float = 13.8        # innovation gate (99.9 % for 2 dof)
    reinit_after: float = 0.6      # s of consecutive rejections before trusting measurements again
    max_speed: float = 30.0        # ft/s (~9.1 m/s, NBA sprint ceiling)
    post_sigma: float = 0.15       # s, final Gaussian smoothing; removes snaps when a long gap re-locks
    observed_window: float = 0.25  # s, a sample is "observed" if a measurement is this close
    court_margin: float = 4.0      # ft outside the court lines still accepted
    box_height_jump: float = 1.2   # box taller than this x its rolling median -> merged box, foot unreliable
    box_median_window: float = 1.0 # s, rolling window for that median


@dataclass
class PlayerPath:
    track_id: int
    team: Optional[str]
    t: np.ndarray                  # (N,) source seconds
    xy: np.ndarray                 # (N, 2) feet
    observed: np.ndarray           # (N,) bool
    n_meas: int = 0
    n_rejected: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def observed_frac(self) -> float:
        return float(self.observed.mean()) if len(self.observed) else 0.0


def apply_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = H @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]


def _h_jacobian(H: np.ndarray, xy: np.ndarray, eps: float = 1.0) -> np.ndarray:
    c = apply_h(H, xy)
    jx = (apply_h(H, xy + [eps, 0]) - c) / eps
    jy = (apply_h(H, xy + [0, eps]) - c) / eps
    return np.stack([jx, jy], axis=1)


class CourtFrames:
    """Per-frame homographies from court.json; ``None`` where the frame is not usable."""

    def __init__(self, court: dict) -> None:
        self.H: dict[int, np.ndarray] = {}
        for fr in court.get("frames", []):
            if fr.get("ok") and fr.get("H_img2court") is not None:
                self.H[int(fr["f"])] = np.asarray(fr["H_img2court"], dtype=float)

    def get(self, f: int) -> Optional[np.ndarray]:
        return self.H.get(f)

    def img_to_court(self, f: int, xy) -> Optional[np.ndarray]:
        H = self.H.get(f)
        return None if H is None else apply_h(H, np.asarray(xy, float))

    def court_to_img(self, f: int, xy) -> Optional[np.ndarray]:
        H = self.H.get(f)
        return None if H is None else apply_h(np.linalg.inv(H), np.asarray(xy, float))


def measurements(track: dict, court: CourtFrames, t0: float, t1: float, cfg: SmoothConfig):
    """Accepted foot measurements of one track in [t0, t1]: list of (t, xy_ft, R_ft2)."""
    out = []
    det = [fr for fr in track["frames"] if fr.get("detected", True) and fr.get("observed", True)]
    dt_ = np.array([fr["t"] for fr in det])
    dh = np.array([fr["bbox"][3] - fr["bbox"][1] for fr in det])
    for fr in track["frames"]:
        t = fr["t"]
        if t < t0 or t > t1 or not fr.get("observed", True) or not fr.get("foot_reliable", True):
            continue
        if fr.get("detected") is False:   # M1 interpolates between detections; keep real boxes only
            continue
        H = court.get(fr["f"])
        if H is None:
            continue
        foot = np.asarray(fr["foot"], float)
        xy = apply_h(H, foot)
        if not np.all(np.isfinite(xy)) or abs(xy[0]) > COURT_X + cfg.court_margin \
                or abs(xy[1]) > COURT_Y + cfg.court_margin:
            continue
        box_h = max(1.0, fr["bbox"][3] - fr["bbox"][1])
        near = np.abs(dt_ - t) <= cfg.box_median_window / 2
        if near.sum() >= 3 and box_h > cfg.box_height_jump * np.median(dh[near]):
            continue   # merged / leg-extended box: its bottom is someone else's feet
        sigma = cfg.foot_px_std + cfg.foot_px_rel * box_h
        J = _h_jacobian(H, foot)
        R = J @ J.T * sigma ** 2 + np.eye(2) * 0.05
        out.append((t, xy, R))
    return out


def _model(dt: float, cfg: SmoothConfig):
    a = np.exp(-dt / cfg.vel_tau)
    g = cfg.vel_tau * (1 - a)
    F = np.eye(4)
    F[0, 2] = F[1, 3] = g
    F[2, 2] = F[3, 3] = a
    q = cfg.accel_std ** 2
    Q1 = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    return F, Q


def smooth_track(times: np.ndarray, meas: list, cfg: SmoothConfig) -> tuple[np.ndarray, np.ndarray, int]:
    """Robust forward Kalman filter + RTS smoother on a fixed time grid.

    Returns (xy (N,2), observed (N,), n_rejected). Requires at least one measurement.
    """
    n = len(times)
    dt = 1.0 / cfg.rate_hz
    F, Q = _model(dt, cfg)
    Hm = np.zeros((2, 4))
    Hm[0, 0] = Hm[1, 1] = 1
    # Bucket measurements onto the grid (several per bucket are fused sequentially).
    buckets: dict[int, list] = {}
    for t, z, R in meas:
        k = int(round((t - times[0]) * cfg.rate_hz))
        if 0 <= k < n:
            buckets.setdefault(k, []).append((z, R))
    first = min(buckets)
    x = np.array([*buckets[first][0][0], 0.0, 0.0])
    P = np.diag([4.0, 4.0, 100.0, 100.0])
    xs_f = np.zeros((n, 4)); Ps_f = np.zeros((n, 4, 4))
    xs_p = np.zeros((n, 4)); Ps_p = np.zeros((n, 4, 4))
    used = np.zeros(n, bool)
    rejected, streak = 0, 0.0
    for k in range(first, n):
        if k > first:
            x = F @ x
            P = F @ P @ F.T + Q
        xs_p[k], Ps_p[k] = x, P
        for z, R in buckets.get(k, []):
            S = Hm @ P @ Hm.T + R
            r = z - Hm @ x
            d2 = float(r @ np.linalg.solve(S, r))
            if d2 > cfg.gate_chi2 and streak < cfg.reinit_after:
                rejected += 1
                streak += dt
                continue
            if d2 > cfg.gate_chi2:   # persistent disagreement: the filter is wrong, not the detector
                P = P + np.diag([d2, d2, 25.0, 25.0])
                S = Hm @ P @ Hm.T + R
            streak = 0.0
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ r
            P = (np.eye(4) - K @ Hm) @ P
            used[k] = True
        if not buckets.get(k):
            streak = max(0.0, streak - dt) if streak else 0.0
        xs_f[k], Ps_f[k] = x, P
    # RTS backward pass.
    xs = xs_f.copy()
    for k in range(n - 2, first - 1, -1):
        C = Ps_f[k] @ F.T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
    xs[:first] = xs[first] * [1, 1, 0, 0]
    xy = _limit_motion(xs[:, :2], dt, cfg)
    if cfg.post_sigma > 0:
        xy = gaussian_filter1d(xy, cfg.post_sigma / dt, axis=0, mode="nearest")
    # observed = an accepted measurement within the window
    win = max(1, int(round(cfg.observed_window * cfg.rate_hz)))
    obs = np.convolve(used.astype(float), np.ones(2 * win + 1), mode="same") > 0
    return xy, obs, rejected


def _limit_motion(xy: np.ndarray, dt: float, cfg: SmoothConfig) -> np.ndarray:
    """Speed clamp only (teleport guard). The RTS output already respects the acceleration model."""
    out = xy.copy()
    vmax = cfg.max_speed * dt
    for k in range(1, len(out)):
        step = out[k] - out[k - 1]
        d = np.linalg.norm(step)
        if d > vmax:
            out[k] = out[k - 1] + step / d * vmax
    return out


def build_paths(tracks: dict, court: dict, t0: float, t1: float, cfg: SmoothConfig = SmoothConfig(),
                track_ids: Optional[list[int]] = None) -> list[PlayerPath]:
    """Smoothed court paths for players (role=player) on a common grid over [t0, t1].

    Players without any measurement in the window are seeded from their closest measurement
    outside it (held, ``observed`` all False); players never measured at all are skipped.
    """
    cf = CourtFrames(court)
    times = np.arange(t0, t1 + 1e-9, 1.0 / cfg.rate_hz)
    paths = []
    for tr in tracks["tracks"]:
        if track_ids is not None and tr["track_id"] not in track_ids:
            continue
        if track_ids is None and tr.get("role") != "player":
            continue
        meas = measurements(tr, cf, t0, t1, cfg)
        if meas:
            xy, obs, rej = smooth_track(times, meas, cfg)
            paths.append(PlayerPath(tr["track_id"], tr.get("team"), times, xy, obs, len(meas), rej))
            continue
        outside = measurements(tr, cf, -1e9, 1e9, cfg)
        if not outside:
            logger.warning("track %s has no usable court measurement; skipped", tr["track_id"])
            continue
        near = min(outside, key=lambda m: min(abs(m[0] - t0), abs(m[0] - t1)))
        xy = np.repeat(near[1][None], len(times), axis=0)
        paths.append(PlayerPath(tr["track_id"], tr.get("team"), times, xy, np.zeros(len(times), bool), 0, 0,
                                {"seeded_from_t": near[0]}))
    return paths


def path_stats(p: PlayerPath) -> dict:
    """Max speed / acceleration / step for QA (ft, s)."""
    dt = np.diff(p.t)
    v = np.diff(p.xy, axis=0) / dt[:, None]
    sp = np.linalg.norm(v, axis=1)
    acc = np.linalg.norm(np.diff(v, axis=0), axis=1) / dt[1:] if len(v) > 1 else np.zeros(1)
    return {"max_speed_ftps": float(sp.max(initial=0)), "p99_accel_ftps2": float(np.percentile(acc, 99)) if len(acc) else 0.0,
            "max_step_ft": float(np.linalg.norm(np.diff(p.xy, axis=0), axis=1).max(initial=0)),
            "observed_frac": p.observed_frac, "n_meas": p.n_meas, "n_rejected": p.n_rejected}
