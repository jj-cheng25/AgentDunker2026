"""Single-frame calibration: keypoint hypotheses -> camera-model refinement against court-line edges."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import camera as cam
from .geometry import KP_COURT, KP_WEIGHT, LINES, EdgeMap, _front, project

LINE_NAMES = list(LINES)
# Lane side lines: arenas paint the key in different widths, and painted-area edges are often stronger than the
# 16 ft lane line itself, so they get little weight in the fit and none in the support score.
LINE_WEIGHT = np.array([0.25 if n.startswith("lane_") else 1.0 for n in LINE_NAMES])


@dataclass
class FrameFit:
    p: np.ndarray                 # camera params
    H: np.ndarray                 # court -> image
    score: float
    stats: dict = field(default_factory=dict)


RIDGE_MIN = 6.0       # minimum stripe response (grey levels)


def match_lines(H: np.ndarray, em: EdgeMap, radius: float, step: int = 1, skip_lane: bool = False):
    """For each projected template line sample, the nearest painted-stripe centre along its normal (source px).

    A candidate must be a local maximum of the stripe response, strong enough, and a ridge oriented along the
    template line (second derivative across the line clearly dominates).  Sub-pixel by parabola fit.
    """
    hw = em.resp.shape
    r = max(2, int(round(radius * em.scale)))
    offs = np.arange(-r - 1, r + 2)
    out_c, out_q, out_n, out_id, n_vis = [], [], [], [], 0
    for li, poly in enumerate(LINES.values()):
        if skip_lane and LINE_WEIGHT[li] < 1:
            continue
        poly = poly[::step]
        ok = _front(H, poly)
        if ok.sum() < 3:
            continue
        p = project(H, poly) * em.scale
        t = np.gradient(p, axis=0)
        tn = np.linalg.norm(t, axis=1)
        t = t / np.maximum(tn, 1e-9)[:, None]
        n = np.c_[-t[:, 1], t[:, 0]]
        vis = ok & (p[:, 0] >= 2) & (p[:, 1] >= 2) & (p[:, 0] < hw[1] - 2) & (p[:, 1] < hw[0] - 2) & (tn > 0.3 * step)
        idx = np.flatnonzero(vis)
        if len(idx) == 0:
            continue
        pi = np.clip(np.round(p[idx]).astype(int), 0, [hw[1] - 1, hw[0] - 1])
        idx = idx[em.mask[pi[:, 1], pi[:, 0]] > 0]
        n_vis += len(idx)
        if len(idx) == 0:
            continue
        nn = n[idx]
        cand = p[idx, None, :] + offs[None, :, None] * nn[:, None, :]
        cx = np.clip(np.round(cand[..., 0]).astype(int), 0, hw[1] - 1)
        cy = np.clip(np.round(cand[..., 1]).astype(int), 0, hw[0] - 1)
        R = em.resp[cy, cx]
        dnn = (em.hxx[cy, cx] * nn[:, None, 0] ** 2 + 2 * em.hxy[cy, cx] * nn[:, None, 0] * nn[:, None, 1]
               + em.hyy[cy, cx] * nn[:, None, 1] ** 2)
        dtt = (em.hxx[cy, cx] * nn[:, None, 1] ** 2 - 2 * em.hxy[cy, cx] * nn[:, None, 0] * nn[:, None, 1]
               + em.hyy[cy, cx] * nn[:, None, 0] ** 2)
        peak = np.zeros_like(R, bool)
        peak[:, 1:-1] = (R[:, 1:-1] >= R[:, :-2]) & (R[:, 1:-1] > R[:, 2:]) & (R[:, 1:-1] > RIDGE_MIN)
        peak &= (dnn < 0) & (np.abs(dnn) > 2.0 * np.abs(dtt))
        peak[:, [0, -1]] = False
        if not peak.any():
            continue
        dist = np.where(peak, np.abs(offs)[None, :], 1e9)
        k = np.argmin(dist, 1)
        has = peak[np.arange(len(k)), k]
        rows = np.flatnonzero(has)
        k = k[rows]
        y0, y1, y2 = R[rows, k - 1], R[rows, k], R[rows, k + 1]
        den = y0 - 2 * y1 + y2
        sub = np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / den, 0.0).clip(-0.5, 0.5)
        q = p[idx[rows]] + (offs[k] + sub)[:, None] * nn[rows]
        out_c.append(poly[idx[rows]])
        out_q.append(q / em.scale)
        out_n.append(nn[rows])
        out_id.append(np.full(len(rows), li))
    if not out_c:
        z = np.zeros((0, 2))
        return z, z, z, np.zeros(0, int), n_vis
    return np.concatenate(out_c), np.concatenate(out_q), np.concatenate(out_n), np.concatenate(out_id), n_vis


def support(H: np.ndarray, em: EdgeMap, tol: float = 3.0):
    """Line support of H: matched fraction within tol px, number of supported lines, visible samples."""
    c, q, n, ids, n_vis = match_lines(H, em, radius=tol, step=2, skip_lane=True)
    if n_vis == 0:
        return 0.0, 0, 0
    per_line = np.bincount(ids, minlength=len(LINE_NAMES))
    return len(c) / n_vis, int((per_line >= 6).sum()), n_vis


_OUTLINE = np.array([(-47, -25), (47, -25), (47, 25), (-47, 25)], float)


def explained(H: np.ndarray, em: EdgeMap, tol: float = 4.0, thr: float = 25.0) -> float:
    """Precision: fraction of stripe pixels on the projected court floor that lie near a projected court line."""
    hw = em.resp.shape
    S = np.diag([em.scale, em.scale, 1.0]) @ H
    floor = np.zeros(hw, np.uint8)
    poly = _OUTLINE + np.sign(_OUTLINE) * 1.0
    dense = np.concatenate([np.linspace(poly[i], poly[(i + 1) % 4], 40) for i in range(4)])
    okf = _front(S, dense)
    if okf.sum() < 3:
        return 0.0
    pf = project(S, dense[okf])
    pf = np.clip(pf, -4 * hw[1], 5 * hw[1])
    cv2.fillPoly(floor, [np.round(pf).astype(np.int32)], 255)
    stripes = (em.resp > thr) & (em.mask > 0) & (floor > 0)
    n = int(stripes.sum())
    if n < 50:
        return 0.0
    lines = np.zeros(hw, np.uint8)
    width = max(1, int(round(2 * tol * em.scale)))
    for li, poly_c in enumerate(LINES.values()):
        ok = _front(S, poly_c)
        if ok.sum() < 2:
            continue
        pp = project(S, poly_c)
        good = ok & (np.abs(pp) < 4 * hw[1]).all(1)
        if good.sum() >= 2:
            cv2.polylines(lines, [np.round(pp[good]).astype(np.int32)], False, 255, width + 2)
    return float((stripes & (lines > 0)).sum() / n)


def frame_score(H: np.ndarray, em: EdgeMap, tol: float = 3.0) -> tuple[float, dict]:
    """F1 of template recall (matched line samples) and floor precision (explained stripe pixels)."""
    rec, nl, nv = support(H, em, tol)
    prec = explained(H, em, tol + 1.0)
    f1 = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
    f1 *= min(nl, 3) / 3.0
    return f1, {"recall": round(rec, 3), "precision": round(prec, 3), "n_lines": nl, "n_vis": nv}


_SAMPLES = None


def _samples(step: int = 2):
    """Template line samples (court ft) with line ids, excluding the arena-dependent lane side lines."""
    global _SAMPLES
    if _SAMPLES is None:
        pts, ids = [], []
        for li, poly in enumerate(LINES.values()):
            if LINE_WEIGHT[li] < 1:
                continue
            pts.append(poly[::step])
            ids.append(np.full(len(poly[::step]), li))
        _SAMPLES = (np.concatenate(pts), np.concatenate(ids))
    return _SAMPLES


def _dt_residual(p, size, em: EdgeMap, sigma: float, pts, nxt):
    """Welsch-robust distance (source px) from projected samples to the nearest ridge of matching orientation."""
    H = cam.to_H(p, size)
    q = project(H, pts) * em.scale
    t = project(H, nxt) * em.scale - q
    hw = em.resp.shape
    inside = _front(H, pts) & (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < hw[1] - 1) & (q[:, 1] < hw[0] - 1)
    r = np.zeros(len(pts))
    if inside.sum() == 0:
        return r
    qi = q[inside]
    ang = (np.arctan2(t[inside, 1], t[inside, 0]) + np.pi / 2) % np.pi       # normal angle
    k = np.round(ang / (np.pi / 4)).astype(int) % 4
    xi, yi = np.round(qi[:, 0]).astype(int), np.round(qi[:, 1]).astype(int)
    usable = em.mask[yi, xi] > 0
    d = np.empty(len(qi))
    for b in range(4):
        m = k == b
        if m.any():
            d[m] = cv2.remap(em.dts[b], qi[m, 0].astype(np.float32).reshape(-1, 1),
                             qi[m, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    d = d / em.scale
    s = sigma
    rr = s * np.sqrt(2.0 * (1.0 - np.exp(-0.5 * (d / s) ** 2)))
    rr[~usable] = 0.0
    r[inside] = rr
    return r


def refine(p0: np.ndarray, em: EdgeMap, size, kp_img=None, kp_court=None, kp_w=None,
           radii=(24.0, 12.0, 6.0), fixed_C: np.ndarray | None = None, prior: tuple | None = None,
           roll_w: float = 200.0, final_radius: float = 4.0, max_nfev: int = 40):
    """Robust camera refinement against painted-line ridges (+ optional keypoints and temporal prior).

    Coarse stages: orientation-binned distance transforms with a shrinking Welsch scale (smooth basin).
    Final stage: sub-pixel ridge centres along each sample's normal.
    fixed_C keeps the camera position fixed (pan/tilt/roll/zoom only); prior = (p_prior, sigma_vec).
    """
    p = np.array(p0, float)
    free = np.arange(4) if fixed_C is not None else np.arange(7)
    if fixed_C is not None:
        p[4:7] = fixed_C
    pts, _ = _samples()
    nxt = np.roll(pts, -1, axis=0)
    nxt[-1] = pts[-1] + (pts[-1] - pts[-2])
    xs = np.r_[0.01, 0.01, 0.01, 0.01, 1, 1, 1][free]

    def extra(pp):
        parts = [[roll_w * cam.roll_of(pp)]]
        if kp_img is not None and len(kp_img):
            parts.append(((project(cam.to_H(pp, size), kp_court) - kp_img) * kp_w[:, None]).ravel())
        if prior is not None:
            parts.append((pp[free] - prior[0][free]) / prior[1][free])
        return np.concatenate([np.ravel(a) for a in parts])

    for sigma in radii:
        def resid(x, sigma=sigma):
            pp = p.copy()
            pp[free] = x
            return np.r_[_dt_residual(pp, size, em, sigma, pts, nxt), extra(pp)]
        p[free] = least_squares(resid, p[free], x_scale=xs, max_nfev=max_nfev, diff_step=1e-3).x
    stats = {}
    if final_radius:
        c, q, n, ids, n_vis = match_lines(cam.to_H(p, size), em, final_radius)
        if len(c) >= 10:
            lw = LINE_WEIGHT[ids]

            def resid(x):
                pp = p.copy()
                pp[free] = x
                return np.r_[lw * ((project(cam.to_H(pp, size), c) - q) * n).sum(1), extra(pp)]
            p[free] = least_squares(resid, p[free], loss="soft_l1", f_scale=1.5, x_scale=xs, max_nfev=max_nfev).x
        stats = {"n_line": int(len(c)), "n_vis": int(n_vis)}
    return p, stats


def keypoint_hypotheses(kp: np.ndarray, size, conf_thr=(0.5, 0.25), n_random: int = 20, max_drop: int = 12):
    """Candidate court->image homographies from in-image court-33 keypoints.

    The model hallucinates a few confident points and misses true ones at low confidence, so hypotheses come
    from two confidence tiers: RANSAC / all-points / leave-out fits on the confident set and confidence-weighted
    random 4-point subsets of the wider set.  Line support picks among them later.
    """
    w, h = size
    inimg = np.isfinite(KP_COURT[:, 0]) & (kp[:, 0] > 0) & (kp[:, 0] < w) & (kp[:, 1] > 0) & (kp[:, 1] < h)
    hyps = []
    rng = np.random.default_rng(0)
    used = np.zeros(0, int)
    for tier, thr in enumerate(conf_thr):
        idx = np.flatnonzero(inimg & (kp[:, 2] >= thr))
        if len(idx) < 4:
            continue
        used = idx
        src, dst = KP_COURT[idx].astype(np.float32), kp[idx, :2].astype(np.float32)
        for method in (cv2.RANSAC, 0):
            Hm, _ = cv2.findHomography(src, dst, method, 15.0)
            if Hm is not None:
                hyps.append(Hm)
        if tier == 0:
            drops = [d for k in (1, 2) if len(idx) - k >= 4 for d in itertools.combinations(range(len(idx)), k)]
            rng.shuffle(drops)
            subsets = [np.setdiff1d(np.arange(len(idx)), d) for d in drops[:max_drop]]
        else:
            pr = kp[idx, 2] / kp[idx, 2].sum()
            subsets = [rng.choice(len(idx), 4, replace=False, p=pr) for _ in range(n_random)]
        for keep in subsets:
            if np.linalg.svd(src[keep] - src[keep].mean(0), compute_uv=False)[-1] < 2.0:
                continue
            Hs, _ = cv2.findHomography(src[keep], dst[keep], 0)
            if Hs is not None:
                hyps.append(Hs)
    return hyps, used


def kp_agreement(H: np.ndarray, kp: np.ndarray, size, conf: float = 0.6, tol_frac: float = 0.02) -> float:
    """(agreeing + 1) / (confident + 1) over confident in-image keypoints; tolerance ~22 px at 1080p wide."""
    w, h = size
    m = np.isfinite(KP_COURT[:, 0]) & (kp[:, 2] >= conf) & (kp[:, 0] > 0) & (kp[:, 0] < w) & (kp[:, 1] > 0) & (kp[:, 1] < h)
    if not m.any():
        return 1.0
    err = np.linalg.norm(project(H, KP_COURT[m]) - kp[m, :2], axis=1)
    return float(((err < tol_frac * w).sum() + 1) / (m.sum() + 1))


def solve_frame(frame_bgr: np.ndarray, kp: np.ndarray | None, em: EdgeMap | None = None, init: list | None = None,
                fixed_C=None) -> FrameFit | None:
    """Best camera for one frame from keypoint hypotheses and/or provided initial params."""
    h, w = frame_bgr.shape[:2]
    size = (w, h)
    em = em or EdgeMap.from_frame(frame_bgr)
    cands = []
    kp_idx = np.zeros(0, int)
    if kp is not None:
        hyps, kp_idx = keypoint_hypotheses(kp, size)
        for H in hyps:
            p = cam.from_H(H / H[2, 2], size, KP_COURT[kp_idx])
            if fixed_C is not None and p is not None and np.all(np.isfinite(p)):
                from .video import fit_rotation_to_H
                p = fit_rotation_to_H(H / H[2, 2], p, size, fixed_C)
            if cam.plausible(p, size):
                cands.append(p)
    for p in init or []:
        if p is not None:
            cands.append(np.array(p, float))
    if not cands:
        return None
    # cheap screening by line support of the raw hypotheses, then full refinement of the best few
    scored = []
    for p in cands:
        f1, _ = frame_score(cam.to_H(p, size), em, tol=8.0)
        scored.append((f1, p))
    scored.sort(key=lambda t: -t[0])
    best = None
    seen = []
    for _, p in scored[:6]:
        if any(np.abs(p[:4] - s[:4]).max() < 0.01 for s in seen):
            continue
        seen.append(p)
        kpi = kp[kp_idx, :2] if len(kp_idx) else None
        kpw = (0.15 * KP_WEIGHT[kp_idx] * np.sqrt(kp[kp_idx, 2])) if len(kp_idx) else None
        pr, st = refine(p, em, size, kpi, KP_COURT[kp_idx] if len(kp_idx) else None, kpw, fixed_C=fixed_C)
        if not cam.plausible(pr, size):
            continue
        score, fs = frame_score(cam.to_H(pr, size), em, tol=3.0)
        st.update(fs)
        if best is None or score > best.score:
            best = FrameFit(pr, cam.to_H(pr, size), score, st)
    return best
