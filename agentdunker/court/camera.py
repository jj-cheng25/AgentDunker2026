"""Pinhole broadcast camera (above the floor => C_z < 0) looking at the court plane z=0 (principal point at image centre, square pixels).

H_court2img = K [r1 r2 -R C].  Parameter vector: [rvec(3), log f, C(3)].
"""
from __future__ import annotations

import cv2
import numpy as np


def K_of(f: float, size) -> np.ndarray:
    w, h = size
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def to_H(p: np.ndarray, size) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(p[:3], float))
    f = float(np.exp(p[3]))
    C = np.asarray(p[4:7], float)
    H = K_of(f, size) @ np.c_[R[:, 0], R[:, 1], -R @ C]
    return H / H[2, 2]


def from_H(H: np.ndarray, size, ref_pts: np.ndarray | None = None, fit: bool = True) -> np.ndarray | None:
    """Camera parameters closest to a court->image homography (None if no valid camera).

    Closed-form decomposition (principal point at centre, square pixels), then -- because a noisy H has one more
    DOF than this camera -- a least-squares fit of all 7 parameters to H's projections of the court in view.
    """
    p = _decompose(H, size, ref_pts)
    if p is None or not fit:
        return p
    return fit_to_H(H, p, size)


def fit_to_H(H: np.ndarray, p0: np.ndarray, size, free=slice(0, 7)) -> np.ndarray:
    from scipy.optimize import least_squares
    w, h = size
    gx, gy = np.meshgrid(np.linspace(-47, 47, 17), np.linspace(-25, 25, 9))
    g = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)]
    q = g @ H.T
    ok = np.abs(q[:, 2]) > 1e-9
    q = q[:, :2] / q[:, 2:3]
    vis = ok & (q[:, 0] > -0.25 * w) & (q[:, 0] < 1.25 * w) & (q[:, 1] > -0.25 * h) & (q[:, 1] < 1.25 * h)
    if vis.sum() < 6:
        return p0
    g, q = g[vis], q[vis]
    p = np.array(p0, float)
    idx = np.arange(7)[free]

    def r(x):
        pp = p.copy()
        pp[idx] = x
        z = g @ to_H(pp, size).T
        return (z[:, :2] / z[:, 2:3] - q).ravel()

    p[idx] = least_squares(r, p[idx], loss="soft_l1", f_scale=10.0, max_nfev=60,
                           x_scale=np.r_[0.01, 0.01, 0.01, 0.01, 1, 1, 1][idx]).x
    return p


def _decompose(H: np.ndarray, size, ref_pts: np.ndarray | None = None) -> np.ndarray | None:
    w, h = size
    M = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]]) @ H
    m1, m2 = M[:, 0], M[:, 1]
    cands = []
    d = m1[2] * m2[2]
    if abs(d) > 1e-12:
        cands.append(-(m1[0] * m2[0] + m1[1] * m2[1]) / d)
    d = m2[2] ** 2 - m1[2] ** 2
    if abs(d) > 1e-12:
        cands.append((m1[0] ** 2 + m1[1] ** 2 - m2[0] ** 2 - m2[1] ** 2) / d)
    f2 = [c for c in cands if np.isfinite(c) and (0.2 * w) ** 2 < c < (30 * w) ** 2]
    f = float(np.sqrt(np.median(f2))) if f2 else float(w)
    A = np.diag([1 / f, 1 / f, 1.0]) @ M
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    if ref_pts is None:
        ref_pts = np.zeros((1, 2))
    for sgn in (1.0, -1.0):
        r1, r2, t = sgn * lam * A[:, 0], sgn * lam * A[:, 1], sgn * lam * A[:, 2]
        Rm = np.c_[r1, r2, np.cross(r1, r2)]
        U, _, Vt = np.linalg.svd(Rm)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            continue
        z = (np.c_[ref_pts, np.zeros(len(ref_pts))] @ Rm.T + t)[:, 2]
        if np.median(z) > 0:
            C = -Rm.T @ t
            rvec, _ = cv2.Rodrigues(Rm)
            return np.r_[rvec.ravel(), np.log(f), C]
    return None


def roll_of(p: np.ndarray) -> float:
    """Sine of camera roll w.r.t. the floor: world-z component of the image x axis."""
    R, _ = cv2.Rodrigues(np.asarray(p[:3], float))
    return float(R[0, 2])


def plausible(p: np.ndarray | None, size) -> bool:
    """Broadcast-camera sanity: near-side camera well above the floor, small roll, sensible focal length.

    Behind-the-basket or opposite-side angles are rejected on purpose (they get ok=false) because accepting them
    lets mirrored or degenerate fits through on normal broadcast frames.
    """
    if p is None or not np.all(np.isfinite(p)):
        return False
    w, _ = size
    f = np.exp(p[3])
    C = p[4:7]
    height = -C[2]           # court frame is right-handed with +y toward the camera, so +z points into the floor
    dist = np.hypot(C[0], C[1])
    return bool(0.25 * w < f < 25 * w and 10.0 < height < 120.0 and C[1] > 20.0 and dist < 400.0
                and abs(roll_of(p)) < 0.2)


def look_at(C, target, f: float) -> np.ndarray:
    """Camera params for a level camera at C (court ft, z<0 is up) looking at target."""
    C, target = np.asarray(C, float), np.asarray(target, float)
    zc = target - C
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, [0.0, 0.0, 1.0])   # image x axis stays parallel to the floor
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    if yc[2] < 0:                          # image y must point toward the floor (+z)
        R = np.stack([-xc, -yc, zc])
    rvec, _ = cv2.Rodrigues(R)
    return np.r_[rvec.ravel(), np.log(f), C]
