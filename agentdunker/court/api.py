"""Contract I/O (court.json) and image <-> court mapping."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import camera as cam

OK_SCORE = 0.15        # minimum final line support for ok=true
CONF_FULL = 0.40       # line support that maps to conf=1


def to_contract(clip, result: dict, video_sha256: str = "") -> dict:
    size = clip.size
    frames = []
    shot_of = {}
    for k, (s0, s1) in enumerate(result["shots"]):
        for i in range(s0, s1):
            shot_of[i] = k
    for i, st in enumerate(result["states"]):
        rec = {"f": i, "t": round(float(clip.pts[i]), 4), "H_img2court": None, "conf": 0.0,
               "reproj_px": None, "ok": False, "src": st.src, "shot": shot_of.get(i, 0), "camera": None}
        if st.p is not None:
            Hc = cam.to_H(st.p, size)
            Hi = np.linalg.inv(Hc)
            Hi = Hi / np.linalg.norm(Hi) * (1.0 if Hi[2, 2] >= 0 else -1.0)   # scale-free; H[2,2] can be ~0
            conf = min(1.0, st.score / CONF_FULL) * (0.7 if st.src == "interpolated" else 1.0)
            rec.update(H_img2court=np.round(Hi, 12).tolist(), conf=round(float(conf), 3),
                       reproj_px=None if not np.isfinite(st.reproj) else round(st.reproj * 1080.0 / size[1], 2),
                       ok=bool(st.score >= OK_SCORE),
                       camera={"f_px": round(float(np.exp(st.p[3])), 1),
                               "C_ft": np.round(st.p[4:7], 2).tolist()})
        frames.append(rec)
    return {"video": {"path": str(clip.path), "sha256": video_sha256, "fps": clip.fps,
                      "width": size[0], "height": size[1], "n_frames": len(frames)},
            "court": {"units": "ft", "origin": "centre court", "x": "[-47, 47] along length",
                      "y": "[-25, 25], +y = near (camera-side) sideline"},
            "shots": [list(s) for s in result["shots"]],
            "frames": frames}


class CourtCalibration:
    """Loaded court.json with per-frame image<->court mapping."""

    def __init__(self, data: dict):
        self.data = data
        self.frames = {int(fr["f"]): fr for fr in data["frames"]}

    @classmethod
    def load(cls, path: str | Path) -> "CourtCalibration":
        return cls(json.loads(Path(path).read_text()))

    def H(self, f: int) -> np.ndarray | None:
        fr = self.frames.get(int(f))
        return None if fr is None or fr["H_img2court"] is None else np.asarray(fr["H_img2court"], float)

    def ok(self, f: int) -> bool:
        fr = self.frames.get(int(f))
        return bool(fr and fr["ok"])

    def img_to_court(self, f: int, xy) -> np.ndarray:
        """Image pixel(s) (x, y) at frame f -> court feet; NaN where frame f has no homography."""
        xy = np.asarray(xy, float)
        pts = xy.reshape(-1, 2)
        H = self.H(f)
        if H is None:
            out = np.full_like(pts, np.nan)
        else:
            q = np.c_[pts, np.ones(len(pts))] @ H.T
            out = q[:, :2] / q[:, 2:3]
        return out.reshape(xy.shape)

    def court_to_img(self, f: int, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        pts = xy.reshape(-1, 2)
        H = self.H(f)
        if H is None:
            return np.full_like(xy, np.nan)
        q = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(H).T
        return (q[:, :2] / q[:, 2:3]).reshape(xy.shape)


def img_to_court(court: CourtCalibration | str | Path, f: int, xy) -> np.ndarray:
    """Convenience: img_to_court('court.json', f, [[x, y], ...])."""
    if not isinstance(court, CourtCalibration):
        court = CourtCalibration.load(court)
    return court.img_to_court(f, xy)
