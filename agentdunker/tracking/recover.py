"""Pass 3: per-identity recovery over all detections (incl. low confidence).

Tracker-level association only starts tracks from confident boxes, and a merged two-person box often
wins the first association while the correct single-person box sits at conf 0.1-0.25. Here each
identity (chain) re-walks its gaps and ends step by step over the unused detections, and swaps
merged (too wide) boxes for an unused single-person box inside them.
"""
from __future__ import annotations

import logging

import numpy as np

from .identities import Chain
from .tracklets import Observations
from .video import warp_box

logger = logging.getLogger(__name__)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.atleast_2d(b)
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def _ioa(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    inner = np.atleast_2d(inner)
    x1, y1 = np.maximum(inner[:, 0], outer[0]), np.maximum(inner[:, 1], outer[1])
    x2, y2 = np.minimum(inner[:, 2], outer[2]), np.minimum(inner[:, 3], outer[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((inner[:, 2] - inner[:, 0]) * (inner[:, 3] - inner[:, 1]) + 1e-9)


class Recovery:
    def __init__(self, o: Observations, chains: list[Chain], iou_min: float = 0.35, max_miss: int = 2) -> None:
        self.o, self.chains, self.iou_min, self.max_miss = o, chains, iou_min, max_miss
        self.stab = [np.stack([warp_box(o.to_ref[f], b) for b in bs]) if len(bs) else np.zeros((0, 4))
                     for f, bs in zip(o.det_frames, o.boxes)]
        self.used: list[set[int]] = [set() for _ in o.det_frames]
        for ch in chains:
            for k, (i, _) in ch.obs.items():
                self.used[k].add(i)
        self.means = np.stack([ch.emb() for ch in chains]) if chains else np.zeros((0, 512))
        self.hard = [ch.jersey() for ch in chains]  # identity pass numbers (None without one)
        self.n_recovered = self.n_swapped = 0

    def _appearance_ok(self, ci: int, k: int, j: int) -> bool:
        """Group-level check only: OSNet cannot reliably tell teammates apart, but it does tell teams apart.

        Teammates are kept apart by detection exclusivity and the motion gate instead; in the identity
        pass a detection labelled with another number is never taken.
        """
        lab = self.o.det_jersey.get((k, j))
        if lab and self.hard[ci] and lab != self.hard[ci]:
            return False
        s = self.means @ self.o.embs[k][j]
        g = self.chains[ci].group
        own = [s[i] for i, ch in enumerate(self.chains) if ch.group == g]
        other = [s[i] for i, ch in enumerate(self.chains) if ch.group != g]
        return s[ci] >= 0.72 and np.mean(own) >= max([np.mean(other)] if other else [0.0]) - 0.01

    def _aspect(self, box: np.ndarray) -> float:
        return float((box[2] - box[0]) / max(1.0, box[3] - box[1]))

    def swap_merged(self, ci: int) -> None:
        """Replace boxes much wider or taller than the slot's local median (±1 s) by an unused box inside them."""
        ch = self.chains[ci]
        ks = sorted(ch.obs)
        wh = np.array([self.o.boxes[k][ch.obs[k][0]][2:] - self.o.boxes[k][ch.obs[k][0]][:2] for k in ks])
        win = max(3, int(round(self.o.info.fps / self.o.timing["stride"])))
        for n, k in enumerate(ks):
            ref = np.median(wh[max(0, n - win):n + win + 1], 0)
            if wh[n, 0] <= 1.35 * ref[0] and wh[n, 1] <= 1.3 * ref[1]:
                continue
            i = ch.obs[k][0]
            big = self.o.boxes[k][i]
            cand = []
            for j in range(len(self.o.boxes[k])):
                if j in self.used[k]:
                    continue
                cwh = self.o.boxes[k][j][2:] - self.o.boxes[k][j][:2]
                if (cwh <= 1.25 * ref).all() and _ioa(self.o.boxes[k][j], big)[0] >= 0.7 and self._appearance_ok(ci, k, j):
                    cand.append(j)
            if not cand:
                continue
            j = max(cand, key=lambda j: _iou(big, self.o.boxes[k][j])[0])
            self.used[k].discard(i)
            self.used[k].add(j)
            ch.obs[k] = (j, float(self.o.confs[k][j]))
            self.n_swapped += 1

    def _walk(self, ci: int, start: int, step: int, stop: int) -> None:
        ch = self.chains[ci]
        cur = self.stab[start][ch.obs[start][0]]
        prev_k = start - step
        vel = np.zeros(4)
        if prev_k in ch.obs:
            vel = 0.5 * (cur - self.stab[prev_k][ch.obs[prev_k][0]])
        misses, k, last = 0, start + step, start
        shot = self.o.shot_of(self.o.det_frames[start])
        while (0 <= k < len(self.o.det_frames) and k != stop and k not in ch.obs and misses <= self.max_miss
               and self.o.shot_of(self.o.det_frames[k]) == shot):
            pred = cur + vel * (k - last) * step
            free = [j for j in range(len(self.stab[k])) if j not in self.used[k]]
            best, bi = None, self.iou_min
            if free:
                ious = _iou(pred, self.stab[k][free])
                for j, v in sorted(zip(free, ious), key=lambda x: -x[1]):
                    if v < bi:
                        break
                    if self._appearance_ok(ci, k, j):
                        best = j
                        break
            if best is None:
                misses += 1
            else:
                box = self.stab[k][best]
                vel = 0.5 * vel + 0.5 * (box - cur) / max(1, abs(k - last))
                cur, last, misses = box, k, 0
                ch.obs[k] = (best, float(self.o.confs[k][best]))
                self.used[k].add(best)
                self.n_recovered += 1
            k += step

    def _neighbour_pred(self, ch, k: int) -> np.ndarray | None:
        """Stabilised box at step k interpolated from the slot's nearest observations on both sides (same shot)."""
        shot = self.o.shot_of(self.o.det_frames[k])
        ks = [j for j in sorted(ch.obs) if j != k and self.o.shot_of(self.o.det_frames[j]) == shot]
        lo = [j for j in ks if j < k][-2:]
        hi = [j for j in ks if j > k][:2]
        if not lo or not hi or hi[0] - lo[-1] > 12:
            return None
        a, b = lo[-1], hi[0]
        A, B = self.stab[a][ch.obs[a][0]], self.stab[b][ch.obs[b][0]]
        return A + (B - A) * (k - a) / (b - a)

    def refine(self, ci: int) -> None:
        """Swap a slot box for an unused detection that matches the slot's own interpolation clearly better."""
        ch = self.chains[ci]
        for k in sorted(ch.obs):
            pred = self._neighbour_pred(ch, k)
            if pred is None:
                continue
            i = ch.obs[k][0]
            cur = _iou(pred, self.stab[k][i])[0]
            free = [j for j in range(len(self.stab[k])) if j not in self.used[k]]
            if not free:
                continue
            ious = _iou(pred, self.stab[k][free])
            j = int(np.argmax(ious))
            if ious[j] >= 0.5 and ious[j] > cur + 0.15 and self._appearance_ok(ci, k, free[j]):
                self.used[k].discard(i)
                self.used[k].add(free[j])
                ch.obs[k] = (free[j], float(self.o.confs[k][free[j]]))
                self.n_swapped += 1

    def fill(self, ci: int) -> None:
        """Inside gaps, take unused detections that match the two-sided interpolation (IoU >= 0.5)."""
        ch = self.chains[ci]
        ks = sorted(ch.obs)
        for a, b in zip(ks[:-1], ks[1:]):
            if b - a <= 1 or self.o.shot_of(self.o.det_frames[a]) != self.o.shot_of(self.o.det_frames[b]):
                continue
            A, B = self.stab[a][ch.obs[a][0]], self.stab[b][ch.obs[b][0]]
            for k in range(a + 1, b):
                free = [j for j in range(len(self.stab[k])) if j not in self.used[k]]
                if not free:
                    continue
                ious = _iou(A + (B - A) * (k - a) / (b - a), self.stab[k][free])
                j = int(np.argmax(ious))
                if ious[j] >= 0.5 and self._appearance_ok(ci, k, free[j]):
                    ch.obs[k] = (free[j], float(self.o.confs[k][free[j]]))
                    self.used[k].add(free[j])
                    self.n_recovered += 1

    def run(self) -> None:
        for ci in range(len(self.chains)):
            self.swap_merged(ci)
        K = len(self.o.det_frames)
        for ci in sorted(range(len(self.chains)), key=lambda i: -len(self.chains[i].obs)):
            ch = self.chains[ci]
            ks = sorted(ch.obs)
            for a, b in zip(ks[:-1], ks[1:]):
                if b - a > 1:
                    self._walk(ci, a, 1, b)
                    self._walk(ci, b, -1, a)
            self._walk(ci, ks[-1], 1, K)
            self._walk(ci, ks[0], -1, -1)
        for ci in range(len(self.chains)):
            self.fill(ci)
            self.refine(ci)
        logger.info("recovery: %d detections recovered, %d boxes swapped", self.n_recovered, self.n_swapped)
