"""Pass 2: tracklets -> roles/teams -> fixed slots (5 per team, 3 referees) -> per-frame tracks."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans

from .models import RoleClassifier
from .tracklets import Observations
from .video import warp_box

logger = logging.getLogger(__name__)

SLOTS = {"A": 5, "B": 5, "REF": 3}


@dataclass
class Tracklet:
    tid: int
    ks: np.ndarray  # detection-step indices
    idx: np.ndarray  # detection index within its step
    boxes: np.ndarray  # image xyxy
    stab: np.ndarray  # camera-stabilised xyxy (frame-0 plane)
    conf: np.ndarray
    emb: np.ndarray  # mean OSNet embedding (normalised)
    color: np.ndarray  # median colour descriptor
    role: np.ndarray | None = None  # P(player, referee, other)
    group: str | None = None  # "A" | "B" | "REF" | None
    group_conf: float = 0.0
    ambiguous: bool = False  # merged-box stretch around an ID switch: never linked
    vlm: str | None = None  # VLM label: team1 | team2 | referee | other
    parent_vlm: str | None = None  # VLM label of the tracklet this piece was split from
    number: str | None = None  # jersey number read by the VLM on this tracklet's own crops
    split_from: int | None = None  # pieces of one split tracklet
    split_kind: str | None = None  # "mixed": VLM says different people (never relinked); "team": appearance flip
    team_margin: float = 0.0  # OSNet sim to team A centre minus team B centre
    jersey: str | None = None  # identity pass only: "<team>#<number>" from confident M3 reads (hard link)

    @property
    def weight(self) -> float:
        return float(self.conf.sum())


def hard_label(o: Observations, ks: np.ndarray, idx: np.ndarray) -> str | None:
    """Identity-pass number of a set of detections if >= 75 % of its labelled detections agree."""
    c = Counter(o.det_jersey[(int(k), int(i))] for k, i in zip(ks, idx) if (int(k), int(i)) in o.det_jersey)
    if not c:
        return None
    lab, n = c.most_common(1)[0]
    return lab if n >= 0.75 * sum(c.values()) else None


def build_tracklets(o: Observations) -> list[Tracklet]:
    rows: dict[int, list[tuple[int, int]]] = {}
    for k, m in enumerate(o.track_of):
        for i, tid in m.items():
            rows.setdefault(tid, []).append((k, i))
    out = []
    for tid, kv in rows.items():
        ks = np.array([k for k, _ in kv])
        idx = np.array([i for _, i in kv])
        boxes = np.stack([o.boxes[k][i] for k, i in kv])
        stab = np.stack([warp_box(o.to_ref[o.det_frames[k]], b) for (k, _), b in zip(kv, boxes)])
        conf = np.array([o.confs[k][i] for k, i in kv])
        emb = np.average(np.stack([o.embs[k][i] for k, i in kv]), axis=0, weights=conf)
        color = np.median(np.stack([o.colors[k][i] for k, i in kv]), 0)
        out.append(Tracklet(tid, ks, idx, boxes, stab, conf, emb / (np.linalg.norm(emb) + 1e-9), color))
    return out


def classify_roles(o: Observations, tls: list[Tracklet], clf: RoleClassifier, budget: int = 240, per: int = 3) -> None:
    """CLIP role probabilities per tracklet from <= ``per`` crops; per-crop results are cached on ``o``."""
    plan: dict[int, list[tuple[int, int]]] = {}
    todo: list[tuple[int, int]] = []
    for t in sorted(tls, key=lambda t: -t.weight):
        keys = [(int(k), int(i)) for k, i in zip(t.ks, t.idx) if (k, i) in o.crops]
        if not keys:
            continue
        cached = [k for k in keys if k in o.role_cache]
        if len(cached) >= min(per, len(keys)):
            plan[t.tid] = cached[:per]
            continue
        if len(todo) >= budget:
            if cached:
                plan[t.tid] = cached
            continue
        pick = [keys[j] for j in np.unique(np.linspace(0, len(keys) - 1, min(per, len(keys))).round().astype(int))]
        plan[t.tid] = pick
        todo.extend(k for k in pick if k not in o.role_cache)
    if todo:
        probs = clf([cv2.imdecode(np.frombuffer(o.crops[k], np.uint8), cv2.IMREAD_COLOR) for k in todo])
        o.role_cache.update(zip(todo, probs))
    for t in tls:
        if t.tid in plan:
            t.role = np.mean([o.role_cache[k] for k in plan[t.tid]], 0)


def inherit_vlm(o: Observations, tls: list[Tracklet]) -> None:
    """Tracklet VLM label / jersey number = those of the VLM-judged crops it contains (None if none or conflicting)."""
    for t in tls:
        keys = [(int(k), int(i)) for k, i in zip(t.ks, t.idx)]
        labs = {o.det_vlm[k] for k in keys if k in o.det_vlm}
        nums = {o.det_number[k] for k in keys if k in getattr(o, "det_number", {})}
        t.vlm = labs.pop() if len(labs) == 1 else None
        t.number = nums.pop() if len(nums) == 1 else None


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def select_tiles(o: Observations, tls: list[Tracklet], max_tiles: int = 40) -> list[tuple[Tracklet, list]]:
    """The strongest, least crowd-like tracklets with two judged crops each (at 1/3 and 2/3 of the track)."""

    def rank(t: Tracklet) -> float:
        other = t.role[2] if t.role is not None else 0.5
        return t.weight * (1.0 - 0.7 * other)

    cands = []
    for t in sorted([t for t in tls if len(t.ks) >= 5], key=rank, reverse=True):
        keys = [(k, i) for k, i in zip(t.ks, t.idx) if (k, i) in o.crops]
        if len(keys) >= 2:
            cands.append((t, [keys[len(keys) // 3], keys[(2 * len(keys)) // 3]]))
        if len(cands) >= max_tiles:
            break
    return cands


def adjudicate_vlm(o: Observations, tls: list[Tracklet], cands: list[tuple[Tracklet, list]]) -> dict | None:
    """One VLM call labelling ``cands`` (see select_tiles); labels/numbers attach to the judged crops.

    Thread-safe against a concurrent classify_roles: it only writes ``o.det_vlm``, ``o.det_number``,
    ``t.vlm`` and ``t.number``.
    """
    from . import vlm

    if not cands:
        return None
    tiles = [[cv2.imdecode(np.frombuffer(o.crops[k], np.uint8), cv2.IMREAD_COLOR) for k in ks] for _, ks in cands]
    res = vlm.label_tiles(vlm.contact_sheet(tiles), len(tiles))
    if res:
        for n, (t, ks) in enumerate(cands, 1):
            lab = res["labels"].get(n)
            if lab:
                o.det_vlm.update({(int(k), int(i)): lab for k, i in ks})
                if n in res["numbers"] and lab != "mixed":
                    o.det_number.update({(int(k), int(i)): res["numbers"][n] for k, i in ks})
        inherit_vlm(o, tls)
        res["tiles"] = {n: int(t.tid) for n, (t, _) in enumerate(cands, 1)}
    return res


def _center(ts: list[Tracklet]) -> np.ndarray:
    c = np.average(np.stack([t.emb for t in ts]), 0, weights=[t.weight for t in ts])
    return c / np.linalg.norm(c)


def assign_groups(tls: list[Tracklet]) -> dict[str, np.ndarray]:
    """Team / referee centroids in OSNet space, then a group per tracklet.

    Centroids come from VLM labels when available (team1/team2/referee), otherwise from 2-means over
    CLIP-player tracklets and CLIP-referee tracklets. VLM-labelled tracklets take their label; the rest
    get the group maximising 0.5*CLIP-role + 0.5*appearance-similarity, or None (bench, crowd, coaches)
    when no group scores at least 0.4. Team A = the brighter jersey.
    """
    by_vlm = {lab: [t for t in tls if t.vlm == lab and not t.ambiguous] for lab in ("team1", "team2", "referee")}
    teams: list[list[Tracklet]] = []
    if by_vlm["team1"] and by_vlm["team2"]:
        teams = [by_vlm["team1"], by_vlm["team2"]]
    else:
        players = [t for t in tls if t.role is not None and t.role[0] >= 0.5 and len(t.ks) >= 3]
        if len(players) >= 2:
            lab = KMeans(2, n_init=10, random_state=0).fit(
                np.stack([t.emb for t in players]), sample_weight=[t.weight for t in players]).labels_
            teams = [[t for t, l in zip(players, lab) if l == j] for j in (0, 1)]
    centers: dict[str, np.ndarray] = {}
    vlm_map: dict[str, str] = {"referee": "REF"}
    bright = np.array([0, 1])
    if teams and all(teams):
        bright = np.argsort([-np.mean([t.color[0] for t in ts]) for ts in teams])
        for name, j in zip("AB", bright):
            centers[name] = _center(teams[j])
            vlm_map[("team1", "team2")[j]] = name
    refs = by_vlm["referee"] or [t for t in tls if t.role is not None and t.role[1] >= 0.5 and len(t.ks) >= 3]
    if refs:
        centers["REF"] = _center(refs)
    # per-clip calibration of "how similar is a member to its own group centre"
    ramps = {"A": (0.80, 0.95), "B": (0.80, 0.95), "REF": (0.75, 0.90)}
    for g, ts in (("A", teams[int(bright[0])] if centers.get("A") is not None and teams else []),
                  ("B", teams[int(bright[1])] if centers.get("B") is not None and teams else []), ("REF", refs)):
        if len(ts) >= 3:
            sims = np.array([float(centers[g] @ t.emb) for t in ts])
            ramps[g] = (float(np.percentile(sims, 10)) - 0.05, float(np.median(sims)))
    for t in tls:
        if "A" in centers and "B" in centers:
            t.team_margin = float(centers["A"] @ t.emb - centers["B"] @ t.emb)
        if t.ambiguous:
            t.group, t.group_conf = None, 0.0
            continue
        if t.vlm not in (None, "mixed"):
            t.group = vlm_map.get(t.vlm)
            t.group_conf = 0.95 if t.group else 0.0
            continue
        if t.vlm == "mixed" or t.parent_vlm in ("team1", "team2", "mixed"):
            if "A" in centers and "B" in centers:  # known player: only the team is in question
                sa, sb = float(centers["A"] @ t.emb), float(centers["B"] @ t.emb)
                t.group, t.group_conf = ("A" if sa > sb else "B"), 0.6 + min(0.35, 5 * abs(sa - sb))
                continue
        if t.parent_vlm == "referee" and "REF" in centers:
            t.group, t.group_conf = "REF", 0.8
            continue
        role = t.role if t.role is not None else np.array([0.3, 0.1, 0.6])
        scores = {}
        for g, c in centers.items():
            sim = float(c @ t.emb)
            scores[g] = 0.5 * (role[1] if g == "REF" else role[0]) + 0.5 * _ramp(sim, *ramps[g])
        team_keys = [g for g in ("A", "B") if g in scores]
        if len(team_keys) == 2:  # a tracklet competes for its closer team only
            scores.pop(min(team_keys, key=lambda g: float(centers[g] @ t.emb)))
        if scores:
            g = max(scores, key=scores.get)
            t.group, t.group_conf = (g if scores[g] >= 0.4 else None), scores[g]
    return centers


def _subset(o: Observations, t: Tracklet, m: np.ndarray, tid: int) -> Tracklet:
    embs = np.stack([o.embs[k][i] for k, i in zip(t.ks[m], t.idx[m])])
    emb = np.average(embs, 0, weights=t.conf[m])
    color = np.median(np.stack([o.colors[k][i] for k, i in zip(t.ks[m], t.idx[m])]), 0)
    keys = [(int(k), int(i)) for k, i in zip(t.ks[m], t.idx[m])]
    nums = {o.det_number[k] for k in keys if k in o.det_number}
    labs = {o.det_vlm[k] for k in keys if k in o.det_vlm}  # the piece holding the judged crops keeps the label
    return Tracklet(tid, t.ks[m], t.idx[m], t.boxes[m], t.stab[m], t.conf[m], emb / np.linalg.norm(emb), color,
                    role=t.role, vlm=labs.pop() if len(labs) == 1 else None, parent_vlm=t.vlm or t.parent_vlm,
                    number=nums.pop() if len(nums) == 1 else None, split_from=t.split_from,
                    split_kind=t.split_kind, jersey=hard_label(o, t.ks[m], t.idx[m]))


def _sign_segments(sgn: np.ndarray, min_run: int) -> list[tuple[int, int, int]]:
    """Runs of equal non-zero sign; runs shorter than ``min_run`` are absorbed by their neighbours."""
    nz = np.flatnonzero(sgn)
    if not len(nz):
        return [(0, len(sgn), 0)]
    filled = sgn[nz[np.clip(np.searchsorted(nz, np.arange(len(sgn))), 0, len(nz) - 1)]]
    segs = []
    for j, v in enumerate(filled):
        if segs and segs[-1][2] == v:
            segs[-1][1] = j + 1
        else:
            segs.append([j, j + 1, int(v)])
    while len(segs) > 1:
        i = min(range(len(segs)), key=lambda i: segs[i][1] - segs[i][0])
        if segs[i][1] - segs[i][0] >= min_run:
            break
        nb = i - 1 if i == len(segs) - 1 or (i > 0 and segs[i - 1][1] - segs[i - 1][0] >= segs[i + 1][1] - segs[i + 1][0]) else i + 1
        lo, hi = min(segs[i][0], segs[nb][0]), max(segs[i][1], segs[nb][1])
        segs[min(i, nb)] = [lo, hi, segs[nb][2]]
        del segs[max(i, nb)]
        k = 1  # re-merge equal neighbours
        while k < len(segs):
            if segs[k][2] == segs[k - 1][2]:
                segs[k - 1][1] = segs[k][1]
                del segs[k]
            else:
                k += 1
    return [tuple(x) for x in segs]


def split_team_switches(o: Observations, tls: list[Tracklet], centers: dict[str, np.ndarray],
                        min_run_s: float = 1.0) -> list[Tracklet]:
    """Cut tracklets whose per-detection appearance holds the other team for >= ``min_run_s``.

    Per-detection team evidence is only ~95 % right (occluders really do fill the box), so shorter
    flips are absorbed. The merged / neutral stretch just before each cut becomes an ambiguous piece
    that is never linked (filled as observed=false).
    """
    if "A" not in centers or "B" not in centers:
        return tls
    min_run = max(4, int(round(min_run_s * o.info.fps / o.timing["stride"])))
    out = []
    for t in tls:
        player_like = t.group in ("A", "B") or t.vlm == "mixed" or (t.role is not None and t.role[0] >= 0.5)
        if len(t.ks) < 2 * min_run or not player_like:
            out.append(t)
            continue
        E = np.stack([o.embs[k][i] for k, i in zip(t.ks, t.idx)])
        d = median_filter(E @ centers["A"] - E @ centers["B"], size=5, mode="nearest")
        sgn = np.where(d > 0.03, 1, np.where(d < -0.03, -1, 0))
        segs = _sign_segments(sgn, min_run)
        if len(segs) == 1:
            out.append(t)
            continue
        wh = t.boxes[:, 2:] - t.boxes[:, :2]
        aspect = wh[:, 0] / np.maximum(wh[:, 1], 1)
        amb = (sgn == 0) | (aspect > 1.3 * np.median(aspect))
        bounds, amb_starts = [0], set()
        for lo, _, _ in segs[1:]:
            z = lo
            while z > bounds[-1] + 1 and amb[z - 1]:
                z -= 1
            if z < lo:
                bounds.append(z)
                amb_starts.add(z)
            bounds.append(lo)
        bounds.append(len(t.ks))
        for n, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            m = np.zeros(len(t.ks), bool)
            m[a:b] = True
            piece = _subset(o, t, m, t.tid * 1000 + n)
            piece.ambiguous = a in amb_starts
            piece.split_from, piece.split_kind = t.tid, "team"
            out.append(piece)
        logger.info("split tracklet %d at %s (team flip)", t.tid, [int(t.ks[lo]) for lo, _, _ in segs[1:]])
    return out


def _snap_to_handover(t: Tracklet, tls: list[Tracklet], cut: int, lo: int, hi: int, win: int) -> int:
    """Move a cut to where a neighbouring tracklet ends or starts next to t (the tracker swapped targets there).

    Only events within ``win`` steps of the appearance cut and inside [lo, hi] are considered; the
    neighbour's box must touch t's box at that moment.
    """
    def touches(a: np.ndarray, b: np.ndarray) -> bool:
        return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

    ks = t.ks
    best = None
    for u in tls:
        if u is t:
            continue
        for k_event, k_box, ubox in ((int(u.ks[-1]) + 1, int(u.ks[-1]), u.stab[-1]), (int(u.ks[0]), int(u.ks[0]), u.stab[0])):
            j = int(np.searchsorted(ks, k_event))
            jb = int(np.searchsorted(ks, k_box))
            if not (lo <= j <= hi) or abs(j - cut) > win or jb >= len(ks) or ks[jb] != k_box:
                continue
            if touches(t.stab[jb], ubox) and (best is None or abs(j - cut) < abs(best - cut)):
                best = j
    return cut if best is None else best


def split_mixed(o: Observations, tls: list[Tracklet], min_side_s: float = 0.3) -> list[Tracklet]:
    """Cut tracklets whose two VLM-judged crops show different people (label "mixed").

    The change point is searched between the two judged crops: the step with the largest OSNet
    distance between the mean embeddings of the windows just before and just after it. This handles
    teammate swaps that the team-flip splitter cannot see.
    """
    w = max(3, int(round(min_side_s * o.info.fps / o.timing["stride"])))
    out = []
    for t in tls:
        pos = [j for j, (k, i) in enumerate(zip(t.ks, t.idx)) if o.det_vlm.get((int(k), int(i))) == "mixed"]
        if t.vlm != "mixed" or len(pos) < 2 or pos[-1] - pos[0] < 2:
            out.append(t)
            continue
        E = np.stack([o.embs[k][i] for k, i in zip(t.ks, t.idx)])
        best, cut = -1.0, None
        for c in range(pos[0] + 1, pos[-1] + 1):
            a, b = E[max(0, c - w):c].mean(0), E[c:c + w].mean(0)
            d = 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            if d > best:
                best, cut = d, c
        cut = _snap_to_handover(t, tls, cut, pos[0] + 1, pos[-1], int(round(1.0 * o.info.fps / o.timing["stride"])))
        for n, (a, b) in enumerate(((0, cut), (cut, len(t.ks)))):
            m = np.zeros(len(t.ks), bool)
            m[a:b] = True
            piece = _subset(o, t, m, t.tid * 1000 + 500 + n)
            piece.parent_vlm, piece.split_from, piece.split_kind = "mixed", t.tid, "mixed"
            out.append(piece)
        logger.info("split mixed tracklet %d at %d (appearance change %.3f)", t.tid, int(t.ks[cut]), best)
    return out


@dataclass
class Chain:
    group: str
    members: list[Tracklet] = field(default_factory=list)
    obs: dict[int, tuple[int, float]] = field(default_factory=dict)  # det step -> (det index, conf)
    shot_links: list[float] = field(default_factory=list)  # appearance similarity of each cross-cut join

    def finalize(self) -> None:
        """Collect member detections; on hand-over overlaps keep the stronger one."""
        self.members.sort(key=lambda m: m.ks[0])
        self.obs = {}
        for m in self.members:
            for k, i, s in zip(m.ks, m.idx, m.conf):
                if int(k) not in self.obs or self.obs[int(k)][1] < s:
                    self.obs[int(k)] = (int(i), float(s))

    def ks(self) -> np.ndarray:
        return np.concatenate([m.ks for m in self.members]) if self.members else np.zeros(0, int)

    def number(self) -> str | None:
        c = Counter(m.number for m in self.members if m.number)
        return c.most_common(1)[0][0] if c else None

    def jersey(self) -> str | None:
        """Identity-pass hard number (weighted majority of labelled members)."""
        c = Counter()
        for m in self.members:
            if m.jersey:
                c[m.jersey] += m.weight
        return c.most_common(1)[0][0] if c else None

    def emb(self) -> np.ndarray:
        e = sum(m.emb * m.weight for m in self.members)
        return e / (np.linalg.norm(e) + 1e-9)


def _foot(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, b[3]])


def _duplicates(a: Tracklet, b: Tracklet) -> bool:
    """Two time-overlapping tracklets can only be one person if their boxes coincide while overlapping."""
    shared, ia, ib = np.intersect1d(a.ks, b.ks, return_indices=True)
    if not len(shared):
        return False
    A, B = a.stab[ia], b.stab[ib]
    x1, y1 = np.maximum(A[:, 0], B[:, 0]), np.maximum(A[:, 1], B[:, 1])
    x2, y2 = np.minimum(A[:, 2], B[:, 2]), np.minimum(A[:, 3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return float(np.median(inter / (area(A) + area(B) - inter + 1e-9))) >= 0.5


def pair_cost(a: Tracklet, b: Tracklet, det_dt: float, max_overlap: int = 3) -> float:
    """Cost of b continuing a (a ends first): stabilised foot-motion feasibility + appearance; inf if impossible.

    Equal VLM-read jersey numbers (~90 % right with gpt-6-astra) relax the gates: duplicate tracks of one
    person may overlap for up to 0.5 s and the motion gate doubles. Different numbers cost +1 (soft).
    """
    if a.jersey and b.jersey and a.jersey != b.jersey:
        return np.inf  # identity pass: different confident numbers never link
    hard = bool(a.jersey and a.jersey == b.jersey)
    same = hard or bool(a.number and a.number == b.number)
    gap = int(b.ks[0]) - int(a.ks[-1])
    if b.ks[-1] <= a.ks[-1] or b.ks[0] <= a.ks[0]:
        return np.inf
    if a.split_from is not None and a.split_from == b.split_from:
        # siblings: VLM-mixed pieces are different people; team-flip pieces only rejoin if their
        # appearance agrees on the team (the flip was an occlusion after all)
        if a.split_kind == "mixed" or b.split_kind == "mixed" or np.sign(a.team_margin) != np.sign(b.team_margin):
            return np.inf
    overlap_ok = max(max_overlap, int(round(0.5 / det_dt))) if same else max_overlap
    if gap < -overlap_ok and not _duplicates(a, b):
        return np.inf
    dt = max(1, abs(gap)) * det_dt
    h = max(10.0, (a.stab[-1, 3] - a.stab[-1, 1] + b.stab[0, 3] - b.stab[0, 1]) / 2)
    r = np.linalg.norm(_foot(a.stab[-1]) - _foot(b.stab[0])) / (h * (0.35 + 2.5 * dt))
    if r > (4.0 if hard else 2.0 if same else 1.0):
        return np.inf
    if hard:
        return 0.1 * r - 0.5  # equal confident numbers: always worth linking
    if a.number and b.number:
        return 0.5 * r if same else r + 1.0
    return r + 2.0 * float(1.0 - a.emb @ b.emb)


def _chain_cover(ts: list[Tracklet], K: int, det_dt: float, max_cost: float) -> list[list[Tracklet]]:
    """At most K disjoint chains maximising sum(reward) - sum(link cost): min-cost flow over a DAG."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_node("S", demand=-K)
    G.add_node("T", demand=K)
    G.add_edge("S", "T", capacity=K, weight=0)
    for i, t in enumerate(ts):
        G.add_edge(("in", i), ("out", i), capacity=1, weight=-int(1000 * t.weight * t.group_conf))
        G.add_edge("S", ("in", i), capacity=1, weight=0)
        G.add_edge(("out", i), "T", capacity=1, weight=0)
    for i, a in enumerate(ts):
        for j, b in enumerate(ts):
            if i != j:
                c = pair_cost(a, b, det_dt)
                if c <= max_cost:
                    G.add_edge(("out", i), ("in", j), capacity=1, weight=int(1000 * c))
    flow = nx.min_cost_flow(G)
    chains = []
    for i in range(len(ts)):
        if flow["S"].get(("in", i), 0) != 1 or flow[("in", i)][("out", i)] != 1:
            continue
        path, cur = [ts[i]], i
        while True:
            nxt = [n[1] for n, v in flow[("out", cur)].items() if n != "T" and v == 1]
            if not nxt:
                break
            cur = nxt[0]
            path.append(ts[cur])
        chains.append(path)
    return chains


def build_chains(o: Observations, tls: list[Tracklet], det_dt: float, max_cost: float = 1.6) -> list[Chain]:
    """Fixed slots per shot (5 per team, 3 referees) by a global chain cover; never across a cut."""
    chains: list[Chain] = []
    for shot in range(len(o.shots)):
        in_shot = [t for t in tls if o.shot_of(o.det_frames[int(t.ks[0])]) == shot]
        for g, K in SLOTS.items():
            ts = [t for t in in_shot if t.group == g]
            if not ts:
                continue
            paths = _merge_numbers(_chain_cover(ts, K, det_dt, max_cost))
            if len(paths) < K:  # identity pass merged slots: refill them from the unused tracklets
                used = {id(t) for p in paths for t in p}
                taken = {t.jersey for p in paths for t in p if t.jersey}
                rest = [t for t in ts if id(t) not in used and t.jersey not in taken]
                paths += _chain_cover(rest, K - len(paths), det_dt, max_cost) if rest else []
            for path in paths:
                c = Chain(g, path)
                c.finalize()
                chains.append(c)
    return chains


def _merge_numbers(paths: list[list[Tracklet]]) -> list[list[Tracklet]]:
    """Identity pass: chains of one shot and team with the same hard number are one player.

    Merged only if they never overlap in time (otherwise one of them holds a misread).
    """
    def label(p: list[Tracklet]) -> str | None:
        c = Counter()
        for t in p:
            if t.jersey:
                c[t.jersey] += t.weight
        return c.most_common(1)[0][0] if c else None

    out: list[list[Tracklet]] = []
    for p in sorted(paths, key=lambda p: -sum(t.weight for t in p)):
        lab = label(p)
        ks = set(np.concatenate([t.ks for t in p]).tolist())
        for q in out:
            if lab and label(q) == lab and not ks & set(np.concatenate([t.ks for t in q]).tolist()):
                q.extend(p)
                q.sort(key=lambda t: t.ks[0])
                logger.info("merged slots by number %s", lab)
                break
        else:
            out.append(p)
    return out


def link_shots(o: Observations, chains: list[Chain], min_sim: float = 0.7) -> list[Chain]:
    """Join per-shot chains of the same group across cuts by OSNet appearance (Hungarian).

    Motion means nothing across a cut, so this is the weakest link in the pipeline; the joined
    identity records ``shot_links`` with the similarity used.
    """
    from scipy.optimize import linear_sum_assignment

    shot_of = lambda c: o.shot_of(o.det_frames[min(c.obs)])
    idents: list[Chain] = []
    for shot in range(len(o.shots)):
        for g in SLOTS:
            cur = [c for c in chains if c.group == g and shot_of(c) == shot]
            prev = [c for c in idents if c.group == g]
            if prev and cur:
                def sim(p: Chain, c: Chain) -> float:
                    if p.jersey() and c.jersey():  # identity pass: numbers decide
                        return 2.0 if p.jersey() == c.jersey() else -1.0
                    bonus = 0.0 if not (p.number() and c.number()) else (0.15 if p.number() == c.number() else -0.15)
                    return float(p.emb() @ c.emb()) + bonus

                S = np.array([[sim(p, c) for c in cur] for p in prev])
                r, q = linear_sum_assignment(-S)
                used = set()
                for i, j in zip(r, q):
                    if S[i, j] >= min_sim:
                        prev[i].members += cur[j].members
                        prev[i].obs.update(cur[j].obs)
                        prev[i].shot_links.append(round(float(S[i, j]), 3))
                        used.add(j)
                cur = [c for j, c in enumerate(cur) if j not in used]
            idents.extend(cur)
    return idents
