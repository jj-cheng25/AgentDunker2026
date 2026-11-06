import numpy as np

from agentdunker.tracking.identities import Tracklet, _chain_cover, _sign_segments


def test_sign_segments_absorbs_short_flips():
    sgn = np.array([1] * 20 + [-1] * 3 + [1] * 10 + [0] * 2 + [-1] * 20)
    assert _sign_segments(sgn, 15) == [(0, 33, 1), (33, 55, -1)]
    assert _sign_segments(np.array([-1] * 4 + [1] * 30 + [-1] * 12 + [1] * 20), 15) == [(0, 66, 1)]


def _tl(tid, k0, k1, x, number=None):
    ks = np.arange(k0, k1)
    box = np.array([x, 100.0, x + 40, 200.0])
    stab = np.repeat(box[None], len(ks), 0)
    emb = np.ones(8) / np.sqrt(8)
    t = Tracklet(tid, ks, np.zeros(len(ks), int), stab.copy(), stab, np.full(len(ks), 0.9), emb, np.zeros(10))
    t.group, t.group_conf, t.number = "A", 0.95, number
    return t


def test_chain_cover_links_fragments_and_respects_slots():
    a, b = _tl(1, 0, 20, 100), _tl(2, 22, 40, 105)  # same player, short gap, nearby
    c = _tl(3, 0, 40, 600)  # another player
    d = _tl(4, 0, 25, 900)  # a weaker concurrent tracklet that cannot fit into 2 slots
    chains = _chain_cover([a, b, c, d], K=2, det_dt=1 / 15, max_cost=1.6)
    ids = sorted(tuple(t.tid for t in ch) for ch in chains)
    assert ids == [(1, 2), (3,)]  # fragments joined into one slot; the weaker extra tracklet dropped


def test_identity_numbers_are_hard_links():
    from agentdunker.tracking.identities import _merge_numbers, pair_cost

    a, b = _tl(1, 0, 20, 100), _tl(2, 22, 40, 105)
    a.jersey, b.jersey = "A#3", "A#7"
    assert pair_cost(a, b, 1 / 15) == np.inf  # different confident numbers never link, however close
    far = _tl(3, 30, 50, 700)  # same number far away: beyond the plain motion gate, still linked
    far.jersey = "A#3"
    assert pair_cost(a, _tl(4, 30, 50, 700), 1 / 15) == np.inf
    assert pair_cost(a, far, 1 / 15) < 0
    merged = _merge_numbers([[a], [far], [b]])
    assert sorted(tuple(t.tid for t in p) for p in merged) == [(1, 3), (2,)]
