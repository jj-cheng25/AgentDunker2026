"""Stages start in parallel threads; their first imports must not race (h04 held-out, 2026-11-20: scipy was left
half-initialised, tracking and court failed at 0.2 s and the film died at the audio mix)."""
import importlib.util
import sys
import threading
import time
import types

import pytest

from agentdunker import config, runner


def test_apply_preloads_the_shared_scientific_stack():
    pytest.importorskip("scipy")
    config.apply()
    for name in config.PRELOAD:
        if importlib.util.find_spec(name.split(".")[0]) is not None:
            assert name in sys.modules, name


def test_stage_imports_run_one_at_a_time(monkeypatch, tmp_path):
    active, peak, lock = [0], [0], threading.Lock()

    def fake_import(name):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)                       # a slow first import (network disk under a 12-film queue)
        with lock:
            active[0] -= 1
        return types.SimpleNamespace(main=lambda argv: 0)

    monkeypatch.setattr(runner, "importlib", types.SimpleNamespace(import_module=fake_import))
    rcs = []
    ts = [threading.Thread(target=lambda m=m: rcs.append(runner._inprocess(m, [], tmp_path / f"{m}.log")))
          for m in ("m1", "m2")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 1 and sorted(rcs) == [(0, ""), (0, "")]
