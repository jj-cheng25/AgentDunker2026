# Performance

Measured 2026-11-06 and 2026-11-10 on a shared 64-core CPU node with **16 threads** for AgentDunker (other jobs kept the load at
12–20), no GPU. Times are wall-clock for `agentdunker run` from an empty output directory. Model-answer caches
were warm only where noted.

## End to end

| run | clip | preset | services | wall time | film |
|---|---|---|---|---|---|
| clean venv, fresh | c01 (22.0 s, 59.94 fps) | quality | all live (Mosshub LLM, MOSS voice) | **1340 s (22.3 min)** | 72.6 s, 15/16 elements A |
| **fresh HOME, first film** (2026-11-10, current pins) | c01 | quality | none (`--offline`) | **663 s (11.1 min)**, intro/outro rendered in the background | 74.7 s ³ |
| same HOME, second film | c01 | quality | none | 659 s (11.0 min) | 74.7 s |
| same machine, offline (2026-11-06, M7 4e1735f) | c01 | quality | none | 629 s (10.5 min) | 71.3 s ³ |
| offline | c01 | fast | none | **501 s (8.3 min)** | 71.3 s |
| offline | c06 (14.8 s) | quality / fast | none | 445 s ¹ / 300 s | 56.7 s / 57.2 s |
| offline | c12 (10.4 s) | quality / fast | none | 366 s / 294 s | 46.4 s / 35.5 s ² |

¹ measured 824 s, of which 379 s was a one-time rebuild of the intro/outro cache (the promo pages had just changed);
see "first run" below. ² fast on c12 shortened the 3D window (tier B), hence the shorter film.
³ 74.7 s vs 71.3 s is intended: M7 88e4ea0 slowed the 3D replay to 0.56× (0.5× for the last phase, closer to the
reference film), so the c01 replay segment grew from 22.8 s to 26.2 s. Offline films also use the 3.2 s freeze
(no official xFG card) instead of 5.2 s. Segments now: intro 6, live 21.95, transition 4.8, replay 26.19,
freeze 3.2, resume 6.55, outro 6. The fast-preset rows were measured at the 11-06 pins.

Stage times (seconds) — the analysis stages run tracking ∥ court, then identity, then events:

| stage | c01 quality, live | c01 quality, offline | c01 fast, offline |
|---|---|---|---|
| tracking (YOLO11 + BoT-SORT + OSNet + CLIP) | 389 | 357 | 198 |
| court calibration (parallel with tracking) | 248 | 223 | 224 |
| identity (jersey VLM, score bug, PBP) | 85 | 4 | 4 |
| events (ball, rim, release pose) | 64 | 58 | 58 |
| commentary (MOSS TTS + ASR read-back + retakes) | 590 | 0 (subtitles) | 0 |
| 3D replay render (3 headless Chromium) | 105 | 108 | 110 |
| other renders + mix + mux | ≈ 105 | ≈ 90 | ≈ 90 |

**Where the time goes**: with all services, the voice (MOSS TTS with ASR read-back and retakes, network-bound,
340–590 s on c01) is the longest single stage, then tracking. With `fast` the tracking stage drops below court
calibration, which becomes the critical path of the analysis (≈ 0.17 s per frame on CPU).

**First run on a machine**: the intro, outro and cover are rendered once from their HTML sources (≈ 390 s on 16
threads) and cached in `paths.cache_dir`. `agentdunker download-models` now does this at the end; if it has not
been done, the first film renders it in the background while tracking and court calibration run (its own progress
line, "intro/outro render — first run only"). Measured 2026-11-10 in a fresh HOME: first film 663 s vs 659 s for
the second, i.e. the overlap costs ≈ 4 s. Before this change the render ran silently after the 3D replay and added
≈ 386 s (commander's clean install, 2026-11-09: 1051 s for the same film).

**Chromium launch check**: `doctor` and `run` start Chromium once (headless, about:blank) to catch missing system
libraries: 0.31–0.36 s per check; a whole `agentdunker doctor` takes ≈ 4.5 s.

## What `fast` costs (M0 benchmark, `bench/score.py`, c01 + c06 + c12, offline)

`fast` = tracking with YOLO11m instead of YOLO11x and detection at 10 Hz instead of 15 Hz; everything else as
`quality`. Same offline condition for both, so the numbers isolate the preset (offline itself lowers tracking
quality because the one VLM role/team adjudication per clip is skipped; with the LLM, c01 quality scores IDF1 0.920).

| metric | quality | fast | fast v1 (960 px, no pose) |
|---|---|---|---|
| player recall / precision | 0.839 / 0.960 | 0.834 / 0.973 | 0.811 / 0.987 |
| IDF1 | **0.732** | **0.646** | 0.646 |
| ID switches | 15 | 34 | 30 |
| team accuracy | 0.915 | 0.874 | 0.877 |
| tracks mixing teams | 5 | 6 | 10 |
| court reprojection median (px) | 4.31 | 4.31 | 4.31 |
| shooter correct | 3/3 | **2/3** (c06 wrong) | 2/3 |
| release-time error | 0.050 s | 0.050 s | 0.139 s |
| made / missed correct | 3/3 | 3/3 | 3/3 |
| wall time, sum of 3 clips | 1440 s ¹ | 1094 s (−24 %) | 1038 s |

The first fast version also shrank the detector input to 960 px and skipped the release pose: 5 % faster than the
current `fast`, but the release-time error tripled, so it was dropped. Court calibration is the same in both presets.

Per clip (quality → fast): IDF1 c01 0.898 → 0.815, c06 0.674 → 0.505, c12 0.626 → 0.616; c06's shooter is lost in
`fast` (and its 3D window with it).

## Against the targets

| target | result |
|---|---|
| 20 s clip, `fast`, 16-core CPU, ≤ 15 min | **met offline** (c01 22 s: 8.3 min). With the MOSS voice the estimate is ≈ 20 min (c01 quality live 22.3 min − ≈ 2 min tracking saved); the voice stage alone is 6–10 min and is network-bound |
| with a GPU, ≤ 5 min | **not measured** (no GPU available). Detection (Ultralytics), ReID, CLIP and the ONNX ball detector move to CUDA automatically; court keypoints (OpenCV DNN) and the Chromium renders stay on CPU, and the voice stage is network-bound |

## Knobs

- `--threads N` (or `run.threads`) caps every stage (torch, ONNX Runtime, OpenCV, BLAS).
- `--preset fast|quality`; `--offline` / `--no-voice` remove the network-bound stages.
- Re-running into the same output directory resumes: every analysis stage is cached by input fingerprint, so a
  re-run after a failure or a change in a later stage only redoes what changed.

Raw data: `tools/bench_presets.py` writes `presets.json` (per-run stage times, tiers, scores). Running it needs the
private benchmark clips.
