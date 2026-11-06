# Architecture

```
agentdunker/
  cli.py  run.py  runner.py  config.py  providers.py  weights.py  doctor.py  textnarration.py  device.py  ui.py
  tracking/   players, referees, teams      (synced from cmd/m1-tracking,   package adk_tracking)
  court/      image → court homography      (cmd/m2-court,      adk_court)
  identity/   jerseys, names, game match    (cmd/m3-identity,   adk_identity)
  replay/     Three.js holographic 3D       (cmd/m4-replay,     adk_replay)
  events/     ball, rim, shot, events, xFG  (cmd/m5-shot,       adk_events)
  narration/  script, voice, alignment      (cmd/m6-narration,  adk_narration)
  film/       shots, director, render, mix  (cmd/m7-integration, agentdunker)
  SYNCED_FROM.json
tools/sync_modules.py
```

## Flow

1. `tracking` ∥ `court`, then `identity`, then `events` (the runner caches each by input fingerprint).
2. `film.shots` cuts the clip into camera shots and labels them wide/close (rule; MOSS-VL veto when available).
3. `film.director` writes `edl.json`: the 3D window is the longest calibrated stretch of a wide shot ending 0.65 s
   before the release (≥ 3 s, else no 3D); phases split at measured tactical facts (passes, drives, cuts, screens,
   help, closeouts from feet × homography); phase titles from the LLM, validated, template fallback.
4. `narration` writes the commentary for every section and voices it (MOSS TTS, ASR read-back, retakes); the 3D
   pacing then follows the lines (audio-first), and `replay` renders exactly that timeline.
5. `film.render` draws nameplates on wide shots, the transition page, the freeze (spotlight + xFG card), the
   resume (ball trail + light wave); `film.audio` mixes to ≈ −21 LUFS; ffmpeg muxes `film.mp4`.

Nothing raises past a stage: a failed stage returns `ok=False`, its element degrades or disappears, and the reason
lands in `report.json`. Worst case: live play with the broadcast sound (plus intro/outro when Chromium is present).

## Runner

- **in-process (default)**: `agentdunker.<stage>.__main__.main(argv)` in the same interpreter; each stage's log
  records (logger `agentdunker.<stage>.*`) and prints go to `work/logs/<stage>.log`.
- **subprocess (`--isolate`)**: `python -m agentdunker.<stage>` per stage, the original M7 runner (timeouts,
  one retry after an OOM kill or a network-disk error).
- **presets**: `runner.PRESETS` adds stage arguments (`fast`: `--detector yolo11m.pt --imgsz 960 --det-hz 10` for
  tracking, `--no-pose` for events). `--threads` replaces every stage's thread option and the BLAS/OpenMP caps.

## Providers

`providers.py` is the only place that sends model requests. The synced modules call `llm_post()` (OpenAI wire
format; Responses API bodies are converted to chat completions for providers that lack it) and `moss_key()` /
`moss_or_llm_post()` for MOSS. Every module keeps its own answer cache keyed by the request hash, so re-runs cost
nothing. See [configuration.md](configuration.md).

## Module sync

The module code is developed on the `cmd/*` branches and copied here by `tools/sync_modules.py` from pinned commits
(read with `git archive`, never from a working tree). The script rewrites the `adk_*` imports and applies a list of
patch rules — path configuration, provider calls, device selection, the in-process runner, two robustness fixes —
each of which must match exactly once or the sync stops. Do not edit the synced directories by hand: change the
module upstream (or add a rule) and re-sync.

```bash
python tools/sync_modules.py                       # pinned commits
python tools/sync_modules.py --commit m7=<rev>     # move one pin for this run (then update MODULES)
python tools/sync_modules.py --check               # CI: is the tree what the pins produce?
```
