# agentdunker.replay (M4): ten-player holo 3D replay

```bash
python -m agentdunker.replay clip.mp4 --tracks tracks.json --court court.json \
    [--identity identity.json] [--events events.json] -o out/replay.mp4 \
    [--start S --end S] [--timeline knots.json | --timeline '[[0,4.0],[4,5.0],[5,5.0],[8,8.0]]'] \
    [--annotations annotations.json] [--compare out/side_by_side.mp4] [--workers 3] [--width 1920 --height 1080 --fps 30] [--no-pip]
```

`--annotations` (tactical graphics: screen, pass, cut, drive, help …): `{"annotations": [...], "clock": "replay"}` in the
renderer format of the hackathon play.json (types `arrow` (path {who, t} or from/to), `zone` (circle at/r, poly points,
preset paint), `link` (ids), `focus` (who), `label` (at, text); colours `off def warn good gold ball`). Players are
`T<track_id>`; times are seconds on the replay clock (from the window start, as in `replay.play.json`), or source PTS
with `"clock": "source_pts"`. Entries with an unknown type, a bad interval, a player not in the replay, or a `ball`
anchor are dropped; intervals are clipped to the replay. `replay.json["annotations"] = {kept, dropped[]}`.
Without the flag: no annotations (unchanged).

`--timeline` (director pacing, e.g. audio-first 3D timing): piecewise-linear knots `[[out_s, src_pts], …]`
(output seconds → source PTS seconds; a JSON file, inline JSON, or `{"timeline": [...]}`). Output times must
increase (a first knot not at 0 is held from 0); source times must not decrease (equal source times = freeze).
The knots set the window (`[first src, last src]`, no automatic trimming) and the render clock exactly: no
constant slow-down and no end hold are added. PIP and the QA video use the same source time. An invalid timeline
falls back to the default pacing with a `degraded` note; `replay.json["timeline_source"]` is `director` or `default`.
Without `--timeline`: constant slow-down to about 24 s (0.5–1.0×) + 1 s end hold.

```python
from agentdunker.replay import make_replay
rep = make_replay("clip.mp4", "tracks.json", "court.json", "out/replay.mp4", identity=None, events=None,
                  timeline=None,      # or [[out_s, src_pts], ...]
                  annotations=None)   # or a path to {"annotations": [...]}
if rep["ok"]: ...   # else skip the 3D segment (rep["reason"])
```

Outputs next to `-o`:
- `replay.mp4`: silent H.264, 1080p30, "原视频" PIP bottom right (same source time as the 3D).
- `replay.json`: report. `ok`, `window` (source PTS), `timeline_source_pts` = `[[output_s, source_pts], …]`
  knots (use these to place narration and subtitles), `observed_frac` per player, `degraded` notes, `render` timing.
- `replay.play.json`: the renderer scene (paths in metres, ball segments, actions, phases).
- `--compare`: QA video, 3D on the left, source on the right with every replay position projected back
  through the inverse homography (filled dot = observed, ring = predicted).

`ok: false` (exit code 0, `reason` set) means there is no calibrated wide-shot window of at least 3 s, or rendering
failed after retries (stalled browsers are restarted 3×). The 3D segment should then be skipped.

## Pipeline

1. **Window**: longest stretch of frames with `court.ok` inside one M1 camera shot (gaps ≤ 0.35 s bridged, 0.5 s
   skipped after a cut). With `events.shot.release_t` it ends 0.65 s before release (M7 shows the shot as a live freeze),
   and it is trimmed back if fewer than 8 players are observed at the end. If the players cross half court, it starts
   1 s before the final crossing (final possession). `--start/--end` or `--timeline` override all of this.
   Basket and offense are judged from the last 2 s.
2. **Court paths** (`trajectory.py`): M1 `foot` on detected, observed, `foot_reliable` frames, dropping merged boxes
   (box height > 1.2 × rolling median) → `H_img2court` (divided by the third component) → per-player robust Kalman
   filter (constant velocity with velocity decay τ = 1.2 s, measurement noise from the homography Jacobian × (8 px +
   3 % box height), χ² gate) + RTS smoother at 30 Hz → speed clamp 9.1 m/s. Gaps and jumps are interpolated; before the
   first / after the last measurement a player is held. Every player is present for the whole window; predicted
   stretches are in `players[].predictedIntervals`.
3. **Scene** (`scene.py`): ≤ 5 players per team with the most measurements; attacked basket from the shooter's
   release position, else the half the players stand on; offense = shooter's team, else the team farther from the
   basket. Renderer frame: x lateral (+ = far sideline), z = metres from the attacked baseline (rotated 180° for the
   right basket, camera on the near side). No clamp: `fullCourt` draws both halves when anyone is past half court.
4. **Motion**: procedural point-cloud bodies (speed-driven gait, facing = ball holder, else matchup / hoop).
   Defenders get the `defend` stance except while running > 3.5 m/s; the shooter gets `shoot` + `jump` at release.
5. **Ball**: only where possession is known: pass chain from `events` (`type: pass`, `actors: [from, to]`,
   optional `flight`), shooter holding ≥ 0.4 s before release. Otherwise the ball is hidden. Never invented.
6. **Names**: `identity.players[track_id]` with `conf ≥ 0.5` → number + `name_zh` (else `name`); team names from
   `team_abbr`. Without identity: M1 `jersey_hint` number + 进攻/防守.
7. **Phases** (minimap panel): events with `label_zh` (+ `note`), else event types.
8. **Render** (`browser.py`, `compose.py`): headless Chromium + SwiftShader (CPU), `holo.renderAt(t)` per frame,
   frames split over `--workers` browsers; time map = constant slow-down to ~24 s (0.5–1.0×) + 1 s end hold.

## Renderer

`web/` is the hackathon holo renderer (`studio/holo_court_import`, three.js 0.170, MIT) with small patches:
hidden-ball intervals, `cameraPreset`, `exportPush`, `exportFitScale`, `fullCourt`, matchup facing, badges without a
number, film-size badges. Licences: three.js MIT; `models/Xbot.glb` is not used (point-cloud bodies only).

## Benchmark

`python -m agentdunker.replay.bench fixtures` writes GT-derived identity/events stand-ins (for M3/M5) to `fixtures/`.
`python -m agentdunker.replay.bench score <clip> <replay.play.json>` scores positions against M0 GT feet;
`python -m agentdunker.replay.bench regress <c01 replay.play.json>` compares with the hackathon film's c01 paths.
`python -m agentdunker.replay.summary <dir>` tabulates a batch.
