# agentdunker.events — shot, ball path and possession events (AgentDunker M5)

```bash
python -m agentdunker.events clip.mp4 --tracks tracks.json [--court court.json] [--identity identity.json] -o events.json \
    [--cache DIR] [--overlay preview.mp4] [--no-vlm] [--no-pose] [--threads 8]
```
```python
from agentdunker.events.pipeline import run
doc = run("clip.mp4", "tracks.json", "court.json", "identity.json", cache_dir="cache/")
```
Inputs: M1 `tracks.json` (required for shooter/holders; a missing file still gives release, made and ball path),
M2 `court.json` (screens, drives, 2/3 points, shot location), M3 `identity.json` (official PBP match + xFG).
The run never fails on uncertainty: no shot found → `shot` fields are null, timelines still come out.

## Output (events.json)
All `t` are source PTS seconds (`t = start_pts + f / fps`), `f` source frame indices.
- `events[]` `{t, type: pass|screen|drive|shot|rebound|other, actors, conf, t_end?, made?, main?, subtype?}`
- `shot` `{release_t, release_f, shooter_track, made, made_conf, made_source, conf, xfg, points, shot_type,
  shot_type_conf, shot_court_xy, shot_distance_ft, ball_path_img [[t,x,y]], rim_img [[t,cx,cy,w]], rim_t, made_t,
  bank_t, official, evidence}`
- `ball_state[]` `{type: hold|dribble|pass|shot|loose, t0, t1, f0, f1, conf, holder | from,to | from}`
- `actions[]` `{track, type: catch|pass|dribble|shoot|jump|screen, t0, t1, f0, f1, conf, pose?, for?}`
- `attempts[]` every attempt; `shot` = the last one.  `offense_team`, `video`, `meta`.

## Method
1. **Ball / rim / shot poses**: DEIM-M fine-tuned on NBA broadcast frames (`ortizeg/basketball-deim-m-640`,
   Apache-2.0; data Roboflow basketball-player-detection-3, CC BY 4.0), full frame at 640, 10 Hz over the clip and
   every frame (≤30 Hz) around each rim visit. Cached per video.
2. **Ball track**: longest path over detections (score − miss cost − over-speed penalty); ball candidates fixed
   relative to the rim (net, logos) are removed. Rim boxes are median-smoothed.
3. **Attempts**: the ball comes to the rim neighbourhood from above. Around each visit, YOLO11x-pose (AGPL-3.0) on a
   ball-centred crop gives wrists; a *hold* = one person keeps a wrist within 2 ball diameters for ≥ 0.15 s; the
   *release* is the end of a hold after which the ball leaves that person (or the view). Each release followed by the
   ball reaching the rim is an attempt (a missed layup and its putback are separate). Shooter = wrist vote over the
   hold (recent frames weigh more; a late hand from the other team is a contest), plus DEIM shot/block pose class.
   Fallback without pose: last contact of an M1 box at the start of the parabolic flight.
4. **Made**: rim-relative path; the last downward crossing of the rim plane inside the rim with the ball falling
   through = made, outside = missed; otherwise a cached `gemini-3.8-flash` check on 12 rim crops; then official PBP
   when M3 matched the game. **Bank**: horizontal reversal above the rim before the ball first reaches it.
5. **Possession**: holder per 10 Hz frame from ball contact with M1 boxes, DEIM `player-in-possession`, and wrists on
   ambiguous frames; runs are cleaned (tight defence, ball flying past defenders) and split into hold / dribble by ball
   height in the holder box; gaps between teammates = pass, otherwise loose.
6. **Actions**: catch/pass around passes, dribble segments, shoot window, jump from DEIM pose classes, screens and
   drives from M2 court positions (speed / distance rules).
7. **Official data**: PBP field goal inside the M3 clock window (name/team/made agreement) → `official`, and xFG from
   `data/gpshot/<game>/<personId>.json` (eventId == actionNumber). `release_t` is never taken from PBP (the game clock
   stops; PBP time is not video time).

## Tools
`python -m agentdunker.events.tools.bench --out DIR` (run on the M0 bench with M1/M2/M3 evidence + score.py),
`python -m agentdunker.events.tools.summary --pred DIR` (shooter / release / made table).

## Licences
DEIM weights Apache-2.0; YOLO11 pose AGPL-3.0 (same as M1); onnxruntime MIT.
