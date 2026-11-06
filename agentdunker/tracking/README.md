# agentdunker.tracking (M1): players, referees, teams

```bash
python -m agentdunker.tracking clip.mp4 -o tracks.json [--overlay overlay.mp4] [--threads 16] [--no-vlm]
```

```python
from agentdunker.tracking import track_video, TrackConfig
res = track_video("clip.mp4", TrackConfig(det_hz=15))   # dict in the PLAN.md tracks.json contract
```

Optional flags (the defaults above are unchanged by them):

| flag | effect |
|---|---|
| `--work-dir DIR` | keep pass 1 (detections, features, CLIP cache) in `DIR/pass1_<hash>.pkl`; a second run of the same video and config reuses it. Only point it at directories you own (pickle). |
| `--identity identity.json --identity-tracks tracks.json` | identity pass: M3's confident numbers (`conf` ≥ 0.7) become hard links. Each `evidence_frames` entry is mapped by box IoU onto the detection in the tracks.json M3 read. Then: a tracklet holding two numbers is cut between them; tracklets or slots of one team with different numbers never link; equal numbers always link, and same-number slots in one shot merge with the freed slots refilled; recovery and cross-cut linking follow the numbers. Adds `tracks[].identity = {jersey, from_tracks}` and `meta.identity`. Measured on 14 clips: **no gain** (see `status/m1.md`). |
| `--max-hold S` | omit `observed=false` frames more than S s before the first / after the last sighting in a shot, and predicted boxes less than half on screen. Raises IDF1 because the M0 scorer counts every box (14 clips: 0.847 → 0.902 at S = 1), but players then drop out of frames (c14: 10 players in 158/271 frames instead of 271/271). |

M7 chain with the identity pass:

```bash
python -m agentdunker.tracking clip.mp4 -o tracks.json --work-dir w
python -m agentdunker.identity clip.mp4 --tracks tracks.json -o identity.json        # M3
python -m agentdunker.tracking clip.mp4 -o tracks_id.json --work-dir w --identity identity.json --identity-tracks tracks.json
```

## Pipeline

1. **Detect** `yolo11x.pt` at `imgsz=1280`, person class, conf ≥ 0.08, on every `round(fps/15)`-th frame.
2. **Camera motion** frame-to-frame similarity transform (sparse LK on a 640 px grey image, people and
   score bug masked, RANSAC), accumulated into a camera-stabilised plane. When < 25 % of points fit,
   the frame is a **flash** (brightness jump; pre-flash frame stays the reference) or a **shot cut**.
3. **Tracklets** ultralytics BoT-SORT per shot, driven with that warp and OSNet-x0.25 (MSMT17) embeddings.
4. **Roles** zero-shot CLIP ViT-L/14 on ≤ 3 crops per tracklet (≤ 240 crops per clip, cached per crop).
5. **VLM adjudication** one call per clip: a numbered sheet of the 40 strongest tracklets (2 crops each) is
   labelled `team1 / team2 / referee / other / mixed` plus a readable jersey number. Default model
   `gpt-6-astra`, fallback `gemini-3.8-flash` (`ADK_VLM_MODEL` overrides, comma-separated order).
   Labels attach to the judged crops. Responses are cached by request hash (`~/.cache/agentdunker.tracking/vlm`).
   No credential or failure → local-only (CLIP role + OSNet 2-means).
6. **Teams** OSNet centroids from the labelled tracklets (team A = brighter jersey). Unlabelled tracklets:
   0.5·CLIP role + 0.5·similarity ramp calibrated on the labelled members; < 0.4 → dropped.
7. **Switch splitting** a tracklet whose per-detection appearance holds the other team for ≥ 1 s is cut
   (shorter flips are occlusions and are absorbed); the merged-box stretch before the cut is discarded.
   A piece keeps the VLM label of the judged crops it contains; other pieces of a known player get
   their team from A-vs-B appearance.
8. **Identity slots** per shot, ≤ 5 per team and 3 referees, as a min-cost-flow chain cover
   (reward = Σconf · group confidence, link cost = stabilised foot-motion feasibility + appearance,
   soft jersey-number cue, ≤ 3 steps hand-over overlap).
9. **Recovery** every slot re-walks its gaps and ends inside its shot over all unused detections
   (conf ≥ 0.08) with an IoU gate and a team-level appearance check; merged two-person boxes are swapped
   for an unused single-person box inside them.
10. **Cross-cut linking** Hungarian on appearance (+ number cue) between consecutive shots.
11. **Export** every frame of every shot in which an identity was observed.

## Output extras (beyond the contract)

| field | meaning |
|---|---|
| `frames[].observed` | `true` on detection frames and between consecutive detections of the slot (regular cadence); `false` for gap interpolation / holding |
| `frames[].detected` | an actual detector box on this frame |
| `frames[].foot_reliable` | `false` when not observed, the box touches the frame bottom, or the foot rises > 10 % of box height above its 0.8 s median track (jump) |
| `tracks[].group_conf` | role × appearance score of the slot's tracklets (0–1) |
| `tracks[].jersey_hint` | majority VLM-read number of the slot (≈ 75–90 % right; a hint for M3, not an identity) |
| `tracks[].source_tracklets`, `shot_links` | merged BoT-SORT tracklets; appearance similarity of each cross-cut join |
| `shots` | `[{start_f, end_f}]` |
| `meta` | config, VLM model and team descriptions, timing |

A track has frames only for shots in which it was seen; inside such a shot every frame is present.

## Degradation

- Fewer than 5 confident tracklets of a team in a shot → fewer slots (never fabricated).
- Inside a shot, before the first / after the last observation a slot is held at its last stabilised
  position with `observed=false` and decaying `conf`.
- VLM unavailable → local roles/teams (clearly weaker on similar-coloured or busy sidelines).

Model weights go to `models/` at the repo root (`ADK_MODEL_DIR` overrides; downloaded on first use,
not committed). Licences: ultralytics YOLO11 and BoT-SORT AGPL-3.0, OSNet MIT, CLIP MIT.
