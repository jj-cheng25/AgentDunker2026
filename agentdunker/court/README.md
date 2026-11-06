# agentdunker.court — per-frame NBA court calibration

Image ↔ court homography for every frame of an NBA broadcast clip (M2 of AgentDunker).

```bash
python -m agentdunker.court clip.mp4 -o court.json                  # contract JSON
python -m agentdunker.court clip.mp4 -o court.json --overlay qa.mp4  # + court lines re-projected per frame and a minimap
python -m agentdunker.court clip.mp4 -o court.json --tracks tracks.json --overlay qa.mp4   # M1 feet on the minimap
```

```python
from agentdunker.court import CourtCalibration, img_to_court
cal = CourtCalibration.load("court.json")
cal.img_to_court(f, [[x, y], ...])   # image px -> court ft (NaN if frame f has no homography)
cal.court_to_img(f, [[X, Y], ...])
cal.ok(f)
img_to_court("court.json", f, [[x, y]])  # one-off convenience
```

## Court frame and contract

Feet, origin at centre court, `x ∈ [-47, 47]` along the length (+x = basket on the right of the main broadcast
view), `y ∈ [-25, 25]`, **+y = near (camera-side) sideline**. `t` is source PTS seconds.

```jsonc
{"video": {...}, "shots": [[0, 125], [125, 1004], ...],
 "frames": [{"f": 0, "t": 0.0, "H_img2court": [[...]] | null, "conf": 0.0, "reproj_px": 1.2, "ok": true,
             "src": "measured|propagated|interpolated|none", "shot": 1,
             "camera": {"f_px": 3500.0, "C_ft": [x, y, z]}}]}
```

- `H_img2court` maps homogeneous image pixels to court feet (scale-free: always divide by the third coordinate). `null` only for shots with no usable court view
  (close-ups, crowd, very low courtside angles); those frames are `ok=false`.
- `conf` ∈ [0, 1] comes from the line-evidence score of the final homography; `ok` = score above threshold.
  `interpolated` frames were outliers or gaps filled from neighbours in the same shot (`conf` × 0.7).
- `reproj_px`: median distance (1080p px) between projected court lines and detected painted lines.
- `camera.C_ft` is the estimated camera position in the court frame (z < 0 is above the floor).

## Method

1. **Keypoints**: court-33 YOLO11m-pose (Roboflow `basketball-court-detection-2`, weights from
   `MTerryJack/Element-PitchCalib-Basketball-Court`, downloaded and sha256-checked on first use).
   Input is stretch-resized to 640×640 as in training (letterboxing shifts keypoints by 40–120 px).
2. **Hypotheses → camera**: RANSAC / leave-out / confidence-weighted 4-point homographies, each fitted by a pinhole
   broadcast camera (focal, rotation, position; roll ≈ 0; camera on the near side 10–120 ft up).
3. **Line evidence**: thin painted-line response (morphological top-hat/black-hat), Hessian ridge test, and a
   clutter mask that drops crowd, players, benches and floor lettering. Painted-area edges (which vary by arena)
   give no response; lane side lines get low weight because some arenas paint the key narrower than 16 ft.
4. **Refinement and selection**: orientation-binned distance-transform alignment with a shrinking robust scale,
   then sub-pixel ridge matching. Hypotheses are ranked by F1 of template recall and floor-stripe precision.
5. **Video**: shot cuts (colour histogram + KLT inlier ratio, dissolves merged); one camera position per shot;
   2 Hz keyframe measurements; KLT inter-frame motion propagates the camera both ways from the best keyframe and
   keyframes compete with the tracked estimate; re-detection when support drops. Smoothing is a complementary
   filter: KLT dead-reckoning carries fast pans, and the slowly varying residual to the line fits is smoothed
   robustly (local quadratic, Gaussian sigma 0.5 s) with outlier replacement. Repeated frames keep their twin's H.

## Evaluation (M0 bench, `python -m agentdunker.court.tools.court_errors` / `bench/score.py`)

See `$CMD/status/m2.md` for the current numbers. Tools: `tools/court_errors.py` (per-point GT errors),
`tools/jitter.py` (frame-to-frame line jitter against KLT image motion), `tools/summarize.py` (batch table),
`tools/detect_persons.py` (stand-in person boxes until M1 tracks exist).

## Limits

- Close-ups and very low courtside cameras (only the far sideline visible) get no homography (`ok=false`).
- Behind-the-basket or opposite-side camera angles are rejected by the camera prior.
- Where a court region is hidden (score bug, crowd, lettered aprons) the homography is extrapolated; errors grow
  toward those image corners.
- Licence: the court-33 ONNX declares AGPL-3.0.
