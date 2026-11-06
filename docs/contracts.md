# Data contracts

Every stage reads and writes plain JSON in `out/work/`. Court coordinates are **feet with the origin at centre
court**: x ∈ [-47, 47] along the long side (+x = the basket on the right of the main camera), y ∈ [-25, 25]
(+y = near sideline). Times: `t` is source-video PTS seconds, `f` the frame index.

```jsonc
// tracks.json (tracking)
{"video": {"path": "", "sha256": "", "fps": 0, "width": 0, "height": 0, "n_frames": 0},
 "tracks": [{"track_id": 1, "role": "player|referee|other", "team": "A|B|null", "team_color": [0,0,0],
             "frames": [{"f": 0, "t": 0.0, "bbox": [0,0,0,0], "foot": [0,0], "conf": 0.0, "observed": true}]}]}
// court.json (court) — H is NOT normalised: divide by the third component
{"frames": [{"f": 0, "H_img2court": [[0,0,0],[0,0,0],[0,0,0]], "conf": 0.0, "reproj_px": 0.0, "ok": true}]}
// identity.json (identity)
{"players": {"<track_id>": {"jersey": "35", "team_abbr": "HOU", "name": "Kevin Durant", "conf": 0.0, "evidence_frames": []}},
 "game_match": {"game_id": "", "period": 0, "clock": ["", ""], "conf": 0.0}}
// events.json (events)
{"events": [{"t": 0.0, "type": "pass|screen|drive|shot|rebound|other", "actors": [1], "conf": 0.0}],
 "shot": {"release_t": 0.0, "shooter_track": 1, "made": true, "xfg": null, "ball_path_img": []}}
// narration.json (narration)
{"lines": [{"start": 0.0, "end": 0.0, "text": "", "audio": "", "event_ref": null}], "mix": "narration_mix.wav"}
```

`observed=false` marks an interpolated or predicted position: drawable, but not a measurement.

Each module writes more fields than the contract (for example `court.json` per-frame `src`/`shot`/`camera`,
`events.json` `ball_state`, `actions`, `attempts`, `shot.rim_img`, `identity.json` `game_match.status`,
Chinese names). Consumers must ignore fields they do not know; producers never remove contract fields.

Film-level outputs:

- `edl.json` — the director's edit decision list: segments (`intro, live, bridge, replay, freeze, resume, outro`)
  with film and source times, speeds, overlays and data sources.
- `report.json` — `elements` (tier A/B/C + reason for every visual element), `modules` (ok, seconds, cached),
  `timing_s`, `model_calls`, `notes`, and `agentdunker` (version, preset, services available, degraded list).
