# agentdunker.identity (M3): jersey numbers, team names, game match, player names

```bash
python -m agentdunker.identity clip.mp4 --tracks tracks.json -o identity.json [--overlay nameplates.mp4] \
       [--max-strong 20] [--no-deduce] [--threads 8]
python -m agentdunker.identity.bencheval --pred <dir>      # <dir>/<clip>/{tracks,identity}.json vs M0 GT
```

```python
from agentdunker.identity import identify_video, IdentityConfig
ident = identify_video("clip.mp4", tracks_dict)          # dict in the PLAN.md identity.json contract
```

Input is M1's `tracks.json` (players with team `A`/`B`). Credentials: `MOSSHUB_API_KEY` or
`~/.config/mosshub/key`. Official data root: `ADK_NBA_DATA` (default: the hackathon `data/` mirror).

## Pipeline

1. **Crops** per player track: detector boxes every ~0.1 s ranked by height, frame-edge cut, overlap with
   other people and box shape; the best 30 are decoded and re-ranked by torso sharpness; 8 spread ≥ 0.4 s
   apart are kept. Outside the box is dimmed so the tracked person stands out.
2. **Jersey read** one `gemini-3.8-flash` call per track on a numbered sheet of the 8 crops; every crop
   gets its own answer and "unreadable" (null) is allowed. Tracks with fewer than 5 agreeing crops get a
   second read by `gpt-6-astra`. Votes are per crop; a crop where the two models disagree counts against.
3. **Score bug** one cheap call on 3 full frames: tricodes, scores, period, game clock, banner.
   `ADK_ID_SCOREBOARD_DIGITS=moss` adds MOSS-VL digit reads (one call per frame, bottom 35 %, gemini as the
   fallback). Off by default: on the 14-clip bench MOSS-VL read 29/42 frames fully right against gemini's 42/42
   (`python -m agentdunker.identity.sbbench --ref reference.json`).
4. **Team naming** one cheap call on front-view crops of each group, restricted to the two score-bug teams.
5. **Game match** candidates = local PBP games of those two teams whose score timeline fits every
   reading (±8 s). Accepted only if exactly one candidate survives, the score state is rare
   (fits ≤ 5 % of all local games; 0-0 or 2-0 never matches) and the read jerseys fit the PBP on-court
   lineup (≥ 75 %, ≥ 50 % when the state fits ≤ 1 % of games). Otherwise `status: uncertain` with the
   candidates.
6. **Numbers** per team, a max-weight one-to-one assignment of tracks to numbers. Unmatched game: ≥ 2
   crops and ≥ 75 % of the track's read crops. Matched game: the number must be on the floor (starters +
   substitutions up to the score-bug clock). Off-lineup reads are dropped as misreads. Open tracks then
   get a strong-model re-read restricted to the free on-court numbers, and the last free number goes to
   the last open track (`source: lineup-deduction`).
   `conf`: evidence share × weight; in a matched game the lineup has checked the number, so it is rescaled to
   0.6–1 (no lineup-checked number was wrong on the bench). A deduced number gets the lowest conf of its four
   teammates, since it is right exactly when they and the lineup are.
7. **Names** from the matched game's boxscore. Without a match: team + number only.

## Output (PLAN contract + extras)

`players[<track_id>]`: `jersey`, `team_abbr`, `name`, `conf`, `evidence_frames`, plus `team` (M1 label),
`team_conf`, `source` (`vote` / `vote+lineup` / `lineup-deduction`), `votes` (per crop), `rejected`,
`person_id`, `name_zh` (贾巴里·史密斯), `name_zh_short` (史密斯), `team_name_zh` (火箭).
Chinese player names come from NBA China (api.nba.cn `inside/gpShot`, one request per named player, cached
per personId in `~/.cache/agentdunker.identity/nba_cn`); they are null when no game is matched or NBA China has no
Chinese name for the player (fall back to `name`). `teams[A|B].name_zh` is the Chinese team name.
`game_match`: `game_id`, `period`, `clock`, `conf`, plus `status` (`matched` / `uncertain` / `none`),
`home`, `away`, `date`, `type`, `candidates` (with `roster_fit` = read jerseys on the floor / read).
Also `teams` (A/B → abbr, conf, source), `scoreboard` readings and `meta.api` (calls per model, live vs
cached, seconds).

## Degradation

- No credential or failed calls → no numbers / no team names; the output still has every player track.
- No score bug, teams not readable, game not in local data, or several candidates → no game, no names;
  numbers only by the stricter unmatched rule.
- `conf` < 0.5 → the overlay shows team only (plain plate).

VLM responses are cached by request hash in `~/.cache/agentdunker.identity/vlm` (`ADK_ID_CACHE`). Models:
`ADK_ID_CHEAP_MODEL` (default `gemini-3.8-flash`), `ADK_ID_STRONG_MODEL` (default `gpt-6-astra`).
No new third-party dependency beyond numpy / scipy / opencv / requests.
