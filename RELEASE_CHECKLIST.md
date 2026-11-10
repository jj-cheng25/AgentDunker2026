# Release checklist

Target: GitHub `jj-cheng25/AgentDunker2026` — **public since 2026-11-06** (user decision). Commit 047f2aa was pushed to `main` by the commander from the
user's Mac with the user's approval (2026-11-06); nothing is pushed from cpu-0. Later pushes go through the commander. **Every push, and
making the repository public, needs the user's explicit go-ahead.**

## A. Decisions for the user

| # | item | why it matters | options / recommendation |
|---|---|---|---|
| A1 | **Demo material** | every film contains NBA broadcast footage (© NBA) | **decided by the user**: README shows the demo film as an inline player (user-attachments, a23877d) plus the `demo-durant` pre-release (1080p, 720p, cover). `docs/media/cover.jpg` stays in the repo, unreferenced |
| A2 | **Mossland voice** (MOSS TTS `voice_id 865098d7…`) | the default voice of the public pipeline; terms for public use not confirmed. The hackathon voice that cloned 于嘉 must **never** be used | confirm with MOSI that the Mossland voice may be the default in an open-source project and in published films |
| A3 | **Official data terms** (api.nba.cn PBP, box scores, xFG, gpShot, season tables; `identity/names_zh.py` calls `api.nba.cn` live for Chinese names) | names, xFG card and season stats depend on it; no data is shipped | decide: keep "bring your own data" (current), or drop the live api.nba.cn call, or get permission. Also: `narration/roster_zh.json` (Chinese player names) and `narration/fixtures/*.json` (event JSON of three benchmark clips) are derived from that data — keep, or replace with a generated roster |
| A4 | **Licence** | AGPL-3.0 is forced by Ultralytics YOLO11 / BoT-SORT and the court-33 model (no licence on its model card; AGPL assumed as a YOLO11 derivative) | keep AGPL-3.0 (current), or buy an Ultralytics enterprise licence / swap to Apache-2.0 detectors (DEIM, RT-DETR from other sources) for a permissive release |
| A5 | **Repository owner and name** | currently `jj-cheng25/AgentDunker2026` | move to an organisation (SII / Fudan / MOSI)? rename to `AgentDunker`? |
| A6 | **Repository description and topics** | the description was the hackathon's AWS AgentCore / CDK text | description **updated by the commander on 2026-11-09** (now mentions the silver award); **topics still empty** — suggest e.g. `basketball nba sports-analytics computer-vision multi-object-tracking 3d-visualization ai-commentary agent` |
| A7 | **PyPI** package `agentdunker` | `pip install agentdunker` instead of a clone; name availability to check | after the first public release |
| A8 | **Hugging Face Space** demo | biggest reach; needs A1/A2/A3 settled and a GPU or long CPU time; must not let visitors upload copyrighted footage to us | later; or a Space that only replays pre-computed contracts |
| A9 | **Logos**: `film/assets/promo/intro.html` and `outro.html` embed **3 base64 PNG logos in the order SII × Fudan × MOSI** (上海创智学院 × 复旦大学 × MOSI 模思; synced from M7 cf8ddb8 and later, identical to the demo cover) | trademark use in a public repo and in every film other people make | the user must confirm each institution's consent to logo use (repo is already public); else make them optional / replace with text |
| A10 | **Mixamo-derived avatar** `Xbot.glb` | downloaded from three.js examples (not committed); Mixamo terms forbid redistributing the raw character | fine as a download; for a Space or Docker image, consider our own avatar |
| A12 | **Commit author identity** | the author address becomes public with the history | ✅ **done 2026-11-10 (user decision)**: GitHub history rewritten to the GitHub noreply address (main 4957d81 → 4957d81, tag demo-durant 047f2aa → 047f2aa); the old commits can stay reachable by SHA on GitHub until GitHub's GC. New commits use the noreply identity |
| A13 | **GitHub Actions** run on every push | `ci.yml` must pass on GitHub's runners (not only locally); it installs CPU torch, runs ruff, `tools/release_scan.py` and pytest | ✅ green on the current main b8e9bd1 (run 42194693950), on 4957d81 (run 42333823105) and on defd8bb (run 42193554529); also green on 047f2aa, 53682cd, a23877d. 0ff953a was red (release scan vs the 4.8 MB replay.webp), fixed by defd8bb |
| A14 | **README "Awards and press"** (4957d81): Silver Award, AWS × NBA Broadcast Code 12H Hackathon; badge + `docs/media/award*.svg` link to the AWS China WeChat article `mp.weixin.qq.com/s/hhVN8YK1eZvnQ_MH6_-vlg` | the award name, the AWS / NBA names and the article link are public claims about third parties | user to confirm the wording matches the official announcement, and that the link stays reachable (WeChat articles can be withdrawn); keep AWS / NBA names factual, no logos |
| A11 | **Mosshub key** | was rejected (401) until 2026-11-06 ~04:30Z; the replaced key passes `agentdunker doctor --ping` | done (keep keys out of the repo) |

## B. Done before the first push (verified)

- [x] `python tools/release_scan.py`: no credentials, no footage, no weights, no file > 2 MB, no machine paths, in the
      tree **and** in the full history.
- [x] Weights, fonts, avatar, CLIP, Chromium are downloads with sha256 (`agentdunker download-models`).
- [x] `LICENSE` (AGPL-3.0) and `THIRD_PARTY_NOTICES.md` (code, models, fonts, deps, data, services).
- [x] No-key path produces a film (`--offline`), and `report.json` lists what degraded.
- [x] CI (`.github/workflows/ci.yml`): lint, scan, unit + synthetic smoke tests; no paid calls, no weight downloads.
- [x] First private push (047f2aa) — by the commander, with the user's approval.
- [ ] Each later push: user go-ahead via the commander (A-items block **public**, not private pushes).
- [x] GitHub Actions green on the latest pushed commit (A13: b8e9bd1, run 42194693950).

## C. Before making the repository public

- [ ] A1, A2, A3, A4, A9 decided (A12 done).
- [x] README demo in place (per A1): inline player (a23877d) + `demo-durant` release; awards section (4957d81, A14).
- [ ] Update the repository description (A6) and topics.
- [ ] Re-run `tools/release_scan.py`; re-sync modules at their final commits (`tools/sync_modules.py`), bump version, tag.
- [ ] Clean-machine install test once more from the public URL.
