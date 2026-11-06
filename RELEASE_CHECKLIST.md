# Release checklist

Target: GitHub `jj-cheng25/AgentDunker2026` — **public since 2026-11-06** (user decision). Commit 9eb3289 was pushed to `main` by the commander from the
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
| A6 | **Repository description** | the GitHub description still describes the hackathon's AWS AgentCore / CDK architecture — outdated | e.g. "NBA clip → narrated holographic 3D tactical breakdown film, fully automatic (Fudan × SII × MOSI)" |
| A7 | **PyPI** package `agentdunker` | `pip install agentdunker` instead of a clone; name availability to check | after the first public release |
| A8 | **Hugging Face Space** demo | biggest reach; needs A1/A2/A3 settled and a GPU or long CPU time; must not let visitors upload copyrighted footage to us | later; or a Space that only replays pre-computed contracts |
| A9 | **Logos**: `film/assets/promo/intro.html` and `outro.html` embed **3 base64 PNG logos (Fudan, SII, MOSI; synced from M7 — the new SII · Fudan · MOSI order of the demo cover is not yet in M7's promo pages)** | trademark use in a public repo and in every film other people make | the user must confirm logo usage before going public; else make them optional / replace with text |
| A10 | **Mixamo-derived avatar** `Xbot.glb` | downloaded from three.js examples (not committed); Mixamo terms forbid redistributing the raw character | fine as a download; for a Space or Docker image, consider our own avatar |
| A12 | **Commit author identity**: all commits are authored as `jj-cheng25 <zxhy6636@gmail.com>` | the e-mail becomes public with the repository (history included) | keep, or rewrite history to a GitHub noreply address *before* going public (after that a rewrite breaks clones) |
| A13 | **GitHub Actions** run on every push | `ci.yml` must pass on GitHub's runners (not only locally); it installs CPU torch, runs ruff, `tools/release_scan.py` and pytest | ✅ green on GitHub for 9eb3289, 53682cd and a23877d (runs 42164600950, 42172939579, 42174309216, 42178862710) |
| A11 | **Mosshub key** | was rejected (401) until 2026-11-06 ~04:30Z; the replaced key passes `agentdunker doctor --ping` | done (keep keys out of the repo) |

## B. Done before the first push (verified)

- [x] `python tools/release_scan.py`: no credentials, no footage, no weights, no file > 2 MB, no machine paths, in the
      tree **and** in the full history.
- [x] Weights, fonts, avatar, CLIP, Chromium are downloads with sha256 (`agentdunker download-models`).
- [x] `LICENSE` (AGPL-3.0) and `THIRD_PARTY_NOTICES.md` (code, models, fonts, deps, data, services).
- [x] No-key path produces a film (`--offline`), and `report.json` lists what degraded.
- [x] CI (`.github/workflows/ci.yml`): lint, scan, unit + synthetic smoke tests; no paid calls, no weight downloads.
- [x] First private push (9eb3289) — by the commander, with the user's approval.
- [ ] Each later push: user go-ahead via the commander (A-items block **public**, not private pushes).
- [x] GitHub Actions green on every pushed commit so far (A13).

## C. Before making the repository public

- [ ] A1, A2, A3, A4, A9, A12 decided.
- [ ] README demo GIF in place (per A1).
- [ ] Update the repository description (A6) and topics.
- [ ] Re-run `tools/release_scan.py`; re-sync modules at their final commits (`tools/sync_modules.py`), bump version, tag.
- [ ] Clean-machine install test once more from the public URL.
