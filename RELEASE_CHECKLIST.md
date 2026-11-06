# Release checklist

Target: GitHub `jj-cheng25/AgentDunker2026` (private, empty). First push: private, for review. **Every push, and
making the repository public, needs the user's explicit go-ahead.**

## A. Decisions for the user

| # | item | why it matters | options / recommendation |
|---|---|---|---|
| A1 | **Demo material** (README GIF, screenshots, sample film) | every film contains NBA broadcast footage (© NBA); the README demo slot is a placeholder | (a) no footage: 3D replay stills only (our renders, no broadcast frame) + architecture diagram; (b) a short clip under a fair-use/commentary argument (legal risk); (c) ask the NBA / a rights holder. Recommendation: (a) for launch, the 3D replay is the most distinctive frame anyway |
| A2 | **Mossland voice** (MOSS TTS `voice_id 865098d7…`) | the default voice of the public pipeline; terms for public use not confirmed. The hackathon voice that cloned 于嘉 must **never** be used | confirm with MOSI that the Mossland voice may be the default in an open-source project and in published films |
| A3 | **Official data terms** (api.nba.cn PBP, box scores, xFG, gpShot, season tables; `identity/names_zh.py` calls `api.nba.cn` live for Chinese names) | names, xFG card and season stats depend on it; no data is shipped | decide: keep "bring your own data" (current), or drop the live api.nba.cn call, or get permission. Also: `narration/roster_zh.json` (Chinese player names) and `narration/fixtures/*.json` (event JSON of three benchmark clips) are derived from that data — keep, or replace with a generated roster |
| A4 | **Licence** | AGPL-3.0 is forced by Ultralytics YOLO11 / BoT-SORT and the court-33 model (no licence on its model card; AGPL assumed as a YOLO11 derivative) | keep AGPL-3.0 (current), or buy an Ultralytics enterprise licence / swap to Apache-2.0 detectors (DEIM, RT-DETR from other sources) for a permissive release |
| A5 | **Repository owner and name** | currently `jj-cheng25/AgentDunker2026` | move to an organisation (Fudan / SII / MOSI)? rename to `AgentDunker`? |
| A6 | **Repository description** | the current GitHub description is the hackathon's AWS AgentCore architecture — outdated | e.g. "NBA clip → narrated holographic 3D tactical breakdown film, fully automatic (Fudan × SII × MOSI)" |
| A7 | **PyPI** package `agentdunker` | `pip install agentdunker` instead of a clone; name availability to check | after the first public release |
| A8 | **Hugging Face Space** demo | biggest reach; needs A1/A2/A3 settled and a GPU or long CPU time; must not let visitors upload copyrighted footage to us | later; or a Space that only replays pre-computed contracts |
| A9 | **Logos** in the intro/outro (Fudan, SII, MOSI embedded in `film/assets/promo/*.html`) | trademark use in a public repo and in films other people make | confirm each institution agrees; else make them optional |
| A10 | **Mixamo-derived avatar** `Xbot.glb` | downloaded from three.js examples (not committed); Mixamo terms forbid redistributing the raw character | fine as a download; for a Space or Docker image, consider our own avatar |
| A11 | **Mosshub key** | currently rejected (HTTP 401) on cpu-0; LLM features (jersey numbers, names) cannot be demonstrated until it is replaced | replace the key |

## B. Done before the first push (verified)

- [x] `python tools/release_scan.py`: no credentials, no footage, no weights, no file > 2 MB, no machine paths, in the
      tree **and** in the full history.
- [x] Weights, fonts, avatar, CLIP, Chromium are downloads with sha256 (`agentdunker download-models`).
- [x] `LICENSE` (AGPL-3.0) and `THIRD_PARTY_NOTICES.md` (code, models, fonts, deps, data, services).
- [x] No-key path produces a film (`--offline`), and `report.json` lists what degraded.
- [x] CI (`.github/workflows/ci.yml`): lint, scan, unit + synthetic smoke tests; no paid calls, no weight downloads.
- [ ] User go-ahead for the push (A-items do not block a **private** push, they block **public**).

## C. Before making the repository public

- [ ] A1, A2, A3, A4, A9 decided.
- [ ] README demo GIF in place (per A1).
- [ ] Update the repository description (A6) and topics.
- [ ] Re-run `tools/release_scan.py`; re-sync modules at their final commits (`tools/sync_modules.py`), bump version, tag.
- [ ] Clean-machine install test once more from the public URL.
