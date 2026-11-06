# Third-party notices

AgentDunker is licensed under **AGPL-3.0** (see `LICENSE`), because the person detector / tracker (Ultralytics
YOLO11, BoT-SORT) and the court keypoint model it depends on are AGPL-3.0.

Nothing below marked *downloaded* is in this repository: `agentdunker download-models` fetches it from the
listed source and checks its sha256 (`agentdunker/weights.py` holds the exact URLs and checksums).

## Code included in this repository

| component | where | licence | source |
|---|---|---|---|
| OSNet model definition (torchreid) | `agentdunker/tracking/_vendor/osnet.py` | MIT, © 2018 Kaiyang Zhou (`_vendor/OSNET_LICENSE`) | KaiyangZhou/deep-person-reid @ f8cd150, unmodified |
| three.js r170 + addons (GLTFLoader, OrbitControls, postprocessing, SkeletonUtils, BufferGeometryUtils) | `agentdunker/replay/web/vendor/three/` | MIT, © 2010-2024 three.js authors (`vendor/three/LICENSE`) | mrdoob/three.js |
| Intro / outro / transition pages and sound scripts | `agentdunker/film/assets/promo/` | project code (AGPL-3.0); the intro/outro embed the logos of Fudan University, Shanghai Innovation Institute and MOSI (trademarks of their owners) | AgentDunker hackathon entry |

## Model weights and assets (downloaded)

| file | used for | licence | source |
|---|---|---|---|
| `yolo11x.pt` (quality) / `yolo11m.pt` (fast) | person detection + BoT-SORT tracking | AGPL-3.0 | Ultralytics assets v8.3.0 |
| `yolo11x-pose.pt` (quality only) | wrist keypoints at the shot release | AGPL-3.0 | Ultralytics assets v8.3.0 |
| `osnet_x0_25_msmt17.pth` | ReID appearance features | MIT | torchreid model zoo (Google Drive link in MODEL_ZOO.md) |
| `clip-vit-large-patch14/` | zero-shot player / referee / other roles | MIT | openai/clip-vit-large-patch14 @ 32bd642 (Hugging Face) |
| `court-33-yolo11m-pose.onnx` | 33 court keypoints → homography | **AGPL-3.0 assumed** (YOLO11m-pose derivative); the model card states no licence; training data: Roboflow Universe *basketball-court-detection-2* | MTerryJack/Element-PitchCalib-Basketball-Court @ 8111196 |
| `deim_m_640.onnx` | ball / rim / player detection | Apache-2.0 | ortizeg/basketball-deim-m-640 @ 1b4f378 |
| `Xbot.glb` | 3D player avatar | MIT (three.js repository); the character originates from Adobe Mixamo | mrdoob/three.js @ 0d61155 `examples/models/gltf/Xbot.glb` |
| Montserrat, Noto Sans SC, Barlow Condensed | film typography | SIL OFL 1.1 (texts in `agentdunker/film/assets/fonts/OFL-*.txt`) | google/fonts @ 24ddc1f |
| Noto Sans CJK SC 2.004 (`NotoSansSC-Regular.otf`) | Chinese labels in the 3D replay page | SIL OFL 1.1 | notofonts/noto-cjk @ Sans2.004 |
| Chromium (Playwright build) | headless rendering of the 3D replay and the transition page | BSD-3-Clause and others (Chromium) | `python -m playwright install chromium` |

## Python dependencies (installed by pip, not included)

| package | licence | | package | licence |
|---|---|---|---|---|
| ultralytics | AGPL-3.0 | | torch, torchvision | BSD-3-Clause |
| opencv-python | Apache-2.0 (+ bundled FFmpeg, LGPL) | | onnxruntime | MIT |
| transformers, huggingface_hub | Apache-2.0 | | numpy, scipy, scikit-learn, networkx | BSD-3-Clause |
| av (PyAV) | BSD-3-Clause (links FFmpeg, LGPL) | | pillow | MIT-CMU (HPND) |
| requests, websocket-client | Apache-2.0 | | soundfile | BSD-3-Clause |
| librosa | ISC | | pypinyin, pyloudnorm, gdown | MIT |
| playwright | Apache-2.0 | | lap | BSD-2-Clause |
| gradio (`[ui]` extra) | Apache-2.0 | | pytest (MIT), ruff (MIT) (`[dev]`) | |

System tool: **FFmpeg** (LGPL-2.1+, GPL when built with libx264, as most distributions do) is called as an external
program and is not distributed with AgentDunker.

## Not included, never to be committed

- **NBA game footage** (all broadcast clips, including the clips the examples and benchmark refer to): © NBA.
- **Official NBA data** (play-by-play, box scores, xFG, season tables from api.nba.cn / nba.com): used only when
  the user points `paths.nba_data` at their own copy; terms of use apply.
- **PL-NBA research dataset**: CC BY-NC 4.0 (non-commercial), not used by the film pipeline.

## Model services (called over the network when configured)

Mosshub (gemini-3.8-flash, gpt-6-astra), MOSS TTS / ASR / MOSS-VL (MOSI) and the Mossland commentary voice, or the
OpenAI / Google Gemini APIs if the user configures them, are governed by their providers' terms. No service is
needed to make a film: without credentials every model step falls back to local rules and templates.
