# Configuration

Precedence: built-in defaults < config file < environment variables < command-line flags.
Config file (TOML), first found: `$ADK_CONFIG`, `./agentdunker.toml`, `~/.config/agentdunker/config.toml`.
Full example: [`examples/agentdunker.toml`](../examples/agentdunker.toml). `agentdunker config` prints the
effective settings (never credentials).

## Model providers

| `[llm] provider` | endpoint | key | default models (fast / strong) |
|---|---|---|---|
| `mosshub` (default) | `https://api.mosshub.cn/v1` | `MOSSHUB_API_KEY` or `~/.config/mosshub/key` | gemini-3.8-flash / gpt-6-astra |
| `openai` | `https://api.openai.com/v1`, or any OpenAI-compatible `base_url` (vLLM, Ollama, LM Studio …) | `OPENAI_API_KEY` | gpt-5-mini / gpt-5 |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai` (Google's official OpenAI-compatible API) | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | gemini-3.8-flash / gemini-3.8-flash |
| `none` | — | — | everything falls back to local rules and templates |

The models must accept images (jersey crops, contact sheets). *fast* is used for most calls; *strong* for the
few hard reads (one role adjudication per clip, ≤ 20 jersey re-reads). Set `responses_api = false` for an
OpenAI-compatible server without `/responses`.

Environment overrides: `ADK_LLM_PROVIDER`, `ADK_LLM_BASE_URL`, `ADK_LLM_FAST_MODEL`, `ADK_LLM_STRONG_MODEL`,
`ADK_LLM_KEY_ENV`; command line: `--provider`.

## Voice and MOSS-VL

`[voice] backend = "moss"` (default) uses MOSS TTS with the Mossland commentary voice and MOSS ASR read-back
(`MOSS_API_KEY` or `~/.config/moss/key`). Without a key, or with `backend = "none"` / `--no-voice`, the same
script is written (LLM if available, else templates that only state measured facts) and shown as subtitles in the
3D replay; the live play keeps the broadcast sound. `[moss] vl_checks = false` turns off the MOSS-VL checks.

## What degrades without which key

| missing | effect (recorded in `report.json` → `agentdunker.services_missing` and `elements`) |
|---|---|
| LLM | tracking: team/role split from CLIP + colour only; identity: no jersey numbers (no names); director: template phase titles; narration: template lines |
| MOSS voice | subtitles instead of voice |
| MOSS-VL | shot-type veto and xFG-card read-back skipped |
| official data (`paths.nba_data`) | no game match → no names, no official xFG card, no season stat on the hero plate |

## Paths, threads, presets, device

| key | env | default |
|---|---|---|
| `paths.model_dir` | `ADK_MODEL_DIR` | `~/.cache/agentdunker/models` |
| `paths.cache_dir` | `ADK_CACHE` | `~/.cache/agentdunker/cache` (model answers, intro/outro renders) |
| `paths.nba_data` | `ADK_NBA_DATA` | `~/.cache/agentdunker/nba_data` (empty) |
| `run.preset` | `ADK_PRESET`, `--preset` | `quality` |
| `run.threads` | `ADK_THREADS`, `--threads` | 16 |
| `run.runner` | `ADK_RUNNER`, `--isolate` | `inprocess` |
| `run.device` | `ADK_DEVICE`, `--device` | `auto` (CUDA when torch sees a GPU) |

Downloads behind the Great Firewall: `export HF_ENDPOINT=https://hf-mirror.com` before `agentdunker download-models`
(CLIP and the Hugging Face weights then come from the mirror; checksums are unchanged).
