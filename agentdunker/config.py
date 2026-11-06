"""One configuration for the whole package: defaults < config file < environment < command line.

Config file (TOML), first found: ``$ADK_CONFIG``, ``./agentdunker.toml``, ``~/.config/agentdunker/config.toml``.
See ``examples/agentdunker.toml`` for every key. Keys are never stored in the config file itself: the file
names the environment variable (or key file) that holds each credential.

``apply()`` exports the settings as the environment variables the analysis modules read at import time
(model names, cache and data directories, thread caps), so it must run before any of them is imported.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import tomllib

HOME_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "agentdunker"

# Per-provider defaults. Every value can be overridden in [llm] or by ADK_LLM_* variables.
PROVIDERS: dict[str, dict[str, Any]] = {
    "mosshub": {   # default: Mosshub serves gemini-3.8-flash and gpt-6-astra behind one OpenAI-format endpoint
        "base_url": "https://api.mosshub.cn/v1", "key_env": "MOSSHUB_API_KEY",
        "key_file": "~/.config/mosshub/key", "fast_model": "gemini-3.8-flash", "strong_model": "gpt-6-astra",
        "responses_api": True,
    },
    "openai": {    # api.openai.com or any OpenAI-compatible server (vLLM, Ollama, LM Studio …) via base_url
        "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "key_file": "",
        "fast_model": "gpt-5-mini", "strong_model": "gpt-5", "responses_api": True,
    },
    "gemini": {    # Google's official OpenAI-compatible Gemini endpoint (chat completions only)
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY",
        "key_file": "", "fast_model": "gemini-3.8-flash", "strong_model": "gemini-3.8-flash",
        "responses_api": False,
    },
    "none": {"base_url": "", "key_env": "", "key_file": "", "fast_model": "", "strong_model": "",
             "responses_api": False},
}

DEFAULTS: dict[str, Any] = {
    "llm": {"provider": "mosshub"},
    "moss": {   # MOSS (MOSI) official endpoint: TTS + ASR (commentary voice) and MOSS-VL (small visual checks)
        "base_url": "https://api.mosi.cn/v1", "key_env": "MOSS_API_KEY", "key_file": "~/.config/moss/key",
        "tts_model": "moss-tts-1.5-flash", "asr_model": "moss-transcribe-diarize-pro", "vl_model": "moss-vl-1.0",
        "voice_id": "865098d7-ca3f-4463-945e-783425f655b3",      # Mossland 世界杯激情体育足球解说
        "vl_checks": True,
    },
    "voice": {"backend": "moss"},          # moss | none (subtitles only)
    "paths": {"model_dir": str(HOME_CACHE / "models"), "cache_dir": str(HOME_CACHE / "cache"),
              "nba_data": str(HOME_CACHE / "nba_data")},
    "run": {"preset": "quality", "threads": 16, "runner": "inprocess", "device": "auto"},
}

ENV = {   # environment variable → (section, key)
    "ADK_LLM_PROVIDER": ("llm", "provider"), "ADK_LLM_BASE_URL": ("llm", "base_url"),
    "ADK_LLM_FAST_MODEL": ("llm", "fast_model"), "ADK_LLM_STRONG_MODEL": ("llm", "strong_model"),
    "ADK_LLM_KEY_ENV": ("llm", "key_env"),
    "ADK_VOICE": ("voice", "backend"), "ADK_MOSS_BASE_URL": ("moss", "base_url"),
    "ADK_VOICE_ID": ("moss", "voice_id"),
    "ADK_MODEL_DIR": ("paths", "model_dir"), "ADK_CACHE": ("paths", "cache_dir"), "ADK_NBA_DATA": ("paths", "nba_data"),
    "ADK_PRESET": ("run", "preset"), "ADK_THREADS": ("run", "threads"), "ADK_RUNNER": ("run", "runner"),
    "ADK_DEVICE": ("run", "device"),
}


@dataclass
class LLM:
    provider: str
    base_url: str
    key_env: str
    key_file: str
    fast_model: str
    strong_model: str
    responses_api: bool


def config_file() -> Optional[Path]:
    for p in (os.environ.get("ADK_CONFIG"), "agentdunker.toml", Path.home() / ".config" / "agentdunker" / "config.toml"):
        if p and Path(p).expanduser().is_file():
            return Path(p).expanduser()
    return None


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@lru_cache(maxsize=1)
def load() -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    f = config_file()
    if f:
        cfg = _merge(cfg, tomllib.loads(f.read_text()))
        cfg["_file"] = str(f)
    for env, (sec, key) in ENV.items():
        if os.environ.get(env):
            cfg.setdefault(sec, {})[key] = os.environ[env]
    cfg["run"]["threads"] = max(1, int(cfg["run"]["threads"]))
    _OVERRIDES.clear()
    return cfg


def override(**kv: Any) -> None:
    """Command-line overrides, e.g. override(run={"preset": "fast"}); clears the cache of derived values."""
    cfg = _merge(get(), {k: v for k, v in kv.items() if v})
    _OVERRIDES.clear()
    _OVERRIDES.update(cfg)
    llm.cache_clear()


_OVERRIDES: dict = {}


def get() -> dict:
    return _OVERRIDES or load()


@lru_cache(maxsize=1)
def llm() -> LLM:
    c = get()["llm"]
    name = str(c.get("provider", "mosshub")).lower()
    if name not in PROVIDERS:
        raise ValueError(f"unknown llm provider {name!r} (one of {', '.join(PROVIDERS)})")
    p = {**PROVIDERS[name], **{k: v for k, v in c.items() if k != "provider" and v not in (None, "")}}
    return LLM(name, str(p["base_url"]).rstrip("/"), p["key_env"], p["key_file"], p["fast_model"],
               p["strong_model"], bool(p["responses_api"]))


def path(name: str) -> Path:
    return Path(get()["paths"][name]).expanduser()


def threads() -> int:
    return int(get()["run"]["threads"])


def apply() -> None:
    """Export the settings for the analysis modules (they read os.environ at import time)."""
    c, l, m = get(), llm(), get()["moss"]
    n = str(threads())
    cache = path("cache_dir")
    env = {
        # thread caps (torch / onnxruntime / OpenCV / BLAS)
        "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "ADK_THREADS": n,
        "ADK_DEVICE": str(c["run"].get("device", "auto")),
        # weights and data
        "ADK_MODEL_DIR": str(path("model_dir")), "ADK_NBA_DATA": str(path("nba_data")),
        "ADK_COURT_MODEL": str(path("model_dir") / "court-33-yolo11m-pose.onnx"),
        # model names: the modules put these into their requests; the provider layer sends them as they are
        "ADK_VLM_MODEL": ",".join(x for x in dict.fromkeys((l.strong_model, l.fast_model)) if x),
        "ADK_ID_CHEAP_MODEL": l.fast_model, "ADK_ID_STRONG_MODEL": l.strong_model or l.fast_model,
        "ADK_EVENTS_VLM_MODEL": l.fast_model, "ADK_CHAT_MODEL": l.fast_model, "ADK_TEXT_MODEL": l.fast_model,
        # MOSS (TTS / ASR / MOSS-VL)
        "MOSS_BASE_URL": m["base_url"].rstrip("/"), "ADK_MOSS_API": m["base_url"].rstrip("/"),
        "ADK_TTS_MODEL": m["tts_model"], "ADK_ASR_MODEL": m["asr_model"], "ADK_VOICE_ID": m["voice_id"],
        "ADK_MOSS_VL_MODEL": m["vl_model"], "ADK_ID_MOSS_VL_MODEL": m["vl_model"],
        # answer caches (keyed by request hash: a re-run never pays twice)
        "ADK_CACHE": str(cache / "director"), "ADK_VLM_CACHE": str(cache / "vlm"),
        "ADK_ID_CACHE": str(cache / "identity"), "ADK_ID_INDEX": str(cache / "identity" / "pbp_compact.json.gz"),
        "PLAYWRIGHT_BROWSERS_PATH": os.environ.get("PLAYWRIGHT_BROWSERS_PATH", str(path("model_dir") / "ms-playwright")),
    }
    if l.provider == "mosshub":
        env["MOSSHUB_BASE_URL"] = l.base_url
    clip = path("model_dir") / "clip-vit-large-patch14"
    if (clip / "model.safetensors").is_file():    # downloaded by `agentdunker download-models`
        env["ADK_CLIP_MODEL"] = str(clip)
    if c["paths"].get("hf_home"):              # optional: where CLIP is cached (default: Hugging Face default)
        env["HF_HOME"] = str(Path(c["paths"]["hf_home"]).expanduser())
    os.environ.update(env)
    try:                                   # torch may not be importable in a minimal environment
        import torch
        torch.set_num_threads(threads())
    except Exception:  # noqa: BLE001
        pass


def public() -> dict:
    """The effective configuration without credentials (for report.json and `doctor`)."""
    c, l = get(), llm()
    return {"file": c.get("_file"), "llm": {"provider": l.provider, "base_url": l.base_url, "fast_model": l.fast_model,
                                            "strong_model": l.strong_model, "key_env": l.key_env},
            "voice": c["voice"]["backend"], "moss": {k: v for k, v in c["moss"].items() if k != "key_file"},
            "paths": c["paths"], "run": c["run"]}
