"""HTTP clients: MOSS TTS / ASR (api.mosi.cn) and the Mosshub chat model.

Keys come from MOSS_API_KEY or ~/.config/moss/key, and MOSSHUB_API_KEY or
~/.config/mosshub/key. Keys are never logged or written to outputs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import requests

from agentdunker import providers as _providers

from . import audio, files

logger = logging.getLogger(__name__)

MOSS_API = os.environ.get("ADK_MOSS_API", "https://api.mosi.cn/v1")
MOSSHUB_API = os.environ.get("MOSSHUB_BASE_URL", "https://api.mosshub.cn/v1").rstrip("/")
# The MOSS platform exposes no dated snapshot for flash; the model id and response
# headers are recorded per take so a later drift can be detected.
TTS_MODEL = os.environ.get("ADK_TTS_MODEL", "moss-tts-1.5-flash")
ASR_MODEL = os.environ.get("ADK_ASR_MODEL", "moss-transcribe-diarize-pro")
VOICE_ID = os.environ.get("ADK_VOICE_ID", "865098d7-ca3f-4463-945e-783425f655b3")  # Mossland 世界杯激情体育足球解说
CHAT_MODEL = os.environ.get("ADK_CHAT_MODEL", "gemini-3.8-flash")
# MOSS answers HTTP 429 above a few concurrent requests; all MOSS calls share this cap (at most 3).
MOSS_CONCURRENCY = max(1, min(3, int(os.environ.get("ADK_MOSS_CONCURRENCY", "3"))))
_MOSS_SLOTS = threading.BoundedSemaphore(MOSS_CONCURRENCY)


def _key(env: str, path: str) -> str:
    value = os.environ.get(env)
    if not value:
        p = Path(path).expanduser()
        if not p.exists():
            raise RuntimeError(f"missing credential: set {env} or write {path}")
        value = p.read_text().strip()
    return value


def _post(url: str, key_env: str, key_path: str, retries: int = 7, timeout: int = 180, **kw) -> requests.Response:
    last = None
    gate = _MOSS_SLOTS if url.startswith(MOSS_API) else threading.BoundedSemaphore(1)
    for attempt in range(retries):
        limited = False
        try:
            with gate:
                r = _providers.moss_or_llm_post(url, key_env, timeout=timeout, **kw)
            if r.status_code == 200:
                return r
            limited = r.status_code == 429
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except requests.RequestException as exc:
            last = type(exc).__name__
        logger.warning("retry %s (%s)", url.rsplit("/", 1)[-1], last.strip())
        time.sleep((4 + 4 * attempt) if limited else (2 + 3 * attempt))
    raise RuntimeError(f"request failed: {url} ({last})")


def tts(text: str, out: Path, expected: float | None = None) -> dict:
    body: dict[str, object] = {"model": TTS_MODEL, "input": text, "voice_id": VOICE_ID, "language": "Chinese",
            "response_format": "wav", "delivery_method": "audio"}
    if expected:
        body["expected_duration_sec"] = round(float(expected), 2)
    for attempt in range(3):
        r = _post(f"{MOSS_API}/audio/speech", "MOSS_API_KEY", "~/.config/moss/key", json=body)
        if r.content[:4] == b"RIFF":
            # Written under a temp name and renamed only once it decodes, so a kill or a full
            # disk never leaves a truncated take under the cached name.
            try:
                with files.atomic_path(out) as tmp:
                    tmp.write_bytes(r.content)
                    if not audio.wav_ok(tmp):
                        raise ValueError(f"undecodable wav ({len(r.content)} bytes)")
                return {"model": TTS_MODEL, "voice": VOICE_ID, "request_id": r.headers.get("x-request-id", "")}
            except (OSError, ValueError) as exc:
                logger.warning("tts take not saved, retrying: %s", exc)
        time.sleep(2)
    raise RuntimeError("tts returned no audio")


def asr(path: Path, keyterms: list[str] | None = None, cache_dir: Path | None = None) -> str:
    digest = hashlib.sha1(Path(path).read_bytes() + json.dumps(keyterms or [], ensure_ascii=False).encode()).hexdigest()[:16]
    cache = cache_dir / f"asr_{digest}.json" if cache_dir else None
    hit = files.read_json(cache) if cache else None
    if isinstance(hit, dict) and isinstance(hit.get("text"), str):
        return hit["text"]
    data = {"model": ASR_MODEL, "response_format": "json"}
    if keyterms:
        data["keyterms"] = json.dumps(keyterms[:20], ensure_ascii=False)
    with open(path, "rb") as f:
        payload = f.read()
    r = _post(f"{MOSS_API}/audio/transcriptions", "MOSS_API_KEY", "~/.config/moss/key",
              files={"file": (Path(path).name, payload)}, data=data, timeout=300)
    text = r.json().get("text", "")
    if cache:
        files.write_text(cache, json.dumps({"text": text}, ensure_ascii=False))
    return text


def chat_json(system: str, user: str, temperature: float = 0.7) -> dict:
    """One JSON-returning chat call on Mosshub. Raises on non-JSON output."""
    body = {"model": CHAT_MODEL, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}}
    r = _post(f"{MOSSHUB_API}/chat/completions", "MOSSHUB_API_KEY", "~/.config/mosshub/key", json=body, timeout=240)
    content = r.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content)
