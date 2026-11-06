"""Every LLM / VLM / TTS / ASR request of the package goes through here.

Two kinds of service:

* **LLM** (text + vision chat; one provider for the whole run, see ``config.PROVIDERS``): the role
  adjudication in tracking, jersey and scoreboard reading in identity, the shot checks in events, the
  commentary script, the director's phase titles. Wire format is OpenAI chat completions (or the
  Responses API for ``gpt-*`` models when the provider has it). Default: Mosshub.
* **MOSS** (MOSI official endpoint): the commentary voice (TTS + ASR read-back) and MOSS-VL checks.

No credential → the functions return ``None`` (or raise ``RuntimeError`` where the caller expects that)
and every caller falls back to local rules; nothing here makes the film fail. Credentials are read
from the environment or a key file, and never logged, cached or written to any output.
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Optional

import requests

from . import config

logger = logging.getLogger(__name__)


def _read_key(env: str, key_file: str) -> Optional[str]:
    if env and os.environ.get(env, "").strip():
        return os.environ[env].strip()
    if key_file:
        p = Path(key_file).expanduser()
        if p.is_file():
            v = p.read_text().strip()
            return v or None
    return None


def llm_key() -> Optional[str]:
    l = config.llm()
    if l.provider == "none":
        return None
    key = _read_key(l.key_env, l.key_file)
    if key is None and l.provider == "gemini":
        key = _read_key("GOOGLE_API_KEY", "")
    if key is None and l.provider == "openai" and "api.openai.com" not in l.base_url:
        key = os.environ.get("OPENAI_API_KEY") or "none"     # local OpenAI-compatible servers often need no key
    return key


def moss_key(purpose: str = "speech") -> Optional[str]:
    """MOSS credential for ``speech`` (TTS/ASR) or ``vl`` (MOSS-VL); None when that use is switched off."""
    c = config.get()
    if purpose == "speech" and c["voice"]["backend"] != "moss":
        return None
    if purpose == "vl" and not c["moss"].get("vl_checks", True):
        return None
    m = c["moss"]
    return _read_key(m.get("key_env", "MOSS_API_KEY"), m.get("key_file", ""))


def available() -> dict:
    """Which services this run can use (for doctor and report.json)."""
    l = config.llm()
    return {"llm": {"provider": l.provider, "ok": llm_key() is not None, "fast_model": l.fast_model,
                    "strong_model": l.strong_model},
            "voice": {"backend": config.get()["voice"]["backend"], "ok": moss_key("speech") is not None},
            "moss_vl": {"ok": moss_key("vl") is not None}}


# ---------------------------------------------------------------- LLM requests
def responses_to_chat(payload: dict) -> dict:
    """OpenAI Responses-API body → chat-completions body (for providers without /responses)."""
    content = []
    for msg in payload.get("input", []):
        for part in msg.get("content", []) if isinstance(msg.get("content"), list) else [
                {"type": "input_text", "text": msg.get("content", "")}]:
            if part.get("type") == "input_image":
                content.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
            elif part.get("type") == "input_text":
                content.append({"type": "text", "text": part["text"]})
    body: dict[str, Any] = {"model": payload["model"], "messages": [{"role": "user", "content": content}]}
    fmt = (payload.get("text") or {}).get("format") or {}
    if fmt.get("type") == "json_schema":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": fmt.get("name", "answer"), "schema": fmt["schema"]}}
    if "max_output_tokens" in payload:
        body["max_tokens"] = payload["max_output_tokens"]
    return body


def llm_post(endpoint: str, payload: dict, timeout: float) -> requests.Response:
    """POST an OpenAI-format request to the configured LLM provider.

    ``endpoint`` is ``chat/completions`` or ``responses``. Raises ``requests.RequestException`` on transport
    errors (callers already handle it) and ``RuntimeError`` when no provider/credential is configured.
    """
    l = config.llm()
    key = llm_key()
    if key is None:
        raise RuntimeError("no LLM provider configured")
    body = payload
    if endpoint == "responses" and not l.responses_api:
        endpoint, body = "chat/completions", responses_to_chat(payload)
    if l.provider == "gemini" and "response_format" in body:
        body = copy.deepcopy(body)
        _gemini_schema(body["response_format"].get("json_schema", {}).get("schema"))
    headers = {"Authorization": f"Bearer {key}"} if key != "none" else {}
    return requests.post(f"{l.base_url}/{endpoint}", json=body, timeout=timeout, headers=headers)


def _gemini_schema(schema: Any) -> None:
    """Gemini's OpenAI layer rejects JSON-schema type unions (["string", "null"]): use nullable instead."""
    if isinstance(schema, dict):
        t = schema.get("type")
        if isinstance(t, list):
            rest = [x for x in t if x != "null"]
            schema["type"] = rest[0] if rest else "string"
            if "null" in t:
                schema["nullable"] = True
        for v in schema.values():
            _gemini_schema(v)
    elif isinstance(schema, list):
        for v in schema:
            _gemini_schema(v)


def moss_or_llm_post(url: str, key_env: str, timeout: float = 180, **kw) -> requests.Response:
    """Narration transport: MOSS TTS/ASR stay on MOSS; the script chat goes to the configured LLM."""
    if key_env == "MOSSHUB_API_KEY":
        endpoint = url.rstrip("/").rsplit("/v1/", 1)[-1] if "/v1/" in url else "chat/completions"
        return llm_post(endpoint, kw["json"], timeout)
    key = moss_key("speech")
    if key is None:
        raise RuntimeError("missing credential: MOSS voice is not configured (MOSS_API_KEY)")
    return requests.post(url, headers={"Authorization": f"Bearer {key}"}, timeout=timeout, **kw)


def ping(timeout: float = 15) -> dict:
    """One authenticated ``GET /models`` per configured service: is the key accepted? (no model call, no cost)"""
    out = {}
    l = config.llm()
    key = llm_key()
    if key is not None:
        out["llm"] = _ping(f"{l.base_url}/models", key, timeout)
    m = config.get()["moss"]
    mk = _read_key(m.get("key_env", "MOSS_API_KEY"), m.get("key_file", ""))
    if mk is not None:
        out["moss"] = _ping(f"{m['base_url'].rstrip('/')}/models", mk, timeout)
    return out


def _ping(url: str, key: str, timeout: float) -> str:
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {key}"} if key != "none" else {}, timeout=timeout)
    except requests.RequestException as e:
        return f"unreachable ({type(e).__name__})"
    return {200: "ok", 401: "key rejected (401)", 403: "key rejected (403)"}.get(r.status_code, f"HTTP {r.status_code}")
