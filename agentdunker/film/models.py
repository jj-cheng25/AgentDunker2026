"""Model calls used by the director: Mosshub ``gemini-3.8-flash`` (text/JSON) and MOSS-VL (image choice).

Credentials: ``MOSSHUB_API_KEY`` or ``~/.config/mosshub/key``; ``MOSS_API_KEY`` or ``~/.config/moss/key``.
They are never written to logs, caches or outputs. Every answer is cached by request hash
(``$ADK_CACHE`` or ``~/.cache/agentdunker``) and every call is recorded in ``LEDGER`` for report.json.
A failure returns ``None``; callers fall back to deterministic rules.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

MOSSHUB_URL = os.environ.get("MOSSHUB_BASE_URL", "https://api.mosshub.cn/v1").rstrip("/")
MOSS_URL = os.environ.get("MOSS_BASE_URL", "https://api.mosi.cn/v1").rstrip("/")
TEXT_MODEL = os.environ.get("ADK_TEXT_MODEL", "gemini-3.8-flash")
MOSS_VL_MODEL = os.environ.get("ADK_MOSS_VL_MODEL", "moss-vl-1.0")
CACHE = Path(os.environ.get("ADK_CACHE", Path.home() / ".cache" / "agentdunker"))
LEDGER: list[dict] = []
_lock = threading.Lock()


def _key(env: str, path: str) -> Optional[str]:
    from agentdunker.providers import llm_key, moss_key
    return moss_key("vl") if env == "MOSS_API_KEY" else llm_key()


def _cached(kind: str, payload: dict) -> tuple[Path, Optional[dict]]:
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    p = CACHE / kind / f"{h}.json"
    return p, (json.loads(p.read_text()) if p.exists() else None)


def _record(model: str, tag: str, cached: bool, ok: bool, seconds: float) -> None:
    with _lock:
        LEDGER.append({"model": model, "tag": tag, "cached": cached, "ok": ok, "seconds": round(seconds, 2)})


def ledger_summary(entries: Optional[list[dict]] = None) -> dict:
    out: dict = {}
    for e in entries if entries is not None else LEDGER:
        m = out.setdefault(e["model"], {"calls": 0, "live": 0, "cached": 0, "failed": 0, "seconds": 0.0, "tags": {}})
        m["calls"] += 1
        m["live" if not e["cached"] else "cached"] += 1
        m["failed"] += not e["ok"]
        m["seconds"] = round(m["seconds"] + e["seconds"], 2)
        m["tags"][e["tag"]] = m["tags"].get(e["tag"], 0) + 1
    return out


def _json_text(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def ask_json(prompt: str, schema: dict, tag: str, model: str = TEXT_MODEL, timeout: float = 120) -> Optional[dict]:
    """Text-only structured answer from Mosshub; None on any failure."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "response_format": {"type": "json_schema", "json_schema": {"name": tag.replace("-", "_"), "schema": schema}}}
    path, body = _cached("text", payload)
    t0 = time.time()
    was_cached = body is not None
    try:
        if body is None:
            key = _key("MOSSHUB_API_KEY", ".config/mosshub/key")
            if not key:
                return None
            for attempt in range(3):
                try:
                    r = __import__("agentdunker.providers", fromlist=["x"]).llm_post("chat/completions", payload, timeout)
                    r.raise_for_status()
                    body = r.json()
                    break
                except (requests.RequestException, ValueError) as e:
                    logger.warning("%s %s attempt %d failed: %s", model, tag, attempt + 1, type(e).__name__)
                    time.sleep(3 * (attempt + 1))
            if body is None:
                _record(model, tag, False, False, time.time() - t0)
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body, ensure_ascii=False))
        ans = _json_text(body["choices"][0]["message"]["content"])
        _record(model, tag, was_cached, True, time.time() - t0)
        return ans
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("%s %s: unparseable answer", model, tag)
        _record(model, tag, was_cached, False, time.time() - t0)
        return None


def _jpeg(img: np.ndarray, max_w: int = 1280) -> bytes:
    im = Image.fromarray(img)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def moss_vl(prompt: str, images: list[np.ndarray], tag: str, timeout: float = 180) -> Optional[str]:
    """MOSS-VL (official endpoint, Responses API with uploaded files). Returns raw text or None."""
    blobs = [_jpeg(im) for im in images]
    fp = {"model": MOSS_VL_MODEL, "prompt": prompt, "images": [hashlib.sha256(b).hexdigest() for b in blobs]}
    path, body = _cached("moss_vl", fp)
    t0 = time.time()
    if body is not None:
        _record(MOSS_VL_MODEL, tag, True, True, time.time() - t0)
        return body["text"]
    key = _key("MOSS_API_KEY", ".config/moss/key")
    if not key:
        return None
    headers = {"Authorization": f"Bearer {key}"}
    try:
        with requests.Session() as s:
            content = [{"type": "input_text", "text": prompt}]
            for i, b in enumerate(blobs):
                r = s.post(f"{MOSS_URL}/files", headers=headers, data={"purpose": "image"},
                           files={"file": (f"img{i}.jpg", b, "image/jpeg")}, timeout=(15, 120))
                r.raise_for_status()
                content.append({"type": "input_image", "file_id": r.json()["id"]})
            r = s.post(f"{MOSS_URL}/responses", headers=headers, timeout=(15, timeout),
                       json={"model": MOSS_VL_MODEL, "input": [{"role": "user", "content": content}],
                             "max_output_tokens": 512})
            r.raise_for_status()
            out = r.json()
        text = "".join(p.get("text", "") for it in out.get("output", []) if it.get("type") == "message"
                       for p in it.get("content", []) if p.get("type") == "output_text")
        if not text.strip():
            raise ValueError("empty answer")
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("moss-vl %s failed: %s", tag, type(e).__name__)
        _record(MOSS_VL_MODEL, tag, False, False, time.time() - t0)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text, "model": out.get("model")}, ensure_ascii=False))
    _record(MOSS_VL_MODEL, tag, False, True, time.time() - t0)
    return text
