"""Mosshub VLM transport: images + prompt + JSON schema -> parsed dict, cached by request hash.

The credential is read from MOSSHUB_API_KEY or ~/.config/mosshub/key and never logged. Every call is
recorded in `LEDGER` (model, cached or live, seconds) so the caller can report paid calls per clip.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from agentdunker import providers as _providers

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("MOSSHUB_BASE_URL", "https://api.mosshub.cn/v1").rstrip("/")
CHEAP = os.environ.get("ADK_ID_CHEAP_MODEL", "gemini-3.8-flash")
STRONG = os.environ.get("ADK_ID_STRONG_MODEL", "gpt-6-astra")
MOSS_URL = os.environ.get("MOSS_BASE_URL", "https://api.mosi.cn/v1").rstrip("/")
MOSS_VL = os.environ.get("ADK_ID_MOSS_VL_MODEL", "moss-vl-1.0")
CACHE = Path(os.environ.get("ADK_ID_CACHE", Path.home() / ".cache" / "agentdunker.identity" / "vlm"))

LEDGER: list[dict] = []
_lock = threading.Lock()


def _key() -> str | None:
    from agentdunker.providers import llm_key
    return llm_key()


def _strict(schema: dict) -> dict:
    """OpenAI strict structured output: every object closed and fully required."""
    out = dict(schema)
    if out.get("type") == "object":
        out["properties"] = {k: _strict(v) for k, v in out["properties"].items()}
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if "items" in out:
        out["items"] = _strict(out["items"])
    return out


def _url(img: np.ndarray, quality: int = 90) -> str:
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def _payload(model: str, urls: list[str], prompt: str, schema: dict, name: str) -> tuple[str, dict]:
    """(endpoint, body): Responses API for gpt-* models, chat completions otherwise."""
    if model.startswith("gpt-"):
        content = [{"type": "input_image", "image_url": u} for u in urls] + [{"type": "input_text", "text": prompt}]
        return "responses", {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": name, "schema": _strict(schema), "strict": True}},
        }
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls] + [{"type": "text", "text": prompt}]
    return "chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
    }


def _answer(body: dict) -> str:
    if "choices" in body:
        return body["choices"][0]["message"]["content"]
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    return "".join(part.get("text", "") for item in body.get("output", []) if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def ask(images: list[np.ndarray], prompt: str, schema: dict, model: str = CHEAP, name: str = "agentdunker.identity",
        tag: str = "", timeout: float = 180.0, retries: int = 2) -> dict | None:
    """Return the parsed JSON answer, or None (no credential, HTTP failure, unparseable answer)."""
    urls = [_url(im) for im in images]
    endpoint, payload = _payload(model, urls, prompt, schema, name)
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cached = CACHE / f"{h}.json"
    t0 = time.time()
    entry = {"model": model, "tag": tag, "cached": cached.exists(), "ok": False}
    try:
        if cached.exists():
            body = json.loads(cached.read_text())
        else:
            key = _key()
            if not key:
                logger.info("vlm: no credential")
                return None
            body = None
            for attempt in range(retries + 1):
                try:
                    r = _providers.llm_post(endpoint, payload, timeout)
                    r.raise_for_status()
                    body = r.json()
                    break
                except (requests.RequestException, ValueError) as e:
                    logger.warning("vlm %s %s: request failed (%s), attempt %d", model, tag, type(e).__name__, attempt + 1)
                    entry["attempts"] = attempt + 1
                    time.sleep(3 * (attempt + 1))
            if body is None:
                return None
            CACHE.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(body))
        try:
            ans = _parse(_answer(body))
        except (KeyError, IndexError, TypeError, ValueError):
            logger.warning("vlm %s %s: unparseable answer", model, tag)
            return None
        entry["ok"] = True
        return ans
    finally:
        entry["seconds"] = round(time.time() - t0, 2)
        with _lock:
            LEDGER.append(entry)


def ask_moss(image: np.ndarray, prompt: str, tag: str = "", timeout: float = 120.0, retries: int = 1) -> dict | None:
    """MOSS-VL (official endpoint: upload the image, then the Responses API) on one image; the first JSON object
    in the answer, or None. Same cache, ledger and credential rules as `ask` (MOSS_API_KEY or ~/.config/moss/key)."""
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    blob = jpg.tobytes()
    fp = {"model": MOSS_VL, "prompt": prompt, "image": hashlib.sha256(blob).hexdigest()}
    cached = CACHE / f"{hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()}.json"
    t0 = time.time()
    entry = {"model": MOSS_VL, "tag": tag, "cached": cached.exists(), "ok": False}
    try:
        if cached.exists():
            text = json.loads(cached.read_text())["text"]
        else:
            key = _providers.moss_key("vl")
            if not key:
                logger.info("moss-vl: no credential")
                return None
            text = None
            for attempt in range(retries + 1):
                try:
                    with requests.Session() as s:
                        h = {"Authorization": f"Bearer {key}"}
                        r = s.post(f"{MOSS_URL}/files", headers=h, data={"purpose": "image"},
                                   files={"file": ("img.jpg", blob, "image/jpeg")}, timeout=(15, 60))
                        r.raise_for_status()
                        content = [{"type": "input_text", "text": prompt},
                                   {"type": "input_image", "file_id": r.json()["id"]}]
                        r = s.post(f"{MOSS_URL}/responses", headers=h, timeout=(15, timeout),
                                   json={"model": MOSS_VL, "input": [{"role": "user", "content": content}],
                                         "max_output_tokens": 512})
                        r.raise_for_status()
                        text = _answer(r.json())
                    break
                except (requests.RequestException, ValueError, KeyError) as e:
                    logger.warning("moss-vl %s: request failed (%s), attempt %d", tag, type(e).__name__, attempt + 1)
                    time.sleep(3 * (attempt + 1))
            if not text:
                return None
            CACHE.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({"text": text}))
        i, j = text.find("{"), text.rfind("}")
        # MOSS-VL sometimes leaves a clock unquoted ("gameClock":1:50); quote it, nothing else is changed
        body = re.sub(r':\s*(\d{1,2}:\d{2}(?:\.\d+)?)\s*(?=[,}])', r': "\1"', text[i:j + 1]) if 0 <= i < j else ""
        try:
            ans = json.loads(body) if body else None
        except ValueError:
            ans = None
        if not isinstance(ans, dict):
            logger.warning("moss-vl %s: unparseable answer", tag)
            return None
        entry["ok"] = True
        return ans
    finally:
        entry["seconds"] = round(time.time() - t0, 2)
        with _lock:
            LEDGER.append(entry)


def ledger_summary(entries: list[dict]) -> dict:
    out: dict = {}
    for e in entries:
        m = out.setdefault(e["model"], {"calls": 0, "live": 0, "cached": 0, "failed": 0, "seconds": 0.0})
        m["calls"] += 1
        m["cached" if e["cached"] else "live"] += 1
        m["failed"] += not e["ok"]
        m["seconds"] = round(m["seconds"] + e["seconds"], 1)
    return out
