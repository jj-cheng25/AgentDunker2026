"""Small, cached VLM checks on narrow frame strips (Mosshub; default gemini-3.8-flash).

The credential comes from MOSSHUB_API_KEY or MOSSHUB_API_KEY_FILE and is never logged. Responses are cached by
request hash, so a re-run never pays twice and never re-samples a frozen request.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import requests

from agentdunker import providers as _providers

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("MOSSHUB_BASE_URL", "https://api.mosshub.cn/v1").rstrip("/")
MODEL = os.environ.get("ADK_EVENTS_VLM_MODEL", "gemini-3.8-flash")
CACHE = Path(os.environ.get("ADK_VLM_CACHE", Path.home() / ".cache" / "agentdunker.events" / "vlm"))


def _key() -> str | None:
    from agentdunker.providers import llm_key
    return llm_key()


def strip(frames: list[np.ndarray], labels: list[str], tile_h: int = 260, cols: int = 6) -> np.ndarray:
    tiles = []
    for im, lab in zip(frames, labels):
        t = cv2.resize(im, (max(8, int(im.shape[1] * tile_h / im.shape[0])), tile_h))
        t = cv2.copyMakeBorder(t, 30, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, lab, (6, 23), 0, 0.7, (0, 0, 200), 2)
        tiles.append(t)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def ask_json(image: np.ndarray, prompt: str, schema: dict, timeout: float = 120.0,
             model: str | None = None) -> dict | None:
    model = model or MODEL
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    payload = {"model": model,
               "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                         {"type": "text", "text": prompt}]}],
               "response_format": {"type": "json_schema", "json_schema": {"name": "agentdunker.events", "schema": schema}}}
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cached = CACHE / f"{h}.json"
    if cached.exists():
        body = json.loads(cached.read_text())
    else:
        key = _key()
        if not key:
            logger.info("vlm: no credential; skipping check")
            return None
        try:
            r = _providers.llm_post("chat/completions", payload, timeout)
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("vlm %s: request failed (%s)", model, type(e).__name__)
            return None
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(body))
    try:
        ans = json.loads(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("vlm %s: unparseable answer", model)
        return None
    ans["_cache"] = h
    ans["_model"] = model
    return ans
