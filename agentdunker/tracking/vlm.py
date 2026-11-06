"""One VLM call per clip: label numbered tracklet crops as team1 / team2 / referee / other.

The model only judges given candidates (no coordinates). Responses are cached by request hash so a
re-run never pays twice. The credential is read from the environment / key file and never logged.
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
# One adjudication call per clip. gpt-6-astra is the default because it labels the tiles more reliably
# (c01/c12 IDF1 0.925/0.863 vs 0.899/0.842 with gemini-3.8-flash); gemini is the automatic fallback.
MODELS = tuple(os.environ.get("ADK_VLM_MODEL", "gpt-6-astra,gemini-3.8-flash").split(","))
CACHE = Path(os.environ.get("ADK_VLM_CACHE", Path.home() / ".cache" / "agentdunker.tracking" / "vlm"))
LABELS = ("team1", "team2", "referee", "other", "mixed")

PROMPT = """These numbered tiles show people tracked in one NBA broadcast clip: two crops per tile, taken at
different times, each centred on the tracked person (others may overlap). Classify every number by the
centred person:
- "team1" / "team2": a player on the court for one of the two teams (group by jersey colour),
- "referee": a game official,
- "other": anyone else (coaches, bench players, staff, fans, photographers),
- "mixed": the two crops are centred on different people.
Use the same team labels consistently. Describe each team's jersey colours.
For players and referees also give the jersey / shirt number if you can read it clearly, else null.
Never guess a number."""

SCHEMA = {
    "type": "object",
    "properties": {
        "team1_jersey": {"type": "string"},
        "team2_jersey": {"type": "string"},
        "people": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "label": {"type": "string", "enum": list(LABELS)},
            "number": {"type": ["string", "null"]}},
            "required": ["id", "label", "number"]}},
    },
    "required": ["team1_jersey", "team2_jersey", "people"],
}


def _key() -> str | None:
    from agentdunker.providers import llm_key
    return llm_key()


def contact_sheet(tiles: list[list[np.ndarray]], tile_h: int = 220, cols: int = 6) -> np.ndarray:
    cells = []
    for n, crops in enumerate(tiles, 1):
        row = [cv2.resize(c, (max(8, int(c.shape[1] * tile_h / c.shape[0])), tile_h)) for c in crops]
        cell = np.hstack(row)
        cell = cv2.copyMakeBorder(cell, 34, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(cell, str(n), (6, 28), 0, 0.9, (0, 0, 200), 2)
        cells.append(cell)
    w = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in cells]
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    return np.vstack([np.hstack(cells[i : i + cols]) for i in range(0, len(cells), cols)])


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


def _payload(model: str, url: str, prompt: str) -> tuple[str, dict]:
    """(endpoint, body): Responses API for gpt-* models, chat completions otherwise."""
    if model.startswith("gpt-"):
        return "responses", {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": url},
                                                   {"type": "input_text", "text": prompt}]}],
            "text": {"format": {"type": "json_schema", "name": "adk_roles", "schema": _strict(SCHEMA), "strict": True}},
        }
    return "chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                  {"type": "text", "text": prompt}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "adk_roles", "schema": SCHEMA}},
    }


def _answer(body: dict) -> str:
    if "choices" in body:
        return body["choices"][0]["message"]["content"]
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    return "".join(part.get("text", "") for item in body.get("output", []) if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")


def label_tiles(sheet: np.ndarray, n: int, timeout: float = 180.0) -> dict | None:
    """Try each configured model in order; None if all fail (the caller then stays local-only)."""
    for model in MODELS:
        res = _label_tiles(sheet, n, timeout, model.strip())
        if res is not None:
            return res
    return None


def _label_tiles(sheet: np.ndarray, n: int, timeout: float, model: str) -> dict | None:
    """Return {"labels": {tile: label}, "numbers": {tile: str}, jersey descriptions} or None on failure."""
    ok, jpg = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    endpoint, payload = _payload(model, url, PROMPT + f"\nNumbers: 1..{n}.")
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cached = CACHE / f"{h}.json"
    if cached.exists():
        body = json.loads(cached.read_text())
    else:
        key = _key()
        if not key:
            logger.info("vlm: no credential, using local role/team only")
            return None
        try:
            r = _providers.llm_post(endpoint, payload, timeout)
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("vlm %s: request failed (%s)", model, type(e).__name__)
            return None
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(body))
    try:
        ans = json.loads(_answer(body))
        labels = {int(p["id"]): p["label"] for p in ans["people"] if p.get("label") in LABELS and 1 <= int(p["id"]) <= n}
        numbers = {int(p["id"]): str(p["number"]).strip().lstrip("#") for p in ans["people"]
                   if p.get("number") not in (None, "", "null") and 1 <= int(p["id"]) <= n
                   and str(p["number"]).strip().lstrip("#").isdigit()}
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("vlm %s: unparseable answer", model)
        return None
    logger.info("vlm %s: %d/%d tiles labelled, %d numbers (%s | %s)", model, len(labels), n, len(numbers),
                ans.get("team1_jersey"), ans.get("team2_jersey"))
    return {"labels": labels, "numbers": numbers, "team1_jersey": ans.get("team1_jersey"), "team2_jersey": ans.get("team2_jersey"),
            "cache": h, "model": model}
