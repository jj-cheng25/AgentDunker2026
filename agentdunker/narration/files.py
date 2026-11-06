"""Atomic writes and tolerant cache reads.

A run killed mid-write (or a disk outage) must not leave a half-written file under the final
name, because takes, ASR results and scripts are reused from disk on the next run.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def atomic_path(path: str | Path) -> Iterator[Path]:
    """Yield a temp path next to `path` (same suffix, so ffmpeg infers the format); it replaces
    `path` only if the block finishes without raising. The temp file is removed otherwise."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part" + path.suffix)
    os.close(fd)
    tmp = Path(name)
    tmp.chmod(0o644)   # mkstemp creates 0600; outputs are read by other modules
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_bytes(path: str | Path, data: bytes) -> Path:
    with atomic_path(path) as tmp, open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return Path(path)


def write_text(path: str | Path, text: str) -> Path:
    return write_bytes(path, text.encode("utf-8"))


def copy(src: str | Path, dst: str | Path) -> Path:
    with atomic_path(dst) as tmp:
        shutil.copy2(src, tmp)
    return Path(dst)


def read_json(path: str | Path) -> Any | None:
    """Cached JSON, or None when it is missing or unreadable (the caller recomputes)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable cache %s (%s)", path.name, type(exc).__name__)
        return None
