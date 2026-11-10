#!/usr/bin/env python3
"""Pre-push scan of everything git would publish (tracked files + full history).

    python tools/release_scan.py            # exit 1 on any finding

Checks: credentials (API-key shapes, private keys, key files), media/weights by extension and by magic bytes,
files over 2 MB, machine-specific paths, and known key values from the local environment (compared by hash,
never printed).
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_BYTES = 2 * 1024 * 1024
BAD_EXT = re.compile(r"\.(mp4|mov|mkv|avi|webm|wav|mp3|m4a|flac|pt|pth|onnx|safetensors|ckpt|bin|glb|ttf|otf|npz|pkl)$", re.I)
SECRET = [
    (re.compile(rb"sk-[A-Za-z0-9_\-]{20,}"), "API key (sk-…)"),
    (re.compile(rb"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(rb"AIza[0-9A-Za-z_\-]{35}"), "Google API key"),
    (re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"), "GitHub token"),
    (re.compile(rb"hf_[A-Za-z0-9]{30,}"), "Hugging Face token"),
    (re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(rb"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"), "hard-coded credential"),
]
PATHS = re.compile(rb"/inspire/|/root/\.config|nba_hackathon")
MAGIC = [(b"\x00\x00\x00", b"ftyp", "MP4/MOV video"), (b"\x1aE\xdf\xa3", None, "Matroska")]
RIFF_MEDIA = (b"WAVE", b"AVI ")                     # RIFF/WEBP is an image and fine
# Reviewed exceptions to the 2 MB rule, pinned by git blob id: a replaced file is flagged again.
ALLOW_LARGE = {
    # README animation: our own 3D render with the broadcast inset masked (4.8 MB)
    "docs/media/replay.webp": "fde68835809320fc33ea314e0ad5920bfd7620f3",
}
ALLOW_PATHS = {"tools/release_scan.py", "tools/sync_modules.py"}   # the scanner and the sync rules name them on purpose


def git(*a: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *a], check=True, capture_output=True).stdout


def local_key_hashes() -> set[str]:
    vals = [v for k, v in os.environ.items() if k.endswith(("API_KEY", "_TOKEN")) and len(v) >= 16]
    for p in ("~/.config/mosshub/key", "~/.config/moss/key"):
        f = Path(p).expanduser()
        if f.is_file():
            vals.append(f.read_text().strip())
    return {hashlib.sha256(v.encode()).hexdigest() for v in vals if v}


def main() -> int:
    findings: list[str] = []
    keys = local_key_hashes()
    files = git("ls-files", "-z").decode().split("\0")
    for rel in filter(None, files):
        p = REPO / rel
        if not p.is_file():
            continue
        data = p.read_bytes()
        allowed = ALLOW_LARGE.get(rel) == git("hash-object", str(p)).decode().strip()
        if len(data) > MAX_BYTES and not allowed:
            findings.append(f"{rel}: {len(data) / 1e6:.1f} MB (> 2 MB)")
        if BAD_EXT.search(rel):
            findings.append(f"{rel}: media/weights extension")
        if data.startswith(b"RIFF") and data[8:12] in RIFF_MEDIA:
            findings.append(f"{rel}: looks like RIFF {data[8:12].decode()} (audio/video)")
        if data.startswith(b"\x7fELF") or re.fullmatch(r"(.*/)?core(\.\d+)?", rel):
            findings.append(f"{rel}: ELF binary or core dump (may hold process memory)")
        for a, b, what in MAGIC:
            if data.startswith(a) and (b is None or data[4:8] == b):
                findings.append(f"{rel}: looks like {what}")
        for rx, what in SECRET:
            if rx.search(data):
                findings.append(f"{rel}: {what}")
        if rel not in ALLOW_PATHS and PATHS.search(data):
            findings.append(f"{rel}: machine-specific path")
        if keys and any(hashlib.sha256(tok).hexdigest() in keys for tok in re.findall(rb"[A-Za-z0-9_\-]{16,}", data)):
            findings.append(f"{rel}: contains a local credential value")
    # history: secrets or big blobs that were committed and later removed still get published
    for line in git("rev-list", "--objects", "--all").decode().splitlines():
        sha, _, name = line.partition(" ")
        if not name:
            continue
        size = int(git("cat-file", "-s", sha))
        if ALLOW_LARGE.get(name) == sha:
            continue
        if size > MAX_BYTES or BAD_EXT.search(name) or re.fullmatch(r"(.*/)?core(\.\d+)?", name):
            findings.append(f"history: {name} ({size / 1e6:.1f} MB)")
    hist = git("log", "--all", "-p", "--no-color")
    for rx, what in SECRET:
        if rx.search(hist):
            findings.append(f"history: {what} in a past diff")
    findings = sorted(set(findings))
    for f in findings:
        print("FOUND", f)
    print(f"scanned {len(files)} tracked files and full history: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
