#!/usr/bin/env python3
"""Copy the module packages from their ``cmd/*`` branches into this repo (re-runnable).

    python tools/sync_modules.py                 # sync every module at the commit pinned in MODULES
    python tools/sync_modules.py --commit m7=abc123 --commit m4=HEAD   # move a pin for this run
    python tools/sync_modules.py --check         # exit 1 if the tree differs from what a sync would write

Files are read from git objects (``git archive <commit> <package>``), never from a worktree, so the
module agents' uncommitted work is not picked up and their worktrees are not touched.

Module logic is never edited by hand in this repo. What has to change for a single installable
package (import paths, hard-coded /inspire paths, the model-provider layer, the in-process runner) is a
``Rule`` below. Every rule must match exactly ``count`` times, or the sync stops: an upstream change that
moves the patched code is noticed instead of silently skipped. ``SYNCED_FROM.json`` records the source
commit of every module and the rules applied to it.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "agentdunker"
DEFAULT_BASE = REPO.parent   # $B: the module worktrees adk-cmd-m1 … adk-cmd-m7 live next to this repo

# module key → (worktree dir, pinned commit, source package, target subpackage)
MODULES = {
    "m1": ("adk-cmd-m1", "4f7005f", "adk_tracking", "tracking"),
    "m2": ("adk-cmd-m2", "85772b7", "adk_court", "court"),
    "m3": ("adk-cmd-m3", "447097e", "adk_identity", "identity"),
    "m4": ("adk-cmd-m4", "bd52595", "adk_replay", "replay"),
    "m5": ("adk-cmd-m5", "e98e145", "adk_events", "events"),
    "m6": ("adk-cmd-m6", "dc2fdba", "adk_narration", "narration"),
    "m7": ("adk-cmd-m7", "cf8ddb8", "agentdunker", "film"),
}

# Development / benchmark helpers that read the private bench ($CMD) or are not used by the film.
EXCLUDE = {
    "m1": [],
    "m2": ["tools/court_errors.py", "tools/summarize.py"],
    "m3": ["bencheval.py", "sbbench.py"],
    "m4": ["bench.py", "check_scenes.py", "summary.py", "fixtures/", "web/models/Xbot.glb"],
    "m5": ["tools/bench.py", "tools/coverage.py"],
    "m6": ["compare_ref.py"],
    # fonts are downloaded by `agentdunker download-models` (NotoSansSC alone is 17 MB)
    "m7": ["assets/fonts/*.ttf", "README.md", ".gitignore"],
}
ALWAYS_EXCLUDE = ("__pycache__/", ".pyc")


@dataclass(frozen=True)
class Rule:
    module: str
    file: str              # path inside the target subpackage
    old: str
    new: str
    why: str
    count: int = 1
    regex: bool = False


GENERIC_WHY = "in-process runner: main() takes argv"

RULES: list[Rule] = [
    # ------------------------------------------------------------------ CLI entry points take argv
    *[Rule(m, "__main__.py", "def main() -> None:", "def main(argv: list[str] | None = None) -> None:", GENERIC_WHY)
      for m in ("m1", "m2", "m3", "m4", "m6")],
    *[Rule(m, "__main__.py", "ap.parse_args()", "ap.parse_args(argv)", GENERIC_WHY) for m in ("m1", "m2", "m3", "m4")],
    Rule("m6", "__main__.py", "args = ap.parse_args()", "args = ap.parse_args(argv)", GENERIC_WHY),

    # ------------------------------------------------------------------ hard-coded paths → configuration
    Rule("m1", "models.py",
         'Path(\n    "/inspire/qb-ilm/project/video-understanding/public/personal/jjcheng/work/nba_hackathon/"\n'
         '    "agentdunker-robust-e2e/studio/artifacts/models/person-reid/osnet-x025-msmt17/osnet_x0_25_msmt17.pth"\n)',
         'MODEL_DIR / "osnet_x0_25_msmt17.pth"  # agentdunker download-models',
         "no machine-specific weight path"),
    Rule("m1", "models.py", 'Path(__file__).resolve().parents[1] / "models"',
         'Path.home() / ".cache" / "agentdunker" / "models"', "weights live in the agentdunker model dir"),
    Rule("m2", "keypoints.py", 'Path(__file__).parent / "models" / "court-33-yolo11m-pose.onnx"',
         'Path(os.environ.get("ADK_MODEL_DIR", Path.home() / ".cache" / "agentdunker" / "models")) / "court-33-yolo11m-pose.onnx"',
         "weights live in the agentdunker model dir"),
    Rule("m3", "games.py",
         '"/inspire/qb-ilm/project/video-understanding/public/personal/jjcheng/work/nba_hackathon/data"',
         'Path.home() / ".cache" / "agentdunker" / "nba_data"', "official data dir is configurable"),
    Rule("m5", "official.py",
         '"/inspire/qb-ilm/project/video-understanding/public/personal/jjcheng/work/nba_hackathon/data"',
         'Path.home() / ".cache" / "agentdunker" / "nba_data"', "official data dir is configurable"),
    Rule("m5", "detector.py", 'Path(__file__).resolve().parent.parent / "models" / "deim_m_640.onnx"',
         'Path(os.environ.get("ADK_MODEL_DIR", Path.home() / ".cache" / "agentdunker" / "models")) / "deim_m_640.onnx"',
         "weights live in the agentdunker model dir"),
    Rule("m5", "pose.py", 'Path(__file__).resolve().parent.parent / "models" / "yolo11x-pose.pt"',
         'Path(os.environ.get("ADK_MODEL_DIR", Path.home() / ".cache" / "agentdunker" / "models")) / "yolo11x-pose.pt"',
         "weights live in the agentdunker model dir"),
    Rule("m7", "season.py", 'Path(__file__).resolve().parents[2] / "data" / "season"',
         'Path(os.environ.get("ADK_NBA_DATA", Path.home() / ".cache" / "agentdunker" / "nba_data")) / "season"',
         "season tables live in the official data dir"),
    Rule("m7", "render/style.py", 'Path(__file__).resolve().parents[1] / "assets" / "fonts"',
         '__import__("agentdunker.weights", fromlist=["x"]).fonts_dir()', "fonts are downloaded assets"),
    Rule("m7", "render/promo.py", 'FONTS = ASSETS / "fonts"',
         'FONTS = __import__("agentdunker.weights", fromlist=["x"]).fonts_dir()', "fonts are downloaded assets"),
    # Xbot.glb (three.js examples / Mixamo) is a downloaded asset, served from the model dir
    Rule("m4", "browser.py", "        super().do_GET()\n",
         "        from agentdunker.weights import serve_web_asset  # downloaded assets (Xbot.glb)\n"
         "        if not serve_web_asset(self, WEB_ROOT):\n"
         "            super().do_GET()\n", "Xbot.glb is a downloaded asset"),
    Rule("m4", "browser.py", 'CJK_FONT_CANDIDATES = (Path(__file__).resolve().parents[1] / "agent/fonts/NotoSansSC-Regular.otf",)',
         'CJK_FONT_CANDIDATES = (__import__("agentdunker.weights", fromlist=["x"]).fonts_dir() / "NotoSansSC-Regular.otf",)',
         "the CJK label font is a downloaded asset (Noto Sans CJK SC 2.004)"),
    Rule("m4", "browser.py", 'raise RuntimeError("chromium is required for the 3D replay renderer")',
         'from agentdunker.weights import chromium_path\n'
         '    if chromium_path():\n'
         '        return chromium_path()\n'
         '    raise RuntimeError("chromium is required for the 3D replay renderer (agentdunker download-models)")',
         "use the Playwright-managed Chromium"),
    Rule("m7", "render/promo.py", '    env = os.environ.get("ADK_CHROMIUM")\n    if env:\n        return env\n',
         '    from agentdunker.weights import chromium_path\n    if chromium_path():\n        return chromium_path()\n',
         "one Chromium lookup for the whole package"),

    # ------------------------------------------------------------------ always a film (M7 robustness)
    Rule("m7", "render/promo.py", '    return build_fixed() / f"{name}_score.wav"\n',
         '    try:\n        return build_fixed() / f"{name}_score.wav"\n'
         '    except Exception as e:  # noqa: BLE001  (no Chromium / fonts: the film goes on without intro and outro)\n'
         '        logger.warning("intro/outro unavailable: %s", str(e).splitlines()[0][:200])\n        return None\n',
         "no Chromium must drop intro/outro, not the film (build_mix accepts None)"),
    Rule("m7", "render/promo.py", 'def cover() -> Path:\n    return build_fixed() / "cover.png"\n',
         'def cover() -> Path:\n    try:\n        return build_fixed() / "cover.png"\n'
         '    except Exception:  # noqa: BLE001  (compose() embeds the cover only if it exists)\n'
         '        return Path("/nonexistent/cover.png")\n', "no Chromium must not fail the mux"),

    # ------------------------------------------------------------------ GPU when available (agentdunker.device)
    Rule("m1", "models.py", "        self.model.load_state_dict(state)\n",
         "        self.model.load_state_dict(state)\n        self.model.to(_device())\n", "ReID on GPU when available"),
    Rule("m1", "models.py", "f = self.model(x).numpy()", "f = self.model(x.to(_device())).cpu().numpy()",
         "ReID on GPU when available"),
    Rule("m1", "models.py", "CLIPModel.from_pretrained(CLIP_NAME).eval()", "CLIPModel.from_pretrained(CLIP_NAME).eval().to(_device())",
         "CLIP on GPU when available"),
    Rule("m1", "models.py", 'self.proc(text=texts, return_tensors="pt", padding=True)',
         'self.proc(text=texts, return_tensors="pt", padding=True).to(_device())', "CLIP on GPU when available"),
    Rule("m1", "models.py", 'self.proc(images=rgb, return_tensors="pt")', 'self.proc(images=rgb, return_tensors="pt").to(_device())',
         "CLIP on GPU when available"),
    Rule("m1", "models.py", ".softmax(-1).numpy()", ".softmax(-1).cpu().numpy()", "CLIP on GPU when available"),
    Rule("m1", "models.py", "import torch\n", "import torch\n\nfrom agentdunker.device import torch_device as _device\n",
         "device helper import"),
    Rule("m5", "detector.py", 'providers=["CPUExecutionProvider"]',
         'providers=__import__("agentdunker.device", fromlist=["x"]).ort_providers()', "ONNX detector on GPU when available"),

    # ------------------------------------------------------------------ model providers (one config)
    # M1 roles VLM, M3 jersey/scoreboard VLM, M5 shot checks, M6 script + TTS/ASR, M7 director + MOSS-VL
    *[Rule(m, "vlm.py", '    v = os.environ.get("MOSSHUB_API_KEY")\n    if v:\n        return v.strip()\n'
           '    p = Path(os.environ.get("MOSSHUB_API_KEY_FILE", Path.home() / ".config" / "mosshub" / "key"))\n'
           '    return p.read_text().strip() if p.exists() else None\n',
           '    from agentdunker.providers import llm_key\n    return llm_key()\n',
           "LLM/VLM credential from the configured provider") for m in ("m1", "m3", "m5")],
    *[Rule(m, "vlm.py", 'requests.post(f"{BASE_URL}/{endpoint}", json=payload, timeout=timeout,\n'
           '                              headers={"Authorization": f"Bearer {key}"})',
           '_providers.llm_post(endpoint, payload, timeout)', "LLM/VLM call through the configured provider")
      for m in ("m1",)],
    Rule("m3", "vlm.py", 'requests.post(f"{BASE_URL}/{endpoint}", json=payload, timeout=timeout,\n'
         '                                      headers={"Authorization": f"Bearer {key}"})',
         '_providers.llm_post(endpoint, payload, timeout)', "LLM/VLM call through the configured provider"),
    Rule("m5", "vlm.py", 'requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=timeout,\n'
         '                              headers={"Authorization": f"Bearer {key}"})',
         '_providers.llm_post("chat/completions", payload, timeout)', "LLM/VLM call through the configured provider"),
    *[Rule(m, "vlm.py", "import requests\n", "import requests\n\nfrom agentdunker import providers as _providers\n",
           "provider layer import") for m in ("m1", "m3", "m5")],
    Rule("m3", "vlm.py",
         '            v = os.environ.get("MOSS_API_KEY")\n'
         '            p = Path.home() / ".config" / "moss" / "key"\n'
         '            key = v.strip() if v else (p.read_text().strip() if p.exists() else None)\n',
         '            key = _providers.moss_key("vl")\n', "MOSS-VL credential from the configured provider"),
    Rule("m6", "moss.py",
         'requests.post(url, headers={"Authorization": "Bearer " + _key(key_env, key_path)}, timeout=timeout, **kw)',
         '_providers.moss_or_llm_post(url, key_env, timeout=timeout, **kw)',
         "TTS/ASR → MOSS provider, script chat → configured LLM provider"),
    Rule("m6", "moss.py", "import requests\n", "import requests\n\nfrom agentdunker import providers as _providers\n",
         "provider layer import"),
    Rule("m7", "models.py", "    v = os.environ.get(env)\n    if v:\n        return v.strip()\n"
         "    p = Path.home() / path\n    return p.read_text().strip() if p.exists() else None\n",
         "    from agentdunker.providers import llm_key, moss_key\n"
         "    return moss_key(\"vl\") if env == \"MOSS_API_KEY\" else llm_key()\n",
         "credentials from the configured provider"),
    Rule("m7", "models.py", 'requests.post(f"{MOSSHUB_URL}/chat/completions", json=payload, timeout=timeout,\n'
         '                                      headers={"Authorization": f"Bearer {key}"})',
         '__import__("agentdunker.providers", fromlist=["x"]).llm_post("chat/completions", payload, timeout)',
         "director LLM call through the configured provider"),

    # ------------------------------------------------------------------ M7 runner: single environment, in-process
    Rule("m7", "modules.py", "def run_module(", "def _subprocess_run_module(", "keep M7's runner as the isolate mode"),
    Rule("m7", "modules.py", 'def module_dir(mod: str) -> Path:', 'def _m7_module_dir(mod: str) -> Path:',
         "module dirs come from agentdunker.runner"),
    Rule("m7", "modules.py", 'def module_python(mod: str) -> str:', 'def _m7_module_python(mod: str) -> str:',
         "the interpreter comes from agentdunker.runner"),
    Rule("m7", "modules.py", "    d = module_dir(mod)\n", "    d = _m7_module_dir(mod)\n", "renamed helper"),
    Rule("m7", "modules.py", r"\Z",
         "\n\n# --- agentdunker: one environment; modules run in-process by default (agentdunker.runner) ---\n"
         "from agentdunker.runner import module_dir, module_python, run_module  # noqa: E402,F401\n",
         "in-process module calls, subprocess isolation optional", regex=True),
]


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def excluded(mod: str, rel: str) -> bool:
    if any(x in rel for x in ALWAYS_EXCLUDE):
        return True
    for pat in EXCLUDE[mod]:
        if pat.endswith("/") and rel.startswith(pat):
            return True
        if "*" in pat and re.fullmatch(re.escape(pat).replace(r"\*", "[^/]*"), rel):
            return True
        if rel == pat:
            return True
    return False


def rewrite_imports(text: str) -> str:
    for _, _, src, dst in (v for k, v in MODULES.items() if k != "m7"):
        text = re.sub(rf"\b{src}\b", f"agentdunker.{dst}", text)
    return text


def apply_rules(mod: str, files: dict[str, bytes]) -> list[dict]:
    applied = []
    for r in (r for r in RULES if r.module == mod):
        if r.file not in files:
            raise SystemExit(f"rule for {mod}/{r.file}: file not in the synced package ({r.why})")
        text = files[r.file].decode()
        n = len(re.findall(r.old, text)) if r.regex else text.count(r.old)
        if n != r.count:
            raise SystemExit(f"rule for {mod}/{r.file} matched {n}x (expected {r.count}): {r.why}\n  old: {r.old[:120]!r}")
        text = re.sub(r.old, lambda _m, new=r.new: new, text) if r.regex else text.replace(r.old, r.new)
        files[r.file] = text.encode()
        applied.append({"file": r.file, "why": r.why})
    return applied


def extract(base: Path, mod: str, commit: str) -> tuple[str, dict[str, bytes]]:
    wt, _, src, _ = MODULES[mod]
    repo = base / wt
    full = git(repo, "rev-parse", f"{commit}^{{commit}}").decode().strip()
    tar = tarfile.open(fileobj=io.BytesIO(git(repo, "archive", full, src)))
    files = {}
    for m in tar.getmembers():
        if not m.isfile():
            continue
        rel = m.name[len(src) + 1:]
        if excluded(mod, rel):
            continue
        data = tar.extractfile(m).read()
        if rel.endswith((".py", ".md", ".js", ".html")):
            data = rewrite_imports(data.decode()).encode()
        files[rel] = data
    return full, files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE, help="directory holding the adk-cmd-m* worktrees")
    ap.add_argument("--commit", action="append", default=[], help="mN=<rev> overrides a pinned commit")
    ap.add_argument("--check", action="store_true", help="only report whether the tree is up to date")
    a = ap.parse_args()
    pins = {k: v[1] for k, v in MODULES.items()}
    for c in a.commit:
        k, v = c.split("=", 1)
        pins[k] = v
    record = {"tool": "tools/sync_modules.py", "modules": {}}
    plan: dict[Path, bytes] = {}
    for mod, (wt, _, src, dst) in MODULES.items():
        full, files = extract(a.base, mod, pins[mod])
        applied = apply_rules(mod, files)
        digest = hashlib.sha256()
        for rel in sorted(files):
            plan[PKG / dst / rel] = files[rel]
            digest.update(rel.encode() + b"\0" + files[rel])
        record["modules"][dst] = {"module": mod, "branch_worktree": wt, "source_package": src, "commit": full,
                                  "files": len(files), "tree_sha256": digest.hexdigest()[:16], "rules": applied}
    plan[PKG / "SYNCED_FROM.json"] = (json.dumps(record, indent=1, ensure_ascii=False) + "\n").encode()
    targets = [PKG / v[3] for v in MODULES.values()]
    existing = {p for t in targets if t.exists() for p in t.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    stale = existing - set(plan)
    changed = [p for p, b in plan.items() if not p.exists() or p.read_bytes() != b]
    if a.check:
        for p in sorted(changed):
            print("differs:", p.relative_to(REPO))
        for p in sorted(stale):
            print("stale:  ", p.relative_to(REPO))
        return 1 if changed or stale else 0
    for p in stale:
        p.unlink()
    for p in changed:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(plan[p])
    for t in targets:                       # drop empty directories left by removed files
        for d in sorted((d for d in t.rglob("*") if d.is_dir()), reverse=True):
            if not any(d.iterdir()):
                shutil.rmtree(d)
    for dst, info in record["modules"].items():
        print(f"{dst:10s} {info['module']} {info['commit'][:8]}  {info['files']:3d} files  {len(info['rules'])} rules")
    print(f"{len(changed)} file(s) written, {len(stale)} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
