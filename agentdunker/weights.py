"""Model weights and binary assets: registry, ``agentdunker download-models``, checksums, licences.

Nothing here is in git. Files go to ``config.path("model_dir")`` (default ``~/.cache/agentdunker/models``),
fonts to ``<model_dir>/fonts``, CLIP to the Hugging Face cache, Chromium to ``<model_dir>/ms-playwright``.
Every file is checked against its sha256 after download; a mismatch is deleted and reported.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from . import config

logger = logging.getLogger(__name__)

UL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
GF = "https://raw.githubusercontent.com/google/fonts/24ddc1f324cac5d9f5aedca126d298f56e3f1640/ofl"
THREE = "https://raw.githubusercontent.com/mrdoob/three.js/0d6115556e60cb940b3063307549ea95a2dfe82b"


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    size: int
    license: str
    used_by: str
    kind: str = "model"            # model | font | web | hf
    presets: tuple = ("fast", "quality")

    def path(self) -> Path:
        base = config.path("model_dir")
        return base / "fonts" / self.name if self.kind == "font" else base / self.name


ASSETS: list[Asset] = [
    Asset("yolo11x.pt", f"{UL}/yolo11x.pt", "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
          114636239, "AGPL-3.0 (Ultralytics)", "tracking: person detector (quality preset)", presets=("quality",)),
    Asset("yolo11m.pt", f"{UL}/yolo11m.pt", "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
          40684120, "AGPL-3.0 (Ultralytics)", "tracking: person detector (fast preset)", presets=("fast",)),
    Asset("osnet_x0_25_msmt17.pth", "https://drive.google.com/uc?export=download&id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e",
          "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18", 9336983,
          "MIT (torchreid model zoo, KaiyangZhou/deep-person-reid)", "tracking: OSNet ReID appearance features"),
    Asset("court-33-yolo11m-pose.onnx",
          "https://huggingface.co/MTerryJack/Element-PitchCalib-Basketball-Court/resolve/"
          "81111962e7bfeb669cfbedad2ff455342dbb5529/weights.onnx",
          "9c00051bd207f5b76c6e492f3672be8dd7cce5a217ab271e971378fca057d26d", 86282807,
          "AGPL-3.0 assumed (YOLO11m-pose derivative; no licence on the model card; trained on Roboflow "
          "Universe basketball-court-detection-2)", "court: 33 court keypoints"),
    Asset("deim_m_640.onnx", "https://huggingface.co/ortizeg/basketball-deim-m-640/resolve/"
          "1b4f378bc1fa8d3a7980768d3b08486b6f3fe357/deim_m_640.onnx",
          "29f575c8127e5eadde6da60cd66c3d0a5873adc0cbfd4e2af9ee35fde339fac7", 77722583, "Apache-2.0",
          "events: ball / rim / player detector"),
    Asset("yolo11x-pose.pt", f"{UL}/yolo11x-pose.pt", "013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8",
          118481010, "AGPL-3.0 (Ultralytics)", "events: wrist keypoints at the release (quality preset)",
          presets=("quality",)),
    Asset("Xbot.glb", f"{THREE}/examples/models/gltf/Xbot.glb",
          "002f8d269de68e5dce3d25195caf390d1aa359bbfaae3fcf4c8dc78ec36c3ba5", 2930032,
          "three.js examples asset (MIT repository; character originally from Adobe Mixamo)",
          "replay: 3D player avatar", kind="web"),
    Asset("Montserrat-VF.ttf", f"{GF}/montserrat/Montserrat%5Bwght%5D.ttf",
          "0f7b311b2f3279e4eef9b2f968bcdbab6e28f4daeb1f049f4f278a902bcd82f7", 744936, "SIL OFL 1.1",
          "film: titles", kind="font"),
    Asset("Montserrat-Italic-VF.ttf", f"{GF}/montserrat/Montserrat-Italic%5Bwght%5D.ttf",
          "51607f316bc020e59f03cbf51543eecffbea501c0b31d73e5b82927c5cca442c", 762376, "SIL OFL 1.1",
          "film: titles", kind="font"),
    Asset("NotoSansSC-VF.ttf", f"{GF}/notosanssc/NotoSansSC%5Bwght%5D.ttf",
          "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da", 17772300, "SIL OFL 1.1",
          "film: Chinese text", kind="font"),
    Asset("BarlowCondensed-Bold.ttf", f"{GF}/barlowcondensed/BarlowCondensed-Bold.ttf",
          "e476562ec9c1e16cf16475895b511f08c804f438cc9a9f80a44ea50a0eeb5b65", 109912, "SIL OFL 1.1",
          "film: numbers", kind="font"),
    Asset("BarlowCondensed-BoldItalic.ttf", f"{GF}/barlowcondensed/BarlowCondensed-BoldItalic.ttf",
          "d8694f65cd70c3e5e600ead37657aab0adb5f63b275d0c1f6f4a7046de222145", 112196, "SIL OFL 1.1",
          "film: numbers", kind="font"),
]
HF_MODELS = [("openai/clip-vit-large-patch14", "MIT", "tracking: zero-shot player / referee / other roles")]
BY_NAME = {a.name: a for a in ASSETS}


def model_dir() -> Path:
    return config.path("model_dir")


def fonts_dir() -> Path:
    """Downloaded fonts; falls back to the packaged licence-only folder so imports never fail."""
    d = model_dir() / "fonts"
    return d if d.is_dir() else Path(__file__).resolve().parent / "film" / "assets" / "fonts"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(a: Asset, verify: bool = False) -> str:
    p = a.path()
    if not p.exists():
        return "missing"
    if p.stat().st_size != a.size:
        return "size mismatch"
    if verify and sha256(p) != a.sha256:
        return "checksum mismatch"
    return "ok"


def _fetch(url: str, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".part")
    if "drive.google.com" in url:
        try:
            import gdown                               # handles the Drive confirmation page
            gdown.download(url, str(tmp), quiet=True)
        except ImportError:
            urllib.request.urlretrieve(url, tmp)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "agentdunker-download"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dst)


def download(names: Optional[Iterable[str]] = None, preset: Optional[str] = None, force: bool = False,
             log=print) -> list[str]:
    """Download (and verify) assets; returns the names that failed."""
    todo = [BY_NAME[n] for n in names] if names else [a for a in ASSETS if not preset or preset in a.presets]
    failed = []
    for a in todo:
        p = a.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not force and status(a, verify=True) == "ok":
            log(f"  ok        {a.name}")
            continue
        log(f"  download  {a.name} ({a.size / 1e6:.1f} MB) ← {a.url.split('/')[2]}")
        try:
            _fetch(a.url, p)
        except Exception as e:  # noqa: BLE001
            log(f"  FAILED    {a.name}: {type(e).__name__}: {e}")
            failed.append(a.name)
            continue
        if sha256(p) != a.sha256:
            p.unlink()
            log(f"  FAILED    {a.name}: checksum mismatch (deleted)")
            failed.append(a.name)
    return failed


def download_hf(log=print) -> list[str]:
    failed = []
    for repo, _, _ in HF_MODELS:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo, allow_patterns=["*.json", "*.txt", "model.safetensors"])
            log(f"  ok        {repo} (Hugging Face cache)")
        except Exception as e:  # noqa: BLE001
            log(f"  FAILED    {repo}: {type(e).__name__}: {e}")
            failed.append(repo)
    return failed


def hf_cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- Chromium (3D replay, transition page)
def chromium_path() -> Optional[str]:
    env = os.environ.get("ADK_CHROMIUM")
    if env:
        return env
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH"), str(model_dir() / "ms-playwright"),
             str(Path.home() / ".cache" / "ms-playwright")]
    for r in filter(None, roots):
        found = sorted(glob.glob(f"{r}/chromium-*/chrome-linux*/chrome"))
        if found:
            return found[-1]
    for name in ("chromium", "chromium-browser", "google-chrome"):
        if shutil.which(name):
            return shutil.which(name)
    return None


def install_chromium(log=print) -> bool:
    env = {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": os.environ.get("PLAYWRIGHT_BROWSERS_PATH",
                                                                    str(model_dir() / "ms-playwright"))}
    log(f"  install   Chromium (Playwright) → {env['PLAYWRIGHT_BROWSERS_PATH']}")
    r = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], env=env)
    return r.returncode == 0 and chromium_path() is not None


def serve_web_asset(handler, web_root: Path) -> bool:
    """For the replay page's HTTP server: serve /models/<asset> from the model dir when not in web/."""
    from urllib.parse import urlsplit
    path = urlsplit(handler.path).path
    if not path.startswith("/models/") or (web_root / path.lstrip("/")).exists():
        return False
    a = BY_NAME.get(path.rsplit("/", 1)[-1])
    if a is None or not a.path().exists():
        return False
    data = a.path().read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", "model/gltf-binary" if a.name.endswith(".glb") else "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    return True
