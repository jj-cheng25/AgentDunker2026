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
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from . import config

logger = logging.getLogger(__name__)

UL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
GF = "https://raw.githubusercontent.com/google/fonts/24ddc1f324cac5d9f5aedca126d298f56e3f1640/ofl"
CLIP_DIR = "clip-vit-large-patch14"
CLIP_REV = "32bd64288804d66eefd0ccbe215aa642df71cc41"
THREE = "https://raw.githubusercontent.com/mrdoob/three.js/0d6115556e60cb940b3063307549ea95a2dfe82b"


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    size: int
    license: str
    used_by: str
    kind: str = "model"            # model | font | web | clip
    presets: tuple = ("fast", "quality")
    mirrors: tuple = ()            # tried in order after ``url``; same bytes (sha256-checked)

    def path(self) -> Path:
        base = config.path("model_dir")
        sub = {"font": "fonts", "clip": CLIP_DIR}.get(self.kind)
        return base / sub / self.name if sub else base / self.name


ASSETS: list[Asset] = [
    Asset("yolo11x.pt", f"{UL}/yolo11x.pt", "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
          114636239, "AGPL-3.0 (Ultralytics)", "tracking: person detector (quality preset)", presets=("quality",)),
    Asset("yolo11m.pt", f"{UL}/yolo11m.pt", "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
          40684120, "AGPL-3.0 (Ultralytics)", "tracking: person detector (fast preset)", presets=("fast",)),
    # The author's Hugging Face copy is byte-identical to the torchreid model-zoo file on Google Drive and works
    # through HF_ENDPOINT mirrors; Drive stays as the fallback.
    Asset("osnet_x0_25_msmt17.pth", "hf://kaiyangzhou/osnet/resolve/a5c5cc037c24235cda3b21085b93ad77c9616224/"
          "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
          "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18", 9336983,
          "MIT (torchreid model zoo, KaiyangZhou/deep-person-reid)", "tracking: OSNet ReID appearance features",
          mirrors=("https://drive.google.com/uc?export=download&id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e",)),
    Asset("court-33-yolo11m-pose.onnx",
          "hf://MTerryJack/Element-PitchCalib-Basketball-Court/resolve/"
          "81111962e7bfeb669cfbedad2ff455342dbb5529/weights.onnx",
          "9c00051bd207f5b76c6e492f3672be8dd7cce5a217ab271e971378fca057d26d", 86282807,
          "AGPL-3.0 assumed (YOLO11m-pose derivative; no licence on the model card; trained on Roboflow "
          "Universe basketball-court-detection-2)", "court: 33 court keypoints"),
    Asset("deim_m_640.onnx", "hf://ortizeg/basketball-deim-m-640/resolve/"
          "1b4f378bc1fa8d3a7980768d3b08486b6f3fe357/deim_m_640.onnx",
          "29f575c8127e5eadde6da60cd66c3d0a5873adc0cbfd4e2af9ee35fde339fac7", 77722583, "Apache-2.0",
          "events: ball / rim / player detector"),
    Asset("yolo11x-pose.pt", f"{UL}/yolo11x-pose.pt", "013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8",
          118481010, "AGPL-3.0 (Ultralytics)", "events: wrist keypoints at the release"),
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
    Asset("NotoSansSC-Regular.otf", "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/Sans/OTF/"
          "SimplifiedChinese/NotoSansCJKsc-Regular.otf",
          "2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b", 16437364, "SIL OFL 1.1",
          "replay: Chinese labels in the 3D page (Noto Sans CJK SC 2.004)", kind="font"),
    Asset("BarlowCondensed-Bold.ttf", f"{GF}/barlowcondensed/BarlowCondensed-Bold.ttf",
          "e476562ec9c1e16cf16475895b511f08c804f438cc9a9f80a44ea50a0eeb5b65", 109912, "SIL OFL 1.1",
          "film: numbers", kind="font"),
    Asset("BarlowCondensed-BoldItalic.ttf", f"{GF}/barlowcondensed/BarlowCondensed-BoldItalic.ttf",
          "d8694f65cd70c3e5e600ead37657aab0adb5f63b275d0c1f6f4a7046de222145", 112196, "SIL OFL 1.1",
          "film: numbers", kind="font"),
]
# CLIP ViT-L/14 (OpenAI, MIT): zero-shot player / referee / other roles. Plain files at a pinned revision, so a
# Hugging Face mirror (HF_ENDPOINT, e.g. https://hf-mirror.com) works without the hub client.
_CLIP = {"config.json": ("8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a", 4519),
         "merges.txt": ("9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a", 524619),
         "preprocessor_config.json": ("910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd", 316),
         "special_tokens_map.json": ("f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf", 389),
         "tokenizer.json": ("a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4", 2224003),
         "tokenizer_config.json": ("deef455e52fa5e8151e339add0582e4235f066009601360999d3a9cda83b1129", 905),
         "vocab.json": ("3f0c4f7d2086b61b38487075278ea9ed04edb53a03cbb045b86c27190fa8fb69", 961143),
         "model.safetensors": ("a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb", 1710540580)}
ASSETS += [Asset(n, f"hf://openai/clip-vit-large-patch14/resolve/{CLIP_REV}/{n}", h, sz, "MIT (OpenAI CLIP)",
                 "tracking: zero-shot player / referee / other roles (CLIP ViT-L/14)", kind="clip")
           for n, (h, sz) in _CLIP.items()]
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


def clip_dir() -> Path:
    return model_dir() / CLIP_DIR


RETRIES = 5          # per source: resumed (HTTP Range) after a dropped connection
UA = {"User-Agent": "agentdunker-download"}


def _resolve(url: str) -> str:
    if url.startswith("hf://"):     # Hugging Face file: honours HF_ENDPOINT (e.g. https://hf-mirror.com)
        return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/" + url[5:]
    return url


def _stream(url: str, tmp: Path, size: int, sleep=time.sleep) -> None:
    """One file into ``tmp``, resuming from what is already there; raises unless exactly ``size`` bytes arrived."""
    for attempt in range(RETRIES + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if size and have > size:
            tmp.unlink()
            have = 0
        if size and have == size:
            return
        try:
            hdr = dict(UA, **({"Range": f"bytes={have}-"} if have else {}))
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
                mode = "ab" if have and r.status == 206 else "wb"        # 200: server ignored Range → start over
                with open(tmp, mode) as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            got = tmp.stat().st_size
            if not size or got == size:
                return
            raise OSError(f"connection closed at {got} of {size} bytes")
        except OSError as e:          # URLError, HTTPError, IncompleteRead, timeouts are all OSError subclasses
            if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404, 410):
                raise
            if attempt == RETRIES:
                raise
            logger.info("download %s: %s; resuming (%d/%d)", tmp.name, e, attempt + 1, RETRIES)
            sleep(min(30, 3 * (attempt + 1)))
    raise OSError("unreachable")


def _fetch(url: str, dst: Path, size: int = 0) -> None:
    url = _resolve(url)
    tmp = dst.with_name(dst.name + ".part")
    if "drive.google.com" in url:
        import gdown  # handles the Drive confirmation page
        for attempt in range(RETRIES + 1):
            if gdown.download(url, str(tmp), quiet=True, resume=True) and (not size or tmp.stat().st_size == size):
                break
            if attempt == RETRIES:
                raise OSError("Google Drive download failed")
            time.sleep(3 * (attempt + 1))
    elif size >= 200_000_000:
        _fetch_ranges(url, tmp, size)
    else:
        _stream(url, tmp, size)
    os.replace(tmp, dst)


def _fetch_ranges(url: str, tmp: Path, size: int, parts: int = 8) -> None:
    """Large files: ``parts`` parallel HTTP range requests (mirrors often throttle each connection)."""
    from concurrent.futures import ThreadPoolExecutor
    step = -(-size // parts)
    pieces = [tmp.with_name(f"{tmp.name}.{i}") for i in range(parts)]

    def get(i: int) -> None:
        a, b = i * step, min(size, (i + 1) * step) - 1
        part = pieces[i]
        for attempt in range(RETRIES + 1):
            have = part.stat().st_size if part.exists() else 0
            if have == b - a + 1:
                return
            try:
                req = urllib.request.Request(url, headers=dict(UA, Range=f"bytes={a + have}-{b}"))
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                    if r.status != 206:
                        raise OSError(f"server ignored the range request (HTTP {r.status})")
                    shutil.copyfileobj(r, f, 1 << 20)
                if part.stat().st_size == b - a + 1:
                    return
                raise OSError("short read")
            except OSError:
                if attempt == RETRIES:
                    raise
                time.sleep(min(30, 3 * (attempt + 1)))

    with ThreadPoolExecutor(parts) as ex:
        list(ex.map(get, range(parts)))
    with open(tmp, "wb") as out:
        for part in pieces:
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 20)
            part.unlink()


def _get(a: Asset, log) -> Optional[str]:
    """Download one asset from its sources; None on success, else the last error."""
    p = a.path()
    err = None
    sources = (a.url, *a.mirrors)
    for i, url in enumerate(sources):
        for check in (1, 2):                  # one fresh retry after a checksum mismatch
            try:
                _fetch(url, p, a.size)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                break
            if sha256(p) == a.sha256:
                return None
            p.unlink()
            err = "checksum mismatch"
            if check == 1:
                log(f"  retry     {a.name}: checksum mismatch, downloading again")
        if i + 1 < len(sources):
            log(f"  fallback  {a.name}: {err}; trying {_resolve(sources[i + 1]).split('/')[2]}")
    return err


def fetch_for_module(name: str, dst) -> Path:
    """For the modules' own "download if missing" paths: same sources, retries and checks as download-models.

    In offline mode nothing is downloaded: the stage fails with a clear message and the film degrades.
    """
    from .providers import offline
    if offline():
        raise RuntimeError(f"{name} is missing and --offline forbids downloads: run `agentdunker download-models`")
    a = BY_NAME.get(name)
    if a is None:
        raise RuntimeError(f"{name} is not a known asset: run `agentdunker download-models`")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for url in (a.url, *a.mirrors):
        try:
            _fetch(url, dst, a.size)
            return dst
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"could not download {name}: {err}; run `agentdunker download-models`")


def download(names: Optional[Iterable[str]] = None, preset: Optional[str] = None, force: bool = False,
             log=print) -> list[str]:
    """Download (and verify) assets; returns the names that failed. Finished files are kept between runs."""
    todo = [BY_NAME[n] for n in names] if names else [a for a in ASSETS if not preset or preset in a.presets]
    failed = []
    for a in todo:
        p = a.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not force and status(a, verify=True) == "ok":
            log(f"  ok        {a.name}")
            continue
        if force and p.exists():
            p.unlink()
        host = _resolve(a.url).split("/")[2]
        log(f"  download  {a.name} ({a.size / 1e6:.1f} MB) ← {host}")
        err = _get(a, log)
        if err:
            log(f"  FAILED    {a.name}: {err}")
            failed.append(a.name)
    return failed


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


# Every Chromium we start: no background networking, no component updates (it downloaded a 7.9 MB .crx during a
# render before), no first-run UI. Used by the replay renderer, the promo pages and the preflight check.
CHROMIUM_QUIET_FLAGS = ["--disable-background-networking", "--disable-component-update", "--no-first-run",
                        "--no-default-browser-check", "--disable-sync", "--disable-default-apps",
                        "--disable-domain-reliability", "--disable-breakpad", "--metrics-recording-only"]
_CHECK: dict = {}


def chromium_check(timeout: float = 30) -> tuple[bool, str]:
    """Start Chromium once (headless, about:blank). (True, path) or (False, first error line). Cached per process."""
    exe = chromium_path()
    if not exe:
        return False, "not installed"
    if exe in _CHECK:
        return _CHECK[exe]
    with tempfile.TemporaryDirectory(prefix="adk-chromium-check-") as d:
        cmd = [exe, "--headless", "--no-sandbox", "--disable-gpu", *CHROMIUM_QUIET_FLAGS, f"--user-data-dir={d}",
               "--dump-dom", "about:blank"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={**os.environ, "TMPDIR": d})
            lines = [l for l in (r.stderr or "").splitlines() if l.strip()]
            bad = r.returncode != 0
            msg = next((l for l in lines if "error" in l.lower()), lines[0] if lines else f"exit code {r.returncode}")
            res = (False, f"exit code {r.returncode}: {msg.strip()[:200]}") if bad else (True, exe)
        except subprocess.TimeoutExpired:
            res = (False, f"did not finish within {timeout:.0f} s")
        except OSError as e:
            res = (False, f"{type(e).__name__}: {e}")
    _CHECK[exe] = res
    return res


def chromium_fix() -> str:
    """The command that installs Chromium's system libraries, with this environment's real path."""
    pw = Path(sys.executable).parent / "playwright"
    return f"sudo {pw if pw.exists() else sys.executable + ' -m playwright'} install-deps chromium"


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
