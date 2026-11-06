"""Model loading: detector, OSNet ReID, CLIP role classifier (all CPU)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from agentdunker.device import torch_device as _device

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("ADK_MODEL_DIR", Path.home() / ".cache" / "agentdunker" / "models"))
OSNET_FALLBACK = MODEL_DIR / "osnet_x0_25_msmt17.pth"  # agentdunker download-models
CLIP_NAME = os.environ.get("ADK_CLIP_MODEL", "openai/clip-vit-large-patch14")

ROLE_PROMPTS = {
    "player": ["a basketball player wearing a jersey and shorts", "a basketball player running on the court"],
    "referee": [
        "a basketball referee wearing a gray shirt and long black pants",
        "a basketball referee wearing a black and white striped shirt",
        "a referee",
    ],
    "other": [
        "a coach in a suit",
        "a fan sitting in the crowd",
        "a person standing on the sideline in a t-shirt",
        "a cameraman holding a camera",
    ],
}
ROLES = tuple(ROLE_PROMPTS)


def set_threads(n: int) -> None:
    torch.set_num_threads(n)
    cv2.setNumThreads(min(n, 8))


def load_detector(name: str):
    from ultralytics import RTDETR, YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    path = MODEL_DIR / name
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(path))
    return (RTDETR if "rtdetr" in name else YOLO)(str(path))


class ReID:
    """OSNet-x0.25 (MSMT17) person embedding, L2-normalised 512-d."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self) -> None:
        from ._vendor.osnet import osnet_x0_25

        path = MODEL_DIR / "osnet_x0_25_msmt17.pth"
        path = path if path.exists() else OSNET_FALLBACK
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.get("state_dict", state).items()}
        self.model = osnet_x0_25(num_classes=state["classifier.weight"].shape[0], pretrained=False).eval()
        self.model.load_state_dict(state)
        self.model.to(_device())

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), np.float32)
        x = np.stack([cv2.resize(c, (128, 256))[:, :, ::-1] for c in crops]).astype(np.float32) / 255.0
        x = torch.from_numpy(((x - self.MEAN) / self.STD).transpose(0, 3, 1, 2).copy())
        with torch.inference_mode():
            f = self.model(x.to(_device())).cpu().numpy()
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)


class RoleClassifier:
    """Zero-shot CLIP: P(player), P(referee), P(other) per crop."""

    def __init__(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(CLIP_NAME).eval().to(_device())
        self.proc = CLIPProcessor.from_pretrained(CLIP_NAME, use_fast=True)
        texts = [p for ps in ROLE_PROMPTS.values() for p in ps]
        self.groups = np.repeat(np.arange(len(ROLES)), [len(ps) for ps in ROLE_PROMPTS.values()])
        with torch.inference_mode():
            t = self.model.get_text_features(**self.proc(text=texts, return_tensors="pt", padding=True).to(_device()))
        self.text = t / t.norm(dim=-1, keepdim=True)

    def __call__(self, crops: list[np.ndarray], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(crops), batch):
            rgb = [c[:, :, ::-1].copy() for c in crops[i : i + batch]]
            with torch.inference_mode():
                f = self.model.get_image_features(**self.proc(images=rgb, return_tensors="pt").to(_device()))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ self.text.T).softmax(-1).cpu().numpy()
            out.append(np.stack([p[:, self.groups == g].sum(1) for g in range(len(ROLES))], 1))
        return np.concatenate(out) if out else np.zeros((0, len(ROLES)))


def torso_color(crop: np.ndarray) -> np.ndarray:
    """Jersey/shorts colour descriptor in Lab: [L,a,b medians + L quartiles] for torso and shorts."""
    h, w = crop.shape[:2]
    feats = []
    for y0, y1 in ((0.15, 0.42), (0.42, 0.58)):
        r = crop[int(h * y0) : max(int(h * y0) + 1, int(h * y1)), int(w * 0.3) : max(int(w * 0.3) + 1, int(w * 0.7))]
        lab = cv2.cvtColor(r, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        med = np.median(lab, 0)
        q = np.percentile(lab[:, 0], [25, 75])
        feats.extend([*med, *q])
    return np.array(feats, np.float32)
