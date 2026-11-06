"""Compute device: ``run.device = auto | cpu | cuda`` (``ADK_DEVICE``). auto = CUDA when torch sees a GPU.

Ultralytics detectors pick the GPU by themselves; this covers the rest (OSNet ReID, CLIP roles, the
ONNX ball detector). Court keypoints run on OpenCV DNN (CPU) in every mode.
"""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def torch_device() -> str:
    want = os.environ.get("ADK_DEVICE", "auto").lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def ort_providers() -> list[str]:
    try:
        import onnxruntime as ort
        have = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return ["CPUExecutionProvider"]
    if torch_device() == "cuda" and "CUDAExecutionProvider" in have:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
