"""agentdunker.court: per-frame NBA court calibration (image <-> court feet, origin at centre court)."""
from .api import CourtCalibration, img_to_court, to_contract

__all__ = ["CourtCalibration", "img_to_court", "to_contract", "calibrate"]


def calibrate(video: str, tracks: str | None = None) -> dict:
    from .__main__ import calibrate as _calibrate
    return _calibrate(video, tracks)
