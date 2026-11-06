"""M4: ten-player holo 3D replay from M1 tracks + M2 court (+ optional M3 identity / M5 events)."""
from .api import make_replay
from .scene import SceneConfig, build_play
from .trajectory import SmoothConfig, build_paths

__all__ = ["make_replay", "build_play", "build_paths", "SceneConfig", "SmoothConfig"]
