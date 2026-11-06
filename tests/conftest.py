import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Every test: no user config, no credentials, caches in tmp."""
    for k in list(os.environ):
        if k.startswith(("ADK_", "MOSS", "OPENAI", "GEMINI", "GOOGLE_API")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ADK_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("ADK_CACHE", str(tmp_path / "cache"))
    from agentdunker import config
    config.load.cache_clear()
    config.llm.cache_clear()
    config._OVERRIDES.clear()
    yield
    config.load.cache_clear()
    config.llm.cache_clear()
    config._OVERRIDES.clear()


@pytest.fixture
def real_models_dir():
    """The developer's model dir, when present (slow tests only)."""
    d = os.environ.get("ADK_TEST_MODEL_DIR")
    return Path(d) if d else None
