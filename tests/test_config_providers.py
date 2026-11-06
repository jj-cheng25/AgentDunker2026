import json

import pytest

from agentdunker import config, providers


def test_defaults_are_mosshub_and_no_key():
    assert config.llm().provider == "mosshub"
    assert config.llm().fast_model == "gemini-3.8-flash"
    assert providers.llm_key() is None
    assert providers.moss_key("speech") is None
    svc = providers.available()
    assert not svc["llm"]["ok"] and not svc["voice"]["ok"] and not svc["moss_vl"]["ok"]


def test_env_and_file_precedence(tmp_path, monkeypatch):
    f = tmp_path / "c.toml"
    f.write_text('[llm]\nprovider = "openai"\nfast_model = "m-file"\n[run]\nthreads = 4\n')
    monkeypatch.setenv("ADK_CONFIG", str(f))
    monkeypatch.setenv("ADK_LLM_FAST_MODEL", "m-env")
    config.load.cache_clear()
    config.llm.cache_clear()
    assert config.llm().provider == "openai"
    assert config.llm().fast_model == "m-env"          # env beats file
    assert config.llm().strong_model == "gpt-5"        # provider default
    assert config.threads() == 4
    config.override(run={"threads": 2})
    assert config.threads() == 2                       # command line beats both


def test_apply_exports_module_env(monkeypatch):
    import os
    config.override(llm={"provider": "gemini"}, run={"threads": 3})
    config.apply()
    assert os.environ["ADK_ID_CHEAP_MODEL"] == config.llm().fast_model
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["ADK_VLM_MODEL"].split(",")[0] == config.llm().strong_model


def test_key_from_env_never_in_public(monkeypatch):
    monkeypatch.setenv("MOSSHUB_API_KEY", "sk-secret-123")
    assert providers.llm_key() == "sk-secret-123"
    assert "sk-secret" not in json.dumps(config.public())


def test_voice_none_disables_moss_speech_only(monkeypatch):
    monkeypatch.setenv("MOSS_API_KEY", "k")
    config.override(voice={"backend": "none"})
    assert providers.moss_key("speech") is None
    assert providers.moss_key("vl") == "k"


def test_responses_to_chat():
    body = {"model": "gpt-x", "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/jpeg;base64,AA"}, {"type": "input_text", "text": "hi"}]}],
        "text": {"format": {"type": "json_schema", "name": "n", "schema": {"type": "object"}, "strict": True}}}
    chat = providers.responses_to_chat(body)
    assert chat["messages"][0]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA"}}
    assert chat["messages"][0]["content"][1] == {"type": "text", "text": "hi"}
    assert chat["response_format"]["json_schema"]["name"] == "n"


def test_gemini_schema_unions():
    s = {"type": "object", "properties": {"n": {"type": ["string", "null"]}}}
    providers._gemini_schema(s)
    assert s["properties"]["n"] == {"type": "string", "nullable": True}


def test_llm_post_routes_and_converts(monkeypatch):
    sent = {}

    def fake_post(url, json=None, timeout=None, headers=None):
        sent.update(url=url, json=json, headers=headers)
        return "resp"

    monkeypatch.setattr(providers.requests, "post", fake_post)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    config.override(llm={"provider": "gemini"})
    payload = {"model": "m", "input": [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}]}
    assert providers.llm_post("responses", payload, 5) == "resp"
    assert sent["url"].endswith("/v1beta/openai/chat/completions")
    assert sent["headers"] == {"Authorization": "Bearer g"}
    assert "messages" in sent["json"]


def test_llm_post_without_key_raises():
    config.override(llm={"provider": "none"})
    with pytest.raises(RuntimeError):
        providers.llm_post("chat/completions", {"model": "x"}, 1)


def test_narration_transport_split(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "llm_post", lambda e, p, t: calls.append(("llm", e)) or "chat")
    assert providers.moss_or_llm_post("https://api.mosshub.cn/v1/chat/completions", "MOSSHUB_API_KEY",
                                      json={"model": "m"}) == "chat"
    assert calls == [("llm", "chat/completions")]
    with pytest.raises(RuntimeError):            # no MOSS key: the narration module falls back
        providers.moss_or_llm_post("https://api.mosi.cn/v1/audio/speech", "MOSS_API_KEY", json={})
