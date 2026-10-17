from config import settings


def test_model_selector_only_includes_providers_with_configured_keys(monkeypatch):
    config = {
        "default_model": "deepseek-chat",
        "models": [
            {"id": "qwen-plus", "api_key_env": "DASHSCOPE_API_KEY"},
            {"id": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
        ],
    }
    monkeypatch.setattr(settings, "_load_yaml", lambda _filename: config)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "configured")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert [item["id"] for item in settings.list_model_presets()] == ["qwen-plus"]
    assert settings.default_model_id() == "qwen-plus"


def test_model_selector_honors_configured_provider(monkeypatch):
    config = {
        "default_model": "deepseek-chat",
        "models": [
            {"id": "qwen-plus", "api_key_env": "DASHSCOPE_API_KEY"},
            {"id": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
        ],
    }
    monkeypatch.setattr(settings, "_load_yaml", lambda _filename: config)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "configured")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "configured")

    assert settings.default_model_id() == "deepseek-chat"
