import anyio
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from api import ChatRequest, app, chat, lifespan, me
from db import repository


def test_vue_frontend_is_served():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "森语" in response.text
    assert "/static/app.js" in response.text


def test_lifespan_trains_local_model_before_database(monkeypatch):
    events = []
    monkeypatch.setattr("api.train_default_model", lambda: events.append("train"))
    monkeypatch.setattr("api.initialize_database", lambda: events.append("database"))

    async def run_lifespan():
        application = type("Application", (), {"state": type("State", (), {})()})()
        async with lifespan(application):
            pass

    anyio.run(run_lifespan)

    assert events == ["train", "database"]


def test_private_endpoints_require_login():
    client = TestClient(app)

    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/models").status_code == 401
    assert client.get("/api/routes/map").status_code == 401


def test_auth_me_clears_transcript_and_returns_only_summary(monkeypatch):
    cleared = []
    monkeypatch.setattr(repository, "clear_messages", lambda user_id: cleared.append(user_id))
    monkeypatch.setattr(repository, "load_summary", lambda _user_id: "持续保留的摘要")

    response = me({"id": 21, "username": "reader", "role": "reader"})

    assert cleared == [21]
    assert response["messages"] == []
    assert response["summary"] == "持续保留的摘要"


def test_chat_streams_text_then_updates_summary_without_persisting_transcript(monkeypatch):
    captured = {}

    class FakeAgent:
        def stream(self, state, stream_mode):
            assert stream_mode == "messages"
            captured["messages"] = state["messages"]
            yield AIMessageChunk(content="逐字"), {"langgraph_node": "model"}
            yield AIMessageChunk(content="输出"), {"langgraph_node": "model"}

    monkeypatch.setattr("api.list_model_presets", lambda: [{"id": "test-model"}])
    monkeypatch.setattr("api.get_model", lambda _model_id: object())
    monkeypatch.setattr("api.get_agent", lambda _model_id: FakeAgent())
    monkeypatch.setattr(repository, "load_summary", lambda _user_id: "已有摘要")
    monkeypatch.setattr("api.update_summary", lambda *_args: "更新后摘要")
    monkeypatch.setattr(
        repository,
        "append_message",
        lambda *_args: (_ for _ in ()).throw(AssertionError("逐条消息不应写入数据库")),
    )

    response = chat(
        ChatRequest(
            message="测试",
            model_id="test-model",
            category="health",
            history=[{"role": "user", "content": "测试"}],
        ),
        {"id": 31},
    )

    async def read_stream():
        return "".join([chunk async for chunk in response.body_iterator])

    body = anyio.run(read_stream)

    assert response.media_type == "text/event-stream"
    assert '"type": "token", "text": "逐字"' in body
    assert '"type": "token", "text": "输出"' in body
    assert '"type": "summary", "summary": "更新后摘要"' in body
    assert '"type":"done"' in body
    assert any("身体健康问答" in str(message.content) for message in captured["messages"] if hasattr(message, "content"))
    assert all(
        "地图步行路线" not in str(message.content) and "分区顺序" not in str(message.content)
        for message in captured["messages"]
        if hasattr(message, "content")
    )


def test_chat_rejects_unknown_category():
    from fastapi import HTTPException
    import pytest

    with pytest.raises(HTTPException, match="咨询分类无效"):
        chat(ChatRequest(message="测试", model_id="test-model", category="routes"), {"id": 31})


def test_chat_sends_fallback_text_when_model_stream_is_empty(monkeypatch):
    class EmptyAgent:
        def stream(self, _state, stream_mode):
            assert stream_mode == "messages"
            return iter(())

    monkeypatch.setattr("api.list_model_presets", lambda: [{"id": "test-model"}])
    monkeypatch.setattr("api.get_model", lambda _model_id: object())
    monkeypatch.setattr("api.get_agent", lambda _model_id: EmptyAgent())
    monkeypatch.setattr(repository, "load_summary", lambda _user_id: "")
    monkeypatch.setattr("api.update_summary", lambda *_args: "")

    response = chat(ChatRequest(message="测试", model_id="test-model"), {"id": 31})

    async def read_stream():
        return "".join([chunk async for chunk in response.body_iterator])

    body = anyio.run(read_stream)

    assert "暂时没有生成可展示的文字" in body


def test_clear_conversation_deletes_messages_and_summary(monkeypatch):
    statements = []
    monkeypatch.setattr(
        repository,
        "execute",
        lambda statement, params: statements.append((statement, params)),
    )

    repository.clear_conversation(42)

    assert statements == [
        ("DELETE FROM chat_messages WHERE user_id=%s", (42,)),
        ("DELETE FROM conversation_memory WHERE user_id=%s", (42,)),
    ]