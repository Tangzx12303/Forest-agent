import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api import ChatRequest, administrator, app, chat
from db import repository


def test_chat_rejects_admin_accounts():
    with pytest.raises(HTTPException) as excinfo:
        chat(ChatRequest(message="测试", model_id="test-model"), {"id": 1, "role": "admin"})

    assert excinfo.value.status_code == 403
    assert "管理员账号不提供对话功能" in excinfo.value.detail


def test_reader_summary_query_only_reads_readers_and_rolling_summaries(monkeypatch):
    captured = {}

    def fake_query_all(sql, params=()):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(repository, "query_all", fake_query_all)

    assert repository.list_reader_summaries() == []
    assert "u.role='reader'" in captured["sql"]
    assert "LEFT JOIN conversation_memory" in captured["sql"]
    assert "chat_messages" not in captured["sql"]


def test_admin_summaries_endpoint_returns_reader_summaries(monkeypatch):
    monkeypatch.setattr(
        repository,
        "list_reader_summaries",
        lambda: [
            {
                "id": 2,
                "username": "reader",
                "display_name": "读者",
                "status": "active",
                "summary": "用户关注森林漫步与睡眠。",
                "updated_at": None,
            }
        ],
    )
    app.dependency_overrides[administrator] = lambda: {"id": 1, "username": "admin", "role": "admin"}
    try:
        response = TestClient(app).get("/api/admin/summaries")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["summaries"][0]["summary"] == "用户关注森林漫步与睡眠。"
    assert response.json()["summaries"][0]["updated_at"] == ""


def test_admin_summaries_endpoint_requires_login():
    app.dependency_overrides.clear()

    assert TestClient(app).get("/api/admin/summaries").status_code == 401
