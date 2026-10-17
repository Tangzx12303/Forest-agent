import anyio
from langchain_core.messages import AIMessageChunk

from api import ChatRequest, chat
from db import repository


class MapAgent:
    def stream(self, _state, stream_mode):
        assert stream_mode == "messages"
        yield (
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "show_virtual_map", "args": "{}", "id": "call-map", "index": 0}
                ],
            ),
            {"langgraph_node": "model"},
        )
        yield (
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "recommend_wellness_routes",
                        "args": '{"goals":"轻松","duration_minutes":90}',
                        "id": "call-route",
                        "index": 0,
                    }
                ],
            ),
            {"langgraph_node": "model"},
        )
        yield AIMessageChunk(content="已展示地图并推荐路线。"), {"langgraph_node": "model"}


def read_body(response) -> str:
    async def consume():
        return "".join([chunk async for chunk in response.body_iterator])

    return anyio.run(consume)


class FakeRouter:
    def predict_tools(self, query, threshold=None):
        del query, threshold
        return [{"tool": "show_virtual_map", "confidence": 0.9}]

    def score_quality(self, query, answer):
        del query, answer
        return 0.88


class HintAgent:
    def __init__(self, captured):
        self.captured = captured

    def stream(self, state, stream_mode):
        assert stream_mode == "messages"
        self.captured["messages"] = state["messages"]
        yield AIMessageChunk(content="回答内容。"), {"langgraph_node": "model"}


def test_chat_injects_router_hint_and_captures_turn(monkeypatch):
    captured = {}
    records = []
    monkeypatch.setattr("api.list_model_presets", lambda: [{"id": "test-model"}])
    monkeypatch.setattr("api.get_model", lambda _model_id: object())
    monkeypatch.setattr("api.get_agent", lambda _model_id: HintAgent(captured))
    monkeypatch.setattr("api.get_tool_router", lambda: FakeRouter())
    monkeypatch.setattr("api.TRAINING_CAPTURE_ENABLED", True)
    monkeypatch.setattr("api.append_turn", lambda record: records.append(record))
    monkeypatch.setattr(repository, "load_summary", lambda _user_id: "")
    monkeypatch.setattr("api.update_summary", lambda *_args: "")

    response = chat(
        ChatRequest(message="森林浴有什么证据？", model_id="test-model", category="health"),
        {"id": 31, "role": "reader"},
    )
    read_body(response)

    assert any("show_virtual_map" in str(message.content) for message in captured["messages"])
    assert len(records) == 1
    assert records[0].query == "森林浴有什么证据？"
    assert records[0].checks["model_quality_score"] == 0.88


def test_chat_streams_tool_and_map_ui_events(monkeypatch):
    monkeypatch.setattr("api.list_model_presets", lambda: [{"id": "test-model"}])
    monkeypatch.setattr("api.get_model", lambda _model_id: object())
    monkeypatch.setattr("api.get_agent", lambda _model_id: MapAgent())
    monkeypatch.setattr(repository, "load_summary", lambda _user_id: "")
    monkeypatch.setattr("api.update_summary", lambda *_args: "")

    response = chat(
        ChatRequest(message="展示地图并推荐路线", model_id="test-model", category="forest"),
        {"id": 31, "role": "reader"},
    )
    body = read_body(response)

    assert '"type": "tool", "name": "show_virtual_map"' in body
    assert '"type": "tool", "name": "recommend_wellness_routes"' in body
    assert '"type": "map"' in body
    assert '"zones"' in body
    assert "已展示地图并推荐路线。" in body
