"""真实模型冒烟测试；仅在显式启用后运行。

PowerShell:
    $env:RUN_LIVE_AGENT_TESTS="1"
    python -m pytest tests/test_live_agent.py -q
"""
import os

import pytest

from agent.evaluation import load_cases, run_case

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_AGENT_TESTS", "").strip().lower() not in {"1", "true", "yes", "on"},
    reason="设置 RUN_LIVE_AGENT_TESTS=1 才会调用真实模型 API",
)


def test_live_agent_selects_map_tool_for_map_question():
    from agent.agent import build_agent
    from config.settings import default_model_id
    from model.factory import build_chat_model
    from rag.vector_store import VectorStoreService

    model_id = default_model_id()
    if not model_id:
        pytest.skip("当前没有可用模型密钥。")
    agent = build_agent(build_chat_model(model_id, streaming=False), VectorStoreService())
    case = next(item for item in load_cases() if item["id"] == "map-zones")

    result = run_case(agent, case)

    assert result["answer"]
    assert "show_virtual_map" in result["tools_used"]
