import json

from training.capture import append_turn, load_turns
from training.features import encode_batch, hashed_ids
from training.inference import ToolRouterService
from training.schema import TurnRecord, evaluate_turn
from training.train import build_rows


def test_evaluate_turn_accepts_cited_safe_answer():
    checks = evaluate_turn(
        "森林浴能治疗焦虑吗？",
        "health",
        ["search_forest_knowledge"],
        "据《森林环境与健康科普》，短期可能有助于放松，但不能替代治疗；如有持续焦虑请就医。",
    )

    assert checks["overall_pass"] is True
    assert checks["cited_source"] is True
    assert checks["safety_reference_ok"] is True
    assert checks["forbidden_hits"] == []


def test_evaluate_turn_flags_forbidden_claim_and_missing_citation():
    checks = evaluate_turn(
        "负氧离子能治病吗？",
        "forest",
        ["search_forest_knowledge"],
        "负氧离子可以根治疾病。",
    )

    assert checks["overall_pass"] is False
    assert checks["citation_ok"] is False
    assert "根治" in checks["forbidden_hits"]


def test_evaluate_turn_requires_route_disclaimer():
    checks = evaluate_turn(
        "从入口到古树怎么走？",
        "forest",
        ["plan_walking_route"],
        "沿主路一直走即可到达，距离 850 米。",
    )

    assert checks["route_disclaimer_ok"] is False
    assert checks["overall_pass"] is False


def test_turn_record_roundtrip():
    record = TurnRecord(
        query="测试问题",
        category="forest",
        tools_used=["show_virtual_map"],
        answer="答案",
        checks={"overall_pass": True},
        model_id="qwen-plus",
    )

    restored = TurnRecord.from_dict(json.loads(json.dumps(record.to_dict(), ensure_ascii=False)))

    assert restored.query == record.query
    assert restored.tools_used == ["show_virtual_map"]
    assert restored.checks == {"overall_pass": True}


def test_hashed_features_are_deterministic_and_offsets_are_built():
    first = hashed_ids("森林浴 shinrin-yoku")
    second = hashed_ids("森林浴 shinrin-yoku")
    flat, offsets = encode_batch(["森林浴", "森林 步道"])

    assert first == second
    assert first
    assert offsets == [0, len(hashed_ids("森林浴"))]
    assert len(flat) == sum(len(hashed_ids(text)) for text in ["森林浴", "森林 步道"])


def test_capture_appends_and_skips_malformed_lines(tmp_path):
    path = tmp_path / "turns.jsonl"
    record = TurnRecord(
        query="记录测试",
        category="forest",
        tools_used=["search_forest_knowledge"],
        answer="据资料说明。",
    )

    assert append_turn(record, path=path) is True
    with path.open("a", encoding="utf-8") as file:
        file.write("{not-json}\n")

    loaded = load_turns(path)
    assert len(loaded) == 1
    assert loaded[0]["query"] == "记录测试"


def test_tool_router_service_is_a_noop_without_checkpoint(tmp_path):
    service = ToolRouterService(path=tmp_path / "missing.pt")

    assert service.available is False
    assert service.predict_tools("森林浴") == []
    assert service.score_quality("问题", "回答") is None
    assert service.load_error


def test_build_rows_combines_golden_cases_and_turns():
    golden = [{"query": "看地图", "expected_tools": ["show_virtual_map"]}]
    turns = [
        {
            "query": "推荐路线",
            "tools_used": ["recommend_wellness_routes"],
            "answer": "示意路线如下。",
            "checks": {"overall_pass": True},
        }
    ]

    rows = build_rows(golden, turns)

    assert [row["query"] for row in rows] == ["看地图", "推荐路线"]
    assert rows[0]["quality"] is None
    assert rows[1]["quality"] == 1.0
