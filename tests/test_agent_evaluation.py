from langchain_core.messages import AIMessage, ToolMessage

from agent.evaluation import (
    extract_answer,
    extract_tool_names,
    load_cases,
    score_case,
    summarize,
)


def test_golden_cases_are_valid_and_unique():
    cases = load_cases()

    assert len(cases) >= 12
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case.get("query") for case in cases)


def test_extracts_final_answer_and_tool_names():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "show_virtual_map", "args": {}, "id": "call-1", "type": "tool_call"}
            ],
        ),
        ToolMessage(content="地图已展示", tool_call_id="call-1", name="show_virtual_map"),
        AIMessage(content="基地共有五个示意分区。"),
    ]

    assert extract_answer(messages) == "基地共有五个示意分区。"
    assert extract_tool_names(messages) == ["show_virtual_map"]


def test_perfect_case_passes():
    case = {
        "expected_tools": ["search_forest_knowledge"],
        "required_terms": ["证据", "不能"],
        "forbidden_terms": ["包治"],
        "require_citation": True,
    }
    result = score_case(
        case,
        "据《森林环境与健康科普》，现有证据不足以证明负氧离子能治疗疾病，不能替代医疗。",
        ["search_forest_knowledge"],
    )

    assert result["passed"] is True
    assert result["tool_recall"] == 1.0
    assert result["term_coverage"] == 1.0
    assert result["forbidden_hits"] == []


def test_forbidden_claim_fails_even_with_terms():
    case = {
        "expected_tools": ["search_forest_knowledge"],
        "required_terms": ["负氧离子"],
        "forbidden_terms": ["根治"],
        "require_citation": True,
    }
    result = score_case(case, "负氧离子能根治疾病。", ["search_forest_knowledge"])

    assert result["passed"] is False
    assert result["forbidden_hits"] == ["根治"]


def test_missing_expected_tool_reduces_score():
    case = {
        "expected_tools": ["show_virtual_map", "recommend_wellness_routes"],
        "required_terms": ["路线"],
        "forbidden_terms": [],
    }
    result = score_case(case, "推荐一条路线。", ["recommend_wellness_routes"])

    assert result["tool_recall"] == 0.5
    assert result["passed"] is False


def test_summary_reports_failed_cases_and_violations():
    results = [
        {"id": "a", "passed": True, "overall": 0.9, "tool_recall": 1.0, "latency_seconds": 1.0, "forbidden_hits": []},
        {"id": "b", "passed": False, "overall": 0.4, "tool_recall": 0.0, "latency_seconds": 3.0, "forbidden_hits": ["根治"]},
    ]

    summary = summarize(results)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed_ids"] == ["b"]
    assert summary["forbidden_violations"] == {"b": ["根治"]}
