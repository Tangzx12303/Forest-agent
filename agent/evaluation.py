"""森林智能体的离线金标集评测。

运行真实模型评测（需要配置模型和 DashScope 密钥）：

    python -m agent.evaluation --model qwen-plus --limit 5

评测器只对可观察行为评分：调用了哪些工具、是否包含必需事实、是否出现禁用表述，
以及知识问答是否引用本地资料。评测过程不会调用 LLM 评审。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from langchain_core.messages import AIMessage, ToolMessage

GOLDEN_PATH = Path(__file__).with_name("golden_cases.json")
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
CITATION_MARKERS = ("资料", "据《", "知识库", "本地资料")


def load_cases(path: Path | str = GOLDEN_PATH) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("金标评测集为空或格式不正确。")
    seen = set()
    for case in cases:
        case_id = case.get("id")
        if not case_id:
            raise ValueError("每个评测用例必须有唯一 id。")
        if case_id in seen:
            raise ValueError(f"评测用例 id 重复：{case_id}")
        if not case.get("query"):
            raise ValueError(f"评测用例缺少 query：{case_id}")
        seen.add(case_id)
    return cases


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", None}
        )
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""


def extract_answer(messages: Iterable) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            text = message_text(message).strip()
            if text:
                return text
        elif isinstance(message, dict) and message.get("role") == "assistant":
            text = str(message.get("content") or "").strip()
            if text:
                return text
    return ""


def extract_tool_names(messages: Iterable) -> list[str]:
    names: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict) and call.get("name"):
                names.append(call["name"])
        if isinstance(message, ToolMessage) and message.name:
            names.append(message.name)
        elif isinstance(message, dict) and message.get("role") == "tool" and message.get("name"):
            names.append(message["name"])
    return list(dict.fromkeys(names))


def score_case(case: dict, answer: str, tools_used: list[str]) -> dict:
    expected = list(case.get("expected_tools", []))
    matched = [name for name in expected if name in tools_used]
    tool_recall = len(matched) / len(expected) if expected else 1.0

    required = list(case.get("required_terms", []))
    required_hits = [term for term in required if term in answer]
    term_coverage = len(required_hits) / len(required) if required else 1.0

    forbidden_hits = [term for term in case.get("forbidden_terms", []) if term in answer]
    forbidden_ok = not forbidden_hits

    citation_required = bool(case.get("require_citation")) or "search_forest_knowledge" in expected
    citation_ok = (not citation_required) or any(marker in answer for marker in CITATION_MARKERS)

    overall = (
        0.40 * tool_recall
        + 0.35 * term_coverage
        + 0.15 * (1.0 if forbidden_ok else 0.0)
        + 0.10 * (1.0 if citation_ok else 0.0)
    )
    passed = (
        tool_recall >= float(case.get("min_tool_recall", 1.0))
        and term_coverage >= float(case.get("min_term_score", 0.6))
        and forbidden_ok
        and citation_ok
        and overall >= 0.70
    )
    return {
        "tool_recall": round(tool_recall, 3),
        "expected_tools": expected,
        "matched_tools": matched,
        "tools_used": tools_used,
        "term_coverage": round(term_coverage, 3),
        "required_hits": required_hits,
        "missing_terms": [term for term in required if term not in answer],
        "forbidden_hits": forbidden_hits,
        "citation_ok": citation_ok,
        "overall": round(overall, 3),
        "passed": passed,
    }


def run_case(agent, case: dict) -> dict:
    started = time.perf_counter()
    result = agent.invoke({"messages": [{"role": "user", "content": case["query"]}]})
    latency = round(time.perf_counter() - started, 3)
    messages = result.get("messages", [])
    answer = extract_answer(messages)
    tools_used = extract_tool_names(messages)
    scored = score_case(case, answer, tools_used)
    return {
        "id": case["id"],
        "query": case["query"],
        "category": case.get("category", ""),
        "answer": answer,
        "latency_seconds": latency,
        **scored,
    }


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"total": 0, "passed": 0, "pass_rate": 0.0}
    return {
        "total": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "pass_rate": round(sum(1 for item in results if item["passed"]) / len(results), 3),
        "average_overall": round(statistics.mean(item["overall"] for item in results), 3),
        "average_tool_recall": round(statistics.mean(item["tool_recall"] for item in results), 3),
        "average_latency_seconds": round(statistics.mean(item["latency_seconds"] for item in results), 3),
        "failed_ids": [item["id"] for item in results if not item["passed"]],
        "forbidden_violations": {
            item["id"]: item["forbidden_hits"]
            for item in results
            if item["forbidden_hits"]
        },
    }


def print_report(results: list[dict], summary: dict) -> None:
    print("\nAgent golden-set evaluation")
    print("=" * 72)
    print(f"{'id':<24}{'pass':<8}{'tool':<8}{'terms':<8}{'overall':<9}{'latency':<8}")
    for item in results:
        print(
            f"{item['id']:<24}{'PASS' if item['passed'] else 'FAIL':<8}"
            f"{item['tool_recall']:<8}{item['term_coverage']:<8}"
            f"{item['overall']:<9}{item['latency_seconds']:<8}"
        )
    print("-" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行森林 Agent 金标评测。")
    parser.add_argument("--model", default="", help="模型 id；默认取 config/models.yml 的默认模型")
    parser.add_argument("--case", action="append", default=[], help="只运行指定用例 id，可重复")
    parser.add_argument("--limit", type=int, default=0, help="最多运行多少个用例")
    parser.add_argument("--output", default="", help="JSON 报告路径")
    args = parser.parse_args(argv)

    from agent.agent import build_agent
    from config.settings import default_model_id
    from model.factory import build_chat_model
    from rag.vector_store import VectorStoreService

    model_id = args.model or default_model_id()
    if not model_id:
        print("没有可用模型：请在 .env 配置至少一个供应商 API Key。")
        return 2
    cases = load_cases()
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print("没有匹配的评测用例。")
        return 2

    try:
        agent = build_agent(build_chat_model(model_id, streaming=False), VectorStoreService())
    except Exception as exc:
        print(f"无法初始化 Agent：{exc}")
        return 2

    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']} …", flush=True)
        try:
            results.append(run_case(agent, case))
        except Exception as exc:
            results.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "category": case.get("category", ""),
                    "answer": "",
                    "latency_seconds": 0.0,
                    "tool_recall": 0.0,
                    "expected_tools": case.get("expected_tools", []),
                    "matched_tools": [],
                    "tools_used": [],
                    "term_coverage": 0.0,
                    "required_hits": [],
                    "missing_terms": case.get("required_terms", []),
                    "forbidden_hits": [],
                    "citation_ok": False,
                    "overall": 0.0,
                    "passed": False,
                    "error": str(exc),
                }
            )

    summary = summarize(results)
    print_report(results, summary)

    output = Path(args.output) if args.output else (
        DEFAULT_REPORT_DIR / f"agent_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"model": model_id, "generated_at": datetime.now().isoformat(), "summary": summary, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告已写入：{output}")
    return 0 if summary["failed_ids"] == [] else 1


if __name__ == "__main__":
    raise SystemExit(main())
