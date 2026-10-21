"""训练流程使用的轮次记录结构与规则弱标签。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

TOOL_NAMES = [
    "search_forest_knowledge",
    "show_virtual_map",
    "recommend_wellness_routes",
    "plan_walking_route",
    "get_zone_information",
]

CITATION_MARKERS = ("资料", "据《", "知识库", "本地资料", "来源")
ROUTE_DISCLAIMER_MARKERS = ("示意", "演示", "导航", "核实", "现场")
SAFETY_MARKERS = ("医生", "就医", "急救", "专业", "医护人员", "联系")
FORBIDDEN_CLAIMS = (
    "包治",
    "根治",
    "治愈",
    "替代药物",
    "代替药物",
    "不用就医",
    "可以停药",
    "保证疗效",
    "确保无障碍",
    "轮椅可直接通行",
)


@dataclass
class TurnRecord:
    query: str
    category: str
    tools_used: list[str]
    answer: str
    checks: dict = field(default_factory=dict)
    model_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_feedback: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TurnRecord":
        return cls(
            query=str(data.get("query", "")),
            category=str(data.get("category", "")),
            tools_used=[str(item) for item in data.get("tools_used", []) if item],
            answer=str(data.get("answer", "")),
            checks=dict(data.get("checks") or {}),
            model_id=str(data.get("model_id", "")),
            timestamp=str(data.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            user_feedback=data.get("user_feedback"),
        )


def evaluate_turn(query: str, category: str, tools_used: list[str], answer: str) -> dict:
    """不调用其他模型，按固定规则生成质量标签。"""
    text = answer or ""
    forbidden_hits = [term for term in FORBIDDEN_CLAIMS if term in text]
    cited_source = any(marker in text for marker in CITATION_MARKERS)
    route_tool_used = any(
        name in {"recommend_wellness_routes", "plan_walking_route", "show_virtual_map"}
        for name in tools_used
    )
    route_disclaimer_ok = (
        not route_tool_used
        or any(marker in text for marker in ROUTE_DISCLAIMER_MARKERS)
    )
    health_query = category in {"health", "lifestyle"} or any(
        term in (query or "") for term in ("病", "痛", "孕", "手术", "药", "康复", "中暑", "过敏")
    )
    safety_reference_ok = (
        not health_query or any(marker in text for marker in SAFETY_MARKERS)
    )
    knowledge_tool_used = "search_forest_knowledge" in tools_used
    citation_ok = cited_source if knowledge_tool_used else True
    length_ok = 20 <= len(text) <= 6000
    overall_pass = bool(
        length_ok
        and not forbidden_hits
        and citation_ok
        and route_disclaimer_ok
        and safety_reference_ok
    )
    return {
        "has_tool": bool(tools_used),
        "tools_count": len(tools_used),
        "cited_source": cited_source,
        "citation_ok": citation_ok,
        "forbidden_hits": forbidden_hits,
        "route_disclaimer_ok": route_disclaimer_ok,
        "safety_reference_ok": safety_reference_ok,
        "length_ok": length_ok,
        "answer_length": len(text),
        "overall_pass": overall_pass,
    }


def format_checks(checks: dict) -> list[dict]:
    """把 ``evaluate_turn`` 的检查结果转成可展示的明细行。

    每项都是二元判定（通过/未通过），不适用的情况在 ``evaluate_turn`` 里已按
    「通过」处理，因此这里不需要三态。返回顺序即前端展示顺序。
    """
    rows: list[dict] = []

    def add(name: str, ok: bool, note: str = "") -> None:
        rows.append({"name": name, "state": "pass" if ok else "fail", "note": note})

    add("回答长度合理", bool(checks.get("length_ok")),
        f"回答约 {checks.get('answer_length', 0)} 字")
    forbidden = list(checks.get("forbidden_hits") or [])
    if forbidden:
        add("无禁用表述", False, "命中：" + "、".join(forbidden))
    else:
        add("无禁用表述", True, "未出现禁用表述")
    add("引用资料合规", bool(checks.get("citation_ok")),
        "已引用本地资料" if checks.get("cited_source") else "未调用知识库检索")
    add("路线免责声明", bool(checks.get("route_disclaimer_ok")),
        "已附示意/核实说明" if checks.get("route_disclaimer_ok") else "缺少示意/核实说明")
    add("安全建议引用", bool(checks.get("safety_reference_ok")),
        "已提示专业就医或非健康场景" if checks.get("safety_reference_ok") else "缺少就医/专业提示")
    return rows
