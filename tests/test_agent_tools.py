from langchain_core.documents import Document

from agent.tools import build_forest_tools, build_map_payload, resolve_zone
from training.schema import TOOL_NAMES


class FakeVectorStore:
    def search(self, query, k=None):
        del query, k
        return [
            Document(
                page_content="森林环境可能有助于短期放松，但不能替代治疗。",
                metadata={"title": "森林环境与健康科普", "source": "森林环境与健康科普.txt"},
            )
        ]


def tool_map():
    return {item.name: item for item in build_forest_tools(FakeVectorStore())}


def test_tool_registry_contains_all_agent_tools():
    assert set(tool_map()) == {
        "search_forest_knowledge",
        "show_virtual_map",
        "recommend_wellness_routes",
        "plan_walking_route",
        "get_zone_information",
    }


def test_training_labels_match_registered_tools():
    assert set(TOOL_NAMES) == set(tool_map())


def test_zone_resolution_accepts_id_name_and_tag():
    assert resolve_zone("visitor_entry")["id"] == "visitor_entry"
    assert resolve_zone("入口")["id"] == "visitor_entry"
    assert resolve_zone("古树")["id"] == "banyan_grove"
    assert resolve_zone("不存在") is None


def test_map_payload_highlights_requested_zone():
    payload = build_map_payload("古树")

    assert payload["focus_zone"] == "banyan_grove"
    assert len(payload["zones"]) == 5
    assert payload["trails"]
    assert "不是实景地图" in payload["source_note"]


def test_show_virtual_map_returns_zone_summary():
    result = tool_map()["show_virtual_map"].invoke({"focus_zone": "古树"})

    assert "虚拟二维" in result
    assert "古树森林体验区" in result
    assert "不是导航" in result


def test_plan_walking_route_resolves_chinese_names():
    result = tool_map()["plan_walking_route"].invoke(
        {"start_zone": "入口与游客服务区", "end_zone": "古树森林体验区"}
    )

    assert "示意路线" in result
    assert "古树森林体验区" in result
    assert "km" in result


def test_recommend_routes_uses_configured_candidates():
    result = tool_map()["recommend_wellness_routes"].invoke(
        {"goals": "溪谷", "duration_minutes": 120, "mobility": "常规步行"}
    )

    assert "森林溪谷体验线" in result
    assert "演示" in result


def test_recommend_routes_rejects_invalid_duration():
    result = tool_map()["recommend_wellness_routes"].invoke(
        {"goals": "轻松", "duration_minutes": 5}
    )

    assert "15 至 600" in result


def test_get_zone_information_marks_unverified_data():
    result = tool_map()["get_zone_information"].invoke({"zone": "入口"})

    assert "入口与游客服务区" in result
    assert "未现场核验" in result


def test_knowledge_search_returns_source_and_title():
    result = tool_map()["search_forest_knowledge"].invoke({"query": "森林浴"})

    assert "森林环境与健康科普" in result
    assert "森林环境与健康科普.txt" in result
    assert "不能替代治疗" in result
