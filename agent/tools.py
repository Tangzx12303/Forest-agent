"""Agent tool registry.

Tools return plain text for the model and, for the map tool, a deterministic
payload that the API can stream to the browser as a UI event.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agent.route_planner import format_route_recommendation, load_base_data
from services.virtual_map import calculate_virtual_route

TOOL_LABELS = {
    "search_forest_knowledge": "知识库检索",
    "show_virtual_map": "虚拟地图",
    "recommend_wellness_routes": "疗养路线推荐",
    "plan_walking_route": "步道计算",
    "get_zone_information": "分区信息",
}

MAP_SOURCE_NOTE = (
    "虚拟平面示意，仅用于分区顺序和内部步道关系展示；不是实景地图、导航数据或现场核验资料。"
)


def resolve_zone(value: str) -> dict | None:
    """Resolve a zone id, Chinese name fragment, or tag to a configured zone."""
    text = (value or "").strip()
    if not text:
        return None
    zones = load_base_data()["zones"]
    for zone in zones:
        if text == zone["id"]:
            return zone
    for zone in zones:
        if text in zone["name"] or zone["name"] in text:
            return zone
    for zone in zones:
        if any(text in tag or tag in text for tag in zone.get("tags", [])):
            return zone
    return None


def build_map_payload(focus_zone: str = "") -> dict:
    """Build the same virtual map payload used by the route page and chat UI."""
    base = load_base_data()
    focus = resolve_zone(focus_zone)
    return {
        "name": "森林康养基地虚拟二维示意图",
        "width": 960,
        "height": 620,
        "base_name": base["base_name"],
        "source_note": MAP_SOURCE_NOTE,
        "zones": base["zones"],
        "trails": base["trails"],
        "focus_zone": focus["id"] if focus else "",
    }


def _zone_hint() -> str:
    zones = load_base_data()["zones"]
    return "、".join(f"{zone['name']}（{zone['id']}）" for zone in zones)


def build_forest_tools(vector_store) -> list[BaseTool]:
    """Create the tool set bound to one vector store instance."""

    @tool
    def search_forest_knowledge(query: str) -> str:
        """检索本地森林康养知识库，覆盖森林生态、森林环境与健康证据、身体活动与慢病安全、
        康复边界、户外急救、特殊人群和基地分区说明。涉及知识、健康或安全问题时优先调用。
        一次问题可以拆成多个子问题多次调用。"""
        docs = vector_store.search(query)
        if not docs:
            return "本地知识库没有检索到相关资料。请说明资料不足，并建议咨询基地官方渠道或医疗专业人员。"
        parts = []
        for index, doc in enumerate(docs, 1):
            title = doc.metadata.get("title") or doc.metadata.get("source") or "本地资料"
            source = doc.metadata.get("source") or "未知来源"
            page = doc.metadata.get("page")
            location = f"{source}，页码 {page}" if page not in (None, "") else source
            parts.append(f"[资料{index}：{title}（{location}）]\n{doc.page_content}")
        return "\n\n".join(parts)

    @tool
    def show_virtual_map(focus_zone: str = "") -> str:
        """在聊天界面展示基地虚拟二维分区与步道示意图，并返回分区概要。用户询问基地地图、
        分区位置、有哪些区域、想同时看地图时调用。focus_zone 可选，填分区中文名或 id；
        地图是演示示意，不是实景导航。"""
        payload = build_map_payload(focus_zone)
        zone_lines = "\n".join(
            f"- {zone['name']}（{zone['id']}，强度：{zone['intensity']}）：{zone['function']}"
            for zone in payload["zones"]
        )
        focus_text = ""
        if focus_zone and payload["focus_zone"]:
            focus_text = f"\n已聚焦分区：{payload['focus_zone']}。"
        elif focus_zone:
            focus_text = f"\n没有找到“{focus_zone}”，已在界面展示全部区域。"
        return (
            "已在界面展示基地虚拟二维示意地图（演示数据，不是导航）。\n"
            f"基地：{payload['base_name']}\n分区概要：\n{zone_lines}{focus_text}\n"
            "点位、连线、距离与无障碍信息均未现场核验，实际出行须以园方资料和现场标识为准。"
        )

    @tool
    def recommend_wellness_routes(
        goals: str,
        duration_minutes: int,
        mobility: str = "常规步行",
    ) -> str:
        """根据体验偏好、可用时长和步行能力，从已配置线路中筛选疗养顺序。
        goals 用逗号或顿号分隔，例如“轻松,植物,亲子”；duration_minutes 为 15 至 600 的整数；
        mobility 可选“常规步行”“慢行/需休息”“轮椅/行动不便”。不得编造配置外路线。"""
        if not 15 <= duration_minutes <= 600:
            return "可用时长应在 15 至 600 分钟之间。请让用户确认预计步行时间。"
        return format_route_recommendation(goals, duration_minutes, mobility)

    @tool
    def plan_walking_route(start_zone: str, end_zone: str) -> str:
        """计算两个基地分区之间的最短示意步道，返回途经分区、距离和预计时间。
        start_zone 与 end_zone 可填中文名称或 id，例如“入口与游客服务区”或“visitor_entry”。
        用户询问从某区走到某区、两点之间怎么走时调用。"""
        start = resolve_zone(start_zone)
        end = resolve_zone(end_zone)
        if not start or not end:
            return f"没有找到对应分区。可用分区：{_zone_hint()}。"
        if start["id"] == end["id"]:
            return "起点和终点不能是同一个分区，请选择两个不同区域。"
        try:
            route = calculate_virtual_route(start["id"], end["id"])
        except ValueError as exc:
            return str(exc)
        zone_by_id = {zone["id"]: zone for zone in load_base_data()["zones"]}
        names = [zone_by_id[zone_id]["name"] for zone_id in route["zone_ids"]]
        return (
            f"示意路线：{' → '.join(names)}\n"
            f"距离：约 {route['distance_km']:.2f} km；预计步行：约 {route['duration_minutes']} 分钟；"
            f"途经 {len(names) - 1} 段已配置步道。\n"
            "说明：路线基于虚拟步道网络和演示距离，不代表实际道路、坡度、台阶或开放状态。"
        )

    @tool
    def get_zone_information(zone: str) -> str:
        """查询单个基地分区的功能定位、适合活动和核实状态。zone 填分区中文名或 id。
        用户询问某个分区有什么、适合谁、是否无障碍时调用。"""
        target = resolve_zone(zone)
        if not target:
            return f"没有找到对应分区。可用分区：{_zone_hint()}。"
        tags = "、".join(target.get("tags", [])) or "无"
        verified = "已核验" if target.get("verified") is True else "未现场核验（演示数据）"
        return (
            f"分区：{target['name']}（{target['id']}）\n"
            f"功能定位：{target['function']}\n"
            f"适合主题：{tags}\n"
            f"活动强度：{target.get('intensity', '未标注')}\n"
            f"数据状态：{verified}\n"
            "无障碍、开放状态、饮水和急救设施需向基地现场核实，系统不作承诺。"
        )

    return [
        search_forest_knowledge,
        show_virtual_map,
        recommend_wellness_routes,
        plan_walking_route,
        get_zone_information,
    ]
