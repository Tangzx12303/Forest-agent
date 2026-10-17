import json
from pathlib import Path

from config.settings import BASE_ZONES_PATH

INTENSITY_RANK = {"低": 1, "中": 2, "高": 3}


def load_base_data(path: Path = BASE_ZONES_PATH) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def select_routes(
    goal: str,
    duration_minutes: int,
    mobility: str = "常规步行",
    path: Path = BASE_ZONES_PATH,
) -> tuple[dict, list[dict]]:
    """Rank only operator-configured route candidates; never invent route geometry."""
    base = load_base_data(path)
    zone_by_id = {zone["id"]: zone for zone in base["zones"]}
    keywords = {word.strip().lower() for word in goal.replace("，", ",").split(",") if word.strip()}
    max_intensity = {"常规步行": 3, "慢行/需休息": 1, "轮椅/行动不便": 0}.get(mobility, 1)
    candidates = []

    for route in base["routes"]:
        if route["duration_minutes"] > duration_minutes:
            continue
        if mobility == "轮椅/行动不便" and route.get("wheelchair_accessible") is not True:
            continue
        route_intensity = INTENSITY_RANK.get(route["intensity"], 3)
        if route_intensity > max_intensity:
            continue

        tags = {tag.lower() for tag in route.get("tags", [])}
        matched = sum(1 for keyword in keywords if any(keyword in tag or tag in keyword for tag in tags))
        candidates.append({**route, "zones": [zone_by_id[item] for item in route["zone_ids"]],
                           "preference_matches": matched})

    candidates.sort(key=lambda item: (-item["preference_matches"], item["duration_minutes"]))
    return base, candidates[:3]


def format_route_recommendation(
    goal: str,
    duration_minutes: int,
    mobility: str = "常规步行",
) -> str:
    base, routes = select_routes(goal, duration_minutes, mobility)
    if not routes:
        if mobility == "轮椅/行动不便":
            return ("当前配置没有经核实的轮椅/行动不便可通行路线，因此不能安全推荐具体线路。"
                    "请联系基地确认无障碍路段、坡度、台阶和卫生间等信息。")
        return f"当前路线配置中没有符合 {duration_minutes} 分钟和“{mobility}”条件的路线。可增加时长或选择更轻松的活动目标。"

    lines = [f"**基地：** {base['base_name']}",
             f"**需求：** {goal or '轻松森林漫步'}；时长不超过 {duration_minutes} 分钟；活动能力：{mobility}", ""]
    for index, route in enumerate(routes, 1):
        zone_names = " → ".join(zone["name"] for zone in route["zones"])
        lines.extend([
            f"### {index}. {route['name']}",
            f"- **距离 / 时长 / 强度：** 约 {route['distance_km']} km / {route['duration_minutes']} 分钟 / {route['intensity']}",
            f"- **功能分区：** {zone_names}",
            f"- **分区说明：** {'；'.join(zone['function'] for zone in route['zones'])}",
            f"- **路况提示：** {route['terrain_note']}",
            "",
        ])
    lines.append(f"> **数据声明：** {base['source_note']}")
    lines.append("> 出发前请以基地官方公告核实开放状态、天气、路况、饮水点与医疗救助信息；本工具不是导航或医疗服务。")
    return "\n".join(lines)
