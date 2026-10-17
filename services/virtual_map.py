import heapq

from agent.route_planner import load_base_data


def calculate_virtual_route(start_id: str, end_id: str) -> dict:
    base = load_base_data()
    zones = {zone["id"]: zone for zone in base["zones"]}
    if start_id not in zones or end_id not in zones:
        raise ValueError("请选择地图中已有的起点和终点分区。")
    if start_id == end_id:
        raise ValueError("起点和终点不能是同一个分区。")

    graph: dict[str, list[tuple[str, int, list[list[int]]]]] = {zone_id: [] for zone_id in zones}
    for trail in base["trails"]:
        source, target = trail["from"], trail["to"]
        points = trail["points"]
        distance = int(trail["distance_m"])
        graph[source].append((target, distance, points))
        graph[target].append((source, distance, list(reversed(points))))

    queue = [(0, start_id)]
    distances = {start_id: 0}
    previous: dict[str, tuple[str, list[list[int]]]] = {}
    while queue:
        distance, current = heapq.heappop(queue)
        if current == end_id:
            break
        if distance != distances[current]:
            continue
        for neighbor, edge_distance, points in graph[current]:
            candidate = distance + edge_distance
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                previous[neighbor] = (current, points)
                heapq.heappush(queue, (candidate, neighbor))

    if end_id not in previous:
        raise ValueError("这两个分区之间暂无已配置的示意步道。")

    segments = []
    zone_ids = [end_id]
    current = end_id
    while current != start_id:
        parent, points = previous[current]
        segments.append(points)
        zone_ids.append(parent)
        current = parent
    segments.reverse()
    coordinates = []
    for segment in segments:
        coordinates.extend(segment if not coordinates else segment[1:])

    distance_m = distances[end_id]
    return {
        "zone_ids": list(reversed(zone_ids)),
        "coordinates": coordinates,
        "distance_km": distance_m / 1000,
        "duration_minutes": max(1, round(distance_m / 1000 / 3 * 60)),
        "instructions": [f"从“{zones[zone_id]['name']}”前往下一分区" for zone_id in list(reversed(zone_ids))[1:]],
        "source": "基地虚拟二维示意步道",
    }