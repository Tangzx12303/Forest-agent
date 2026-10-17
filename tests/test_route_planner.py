from agent.route_planner import format_route_recommendation, load_base_data, select_routes


def test_route_selection_matches_goal_and_time():
    _, routes = select_routes("植物,轻松", 90)
    assert routes
    assert routes[0]["id"] == "gentle_nature"
    assert all(route["duration_minutes"] <= 90 for route in routes)


def test_route_selection_changes_with_wellness_goal():
    _, stream_routes = select_routes("溪谷", 120)
    _, plant_routes = select_routes("植物,亲子", 90)

    assert stream_routes[0]["id"] == "forest_stream"
    assert plant_routes[0]["id"] == "gentle_nature"


def test_demo_zones_have_virtual_map_positions():
    base = load_base_data()

    assert all(zone.get("map_position") for zone in base["zones"])
    assert all(len(zone["map_position"]) == 2 for zone in base["zones"])


def test_slow_pace_excludes_moderate_and_high_routes():
    _, routes = select_routes("森林步道", 240, "慢行/需休息")
    assert routes
    assert all(route["intensity"] == "低" for route in routes)


def test_unverified_accessibility_is_not_claimed():
    response = format_route_recommendation("自然观察", 240, "轮椅/行动不便")
    assert "没有经核实的轮椅" in response
