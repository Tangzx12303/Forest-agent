from agent.route_planner import load_base_data


def test_every_route_references_existing_zones():
    base = load_base_data()
    zone_ids = {zone["id"] for zone in base["zones"]}
    assert base["routes"]
    assert all(set(route["zone_ids"]) <= zone_ids for route in base["routes"])


def test_route_candidates_have_explicit_demo_metadata():
    base = load_base_data()
    assert "演示数据" in base["source_note"]
    assert all(route.get("wheelchair_accessible") is None for route in base["routes"])
