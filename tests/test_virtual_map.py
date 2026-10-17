import pytest

from services.virtual_map import calculate_virtual_route


def test_virtual_route_uses_shortest_configured_trail_path():
    route = calculate_virtual_route("visitor_entry", "forest_ridge")

    assert route["zone_ids"] == ["visitor_entry", "banyan_grove", "forest_ridge"]
    assert route["coordinates"][0] == [130, 500]
    assert route["coordinates"][-1] == [820, 170]
    assert route["distance_km"] == 1.9
    assert route["source"] == "基地虚拟二维示意步道"


def test_virtual_route_supports_reverse_direction():
    route = calculate_virtual_route("forest_ridge", "visitor_entry")

    assert route["zone_ids"] == ["forest_ridge", "banyan_grove", "visitor_entry"]
    assert route["coordinates"][0] == [820, 170]
    assert route["coordinates"][-1] == [130, 500]


@pytest.mark.parametrize(
    ("start_id", "end_id", "message"),
    [
        ("missing", "visitor_entry", "已有的起点和终点"),
        ("visitor_entry", "visitor_entry", "不能是同一个分区"),
    ],
)
def test_virtual_route_rejects_invalid_endpoints(start_id, end_id, message):
    with pytest.raises(ValueError, match=message):
        calculate_virtual_route(start_id, end_id)