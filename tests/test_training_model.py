import pytest

torch = pytest.importorskip("torch")

from training.model import ToolRouter  # noqa: E402
from training.schema import TOOL_NAMES  # noqa: E402
from training.train import train_model  # noqa: E402


def test_model_forward_returns_tool_and_quality_outputs():
    model = ToolRouter(len(TOOL_NAMES), buckets=256, embedding_dim=16, hidden=24)
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)

    tool_logits, quality_logit = model(ids, offsets)

    assert tool_logits.shape == (2, len(TOOL_NAMES))
    assert quality_logit.shape == (2,)


def test_train_model_runs_on_tiny_dataset():
    rows = [
        {
            "query": "展示基地地图",
            "tools": ["show_virtual_map"],
            "answer": "已展示虚拟示意地图。",
            "quality": 1.0,
        },
        {
            "query": "推荐一条轻松路线",
            "tools": ["recommend_wellness_routes"],
            "answer": "推荐演示路线，须以现场核实为准。",
            "quality": 1.0,
        },
        {
            "query": "森林浴有什么证据",
            "tools": ["search_forest_knowledge"],
            "answer": "据资料，现有证据只支持短期放松。",
            "quality": 1.0,
        },
        {
            "query": "户外中暑怎么办",
            "tools": ["search_forest_knowledge"],
            "answer": "据资料，应立即急救并就医。",
            "quality": 1.0,
        },
        {
            "query": "从入口到古树怎么走",
            "tools": ["plan_walking_route"],
            "answer": "示意路线约 0.85 km，不是导航数据。",
            "quality": 1.0,
        },
    ]

    checkpoint, metrics = train_model(
        rows, epochs=2, buckets=512, batch_size=2, learning_rate=5e-3
    )

    assert checkpoint["samples"] == 5
    assert checkpoint["tools"] == TOOL_NAMES
    assert "macro_f1" in metrics
    assert "quality_accuracy" in metrics
