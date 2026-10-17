"""Train the optional local tool router / quality scorer.

Example:
    pip install -r requirements.txt
    set TRAINING_CAPTURE=true        # PowerShell: $env:TRAINING_CAPTURE="true"
    python -m training.train --epochs 40

Training data comes from two sources:
1. agent/golden_cases.json (query -> expected tools);
2. data/training/turns.jsonl captured from real chat turns, including
   deterministic quality checks. Raw chat text is only logged when
   TRAINING_CAPTURE is enabled.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from agent.evaluation import GOLDEN_PATH, load_cases
from config.settings import TOOL_ROUTER_MODEL_PATH, TRAINING_LOG_PATH
from training.capture import load_turns
from training.features import DEFAULT_BUCKETS, encode_batch
from training.schema import TOOL_NAMES


def build_rows(golden_cases: list[dict], turns: list[dict]) -> list[dict]:
    rows = []
    for case in golden_cases:
        rows.append(
            {
                "query": case["query"],
                "tools": [name for name in case.get("expected_tools", []) if name in TOOL_NAMES],
                "answer": "",
                "quality": None,
            }
        )
    for turn in turns:
        query = str(turn.get("query", "")).strip()
        if not query:
            continue
        checks = turn.get("checks") or {}
        rows.append(
            {
                "query": query,
                "tools": [name for name in turn.get("tools_used", []) if name in TOOL_NAMES],
                "answer": str(turn.get("answer", "")),
                "quality": 1.0 if checks.get("overall_pass") else 0.0,
            }
        )
    return rows


def _batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _encode(torch, texts: list[str], buckets: int, device):
    flat, offsets = encode_batch(texts, buckets)
    return (
        torch.tensor(flat, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
    )


def _evaluate(torch, model, rows: list[dict], buckets: int, device) -> dict:
    if not rows:
        return {}
    model.eval()
    with torch.no_grad():
        ids, offsets = _encode(torch, [row["query"] for row in rows], buckets, device)
        tool_logits, _ = model(ids, offsets)
        probabilities = torch.sigmoid(tool_logits).cpu()
    labels = torch.tensor(
        [[1.0 if name in row["tools"] else 0.0 for name in TOOL_NAMES] for row in rows]
    )
    predictions = (probabilities >= 0.5).float()
    metrics = {}
    f1s = []
    for index, name in enumerate(TOOL_NAMES):
        tp = float(((predictions[:, index] == 1) & (labels[:, index] == 1)).sum())
        fp = float(((predictions[:, index] == 1) & (labels[:, index] == 0)).sum())
        fn = float(((predictions[:, index] == 0) & (labels[:, index] == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        metrics[name] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": int(labels[:, index].sum().item()),
        }
    metrics["macro_f1"] = round(sum(f1s) / len(f1s), 3)

    quality_rows = [row for row in rows if row["answer"]]
    if quality_rows:
        q_ids, q_offsets = _encode(
            torch,
            [f"{row['query']}\n{row['answer']}" for row in quality_rows],
            buckets,
            device,
        )
        _, q_logits = model(q_ids, q_offsets)
        q_predictions = (torch.sigmoid(q_logits).cpu() >= 0.5).float()
        q_labels = torch.tensor([row["quality"] for row in quality_rows])
        metrics["quality_accuracy"] = round(
            float((q_predictions == q_labels).float().mean().item()), 3
        )
    return metrics


def train_model(
    rows: list[dict],
    *,
    epochs: int = 30,
    buckets: int = DEFAULT_BUCKETS,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """Train and return (checkpoint, metrics). Raises ImportError without torch."""
    import torch
    from torch import nn

    from training.model import ToolRouter

    rows = [row for row in rows if row.get("query")]
    if not rows:
        raise ValueError("没有可用训练样本。")

    random.seed(seed)
    torch.manual_seed(seed)
    shuffled = list(rows)
    random.shuffle(shuffled)
    if len(shuffled) >= 5:
        split = max(1, int(len(shuffled) * 0.8))
        train_rows, validation_rows = shuffled[:split], shuffled[split:]
    else:
        train_rows, validation_rows = shuffled, shuffled

    model = ToolRouter(len(TOOL_NAMES), buckets=buckets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tool_loss_fn = nn.BCEWithLogitsLoss()
    quality_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(max(1, epochs)):
        model.train()
        random.shuffle(train_rows)
        epoch_loss = 0.0
        for batch in _batches(train_rows, max(1, batch_size)):
            optimizer.zero_grad()
            ids, offsets = _encode(torch, [row["query"] for row in batch], buckets, device)
            tool_logits, _ = model(ids, offsets)
            labels = torch.tensor(
                [[1.0 if name in row["tools"] else 0.0 for name in TOOL_NAMES] for row in batch],
                device=device,
            )
            loss = tool_loss_fn(tool_logits, labels)

            quality_batch = [row for row in batch if row["answer"]]
            if quality_batch:
                q_ids, q_offsets = _encode(
                    torch,
                    [f"{row['query']}\n{row['answer']}" for row in quality_batch],
                    buckets,
                    device,
                )
                _, q_logits = model(q_ids, q_offsets)
                q_labels = torch.tensor(
                    [row["quality"] if row["quality"] is not None else 1.0 for row in quality_batch],
                    device=device,
                )
                loss = loss + 0.5 * quality_loss_fn(q_logits, q_labels)

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1}/{epochs} loss={epoch_loss / max(1, len(train_rows)):.4f}")

    metrics = _evaluate(torch, model, validation_rows, buckets, device)
    checkpoint = {
        "state_dict": model.state_dict(),
        "tools": TOOL_NAMES,
        "buckets": buckets,
        "embedding_dim": model.embedding_dim,
        "hidden": model.hidden,
        "metrics": metrics,
        "samples": len(rows),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    return checkpoint, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练本地 PyTorch 工具路由/质量模型。")
    parser.add_argument("--data", default=str(TRAINING_LOG_PATH), help="对话记录 JSONL 路径")
    parser.add_argument("--output", default=str(TOOL_ROUTER_MODEL_PATH), help="模型输出路径")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument("--no-golden", action="store_true", help="不使用金标评测集")
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("尚未安装 PyTorch。请先运行：python -m pip install -r requirements.txt")
        return 2

    turns = load_turns(args.data)
    golden = [] if args.no_golden else load_cases(GOLDEN_PATH)
    rows = build_rows(golden, turns)
    if not rows:
        print(
            "没有训练样本。可在 .env 设置 TRAINING_CAPTURE=true 记录真实对话，"
            "或保留 golden_cases.json 作为基础训练集。"
        )
        return 2
    print(f"训练样本：{len(rows)}（金标 {len(golden)}，对话记录 {len(turns)}）")

    checkpoint, metrics = train_model(
        rows,
        epochs=args.epochs,
        buckets=args.buckets,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(checkpoint, output)
    print(f"模型已保存：{output}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
