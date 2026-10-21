"""训练必需的本地工具路由和回答质量评分模型。"""
from __future__ import annotations

import argparse
import json
import random
import copy
from datetime import datetime, timezone
from pathlib import Path

from agent.evaluation import GOLDEN_PATH, load_cases
from config.settings import (
    TOOL_ROUTER_MODEL_PATH,
    TRAINING_LOG_PATH,
    TRAINING_SAMPLES_PATH,
)
from training.capture import load_turns
from training.features import DEFAULT_BUCKETS, encode_batch
from training.schema import TOOL_NAMES, evaluate_turn


def load_seed_samples(path: Path | str = TRAINING_SAMPLES_PATH) -> list[dict]:
    samples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        tools = [name for name in item.get("tools", []) if name in TOOL_NAMES]
        answer = str(item.get("answer", ""))
        category = str(item.get("category", "forest"))
        checks = evaluate_turn(query, category, tools, answer) if answer else {}
        samples.append({
            "query": query,
            "tools": tools,
            "answer": answer,
            "quality": float(checks["overall_pass"]) if answer else None,
        })
    return samples


def build_rows(
    golden_cases: list[dict],
    turns: list[dict],
    seed_samples: list[dict] | None = None,
) -> list[dict]:
    rows = []
    for sample in seed_samples or []:
        rows.append({
            "query": sample["query"],
            "tools": [name for name in sample.get("tools", []) if name in TOOL_NAMES],
            "answer": str(sample.get("answer", "")),
            "quality": sample.get("quality"),
        })
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


def _split_rows(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    if len(rows) < 10:
        return list(rows), list(rows)

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * 0.2))
    validation_rows = []
    validation_ids = set()

    def add_representative(predicate) -> None:
        candidates = [row for row in shuffled if predicate(row)]
        if candidates and len(candidates) > 1:
            candidate = next(
                (row for row in candidates if id(row) not in validation_ids), None
            )
            if candidate is not None and len(validation_rows) < validation_size:
                validation_rows.append(candidate)
                validation_ids.add(id(candidate))

    for name in TOOL_NAMES:
        add_representative(lambda row, tool=name: tool in row["tools"])
    for label in (0.0, 1.0):
        add_representative(
            lambda row, expected=label: row["quality"] == expected
            and row["answer"]
        )
    for row in shuffled:
        if len(validation_rows) >= validation_size:
            break
        if id(row) not in validation_ids:
            validation_rows.append(row)
            validation_ids.add(id(row))

    train_rows = [row for row in shuffled if id(row) not in validation_ids]
    return train_rows, validation_rows


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
    """执行训练并返回（检查点、指标）；未安装 torch 时抛出 ImportError。"""
    import torch
    from torch import nn

    from training.model import ToolRouter

    rows = [row for row in rows if row.get("query")]
    if not rows:
        raise ValueError("没有可用训练样本。")

    random.seed(seed)
    torch.manual_seed(seed)
    train_rows, validation_rows = _split_rows(rows, seed)

    model = ToolRouter(len(TOOL_NAMES), buckets=buckets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    positive_counts = [
        sum(name in row["tools"] for row in train_rows)
        for name in TOOL_NAMES
    ]
    negative_counts = [len(train_rows) - count for count in positive_counts]
    tool_weights = torch.tensor(
        [min(10.0, negatives / max(1, positives))
         for positives, negatives in zip(positive_counts, negative_counts)],
        dtype=torch.float32,
        device=device,
    )
    tool_loss_fn = nn.BCEWithLogitsLoss(pos_weight=tool_weights)
    quality_loss_fn = nn.BCEWithLogitsLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    stale_epochs = 0
    patience = 3
    validation_rows = validation_rows or train_rows

    def validation_loss() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in _batches(validation_rows, max(1, batch_size)):
                ids, offsets = _encode(torch, [row["query"] for row in batch], buckets, device)
                tool_logits, _ = model(ids, offsets)
                labels = torch.tensor(
                    [[1.0 if name in row["tools"] else 0.0 for name in TOOL_NAMES] for row in batch],
                    device=device,
                )
                loss = tool_loss_fn(tool_logits, labels)
                quality_batch = [row for row in batch if row["answer"] and row["quality"] is not None]
                if quality_batch:
                    q_ids, q_offsets = _encode(
                        torch,
                        [f"{row['query']}\n{row['answer']}" for row in quality_batch],
                        buckets,
                        device,
                    )
                    _, q_logits = model(q_ids, q_offsets)
                    q_labels = torch.tensor([row["quality"] for row in quality_batch], device=device)
                    loss = loss + quality_loss_fn(q_logits, q_labels)
                losses.append(float(loss.item()))
        return sum(losses) / max(1, len(losses))

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

            quality_batch = [row for row in batch if row["answer"] and row["quality"] is not None]
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
                loss = loss + quality_loss_fn(q_logits, q_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        current_validation_loss = validation_loss()
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    metrics = _evaluate(torch, model, validation_rows, buckets, device)
    checkpoint = {
        "state_dict": model.state_dict(),
        "tools": TOOL_NAMES,
        "buckets": buckets,
        "embedding_dim": model.embedding_dim,
        "hidden": model.hidden,
        "metrics": metrics,
        "samples": len(rows),
        "epochs_trained": epoch + 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    return checkpoint, metrics


def train_default_model(
    *,
    output: Path | str = TOOL_ROUTER_MODEL_PATH,
    epochs: int = 12,
    batch_size: int = 16,
) -> dict:
    """使用项目内置数据训练，并保存应用启动所需的检查点。"""
    import torch

    rows = build_rows(
        load_cases(GOLDEN_PATH),
        load_turns(TRAINING_LOG_PATH),
        load_seed_samples(),
    )
    if not rows:
        raise RuntimeError("内置训练样本为空，无法启动本地模型。")
    checkpoint, metrics = train_model(rows, epochs=epochs, batch_size=batch_size)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"本地工具路由模型已训练：{len(rows)} 条样本，验证指标 {metrics}")
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练本地 PyTorch 工具路由/质量模型。")
    parser.add_argument("--data", default=str(TRAINING_LOG_PATH), help="对话记录 JSONL 路径")
    parser.add_argument("--output", default=str(TOOL_ROUTER_MODEL_PATH), help="模型输出路径")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("尚未安装 PyTorch。请先运行：python -m pip install -r requirements.txt")
        return 2

    turns = load_turns(args.data)
    golden = load_cases(GOLDEN_PATH)
    seed_samples = load_seed_samples()
    rows = build_rows(golden, turns, seed_samples)
    if not rows:
        print("内置训练样本为空，无法启动本地模型训练。")
        return 2
    print(
        f"训练样本：{len(rows)}（内置 {len(seed_samples)}，"
        f"金标 {len(golden)}，本地记录 {len(turns)}）"
    )

    checkpoint, metrics = train_model(
        rows,
        epochs=args.epochs,
        buckets=args.buckets,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import torch

    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"模型已保存：{output}")
    print(f"训练轮数：{checkpoint['epochs_trained']}/{args.epochs}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
