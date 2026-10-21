"""仅追加写入本地轮次记录，供训练流程使用。"""
from __future__ import annotations

import json
from pathlib import Path

from config.settings import TRAINING_CAPTURE_ENABLED, TRAINING_LOG_PATH
from training.schema import TurnRecord


def append_turn(record: TurnRecord | dict, path: Path | str | None = None) -> bool:
    """追加一条 JSONL 记录。

    未指定 ``path`` 时，必须启用全局 TRAINING_CAPTURE 开关。
    显式传入路径用于测试和手动导入。
    """
    if path is None and not TRAINING_CAPTURE_ENABLED:
        return False
    data = record.to_dict() if isinstance(record, TurnRecord) else dict(record)
    data["query"] = str(data.get("query", ""))[:4000]
    data["answer"] = str(data.get("answer", ""))[:8000]
    target = Path(path) if path is not None else Path(TRAINING_LOG_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")
    return True


def load_turns(path: Path | str | None = None) -> list[dict]:
    target = Path(path) if path is not None else Path(TRAINING_LOG_PATH)
    if not target.exists():
        return []
    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("query") and isinstance(item.get("tools_used", []), list):
            records.append(item)
    return records
