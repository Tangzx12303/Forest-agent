"""Append-only local capture of turn records for the optional training loop."""
from __future__ import annotations

import json
from pathlib import Path

from config.settings import TRAINING_CAPTURE_ENABLED, TRAINING_LOG_PATH
from training.schema import TurnRecord


def append_turn(record: TurnRecord | dict, path: Path | str | None = None) -> bool:
    """Append one JSONL record.

    When ``path`` is omitted the global TRAINING_CAPTURE switch must be enabled.
    Passing an explicit path is intended for tests and manual imports.
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
