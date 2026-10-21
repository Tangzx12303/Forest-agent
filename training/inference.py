"""本地工具路由和质量判定头的延迟推理服务。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config.settings import TOOL_ROUTER_HINT_THRESHOLD, TOOL_ROUTER_MODEL_PATH
from training.features import encode_batch


class ToolRouterService:
    """首次使用时加载 PyTorch 检查点；不可用时返回空结果。"""

    def __init__(self, path: Path | str | None = None, threshold: float | None = None) -> None:
        self.path = Path(path) if path is not None else Path(TOOL_ROUTER_MODEL_PATH)
        self.threshold = threshold if threshold is not None else TOOL_ROUTER_HINT_THRESHOLD
        self._loaded = False
        self._torch = None
        self._model = None
        self._tools: list[str] = []
        self._buckets = 0
        self.load_error = ""

    def _load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            import torch

            from training.model import ToolRouter
        except ImportError:
            self.load_error = "PyTorch 未安装"
            return False
        if not self.path.exists():
            self.load_error = f"模型文件不存在：{self.path}"
            return False
        try:
            try:
                checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(self.path, map_location="cpu")
            self._tools = list(checkpoint["tools"])
            model = ToolRouter(
                len(self._tools),
                buckets=int(checkpoint["buckets"]),
                embedding_dim=int(checkpoint.get("embedding_dim", 64)),
                hidden=int(checkpoint.get("hidden", 96)),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self._model = model
            self._buckets = int(checkpoint["buckets"])
            self._torch = torch
            return True
        except Exception as exc:
            self.load_error = f"模型加载失败：{exc}"
            return False

    @property
    def available(self) -> bool:
        return self._load()

    def predict_tools(self, query: str, threshold: float | None = None) -> list[dict]:
        if not query.strip() or not self._load():
            return []
        limit = self.threshold if threshold is None else threshold
        torch = self._torch
        flat, offsets = encode_batch([query], self._buckets)
        with torch.no_grad():
            ids = torch.tensor(flat, dtype=torch.long)
            offset_tensor = torch.tensor(offsets, dtype=torch.long)
            tool_logits, _ = self._model(ids, offset_tensor)
            probabilities = torch.sigmoid(tool_logits)[0].tolist()
        ranked = sorted(zip(self._tools, probabilities), key=lambda item: item[1], reverse=True)
        return [
            {"tool": name, "confidence": round(float(probability), 3)}
            for name, probability in ranked
            if probability >= limit
        ][:3]

    def score_quality(self, query: str, answer: str) -> float | None:
        if not answer.strip() or not self._load():
            return None
        torch = self._torch
        flat, offsets = encode_batch([f"{query}\n{answer}"], self._buckets)
        with torch.no_grad():
            ids = torch.tensor(flat, dtype=torch.long)
            offset_tensor = torch.tensor(offsets, dtype=torch.long)
            _, quality_logit = self._model(ids, offset_tensor)
            return round(float(torch.sigmoid(quality_logit).item()), 3)


@lru_cache(maxsize=1)
def get_tool_router() -> ToolRouterService:
    return ToolRouterService()
