"""PyTorch 多任务模型：多标签工具路由头和二分类回答质量判定头。"""
from __future__ import annotations

import torch
from torch import nn

from training.features import DEFAULT_BUCKETS


class ToolRouter(nn.Module):
    def __init__(
        self,
        num_tools: int,
        buckets: int = DEFAULT_BUCKETS,
        embedding_dim: int = 32,
        hidden: int = 48,
    ) -> None:
        super().__init__()
        self.num_tools = num_tools
        self.buckets = buckets
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.embedding = nn.EmbeddingBag(buckets, embedding_dim, mode="mean")
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.tool_head = nn.Linear(hidden, num_tools)
        self.quality_head = nn.Linear(hidden, 1)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor):
        features = self.trunk(self.embedding(ids, offsets))
        return self.tool_head(features), self.quality_head(features).squeeze(-1)
