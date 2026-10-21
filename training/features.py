"""确定性哈希字符/词语 n-gram 特征，无需词汇表文件。"""
from __future__ import annotations

import hashlib
import re

DEFAULT_BUCKETS = 2**13
WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def char_ngrams(text: str, n_min: int = 2, n_max: int = 4):
    normalized = normalize(text)
    for size in range(n_min, n_max + 1):
        for index in range(max(0, len(normalized) - size + 1)):
            token = normalized[index:index + size]
            if token.strip():
                yield token


def _bucket(token: str, buckets: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


def hashed_ids(text: str, buckets: int = DEFAULT_BUCKETS) -> list[int]:
    ids = [_bucket(token, buckets) for token in char_ngrams(text)]
    ids.extend(_bucket(f"w:{word}", buckets) for word in WORD_RE.findall(normalize(text)))
    if not ids:
        ids.append(_bucket("<empty>", buckets))
    return ids


def encode_batch(texts: list[str], buckets: int = DEFAULT_BUCKETS) -> tuple[list[int], list[int]]:
    """展平特征编号，并返回供 torch.nn.EmbeddingBag 使用的 (flat_ids, offsets)。"""
    flat: list[int] = []
    offsets: list[int] = []
    for text in texts:
        offsets.append(len(flat))
        flat.extend(hashed_ids(text, buckets))
    return flat, offsets
