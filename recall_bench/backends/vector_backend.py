# -*- coding: utf-8 -*-
"""向量存储适配器：LAAP MemoryVectorStore + 零模型依赖的哈希嵌入。

设计
----
- 底层用 ``laap.engine.memory.vector_store.MemoryVectorStore``（384 维 cosine）；
- 文本→向量用 sklearn ``HashingVectorizer``（字符 2-3 gram 哈希），
  无需下载 embedding 模型——任何环境可复现；
- 语义层效果取决于 n-gram 哈希的区分度：它捕捉词汇重叠，不是深层语义，
  但与 LIKE 相比已覆盖"措辞部分重叠"的召回。
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from recall_bench.backends.base import MemoryItem

_EMBED_DIM = 384


class VectorBackend:
    """包装 MemoryVectorStore 的向量检索后端。"""

    name = "laap_vector"

    def __init__(self, vector_store=None) -> None:
        if vector_store is not None:
            self._vs = vector_store
        else:
            from laap.engine.memory.vector_store import MemoryVectorStore

            self._vs = MemoryVectorStore(dimension=_EMBED_DIM)
        self._hasher = HashingVectorizer(
            n_features=_EMBED_DIM,
            analyzer="char_wb",
            ngram_range=(2, 3),
            alternate_sign=False,
        )
        self._docs: dict = {}  # vs_id -> content

    # ── 协议 ────────────────────────────────────────────────────────────

    def start_session(self, title: str = "") -> str:
        return title

    def store(self, role: str, content: str, tags: str = "") -> int:
        vec = self._embed(content)
        vid = self._vs.store(
            type(self)._record(vid=str(len(self._docs) + 1), vector=vec, content=content)
        )
        self._docs[vid] = content
        return int(vid)

    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        q = self._embed(query)
        records = self._vs.search(q, top_k=limit)
        results = []
        for r in records:
            payload = getattr(r, "payload", {}) or {}
            content = payload.get("content") or self._docs.get(r.id, "")
            if content:
                results.append(MemoryItem(id=int(r.id) if str(r.id).isdigit() else len(results) + 1,
                                          content=content))
        return results

    def cleanup(self, ids: List[int], session_id: str = "") -> None:
        for i in ids:
            self._vs.delete(str(i))
            self._docs.pop(str(i), None)

    # ── 内部 ────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        return self._hasher.transform([text]).toarray().ravel().astype(np.float64)

    @staticmethod
    def _record(vid: str, vector: np.ndarray, content: str):
        from laap.engine.memory.vector_store import VectorRecord

        return VectorRecord(id=vid, vector=vector, payload={"content": content})
