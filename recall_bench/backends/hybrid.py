# -*- coding: utf-8 -*-
"""混合检索后端：LIKE 精确层 + 字符 n-gram TF-IDF 语义层。

设计动机（对应 LAAP 记忆检索升级 P0-①）：
- 纯 LIKE 子串匹配的 recall 基线仅 25%（生产）/ 10%（隔离）——瓶颈是匹配机制；
- 本后端在任意 MemoryBackend 之上叠加 TF-IDF 语义近似层，无需分词器
  （中文用字符 2-3 gram），零第三方新依赖（numpy + sklearn 即可）；
- 包装模式：不修改底层后端，符合"评估器外置/最小侵入"原则。

用法::

    from recall_bench.backends.laap_vault import LaapVaultBackend
    from recall_bench.backends.hybrid import HybridBackend

    base = LaapVaultBackend(db_path=tmp_db)
    hybrid = HybridBackend(base)
    report = run_benchmark(hybrid, tag="RB_HYBRID")
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from recall_bench.backends.base import MemoryBackend, MemoryItem, ensure_backend


class HybridBackend:
    """在任意 MemoryBackend 之上叠加 TF-IDF 语义检索。

    检索策略（融合排序）：
    1. 底层 LIKE 精确匹配（快速，保精确）；
    2. TF-IDF 字符 n-gram 余弦相似度（语义近似，覆盖措辞差异）；
    3. 结果按 id 去重合并，语义层结果排在 LIKE 命中之后。
    """

    name = "hybrid"

    def __init__(
        self,
        base: MemoryBackend,
        ngram_range: tuple = (2, 3),
        max_features: int = 20000,
    ) -> None:
        self.base = ensure_backend(base)
        self.name = f"hybrid+{self.base.name}"
        self.ngram_range = ngram_range
        self.max_features = max_features
        self._contents: List[str] = []
        self._ids: List[int] = []
        self._tfidf: Optional[TfidfVectorizer] = None
        self._matrix: Optional[np.ndarray] = None
        self._dirty: bool = False

    # ── 协议方法 ────────────────────────────────────────────────────────

    def start_session(self, title: str = "") -> str:
        return self.base.start_session(title=title)

    def store(self, role: str, content: str, tags: str = "") -> int:
        eid = self.base.store(role=role, content=content, tags=tags)
        self._contents.append(content)
        self._ids.append(eid)
        self._dirty = True
        return eid

    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        # 惰性重建：语料有变化时先重建 TF-IDF
        if self._dirty:
            self.rebuild()
        # 1. 底层 LIKE 精确层
        like_hits = self.base.search(query, limit=limit)
        seen: set = {r.id for r in like_hits}
        results: List[MemoryItem] = list(like_hits)

        # 2. TF-IDF 语义层（语料不足时不启用）
        if self._tfidf is not None and self._matrix is not None and len(self._ids) > 0:
            q_vec = self._tfidf.transform([_normalize(query)])
            sims = (self._matrix @ q_vec.T).toarray().ravel()
            top_idx = np.argsort(-sims)[: limit * 2]
            for idx in top_idx:
                if len(results) >= limit:
                    break
                sid = self._ids[idx]
                if sid in seen:
                    continue
                # 低相似度截断，避免噪声
                if sims[idx] < 0.05:
                    break
                seen.add(sid)
                results.append(
                    MemoryItem(
                        id=sid,
                        content=self._contents[idx],
                        timestamp="",
                    )
                )
        return results[:limit]

    def cleanup(self, ids: List[int], session_id: str = "") -> None:
        self.base.cleanup(ids, session_id=session_id)
        # 同步移除本地语料
        id_set = set(ids)
        keep = [(c, i) for c, i in zip(self._contents, self._ids) if i not in id_set]
        self._contents = [c for c, _ in keep]
        self._ids = [i for _, i in keep]
        self._dirty = True
        self.rebuild()

    # ── 语义层维护 ──────────────────────────────────────────────────────

    def rebuild(self) -> None:
        """重建 TF-IDF 索引（全量，简单可靠）。"""
        if len(self._contents) < 3:
            self._tfidf = None
            self._matrix = None
            self._dirty = False
            return
        self._tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )
        self._matrix = self._tfidf.fit_transform([_normalize(c) for c in self._contents])
        self._dirty = False

    def stats(self) -> Dict:
        return {
            "docs": len(self._contents),
            "tfidf_active": self._tfidf is not None,
            "features": getattr(self._tfidf, "max_features_", 0),
        }

    def prewarm(self, contents: List[str]) -> None:
        """预热语料：从底层后端加载既有内容（不写入底层）。

        用于生产规模验证——让语义层面对真实库的噪声水平，而不是只
        面对本次 probe 写入的少量条目。
        """
        self._contents = list(contents)
        self._ids = list(range(1, len(contents) + 1))
        self._dirty = True
        self.rebuild()


def _normalize(text: str) -> str:
    """轻量归一化：小写 + 空白折叠（保留中文）。"""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
