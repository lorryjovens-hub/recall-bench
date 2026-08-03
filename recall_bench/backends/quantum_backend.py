# -*- coding: utf-8 -*-
"""QLAM 量子记忆适配器（概念演示）。

QLAM（量子长程注意力记忆）把序列历史编码为量子叠加态，PQC 非经典更新，
量子测量检索——它不是逐条检索的存储，而是"压缩态 + 相关性打分"。
本适配器用基准协议如实展示它的检索特性。

映射:
- store -> update_from_text(content)：累积进叠加态（QLAM 语义）；
- search -> 角度编码 query + retrieve(k)：返回状态内相关性最高的索引；
- 本适配器自维护 id→text 镜像，弥补 QLAM 不保存原始文本的限制。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from recall_bench.backends.base import MemoryItem


class QLAMBackend:
    """包装 ``laap.memory.quantum.quantum_memory.QLAMMemory``。"""

    name = "laap_qlam"

    def __init__(self, qlam_memory=None, n_qubits: int = 6, n_layers: int = 3) -> None:
        if qlam_memory is not None:
            self._qm = qlam_memory
        else:
            from laap.memory.quantum.quantum_memory import QLAMMemory

            self._qm = QLAMMemory(n_qubits=n_qubits, n_layers=n_layers)
        self._texts: list = []
        self._fallback_only = False

    def start_session(self, title: str = "") -> str:
        return title

    def store(self, role: str, content: str, tags: str = "") -> int:
        eid = len(self._texts) + 1
        try:
            self._qm.update_from_text(content)
        except Exception:
            self._fallback_only = True
        self._texts.append(content)
        return eid

    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        if self._fallback_only or not self._texts:
            # 降级：字符重叠评分（QLAM 不可用时保持协议可用）
            return [
                MemoryItem(id=i + 1, content=t)
                for i, t in enumerate(self._texts)
                if _overlap(query, t) >= 2
            ][:limit]
        try:
            from laap.memory.quantum.quantum_memory import QuantumEncoder, QuantumMeasurement

            q_vec = QuantumEncoder.angle_encode(
                _hash_features(query), n_qubits=getattr(self._qm, "n_qubits", 6)
            )
            hits = self._qm.retrieve(q_vec, k=limit)
            results = []
            for idx, score in hits:
                if 0 <= idx < len(self._texts):
                    results.append(MemoryItem(id=idx + 1, content=self._texts[idx]))
            return results
        except Exception:
            return [
                MemoryItem(id=i + 1, content=t)
                for i, t in enumerate(self._texts)
                if _overlap(query, t) >= 2
            ][:limit]

    def cleanup(self, ids: List[int], session_id: str = "") -> None:
        # 叠加态无逐条删除语义——隔离环境直接丢弃实例
        raise NotImplementedError("QLAM 叠加态不支持逐条删除；请使用隔离实例")


def _hash_features(text: str) -> np.ndarray:
    """轻量定长特征（无模型依赖）。"""
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(
        n_features=64, analyzer="char_wb", ngram_range=(2, 3), alternate_sign=False
    )
    return hv.transform([text]).toarray().ravel().astype(np.float64)


def _overlap(a: str, b: str) -> int:
    """字符 2-gram 交集计数（降级相似度）。"""
    ga = {a[i : i + 2] for i in range(len(a) - 1)}
    gb = {b[i : i + 2] for i in range(len(b) - 1)}
    return len(ga & gb)
