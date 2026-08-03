# -*- coding: utf-8 -*-
"""HierarchicalMemory 适配器（LAAP L1-L4 分层记忆）。

映射说明（有损，基准会如实测出）:
- store -> remember(content, tags)：写入分层记忆；
- search -> recall(query_tags)：分层记忆是**标签检索**，将自然语言 query
  拆为字符 n-gram 作为标签——这是有损映射，正好用基准量化其 recall 水平；
- cleanup：HierarchicalMemory 无按 id 删除接口，跳过并提示（隔离环境用
  临时实例，直接丢弃即可）。

可选导入：未安装 LAAP 时跳过。
"""

from __future__ import annotations

import re
from typing import List, Optional

from recall_bench.backends.base import MemoryItem


class HierarchicalBackend:
    """包装 ``laap.memory.hierarchical.HierarchicalMemory``。"""

    name = "laap_hierarchy"

    def __init__(self, hierarchical_memory=None, use_rust: bool = False) -> None:
        if hierarchical_memory is not None:
            self._hm = hierarchical_memory
        else:
            from laap.memory.hierarchical import HierarchicalMemory

            self._hm = HierarchicalMemory(use_rust=use_rust)
        self._items: dict = {}  # id -> MemoryItem 镜像（弥补无按 id 查询）

    def start_session(self, title: str = "") -> str:
        return title

    def store(self, role: str, content: str, tags: str = "") -> int:
        eid = len(self._items) + 1
        self._hm.remember(content, tags=[t for t in tags.split(",") if t])
        self._items[eid] = MemoryItem(id=eid, content=content, tags=tags)
        return eid

    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        # 字符 n-gram 标签映射
        tags = _ngram_tags(query)
        rows = self._hm.recall(query_tags=tags, limit=limit)
        results = []
        seen = set()
        for r in rows:
            c = getattr(r, "content", str(r))
            if c in seen:
                continue
            seen.add(c)
            # 尽量对齐本包写入的镜像
            item = next((v for v in self._items.values() if v.content == c), None)
            results.append(item or MemoryItem(id=len(results) + 1, content=c))
        return results

    def cleanup(self, ids: List[int], session_id: str = "") -> None:
        # HierarchicalMemory 无按 id 删除接口——隔离环境直接丢弃实例
        raise NotImplementedError(
            "HierarchicalMemory 不支持按 id 删除；请使用隔离实例"
        )


def _ngram_tags(text: str, n_range=(2, 3)) -> List[str]:
    """字符 n-gram 标签化（中文无分词依赖）。"""
    text = re.sub(r"\s+", "", text)
    tags = set()
    for n in n_range:
        tags.update(text[i : i + n] for i in range(len(text) - n + 1))
    # 控制标签数量，避免噪声
    return sorted(tags)[:40]
