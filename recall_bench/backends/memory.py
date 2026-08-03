# -*- coding: utf-8 -*-
"""内置后端：纯内存参考实现（零依赖，可用作基线对照与测试）。"""

from __future__ import annotations

import re
import time
from typing import Dict, List

from recall_bench.backends.base import MemoryItem


class InMemoryBackend:
    """纯内存记忆后端。

    检索策略：子串匹配（模拟传统 LIKE 语义）。正因为 query 与内容措辞
    不同时子串匹配会漏召，本后端可以复现"低 recall"现象——用于展示
    基准方法论的有效性，也作为任何真实后端的对照基线。
    """

    name = "memory"

    def __init__(self) -> None:
        self._items: Dict[int, MemoryItem] = {}
        self._next_id: int = 1
        self._session: str = ""

    def start_session(self, title: str = "") -> str:
        self._session = f"session_{int(time.time())}"
        return self._session

    def store(self, role: str, content: str, tags: str = "") -> int:
        item = MemoryItem(
            id=self._next_id,
            content=content,
            session_id=self._session,
            tags=tags,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._items[self._next_id] = item
        self._next_id += 1
        return item.id

    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        results: List[MemoryItem] = []
        seen: set = set()

        # 策略 1: 全 query 子串匹配
        for item in self._items.values():
            if item.id in seen:
                continue
            if query in item.content:
                seen.add(item.id)
                results.append(item)
            if len(results) >= limit:
                return results

        # 策略 2: 关键词拆分匹配
        keywords = [k for k in re.split(r"[，,。.\s]+", query) if len(k) >= 2]
        for kw in keywords:
            if len(results) >= limit:
                break
            for item in self._items.values():
                if item.id in seen:
                    continue
                if kw in item.content:
                    seen.add(item.id)
                    results.append(item)
                if len(results) >= limit:
                    break
        return results

    def cleanup(self, ids: List[int], session_id: str = "") -> None:
        for i in ids:
            self._items.pop(i, None)
