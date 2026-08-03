# -*- coding: utf-8 -*-
"""backends 包：记忆后端适配器 + 关系/自我模型协议。"""

from recall_bench.backends.base import MemoryBackend, MemoryItem, ensure_backend
from recall_bench.backends.memory import InMemoryBackend
from recall_bench.backends.hybrid import HybridBackend

__all__ = [
    "MemoryBackend",
    "MemoryItem",
    "ensure_backend",
    "InMemoryBackend",
    "HybridBackend",
]
