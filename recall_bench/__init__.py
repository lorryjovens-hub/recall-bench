# -*- coding: utf-8 -*-
"""recall-bench: 跨会话记忆 Recall 基准（开源方法论包）。

设计原则
--------
- 与任何记忆系统实现解耦：通过 ``MemoryBackend`` 协议运行，可以测任何
  提供 store/search 的记忆后端（SQLite vault、向量库、托管 API……）。
- 零第三方依赖，Python >= 3.10。
- 带唯一标记的写入-检索协议：标记不出现在检索 query 中，命中判定客观。

起源
----
本包源自 LAAP（Living Agent Application Protocol）项目的架构实践——
跨会话记忆是"部署后持续学习"（Ilya Sutskever 2025）与"记忆生命周期成为
智能本身的一部分"（Lilian Weng 2026）的工程验证。第一版基线于 2026-08-02
在 LAAP 生产记忆库上跑出 recall 25.0%（LIKE 子串匹配），本包将该方法论
通用化，供任何记忆系统复测与对比。
"""

from recall_bench.backends.base import MemoryBackend, MemoryItem
from recall_bench.probe_set import build_probe_set, load_probe_set, default_tag
from recall_bench.runner import run_benchmark, BenchmarkReport

__version__ = "0.1.0"

__all__ = [
    "MemoryBackend",
    "MemoryItem",
    "build_probe_set",
    "load_probe_set",
    "default_tag",
    "run_benchmark",
    "BenchmarkReport",
    "__version__",
]
