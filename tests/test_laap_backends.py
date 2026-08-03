# -*- coding: utf-8 -*-
"""LAAP 生态适配器测试：记忆三件套 + 关系模型 + 自我模型。

仅当 LAAP 环境可用时运行（系统 python 已具备 numpy/sklearn，LAAP 可导入）。
"""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_LAAP_OK = True
try:
    sys.path.insert(0, r"D:\LAAP")
    import laap.memory.hierarchical  # noqa: F401
    import laap.engine.memory.vector_store  # noqa: F401
except Exception:  # pragma: no cover
    _LAAP_OK = False

pytestmark = pytest.mark.skipif(not _LAAP_OK, reason="LAAP 环境不可用")


def test_vector_backend_recall():
    """向量后端（哈希嵌入）应跑通基准协议。"""
    from recall_bench.backends.vector_backend import VectorBackend
    from recall_bench.runner import run_benchmark

    report = run_benchmark(VectorBackend(), tag="RB_VEC")
    assert report.total == 20
    assert 0.0 <= report.recall_rate <= 1.0
    assert report.backend == "laap_vector"


def test_hierarchy_backend_protocol():
    """分层记忆后端应跑通协议（标签映射为有损）。"""
    from recall_bench.backends.hierarchy_backend import HierarchicalBackend
    from recall_bench.runner import run_benchmark

    report = run_benchmark(HierarchicalBackend(), tag="RB_HIER")
    assert report.total == 20
    assert report.backend == "laap_hierarchy"


def test_qlam_backend_protocol():
    """量子记忆后端应跑通协议（含降级路径）。"""
    from recall_bench.backends.quantum_backend import QLAMBackend
    from recall_bench.runner import run_benchmark

    report = run_benchmark(QLAMBackend(), tag="RB_QLAM")
    assert report.total == 20
    assert report.backend == "laap_qlam"


def test_relation_backend_triples():
    """关系模型：三元组写入与查询回路。"""
    from recall_bench.backends.relation_model import (
        KnowledgeGraphBackend,
        RelationTriple,
    )

    kg = KnowledgeGraphBackend()  # 内存 SQLite
    rid = kg.add_triple(RelationTriple("aris", "belongs_to", "laap"))
    assert rid
    stats = kg.stats()
    assert stats.get("entities", 0) >= 2 or stats.get("triples", 0) >= 1
    # 实体查询不崩溃
    kg.query_entity("aris", hops=1)


def test_self_model_backend_observe_reflect():
    """自我模型：观测→反射回路。"""
    from recall_bench.backends.self_model import SelfVectorPoolBackend

    sm = SelfVectorPoolBackend()
    sm.observe({"needs": {"certainty": 0.7, "competence": 0.5}})
    snap = sm.reflect()
    assert "needs" in snap
    assert snap["needs"].get("certainty") == pytest.approx(0.7, abs=1e-4)
    # 向量接口可用
    v = sm.vector()
    assert v is not None
