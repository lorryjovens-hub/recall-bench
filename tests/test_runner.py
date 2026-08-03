# -*- coding: utf-8 -*-
"""recall-bench 核心测试（零依赖：只用内置 InMemoryBackend）。"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from recall_bench.backends.memory import InMemoryBackend  # noqa: E402
from recall_bench.probe_set import (  # noqa: E402
    build_probe_set,
    default_tag,
    load_probe_set,
)
from recall_bench.runner import run_benchmark  # noqa: E402


def test_probe_set_structure():
    """抽查集：20 条、10 域、token 唯一且不出现在 query 中。"""
    tag = "RB_TEST"
    probe = build_probe_set(tag)
    assert len(probe) == 20
    domains = {p["domain"] for p in probe}
    assert len(domains) == 10
    tokens = [p["token"] for p in probe]
    assert len(set(tokens)) == 20
    for p in probe:
        assert p["token"] in p["content"]
        assert p["token"] not in p["query"]


def test_run_benchmark_memory_backend():
    """内存后端可跑通完整基准，报告结构完整。"""
    backend = InMemoryBackend()
    report = run_benchmark(backend, tag="RB_TEST", live=False)
    d = report.to_dict()
    assert d["total"] == 20
    assert 0 <= d["recall_rate"] <= 1.0
    assert d["backend"] == "memory"
    assert len(d["by_domain"]) == 10
    assert len(d["per_item"]) == 20
    assert "short<=10" in d["by_query_len"]
    # 命中数 = 各域命中之和
    domain_hits = sum(v["hit"] for v in d["by_domain"].values())
    assert domain_hits == d["hits"]


def test_run_benchmark_cleanup():
    """live 模式跑完后，测试记录应被清理（内存后端直接删）。"""
    backend = InMemoryBackend()
    report = run_benchmark(backend, tag="RB_TEST", live=True)
    remaining = [i for i in backend._items.values() if "RB_TEST" in i.content]
    assert remaining == []


def test_custom_probe_file():
    """自定义抽查集 JSON 可加载。"""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump([
            {"content": "用户喜欢早起", "domain": "习惯", "query": "起床时间"},
            {"content": "项目采用微服务架构", "domain": "项目", "query": "架构风格"},
        ], f, ensure_ascii=False)
        path = f.name
    try:
        probe = load_probe_set(path, "RB_CUSTOM")
        assert len(probe) == 2
        assert all("RB_CUSTOM" in p["token"] for p in probe)
        assert probe[0]["domain"] == "习惯"
        # 完整条目可直接透传
        probe2 = load_probe_set(path, "RB_CUSTOM")
        assert probe2[0]["token"].startswith("RB_CUSTOM")
    finally:
        os.unlink(path)


def test_cli_run_memory_backend():
    """CLI: run --backend memory 应输出报告。"""
    from recall_bench.cli import main

    out = os.path.join(tempfile.mkdtemp(), "report.json")
    rc = main(["run", "--backend", "memory", "--output", out])
    assert rc == 0
    with open(out, "r", encoding="utf-8") as f:
        d = json.load(f)
    assert d["total"] == 20
    assert d["mode"] == "clean"


def test_default_tag_format():
    assert default_tag().startswith("RB_20")


def test_hybrid_backend_improves_recall():
    """混合检索应显著优于纯 LIKE（同协议、同数据、仅换策略）。"""
    from recall_bench.backends.hybrid import HybridBackend

    like = InMemoryBackend()
    r_like = run_benchmark(like, tag="RB_AB")

    hybrid = HybridBackend(InMemoryBackend())
    r_hybrid = run_benchmark(hybrid, tag="RB_AB")

    assert r_hybrid.recall_rate >= r_like.recall_rate
    assert r_hybrid.recall_rate >= 0.5, f"hybrid 应显著提升 recall: {r_hybrid.recall_rate}"
    assert r_hybrid.backend.startswith("hybrid")


def test_hybrid_cleanup_keeps_corpus_consistent():
    """live 清理后语料应与底层一致（无悬空 id）。"""
    from recall_bench.backends.hybrid import HybridBackend

    base = InMemoryBackend()
    hybrid = HybridBackend(base)
    run_benchmark(hybrid, tag="RB_CLEAN", live=True)
    assert len(hybrid._contents) == 0
    assert len(hybrid._ids) == 0
    # 清理后重新检索不应崩溃
    assert hybrid.search("量子记忆压缩") == []
