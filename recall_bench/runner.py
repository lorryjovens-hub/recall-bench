# -*- coding: utf-8 -*-
"""基准运行器：写入 → 检索 → 统计 → 报告。

核心逻辑与记忆后端实现无关，只依赖 MemoryBackend 协议。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from recall_bench.backends.base import MemoryBackend, ensure_backend
from recall_bench.probe_set import build_probe_set


@dataclass
class BenchmarkReport:
    """一次基准运行的完整报告（可 JSON 序列化）。"""

    baseline: str = "recall_v0.1"
    tag: str = ""
    mode: str = "clean"  # clean=隔离库 / live=生产库（自动清理）
    timestamp: str = ""
    total: int = 0
    hits: int = 0
    recall_rate: float = 0.0
    avg_latency_ms: float = 0.0
    write_ms_total: float = 0.0
    by_domain: Dict[str, Any] = field(default_factory=dict)
    by_query_len: Dict[str, Any] = field(default_factory=dict)
    per_item: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_benchmark(
    backend: MemoryBackend,
    tag: str,
    live: bool = False,
    probe_set: Optional[List[Dict[str, Any]]] = None,
    cleanup: bool = True,
) -> BenchmarkReport:
    """执行一次 recall 基准。

    Args:
        backend: 实现 MemoryBackend 协议的记忆后端。
        tag: 唯一标记前缀（每条事实追加 ``{tag}_{i:02d}``）。
        live: True=不清空后端（生产库模式，跑后自动清理写入的记录）；
              False=假定后端为隔离环境。
        probe_set: 自定义抽查集；None 则用默认 20 条。
        cleanup: 是否在结束后清理写入的记录（live 模式建议 True）。

    Returns:
        BenchmarkReport
    """
    backend = ensure_backend(backend)
    probe = probe_set or build_probe_set(tag)
    cleanup_ids: List[int] = []

    session_id = backend.start_session(title=f"recall_bench_{tag}")

    # ── 写入 ────────────────────────────────────────────────────────────
    t0 = time.time()
    for item in probe:
        eid = backend.store(
            role="aris", content=item["content"], tags=item["tags"]
        )
        item["entry_id"] = eid
        cleanup_ids.append(eid)
    write_ms = (time.time() - t0) * 1000

    # ── 检索 ────────────────────────────────────────────────────────────
    results_per_item: List[Dict[str, Any]] = []
    hits = 0
    latencies: List[float] = []
    for item in probe:
        t0 = time.time()
        rows = backend.search(item["query"], limit=10)
        lat_ms = (time.time() - t0) * 1000
        latencies.append(lat_ms)
        hit = any(item["token"] in r.content for r in rows)
        if hit:
            hits += 1
        results_per_item.append({
            "domain": item["domain"],
            "query": item["query"],
            "query_len": len(item["query"]),
            "hit": hit,
            "latency_ms": round(lat_ms, 1),
            "top_result": (rows[0].content[:60] if rows else ""),
        })

    # ── 清理（live 模式移除测试写入，保持生产库干净）───────────────────
    if live and cleanup:
        try:
            backend.cleanup(cleanup_ids, session_id=session_id)
        except NotImplementedError:  # pragma: no cover
            print("[warn] 后端未实现 cleanup，测试记录可能残留")

    # ── 汇总 ────────────────────────────────────────────────────────────
    n = len(probe)
    by_domain: Dict[str, Dict[str, Any]] = {}
    for r in results_per_item:
        d = by_domain.setdefault(r["domain"], {"total": 0, "hit": 0, "lat": []})
        d["total"] += 1
        d["hit"] += 1 if r["hit"] else 0
        d["lat"].append(r["latency_ms"])
    domain_stats = {
        d: {
            "recall_rate": round(v["hit"] / v["total"], 3),
            "total": v["total"],
            "hit": v["hit"],
            "avg_latency_ms": round(sum(v["lat"]) / len(v["lat"]), 1),
        }
        for d, v in sorted(by_domain.items())
    }

    def _rate(items: List[Dict[str, Any]]) -> Optional[float]:
        return round(sum(1 for r in items if r["hit"]) / len(items), 3) if items else None

    short_q = [r for r in results_per_item if r["query_len"] <= 10]
    long_q = [r for r in results_per_item if r["query_len"] > 10]

    report = BenchmarkReport(
        tag=tag,
        mode="live" if live else "clean",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        total=n,
        hits=hits,
        recall_rate=round(hits / n, 3),
        avg_latency_ms=round(sum(latencies) / len(latencies), 1),
        write_ms_total=round(write_ms, 1),
        by_domain=domain_stats,
        by_query_len={
            "short<=10": {"total": len(short_q), "recall_rate": _rate(short_q)},
            "long>10": {"total": len(long_q), "recall_rate": _rate(long_q)},
        },
        per_item=results_per_item,
        backend=backend.name,
    )
    return report


def print_report(report: BenchmarkReport) -> None:
    """控制台人类可读输出。"""
    print("\n" + "=" * 52)
    print(f"RECALL BENCHMARK  backend={report.backend}  mode={report.mode}  "
          f"{report.timestamp}")
    print(f"  total {report.total} | hits {report.hits} | "
          f"recall_rate = {report.recall_rate:.1%}")
    print(f"  avg_latency {report.avg_latency_ms} ms | "
          f"write {report.write_ms_total} ms")
    print("-" * 52)
    for domain, s in report.by_domain.items():
        bar = "#" * int(s["recall_rate"] * 20)
        print(f"  {domain:<6} {s['recall_rate']:6.1%}  {bar:<20} "
              f"{s['hit']}/{s['total']}")
    print("=" * 52)
