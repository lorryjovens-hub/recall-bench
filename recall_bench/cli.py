# -*- coding: utf-8 -*-
"""命令行入口。

用法::

    # 用内置内存后端跑一次隔离基准（零依赖，立即可跑）
    python -m recall_bench.cli run

    # 用 LAAP MemoryVault 后端（需 LAAP 环境；--live 生产库模式自动清理）
    python -m recall_bench.cli run --backend laap_vault
    python -m recall_bench.cli run --backend laap_vault --live

    # 自定义抽查集 + 输出 JSON 报告
    python -m recall_bench.cli run --probe-file my_probe.json --output report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="recall-bench",
                                 description="跨会话记忆 recall 基准")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="执行一次基准")
    run.add_argument("--backend", default="memory",
                     choices=["memory", "laap_vault", "hybrid"],
                     help="记忆后端（默认 memory，零依赖）")
    run.add_argument("--live", action="store_true",
                     help="live 模式：不清空后端（生产库），跑后自动清理")
    run.add_argument("--probe-file", default="",
                     help="自定义抽查集 JSON（默认内置 20 条中性事实）")
    run.add_argument("--tag", default="",
                     help="唯一标记前缀（默认 RB_YYYYMMDD）")
    run.add_argument("--output", default="",
                     help="JSON 报告输出路径（默认打印到控制台）")
    run.add_argument("--db-path", default="",
                     help="laap_vault 后端隔离模式数据库路径（默认生产库）")

    sub.add_parser("version", help="显示版本")
    return ap


def _make_backend(name: str, db_path: str):
    if name == "memory":
        from recall_bench.backends.memory import InMemoryBackend

        return InMemoryBackend()
    if name == "laap_vault":
        from recall_bench.backends.laap_vault import LaapVaultBackend

        return LaapVaultBackend(db_path=db_path or None)
    if name == "hybrid":
        from recall_bench.backends.hybrid import HybridBackend
        from recall_bench.backends.memory import InMemoryBackend

        # hybrid 默认包装内存后端（零依赖可跑）；laap 组合用 --db-path 指定
        base = LaapVaultBackend(db_path=db_path or None) if db_path else InMemoryBackend()
        return HybridBackend(base)
    raise ValueError(f"未知后端: {name}")


def _cmd_run(args: argparse.Namespace) -> int:
    from recall_bench.probe_set import default_tag, load_probe_set
    from recall_bench.runner import run_benchmark, print_report

    tag = args.tag or default_tag()
    backend = _make_backend(args.backend, args.db_path)

    probe = None
    if args.probe_file:
        probe = load_probe_set(args.probe_file, tag)

    report = run_benchmark(
        backend=backend, tag=tag, live=args.live, probe_set=probe,
    )
    print_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"报告已写入: {args.output}")
    return 0


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "version":
        from recall_bench import __version__

        print(f"recall-bench {__version__}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
