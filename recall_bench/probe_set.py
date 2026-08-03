# -*- coding: utf-8 -*-
"""抽查集：跨会话 recall 的客观探测。

每条事实满足三个约束：
1. 含唯一标记（token），用于客观命中判定与 live 模式清理；
2. 检索 query 不含标记——模拟"用户记得内容大意，不记得标记"；
3. query 与 content 措辞不同——测语义召回，而非字符串复制。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

#: 默认抽查集：10 个主题域 × 2 条 = 20 条中性事实（不绑定任何具体项目）。
#: 格式: (content, domain, query)
_DEFAULT_FACTS: List[tuple] = [
    ("用户近期在调研量子记忆压缩算法的能耗表现", "技术",
     "量子记忆压缩的能耗研究进展如何"),
    ("系统采用四层分级架构来管理长期记忆", "技术",
     "长期记忆的分层设计是怎样的"),
    ("团队习惯在深夜进行架构设计讨论", "习惯",
     "架构设计讨论一般在什么时间进行"),
    ("产品计划在下季度发布公开评测报告", "目标",
     "评测报告预计什么时候发布"),
    ("桌面端应用最近迁移到了新的会话框架", "项目",
     "桌面端迁移到了什么框架"),
    ("实验室采购了一批新的 GPU 用于推理加速", "资源",
     "新采购的 GPU 主要用在哪里"),
    ("用户偏好深色主题的界面配色方案", "偏好",
     "界面配色上用户喜欢哪种风格"),
    ("上周完成了错误反思模块的第一轮迭代", "事件",
     "错误反思模块迭代到第几轮了"),
    ("长上下文场景下的 token 消耗明显下降", "效率",
     "长上下文场景的 token 效率"),
    ("每日凌晨四时执行记忆巩固任务", "习惯",
     "记忆巩固任务在什么时候运行"),
    ("用户认为评估器应当独立于改进循环之外", "观点",
     "评估器应该放在循环里面还是外面"),
    ("编排层支持并行子任务的回收与合并", "项目",
     "编排层能不能并行回收子任务"),
    ("用户关注递归自我改进的安全边界问题", "兴趣",
     "递归自我改进存在哪些安全边界"),
    ("会话记录已经同步到了移动端", "事件",
     "会话记录同步到了哪些端"),
    ("检索使用向量索引与关键词混合策略", "技术",
     "记忆检索用的是什么策略"),
    ("新版本在代码生成基准上提升三成", "目标",
     "新版本的代码生成能力提升多少"),
    ("用户最近在调试跨平台的数据同步", "项目",
     "跨平台数据同步调试得如何"),
    ("原型沙箱启用了只读文件系统保护", "技术",
     "原型沙箱的文件保护方式"),
    ("会议记录存放在项目仓库的文档目录", "习惯",
     "会议记录一般存放在哪里"),
    ("用户正在准备认知架构的对照笔记", "目标",
     "认知架构对照笔记进展如何"),
]


def build_probe_set(tag: str,
                    facts: List[tuple] | None = None) -> List[Dict[str, Any]]:
    """构造抽查集。

    Args:
        tag: 唯一标记前缀，形如 ``RB_20260803``；追加到每条事实，用于
            命中判定与 live 模式清理。
        facts: 可选自定义事实列表 [(content, domain, query), ...]。

    Returns:
        probe 条目列表: {token, content, domain, query, tags}
    """
    facts = facts or _DEFAULT_FACTS
    probe = []
    for i, (content, domain, query) in enumerate(facts):
        token = f"{tag}_{i:02d}"
        probe.append({
            "token": token,
            "content": f"{content}（标记:{token}）",
            "domain": domain,
            "query": query,
            "tags": f"recall_bench,{domain}",
        })
    return probe


def load_probe_set(path: str, tag: str) -> List[Dict[str, Any]]:
    """从 JSON 文件加载自定义抽查集。

    JSON 结构::

        [{"content": "...", "domain": "技术", "query": "..."}, ...]

    也可直接传入由 ``build_probe_set`` 生成的完整条目（含 token）。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"probe 文件格式错误: {path}")

    if "token" in data[0]:
        return data  # 已是完整条目

    probe = []
    for i, item in enumerate(data):
        token = f"{tag}_{i:02d}"
        probe.append({
            "token": token,
            "content": f"{item['content']}（标记:{token}）",
            "domain": item.get("domain", "general"),
            "query": item["query"],
            "tags": f"recall_bench,{item.get('domain', 'general')}",
        })
    return probe


def default_tag() -> str:
    """默认标记：RB_YYYYMMDD。"""
    return f"RB_{time.strftime('%Y%m%d')}"
