# -*- coding: utf-8 -*-
"""后端抽象：任何记忆系统只需实现这个协议即可被基准测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable


@dataclass
class MemoryItem:
    """基准测试关心的最小记忆条目。"""

    id: int
    content: str
    session_id: str = ""
    tags: str = ""
    timestamp: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"MemoryItem(id={self.id}, content={self.content[:40]!r})"


@runtime_checkable
class MemoryBackend(Protocol):
    """记忆后端协议。

    实现方最少需要 ``store`` 与 ``search``；``cleanup`` 用于 live 模式下
    移除测试写入的记录（可选，未实现则跳过清理并给出提示）。
    """

    #: 后端名称（报告中的 mode 标识）
    name: str

    def store(self, role: str, content: str, tags: str = "") -> int:
        """写入一条记忆，返回条目 id。"""
        ...

    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """按查询检索记忆，返回条目列表（含 content 字段）。"""
        ...

    def cleanup(self, ids: List[int], session_id: str = "") -> None:
        """删除测试写入的记录（live 模式自动清理用）。"""
        ...

    def start_session(self, title: str = "") -> str:  # pragma: no cover
        """开启一个会话（可选；默认返回空串表示无会话概念）。"""
        return ""


def ensure_backend(backend: Any) -> MemoryBackend:
    """运行时校验对象满足 MemoryBackend 协议，便于给出友好报错。"""
    if not isinstance(backend, MemoryBackend):
        raise TypeError(
            "backend 必须实现 MemoryBackend 协议（store/search/cleanup/name）。"
            f" 收到: {type(backend)!r}"
        )
    return backend
