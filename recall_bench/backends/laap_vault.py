# -*- coding: utf-8 -*-
"""可选后端：LAAP MemoryVault 适配器（示例：如何接入真实 SQLite 记忆库）。

本模块带条件导入：未安装 LAAP 时跳过，不影响包本体使用。
"""

from __future__ import annotations

from typing import List, Optional

from recall_bench.backends.base import MemoryItem


class LaapVaultBackend:
    """适配 LAAP 的 ``laap.memory_vault.vault_manager.MemoryVault``。

    注意：LAAP 的 ``_get_connection()`` 硬编码模块级 DB_PATH，因此本适配器
    通过 monkeypatch ``vault_manager.DB_PATH`` 支持隔离模式（clean）。
    """

    name = "laap_vault"

    def __init__(self, db_path: Optional[str] = None) -> None:
        import importlib

        importlib.import_module("laap.memory_vault.vault_manager")
        import sys

        self._vmodule = sys.modules["laap.memory_vault.vault_manager"]
        if db_path is not None:
            # 隔离模式：重定向到临时库
            self._vmodule.DB_PATH = db_path
        from laap.memory_vault.vault_manager import MemoryVault

        self._vault = MemoryVault(db_path=db_path)
        self._session_id: str = ""

    def start_session(self, title: str = "") -> str:
        self._session_id = self._vault.start_session(title=title)
        return self._session_id

    def store(self, role: str, content: str, tags: str = "") -> int:
        return self._vault.store(
            role=role, content=content, tags=tags, session_id=self._session_id
        )

    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        rows = self._vault.search(query, limit=limit)
        return [
            MemoryItem(
                id=r.id,
                content=r.content,
                session_id=r.session_id,
                tags=r.tags,
                timestamp=str(r.timestamp),
            )
            for r in rows
        ]

    def cleanup(self, ids: List[int], session_id: str = "") -> None:
        if not ids:
            return
        conn = self._vmodule._get_connection()
        try:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"DELETE FROM conversations WHERE id IN ({placeholders})", ids
            )
            if session_id:
                conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?", (session_id,)
                )
            conn.commit()
        finally:
            conn.close()
