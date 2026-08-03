# -*- coding: utf-8 -*-
"""关系模型抽象：RelationBackend 协议 + LAAP KnowledgeGraph 适配器。

背景
----
"统一与激活"计划的第二步：记忆层（MemoryBackend）之后，关系层也应拥有
统一协议——让任何关系存储（知识图谱、关联记忆、向量关系）可互换、可基准。

协议方法
--------
- add_triple / add_from_text : 写入（subject, predicate, object 三元组）
- query_entity(label, hops)  : 实体邻域查询
- neighbors(label)           : 一跳邻居
- find_path(a, b, max_hops)  : 关系路径
- stats()                    : 规模统计
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class RelationTriple:
    """关系三元组。"""

    subject: str
    predicate: str
    object: str
    importance: float = 0.5

    def to_tuple(self) -> tuple:
        return (self.subject, self.predicate, self.object)


@runtime_checkable
class RelationBackend(Protocol):
    """关系记忆协议。"""

    name: str

    def add_triple(self, triple: RelationTriple) -> str: ...

    def add_from_text(self, text: str, source_memory_id: str = "") -> List[str]: ...

    def query_entity(self, label: str, hops: int = 1) -> Dict[str, Any]: ...

    def neighbors(self, label: str) -> List[Dict[str, Any]]: ...

    def find_path(self, a: str, b: str, max_hops: int = 3) -> Optional[List[Dict[str, str]]]: ...

    def stats(self) -> Dict[str, int]: ...


class KnowledgeGraphBackend:
    """包装 ``laap.memory.knowledge_graph.KnowledgeGraph``（SQLite 三元组库）。"""

    name = "laap_kg"

    def __init__(self, knowledge_graph=None, db_path: Optional[str] = None) -> None:
        if knowledge_graph is not None:
            self._kg = knowledge_graph
        else:
            from pathlib import Path
            import tempfile

            from laap.memory.knowledge_graph import KnowledgeGraph

            if not db_path:
                # SQLite ":memory:" 每次连接都是新库，改用临时文件
                fd, tmp = tempfile.mkstemp(suffix="_kg.db")
                import os

                os.close(fd)
                db_path = tmp
            self._kg = KnowledgeGraph(db_path=Path(db_path))
        self._triples: List[RelationTriple] = []

    def add_triple(self, triple: RelationTriple) -> str:
        from laap.memory.knowledge_graph import Triple

        t = Triple(subject=triple.subject, relation=triple.predicate,
                   object=triple.object, importance=triple.importance)
        rid = self._kg.add_triple(t)
        self._triples.append(triple)
        return rid

    def add_from_text(self, text: str, source_memory_id: str = "") -> List[str]:
        from laap.memory.knowledge_graph import extract_triples

        ids = []
        for t in extract_triples(text):
            ids.append(self._kg.add_triple(t))
        return ids

    def query_entity(self, label: str, hops: int = 1) -> Dict[str, Any]:
        return self._kg.query_entity(label, hops=hops)

    def neighbors(self, label: str) -> List[Dict[str, Any]]:
        return self._kg.neighbors(label)

    def find_path(self, a: str, b: str, max_hops: int = 3) -> Optional[List[Dict[str, str]]]:
        return self._kg.find_path(a, b, max_hops=max_hops)

    def stats(self) -> Dict[str, int]:
        return self._kg.stats()
