# -*- coding: utf-8 -*-
"""自我模型抽象：SelfModelBackend 协议 + LAAP SelfVectorPool 适配器。

背景
----
"统一与激活"计划的第三步：自我模型（个体身份、需求状态、自由能、语义
投影）也应拥有统一协议。这是个体型 RSI 的核心差异层——外部没有等价物。

协议方法
--------
- observe(state)  : 观测并合并一次状态更新（需求/情绪/自由能）
- vector()        : 当前自我向量（身份锚点）
- reflect()       : 自我模型快照（needs/fep/meta，可审计）
- save()          : 持久化
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class SelfModelBackend(Protocol):
    """自我模型协议。"""

    name: str

    def observe(self, state: Dict[str, Any]) -> None: ...

    def vector(self) -> Any: ...

    def reflect(self) -> Dict[str, Any]: ...

    def save(self) -> bool: ...


class SelfVectorPoolBackend:
    """包装 ``self_model_bridge.self_vector_pool.SelfVectorPool``。

    映射:
    - observe(needs/fep) -> set_needs / set_fep（合并状态更新）
    - vector()           -> semantic_projection（身份锚点向量）
    - reflect()          -> needs + fep + meta + cycle 快照
    """

    name = "laap_self"

    def __init__(self, pool=None, path: Optional[str] = None) -> None:
        if pool is not None:
            self._pool = pool
        else:
            import sys

            sys.path.insert(0, r"D:\LAAP")
            from self_model_bridge.self_vector_pool import SelfVectorPool

            self._pool = SelfVectorPool(path=path)
        self._last_observe: Dict[str, Any] = {}

    def observe(self, state: Dict[str, Any]) -> None:
        needs = state.get("needs")
        if needs:
            self._pool.set_needs(needs)
        fep = state.get("fep")
        if fep:
            self._pool.set_fep(
                vfe=fep.get("vfe", 0.0), efe=fep.get("efe", 0.0),
            )
        meta = state.get("meta")
        if meta:
            self._pool.set_meta(**meta)
        proj = state.get("semantic_projection")
        if proj is not None:
            self._pool.set_semantic_projection(proj)
        self._last_observe = state

    def vector(self) -> Any:
        return self._pool.get_semantic_projection()

    def reflect(self) -> Dict[str, Any]:
        return {
            "needs": self._pool.get_needs(),
            "fep": self._pool.get_fep(),
            "meta": self._pool.get_meta(),
            "fusion_weights": self._pool.get_fusion_weights(),
            "last_observe_keys": list(self._last_observe.keys()),
        }

    def save(self) -> bool:
        return bool(self._pool.save())
