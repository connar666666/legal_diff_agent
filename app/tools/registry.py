"""运行时服务注册，供各 tool 获取 LawService / CaseService；并缓存工具名 -> BaseTool。"""

from __future__ import annotations

from typing import Any, Optional

from app.services.case_service import CaseService
from app.services.law_service import LawService

_law: Optional[LawService] = None
_case: Optional[CaseService] = None
_tools_by_name: dict[str, Any] = {}


def set_services(law: Optional[LawService], case: Optional[CaseService]) -> None:
    global _law, _case
    _law = law
    _case = case


def get_law_service() -> Optional[LawService]:
    return _law


def get_case_service() -> Optional[CaseService]:
    return _case


def register_tools(tools: list[Any]) -> None:
    """将 LangChain 工具列表登记到内存映射（通常在构图或启动时调用一次）。"""
    global _tools_by_name
    for t in tools:
        n = getattr(t, "name", None)
        if n:
            _tools_by_name[str(n)] = t


def get_tool(name: str) -> Optional[Any]:
    """按名称取工具；若尚未 register_tools，则懒加载 all_tools()。"""
    if name in _tools_by_name:
        return _tools_by_name[name]
    from app.tools.export_tool import all_tools

    register_tools(all_tools())
    return _tools_by_name.get(name)
