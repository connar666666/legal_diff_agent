"""按名称调用 LangChain BaseTool（与 `create_react_agent` / ToolNode 使用同一套工具定义）。"""

from __future__ import annotations

import json
from typing import Any, Optional

from app.tools import registry as tool_registry


class ToolExecutor:
    """轻量执行器：便于单测、脚本或自定义节点复用，不替代 LangGraph ToolNode。"""

    def __init__(self, tools: Optional[list[Any]] = None) -> None:
        if tools is None:
            from app.tools.export_tool import all_tools

            tools = all_tools()
        self._tools = {getattr(t, "name", ""): t for t in tools if getattr(t, "name", None)}

    def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            tool = tool_registry.get_tool(name)
        if tool is None:
            return json.dumps({"ok": False, "error": f"unknown tool: {name}"}, ensure_ascii=False)
        return tool.invoke(arguments)
