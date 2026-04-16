"""ToolExecutor 单测。"""

import json

from app.services.law_service import LawService
from app.services.tool_executor import ToolExecutor
from app.tools import registry
from app.tools.search_law_tool import search_law_tool


def test_executor_unknown_tool():
    ex = ToolExecutor([search_law_tool])
    out = json.loads(ex.invoke("missing_tool_xyz", {}))
    assert out.get("ok") is False


def test_executor_search_law():
    registry.set_services(LawService(), None)
    ex = ToolExecutor([search_law_tool])
    out = ex.invoke("search_law_tool", {"query": "测", "jurisdiction": ""})
    data = json.loads(out)
    assert "ok" in data
