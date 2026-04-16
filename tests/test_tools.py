"""工具层测试（无索引）。"""

import json

from app.services.law_service import LawService
from app.tools import registry
from app.tools.search_law_tool import search_law_tool


def test_search_law_without_index():
    registry.set_services(LawService(), None)
    out = search_law_tool.invoke({"query": "测试", "jurisdiction": ""})
    data = json.loads(out)
    assert data.get("ok") is False
