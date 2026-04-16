"""web_search_tool 与路由（不依赖外网时可跳过集成测试）。"""

from app.graph.routing import classify_intent
from app.tools.web_search_tool import _clean_query_for_search


def test_classify_web_url():
    assert classify_intent("请看 https://example.com/a.pdf 说明") == "web"


def test_classify_web_search_keyword():
    assert classify_intent("帮我网上查一下深圳公积金政策") == "web"


def test_clean_query_strips_url():
    q = _clean_query_for_search("请查 https://a.com/b 深圳 政策")
    assert "http" not in q.lower()
    assert "深圳" in q
