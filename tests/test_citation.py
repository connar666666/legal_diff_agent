"""引用格式化。"""

from app.utils.citation import format_law_citation


def test_format_law_citation_full():
    s = format_law_citation("上海市物业管理条例", "第十二条", "（一）")
    assert "上海市物业管理条例" in s
    assert "第十二条" in s
    assert "（一）" in s


def test_format_law_citation_article_only():
    assert format_law_citation("", "第五条", "") == "第五条"


def test_format_law_citation_empty():
    assert format_law_citation("", "", "") == ""
