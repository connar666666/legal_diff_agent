"""结构化输出解析测试。"""

from app.utils.output_parser import parse_tool_calls_from_text


def test_parse_tool_calls_from_text_basic():
    text = (
        '<tool_call>{"name":"search_law_tool","arguments":'
        '{"query":"劳动合同","jurisdiction":""},"id":"c1","type":"tool_call"}</tool_call>'
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "search_law_tool"
    assert calls[0]["args"]["query"] == "劳动合同"


def test_parse_empty():
    assert parse_tool_calls_from_text("") == []
    assert parse_tool_calls_from_text("无标签") == []
