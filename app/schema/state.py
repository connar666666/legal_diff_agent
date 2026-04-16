"""LangGraph 状态定义。"""

from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """工作流状态：消息流 + 可选路由与中间结果。"""

    messages: Annotated[list[AnyMessage], add_messages]
    intent: Optional[Literal["law", "case", "compare", "export", "general"]]
    last_tool_results: list[dict[str, Any]]
    scratch: dict[str, Any]


def default_state() -> GraphState:
    return {
        "messages": [],
        "intent": None,
        "last_tool_results": [],
        "scratch": {},
    }
