"""编译 LangGraph 图：统一经 `get_chat_model_for_agent` + `create_react_agent`。"""

from __future__ import annotations

from typing import Any, Optional

from app.graph.prompts import SYSTEM_PROMPT
from app.llm.model_factory import get_chat_model_for_agent
from app.tools import registry as tool_registry
from app.tools.export_tool import all_tools


def build_agent_graph(debug: bool = False):
    """
    构建并编译可执行的图。
    """
    from langgraph.prebuilt import create_react_agent

    tools = all_tools()
    tool_registry.register_tools(tools)
    model = get_chat_model_for_agent()
    graph = create_react_agent(
        model,
        tools,
        prompt=SYSTEM_PROMPT,
        debug=debug,
    )
    return graph


def invoke_chat(
    graph,
    user_text: str,
    thread_id: str = "default",
    prior_messages: Optional[list[Any]] = None,
    *,
    debug_tools: bool = False,
) -> dict[str, Any]:
    """单次对话封装（同步）。"""
    from langchain_core.messages import HumanMessage
    from langchain_core.callbacks import BaseCallbackHandler

    config = {"configurable": {"thread_id": thread_id}}
    prior_messages = prior_messages or []

    if debug_tools and settings.tool_debug_enabled:
        from app.utils.debug_logger import JSONLWriter, ToolCallDebugCallbackHandler

        writer = JSONLWriter(settings.tool_debug_log_path)
        callbacks: list[BaseCallbackHandler] = [
            ToolCallDebugCallbackHandler(writer, run_scope="graph_run")
        ]
        # RunnableConfig supports "callbacks" in config dict
        config["callbacks"] = callbacks

    # recursion_limit 防止工具-模型无限循环（默认25，这里限制为10轮）
    config["recursion_limit"] = 10

    return graph.invoke(
        {"messages": [*prior_messages, HumanMessage(content=user_text)]},
        config=config,
    )
