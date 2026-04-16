"""图节点：预构建 ReAct 之外的辅助函数；自定义 StateGraph 时可复用 LLM / Tool 工厂。"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.graph.prompts import SYSTEM_PROMPT


def get_llm_runnable():
    """
    与 `build_agent_graph` 相同的聊天模型（Transformers tool-calling 适配器或 Ollama）。
    用于自行拼装 `StateGraph` 时的 LLM 节点。
    """
    from app.llm.model_factory import get_chat_model_for_agent

    return get_chat_model_for_agent()


def get_tool_node():
    """
    LangGraph 预置 `ToolNode`（即自定义图中的 tool 节点），绑定 `all_tools()`。
    与 `create_react_agent` 内使用的工具集合一致。
    """
    from langgraph.prebuilt import ToolNode

    from app.tools.export_tool import all_tools

    return ToolNode(all_tools())


def prepend_system(messages: list[BaseMessage]) -> list[BaseMessage]:
    """在消息列表前插入系统消息（自定义 StateGraph 时使用）。"""
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)


def last_user_text(messages: list[BaseMessage]) -> str:
    """取最近一条用户文本。"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


def summarize_tool_output_for_log(tool_name: str, content: str, max_len: int = 500) -> str:
    """日志用截断。"""
    c = content.replace("\n", " ")
    if len(c) > max_len:
        c = c[:max_len] + "…"
    return f"{tool_name}: {c}"
