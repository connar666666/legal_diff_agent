"""为 LangGraph / LangChain 构造 Chat 模型（统一入口）。"""

from __future__ import annotations

from typing import Any, Optional

from app.config import settings


def get_chat_model(model: Optional[str] = None, temperature: float = 0.2):
    """
    Ollama 后端：返回已配置好的 ChatOllama（支持工具调用与流式，取决于 Ollama 版本）。
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "需要安装 langchain-ollama：pip install langchain-ollama"
        ) from e

    return ChatOllama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


def get_chat_model_for_agent() -> Any:
    """
    与 `build_agent_graph` 一致：按 `settings.llm_backend` 选择
    - `transformers`：`LocalTransformersToolCallingChatModel`（`<tool_call>` 协议）
    - 其它：默认 `get_chat_model()`（Ollama）
    """
    if settings.llm_backend == "transformers":
        from app.llm.transformers_tool_calling_chat_model import LocalTransformersToolCallingChatModel

        return LocalTransformersToolCallingChatModel()
    return get_chat_model()
