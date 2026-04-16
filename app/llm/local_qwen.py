"""本地 Qwen / 兼容端点的统一入口（按场景分流）。

- **LangGraph / 工具调用**：请用 `app.llm.model_factory.get_chat_model_for_agent()`，
  在 `llm_backend=transformers` 时使用 `LocalTransformersToolCallingChatModel`，
  否则为 `ChatOllama`。
- **无 LangChain 的 HTTP 调试**：本文件的 `ollama_chat` / `ollama_generate`。
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def ollama_chat(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    base_url: Optional[str] = None,
) -> str:
    """
    调用 POST `{base}/api/chat`，返回 assistant 文本内容。
    messages 形如 [{"role":"user","content":"..."}, ...]
    """
    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/chat"
    payload = {
        "model": model or settings.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    with httpx.Client(timeout=settings.ollama_timeout_s) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    msg = data.get("message") or {}
    content = msg.get("content") or ""
    return content.strip()


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    base_url: Optional[str] = None,
) -> str:
    """调用 `/api/generate`（单轮补全）。"""
    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/generate"
    payload = {
        "model": model or settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    with httpx.Client(timeout=settings.ollama_timeout_s) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    return (data.get("response") or "").strip()


def ollama_list_models(base_url: Optional[str] = None) -> list[str]:
    """列出本地已拉取的模型名（Ollama `/api/tags`）。"""
    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/tags"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception as e:
        logger.warning("Could not list Ollama models: %s", e)
        return []


def get_unified_chat_model():
    """与 `build_agent_graph` 使用同一套模型选择逻辑（见 `model_factory`）。"""
    from app.llm.model_factory import get_chat_model_for_agent

    return get_chat_model_for_agent()
