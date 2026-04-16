"""本地 Transformers -> LangGraph 工具调用适配器（输出 AIMessage.tool_calls）。"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Iterable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableBinding

from app.llm.local_transformers import get_local_generator
from app.utils.debug_logger import JSONLWriter
from app.graph.routing import classify_intent
from app.utils.output_parser import (
    build_tool_instruction_for_prompt,
    parse_tool_calls_from_text,
)

logger = logging.getLogger(__name__)


def _messages_to_chat_template_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": str(m.content)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": str(m.content)})
        elif isinstance(m, ToolMessage):
            # Qwen chat template 通常不认识 tool role；这里把 tool 输出当普通用户内容喂回去
            name = getattr(m, "tool_call_id", "") or "tool"
            out.append({"role": "user", "content": f"[TOOL_RESULT {name}] {str(m.content)}"})
        else:
            out.append({"role": "user", "content": str(getattr(m, "content", ""))})
    return out


def _infer_intent_from_messages(messages: list[BaseMessage]) -> str:
    """从最后一条 user 句子推断意图（用于决定是否强制 tool_call）。"""
    from langchain_core.messages import HumanMessage

    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            try:
                return str(classify_intent(str(getattr(m, "content", "") or "")))
            except Exception:
                break
    return "general"


def _last_user_text(messages: list[BaseMessage]) -> str:
    from langchain_core.messages import HumanMessage

    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(getattr(m, "content", "") or "")
    return ""


def _guess_jurisdictions(text: str) -> list[str]:
    candidates = ["北京", "上海", "深圳", "广州", "杭州", "天津", "重庆", "南京", "苏州"]
    out = [c for c in candidates if c in text]
    return out[:3]


def _fallback_tool_calls(intent: str, user_text: str) -> list[dict[str, Any]]:
    """模型两次都没给出 tool_call 时的兜底规划，保证 React 工具链可执行。"""
    tool_calls: list[dict[str, Any]] = []
    cities = _guess_jurisdictions(user_text)

    def _new_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": name,
            "args": args,
            "id": str(uuid.uuid4()),
            "type": "tool_call",
        }

    if intent == "law":
        if cities:
            for city in cities:
                tool_calls.append(
                    _new_call("search_law_tool", {"query": user_text, "jurisdiction": city})
                )
        else:
            tool_calls.append(
                _new_call("search_law_tool", {"query": user_text, "jurisdiction": ""})
            )
    elif intent == "case":
        tool_calls.append(_new_call("search_case_tool", {"query": user_text}))
    elif intent == "compare":
        if len(cities) >= 2:
            tool_calls.append(
                _new_call(
                    "compare_tool",
                    {"topic": user_text, "jurisdiction_a": cities[0], "jurisdiction_b": cities[1]},
                )
            )
        else:
            # compare 缺城市时先查法规
            tool_calls.append(
                _new_call("search_law_tool", {"query": user_text, "jurisdiction": ""})
            )
    elif intent == "export":
        tool_calls.append(
            _new_call("export_tool", {"content": "自动导出占位：请使用上一轮回答内容。", "filename": "export.md"})
        )
    elif intent == "web":
        urls = re.findall(r"https?://[^\s<>\"']+", user_text)
        if urls:
            tool_calls.append(
                _new_call(
                    "fetch_law_primary_source_tool",
                    {
                        "url": urls[0],
                        "law_name": "用户提供的链接",
                        "jurisdiction": "",
                        "note": "web_intent_fallback",
                    },
                )
            )
        else:
            tool_calls.append(
                _new_call("web_search_tool", {"query": user_text, "max_results": 8})
            )
    else:
        # general：通用网页搜索（不依赖本地法规索引）；需要法条时再让模型调用 search_*
        tool_calls.append(_new_call("web_search_tool", {"query": user_text, "max_results": 8}))

    return tool_calls


class LocalTransformersToolCallingChatModel(BaseChatModel):
    """把本地 Transformers 的生成结果转换成 LangChain AIMessage.tool_calls。"""

    def __init__(self) -> None:
        super().__init__()
        self._last_tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "local-transformers-tool-calling"

    def bind_tools(self, tools: list[Any], **kwargs: Any):  # type: ignore[override]
        # create_react_agent 会调用 bind_tools；把 tools 以 RunnableBinding 的形式传下去
        return RunnableBinding(bound=self, kwargs={"tools": tools})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        *,
        tools: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = tools or []
        self._last_tools = list(tools)

        from app.config import settings

        writer: Optional[JSONLWriter] = (
            JSONLWriter(settings.tool_debug_log_path) if settings.tool_debug_enabled else None
        )

        # 追加工具调用说明（仅在 tools 存在时）
        chat_dicts = _messages_to_chat_template_dicts(messages)
        intent = _infer_intent_from_messages(messages)
        user_text = _last_user_text(messages)

        has_tool_results = any(isinstance(m, ToolMessage) for m in messages)

        # 若工具返回“索引未加载”这类可恢复缺口，则允许再次强制 tool_call，
        # 以驱动 discover/fetch/build 等补齐流程。
        # 只看“最新一次工具输出”，避免历史缺口错误导致后续也一直 force_tool_call。
        last_tool: Optional[ToolMessage] = None
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                last_tool = m
                break

        last_content = str(getattr(last_tool, "content", "") or "") if last_tool else ""
        needs_law_index_fix = "法规索引未加载" in last_content
        already_built = ("\"index_ready\"" in last_content) or ("\"skipped\"" in last_content)

        force_tool_call = bool(tools) and intent in ("law", "case", "compare", "export", "web") and (
            not has_tool_results or (needs_law_index_fix and not already_built)
        )

        if tools:
            # 如果没有缺口修复需求，则工具结果后进入“回答模式（不再调用工具）”。
            # 但当检测到“法规索引未加载”等可恢复缺口时，必须允许模型继续调用工具补齐。
            if has_tool_results and not (needs_law_index_fix and not already_built):
                summary_system_msg = [
                    {
                        "role": "system",
                        "content": (
                            "工具检索已完成。请根据上方 [TOOL_RESULT] 中的信息直接回答用户的问题。"
                            "不要再调用任何工具，不要输出 <tool_call> 标签，直接给出完整的文字回答。"
                            "如果工具返回了错误（如索引未构建），请告知用户需要先运行数据索引构建脚本，"
                            "并简要解释如何操作（如 `python scripts/build_law_index.py`）。"
                        ),
                    }
                ]
                chat_dicts = summary_system_msg + chat_dicts
            else:
                tool_instruction = build_tool_instruction_for_prompt(tools)
                tool_system_msg = [{"role": "system", "content": tool_instruction}]
                if force_tool_call:
                    tool_system_msg.insert(
                        0,
                        {
                            "role": "system",
                            "content": (
                                f"IMPORTANT: 当前检测到意图为 `{intent}`。"
                                "你必须输出一个或多个 `<tool_call>{...}</tool_call>`，"
                                "并且只允许包含 `<tool_call>` 标签；不要输出任何其它文本（包括 <think>）。"
                            ),
                        },
                    )
                chat_dicts = tool_system_msg + chat_dicts

        if writer is not None:
            writer.write(
                {
                    "event": "transformers_generate_input",
                    "tool_count": len(tools),
                    "intent": intent,
                    "force_tool_call": force_tool_call,
                    "messages_preview": [
                        {"role": m.get("role", ""), "content": (m.get("content", "")[:500] if isinstance(m.get("content", ""), str) else "")}
                        for m in chat_dicts[:8]
                    ],
                }
            )

        text = get_local_generator().chat(chat_dicts, clean_think=True)

        if writer is not None:
            writer.write(
                {
                    "event": "transformers_generate_raw",
                    "raw_preview": (text or "")[:8000],
                }
            )
        tool_calls = parse_tool_calls_from_text(text)

        if writer is not None:
            writer.write(
                {
                    "event": "transformers_tool_calls_parsed",
                    "tool_calls": [
                        {"name": c.get("name"), "id": c.get("id"), "args_preview": str(c.get("args", {}))[:2000]}
                        for c in tool_calls
                    ],
                    "tool_calls_count": len(tool_calls),
                }
            )

        # 解析失败：如果强制 tool_call 的意图没有解析到 tool_calls，进行一次重试（更强硬的 system 指令）
        if force_tool_call and tools and not tool_calls:
            if writer is not None:
                writer.write(
                    {
                        "event": "transformers_tool_calls_retry",
                        "reason": "force_tool_call but parsed 0 tool_calls",
                    }
                )

            strict_msg = [
                {
                    "role": "system",
                    "content": f"STRICT MODE: intent=`{intent}`. 你必须只输出 `<tool_call>{{...}}</tool_call>` 标签块，不要输出任何其他字符/文本。若不需要工具也必须仍然输出一个 tool_call（用于检索/对比）。",
                },
            ] + chat_dicts

            text2 = get_local_generator().chat(strict_msg, clean_think=True)
            if writer is not None:
                writer.write(
                    {
                        "event": "transformers_generate_raw_retry",
                        "raw_preview": (text2 or "")[:8000],
                    }
                )
            tool_calls2 = parse_tool_calls_from_text(text2)
            if writer is not None:
                writer.write(
                    {
                        "event": "transformers_tool_calls_parsed_retry",
                        "tool_calls_count": len(tool_calls2),
                        "tool_calls": [
                            {"name": c.get("name"), "id": c.get("id")}
                            for c in tool_calls2
                        ],
                    }
                )
            tool_calls = tool_calls2

        # 兜底：模型在强制模式仍不输出 tool_call，后端按规则生成，确保工具链执行
        if force_tool_call and tools and not tool_calls:
            if needs_law_index_fix:
                # 缺口自动补齐：一键发现+下载+入库+重建索引
                cities = _guess_jurisdictions(user_text)
                guessed_law_name = "道路交通安全法"
                if any(k in user_text for k in ["民法典", "侵权责任"]):
                    guessed_law_name = "民法典"

                tool_calls = [
                    {
                        "name": "auto_import_law_primary_source_tool",
                        "args": {
                            "law_name": guessed_law_name,
                            "jurisdiction": cities[0] if cities else "",
                        },
                        "id": str(uuid.uuid4()),
                        "type": "tool_call",
                    }
                ]
            else:
                tool_calls = _fallback_tool_calls(intent, user_text)
            if writer is not None:
                writer.write(
                    {
                        "event": "transformers_tool_calls_fallback",
                        "reason": "model did not output parsable tool_calls after retry",
                        "intent": intent,
                        "user_text_preview": user_text[:500],
                        "tool_calls": [
                            {"name": c.get("name"), "id": c.get("id"), "args_preview": str(c.get("args", {}))[:1000]}
                            for c in tool_calls
                        ],
                    }
                )

        if tool_calls:
            # 工具调用模式：content 可为空；ToolNode 执行后再由模型继续生成最终回答。
            msg = AIMessage(content="", tool_calls=tool_calls)
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # 普通回答模式
        cleaned = text.strip()
        msg = AIMessage(content=cleaned)
        return ChatResult(generations=[ChatGeneration(message=msg)])

