"""Baseline 2：同底座 Transformers + 网页搜索摘要拼上下文（不入库、不走 Agent 工具链）。"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.llm.local_transformers import get_local_generator
from app.tools.web_search_tool import web_search_tool

logger = logging.getLogger(__name__)

_SEARCH_QUERY_SYS = (
    "你是搜索助手。用户将提出与法律、政策或事实相关的问题。\n"
    "请只输出一行：你认为最适合用于网页搜索的简短关键词或短语（建议 8–40 个汉字），"
    "不要解释、不要标点列表、不要输出第二行。"
)

_ANSWER_SYS = (
    "你是法律信息助手。你将看到「用户问题」和「网页搜索摘要」。\n"
    "请仅根据摘要中明确出现的内容作答；若摘要不足以支持具体「第×条」或法规全称，请明确说明并勿编造条号。\n"
    "如引用网页信息，可简要注明来自搜索摘要；不要使用 <tool_call>。"
)


def _first_line(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    return t.split("\n")[0].strip()[:500]


def generate_search_query(question: str) -> str:
    gen = get_local_generator()
    raw = gen.chat(
        [
            {"role": "system", "content": _SEARCH_QUERY_SYS},
            {"role": "user", "content": question},
        ],
        clean_think=True,
    )
    q = _first_line(raw)
    return q or question[:200]


def run_web_search(query: str, max_results: int = 8) -> dict[str, Any]:
    raw = web_search_tool.invoke({"query": query, "max_results": max_results})
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid tool json", "raw": str(raw)[:2000]}


def format_evidence(search_payload: dict[str, Any], max_chars: int = 12000) -> str:
    if not search_payload.get("ok"):
        return f"[搜索失败] {search_payload.get('error', '')}"
    lines: list[str] = []
    for i, row in enumerate(search_payload.get("results") or [], start=1):
        title = row.get("title") or ""
        url = row.get("url") or ""
        snip = row.get("snippet") or ""
        lines.append(f"{i}. {title}\n   URL: {url}\n   摘要: {snip}\n")
    blob = "\n".join(lines)
    return blob[:max_chars]


def run_baseline_web_rag(
    question: str,
    *,
    max_results: int = 8,
    second_search_on_empty: bool = True,
) -> dict[str, Any]:
    """
    两阶段：模型产搜索词 → DDG 搜索 → 模型根据摘要作答。
    返回 dict：answer, search_queries, search_payloads, evidence_excerpt
    """
    queries: list[str] = []
    payloads: list[dict[str, Any]] = []

    q1 = generate_search_query(question)
    queries.append(q1)
    p1 = run_web_search(q1, max_results=max_results)
    payloads.append(p1)

    if second_search_on_empty and not p1.get("ok") and q1.strip() != question.strip()[:500]:
        q2 = question[:200]
        queries.append(q2)
        p2 = run_web_search(q2, max_results=max_results)
        payloads.append(p2)
        evidence = format_evidence(p2 if p2.get("ok") else p1)
    else:
        evidence = format_evidence(p1)

    gen = get_local_generator()
    user_block = f"用户问题：\n{question}\n\n以下为网页搜索摘要（可能不完整）：\n\n{evidence}\n\n请作答。"
    answer = gen.chat(
        [
            {"role": "system", "content": _ANSWER_SYS},
            {"role": "user", "content": user_block},
        ],
        clean_think=True,
    )
    return {
        "answer": (answer or "").strip(),
        "search_queries": queries,
        "search_payloads": payloads,
        "evidence_excerpt": evidence[:8000],
    }
