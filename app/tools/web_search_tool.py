"""通用网页搜索（DuckDuckGo HTML），不依赖嵌入模型或本地索引。"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.tools import tool

from app.config import settings
from app.utils.ddg_html import search_ddg_html_results

logger = logging.getLogger(__name__)


def _clean_query_for_search(user_text: str) -> str:
    """从整句里抽出适合搜索的查询（去掉 URL 等噪声）。"""
    t = (user_text or "").strip()
    if not t:
        return ""
    # 去掉 http(s) URL，保留其余描述
    t = re.sub(r"https?://[^\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) >= 2 else (user_text or "").strip()


@tool
def web_search_tool(query: str, max_results: int = 8) -> str:
    """
    通用网页搜索：用于事实查证、政策/新闻、网页来源线索等（非正式法律意见）。

    query: 搜索关键词或简短问题（中文/英文均可）。
    max_results: 返回条数上限，默认 8。
    返回 JSON：含标题、链接、部分摘要 snippet；若抓取失败会返回 ok=false。
    """
    q = _clean_query_for_search(query)
    if not q:
        return json.dumps({"ok": False, "error": "query 为空"}, ensure_ascii=False)

    mr = max(1, min(int(max_results or 8), settings.web_search_max_results))
    rows = search_ddg_html_results(q, max_results=mr)

    if not rows:
        return json.dumps(
            {
                "ok": False,
                "error": "未获取到搜索结果（DuckDuckGo HTML 无结果或解析失败/网络异常）",
                "query": q,
                "hint": "可缩短关键词重试，或检查网络与 DDG 可访问性",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "ok": True,
            "query": q,
            "engine": "duckduckgo_html",
            "results": rows,
        },
        ensure_ascii=False,
    )
