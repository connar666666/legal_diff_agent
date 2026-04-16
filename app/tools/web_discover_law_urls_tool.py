"""根据“法典/法律名称 + 地点/法域”在线发现候选一手法律页面 URL。"""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import quote_plus

import httpx
from langchain_core.tools import tool

from app.config import settings
from app.utils.ddg_html import extract_ddg_links

logger = logging.getLogger(__name__)


@tool
def discover_law_urls_tool(
    law_name: str,
    jurisdiction: str = "",
    *,
    official_domains: str = "",
    max_results: int = 10,
) -> str:
    """
    在线发现“可能的一手法律/法规页面 URL”。

    law_name: 法典/法律名称（中文优先，如“道路交通安全法”“道路交通安全条例”）
    jurisdiction: 地点/法域（如“深圳/广东/北京/全国”等）
    official_domains: 优先域名列表（逗号分隔）。为空时使用 settings.official_law_domains。
    """

    # 组装检索词：把法域信息加进去，提高匹配概率
    query_parts = [law_name.strip()]
    if jurisdiction:
        query_parts.append(jurisdiction.strip())
    query = " ".join([p for p in query_parts if p])

    domains = official_domains.strip() or settings.official_law_domains
    domain_list = [d.strip() for d in domains.split(",") if d.strip()]

    # 1) 优先官方站点：用 site:domain 搜索
    search_timeout = settings.web_search_timeout_s
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LegalDiffAgent/1.0)"}
    urls: list[str] = []

    with httpx.Client(timeout=search_timeout, follow_redirects=True, headers=headers) as client:
        for domain in domain_list:
            q = f"{query} site:{domain}"
            ddg_url = f"https://duckduckgo.com/html/?q={quote_plus(q)}&kl=zh-cn"
            try:
                resp = client.get(ddg_url)
                resp.raise_for_status()
            except Exception as e:
                logger.warning("duckduckgo official search failed (%s): %s", domain, e)
                continue

            found = extract_ddg_links(resp.text)
            # 简单过滤：避免明显的“法律条文解读/工具软件页面”等非正文倾向
            found = [u for u in found if "pdf" not in u.lower()]
            urls.extend(found)
            if urls:
                break

    # 去重
    seen2: set[str] = set()
    unique_official: list[str] = []
    for u in urls:
        if u not in seen2:
            seen2.add(u)
            unique_official.append(u)

    # 2) 若官方阶段无结果：通用搜索
    stage = "official"
    final_list = unique_official
    if not final_list:
        stage = "general"
        ddg_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}&kl=zh-cn"
        with httpx.Client(timeout=search_timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(ddg_url)
            resp.raise_for_status()
            final_list = extract_ddg_links(resp.text)

    # 限制数量
    final_list = final_list[: max_results if max_results > 0 else 10]

    # 轻量去噪：剔除明显的空壳/跳转
    final_list = [
        u
        for u in final_list
        if u and not re.search(r"(login|signin|javascript:)", u, flags=re.IGNORECASE)
    ]

    return json.dumps(
        {
            "ok": True,
            "stage": stage,
            "query": query,
            "law_name": law_name,
            "jurisdiction": jurisdiction,
            "official_candidates": unique_official[: max_results],
            "candidates": final_list,
        },
        ensure_ascii=False,
    )

