"""DuckDuckGo HTML 结果页解析（无官方 API，页面结构变化可能导致失效）。"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger(__name__)


def extract_ddg_links(html: str, *, base_url: str = "https://duckduckgo.com") -> list[str]:
    """从 DDG HTML 结果页提取外链 URL（去重保序）。"""
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []

    for a in soup.select("a.result__a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue

        if href.startswith("/l/"):
            parsed = urlparse(base_url + href)
            qs = parse_qs(parsed.query)
            uddg = qs.get("uddg", [])
            if uddg:
                links.append(uddg[0])
                continue

        if href.startswith("http://") or href.startswith("https://"):
            links.append(href)

    seen: set[str] = set()
    out: list[str] = []
    for x in links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def search_ddg_html_results(
    query: str,
    *,
    max_results: int = 8,
    region: str = "zh-cn",
    timeout_s: float | None = None,
) -> list[dict[str, Any]]:
    """
    请求 DDG HTML 搜索，返回带标题与链接的结果列表。
    """
    q = (query or "").strip()
    if not q:
        return []

    timeout = timeout_s if timeout_s is not None else settings.web_search_timeout_s
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LegalDiffAgent/1.0; +https://github.com)"}
    ddg_url = f"https://duckduckgo.com/html/?q={quote_plus(q)}&kl={region}"

    out: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(ddg_url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.warning("duckduckgo html search failed: %s", e)
        return []

    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()

    for a in soup.select("a.result__a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue

        resolved = href
        if href.startswith("/l/"):
            parsed = urlparse("https://duckduckgo.com" + href)
            qs = parse_qs(parsed.query)
            uddg = qs.get("uddg", [])
            if uddg:
                resolved = uddg[0]
            else:
                continue
        elif not (href.startswith("http://") or href.startswith("https://")):
            continue

        if resolved in seen:
            continue
        seen.add(resolved)

        title = a.get_text(strip=True) or resolved
        snippet = ""
        parent = a.parent
        if parent is not None:
            sn = parent.select_one(".result__snippet")
            if sn is not None:
                snippet = sn.get_text(strip=True)

        out.append({"title": title, "url": resolved, "snippet": snippet})
        if len(out) >= max_results:
            break

    return out
