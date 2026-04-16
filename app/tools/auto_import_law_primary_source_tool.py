"""当检测到法规索引缺口时：自动发现候选 URL -> 下载 -> 写入 law.txt -> 重建索引。"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from app.config import settings
from app.tools import registry
from app.utils.download_filename import stable_filename_for_download

logger = logging.getLogger(__name__)


def _extract_ddg_links(html: str, *, base_url: str = "https://duckduckgo.com") -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for a in soup.select("a.result__a[href]"):
        href = (a.get("href") or "").strip()
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
    # 去重但保持顺序
    seen: set[str] = set()
    out: list[str] = []
    for x in links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _law_name_guess(user_text: str) -> str:
    t = user_text or ""
    # 粗粒度关键词 -> 法典名猜测（用于自动补齐缺口的兜底）
    if any(k in t for k in ["道路交通安全", "不礼让行人", "应急车道", "高速"]):
        return "道路交通安全法"
    if any(k in t for k in ["民法典", "侵权责任"]):
        return "民法典"
    return "道路交通安全法"


def _index_ready() -> bool:
    return (
        settings.resolve_law_bm25().exists()
        and settings.resolve_law_vector().exists()
        and (settings.data_index / "law_texts.json").exists()
    )


@tool
def auto_import_law_primary_source_tool(
    law_name: str = "",
    jurisdiction: str = "",
    *,
    official_domains: str = "",
    max_pages: int = 1,
    rebuild: bool = True,
) -> str:
    """
    自动补齐法规索引缺口（离线索引重建之前）。
    - 发现：优先官方域名 site:xxx 搜索候选
    - 下载：把候选页面保存到 data/raw/laws/
    - 归档：更新 data/raw/law.txt（JSONL 记录 law_name <-> url）
    - 索引：重建 data/index/ 的法规索引并让运行时 LawService 重载
    """

    # 如果索引已就绪，直接跳过
    if _index_ready() and rebuild:
        return json.dumps({"ok": True, "skipped": True, "reason": "law index 已就绪"}, ensure_ascii=False)

    # 若调用方未显式给出 law_name，则用非常粗的兜底猜测（用于触发补齐流程）
    law_name = (law_name or "").strip() or _law_name_guess(jurisdiction)
    query_parts = [law_name]
    if jurisdiction:
        query_parts.append(jurisdiction)
    query = " ".join([p for p in query_parts if p])

    domains = (official_domains.strip() or settings.official_law_domains).strip()
    domain_list = [d.strip() for d in domains.split(",") if d.strip()]
    # fail-fast：仅尝试少量域名，避免官方阶段网络不可用时耗时过久
    domain_list = domain_list[:1]

    headers = {"User-Agent": "Mozilla/5.0 (compatible; LegalDiffAgent/1.0)"}
    # fail-fast：避免网络受限时无限等待
    search_timeout = httpx.Timeout(connect=3.0, read=8.0, write=8.0, pool=3.0)
    download_timeout = httpx.Timeout(connect=3.0, read=12.0, write=12.0, pool=3.0)

    candidates: list[str] = []
    stage = "official"
    with httpx.Client(timeout=search_timeout, follow_redirects=True, headers=headers) as client:
        # official stage
        for domain in domain_list:
            q = f"{query} site:{domain}"
            ddg_url = f"https://duckduckgo.com/html/?q={quote_plus(q)}&kl=zh-cn"
            try:
                resp = client.get(ddg_url)
                resp.raise_for_status()
            except Exception as e:
                logger.warning("duckduckgo official search failed (%s): %s", domain, e)
                continue
            found = _extract_ddg_links(resp.text)
            found = [u for u in found if "pdf" not in u.lower()]
            candidates.extend(found)
            if candidates:
                break

        # general stage
        if not candidates:
            stage = "general"
            ddg_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}&kl=zh-cn"
            try:
                resp = client.get(ddg_url)
                resp.raise_for_status()
                candidates = _extract_ddg_links(resp.text)
            except Exception as e:
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"通用搜索失败：{e}",
                        "stage": stage,
                        "query": query,
                    },
                    ensure_ascii=False,
                )

    # 去噪
    candidates = [
        u
        for u in candidates
        if u
        and not re.search(r"(login|signin|javascript:)", u, flags=re.IGNORECASE)
        and len(u) < 5000
    ]
    candidates = candidates[: max_pages if max_pages > 0 else 1]
    if not candidates:
        return json.dumps(
            {
                "ok": False,
                "error": "未发现候选法律页面 URL",
                "stage": stage,
                "query": query,
            },
            ensure_ascii=False,
        )

    # 下载并归档
    settings.data_raw_laws.mkdir(parents=True, exist_ok=True)
    registry_path = settings.law_url_registry_path
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    downloaded: list[dict[str, str]] = []

    # 避免重复写入：简单读取已有映射
    existing: set[tuple[str, str]] = set()
    if registry_path.exists():
        for line in registry_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("law_name") and obj.get("url"):
                existing.add((str(obj["law_name"]), str(obj["url"])))

    for url in candidates:
        local_path: Path | None = None
        try:
            settings.data_raw_laws.mkdir(parents=True, exist_ok=True)
            with httpx.Client(timeout=download_timeout, follow_redirects=True, headers=headers) as client:
                resp = client.get(url)
                resp.raise_for_status()
                ct = resp.headers.get("content-type")
                filename = stable_filename_for_download(url, ct)
                local_path = settings.data_raw_laws / filename
                if not local_path.exists() or local_path.stat().st_size == 0:
                    local_path.write_bytes(resp.content)
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"下载候选页面失败: {e}",
                    "url": url,
                    "local_path": str(local_path) if local_path else "",
                    "stage": stage,
                },
                ensure_ascii=False,
            )

        downloaded.append({"url": url, "local_path": str(local_path)})

        key = (law_name, url)
        if key not in existing:
            record = {
                "law_name": law_name,
                "jurisdiction": jurisdiction,
                "url": url,
                "local_path": str(local_path),
                "note": f"auto_import stage={stage}",
                "ts": ts,
            }
            with registry_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing.add(key)

    # 重建索引并重载（仅当至少下载了一个页面时，避免空重建）
    if rebuild and downloaded:
        repo_root = settings.project_root
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "build_law_index.py"),
            str(settings.data_raw_laws),
            "--index-dir",
            str(settings.data_index),
        ]
        try:
            subprocess.run(cmd, check=True, cwd=str(repo_root))
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"build_law_index 执行失败: {e}",
                    "cmd": cmd,
                    "downloaded": downloaded,
                },
                ensure_ascii=False,
            )

        svc = registry.get_law_service()
        if svc:
            svc.load_from_paths()

    return json.dumps(
        {
            "ok": True,
            "stage": stage,
            "query": query,
            "law_name": law_name,
            "jurisdiction": jurisdiction,
            "downloaded": downloaded,
            "index_ready": _index_ready(),
        },
        ensure_ascii=False,
    )

