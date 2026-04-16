"""下载候选法律页面到 data/raw/laws，并更新 law_url_registry_path 映射文件。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import httpx
from langchain_core.tools import tool

from app.config import settings
from app.utils.download_filename import stable_filename_for_download

logger = logging.getLogger(__name__)


@tool
def fetch_law_primary_source_tool(
    url: str,
    law_name: str,
    jurisdiction: str = "",
    *,
    note: str = "",
) -> str:
    """
    下载一手法律/法规页面或 PDF，并保存到 data/raw/laws/。

    扩展名由 URL 与 HTTP Content-Type 决定（PDF 存为 .pdf，便于索引阶段用 pypdf 解析）。
    同时更新 data/raw/law.txt（JSONL，便于后续快速反查 URL）。
    """

    if not url or not url.startswith(("http://", "https://")):
        return json.dumps({"ok": False, "error": "URL 格式不合法"}, ensure_ascii=False)

    out_dir: Path = settings.data_raw_laws
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; LegalDiffAgent/1.0)"}
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type")
        filename = stable_filename_for_download(url, content_type)
        local_path = out_dir / filename

        # 已存在同名且非空则跳过写入（换扩展名后会重新拉取为新文件）
        if not local_path.exists() or local_path.stat().st_size == 0:
            local_path.write_bytes(resp.content)
            logger.info("Fetched %s -> %s (content-type=%s)", url, local_path, content_type)

    # 更新 URL 映射（JSONL：每行一个映射记录）
    reg_path = settings.law_url_registry_path
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    record = {
        "law_name": law_name,
        "jurisdiction": jurisdiction,
        "url": url,
        "local_path": str(local_path),
        "note": note,
        "ts": ts,
    }

    # 若同一 law_name+url 已存在则不重复追加（线性查找即可，文件通常不大）
    exists = False
    if reg_path.exists():
        try:
            for line in reg_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("law_name") == law_name and obj.get("url") == url:
                    exists = True
                    break
        except Exception:
            exists = False

    if not exists:
        with reg_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return json.dumps(
        {
            "ok": True,
            "url": url,
            "law_name": law_name,
            "jurisdiction": jurisdiction,
            "local_path": str(local_path),
            "registry_path": str(reg_path),
        },
        ensure_ascii=False,
    )

