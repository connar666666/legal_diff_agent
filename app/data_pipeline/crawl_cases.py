"""案例抓取：首版以手工或公开接口为准；此处预留下载占位。"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse
import hashlib

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def download_case_page(url: str, dest_dir: Path | None = None) -> Path:
    """将案例页面保存到 data/raw/cases/。"""
    dest = dest_dir or settings.data_raw_cases
    dest.mkdir(parents=True, exist_ok=True)

    # 为避免一次下载多个 URL 时互相覆盖：根据 URL 命名；若 URL 不含合适文件名则使用哈希。
    parsed = urlparse(url)
    base = Path(parsed.path).name
    if not base:
        base = f"case_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}.html"
    elif not base.endswith((".html", ".htm")):
        base = base + ".html"

    out = dest / base
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        out.write_bytes(r.content)
    logger.info("Saved %s", out)
    return out
