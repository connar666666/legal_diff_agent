"""法规抓取：首版建议手动下载后放入 data/raw/laws/，本脚本预留 HTTP 下载入口。"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def download_to_raw(url: str, dest_dir: Path | None = None) -> Path:
    """
    将 URL 内容保存为 raw 文件（需自行保证来源与版权合规）。
    """
    dest = dest_dir or settings.data_raw_laws
    dest.mkdir(parents=True, exist_ok=True)
    name = Path(urlparse(url).path).name or "download.html"
    out = dest / name
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        out.write_bytes(r.content)
    logger.info("Saved %s", out)
    return out
