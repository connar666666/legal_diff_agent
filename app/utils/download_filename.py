"""根据 URL 与 HTTP Content-Type 生成稳定的本地文件名（避免 PDF 被误存为 .html）。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse


def stable_filename_for_download(url: str, content_type: str | None = None) -> str:
    """
    返回仅文件名（不含目录）。
    PDF 链接应保存为 .pdf；HTML 为 .html，便于后续 parse_file 分支处理。
    """
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    parsed = urlparse(url)
    base = Path(parsed.path).name or "law"
    stem = Path(base).stem if "." in base else base
    if len(stem) > 120:
        stem = stem[:120]
    suffix = Path(base).suffix.lower() if "." in base else ""
    ct = (content_type or "").split(";")[0].strip().lower()

    # 优先依据 Content-Type（部分站点 URL 无后缀但返回 PDF）
    if "application/pdf" in ct or ct == "application/x-pdf":
        return f"{stem}_{h}.pdf"
    if "text/html" in ct or "application/xhtml" in ct:
        return f"{stem}_{h}.html"

    # 再依据 URL 路径
    if suffix == ".pdf":
        return f"{stem}_{h}.pdf"
    if suffix in (".html", ".htm", ".xhtml"):
        return f"{stem}_{h}.html"

    # 模糊兜底
    if "pdf" in ct and "html" not in ct:
        return f"{stem}_{h}.pdf"

    # 历史行为：未知类型默认 html（网页为主）；若实为二进制可再依赖 Content-Type 分支
    return f"{stem}_{h}.html"
