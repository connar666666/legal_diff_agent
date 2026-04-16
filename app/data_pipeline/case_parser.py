"""案例页面解析：把 HTML 文本抽取成可索引的文本。"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from bs4 import BeautifulSoup

from app.utils.text_utils import normalize_whitespace


def parse_case_html_to_text(raw_html: str, *, source_url: str = "") -> Tuple[str, str]:
    """从案例 HTML 提取标题与正文文本。"""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    title = ""
    # 常见标题位置：h1/h2/title
    h = soup.find(["h1", "h2", "title"])
    if h:
        title = normalize_whitespace(h.get_text(" ", strip=True))
    if not title:
        # fallback：source_url
        title = source_url.rsplit("/", 1)[-1] if source_url else "未命名案例"

    body = soup.get_text("\n", strip=False)
    text = normalize_whitespace(body)
    return title, text


def parse_case_file(path: Path, *, source_url: str = "") -> Tuple[str, str]:
    """文件 -> 标题/文本。自动根据扩展名读取为 HTML 或纯文本。"""
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".html", ".htm", ".xhtml"):
        return parse_case_html_to_text(raw, source_url=source_url or str(path))

    # 纯文本：以第一行当作标题
    first = raw.splitlines()[0].strip() if raw else ""
    title = first[:200] if first else path.stem
    text = normalize_whitespace(raw)
    return title, text

