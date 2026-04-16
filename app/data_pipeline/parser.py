"""将原始网页/HTML/纯文本解析为结构化法规记录。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from app.data_pipeline.chunker import chunk_law_text
from app.schema.models import LawDocumentMeta, LawChunkRecord
from app.utils.text_utils import normalize_whitespace, strip_html_noise


def _guess_jurisdiction_from_title(title: str) -> str:
    """从标题粗略猜测法域（可后续替换为规则库）。"""
    if not title:
        return ""
    if "中华人民共和国" in title or "全国" in title:
        return "全国"
    provinces = ("省", "市", "自治区", "特别行政区")
    for p in provinces:
        if p in title[:20]:
            return title[:20].strip()
    return ""


def parse_html_to_law_text(raw_html: str, source_url: str = "", raw_path: str | None = None) -> tuple[LawDocumentMeta, str]:
    """从 HTML 提取正文与标题。"""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    title = ""
    h = soup.find(["h1", "h2", "title"])
    if h:
        title = normalize_whitespace(h.get_text(" ", strip=True))
    body = soup.get_text("\n", strip=False)
    text = strip_html_noise(normalize_whitespace(body))
    meta = LawDocumentMeta(
        title=title or "未命名法规",
        jurisdiction=_guess_jurisdiction_from_title(title),
        source_type="法规",
        source_url=source_url,
        raw_path=raw_path,
    )
    return meta, text


def parse_plain_text_file(path: Path, source_url: str = "") -> tuple[LawDocumentMeta, str]:
    """读取纯文本文件。"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    first_line = raw.splitlines()[0].strip() if raw else ""
    title = first_line[:200] if first_line else path.stem
    meta = LawDocumentMeta(
        title=title,
        jurisdiction=_guess_jurisdiction_from_title(title),
        source_type="法规",
        source_url=source_url,
        raw_path=str(path),
    )
    text = normalize_whitespace(raw)
    return meta, text


def parse_pdf_file(path: Path, source_url: str = "") -> tuple[LawDocumentMeta, str]:
    """从 PDF 提取文本（扫描版无文本层时可能几乎为空）。"""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("解析 PDF 需要安装 pypdf：pip install pypdf") from e

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    text = normalize_whitespace("\n".join(parts))
    title = path.stem[:200]
    if not text.strip():
        text = f"[PDF 未提取到文本，可能为扫描件，请使用 OCR 或换文本版来源：{path.name}]"
    meta = LawDocumentMeta(
        title=title,
        jurisdiction=_guess_jurisdiction_from_title(title),
        source_type="法规",
        source_url=source_url,
        raw_path=str(path),
    )
    return meta, text


def parse_file(
    path: Path,
    source_url: str = "",
    encoding: str = "utf-8",
) -> tuple[LawDocumentMeta, str]:
    """按扩展名选择解析器。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf_file(path, source_url=source_url)
    if suffix in (".html", ".htm", ".xhtml"):
        raw = path.read_text(encoding=encoding, errors="replace")
        return parse_html_to_law_text(raw, source_url=source_url, raw_path=str(path))
    return parse_plain_text_file(path, source_url=source_url)


def law_text_to_chunks(
    meta: LawDocumentMeta,
    text: str,
    doc_id: str,
) -> list[LawChunkRecord]:
    """
    将整篇法规文本转为结构化片段列表（先按条，再按款/项细分）。
    实现委托 `chunker.chunk_law_text`，与流水线共用同一套切分逻辑。
    """
    return chunk_law_text(meta, text, doc_id)


def parse_and_chunk(
    path: Path,
    doc_id: Optional[str] = None,
    source_url: str = "",
) -> list[LawChunkRecord]:
    """便捷：文件 -> 解析 -> 切条。"""
    meta, text = parse_file(path, source_url=source_url)
    did = doc_id or path.stem
    return law_text_to_chunks(meta, text, did)
