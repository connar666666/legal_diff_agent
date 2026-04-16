"""文本清洗与轻量规范化。"""

import re
import unicodedata
from typing import Iterable


def normalize_whitespace(text: str) -> str:
    """合并空白、去除首尾空白。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html_noise(text: str) -> str:
    """去掉常见 HTML 残留标记（非完整解析，仅作后处理）。"""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(text)


def iter_article_markers(text: str) -> Iterable[tuple[int, str]]:
    """粗略定位「第…条」在文本中的位置，供分块参考。"""
    pattern = re.compile(r"(第[零一二三四五六七八九十百千万\d]+条)")
    for m in pattern.finditer(text):
        yield m.start(), m.group(1)
