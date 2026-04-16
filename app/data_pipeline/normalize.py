"""处理后的字段规范化（法域、标题、编号格式等）。"""

from __future__ import annotations

import re

from app.utils.text_utils import normalize_whitespace


def normalize_title(title: str) -> str:
    t = normalize_whitespace(title)
    t = re.sub(r"[\s　]+", " ", t)
    return t.strip()


def normalize_jurisdiction(raw: str) -> str:
    """粗清洗法域字符串。"""
    return normalize_whitespace(raw)
