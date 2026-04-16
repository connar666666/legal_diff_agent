"""重排序占位：后续可接入 Cross-Encoder 或 LLM 重排。"""

from __future__ import annotations

from typing import Any


def identity_rerank(
    query: str,
    hits: list[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    """默认不重排，直接返回。"""
    return list(hits)
