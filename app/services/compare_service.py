"""多地法规结构化对比（基于检索结果做启发式对齐）。"""

from __future__ import annotations

from typing import Any

from app.schema.models import CompareRow
from app.services.law_service import LawService


def compare_jurisdictions(
    law_service: LawService,
    topic: str,
    jurisdiction_a: str,
    jurisdiction_b: str,
    max_items: int = 8,
) -> list[dict[str, Any]]:
    """
    对两个法域各检索若干条，按主题做简单两列对比（非深度语义对齐，便于首版展示）。
    """
    hits_a = law_service.search(
        f"{topic} {jurisdiction_a}", jurisdiction=jurisdiction_a, top_k=max_items
    )
    hits_b = law_service.search(
        f"{topic} {jurisdiction_b}", jurisdiction=jurisdiction_b, top_k=max_items
    )
    rows: list[CompareRow] = []
    n = max(len(hits_a), len(hits_b))
    for i in range(min(n, max_items)):
        ta = hits_a[i]["text"] if i < len(hits_a) else ""
        tb = hits_b[i]["text"] if i < len(hits_b) else ""
        rows.append(
            CompareRow(
                aspect=f"检索片段 {i + 1}",
                jurisdiction_a=jurisdiction_a,
                jurisdiction_b=jurisdiction_b,
                content_a=ta[:2000],
                content_b=tb[:2000],
                note="启发式对齐，正式产品需条文级对齐模型",
            )
        )
    return [r.model_dump() for r in rows]
