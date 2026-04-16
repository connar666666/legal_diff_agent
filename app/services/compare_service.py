"""多地法规对比：条文级语义对齐（句向量）+ 检索排名兜底。"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.schema.models import CompareRow
from app.services.article_alignment import semantic_align_jurisdictions
from app.services.law_service import LawService


def compare_jurisdictions(
    law_service: LawService,
    topic: str,
    jurisdiction_a: str,
    jurisdiction_b: str,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """
    对两个法域在某一主题下的法规片段做对比。

    1. 各自检索较大候选池（compare_retrieval_top_k）
    2. 使用与建索引相同的句向量模型，对候选正文编码
    3. 余弦相似度矩阵 + 贪心配对 → 条文片段级语义对齐
    4. 若配对为空或嵌入失败，回退为「检索排名并列」
    """
    top_k = max_items or settings.compare_retrieval_top_k
    q_base = f"{topic} {jurisdiction_a}".strip()
    q_b = f"{topic} {jurisdiction_b}".strip()

    hits_a = law_service.search(q_base, jurisdiction=jurisdiction_a, top_k=top_k)
    hits_b = law_service.search(q_b, jurisdiction=jurisdiction_b, top_k=top_k)

    rows: list[CompareRow] = semantic_align_jurisdictions(
        hits_a,
        hits_b,
        topic=topic,
        jurisdiction_a=jurisdiction_a,
        jurisdiction_b=jurisdiction_b,
    )
    return [r.model_dump() for r in rows]
