"""条文级语义对齐：在两地检索候选片段之间，用句向量余弦相似度做贪心最优配对。"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from app.config import settings
from app.retrieval.embedding import encode_texts
from app.schema.models import CompareRow

logger = logging.getLogger(__name__)

_ARTICLE = re.compile(r"(第[零一二三四五六七八九十百千万\d]+条)")


def extract_article_hint(text: str, search_window: int = 200) -> str:
    """从片段正文前部提取「第…条」作为展示线索（索引中未必单独存条号）。"""
    t = (text or "").strip()
    if not t:
        return ""
    head = t[:search_window]
    m = _ARTICLE.search(head)
    if m:
        return m.group(1)
    m2 = _ARTICLE.search(t)
    if m2:
        return m2.group(1)
    return ""


def _prep_for_embedding(topic: str, text: str) -> str:
    """主题 + 正文，便于同一主题下的跨地语义对齐。"""
    body = (text or "").strip()
    if len(body) > 2400:
        body = body[:2400]
    if topic:
        return f"{topic.strip()}\n{body}"
    return body


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, d), b: (m, d) -> (n, m)"""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_n @ b_n.T


def greedy_pair_by_similarity(
    sim: np.ndarray,
    min_sim: float,
    max_pairs: int,
) -> list[tuple[int, int, float]]:
    """按相似度全局降序贪心配对，每个 A、B 片段至多参与一对。"""
    n, m = sim.shape
    pairs: list[tuple[int, int, float]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    cand: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(m):
            cand.append((float(sim[i, j]), i, j))
    cand.sort(reverse=True, key=lambda x: x[0])
    for s, i, j in cand:
        if s < min_sim:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, s))
        if len(pairs) >= max_pairs:
            break
    # 稳定展示：按 A 侧索引排序
    pairs.sort(key=lambda x: x[0])
    return pairs


def semantic_align_jurisdictions(
    hits_a: list[dict[str, Any]],
    hits_b: list[dict[str, Any]],
    topic: str,
    jurisdiction_a: str,
    jurisdiction_b: str,
    *,
    max_pairs: int | None = None,
    min_similarity: float | None = None,
) -> list[CompareRow]:
    """
    对两地检索结果做「条文片段级」语义对齐：
    句向量编码（与索引构建同一套 embedding）→ 余弦相似度矩阵 → 贪心最优配对。
    """
    max_pairs = max_pairs or settings.compare_max_aligned_pairs
    min_similarity = (
        min_similarity
        if min_similarity is not None
        else settings.compare_semantic_min_similarity
    )

    if not hits_a and not hits_b:
        return []

    if not hits_a:
        return [
            CompareRow(
                aspect=f"仅检索到「{jurisdiction_b}」",
                jurisdiction_a=jurisdiction_a,
                jurisdiction_b=jurisdiction_b,
                content_a="（未检索到该法域下与主题相关的法规片段）",
                content_b=(hits_b[0].get("text") or "")[:2000],
                note="请为法域 A 补充法规数据或调整关键词后重建索引。",
                alignment_method="single_side",
            )
        ]

    if not hits_b:
        return [
            CompareRow(
                aspect=f"仅检索到「{jurisdiction_a}」",
                jurisdiction_a=jurisdiction_a,
                jurisdiction_b=jurisdiction_b,
                content_a=(hits_a[0].get("text") or "")[:2000],
                content_b="（未检索到该法域下与主题相关的法规片段）",
                note="请为法域 B 补充法规数据或调整关键词后重建索引。",
                alignment_method="single_side",
            )
        ]

    texts_a = [_prep_for_embedding(topic, h.get("text", "")) for h in hits_a]
    texts_b = [_prep_for_embedding(topic, h.get("text", "")) for h in hits_b]

    try:
        emb_a = encode_texts(texts_a)
        emb_b = encode_texts(texts_b)
    except Exception as e:
        logger.exception("embedding failed in semantic alignment: %s", e)
        return _fallback_index_pairs(hits_a, hits_b, topic, jurisdiction_a, jurisdiction_b)

    sim = _cosine_similarity_matrix(emb_a, emb_b)
    pairs = greedy_pair_by_similarity(sim, min_similarity, max_pairs)

    rows: list[CompareRow] = []
    for rank, (ia, ib, score) in enumerate(pairs, start=1):
        ha, hb = hits_a[ia], hits_b[ib]
        raw_a = (ha.get("text") or "")[:2000]
        raw_b = (hb.get("text") or "")[:2000]
        hint_a = extract_article_hint(raw_a)
        hint_b = extract_article_hint(raw_b)
        label_a = hint_a or f"片段 {ia + 1}"
        label_b = hint_b or f"片段 {ib + 1}"
        aspect = f"语义对齐 #{rank}：{label_a} ↔ {label_b}（相似度 {score:.2f}）"

        rows.append(
            CompareRow(
                aspect=aspect,
                jurisdiction_a=jurisdiction_a,
                jurisdiction_b=jurisdiction_b,
                content_a=raw_a,
                content_b=raw_b,
                note="基于句向量余弦相似度的跨法域片段配对；条号取自正文前缀，若无法识别则显示「片段」编号。",
                similarity_score=round(score, 4),
                chunk_id_a=str(ha.get("id", "")),
                chunk_id_b=str(hb.get("id", "")),
                article_hint_a=hint_a,
                article_hint_b=hint_b,
                alignment_method="semantic_embedding_greedy",
            )
        )

    if not rows:
        return _fallback_index_pairs(hits_a, hits_b, topic, jurisdiction_a, jurisdiction_b)

    return rows


def _fallback_index_pairs(
    hits_a: list[dict[str, Any]],
    hits_b: list[dict[str, Any]],
    topic: str,
    jurisdiction_a: str,
    jurisdiction_b: str,
) -> list[CompareRow]:
    """相似度均低于阈值或配对为空时，回退为按检索排名并列（并标明）。"""
    n = min(len(hits_a), len(hits_b), settings.compare_max_aligned_pairs)
    rows: list[CompareRow] = []
    for i in range(n):
        ta = (hits_a[i].get("text") or "")[:2000]
        tb = (hits_b[i].get("text") or "")[:2000]
        rows.append(
            CompareRow(
                aspect=f"检索排名对齐 #{i + 1}（语义配对未达阈值，主题：{topic[:40]}）",
                jurisdiction_a=jurisdiction_a,
                jurisdiction_b=jurisdiction_b,
                content_a=ta,
                content_b=tb,
                note="未找到足够高相似度的跨法域配对，已按检索排名并列展示。",
                alignment_method="retrieval_rank_fallback",
            )
        )
    return rows
