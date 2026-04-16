"""BM25 与向量召回的加权融合（RRF 风格简化版）。"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from app.config import settings
from app.retrieval.bm25_index import BM25LawIndex
from app.retrieval.embedding import encode_query
from app.retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse_weighted(
    bm25_hits: list[tuple[str, float]],
    vec_hits: list[tuple[str, float]],
    bm25_w: float,
    vec_w: float,
) -> list[tuple[str, float]]:
    """将两组分数分别 min-max 后加权求和。"""
    b: dict[str, float] = {i: s for i, s in bm25_hits}
    v: dict[str, float] = {i: s for i, s in vec_hits}
    keys = set(b) | set(v)
    if not keys:
        return []
    nb = _min_max({k: b.get(k, 0.0) for k in keys})
    nv = _min_max({k: v.get(k, 0.0) for k in keys})
    fused: dict[str, float] = {}
    for k in keys:
        fused[k] = bm25_w * nb.get(k, 0.0) + vec_w * nv.get(k, 0.0)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """法规或案例域的混合检索。"""

    def __init__(
        self,
        bm25: BM25LawIndex,
        vectors: FaissVectorStore,
        id_to_text: dict[str, str],
        bm25_top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        fusion_top_k: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
    ) -> None:
        self.bm25 = bm25
        self.vectors = vectors
        self.id_to_text = id_to_text
        self.bm25_top_k = bm25_top_k or settings.bm25_top_k
        self.vector_top_k = vector_top_k or settings.vector_top_k
        self.fusion_top_k = fusion_top_k or settings.hybrid_fusion_top_k
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight
        self.vector_weight = vector_weight if vector_weight is not None else settings.vector_weight

    def retrieve(self, query: str) -> list[tuple[str, float, str]]:
        """
        返回 [(id, fused_score, text), ...] 按融合分降序。
        """
        q = query.strip()
        if not q:
            return []
        bm_hits = self.bm25.search(q, top_k=self.bm25_top_k)
        qv = encode_query(q)
        vec_hits = self.vectors.search(qv, top_k=self.vector_top_k)
        fused = fuse_weighted(
            bm_hits,
            vec_hits,
            self.bm25_weight,
            self.vector_weight,
        )[: self.fusion_top_k]
        out: list[tuple[str, float, str]] = []
        for doc_id, score in fused:
            text = self.id_to_text.get(doc_id, "")
            out.append((doc_id, score, text))
        return out


def build_hybrid_from_pairs(
    id_text_pairs: Iterable[tuple[str, str]],
    dim: int,
) -> HybridRetriever:
    """从 (id, text) 构建 BM25 + 向量库 + 混合检索器。"""
    pairs = list(id_text_pairs)
    id_to_text = {i: t for i, t in pairs}

    bm25 = BM25LawIndex()
    bm25.build(pairs)

    from app.retrieval.embedding import encode_texts

    texts = [t for _, t in pairs]
    ids = [i for i, _ in pairs]
    mat = encode_texts(texts)
    if mat.shape[1] != dim:
        dim = mat.shape[1]
    store = FaissVectorStore(dim=dim)
    store.add(ids, mat)

    return HybridRetriever(bm25, store, id_to_text)
