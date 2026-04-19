"""Cross-encoder 重排序：在 BM25 + 向量融合候选上做精排。"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_ce_model: Optional[Any] = None
_ce_init_attempted: bool = False


def _get_cross_encoder() -> Optional[Any]:
    global _ce_model, _ce_init_attempted
    if _ce_init_attempted:
        return _ce_model
    _ce_init_attempted = True
    if not settings.rerank_enabled:
        return None
    try:
        from sentence_transformers import CrossEncoder

        _ce_model = CrossEncoder(
            settings.rerank_model_name,
            max_length=settings.rerank_max_length,
        )
        logger.info("Loaded CrossEncoder reranker: %s", settings.rerank_model_name)
    except Exception as e:
        logger.warning(
            "CrossEncoder reranker unavailable (%s); continuing without rerank.",
            e,
        )
        _ce_model = None
    return _ce_model


def reset_cross_encoder_cache() -> None:
    """测试用：清除懒加载的 reranker 实例。"""
    global _ce_model, _ce_init_attempted
    _ce_model = None
    _ce_init_attempted = False


def cross_encoder_rerank(
    query: str,
    hits: list[tuple[str, float, str]],
    *,
    batch_size: Optional[int] = None,
) -> list[tuple[str, float, str]]:
    """
    对候选 (id, fused_score, text) 用 CrossEncoder 打分并重排。
    返回的 score 为 CE 相关性分数（越高越相关）。
    """
    if not hits:
        return []
    model = _get_cross_encoder()
    if model is None:
        return list(hits)

    lim = max(1, settings.rerank_max_passage_chars)
    pairs = [[query, (text or "")[:lim]] for _, _, text in hits]
    bs = batch_size if batch_size is not None else settings.rerank_batch_size
    raw = model.predict(
        pairs,
        batch_size=bs,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    scores = np.asarray(raw, dtype=np.float64).reshape(-1)

    ranked: list[tuple[str, float, str]] = [
        (doc_id, float(ce), text) for (doc_id, _, text), ce in zip(hits, scores)
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def maybe_rerank(
    query: str,
    hits: list[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    """配置开启且模型可用时精排，否则保持融合顺序与分数。"""
    if not settings.rerank_enabled or not hits:
        return list(hits)
    return cross_encoder_rerank(query, hits)


def identity_rerank(
    query: str,
    hits: list[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    """兼容旧接口：不做重排。"""
    return list(hits)
