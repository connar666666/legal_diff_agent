"""文本嵌入：优先 sentence-transformers，失败时回退到确定性哈希向量（仅开发/测试）。"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_embedder: Optional[object] = None
_embed_dim: int = 384


def _hash_embedding(text: str, dim: int = _embed_dim) -> np.ndarray:
    """确定性伪嵌入：无模型时用于跑通流水线。"""
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)


def get_embedder():
    """懒加载 SentenceTransformer。"""
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(settings.embedding_model_name)
        logger.info("Loaded embedding model: %s", settings.embedding_model_name)
    except Exception as e:
        logger.warning(
            "sentence-transformers unavailable (%s); using hash embeddings.", e
        )
        _embedder = None
    return _embedder


def embedding_dim() -> int:
    m = get_embedder()
    if m is not None:
        return int(m.get_sentence_embedding_dimension())
    return _embed_dim


def encode_texts(texts: Iterable[str], batch_size: Optional[int] = None) -> np.ndarray:
    """将一批文本编码为 (N, D) float32。"""
    texts = list(texts)
    if not texts:
        return np.zeros((0, embedding_dim()), dtype=np.float32)
    m = get_embedder()
    bs = batch_size or settings.embedding_batch_size
    if m is not None:
        arr = m.encode(
            texts,
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(arr, dtype=np.float32)
    dim = embedding_dim()
    out = np.stack([_hash_embedding(t, dim) for t in texts])
    return out


def encode_query(query: str) -> np.ndarray:
    """单条查询向量。"""
    return encode_texts([query])[0]
