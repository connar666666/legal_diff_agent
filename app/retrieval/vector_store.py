"""FAISS 向量库封装：存 id 列表与 L2 归一化后的内积检索。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)


class FaissVectorStore:
    """id 与向量一一对应；使用 IndexFlatIP + 归一化向量等价于 cosine。"""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._index: Optional[faiss.IndexFlatIP] = None

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N, {self.dim}), got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        vectors = _l2_normalize(vectors.astype(np.float32))
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dim)
        start = len(self._ids)
        self._ids.extend(ids)
        self._index.add(vectors)
        logger.debug("Added vectors %s-%s", start, len(self._ids) - 1)

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        if self._index is None or self.size == 0:
            return []
        q = query_vec.astype(np.float32).reshape(1, -1)
        q = _l2_normalize(q)
        scores, indices = self._index.search(q, min(top_k, self.size))
        out: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            out.append((self._ids[idx], float(score)))
        return out

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        meta = {"ids": self._ids, "dim": self.dim}
        (dir_path / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        if self._index is not None:
            faiss.write_index(self._index, str(dir_path / "index.faiss"))
        logger.info("FAISS store saved: %s (n=%d)", dir_path, len(self._ids))

    def load(self, dir_path: Path) -> None:
        meta_path = dir_path / "meta.json"
        idx_path = dir_path / "index.faiss"
        if not meta_path.exists() or not idx_path.exists():
            raise FileNotFoundError(f"Missing FAISS files under {dir_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._ids = meta["ids"]
        self.dim = int(meta["dim"])
        self._index = faiss.read_index(str(idx_path))
        logger.info("FAISS store loaded: %s (n=%d)", dir_path, len(self._ids))
