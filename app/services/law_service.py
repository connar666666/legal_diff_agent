"""法规检索服务：基于混合索引加载与查询。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.retrieval.bm25_index import BM25LawIndex
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


def _default_texts_path(index_dir: Path) -> Path:
    return index_dir / "law_texts.json"


class LawService:
    """封装法规域混合检索。"""

    def __init__(self, retriever: Optional[HybridRetriever] = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> Optional[HybridRetriever]:
        return self._retriever

    def load_from_paths(
        self,
        bm25_path: Optional[Path] = None,
        vector_dir: Optional[Path] = None,
        texts_path: Optional[Path] = None,
    ) -> bool:
        """
        从磁盘加载 BM25 + FAISS + id->text 映射。
        texts 文件为 JSON 对象：{ "chunk_id": "全文..." }。
        """
        bp = bm25_path or settings.resolve_law_bm25()
        vd = vector_dir or settings.resolve_law_vector()
        tp = texts_path or _default_texts_path(settings.data_index)

        if not bp.exists() or not vd.exists() or not tp.exists():
            logger.warning(
                "Law index incomplete (bm25=%s, vec=%s, texts=%s)",
                bp.exists(),
                vd.exists(),
                tp.exists(),
            )
            self._retriever = None
            return False

        id_to_text: dict[str, str] = json.loads(tp.read_text(encoding="utf-8"))
        bm25 = BM25LawIndex()
        bm25.load(bp)
        store = FaissVectorStore(dim=1)  # placeholder, load overwrites
        store.load(vd)

        self._retriever = HybridRetriever(bm25, store, id_to_text)
        logger.info("LawService loaded: %d chunks", len(id_to_text))
        return True

    def search(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        检索法规片段。可选按法域过滤（在 meta 未入索引时仅关键词过滤标题占位）。
        """
        if not self._retriever:
            return []
        hits = self._retriever.retrieve(query)
        if top_k is not None:
            hits = hits[:top_k]
        out: list[dict[str, Any]] = []
        for doc_id, score, text in hits:
            if jurisdiction and jurisdiction not in text:
                continue
            out.append(
                {
                    "id": doc_id,
                    "score": score,
                    "text": text,
                    "jurisdiction_filter": jurisdiction or "",
                }
            )
        return out
