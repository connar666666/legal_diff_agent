"""案例检索服务（与法规索引并行）。"""

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


def _case_texts_path(index_dir: Path) -> Path:
    return index_dir / "case_texts.json"


class CaseService:
    def __init__(self, retriever: Optional[HybridRetriever] = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> Optional[HybridRetriever]:
        """兼容工具层对 retriever 属性的访问。"""
        return self._retriever

    def load_from_paths(
        self,
        bm25_path: Optional[Path] = None,
        vector_dir: Optional[Path] = None,
        texts_path: Optional[Path] = None,
    ) -> bool:
        bp = bm25_path or settings.resolve_case_bm25()
        vd = vector_dir or settings.resolve_case_vector()
        tp = texts_path or _case_texts_path(settings.data_index)

        if not bp.exists() or not vd.exists() or not tp.exists():
            logger.warning("Case index incomplete; case search disabled.")
            self._retriever = None
            return False

        id_to_text = json.loads(tp.read_text(encoding="utf-8"))
        bm25 = BM25LawIndex()
        bm25.load(bp)
        store = FaissVectorStore(dim=1)
        store.load(vd)
        self._retriever = HybridRetriever(bm25, store, id_to_text)
        logger.info("CaseService loaded: %d snippets", len(id_to_text))
        return True

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        if not self._retriever:
            return []
        hits = self._retriever.retrieve(query)
        if top_k is not None:
            hits = hits[:top_k]
        return [
            {"id": i, "score": s, "snippet": t, "kind": "case"}
            for i, s, t in hits
        ]
