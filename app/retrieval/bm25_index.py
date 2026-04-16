"""BM25 关键词索引（适合法条号、专有名词命中）。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize_zh_en(text: str) -> list[str]:
    """中英混合简单分词：连续汉字、连续字母数字各为 token。"""
    if not text:
        return []
    parts = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
    return parts


class BM25LawIndex:
    """可持久化的 BM25 索引。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    @property
    def size(self) -> int:
        return len(self._ids)

    def build(self, id_text_pairs: Iterable[tuple[str, str]]) -> None:
        self._ids = []
        self._tokenized = []
        for doc_id, text in id_text_pairs:
            self._ids.append(doc_id)
            self._tokenized.append(tokenize_zh_en(text))
        if not self._tokenized:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if not self._bm25 or not query.strip():
            return []
        q = tokenize_zh_en(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        out: list[tuple[str, float]] = []
        for i, s in ranked:
            # ATIRE BM25 可能为负分，仍表示相对排序，勿按 <=0 丢弃
            out.append((self._ids[i], float(s)))
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ids": self._ids, "tokenized": self._tokenized}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info("BM25 index saved: %s (%d docs)", path, len(self._ids))

    def load(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._ids = raw["ids"]
        self._tokenized = raw["tokenized"]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        logger.info("BM25 index loaded: %s (%d docs)", path, len(self._ids))
