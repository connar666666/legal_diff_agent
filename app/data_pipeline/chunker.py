"""法规文本智能切分（按条/款/项），供流水线或独立调用。"""

from __future__ import annotations

import re
from typing import Iterable

from app.schema.models import LawChunkRecord, LawDocumentMeta
from app.utils.text_utils import normalize_whitespace

# 与 parser.law_text_to_chunks 共享的条号模式
_ARTICLE = re.compile(r"(第[零一二三四五六七八九十百千万\d]+条)")
_KUAN = re.compile(r"(（[一二三四五六七八九十]+）|（\d+）)")


def split_by_articles(text: str) -> list[tuple[str, str]]:
    """按「第…条」切分，返回 (条号, 正文)。"""
    text = normalize_whitespace(text)
    matches = list(_ARTICLE.finditer(text))
    if not matches:
        return [("", text)]
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        label = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((label, text[start:end].strip()))
    return out


def split_article_by_kuan(
    article_label: str,
    body: str,
    max_chars: int = 800,
) -> list[dict[str, str]]:
    """
    单条过长时按「款」切分。
    返回 [{"article_label", "sub_label", "text"}, ...]
    """
    body = body.strip()
    if len(body) <= max_chars:
        return [{"article_label": article_label, "sub_label": "", "text": body}]

    subs = list(_KUAN.finditer(body))
    if len(subs) <= 1:
        return [{"article_label": article_label, "sub_label": "", "text": body}]

    pieces: list[dict[str, str]] = []
    for j, sm in enumerate(subs):
        s_start = sm.start()
        s_end = subs[j + 1].start() if j + 1 < len(subs) else len(body)
        piece = body[s_start:s_end].strip()
        if piece:
            pieces.append(
                {
                    "article_label": article_label,
                    "sub_label": sm.group(1),
                    "text": piece,
                }
            )
    return pieces or [{"article_label": article_label, "sub_label": "", "text": body}]


def chunk_law_text(
    meta: LawDocumentMeta,
    full_text: str,
    doc_id: str,
    max_article_chars: int = 800,
) -> list[LawChunkRecord]:
    """
    对整篇法规做智能切分：优先按条，其次按款。
    """
    chunks: list[LawChunkRecord] = []
    article_parts = split_by_articles(full_text)
    idx = 0
    for article_label, body in article_parts:
        if not body:
            continue
        for part in split_article_by_kuan(article_label or "全文", body, max_article_chars):
            cid = f"{doc_id}:{idx}"
            idx += 1
            extra = {}
            if part.get("sub_label"):
                extra["sub_label"] = part["sub_label"]
            chunks.append(
                LawChunkRecord(
                    id=cid,
                    doc_id=doc_id,
                    article_label=part["article_label"],
                    text=part["text"],
                    meta=meta,
                    extra=extra,
                )
            )
    if not chunks:
        chunks.append(
            LawChunkRecord(
                id=f"{doc_id}:0",
                doc_id=doc_id,
                article_label="全文",
                text=normalize_whitespace(full_text),
                meta=meta,
            )
        )
    return chunks


def iter_chunks_for_index(chunks: Iterable[LawChunkRecord]) -> Iterable[tuple[str, str]]:
    """为 BM25/向量索引产出 (id, text)。"""
    for c in chunks:
        yield c.id, c.text
