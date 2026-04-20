"""从回答中抽取法条引用并与金标准对齐（启发式）。"""

from __future__ import annotations

import re
from typing import Any


# 《…》+ 第…条
_CITE_BOOK = re.compile(
    r"《[^》]{1,120}》\s*第[零一二三四五六七八九十百千万\d]+条",
)
# 单独「第…条」
_CITE_ART = re.compile(r"第[零一二三四五六七八九十百千万\d]+条")


def normalize_cite_text(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"\s+", "", t)
    return t


def extract_law_citations(answer: str) -> list[str]:
    """从模型回答中抽取引用串（去重保序）。"""
    text = answer or ""
    found: list[str] = []
    seen: set[str] = set()
    for m in _CITE_BOOK.finditer(text):
        c = m.group(0).strip()
        n = normalize_cite_text(c)
        if n and n not in seen:
            seen.add(n)
            found.append(c)
    for m in _CITE_ART.finditer(text):
        c = m.group(0).strip()
        n = normalize_cite_text(c)
        if n and n not in seen:
            seen.add(n)
            found.append(c)
    return found


def _match_one_extracted_to_gold(extracted_norm: str, gold_norm: str) -> bool:
    if not gold_norm:
        return False
    if gold_norm in extracted_norm or extracted_norm in gold_norm:
        return True
    # 条号对齐：金标准可能只写「第十二条」
    if "第" in gold_norm and "条" in gold_norm and gold_norm in extracted_norm:
        return True
    return False


def gold_citation_scores(extracted: list[str], gold_citations: list[str]) -> dict[str, Any]:
    """
    粗粒度 precision / recall / F1：
    - 每个 gold 是否被任一 extracted 覆盖 → recall
    - 每个 extracted 是否命中任一 gold → precision
    """
    golds = [normalize_cite_text(g) for g in gold_citations if g.strip()]
    exts = [normalize_cite_text(e) for e in extracted if e.strip()]
    if not golds and not exts:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp_gold": 0, "tp_ext": 0, "n_gold": 0, "n_ext": 0}
    if not golds:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "tp_gold": 0, "tp_ext": 0, "n_gold": 0, "n_ext": len(exts)}
    if not exts:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "tp_gold": 0, "tp_ext": 0, "n_gold": len(golds), "n_ext": 0}

    gold_hit = [any(_match_one_extracted_to_gold(e, g) for e in exts) for g in golds]
    ext_hit = [any(_match_one_extracted_to_gold(e, g) for g in golds) for e in exts]

    tp_g = sum(1 for x in gold_hit if x)
    tp_e = sum(1 for x in ext_hit if x)
    prec = tp_e / len(exts) if exts else 0.0
    rec = tp_g / len(golds) if golds else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp_gold": tp_g,
        "tp_ext": tp_e,
        "n_gold": len(golds),
        "n_ext": len(exts),
    }


def search_query_relevance(question: str, search_query: str, gold_keywords: list[str]) -> dict[str, Any]:
    """
    粗测检索词是否偏题：字符级重合 + 可选 gold_keywords 命中率。
    返回 overlap ∈ [0,1]（越高越相关，仅启发式）。
    """
    q = normalize_cite_text(question)
    s = normalize_cite_text(search_query)
    if not q or not s:
        return {"overlap": 0.0, "keyword_hit_rate": None}
    qs = set(q)
    ss = set(s)
    inter = len(qs & ss)
    union = len(qs | ss) or 1
    overlap = inter / union
    kw_rate = None
    if gold_keywords:
        hits = sum(1 for k in gold_keywords if normalize_cite_text(k) in s)
        kw_rate = hits / len(gold_keywords)
    return {"overlap": round(overlap, 4), "keyword_hit_rate": kw_rate}


def citation_supported_by_evidence(citations: list[str], evidence: str) -> dict[str, Any]:
    """每条抽取引用是否在 evidence 文本中有归一化子串支撑。"""
    ev = normalize_cite_text(evidence or "")
    supported = []
    for c in citations:
        n = normalize_cite_text(c)
        supported.append(bool(n and n in ev))
    rate = sum(1 for x in supported if x) / len(supported) if supported else 1.0
    return {"supported_rate": round(rate, 4), "per_citation_supported": supported}
