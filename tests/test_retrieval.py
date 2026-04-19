"""检索模块单元测试。"""

from app.retrieval.bm25_index import BM25LawIndex, tokenize_zh_en
from app.retrieval.hybrid_retriever import fuse_weighted
from app.retrieval.rerank import maybe_rerank
from app.config import settings


def test_tokenize():
    assert "高空" in "".join(tokenize_zh_en("高空抛物侵权责任"))


def test_bm25_and_fuse():
    # 需有跨文档重复词，rank_bm25 的 ATIRE idf 在极小语料、词仅出现一次时 idf 可能为 0，导致分数全 0
    pairs = [
        ("a", "第一千二百五十四条 禁止从建筑物中抛掷物品 侵权责任"),
        ("b", "饲养动物损害责任 侵权责任"),
    ]
    bm25 = BM25LawIndex()
    bm25.build(pairs)
    hits = bm25.search("侵权责任", top_k=5)
    assert hits

    fused = fuse_weighted([("a", 1.0)], [("b", 0.5)], 0.5, 0.5)
    assert fused


def test_rerank_disabled_passthrough():
    prev = settings.rerank_enabled
    settings.rerank_enabled = False
    try:
        hits = [("a", 0.9, "foo"), ("b", 0.1, "bar")]
        assert maybe_rerank("q", hits) == hits
    finally:
        settings.rerank_enabled = prev
