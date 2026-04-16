"""条文语义对齐与条号提取测试。"""

import numpy as np

from app.services.article_alignment import (
    extract_article_hint,
    greedy_pair_by_similarity,
    semantic_align_jurisdictions,
)


def test_extract_article_hint():
    assert extract_article_hint("第十二条 foo") == "第十二条"
    assert extract_article_hint("前言") == ""


def test_greedy_pair_prefers_high_sim():
    sim = np.array([[0.9, 0.2], [0.3, 0.85]])
    pairs = greedy_pair_by_similarity(sim, min_sim=0.25, max_pairs=10)
    assert len(pairs) == 2
    assert pairs[0][2] >= 0.85


def test_semantic_align_single_side():
    rows = semantic_align_jurisdictions(
        [],
        [{"id": "b:0", "text": "第二条 测试", "score": 0.5}],
        topic="测试",
        jurisdiction_a="上海",
        jurisdiction_b="深圳",
    )
    assert len(rows) == 1
    assert rows[0].alignment_method == "single_side"
