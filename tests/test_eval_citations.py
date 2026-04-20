"""评测指标单元测试（不触网、不加载大模型）。"""

from app.eval.citations import (
    extract_law_citations,
    gold_citation_scores,
    search_query_relevance,
)
from app.eval.dataset import load_eval_jsonl


def test_extract_citations():
    t = "依据《民法典》第一千二百五十四条，禁止高空抛物。"
    xs = extract_law_citations(t)
    assert any("第一千二百五十四条" in x for x in xs)


def test_gold_f1_perfect():
    # 抽取与金标准条号对齐（法典全称略称不一致时不强求一条 extracted 覆盖两个别名）
    g = gold_citation_scores(
        ["《民法典》第一千二百五十四条"],
        ["第一千二百五十四条"],
    )
    assert g["f1"] == 1.0


def test_search_overlap():
    r = search_query_relevance("上海物业管理收益归属", "上海 物业管理 收益", ["上海", "物业"])
    assert r["overlap"] > 0
    assert r["keyword_hit_rate"] == 1.0


def test_load_sample_jsonl():
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "data" / "eval" / "examples" / "sample_questions.jsonl"
    items = load_eval_jsonl(p)
    assert len(items) >= 1
    assert items[0].question
