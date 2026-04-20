"""批量跑 Baseline 2 与完整 Agent，并写 JSONL 结果。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage

from app.eval.baseline_web_rag import run_baseline_web_rag
from app.eval.citations import (
    citation_supported_by_evidence,
    extract_law_citations,
    gold_citation_scores,
    search_query_relevance,
)
from app.eval.dataset import EvalItem, load_eval_jsonl
from app.graph.builder import build_agent_graph, invoke_chat
from app.main import bootstrap_services

logger = logging.getLogger(__name__)

SystemName = Literal["baseline_web_rag", "full_agent"]


def _last_assistant_text(messages: list[Any]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage):
            c = getattr(m, "content", "") or ""
            if isinstance(c, str) and c.strip():
                return c.strip()
    return ""


def _agent_evidence_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for m in messages or []:
        if isinstance(m, ToolMessage):
            parts.append(str(getattr(m, "content", "") or ""))
    return "\n".join(parts)[:20000]


def run_one_full_agent(item: EvalItem, graph: Any, *, debug_tools: bool = False) -> dict[str, Any]:
    out = invoke_chat(
        graph,
        item.question,
        thread_id=f"eval-{item.id}",
        debug_tools=debug_tools,
    )
    msgs = out.get("messages") or []
    answer = _last_assistant_text(msgs)
    evidence = _agent_evidence_text(msgs)
    return {
        "system": "full_agent",
        "item_id": item.id,
        "task_type": item.task_type,
        "question": item.question,
        "answer": answer,
        "gold_citations": item.gold_citations,
        "gold_keywords": item.gold_keywords,
        "evidence_excerpt": evidence[:8000],
        "raw_message_count": len(msgs),
    }


def run_one_baseline(item: EvalItem, *, max_results: int = 8) -> dict[str, Any]:
    base = run_baseline_web_rag(item.question, max_results=max_results)
    return {
        "system": "baseline_web_rag",
        "item_id": item.id,
        "task_type": item.task_type,
        "question": item.question,
        "answer": base["answer"],
        "gold_citations": item.gold_citations,
        "gold_keywords": item.gold_keywords,
        "search_queries": base["search_queries"],
        "search_payloads": base["search_payloads"],
        "evidence_excerpt": base["evidence_excerpt"],
    }


def attach_metrics(record: dict[str, Any]) -> dict[str, Any]:
    ans = record.get("answer") or ""
    gold = list(record.get("gold_citations") or [])
    ev = record.get("evidence_excerpt") or ""
    cites = extract_law_citations(ans)
    record["citations_extracted"] = cites
    record["gold_metrics"] = gold_citation_scores(cites, gold)
    record["evidence_support"] = citation_supported_by_evidence(cites, ev)
    if record.get("system") == "baseline_web_rag":
        qs = record.get("search_queries") or []
        if qs:
            rel = search_query_relevance(record.get("question") or "", qs[0], record.get("gold_keywords") or [])
            record["search_query_relevance"] = rel
    return record


def run_eval_batch(
    dataset_path: Path,
    output_path: Path,
    systems: list[SystemName],
    *,
    limit: int | None = None,
    max_search_results: int = 8,
    debug_tools: bool = False,
) -> None:
    """
    对 JSONL 评测集逐条运行并写入 output_path（每行一个完整 JSON）。
    """
    items = load_eval_jsonl(Path(dataset_path))
    if limit is not None:
        items = items[: max(0, limit)]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = None
    if "full_agent" in systems:
        bootstrap_services()
        graph = build_agent_graph()

    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            for sys in systems:
                try:
                    if sys == "full_agent":
                        assert graph is not None
                        rec = run_one_full_agent(item, graph, debug_tools=debug_tools)
                    else:
                        rec = run_one_baseline(item, max_results=max_search_results)
                    rec = attach_metrics(rec)
                    rec["ok"] = True
                    rec["error"] = None
                except Exception as e:
                    logger.exception("eval failed item=%s system=%s", item.id, sys)
                    rec = {
                        "ok": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "system": sys,
                        "item_id": item.id,
                        "question": item.question,
                    }
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def aggregate_results(results_path: Path) -> dict[str, Any]:
    """读取 run_eval_batch 产出的 JSONL，按 system 汇总均值。"""
    path = Path(results_path)
    by_sys: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not row.get("ok"):
            continue
        s = row.get("system") or "unknown"
        by_sys.setdefault(s, []).append(row)

    summary: dict[str, Any] = {}
    for s, rows in by_sys.items():
        n = len(rows)
        if n == 0:
            continue
        f1s = [float(r["gold_metrics"]["f1"]) for r in rows if r.get("gold_metrics")]
        sup = [float(r["evidence_support"]["supported_rate"]) for r in rows if r.get("evidence_support")]
        overlaps = [
            float(r["search_query_relevance"]["overlap"])
            for r in rows
            if r.get("search_query_relevance") is not None
        ]
        summary[s] = {
            "n": n,
            "gold_f1_mean": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "evidence_supported_rate_mean": round(sum(sup) / len(sup), 4) if sup else None,
            "search_query_overlap_mean": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
        }
    return summary
