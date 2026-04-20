#!/usr/bin/env python3
"""
评测实验：Baseline 2（网页摘要 RAG）vs 完整 Agent。

用法示例::

    export PYTHONPATH=.
    # 仅跑 baseline（无需本地法规索引）
    python scripts/run_eval_experiment.py run --dataset data/eval/examples/sample_questions.jsonl \\
        --systems baseline_web_rag --out data/outputs/eval/run_baseline.jsonl --limit 5

    # 跑两边（需已构建法规索引以便 Agent 检索）
    python scripts/run_eval_experiment.py run --dataset data/eval/examples/sample_questions.jsonl \\
        --systems baseline_web_rag,full_agent --out data/outputs/eval/run_compare.jsonl

    python scripts/run_eval_experiment.py aggregate --results data/outputs/eval/run_compare.jsonl

多基座：更换 `.env` 或环境中的模型路径后分别跑 `--out`，再对比各 out 的 aggregate 结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.eval.runner import aggregate_results, run_eval_batch


def _parse_systems(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    allowed = {"baseline_web_rag", "full_agent"}
    for p in parts:
        if p not in allowed:
            raise SystemExit(f"unknown system: {p}, allowed: {allowed}")
    return parts  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser(description="Legal eval: baseline web RAG vs full agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="跑评测并写 JSONL")
    p_run.add_argument("--dataset", type=Path, required=True, help="JSONL 评测集路径")
    p_run.add_argument("--out", type=Path, required=True, help="输出 JSONL")
    p_run.add_argument(
        "--systems",
        type=str,
        default="baseline_web_rag,full_agent",
        help="逗号分隔: baseline_web_rag,full_agent",
    )
    p_run.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试用）")
    p_run.add_argument("--max-search-results", type=int, default=8)
    p_run.add_argument("--debug-tools", action="store_true")

    p_agg = sub.add_parser("aggregate", help="汇总 JSONL 指标")
    p_agg.add_argument("--results", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "run":
        systems = _parse_systems(args.systems)
        run_eval_batch(
            args.dataset,
            args.out,
            systems,  # type: ignore[arg-type]
            limit=args.limit,
            max_search_results=args.max_search_results,
            debug_tools=args.debug_tools,
        )
        print(f"Wrote: {args.out}")
    elif args.cmd == "aggregate":
        s = aggregate_results(args.results)
        print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
