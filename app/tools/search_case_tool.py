"""检索司法案例片段。"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app.config import settings
from app.tools import registry


@tool
def search_case_tool(query: str) -> str:
    """
    根据描述检索相关裁判文书/案例摘要。
    query: 案由、关键词或事实描述。
    """
    svc = registry.get_case_service()
    if not svc or not svc.retriever:
        bm25_path = settings.resolve_case_bm25()
        faiss_dir = settings.resolve_case_vector()
        texts_path = settings.data_index / "case_texts.json"

        expected = {
            "case_bm25": str(bm25_path),
            "case_faiss_dir": str(faiss_dir),
            "case_texts": str(texts_path),
        }
        missing = {
            "case_bm25": not bm25_path.exists(),
            "case_faiss_dir": not faiss_dir.exists(),
            "case_texts": not texts_path.exists(),
        }
        return json.dumps(
            {
                "ok": False,
                "error": "案例索引未加载：当前缺少案例检索索引（BM25/FAISS/id->text）。",
                "missing": missing,
                "expected": expected,
                "next_steps": {
                    "prepare_raw": f"准备裁判文书页面或已抽取的案例片段。项目当前用于构建索引的输入是 JSONL：每行 {{\"id\":\"...\",\"text\":\"...\"}}。建议输出到：{settings.data_processed_cases / 'case_snippets.jsonl'}",
                    "build_index_cmd": (
                        f"python scripts/build_case_index.py {settings.data_processed_cases / 'case_snippets.jsonl'}"
                    ),
                    "optional_import_cmd": (
                        f"如果你有公开页面 URL 列表：准备 {settings.data_raw_cases / 'urls.txt'} "
                        f"（每行一个 URL），再运行："
                        f"python scripts/import_cases_from_urls.py --urls-file {settings.data_raw_cases / 'urls.txt'} --out-dir {settings.data_raw_cases}"
                    ),
                },
            },
            ensure_ascii=False,
        )
    rows = svc.search(query, top_k=10)
    return json.dumps({"ok": True, "hits": rows}, ensure_ascii=False)
