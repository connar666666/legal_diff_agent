"""按主题、法域检索法规片段。"""

from __future__ import annotations

import json
from langchain_core.tools import tool

from app.config import settings
from app.tools import registry


@tool
def search_law_tool(query: str, jurisdiction: str = "") -> str:
    """
    根据用户问题检索相关法规条文片段。
    query: 自然语言问题或关键词（可含条号、法律术语）。
    jurisdiction: 可选法域过滤，如「北京」「全国」；未知则留空。
    返回带条号/得分的片段摘要（JSON 字符串）。
    """
    svc = registry.get_law_service()
    if not svc or not svc.retriever:
        bm25_path = settings.resolve_law_bm25()
        faiss_dir = settings.resolve_law_vector()
        texts_path = settings.data_index / "law_texts.json"

        expected = {
            "law_bm25": str(bm25_path),
            "law_faiss_dir": str(faiss_dir),
            "law_texts": str(texts_path),
        }
        missing = {
            "law_bm25": not bm25_path.exists(),
            "law_faiss_dir": not faiss_dir.exists(),
            "law_texts": not texts_path.exists(),
        }
        return json.dumps(
            {
                "ok": False,
                "error": "法规索引未加载：当前缺少法规检索索引（BM25/FAISS/id->text）。",
                "missing": missing,
                "expected": expected,
                "next_steps": {
                    "prepare_raw": f"把法规原文（`.txt/.html/.htm`）放入：{settings.data_raw_laws}",
                    "build_index_cmd": f"python scripts/build_law_index.py {settings.data_raw_laws}",
                    "optional_import_cmd": (
                        "如果你只有公开页面 URL 列表，可先准备："
                        f"{settings.data_raw_laws / 'urls.txt'}（每行一个 URL）\n"
                        f"再运行：python scripts/import_laws_from_urls.py --urls-file {settings.data_raw_laws / 'urls.txt'} --out-dir {settings.data_raw_laws}"
                    ),
                },
            },
            ensure_ascii=False,
        )
    rows = svc.search(query, jurisdiction=jurisdiction or None, top_k=10)
    return json.dumps({"ok": True, "hits": rows}, ensure_ascii=False)
