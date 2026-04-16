"""多地法规对比。"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app.services import compare_service
from app.config import settings
from app.tools import registry


@tool
def compare_tool(topic: str, jurisdiction_a: str, jurisdiction_b: str) -> str:
    """
    对两个法域在某一主题下的法规片段做并列对比（依赖已构建的本地法规索引）。

    对齐方式：在两地各自检索到的候选片段上，使用与建索引相同的句向量模型计算
    余弦相似度，并按阈值做贪心最优配对（条文片段级语义对齐）；无法配对时回退为检索排名并列。

    topic: 主题，如「高空抛物」「物业管理」。
    jurisdiction_a / jurisdiction_b: 两个法域名称或简称。
    若某一侧本地无材料或检索为空，应先调用 discover_law_urls_tool / auto_import_law_primary_source_tool
    按「法规名称 + 该法域」在线发现 URL 并拉取、重建索引后再对比；无需用户必须粘贴链接。
    """
    law = registry.get_law_service()
    if not law or not law.retriever:
        return json.dumps(
            {
                "ok": False,
                "error": "法规索引未加载。",
                "expected": {
                    "law_bm25": str(settings.resolve_law_bm25()),
                    "law_faiss_dir": str(settings.resolve_law_vector()),
                    "law_texts": str(settings.data_index / "law_texts.json"),
                },
            },
            ensure_ascii=False,
        )
    rows = compare_service.compare_jurisdictions(
        law, topic, jurisdiction_a, jurisdiction_b
    )
    return json.dumps({"ok": True, "rows": rows}, ensure_ascii=False)
