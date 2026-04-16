"""将助手总结导出为 Markdown 文件。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from app.config import settings


@tool
def export_tool(content: str, filename: str = "export.md") -> str:
    """
    将最终回答或报告写入 data/outputs 下的文件。
    content: Markdown 或纯文本正文。
    filename: 文件名，应使用 .md 或 .txt 后缀。
    """
    safe = Path(filename).name
    if not safe or ".." in safe:
        return json.dumps({"ok": False, "error": "非法文件名"}, ensure_ascii=False)
    out_dir = settings.data_outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / safe
    header = f"<!-- generated {datetime.utcnow().isoformat()}Z -->\n\n"
    path.write_text(header + content, encoding="utf-8")
    return json.dumps(
        {"ok": True, "path": str(path), "format": "markdown"},
        ensure_ascii=False,
    )


def all_tools():
    """供构图绑定的工具列表。"""
    from app.tools.search_law_tool import search_law_tool
    from app.tools.search_case_tool import search_case_tool
    from app.tools.compare_tool import compare_tool
    from app.tools.export_tool import export_tool
    from app.tools.web_discover_law_urls_tool import discover_law_urls_tool
    from app.tools.fetch_law_url_and_store_tool import fetch_law_primary_source_tool
    from app.tools.lookup_law_url_tool import lookup_law_url_tool
    from app.tools.build_law_index_tool import build_law_index_tool
    from app.tools.auto_import_law_primary_source_tool import auto_import_law_primary_source_tool
    from app.tools.web_search_tool import web_search_tool

    return [
        web_search_tool,
        search_law_tool,
        search_case_tool,
        compare_tool,
        lookup_law_url_tool,
        discover_law_urls_tool,
        fetch_law_primary_source_tool,
        build_law_index_tool,
        auto_import_law_primary_source_tool,
        export_tool,
    ]
