"""根据 data/raw/laws/ 重新构建 data/index/ 的法规检索索引，并让 LawService 重新加载。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

from app.config import settings
from app.tools import registry


def _index_ready() -> bool:
    return (
        settings.resolve_law_bm25().exists()
        and settings.resolve_law_vector().exists()
        and (settings.data_index / "law_texts.json").exists()
    )


@tool
def build_law_index_tool(force: bool = False) -> str:
    """
    重建法规索引（BM25 + FAISS + id->text）。
    """

    if not force and _index_ready():
        return json.dumps({"ok": True, "skipped": True, "reason": "law index 已就绪"}, ensure_ascii=False)

    # 用脚本构建，复用现有 chunker/parser 逻辑
    repo_root = settings.project_root
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "build_law_index.py"),
        str(settings.data_raw_laws),
        "--index-dir",
        str(settings.data_index),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=str(repo_root))
    except Exception as e:
        return json.dumps({"ok": False, "error": f"build_law_index 执行失败: {e}", "cmd": cmd}, ensure_ascii=False)

    # 让当前运行时 LawService 重新加载磁盘索引
    svc = registry.get_law_service()
    if svc:
        svc.load_from_paths()

    return json.dumps(
        {"ok": True, "skipped": False, "index_dir": str(settings.data_index), "index_ready": _index_ready()},
        ensure_ascii=False,
    )

