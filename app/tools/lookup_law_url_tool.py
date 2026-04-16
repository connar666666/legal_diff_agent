"""从 data/raw/law.txt（JSONL）反查某个法典/法律名称对应的已知一手 URL。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from app.config import settings


@tool
def lookup_law_url_tool(law_name: str, jurisdiction: str = "") -> str:
    """
    lookup：给定法典/法律名称，返回 data/raw/law.txt 中已知的 URL 列表。

    law_name: 例如“道路交通安全法”“道路交通安全条例”
    jurisdiction: 例如“深圳/广东/北京/全国”（为空则不做过滤）
    """

    reg_path: Path = settings.law_url_registry_path
    if not reg_path.exists():
        return json.dumps({"ok": False, "error": "law_url_registry 不存在，请先抓取并写入"}, ensure_ascii=False)

    matches: list[dict[str, str]] = []
    for line in reg_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        if obj.get("law_name") != law_name:
            continue
        if jurisdiction and obj.get("jurisdiction") != jurisdiction:
            continue

        matches.append(
            {
                "url": str(obj.get("url", "")),
                "local_path": str(obj.get("local_path", "")),
                "note": str(obj.get("note", "")),
                "ts": str(obj.get("ts", "")),
            }
        )

    if not matches:
        return json.dumps(
            {"ok": False, "error": "未找到已知 URL", "law_name": law_name, "jurisdiction": jurisdiction},
            ensure_ascii=False,
        )

    return json.dumps({"ok": True, "law_name": law_name, "jurisdiction": jurisdiction, "urls": matches}, ensure_ascii=False)

