"""法规片段显式引用字符串（法规名 + 条号 + 款）。"""

from __future__ import annotations


def format_law_citation(
    law_title: str,
    article_label: str,
    sub_label: str = "",
) -> str:
    """
    生成作答用引用串，例如：《上海市住宅物业管理条例》第十二条（一）
    law_title / article_label / sub_label 来自索引 law_chunk_meta。
    """
    t = (law_title or "").strip()
    a = (article_label or "").strip()
    s = (sub_label or "").strip()
    if not t and not a and not s:
        return ""
    head = f"《{t}》" if t else ""
    body = a
    if s and s not in a:
        body = f"{a}{s}" if a else s
    elif not a and s:
        body = s
    return f"{head}{body}" if head or body else ""
