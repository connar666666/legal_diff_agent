"""任务类型路由：根据用户输入粗分类，便于扩展多分支图。"""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["law", "case", "compare", "export", "web", "general"]


def classify_intent(user_text: str) -> Intent:
    """
    基于关键词的轻量意图分类（可替换为 LLM 分类节点）。
    """
    t = user_text.strip()
    if not t:
        return "general"
    tl = t.lower()

    # 含 URL 或明确要求联网检索：优先走通用网页搜索路径
    if re.search(r"https?://|www\.", tl):
        return "web"
    if re.search(r"搜索|检索|网上查|联网查|查一下.*网|浏览器", tl):
        return "web"

    if re.search(r"导出|保存|写入|markdown|\.md", tl):
        return "export"
    if re.search(r"对比|比较|差异|两地|跨省|a省|b省", tl):
        return "compare"
    if re.search(r"案例|判决|裁判文书|高院|最高法", tl):
        return "case"
    if re.search(
        r"法条|法规|条例|规定|第.+条|侵权|合同|物权|刑法|民法典|行政",
        tl,
    ):
        return "law"
    return "general"


def should_use_tools(intent: Intent) -> bool:
    """是否预期需要检索/导出工具。"""
    return intent in ("law", "case", "compare", "export", "web", "general")
