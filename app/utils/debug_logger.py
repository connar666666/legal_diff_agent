"""调试日志：记录工具调用链路（模型输出、解析结果、工具输入输出）。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class JSONLWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", int(time.time() * 1000))
        with self.path.open("a", encoding="utf-8") as f:
            # 兜底：UUID / Path 等不可 JSON 序列化对象转成字符串
            safe = {}
            for k, v in record.items():
                try:
                    json.dumps(v)
                    safe[k] = v
                except Exception:
                    safe[k] = str(v)
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")


class ToolCallDebugCallbackHandler(BaseCallbackHandler):
    """LangChain 回调：记录 LLM / Tool 的调用输入输出。"""

    def __init__(self, writer: JSONLWriter, *, run_scope: str = "run") -> None:
        self.writer = writer
        self.run_scope = run_scope

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:  # type: ignore[override]
        run_id = kwargs.get("run_id")
        self.writer.write(
            {
                "event": "llm_start",
                "scope": self.run_scope,
                "run_id": str(run_id) if run_id is not None else None,
                "serialized": serialized,
                "prompt": prompts[0] if prompts else "",
            }
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
        run_id = kwargs.get("run_id")
        # response 可能较大，避免直接全量写入；优先写文本片段
        content = ""
        try:
            content = getattr(response.generations[0], "text", "") or ""
        except Exception:
            content = str(response)[:2000]
        self.writer.write(
            {
                "event": "llm_end",
                "scope": self.run_scope,
                "run_id": str(run_id) if run_id is not None else None,
                "preview": content[:2000],
            }
        )

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:  # type: ignore[override]
        run_id = kwargs.get("run_id")
        tool_name = serialized.get("name") if isinstance(serialized, dict) else None
        self.writer.write(
            {
                "event": "tool_start",
                "scope": self.run_scope,
                "run_id": str(run_id) if run_id is not None else None,
                "tool_name": tool_name,
                "input": input_str,
            }
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:  # type: ignore[override]
        run_id = kwargs.get("run_id")
        self.writer.write(
            {
                "event": "tool_end",
                "scope": self.run_scope,
                "run_id": str(run_id) if run_id is not None else None,
                "output_preview": str(output)[:4000],
            }
        )

