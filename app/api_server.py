"""HTTP 层：把现有 LangGraph 智能体暴露给 Demo UI（自然语言 → 工具链 → 回答）。

与「粘贴两段全文 + /api/compare」不同：本端点走与 CLI (`python -m app.main`) **相同**
的 `invoke_chat` 路径，因而会按 `app/graph/prompts.py` + `SKILLS.md` 调用
lookup / discover / fetch / build_law_index / compare_tool 等，无需另写一套业务逻辑。

运行示例::

    export PYTHONPATH=.
    uvicorn app.api_server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pydantic import BaseModel, Field

from app.config import settings
from app.graph.builder import build_agent_graph, invoke_chat
from app.main import bootstrap_services

_graph: Any = None


@asynccontextmanager
async def lifespan(app: Any):
    global _graph
    bootstrap_services()
    _graph = build_agent_graph()
    yield


try:
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="legal_diff_agent", lifespan=lifespan)
except ImportError as e:  # pragma: no cover
    raise ImportError("请安装: pip install fastapi uvicorn") from e


def _strip_thinking_block(text: str) -> str:
    return re.sub(
        r"<redacted_thinking>.*?</think>", "", text or "", flags=re.DOTALL
    ).strip()


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """供前端展示「索引/检索/下载/对比」步骤：从消息序列提取 tool 与正文线索。"""
    out: list[dict[str, Any]] = []
    for m in messages:
        name = type(m).__name__
        row: dict[str, Any] = {"kind": name}
        content = getattr(m, "content", None)
        if content is not None:
            row["content"] = content if isinstance(content, str) else str(content)[:8000]
        tcalls = getattr(m, "tool_calls", None)
        if tcalls:
            row["tool_calls"] = [
                {
                    "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""),
                    "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}),
                }
                for tc in tcalls
            ]
        tool_name = getattr(m, "name", None)
        if tool_name:
            row["tool_name"] = tool_name
        out.append(row)
    return out


def _last_assistant_text(messages: list[Any]) -> str:
    from langchain_core.messages import AIMessage

    for m in reversed(messages or []):
        if isinstance(m, AIMessage):
            c = getattr(m, "content", "") or ""
            if isinstance(c, str) and c.strip():
                return _strip_thinking_block(c)
    return ""


class AgentQueryBody(BaseModel):
    question: str = Field(..., description="自然语言问题，例如两地法规对比、补索引等")
    thread_id: str = Field("demo-ui", description="多轮对话线程 id，与 CLI --thread-id 一致")
    tool_debug: bool = Field(
        False,
        description="为 true 时写入 data/outputs/tool_call_debug.jsonl 并打开回调（略慢）",
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "legal_diff_agent", "data_raw_laws": str(settings.data_raw_laws)}


@app.post("/api/agent/query")
def agent_query(body: AgentQueryBody) -> dict[str, Any]:
    """
    用自然语言驱动**与 CLI 相同**的智能体与工具链（本地 raw + 建索引 + 缺失则联网拉取 + 对比）。
    """
    if not (body.question or "").strip():
        raise HTTPException(status_code=400, detail="question 为空")

    if _graph is None:
        raise HTTPException(status_code=503, detail="图未初始化")

    prev = settings.tool_debug_enabled
    if body.tool_debug:
        settings.tool_debug_enabled = True

    try:
        out = invoke_chat(
            _graph,
            body.question.strip(),
            thread_id=body.thread_id,
            debug_tools=body.tool_debug,
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "error_type": type(e).__name__}
    finally:
        settings.tool_debug_enabled = prev

    msgs = out.get("messages") or []
    answer = _last_assistant_text(msgs)
    if not answer:
        answer = "(模型无文字回答，请查看 message_trace 或开启 tool_debug)"

    return {
        "ok": True,
        "answer": answer,
        "message_trace": _serialize_messages(msgs),
        "tool_debug_path": str(settings.tool_debug_log_path) if body.tool_debug else None,
    }


@app.post("/api/agent/query/raw")
def agent_query_raw(body: AgentQueryBody) -> dict[str, Any]:
    """同上，但额外返回 invoke 的原始结构（默认序列化，便于排查）。"""
    if not (body.question or "").strip():
        raise HTTPException(status_code=400, detail="question 为空")

    if _graph is None:
        raise HTTPException(status_code=503, detail="图未初始化")

    prev = settings.tool_debug_enabled
    if body.tool_debug:
        settings.tool_debug_enabled = True

    try:
        out = invoke_chat(
            _graph,
            body.question.strip(),
            thread_id=body.thread_id,
            debug_tools=body.tool_debug,
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "error_type": type(e).__name__}
    finally:
        settings.tool_debug_enabled = prev

    msgs = out.get("messages") or []
    answer = _last_assistant_text(msgs)
    return {
        "ok": True,
        "answer": answer,
        "message_trace": _serialize_messages(msgs),
        "invoke_raw": json.loads(json.dumps(out, default=str)),
    }
