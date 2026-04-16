"""对外/工具层统一响应形状。"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """工具返回的通用封装。"""

    ok: bool = True
    error: Optional[str] = None
    data: Any = None
    format_hint: Literal["markdown", "json", "text"] = "markdown"


class ExportPayload(BaseModel):
    """导出工具载荷。"""

    content: str
    path: Optional[str] = None
    mime: str = "text/markdown"
