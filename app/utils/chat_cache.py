"""兼容别名：对话 SQLite 缓存，实现见 `app.memory.sqlite_store`。"""

from __future__ import annotations

from app.memory.sqlite_store import CachedMessage, ThreadHistoryStore

# 历史代码与文档中的名称
ChatCache = ThreadHistoryStore

__all__ = ["CachedMessage", "ChatCache", "ThreadHistoryStore"]
