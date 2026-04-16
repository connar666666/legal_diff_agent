"""SQLite 线程历史：保存 / 恢复 user、assistant 消息列表。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CachedMessage:
    role: str  # "user" | "assistant"
    content: str


class ThreadHistoryStore:
    """
    按 thread_id 存取 JSON 消息列表，供 CLI / API 恢复多轮上下文。
    与 `app.utils.chat_cache.ChatCache` 为同一实现（后者为兼容别名）。
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
                );
                """
            )

    def reset_thread(self, thread_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

    def load(self, thread_id: str) -> list[CachedMessage]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT messages_json FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if not row:
                return []
            raw = json.loads(row[0])
            out: list[CachedMessage] = []
            for item in raw:
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "") or "")
                if role and content is not None:
                    out.append(CachedMessage(role=role, content=content))
            return out

    def save(self, thread_id: str, messages: list[CachedMessage]) -> None:
        payload: list[dict[str, Any]] = [{"role": m.role, "content": m.content} for m in messages]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO threads(thread_id, messages_json, updated_at)
                VALUES(?, ?, strftime('%s','now'))
                ON CONFLICT(thread_id) DO UPDATE SET
                    messages_json=excluded.messages_json,
                    updated_at=excluded.updated_at
                """,
                (thread_id, json.dumps(payload, ensure_ascii=False)),
            )
