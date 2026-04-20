"""评测集 JSONL 加载与校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal


TaskType = Literal["retrieve", "compare", "general", "refuse"]


@dataclass
class EvalItem:
    """单条评测样本。"""

    id: str
    question: str
    task_type: TaskType = "retrieve"
    gold_citations: list[str] = field(default_factory=list)
    gold_keywords: list[str] = field(default_factory=list)
    jurisdiction: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _coerce_task_type(v: Any) -> TaskType:
    s = str(v or "retrieve").lower().strip()
    if s in ("retrieve", "compare", "general", "refuse"):
        return s  # type: ignore[return-value]
    return "retrieve"


def load_eval_jsonl(path: Path) -> list[EvalItem]:
    """每行一个 JSON 对象，字段见 data/eval/README.md。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    items: list[EvalItem] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
        eid = str(obj.get("id") or f"line_{line_no}")
        q = str(obj.get("question") or "").strip()
        if not q:
            continue
        gc = obj.get("gold_citations")
        if gc is None:
            gc = []
        if not isinstance(gc, list):
            gc = [str(gc)]
        gk = obj.get("gold_keywords")
        if gk is None:
            gk = []
        if not isinstance(gk, list):
            gk = [str(gk)]
        items.append(
            EvalItem(
                id=eid,
                question=q,
                task_type=_coerce_task_type(obj.get("task_type")),
                gold_citations=[str(x).strip() for x in gc if str(x).strip()],
                gold_keywords=[str(x).strip() for x in gk if str(x).strip()],
                jurisdiction=str(obj.get("jurisdiction") or "").strip(),
                extra={k: v for k, v in obj.items() if k not in {"id", "question", "task_type", "gold_citations", "gold_keywords", "jurisdiction"}},
            )
        )
    return items


def iter_eval_jsonl(path: Path) -> Iterator[EvalItem]:
    yield from load_eval_jsonl(path)
