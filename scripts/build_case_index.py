#!/usr/bin/env python3
"""从案例片段 JSONL（每行 {\"id\",\"text\"}）构建案例索引。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import settings
from app.retrieval.bm25_index import BM25LawIndex
from app.retrieval.embedding import encode_texts, embedding_dim
from app.retrieval.vector_store import FaissVectorStore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path, help="每行一个 JSON 对象，含 id 与 text")
    p.add_argument("--index-dir", type=Path, default=settings.data_index)
    args = p.parse_args()

    args.index_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        pairs.append((obj["id"], obj["text"]))

    id_to_text = {i: t for i, t in pairs}
    (args.index_dir / "case_texts.json").write_text(
        json.dumps(id_to_text, ensure_ascii=False), encoding="utf-8"
    )

    bm25 = BM25LawIndex()
    bm25.build(pairs)
    bm25.save(args.index_dir / "case_bm25.json")

    dim = embedding_dim()
    mat = encode_texts([t for _, t in pairs])
    store = FaissVectorStore(dim=dim)
    store.add([i for i, _ in pairs], mat)
    store.save(args.index_dir / "case_faiss")

    print(f"Indexed {len(pairs)} case snippets -> {args.index_dir}")


if __name__ == "__main__":
    main()
