#!/usr/bin/env python3
"""从 processed 法规 JSONL 或纯文本构建 law_bm25 + law_faiss + law_texts.json。"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import settings
from app.data_pipeline.chunker import chunk_law_text
from app.data_pipeline.parser import parse_file
from app.retrieval.bm25_index import BM25LawIndex
from app.retrieval.embedding import encode_texts, embedding_dim
from app.retrieval.vector_store import FaissVectorStore


def _resolve_input_paths(raw: list[Path], cwd: Path) -> list[Path]:
    """把 shell 未展开的 *.txt、目录等解析成真实文件列表。"""
    out: list[Path] = []
    for p in raw:
        s = str(p)
        # bash 在「无匹配」时会把字面量 data/raw/laws/*.txt 原样传给 Python，这里用 glob 再解一次
        if "*" in s or "?" in s:
            matches = sorted(glob_mod.glob(s))
            if not matches:
                matches = sorted(glob_mod.glob(str(cwd / s)))
            for m in matches:
                mp = Path(m)
                if mp.is_file():
                    out.append(mp)
            continue
        rp = p if p.is_absolute() else (cwd / p).resolve()
        if rp.is_dir():
            for pat in ("*.txt", "*.html", "*.htm", "*.pdf"):
                out.extend(sorted(rp.glob(pat)))
        elif rp.is_file():
            out.append(rp)
    # 去重且保持顺序
    seen: set[Path] = set()
    unique: list[Path] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return unique


def main() -> None:
    p = argparse.ArgumentParser(
        description="从法规文本/HTML 构建 BM25 + FAISS + law_texts 索引。",
    )
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[],
        help="法规文本或 HTML 文件路径；也可传目录（自动收集 .txt/.html）；支持通配符",
    )
    p.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="等价于额外传入该目录（收集其中 .txt/.html）",
    )
    p.add_argument("--index-dir", type=Path, default=settings.data_index)
    args = p.parse_args()

    cwd = Path.cwd()
    raw_paths: list[Path] = list(args.inputs)
    if args.dir is not None:
        raw_paths.append(args.dir)

    if not raw_paths:
        print(
            "错误：未指定任何输入。\n"
            "用法示例：\n"
            "  python scripts/build_law_index.py data/raw/laws\n"
            "  python scripts/build_law_index.py --dir data/raw/laws\n"
            "  python scripts/build_law_index.py data/raw/laws/某法.txt\n"
            "请先把法规正文保存为 .txt / .html / .pdf 再放入 data/raw/laws/。",
            file=sys.stderr,
        )
        sys.exit(2)

    input_paths = _resolve_input_paths(raw_paths, cwd)
    if not input_paths:
        print(
            "错误：未找到任何可索引的文件。\n"
            "常见原因：\n"
            "  1) data/raw/laws/ 下还没有 .txt / .html / .pdf 文件（空目录时 shell 的 *.txt 不会展开，"
            "脚本已尝试在程序内做通配，仍无匹配）。\n"
            "  2) 路径写错。\n"
            "请先放入至少一个法规文本文件，再运行本脚本。",
            file=sys.stderr,
        )
        sys.exit(2)

    args.index_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []

    for path in input_paths:
        meta, text = parse_file(path)
        doc_id = path.stem
        chunks = chunk_law_text(meta, text, doc_id)
        for c in chunks:
            pairs.append((c.id, c.text))

    id_to_text = {i: t for i, t in pairs}
    (args.index_dir / "law_texts.json").write_text(
        json.dumps(id_to_text, ensure_ascii=False), encoding="utf-8"
    )

    bm25 = BM25LawIndex()
    bm25.build(pairs)
    bm25.save(args.index_dir / "law_bm25.json")

    dim = embedding_dim()
    mat = encode_texts([t for _, t in pairs])
    store = FaissVectorStore(dim=dim)
    store.add([i for i, _ in pairs], mat)
    store.save(args.index_dir / "law_faiss")

    print(f"Indexed {len(pairs)} chunks -> {args.index_dir}")


if __name__ == "__main__":
    main()
