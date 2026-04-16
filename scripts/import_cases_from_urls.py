#!/usr/bin/env python3
"""从 URL 列表抓取裁判文书页面 -> 生成案例 JSONL -> 可用于 build_case_index.py。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import settings
from app.data_pipeline.case_parser import parse_case_file
from app.data_pipeline.crawl_cases import download_case_page


def main() -> None:
    p = argparse.ArgumentParser(description="导入案例 URL -> 生成 case_snippets.jsonl")
    p.add_argument(
        "--urls-file",
        type=Path,
        required=True,
        help="URL 列表文件：每行一个 URL，空行/以 # 开头会跳过。",
    )
    p.add_argument("--out-dir", type=Path, default=settings.data_raw_cases, help="下载页面保存目录")
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=settings.data_processed_cases / "case_snippets.jsonl",
        help="输出 JSONL 路径：每行 {\"id\",\"text\"}",
    )
    p.add_argument("--max-pages", type=int, default=0, help="0 表示不限制")
    p.add_argument("--skip-existing", action="store_true", help="页面/条目已存在则跳过下载与解析")
    p.add_argument("--max-chars", type=int, default=12000, help="每条案例文本最大字符数（截断）")
    args = p.parse_args()

    if not args.urls_file.exists():
        print(f"错误：urls 文件不存在：{args.urls_file}", file=sys.stderr)
        sys.exit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    urls: list[str] = []
    for line in args.urls_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)

    if args.max_pages and args.max_pages > 0:
        urls = urls[: args.max_pages]

    existing_ids: set[str] = set()
    if args.skip_existing and args.output_jsonl.exists():
        for line in args.output_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    existing_ids.add(str(obj["id"]))
            except Exception:
                continue

    rows: list[dict[str, str]] = []
    for url in urls:
        case_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        if case_id in existing_ids:
            continue

        html_path = download_case_page(url, dest_dir=args.out_dir)
        title, text = parse_case_file(html_path, source_url=url)
        snippet = text
        if args.max_chars and args.max_chars > 0 and len(snippet) > args.max_chars:
            snippet = snippet[: args.max_chars]

        # title 放前面，提升检索可读性
        if title:
            snippet = f"{title}\n\n{snippet}"

        rows.append({"id": case_id, "text": snippet})

    # 追加到 JSONL（而不是覆盖），方便增量导入
    mode = "a" if args.skip_existing else "w"
    with args.output_jsonl.open(mode, encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Imported cases: {len(rows)} -> {args.output_jsonl}")


if __name__ == "__main__":
    main()

