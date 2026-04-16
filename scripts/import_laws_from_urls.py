#!/usr/bin/env python3
"""从 URL 列表抓取法规页面并保存到 data/raw/laws/。

注意：本脚本只负责“保存原始页面”。你仍需要运行 build_law_index.py
把原文转为 BM25/FAISS 索引。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import settings
from app.data_pipeline.crawl_laws import download_to_raw


def _default_out_path_for_url(url: str, out_dir: Path) -> Path:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "download.html"
    if not name.endswith((".html", ".htm", ".txt")):
        name = name + ".html"
    return out_dir / name


def main() -> None:
    p = argparse.ArgumentParser(description="导入法规 URL -> data/raw/laws/")
    p.add_argument(
        "--urls-file",
        type=Path,
        required=True,
        help="URL 列表文件：每行一个 URL，空行/以 # 开头的行会跳过。",
    )
    p.add_argument("--out-dir", type=Path, default=settings.data_raw_laws)
    p.add_argument("--max-pages", type=int, default=0, help="0 表示不限制。")
    p.add_argument("--skip-existing", action="store_true", help="如果目标文件存在则跳过下载。")
    args = p.parse_args()

    if not args.urls_file.exists():
        print(f"错误：urls 文件不存在：{args.urls_file}", file=sys.stderr)
        sys.exit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    urls: list[str] = []
    for line in args.urls_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)

    if args.max_pages and args.max_pages > 0:
        urls = urls[: args.max_pages]

    saved = 0
    for url in urls:
        target = _default_out_path_for_url(url, args.out_dir)
        if args.skip_existing and target.exists():
            continue
        download_to_raw(url, dest_dir=args.out_dir)
        saved += 1

    print(f"Imported laws: {saved} pages -> {args.out_dir}")


if __name__ == "__main__":
    main()

