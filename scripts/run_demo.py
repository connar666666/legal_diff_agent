#!/usr/bin/env python3
"""演示：加载服务并调用图（需 Ollama）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.graph.builder import build_agent_graph, invoke_chat
from app.main import bootstrap_services


def main() -> None:
    bootstrap_services()
    g = build_agent_graph()
    out = invoke_chat(g, "请用一句话说明你能做什么。")
    print(out.get("messages", out))


if __name__ == "__main__":
    main()
