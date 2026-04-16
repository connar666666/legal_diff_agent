"""入口：加载配置与索引、编译 LangGraph、命令行对话。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

# 确保以 `python app/main.py` 运行时也能找到包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import settings
from app.graph.builder import build_agent_graph, invoke_chat
from app.graph.routing import classify_intent
from app.services.case_service import CaseService
from app.services.law_service import LawService
from app.tools import registry
from app.utils.logger import setup_logging
from app.utils.chat_cache import CachedMessage, ChatCache


def _print_ollama_connection_help() -> None:
    print(
        "无法连接 Ollama（连接被拒绝）。请检查：\n"
        "  1. 本机已启动 Ollama 服务（例如终端执行 `ollama serve`，或使用系统里的 Ollama 应用）。\n"
        f"  2. 当前配置的地址为 {settings.ollama_base_url}，可在项目根目录 `.env` 中设置 OLLAMA_BASE_URL。\n"
        f"  3. 已拉取与配置一致的模型：`ollama pull {settings.ollama_model}`\n",
        file=sys.stderr,
        flush=True,
    )


def bootstrap_services() -> tuple[LawService, CaseService]:
    """加载法规/案例索引（若文件不存在则服务为空，仍可启动）。"""
    law = LawService()
    law.load_from_paths()
    case = CaseService()
    case.load_from_paths()
    registry.set_services(law, case)
    return law, case


def main() -> None:
    setup_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Legal Diff Agent CLI")
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="",
        help="单次提问后退出；不传则进入交互循环",
    )
    parser.add_argument("--thread-id", type=str, default="cli")
    parser.add_argument("--show-intent", action="store_true", help="打印粗分类意图")
    parser.add_argument(
        "--debug-tools",
        action="store_true",
        help="开启工具调用调试日志（写入 data/outputs/tool_call_debug.jsonl）",
    )
    parser.add_argument(
        "--reset-thread",
        action="store_true",
        help="重置当前 thread 的对话缓存（仅影响该 thread）。",
    )
    args = parser.parse_args()

    # debug 开关尽量走运行时配置
    if args.debug_tools:
        settings.tool_debug_enabled = True

    bootstrap_services()
    graph = build_agent_graph()

    cache = ChatCache(settings.thread_history_db_path)

    if args.reset_thread:
        cache.reset_thread(args.thread_id)

    def cached_to_langchain_messages(cached: list[CachedMessage]):
        from langchain_core.messages import AIMessage, HumanMessage  # type: ignore[import-not-found]

        out = []
        for m in cached:
            if m.role == "user":
                out.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                out.append(AIMessage(content=m.content))
        return out

    def langchain_messages_to_cached(messages) -> list[CachedMessage]:
        from langchain_core.messages import AIMessage, HumanMessage  # type: ignore[import-not-found]

        out: list[CachedMessage] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                out.append(CachedMessage(role="user", content=str(getattr(m, "content", ""))))
            elif isinstance(m, AIMessage):
                out.append(
                    CachedMessage(role="assistant", content=str(getattr(m, "content", "")))
                )
        return out

    def run_one(text: str) -> bool:
        if args.show_intent:
            print("intent:", classify_intent(text), flush=True)
        try:
            prior_cached = cache.load(args.thread_id)
            prior_messages = cached_to_langchain_messages(prior_cached)
            out = invoke_chat(
                graph,
                text,
                thread_id=args.thread_id,
                prior_messages=prior_messages,
                debug_tools=args.debug_tools,
            )
        except httpx.ConnectError:
            _print_ollama_connection_help()
            return False
        msgs = out.get("messages") or []
        if msgs:
            last = msgs[-1]
            content = getattr(last, "content", str(last))
            # 兜底：过滤掉模型输出中残留的 <think>...</think> 推理块
            import re as _re
            content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
            if not content:
                content = "(模型无文字回答)"
            print(content, flush=True)
        else:
            print(json.dumps(out, ensure_ascii=False, default=str), flush=True)

        # 持久化：保存用户 + 模型当前轮输出后的完整 user/assistant 历史
        cache.save(args.thread_id, langchain_messages_to_cached(msgs))
        return True

    if args.query:
        if not run_one(args.query):
            sys.exit(1)
        return

    print("Legal Diff Agent — 输入问题，空行或 Ctrl+D 结束。", flush=True)
    try:
        while True:
            line = input("> ").strip()
            if not line:
                continue
            if not run_one(line):
                continue
    except EOFError:
        pass


if __name__ == "__main__":
    main()
