"""模型结构化输出解析：本地 Qwen 系常用的 `<tool_call>` JSON 协议。"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)


def normalize_tool_spec(tool: Any) -> tuple[str, str]:
    """返回 (name, description)。"""
    if isinstance(tool, dict):
        if tool.get("type") == "function" and tool.get("function"):
            name = tool["function"].get("name") or tool.get("name") or "unknown_tool"
            instr = tool["function"].get("description") or ""
            return str(name), str(instr)
        if tool.get("name"):
            return str(tool["name"]), str(tool.get("description") or "")
        return "unknown_tool", str(tool)

    name = getattr(tool, "name", None) or getattr(tool, "__name__", None) or "unknown_tool"
    desc = getattr(tool, "description", None) or ""
    return str(name), str(desc)


def extract_tool_param_names(tool: Any) -> list[str]:
    """尽量从工具定义推断参数名，用于提示模型。"""
    if isinstance(tool, dict):
        fn = tool.get("function") or {}
        parameters = fn.get("parameters") or {}
        props = parameters.get("properties") if isinstance(parameters, dict) else None
        if isinstance(props, dict):
            return [str(k) for k in props.keys()]
        return []

    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_fields"):
        return [str(k) for k in args_schema.model_fields.keys()]

    fn = getattr(tool, "func", None)
    ann = getattr(fn, "__annotations__", None)
    if ann and isinstance(ann, dict):
        return [str(k) for k in ann.keys() if k != "return"]

    return []


def build_tool_instruction_for_prompt(tools: list[Any]) -> str:
    """生成注入到 system 的工具说明（与 LocalTransformersToolCallingChatModel 一致）。"""
    lines = [
        "你可以使用以下工具解决任务。",
        "当你需要调用工具时，你必须只输出一个或多个如下标签（不输出其他文本）：",
        "格式：",
        '<tool_call>{"name":"<tool_name>","arguments":{...},"id":"call_1","type":"tool_call"}</tool_call>',
        "",
        "工具列表：",
    ]
    for t in tools:
        name, desc = normalize_tool_spec(t)
        params = extract_tool_param_names(t)
        if desc:
            if params:
                lines.append(f"- {name}({', '.join(params)}): {desc}")
            else:
                lines.append(f"- {name}: {desc}")
        else:
            if params:
                lines.append(f"- {name}({', '.join(params)})")
            else:
                lines.append(f"- {name}")
    lines.append("")
    lines.append("如果不需要工具，直接输出最终答案文本（不要包含 <tool_call> 标签）。")
    return "\n".join(lines)


def parse_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """
    从模型整段文本中解析 `<tool_call>...</tool_call>`，返回 LangChain 风格的 tool_calls 列表元素：
    每项含 name, args, id, type。
    """
    tool_calls: list[dict[str, Any]] = []
    for m in TOOL_CALL_TAG_RE.finditer(text or ""):
        payload = m.group(1).strip()
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        raw = payload[start : end + 1]
        parse_candidates = [raw]
        parse_candidates.append(raw.replace("{{", "{").replace("}}", "}"))
        parse_candidates.append(
            raw.replace("```json", "").replace("```", "").strip().replace("{{", "{").replace("}}", "}")
        )

        obj = None
        for cand in parse_candidates:
            try:
                obj = json.loads(cand)
                break
            except json.JSONDecodeError:
                obj = None
        if obj is None:
            continue

        name = obj.get("name") or obj.get("tool_name")
        arguments = obj.get("arguments") or obj.get("args") or {}
        tool_call_id = obj.get("id") or obj.get("tool_call_id") or str(uuid.uuid4())

        if not name:
            continue
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}

        tool_calls.append(
            {
                "name": str(name),
                "args": arguments,
                "id": str(tool_call_id),
                "type": "tool_call",
            }
        )
    return tool_calls
