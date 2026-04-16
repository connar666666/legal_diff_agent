"""本地 Transformers（GPU/CPU）直接加载模型，不依赖 Ollama。

工具调用由 `LocalTransformersToolCallingChatModel` + `parse_tool_calls_from_text` 适配；
本模块提供 `LocalTransformersGenerator.chat` 作为底层生成。
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings

logger = logging.getLogger(__name__)


_GEN: Optional["LocalTransformersGenerator"] = None


def _dtype_from_name(name: str) -> torch.dtype:
    n = (name or "").lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16"):
        return torch.float16
    return torch.float32


class LocalTransformersGenerator:
    """把 messages 转成 chat_template，再用 generate 得到回复文本。"""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> None:
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        logger.info("Loading transformers model: %s", self.model_path)
        dtype = _dtype_from_name(self.dtype_name)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()
        logger.info("Transformers model loaded.")

    def chat(self, messages: list[dict[str, str]], *, clean_think: bool = True) -> str:
        """messages: [{role: 'system'|'user'|'assistant', content: str}, ...]"""
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        # 生成提示：使用 tokenizer chat template（例如 Qwen）
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
        }
        # 低温时仍允许 sampling；温度=0 则改为贪心
        if self.temperature is not None and self.temperature <= 1e-6:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(self.temperature)
            gen_kwargs["top_p"] = float(self.top_p)

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # generate 的输出包含 prompt + 新生成 token；只解码新增部分，避免把 system/user 也重复返回
        input_len = int(inputs["input_ids"].shape[-1])
        new_ids = outputs[0][input_len:]
        decoded = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        decoded = decoded.strip()

        if clean_think:
            # Qwen/部分指令模型可能会先输出 <think>...</think> 推理片段；清理掉，避免露出思考过程/导致截断。
            if "<think>" in decoded:
                # 常见形态：<think>...</think> final
                if "</think>" in decoded:
                    decoded = decoded.split("</think>", 1)[-1].strip()
                else:
                    decoded = decoded.split("<think>", 1)[-1].strip()

            # 兜底：如果模型没有严格用 </think>，用正则尽量剔除
            decoded = re.sub(
                r"<think>.*?</think>", "", decoded, flags=re.DOTALL
            ).strip()
        return decoded


def get_local_generator() -> LocalTransformersGenerator:
    global _GEN
    if _GEN is None:
        _GEN = LocalTransformersGenerator(
            model_path=settings.local_model_path,
            trust_remote_code=settings.local_trust_remote_code,
            dtype=settings.local_dtype,
            max_new_tokens=settings.local_max_new_tokens,
            temperature=settings.local_temperature,
            top_p=settings.local_top_p,
        )
    return _GEN

