"""Context-bound LLM clients. Sources are added by code, never invented by a model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol


SYSTEM_PROMPT = """你是叁鼎芯供应链商品知识助手。
只能依据给出的“检索证据”回答，不能使用常识补全或猜测。
不得编造库存、实时交期、最终报价、审批结果、认证或未写明的商品参数。
若证据无法直接回答，必须只输出：{handoff}
回答使用简洁中文；不要创建来源标记或引用编号，来源由系统在回答后统一返回。"""


class LLMProvider(Protocol):
    def answer(self, question: str, context: str, handoff_message: str) -> str: ...


class OpenAICompatibleLLM:
    """Minimal stdlib client for an OpenAI Chat Completions-compatible endpoint."""

    def __init__(self, api_base: str, api_key: str, model: str, timeout_seconds: int = 45) -> None:
        if not api_key or api_key == "replace-me":
            raise ValueError("LLM_API_KEY is not configured; copy .env.example and set a real key")
        self._url = f"{api_base.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds

    def answer(self, question: str, context: str, handoff_message: str) -> str:
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(handoff=handoff_message)},
                {
                    "role": "user",
                    "content": f"检索证据：\n{context}\n\n用户问题：{question}",
                },
            ],
        }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
        try:
            answer = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RuntimeError("LLM response did not contain choices[0].message.content") from exc
        return answer


class ContextEchoLLM:
    """A test-only deterministic stand-in that proves context reaches the LLM boundary."""

    def answer(self, question: str, context: str, handoff_message: str) -> str:
        if not context.strip():
            return handoff_message
        # Omit only the orchestration headers. Joining every retrieved chunk is
        # important because a Markdown title and its facts may be split apart.
        evidence_body = "\n".join(
            line
            for line in context.splitlines()
            if not line.startswith("[证据 ") and not line.startswith("product_id=")
        )
        return f"根据已检索资料：{evidence_body[:360]}"
