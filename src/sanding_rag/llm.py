"""Context-bound LLM clients. Sources are added by code, never invented by a model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol


SYSTEM_PROMPT = """你是叁鼎芯供应链商品知识助手。
只能依据给出的“检索证据”回答，不能使用常识补全或猜测。
不得编造库存、实时交期、最终报价、审批结果、认证或未写明的商品参数。
每段检索证据都有由系统分配的 source id（如 S1、S2）。
只要问题明确比较多个商品，且每个商品都有公开证据，应分别回答；“比较”本身不是转人工理由。公开展示价可以回答，但不得把它说成订单价、批量价或经销价。
只能输出一个可解析的 JSON 对象，不能使用 Markdown 代码块或附加说明，格式必须是：
{{"answer":"简洁中文回答或固定转人工文案","used_source_ids":["S1"]}}
``used_source_ids`` 必须列出支撑答案的所有且仅有的 source id，不能杜撰或遗漏。不得在 ``answer`` 中写 source id、product_id 或其他内部字段。
若证据无法直接回答，``answer`` 必须严格等于：{handoff}，且 ``used_source_ids`` 必须是空数组。"""


class LLMRuntimeError(RuntimeError):
    """A provider transport/authentication/envelope failure, not a model badcase."""


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
        except urllib.error.HTTPError as exc:
            # Do not echo a provider body: it can contain request-specific or
            # sensitive diagnostics. HTTP status is enough for the evaluation
            # runner to classify authentication, model, quota and server faults.
            raise LLMRuntimeError(f"LLM request failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMRuntimeError(f"LLM request failed: {type(exc).__name__}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMRuntimeError("LLM response was not a valid JSON envelope") from exc
        try:
            answer = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMRuntimeError("LLM response did not contain choices[0].message.content") from exc
        return answer


class ContextEchoLLM:
    """A test-only deterministic stand-in that proves context reaches the LLM boundary."""

    def answer(self, question: str, context: str, handoff_message: str) -> str:
        if not context.strip():
            return json.dumps({"answer": handoff_message, "used_source_ids": []}, ensure_ascii=False)
        # Omit only the orchestration headers. Joining every retrieved chunk is
        # important because a Markdown title and its facts may be split apart.
        evidence_body = "\n".join(
            line
            for line in context.splitlines()
            if not line.startswith("[S") and not line.startswith("公开证据：")
        )
        source_ids = [
            line.removeprefix("[").removesuffix("]")
            for line in context.splitlines()
            if line.startswith("[S") and line.endswith("]")
        ]
        return json.dumps(
            {
                "answer": f"根据已检索资料：{evidence_body[:360]}",
                "used_source_ids": source_ids,
            },
            ensure_ascii=False,
        )
