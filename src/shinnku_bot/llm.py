"""Model access through the OpenAI-compatible gateway."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ModelReply:
    tool_calls: tuple[ToolCall, ...] = ()
    content: str = ""
    reasoning: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0


class ChatModel(Protocol):
    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, thinking: bool | None = None
    ) -> ModelReply:
        """`thinking` overrides the model's own setting for this one call."""
        ...


def _parse_args(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _int(obj: Any, name: str) -> int:
    return int(getattr(obj, name, 0) or 0) if obj is not None else 0


class GatewayModel:
    """One chat-completions call that must end in a tool call."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        thinking: bool = True,
        timeout_s: float = 60.0,
        client: AsyncOpenAI | None = None,
        effort: str = "low",
        # Thinking counts against this too. At 1024 the model now and then spent it
        # all on thinking and never got to speak.
        max_tokens: int = 8192,
    ) -> None:
        self.model = model
        self.thinking = thinking
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={"x-api-key": api_key},
            timeout=timeout_s,
            max_retries=1,
        )

    def request_kwargs(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], thinking: bool | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "max_tokens": self.max_tokens,
        }
        if self.thinking if thinking is None else thinking:
            kwargs["reasoning_effort"] = self.effort
        else:
            # The only switch this gateway honours for turning reasoning off.
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return kwargs

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, thinking: bool | None = None
    ) -> ModelReply:
        started = time.monotonic()
        response = await self._client.chat.completions.create(
            **self.request_kwargs(messages, tools, thinking)
        )
        latency = time.monotonic() - started

        message = response.choices[0].message
        calls = tuple(
            ToolCall(name=call.function.name, args=_parse_args(call.function.arguments))
            for call in (message.tool_calls or [])
            if getattr(call, "function", None) is not None
        )
        usage = response.usage
        return ModelReply(
            tool_calls=calls,
            content=message.content or "",
            reasoning=getattr(message, "reasoning_content", None) or "",
            prompt_tokens=_int(usage, "prompt_tokens"),
            cached_tokens=_int(getattr(usage, "prompt_tokens_details", None), "cached_tokens"),
            completion_tokens=_int(usage, "completion_tokens"),
            reasoning_tokens=_int(
                getattr(usage, "completion_tokens_details", None), "reasoning_tokens"
            ),
            latency_s=latency,
        )
