"""Model access through the OpenAI-compatible gateway."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, APITimeoutError, AsyncOpenAI

log = logging.getLogger(__name__)

# How long the backup answers alone after the main model fails.
FALLBACK_S = 30 * 60
# Without one the codex models send each call to whichever cache, and mostly miss.
CACHE_KEY = "shinnku-bot"


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
        # The SDK's own retries. The live bot sets 0: on a 429 the SDK first waits out
        # retry-after, hidden inside one call, where the bot cannot see or bound it.
        max_retries: int = 1,
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
            max_retries=max_retries,
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
            "prompt_cache_key": CACHE_KEY,
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
        try:
            response = await self._client.chat.completions.create(
                **self.request_kwargs(messages, tools, thinking)
            )
        except APITimeoutError as exc:
            raise TimeoutError("the model did not answer in time") from exc
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


class FallbackModel:
    """The main model, with a backup that takes over for FALLBACK_S once the main one fails.

    Any gateway error counts: quota gone, key refused, gateway down. A timeout does
    not; the caller already retries those without thinking. When the backup fails
    too, the main model gets one more try: the quota rolls, so it may be back.
    """

    def __init__(
        self, main: ChatModel, backup: ChatModel, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.main = main
        self.backup = backup
        self._clock = clock
        self._main_from = 0.0

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, thinking: bool | None = None
    ) -> ModelReply:
        tried_main = self._clock() >= self._main_from
        if tried_main:
            try:
                return await self.main.complete(messages, tools, thinking=thinking)
            except APIError as exc:
                # Not the message: it may quote the prompt.
                log.warning(
                    "main model failed (%s %s); backup for %d min",
                    type(exc).__name__,
                    getattr(exc, "status_code", ""),
                    FALLBACK_S // 60,
                )
                self._main_from = self._clock() + FALLBACK_S
        try:
            return await self.backup.complete(messages, tools, thinking=thinking)
        except APIError as exc:
            if tried_main:
                raise
            log.warning(
                "backup failed (%s %s); trying the main model",
                type(exc).__name__,
                getattr(exc, "status_code", ""),
            )
        reply = await self.main.complete(messages, tools, thinking=thinking)
        self._main_from = 0.0  # it is back
        return reply
