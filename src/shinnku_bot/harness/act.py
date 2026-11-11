"""What I can do in a chat. Speaking is a tool call; so is saying nothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shinnku_bot.llm import ModelReply

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": "在聊天里说话。一条消息是数组里的一个元素，可以像平时那样连着发几条。",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["messages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stay_silent",
            "description": "这次不接话。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass(frozen=True)
class Say:
    messages: tuple[str, ...]


@dataclass(frozen=True)
class Silent:
    pass


Action = Say | Silent


def _as_messages(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def parse_action(reply: ModelReply) -> Action | None:
    """Read the model's tool calls. None means it produced nothing usable."""
    silent = False
    for call in reply.tool_calls:
        if call.name == "say":
            messages = _as_messages(call.args.get("messages"))
            if messages:
                return Say(messages)
        elif call.name == "stay_silent":
            silent = True
    if silent:
        return Silent()
    # Some backends answer in plain text even when a tool call is required.
    lines = _as_messages(reply.content.splitlines())
    return Say(lines) if lines else None
