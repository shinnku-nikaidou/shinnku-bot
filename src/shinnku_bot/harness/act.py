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
            "description": (
                "在聊天里说话。一条消息是数组里的一个元素，可以像平时那样连着发几条。"
                "回清单里的某一句，就把它的编号写进 to。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}, "to": {"type": "integer"}},
                            "required": ["text"],
                        },
                    },
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


# One message: the number of the waiting question it answers (None if none), and its text.
Line = tuple[int | None, str]


@dataclass(frozen=True)
class Say:
    messages: tuple[Line, ...]


@dataclass(frozen=True)
class Silent:
    pass


Action = Say | Silent


def _line(item: Any) -> Line | None:
    """A message as {"text", "to"}, or as a bare string from a backend that ignores the schema."""
    if isinstance(item, str):
        return (None, item) if item.strip() else None
    if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
        return None
    to = item.get("to")
    return (to if isinstance(to, int) and not isinstance(to, bool) else None, item["text"])


def _as_messages(value: Any) -> tuple[Line, ...]:
    if isinstance(value, str | dict):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(line for line in map(_line, value) if line is not None)


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
