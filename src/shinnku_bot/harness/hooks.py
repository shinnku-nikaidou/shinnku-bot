"""Checks on what I am about to send. Plain code, no model calls."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class StyleLimits:
    """Placeholder limits until the style fingerprint supplies measured ones."""

    max_len: int = 80
    # Messages in one answer to one question.
    max_burst: int = 4
    strip_period: bool = True


# What I tell myself when a draft is off, appended before the second try.
RETRY_NOTES = {
    "ai_tone": "（刚才想发的那句不像我会说的话，太像客服了。我跟朋友说话不会这样。）",
    "too_long": "（我一条消息不会写这么长。话多就拆成几条短的，或者干脆少说点。）",
}

_AI_TONE = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"作为(?:一个|一名|一位)?\s*(?:AI|人工智能|语言模型|智能助手|助手|机器人)",
        r"(?:AI|人工智能)\s*(?:助手|语言模型)",
        r"我(?:无法|不能|没办法)(?:为你|为您)?(?:提供|协助|满足)",
        r"希望.{0,8}对(?:你|您)有(?:所)?帮助",
        r"很高兴(?:为你|为您|能够?)",
        r"有什么(?:可以|能)(?:帮|为)(?:你|您)",
        r"如果(?:你|您)(?:还)?有(?:其他|任何|什么)(?:问题|疑问|需要)",
        r"为您|对您|您好",
        r"首先[，,：:].{0,80}其次",
        r"以下是",
    )
]

_EMPHASIS = re.compile(r"\*\*|__|~~|`")
_LIST_MARK = re.compile(r"^\s*(?:[-*•·]\s+|#{1,6}\s+|\d+、\s*|\d+[.)]\s+)")
_SPEAKER = re.compile(r"^\s*(?:我|shinnku|真红)\s*[:：]\s*", re.IGNORECASE)
_LONE_PERIOD = re.compile(r"(?<![。.])。$")
_QUOTES = (("“", "”"), ('"', '"'), ("「", "」"), ("『", "』"))


def _clean_line(line: str, limits: StyleLimits) -> str:
    line = _SPEAKER.sub("", line)
    line = _LIST_MARK.sub("", line)
    line = _EMPHASIS.sub("", line).strip()
    for left, right in _QUOTES:
        if len(line) >= 2 and line.startswith(left) and line.endswith(right):
            inner = line[len(left) : -len(right)]
            if left not in inner and right not in inner:
                line = inner.strip()
            break
    if limits.strip_period:
        line = _LONE_PERIOD.sub("", line)
    return line.strip()


def clean(messages: Sequence[str], limits: StyleLimits = StyleLimits()) -> tuple[str, ...]:
    """Strip markup and split multi-line drafts into separate messages."""
    out: list[str] = []
    for message in messages:
        for line in message.splitlines():
            line = _clean_line(line, limits)
            if line:
                out.append(line)
    return tuple(out)


def violations_of(lines: Sequence[str], limits: StyleLimits = StyleLimits()) -> tuple[str, ...]:
    """What is off about these cleaned lines."""
    found: list[str] = []
    joined = "\n".join(lines)
    if any(pattern.search(joined) for pattern in _AI_TONE):
        found.append("ai_tone")
    if any(len(line) > limits.max_len for line in lines):
        found.append("too_long")
    return tuple(found)


def fit(lines: Sequence[str], cap: int) -> tuple[str, ...]:
    """At most `cap` messages. The overflow joins the last one, a line each: nothing is dropped."""
    if len(lines) <= cap:
        return tuple(lines)
    return (*lines[: cap - 1], "\n".join(lines[cap - 1 :]))
