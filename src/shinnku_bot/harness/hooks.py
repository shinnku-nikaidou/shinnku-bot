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


# What I tell myself when a draft sounds like a help desk, appended before the second try.
# A draft that is only too long goes out as it is: the second draft was mostly just as long.
AI_TONE_NOTE = "（刚才想发的那句不像我会说的话，太像客服了。我跟朋友说话不会这样。）"

# Code comes fenced like this, and goes out as one message shown as code.
FENCE = "```"
# Telegram takes up to 4096 characters a message.
CODE_MAX = 4000

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
        # Citing what was looked up, which I never say I did.
        r"(?:搜索|查询|检索)结果",
        r"根据.{0,10}(?:搜索|资料|报道|信息|网上)",
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


def code_of(message: str) -> str | None:
    """The code in a message `clean` made from a fenced block, else None."""
    start, end = f"{FENCE}\n", f"\n{FENCE}"
    if message.startswith(start) and message.endswith(end) and len(message) > len(start) + len(end):
        return message[len(start) : -len(end)]
    return None


def _fenced(lines: list[str]) -> list[str]:
    """A block of code as fenced messages that fit in Telegram, cut between lines."""
    pieces = [line[i : i + CODE_MAX] for line in lines for i in range(0, max(len(line), 1), CODE_MAX)]
    while pieces and not pieces[-1].strip():
        pieces.pop()
    while pieces and not pieces[0].strip():
        pieces.pop(0)
    chunks: list[list[str]] = []
    size = 0
    for piece in pieces:
        if not chunks or size + len(piece) + 1 > CODE_MAX:
            chunks.append([])
            size = 0
        chunks[-1].append(piece)
        size += len(piece) + 1
    return [f"{FENCE}\n" + "\n".join(chunk) + f"\n{FENCE}" for chunk in chunks]


def clean(messages: Sequence[str], limits: StyleLimits = StyleLimits()) -> tuple[str, ...]:
    """Strip markup and split multi-line drafts into separate messages.

    A block fenced with ``` stays one message, left as it is, so it goes out as code.
    """
    out: list[str] = []
    for message in messages:
        code: list[str] | None = None
        for line in message.splitlines():
            bare = line.strip()
            if code is None and len(bare) > 2 * len(FENCE) and bare.startswith(FENCE) and bare.endswith(FENCE):
                out.extend(_fenced([bare[len(FENCE) : -len(FENCE)]]))  # code on one line
            elif bare.startswith(FENCE):  # opens a block (perhaps naming a language) or closes it
                if code is None:
                    code = []
                else:
                    out.extend(_fenced(code))
                    code = None
            elif code is not None:
                code.append(line)
            elif line := _clean_line(line, limits):
                out.append(line)
        if code is not None:  # never closed
            out.extend(_fenced(code))
    return tuple(out)


def violations_of(lines: Sequence[str], limits: StyleLimits = StyleLimits()) -> tuple[str, ...]:
    """What is off about these cleaned lines. Code is left alone."""
    prose = [line for line in lines if code_of(line) is None]
    found: list[str] = []
    joined = "\n".join(prose)
    if any(pattern.search(joined) for pattern in _AI_TONE):
        found.append("ai_tone")
    if any(len(line) > limits.max_len for line in prose):
        found.append("too_long")
    return tuple(found)


def fit(lines: Sequence[str], cap: int) -> tuple[str, ...]:
    """At most `cap` messages. The overflow joins the last one, a line each: nothing is dropped."""
    if len(lines) <= cap:
        return tuple(lines)
    return (*lines[: cap - 1], "\n".join(lines[cap - 1 :]))
