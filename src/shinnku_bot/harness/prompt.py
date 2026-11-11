"""Assemble what the model sees, most stable part first so the prefix caches."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

_WEEKDAYS = "一二三四五六日"
FACTS_INTRO = "关于我自己，这些我心里有数："
ANCHOR_INTRO = "下面是我以前聊天的一些片段，原样记着的。带「我:」的是我说的话。"
GROUP_INTRO = "这个群在我眼里："
PEOPLE_INTRO = "眼前这几个人："
DIARY_INTRO = "前几天这里的事，我还记得："
MEMORY_INTRO = "眼下这情形，让我想起以前这几次："
UNPROMPTED = "没人叫我，我只是在看；有想说的就说，没有就不说"


def scene_header(
    title: str | None,
    is_private: bool,
    bot_username: str | None,
    now: datetime,
    unprompted: bool = False,
) -> str:
    """One line of where and when, e.g. （群「某群」，周三 21:14，我在这儿的账号是 @x）."""
    if is_private:
        where = "私聊"
    else:
        where = f"群「{title}」" if title else "群聊"
    when = f"周{_WEEKDAYS[now.weekday()]} {now:%H:%M}"
    handle = f"，我在这儿的账号是 @{bot_username}" if bot_username else ""
    aside = f"，{UNPROMPTED}" if unprompted else ""
    return f"（{where}，{when}{handle}{aside}）"


def render_around(
    group: str = "",
    people: Sequence[tuple[str, str]] = (),
    diary: Sequence[tuple[str, str]] = (),
) -> str:
    """What I know about this place: the group, who is here, the last few days.

    `people` is (name as shown in the chat, card); `diary` is (date, entry).
    """
    parts: list[str] = []
    if group.strip():
        parts.append(f"{GROUP_INTRO}\n{group.strip()}")
    if people:
        cards = "\n".join(f"「{name}」{card.strip()}" for name, card in people)
        parts.append(f"{PEOPLE_INTRO}\n{cards}")
    if diary:
        days = "\n".join(f"{date}：{entry.strip()}" for date, entry in diary)
        parts.append(f"{DIARY_INTRO}\n{days}")
    return "\n\n".join(parts)


def build_messages(
    self_text: str,
    header: str,
    turns: Sequence[dict[str, str]],
    notes: Sequence[str] = (),
    anchor: str = "",
    memory: str = "",
    facts: str = "",
    around: str = "",
) -> list[dict[str, str]] | None:
    """Identity, facts and fixed sample; this place and recalled moments; the chat.

    The first message never changes between requests, so it is what the gateway
    caches. Returns None when there is nothing to answer: no turns, or I spoke last.
    """
    if not turns or turns[-1]["role"] != "user":
        return None
    system = self_text.strip()
    if facts.strip():
        system += f"\n\n{FACTS_INTRO}\n\n{facts.strip()}"
    if anchor.strip():
        system += f"\n\n{ANCHOR_INTRO}\n\n{anchor.strip()}"
    messages = [{"role": "system", "content": system}]

    second = [around.strip()] if around.strip() else []
    if memory.strip():
        second.append(f"{MEMORY_INTRO}\n\n{memory.strip()}")
    if second:
        messages.append({"role": "system", "content": "\n\n".join(second)})

    body = [dict(turn) for turn in turns]
    if body[0]["role"] == "user":
        body[0]["content"] = f"{header}\n{body[0]['content']}"
    else:
        body.insert(0, {"role": "user", "content": header})
    messages.extend(body)
    messages.extend({"role": "system", "content": note} for note in notes)
    return messages
