"""Each night: write down what happened in each group yesterday, in my own words.

The diary is what lets me remember yesterday. What the owner said and what the
bot said are both mine here: to the diary they are the same person.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from shinnku_bot.config import Settings
from shinnku_bot.corpus.distill import _tool, ask, lint
from shinnku_bot.harness.persona import export_chat_id
from shinnku_bot.harness.transcript import render_lines
from shinnku_bot.llm import ChatModel
from shinnku_bot.store import Msg, Store

MIN_MESSAGES = 5
MAX_MESSAGES = 1500
AROUND_ME = 20
ENTRY_MAX_CHARS = 400

DIARY_TOOL = _tool("write_diary", "写下这一天的日记。", {"entry": {"type": "string"}})
DIARY_SYSTEM = """下面是某个群在 {day} 这一天的聊天记录，「我:」开头的是我说的。请替我写这一天的日记，调用 write_diary。

entry 用第一人称，最多 5 句：群里聊了什么、我说了什么、我答应过或提到要做的事、谁说了以后聊天用得上的事。没什么可记的就只写一句。不记别人的健康、财务、住址这类隐私。"""


def day_bounds(day: date, timezone: str) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, tzinfo=ZoneInfo(timezone))
    return int(start.timestamp()), int((start + timedelta(days=1)).timestamp())


def as_mine(msgs: Sequence[Msg], owner_id: int) -> list[Msg]:
    """The owner's own lines count as mine, like the bot's."""
    return [dataclasses.replace(m, is_me=True) if m.user_id == owner_id else m for m in msgs]


def select(msgs: Sequence[Msg], limit: int = MAX_MESSAGES, around: int = AROUND_ME) -> list[Msg]:
    """On a very busy day keep the stretches around what I said; the newest if still too many."""
    if len(msgs) <= limit:
        return list(msgs)
    keep: set[int] = set()
    for index, msg in enumerate(msgs):
        if msg.is_me:
            keep.update(range(max(0, index - around), min(len(msgs), index + around + 1)))
    chosen = [msgs[i] for i in sorted(keep)] if keep else list(msgs)
    return chosen[-limit:]


async def digest_chat(
    model: ChatModel,
    store: Store,
    persona_dir: Path,
    chat_id: int,
    day: date,
    timezone: str,
    owner_id: int,
    force: bool = False,
) -> str:
    """Write one chat's diary for one day. Returns written, kept, quiet, failed or rejected."""
    path = Path(persona_dir) / "diary" / str(export_chat_id(chat_id)) / f"{day:%Y-%m-%d}.md"
    if path.is_file() and not force:
        return "kept"
    msgs = as_mine(store.between(chat_id, *day_bounds(day, timezone)), owner_id)
    if len(msgs) < MIN_MESSAGES:
        return "quiet"
    lines = render_lines(select(msgs))
    args = await ask(model, DIARY_SYSTEM.format(day=f"{day:%Y-%m-%d}"), "\n".join(lines), DIARY_TOOL)
    if args is None:
        return "failed"
    entry = " ".join(str(args.get("entry", "")).split())[:ENTRY_MAX_CHARS]
    if not entry or lint(entry, 1000):
        return "rejected"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(entry + "\n", encoding="utf-8")
    return "written"


async def digest(
    settings: Settings,
    store: Store,
    model: ChatModel,
    day: date | None = None,
    force: bool = False,
    progress: Callable[[str], None] = lambda line: None,
) -> dict[str, int]:
    """Yesterday's diary (or `day`'s) for every group the bot may speak in."""
    day = day or (datetime.now(ZoneInfo(settings.timezone)).date() - timedelta(days=1))
    counts: dict[str, int] = {}
    for chat_id in store.chat_ids():
        if chat_id >= 0:
            continue  # private chats keep no diary
        if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
            continue
        try:
            outcome = await digest_chat(
                model, store, settings.persona_dir, chat_id, day, settings.timezone, settings.owner_id, force
            )
        except Exception as exc:  # noqa: BLE001 - the message may quote chat text
            outcome = "failed"
            progress(f"chat={chat_id} error={type(exc).__name__}")
        counts[outcome] = counts.get(outcome, 0) + 1
        progress(f"chat={chat_id} day={day} {outcome}")
    return counts
