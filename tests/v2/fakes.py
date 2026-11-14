"""Stand-ins for the model, Telegram and settings. All content here is made up."""

from __future__ import annotations

import asyncio
import itertools
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shinnku_bot.config import Settings
from shinnku_bot.harness.lookup import Hit
from shinnku_bot.llm import ModelReply, ToolCall
from shinnku_bot.store import Msg

BOT_ID = 999
BOT_USERNAME = "akitsu_bot"


def make_settings(data_dir: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "tg_token": "test-token",
        "owner_id": 1,
        "llm_base_url": "http://gateway.test/v1",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "data_dir": data_dir,
        "quiet_window_s": 0.03,
        "max_wait_s": 0.3,
        "trace": False,
        # The exam check is one more model call; the tests that want it turn it on.
        "brush_off_exams": False,
    }
    values.update(overrides)
    return Settings(**values)


def say(*messages: str, reasoning: str = "") -> ModelReply:
    return ModelReply(
        tool_calls=(ToolCall("say", {"messages": list(messages)}),), reasoning=reasoning
    )


def say_to(*messages: tuple[int | None, str]) -> ModelReply:
    """A say call whose messages name the waiting question they answer."""
    items = [{"text": text} if to is None else {"text": text, "to": to} for to, text in messages]
    return ModelReply(tool_calls=(ToolCall("say", {"messages": items}),))


def silent() -> ModelReply:
    return ModelReply(tool_calls=(ToolCall("stay_silent", {}),))


class FakeModel:
    """Returns the queued replies in order; the last one repeats."""

    def __init__(self, *replies: ModelReply, delay: float = 0.0) -> None:
        self.replies = list(replies)
        self.delay = delay
        self.calls: list[list[dict[str, Any]]] = []
        self.options: list[dict[str, Any]] = []
        self.finished = 0

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **options: Any
    ) -> ModelReply:
        self.calls.append(messages)
        self.options.append(options)
        if self.delay:
            await asyncio.sleep(self.delay)
        self.finished += 1
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


class ToolModel:
    """Answers structured calls by tool name: a dict, or a function of the user text.

    A tool with no answer gets a reply without a tool call, which callers treat as failure.
    """

    def __init__(self, **answers: Any) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str, str]] = []

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        name = tools[0]["function"]["name"]
        self.calls.append((name, messages[0]["content"], messages[-1]["content"]))
        answer = self.answers.get(name)
        if callable(answer):
            answer = answer(messages[-1]["content"])
        if answer is None:
            return ModelReply()
        return ModelReply(tool_calls=(ToolCall(name, answer),))

    def asked(self, name: str) -> list[str]:
        """The user texts sent with each call of this tool."""
        return [user for tool, _, user in self.calls if tool == name]


class FakeSearch:
    """Gives the same hits for every query, after `delay`; raises `error` if there is one."""

    def __init__(self, *hits: Hit, error: Exception | None = None, delay: float = 0.0) -> None:
        self.hits = list(hits)
        self.error = error
        self.delay = delay
        self.queries: list[str] = []

    async def __call__(self, query: str) -> list[Hit]:
        self.queries.append(query)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.hits


class FakeOutbox:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, int | None]] = []
        self.typing_calls = 0
        self._ids = itertools.count(1000)

    async def typing(self, chat_id: int) -> None:
        self.typing_calls += 1

    async def send(self, chat_id: int, text: str, reply_to: int | None) -> int:
        self.sent.append((chat_id, text, reply_to))
        return next(self._ids)

    @property
    def texts(self) -> list[str]:
        return [text for _, text, _ in self.sent]


async def no_sleep(seconds: float) -> None:
    return None


def msg(message_id: int, text: str, user_id: int = 7, name: str = "小林", **kwargs: Any) -> Msg:
    values: dict[str, Any] = {
        "chat_id": 100,
        "message_id": message_id,
        "ts": 1_800_000_000 + message_id,
        "user_id": user_id,
        "user_name": name,
        "is_me": False,
        "text": text,
    }
    values.update(kwargs)
    return Msg(**values)


def tg_user(user_id: int, is_bot: bool = False, full_name: str = "小林", username: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=user_id, is_bot=is_bot, full_name=full_name, username=username, first_name=full_name
    )


def tg_entity(kind: str, offset: int, length: int, user: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(type=kind, offset=offset, length=length, user=user)


def tg_message(
    message_id: int = 1,
    text: str | None = "在吗",
    user: SimpleNamespace | None = None,
    chat_type: str = "supergroup",
    reply_to: SimpleNamespace | None = None,
    entities: list[SimpleNamespace] | None = None,
    **extra: Any,
) -> SimpleNamespace:
    values: dict[str, Any] = {
        "message_id": message_id,
        "chat_id": 100,
        "chat": SimpleNamespace(type=chat_type, title="测试群"),
        "from_user": user if user is not None else tg_user(7),
        "text": text,
        "caption": None,
        "entities": entities,
        "caption_entities": None,
        "reply_to_message": reply_to,
        "date": datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc),
        "sticker": None,
        "photo": None,
        "voice": None,
        "video": None,
        "video_note": None,
        "animation": None,
        "document": None,
    }
    values.update(extra)
    return SimpleNamespace(**values)
