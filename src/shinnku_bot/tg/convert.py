"""Telegram message -> the neutral Msg the harness works with."""

from __future__ import annotations

import time
from typing import Any

from shinnku_bot.store import Msg


def kind_and_text(message: Any) -> tuple[str, str]:
    """Classify a message and pick the text that stands for it."""
    if message.text is not None:
        return "text", message.text
    caption = getattr(message, "caption", None) or ""
    sticker = getattr(message, "sticker", None)
    if sticker is not None:
        return "sticker", getattr(sticker, "emoji", None) or ""
    if getattr(message, "photo", None):
        return "photo", caption
    if getattr(message, "voice", None) is not None:
        return "voice", caption
    if getattr(message, "video", None) is not None or getattr(message, "video_note", None) is not None:
        return "video", caption
    # Telegram sets both `animation` and `document` on GIFs, so check this first.
    if getattr(message, "animation", None) is not None:
        return "animation", caption
    if getattr(message, "document", None) is not None:
        return "document", caption
    return "other", caption


def display_name(user: Any) -> str:
    return (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or getattr(user, "first_name", None)
        or str(user.id)
    )


def to_msg(message: Any, bot_id: int) -> Msg:
    user = message.from_user
    kind, text = kind_and_text(message)
    reply = message.reply_to_message
    date = getattr(message, "date", None)
    return Msg(
        chat_id=message.chat_id,
        message_id=message.message_id,
        ts=int(date.timestamp()) if date is not None else int(time.time()),
        user_id=user.id if user is not None else None,
        user_name=display_name(user) if user is not None else "某人",
        is_me=user is not None and user.id == bot_id,
        kind=kind,
        text=text,
        is_bot=bool(user is not None and user.is_bot),
        reply_to=reply.message_id if reply is not None else None,
    )
