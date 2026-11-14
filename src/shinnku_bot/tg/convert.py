"""Telegram message -> the neutral Msg the harness works with."""

from __future__ import annotations

import time
from typing import Any

from shinnku_bot.harness.transcript import render_body
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


# Big enough to read the text in a screenshot.
PHOTO_SIDE = 1280


def picture(message: Any) -> tuple[str, str] | None:
    """The image to look at for a photo, sticker or GIF, as (file_id, file_unique_id).

    Animated and video stickers and GIFs are looked at through their still thumbnail.
    """
    if getattr(message, "photo", None):
        # Telegram lists the sizes from small to large.
        sizes = [p for p in message.photo if max(p.width, p.height) <= PHOTO_SIDE]
        file = (sizes or message.photo[:1])[-1]
    elif (sticker := getattr(message, "sticker", None)) is not None:
        file = sticker.thumbnail if sticker.is_animated or sticker.is_video else sticker
    elif (animation := getattr(message, "animation", None)) is not None:
        file = animation.thumbnail
    else:
        return None
    return (file.file_id, file.file_unique_id) if file is not None else None


def display_name(user: Any) -> str:
    return (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or getattr(user, "first_name", None)
        or str(user.id)
    )


def quoted(message: Any, bot_id: int) -> str:
    """What a reply quotes: the part the sender picked, else the whole message it answers.

    Telegram sends both with the reply, so this works even for messages I never saw.
    """
    part = getattr(message, "quote", None)
    if part is not None and part.text:
        return part.text
    reply = message.reply_to_message
    # In a forum topic every message "replies" to the service message that opened the topic.
    if reply is None or getattr(reply, "forum_topic_created", None) is not None:
        return ""
    return render_body(to_msg(reply, bot_id))


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
        quote=quoted(message, bot_id),
    )
