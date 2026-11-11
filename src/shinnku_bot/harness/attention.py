"""Should this message wake me up, and am I allowed to answer here right now."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from shinnku_bot.config import Settings
from shinnku_bot.store import ChatState


def _utf16_slice(text: str, offset: int, length: int) -> str:
    """Telegram entity offsets count UTF-16 code units, not Python characters."""
    data = text.encode("utf-16-le")
    return data[offset * 2 : (offset + length) * 2].decode("utf-16-le", errors="ignore")


def mentions_bot(message: Any, bot_id: int, bot_username: str | None) -> bool:
    if message.text is not None:
        text, entities = message.text, message.entities
    else:
        text = getattr(message, "caption", None) or ""
        entities = getattr(message, "caption_entities", None)
    for entity in entities or ():
        if entity.type == "text_mention":
            user = getattr(entity, "user", None)
            if user is not None and user.id == bot_id:
                return True
        elif entity.type == "mention" and bot_username:
            handle = _utf16_slice(text, entity.offset, entity.length)
            if handle.lstrip("@").lower() == bot_username.lower():
                return True
    return False


def addressed(
    message: Any,
    bot_id: int,
    bot_username: str | None,
    name_triggers: Sequence[str],
) -> str | None:
    """Why this message is aimed at me, or None if it is not.

    Returns "private", "reply", "mention" or "name".
    """
    user = message.from_user
    if user is None or user.is_bot:
        return None
    if message.chat.type == "private":
        return "private"
    reply = message.reply_to_message
    if reply is not None and reply.from_user is not None and reply.from_user.id == bot_id:
        return "reply"
    if mentions_bot(message, bot_id, bot_username):
        return "mention"
    text = (message.text or getattr(message, "caption", None) or "").lower()
    if any(name and name.lower() in text for name in name_triggers):
        return "name"
    return None


def blocked(
    settings: Settings, state: ChatState, chat_id: int, is_private: bool, now: float
) -> str | None:
    """Why I must not answer in this chat right now, or None if I may."""
    if is_private:
        if not settings.allow_private:
            return "private_off"
    elif settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
        return "chat_not_allowed"
    if state.quiet_until > now:
        return "quiet"
    return None

