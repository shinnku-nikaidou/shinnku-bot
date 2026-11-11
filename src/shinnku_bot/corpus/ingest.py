"""Load exports into the corpus database."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shinnku_bot.corpus.db import CorpusDB, Row
from shinnku_bot.corpus.export import ExportReader
from shinnku_bot.store import Msg

_SENDER = re.compile(r"^(user|channel)(\d+)$")
_LINK_ONLY = re.compile(r"^\s*https?://\S+\s*$")
_VIDEO = {"video_file", "video_message"}
_UNSUPPORTED = (
    "poll",
    "location_information",
    "contact_information",
    "game_title",
    "giveaway_information",
)


def sender_id(raw: Any) -> int | None:
    """`user123` -> 123, `channel456` -> -456."""
    match = _SENDER.match(raw) if isinstance(raw, str) else None
    if match is None:
        return None
    number = int(match.group(2))
    return number if match.group(1) == "user" else -number


def sender_name(name: Any, sid: int | None) -> str:
    """The display name, or a stable stand-in for deleted accounts."""
    if isinstance(name, str) and name.strip():
        return name
    return f"路人{abs(sid) % 100:02d}" if sid is not None else "某人"


def flatten_text(text: Any) -> str:
    """Export text is a string, or a list mixing strings and {type, text} parts."""
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in text
            if isinstance(part, (str, dict))
        )
    return ""


def kind_of(message: dict[str, Any], text: str) -> str:
    media = message.get("media_type")
    if media == "sticker":
        return "sticker"
    if media in _VIDEO:
        return "video"
    if media == "voice_message":
        return "voice"
    if media == "animation":
        return "animation"
    if "photo" in message:
        return "photo"
    if "file" in message or media == "audio_file":
        return "document"
    if any(key in message for key in _UNSUPPORTED):
        return "other"
    return "text" if text else "other"


def to_row(message: dict[str, Any], chat_id: int, owner_id: int) -> Row | None:
    """Convert one export message. Service messages and empty ones give None."""
    if message.get("type") != "message":
        return None
    try:
        msg_id = int(message["id"])
        ts = int(message["date_unixtime"])
    except (KeyError, TypeError, ValueError):
        return None
    text = flatten_text(message.get("text"))
    kind = kind_of(message, text)
    if kind == "sticker":
        emoji = message.get("sticker_emoji")
        text = emoji if isinstance(emoji, str) else ""
    elif kind == "other" and not text:
        return None
    sid = sender_id(message.get("from_id"))
    reply_to = message.get("reply_to_message_id")
    canned = (
        "forwarded_from" in message
        or "via_bot" in message
        or text.lstrip().startswith("/")
        or bool(_LINK_ONLY.match(text))
    )
    return Row(
        Msg(
            chat_id=chat_id,
            message_id=msg_id,
            ts=ts,
            user_id=sid,
            user_name=sender_name(message.get("from"), sid),
            is_me=sid == owner_id,
            kind=kind,
            text=text,
            reply_to=reply_to if isinstance(reply_to, int) else None,
        ),
        canned,
    )


@dataclass
class IngestResult:
    source: str
    chat_id: int
    chat_kind: str
    complete: bool
    read: int = 0
    stored: int = 0
    mine: int = 0
    kinds: Counter = field(default_factory=Counter)
    first_ts: int = 0
    last_ts: int = 0


def ingest(db: CorpusDB, path: Path, owner_id: int) -> IngestResult:
    """Replace this chat's messages in `db` with the contents of one export."""
    path = Path(path)
    with ExportReader(path) as reader:
        header = reader.header
        chat_id = header.get("id")
        if not isinstance(chat_id, int):
            raise ValueError("export has no numeric chat id")
        result = IngestResult(
            source=f"{path.parent.name}/{path.name}",
            chat_id=chat_id,
            chat_kind=str(header.get("type") or ""),
            complete=True,
        )

        def rows() -> Iterator[Row]:
            for message in reader:
                result.read += 1
                row = to_row(message, chat_id, owner_id)
                if row is None:
                    continue
                result.kinds[row.msg.kind] += 1
                result.mine += row.msg.is_me
                result.first_ts = result.first_ts or row.msg.ts
                result.last_ts = row.msg.ts
                yield row

        title = header.get("name")
        result.stored = db.replace_chat(
            chat_id, result.chat_kind, title if isinstance(title, str) else "", result.source, rows()
        )
        result.complete = not reader.truncated
    if not result.complete:
        db.mark_incomplete(chat_id)
    return result
