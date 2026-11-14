"""Runtime message store: what the bot has seen and said, per chat."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

KINDS = ("text", "photo", "sticker", "voice", "video", "animation", "document", "other")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    chat_id    INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    ts         INTEGER NOT NULL,
    user_id    INTEGER,
    user_name  TEXT    NOT NULL,
    is_me      INTEGER NOT NULL,
    is_bot     INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    text       TEXT    NOT NULL,
    reply_to   INTEGER,
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages (chat_id, ts);
CREATE TABLE IF NOT EXISTS chats (
    chat_id     INTEGER PRIMARY KEY,
    talk_level  INTEGER NOT NULL DEFAULT 0,
    quiet_until INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS pictures (
    chat_id    INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    file_id    TEXT    NOT NULL,
    file_uid   TEXT    NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
CREATE TABLE IF NOT EXISTS seen (
    file_uid TEXT PRIMARY KEY,
    text     TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Msg:
    """One chat message in the neutral form shared by the live bot and the corpus."""

    chat_id: int
    message_id: int
    ts: int
    user_id: int | None
    user_name: str
    is_me: bool
    kind: str = "text"
    # Text, caption, or (for stickers) the sticker's emoji.
    text: str = ""
    is_bot: bool = False
    reply_to: int | None = None
    # What a photo, sticker or GIF shows, once I have looked at it. Not a column
    # of messages: the harness fills it in from the pictures I have seen.
    seen: str = ""


@dataclass(frozen=True)
class ChatState:
    talk_level: int = 0
    quiet_until: int = 0


def _to_msg(row: sqlite3.Row) -> Msg:
    return Msg(
        chat_id=row["chat_id"],
        message_id=row["message_id"],
        ts=row["ts"],
        user_id=row["user_id"],
        user_name=row["user_name"],
        is_me=bool(row["is_me"]),
        kind=row["kind"],
        text=row["text"],
        is_bot=bool(row["is_bot"]),
        reply_to=row["reply_to"],
    )


class Store:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def record(self, msg: Msg) -> None:
        self._db.execute(
            """
            INSERT INTO messages
                (chat_id, message_id, ts, user_id, user_name, is_me, is_bot, kind, text, reply_to)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, message_id) DO UPDATE SET
                ts = excluded.ts, user_id = excluded.user_id, user_name = excluded.user_name,
                is_me = excluded.is_me, is_bot = excluded.is_bot, kind = excluded.kind,
                text = excluded.text, reply_to = excluded.reply_to
            """,
            (
                msg.chat_id,
                msg.message_id,
                msg.ts,
                msg.user_id,
                msg.user_name,
                int(msg.is_me),
                int(msg.is_bot),
                msg.kind,
                msg.text,
                msg.reply_to,
            ),
        )
        self._db.commit()

    def get(self, chat_id: int, message_id: int) -> Msg | None:
        row = self._db.execute(
            "SELECT * FROM messages WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        ).fetchone()
        return _to_msg(row) if row else None

    def recent(
        self, chat_id: int, limit: int, max_age_s: int, now: float | None = None
    ) -> list[Msg]:
        """The latest `limit` messages no older than `max_age_s`, oldest first."""
        cutoff = int((time.time() if now is None else now) - max_age_s)
        rows = self._db.execute(
            """
            SELECT * FROM messages
            WHERE chat_id = ? AND ts >= ?
            ORDER BY message_id DESC
            LIMIT ?
            """,
            (chat_id, cutoff, limit),
        ).fetchall()
        return [_to_msg(row) for row in reversed(rows)]

    def between(self, chat_id: int, start_ts: int, end_ts: int) -> list[Msg]:
        """Messages with start_ts <= ts < end_ts, oldest first."""
        rows = self._db.execute(
            "SELECT * FROM messages WHERE chat_id = ? AND ts >= ? AND ts < ? ORDER BY message_id",
            (chat_id, int(start_ts), int(end_ts)),
        ).fetchall()
        return [_to_msg(row) for row in rows]

    def newer(self, chat_id: int, message_id: int) -> int:
        """How many messages from others came after this one."""
        return self._db.execute(
            "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND message_id > ? AND is_me = 0",
            (chat_id, message_id),
        ).fetchone()[0]

    def chat_ids(self) -> list[int]:
        rows = self._db.execute("SELECT DISTINCT chat_id FROM messages ORDER BY chat_id").fetchall()
        return [row[0] for row in rows]

    def count(self, chat_id: int) -> int:
        return self._db.execute(
            "SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)
        ).fetchone()[0]

    def chat_state(self, chat_id: int) -> ChatState:
        row = self._db.execute(
            "SELECT talk_level, quiet_until FROM chats WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return ChatState(row["talk_level"], row["quiet_until"]) if row else ChatState()

    def _set(self, chat_id: int, column: str, value: int) -> None:
        self._db.execute("INSERT OR IGNORE INTO chats (chat_id) VALUES (?)", (chat_id,))
        self._db.execute(f"UPDATE chats SET {column} = ? WHERE chat_id = ?", (value, chat_id))
        self._db.commit()

    def set_quiet_until(self, chat_id: int, until_ts: int) -> None:
        self._set(chat_id, "quiet_until", int(until_ts))

    def set_talk_level(self, chat_id: int, level: int) -> None:
        self._set(chat_id, "talk_level", int(level))

    def add_picture(self, chat_id: int, message_id: int, file_id: str, file_uid: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO pictures VALUES (?, ?, ?, ?)",
            (chat_id, message_id, file_id, file_uid),
        )
        self._db.commit()

    def picture(self, chat_id: int, message_id: int) -> tuple[str, str, str] | None:
        """(file_id, file_uid, what it shows) of a message's picture; the last is empty until seen."""
        row = self._db.execute(
            """
            SELECT file_id, file_uid, COALESCE(seen.text, '') FROM pictures
            LEFT JOIN seen USING (file_uid)
            WHERE chat_id = ? AND message_id = ?
            """,
            (chat_id, message_id),
        ).fetchone()
        return tuple(row) if row else None

    def set_seen(self, file_uid: str, text: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO seen VALUES (?, ?)", (file_uid, text))
        self._db.commit()
