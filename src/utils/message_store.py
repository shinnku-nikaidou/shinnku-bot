# encoding: utf-8
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from telegram import Message, Update
from telegram.ext import ContextTypes

from configurations import settings


@dataclass(frozen=True)
class StoredMessage:
    chat_id: int
    message_id: int
    from_user_id: Optional[int]
    from_is_bot: bool
    from_name: str
    text: str
    reply_to_message_id: Optional[int]
    date: int


def _db_path(db_path: Optional[str] = None) -> str:
    return db_path or settings.MESSAGE_CONTEXT_DB_PATH


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = Path(_db_path(db_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            chat_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            from_user_id INTEGER,
            from_is_bot INTEGER NOT NULL,
            from_name TEXT NOT NULL,
            text TEXT NOT NULL,
            reply_to_message_id INTEGER,
            date INTEGER NOT NULL,
            PRIMARY KEY (chat_id, message_id)
        )
        """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_date
        ON messages (date)
        """)
    return conn


def _message_timestamp(message: Message) -> int:
    message_date = message.date
    if isinstance(message_date, datetime):
        if message_date.tzinfo is None:
            message_date = message_date.replace(tzinfo=timezone.utc)
        return int(message_date.timestamp())
    return int(time.time())


def _display_name(message: Message) -> str:
    user = message.from_user
    if user is None:
        return str(message.chat_id)
    return user.full_name or user.username or user.first_name or str(user.id)


def _stored_from_row(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        chat_id=row["chat_id"],
        message_id=row["message_id"],
        from_user_id=row["from_user_id"],
        from_is_bot=bool(row["from_is_bot"]),
        from_name=row["from_name"],
        text=row["text"],
        reply_to_message_id=row["reply_to_message_id"],
        date=row["date"],
    )


def record_message(
    message: Message,
    db_path: Optional[str] = None,
    reply_to_message_id: Optional[int] = None,
) -> None:
    text = message.text
    if text is None:
        return

    user = message.from_user
    if reply_to_message_id is None and message.reply_to_message is not None:
        reply_to_message_id = message.reply_to_message.message_id

    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO messages (
                chat_id,
                message_id,
                from_user_id,
                from_is_bot,
                from_name,
                text,
                reply_to_message_id,
                date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, message_id) DO UPDATE SET
                from_user_id = excluded.from_user_id,
                from_is_bot = excluded.from_is_bot,
                from_name = excluded.from_name,
                text = excluded.text,
                reply_to_message_id = excluded.reply_to_message_id,
                date = excluded.date
            """,
            (
                message.chat_id,
                message.message_id,
                user.id if user else None,
                int(bool(user and user.is_bot)),
                _display_name(message),
                text,
                reply_to_message_id,
                _message_timestamp(message),
            ),
        )
        conn.commit()
    finally:
        conn.close()


async def record_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    # pylint: disable=unused-argument
    if update.message is not None:
        record_message(update.message)


def build_reply_chain(
    chat_id: int,
    message_id: int,
    limit: int = 20,
    max_age_days: int = 30,
    db_path: Optional[str] = None,
) -> list[StoredMessage]:
    cutoff = int(
        (datetime.now(timezone.utc) - timedelta(days=max_age_days)).timestamp()
    )
    current_message_id: Optional[int] = message_id
    seen: set[int] = set()
    chain: list[StoredMessage] = []

    conn = _connect(db_path)
    try:
        while current_message_id is not None and len(chain) < limit:
            if current_message_id in seen:
                break
            seen.add(current_message_id)

            row = conn.execute(
                """
                SELECT
                    chat_id,
                    message_id,
                    from_user_id,
                    from_is_bot,
                    from_name,
                    text,
                    reply_to_message_id,
                    date
                FROM messages
                WHERE chat_id = ? AND message_id = ?
                """,
                (chat_id, current_message_id),
            ).fetchone()
            if row is None or row["date"] < cutoff:
                break

            stored = _stored_from_row(row)
            chain.append(stored)
            current_message_id = stored.reply_to_message_id
    finally:
        conn.close()

    return list(reversed(chain))


def cleanup_old_messages(days: int = 30, db_path: Optional[str] = None) -> None:
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM messages WHERE date < ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()
