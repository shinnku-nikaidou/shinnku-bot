"""The corpus database: every exported message, and the reply pairs cut from them."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from shinnku_bot.store import Msg

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat (
    chat_id  INTEGER PRIMARY KEY,
    kind     TEXT    NOT NULL,
    title    TEXT    NOT NULL,
    messages INTEGER NOT NULL,
    complete INTEGER NOT NULL,
    source   TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS msg (
    chat_id     INTEGER NOT NULL,
    msg_id      INTEGER NOT NULL,
    ts          INTEGER NOT NULL,
    sender_id   INTEGER,
    sender_name TEXT    NOT NULL,
    is_me       INTEGER NOT NULL,
    kind        TEXT    NOT NULL,
    text        TEXT    NOT NULL,
    reply_to    INTEGER,
    -- Forwarded, sent through an inline bot, a /command, or nothing but a link:
    -- not something I typed as conversation.
    canned      INTEGER NOT NULL,
    PRIMARY KEY (chat_id, msg_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS pair (
    pair_id      INTEGER PRIMARY KEY,
    chat_id      INTEGER NOT NULL,
    first_id     INTEGER NOT NULL,
    last_id      INTEGER NOT NULL,
    ctx_first_id INTEGER NOT NULL,
    ts           INTEGER NOT NULL,
    n_reply      INTEGER NOT NULL,
    n_ctx        INTEGER NOT NULL,
    reply_chars  INTEGER NOT NULL,
    quoted       INTEGER NOT NULL,
    split        TEXT    NOT NULL DEFAULT 'train'
);
CREATE INDEX IF NOT EXISTS idx_pair_split ON pair (split, ts);
"""


@dataclass(frozen=True)
class Row:
    """One corpus message: a Msg plus the corpus-only `canned` flag."""

    msg: Msg
    canned: bool


@dataclass(frozen=True)
class Pair:
    pair_id: int
    chat_id: int
    first_id: int
    last_id: int
    ctx_first_id: int
    ts: int
    n_reply: int
    n_ctx: int
    reply_chars: int
    quoted: bool
    split: str


def _row(r: sqlite3.Row) -> Row:
    return Row(
        Msg(
            chat_id=r["chat_id"],
            message_id=r["msg_id"],
            ts=r["ts"],
            user_id=r["sender_id"],
            user_name=r["sender_name"],
            is_me=bool(r["is_me"]),
            kind=r["kind"],
            text=r["text"],
            reply_to=r["reply_to"],
        ),
        bool(r["canned"]),
    )


def _pair(r: sqlite3.Row) -> Pair:
    return Pair(
        r["pair_id"],
        r["chat_id"],
        r["first_id"],
        r["last_id"],
        r["ctx_first_id"],
        r["ts"],
        r["n_reply"],
        r["n_ctx"],
        r["reply_chars"],
        bool(r["quoted"]),
        r["split"],
    )


class CorpusDB:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # -- messages -----------------------------------------------------------

    def replace_chat(
        self, chat_id: int, kind: str, title: str, source: str, rows: Iterable[Row]
    ) -> int:
        """Drop whatever was stored for this chat and load `rows` in its place."""
        self.db.execute("DELETE FROM msg WHERE chat_id = ?", (chat_id,))
        self.db.execute("DELETE FROM chat WHERE chat_id = ?", (chat_id,))
        count = 0
        batch: list[tuple] = []

        def flush() -> None:
            self.db.executemany("INSERT OR REPLACE INTO msg VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
            batch.clear()

        for row in rows:
            m = row.msg
            batch.append(
                (
                    chat_id,
                    m.message_id,
                    m.ts,
                    m.user_id,
                    m.user_name,
                    int(m.is_me),
                    m.kind,
                    m.text,
                    m.reply_to,
                    int(row.canned),
                )
            )
            count += 1
            if len(batch) >= 5000:
                flush()
        flush()
        self.db.execute(
            "INSERT INTO chat VALUES (?,?,?,?,?,?)", (chat_id, kind, title, count, 1, source)
        )
        self.db.commit()
        return count

    def mark_incomplete(self, chat_id: int) -> None:
        self.db.execute("UPDATE chat SET complete = 0 WHERE chat_id = ?", (chat_id,))
        self.db.commit()

    def chat_ids(self) -> list[int]:
        return [r[0] for r in self.db.execute("SELECT chat_id FROM chat ORDER BY chat_id")]

    def chat_title(self, chat_id: int) -> str:
        row = self.db.execute("SELECT title FROM chat WHERE chat_id = ?", (chat_id,)).fetchone()
        return row[0] if row else ""

    def rows(self, chat_id: int) -> Iterator[Row]:
        cursor = self.db.execute("SELECT * FROM msg WHERE chat_id = ? ORDER BY msg_id", (chat_id,))
        for r in cursor:
            yield _row(r)

    def span(self, chat_id: int, first_id: int, last_id: int) -> list[Msg]:
        """Messages with first_id <= id <= last_id, in order."""
        cursor = self.db.execute(
            "SELECT * FROM msg WHERE chat_id = ? AND msg_id BETWEEN ? AND ? ORDER BY msg_id",
            (chat_id, first_id, last_id),
        )
        return [_row(r).msg for r in cursor]

    def before(self, chat_id: int, msg_id: int, limit: int, max_age_s: int | None = None) -> list[Msg]:
        """The `limit` messages up to and including `msg_id`, in order."""
        rows = self.db.execute(
            "SELECT * FROM msg WHERE chat_id = ? AND msg_id <= ? ORDER BY msg_id DESC LIMIT ?",
            (chat_id, msg_id, limit),
        ).fetchall()
        msgs = [_row(r).msg for r in reversed(rows)]
        if max_age_s is not None and msgs:
            msgs = [m for m in msgs if msgs[-1].ts - m.ts <= max_age_s]
        return msgs

    def count(self, where: str = "1", params: tuple = ()) -> int:
        return self.db.execute(f"SELECT COUNT(*) FROM msg WHERE {where}", params).fetchone()[0]

    # -- pairs --------------------------------------------------------------

    def replace_pairs(self, pairs: Iterable[tuple]) -> int:
        self.db.execute("DELETE FROM pair")
        cursor = self.db.executemany(
            "INSERT INTO pair (chat_id, first_id, last_id, ctx_first_id, ts, n_reply, n_ctx,"
            " reply_chars, quoted) VALUES (?,?,?,?,?,?,?,?,?)",
            pairs,
        )
        self.db.commit()
        return cursor.rowcount

    def pairs(self, split: str | None = None, chat_id: int | None = None) -> Iterator[Pair]:
        clauses, params = [], []
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if chat_id is not None:
            clauses.append("chat_id = ?")
            params.append(chat_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        for r in self.db.execute(f"SELECT * FROM pair{where} ORDER BY ts, pair_id", params):
            yield _pair(r)

    def pair(self, pair_id: int) -> Pair | None:
        row = self.db.execute("SELECT * FROM pair WHERE pair_id = ?", (pair_id,)).fetchone()
        return _pair(row) if row else None

    def pair_counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT split, COUNT(*) FROM pair GROUP BY split").fetchall())

    def context_of(self, pair: Pair) -> list[Msg]:
        return self.span(pair.chat_id, pair.ctx_first_id, pair.first_id - 1)

    def reply_of(self, pair: Pair) -> list[Msg]:
        return self.span(pair.chat_id, pair.first_id, pair.last_id)
