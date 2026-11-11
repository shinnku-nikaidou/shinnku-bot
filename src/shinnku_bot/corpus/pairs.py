"""Cut the corpus into (what was being said -> what I said next) pairs."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from shinnku_bot.corpus.db import CorpusDB, Row


@dataclass(frozen=True)
class PairParams:
    # A silence this long ends a conversation; context never crosses it.
    session_gap_s: int = 30 * 60
    # My consecutive lines closer than this are one burst.
    burst_gap_s: int = 120
    # The same window the live bot sees.
    context_messages: int = 30
    context_max_age_s: int = 6 * 3600


def _usable(row: Row) -> bool:
    """Something I typed as conversation, as opposed to forwards, commands, media."""
    return row.msg.kind == "text" and bool(row.msg.text.strip()) and not row.canned


def cut_pairs(db: CorpusDB, chat_id: int, params: PairParams = PairParams()) -> Iterator[tuple]:
    """Yield pair rows for one chat, in the column order `replace_pairs` expects.

    A pair is one burst of mine whose every line is usable, preceded in the same
    conversation by at least one line from someone else.
    """
    window: deque[Row] = deque(maxlen=params.context_messages)
    burst: list[Row] = []

    def flush() -> Iterator[tuple]:
        if not burst:
            return
        first = burst[0].msg
        while window and first.ts - window[0].msg.ts > params.context_max_age_s:
            window.popleft()
        if all(_usable(row) for row in burst) and any(not row.msg.is_me for row in window):
            yield (
                chat_id,
                first.message_id,
                burst[-1].msg.message_id,
                window[0].msg.message_id,
                first.ts,
                len(burst),
                len(window),
                sum(len(row.msg.text) for row in burst),
                int(first.reply_to is not None),
            )
        window.extend(burst)
        burst.clear()

    last_ts: int | None = None
    for row in db.rows(chat_id):
        ts = row.msg.ts
        if last_ts is not None and ts - last_ts > params.session_gap_s:
            yield from flush()
            window.clear()
        last_ts = ts
        if row.msg.is_me:
            if burst and ts - burst[-1].msg.ts > params.burst_gap_s:
                yield from flush()
            burst.append(row)
        else:
            yield from flush()
            window.append(row)
    yield from flush()


def build_pairs(db: CorpusDB, params: PairParams = PairParams()) -> dict[int, int]:
    """Rebuild the pair table for every chat. Returns pairs per chat."""

    def every() -> Iterator[tuple]:
        for chat_id in db.chat_ids():
            yield from cut_pairs(db, chat_id, params)

    db.replace_pairs(every())
    return dict(db.db.execute("SELECT chat_id, COUNT(*) FROM pair GROUP BY chat_id").fetchall())


def assign_holdout(db: CorpusDB, per_chat: int = 300, min_train: int = 1000) -> dict[str, int]:
    """Hold out the most recent pairs of each chat as the test set.

    The split is by time, so nothing that came after a test pair in its own chat
    is ever used to build indexes, samples or the persona. Chats too small to
    leave `min_train` training pairs contribute no test pairs.
    """
    db.db.execute("UPDATE pair SET split = 'train'")
    for chat_id in db.chat_ids():
        total = db.db.execute("SELECT COUNT(*) FROM pair WHERE chat_id = ?", (chat_id,)).fetchone()[0]
        if total - per_chat < min_train:
            continue
        cutoff = db.db.execute(
            "SELECT ts FROM pair WHERE chat_id = ? ORDER BY ts DESC, pair_id DESC LIMIT 1 OFFSET ?",
            (chat_id, per_chat - 1),
        ).fetchone()[0]
        db.db.execute("UPDATE pair SET split = 'test' WHERE chat_id = ? AND ts >= ?", (chat_id, cutoff))
    db.db.commit()
    return db.pair_counts()


def train_cutoffs(db: CorpusDB) -> dict[int, int]:
    """For each chat, the time of its first test pair (or a far-future time)."""
    cutoffs = {chat_id: 2**62 for chat_id in db.chat_ids()}
    rows = db.db.execute("SELECT chat_id, MIN(ts) FROM pair WHERE split = 'test' GROUP BY chat_id")
    cutoffs.update(dict(rows.fetchall()))
    return cutoffs
