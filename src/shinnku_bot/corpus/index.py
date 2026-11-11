"""Build the recall index over the training pairs: trigram FTS plus vectors."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.harness.recall import Embedder
from shinnku_bot.harness.transcript import tail_text

BATCH = 2048


def _digest(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8")).digest()


def _previous_vectors(
    db: CorpusDB, index_dir: Path, model: str
) -> tuple[dict[bytes, int], np.ndarray | None]:
    """Vectors from the last build, keyed by the text they were computed from.

    Pair ids change whenever pairs are re-cut, but most of the text does not,
    so a rebuild after more history arrives only has to embed what is new.
    """
    try:
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if meta.get("model") != model:
            return {}, None
        ids = np.load(index_dir / "tail_ids.npy")
        vectors = np.load(index_dir / "tail.npy", mmap_mode="r")
        row_of = {int(pair_id): row for row, pair_id in enumerate(ids)}
        known = {
            _digest(tail): row_of[rowid]
            for rowid, tail in db.db.execute("SELECT rowid, tail FROM pair_fts")
            if rowid in row_of
        }
        return known, vectors
    except (OSError, ValueError, KeyError, sqlite3.OperationalError):
        return {}, None


def build_index(
    db: CorpusDB,
    index_dir: Path,
    embedder: Embedder,
    progress: Callable[[str], None] = lambda line: None,
) -> dict[str, object]:
    """Index what was being said just before each of my training replies.

    Held-out pairs are never indexed. Returns the metadata that is also written
    to `index_dir/meta.json`.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    known, old_vectors = _previous_vectors(db, index_dir, embedder.name)

    db.db.execute("DROP TABLE IF EXISTS pair_fts")
    db.db.execute("CREATE VIRTUAL TABLE pair_fts USING fts5(tail, tokenize='trigram')")
    ids: list[int] = []
    chats: list[int] = []
    stamps: list[int] = []
    tails: list[str] = []
    for pair in db.pairs("train"):
        tail = tail_text(db.context_of(pair))
        if not tail.strip():
            continue
        ids.append(pair.pair_id)
        chats.append(pair.chat_id)
        stamps.append(pair.ts)
        tails.append(tail)
    db.db.executemany("INSERT INTO pair_fts (rowid, tail) VALUES (?, ?)", zip(ids, tails))
    db.db.commit()
    progress(f"indexed text for {len(ids)} pairs in {time.monotonic() - started:.0f}s")

    dim = int(embedder.embed(["维度"]).shape[1])
    if old_vectors is not None and old_vectors.shape[1] != dim:
        known, old_vectors = {}, None

    # Written beside the live file and swapped in at the end, so a running bot
    # keeps reading the old index until the new one is complete.
    fresh_path = index_dir / "tail.new.npy"
    vectors = np.lib.format.open_memmap(fresh_path, mode="w+", dtype=np.float32, shape=(len(ids), dim))
    missing: list[int] = []
    for row, tail in enumerate(tails):
        old_row = known.get(_digest(tail))
        if old_row is None:
            missing.append(row)
        else:
            vectors[row] = old_vectors[old_row]  # type: ignore[index]
    progress(f"reused {len(tails) - len(missing)} vectors, {len(missing)} to embed")

    for start in range(0, len(missing), BATCH):
        rows = missing[start : start + BATCH]
        vectors[rows] = embedder.embed([tails[row] for row in rows])
        done = start + len(rows)
        if done == len(missing) or (start // BATCH) % 5 == 4:
            progress(f"embedded {done}/{len(missing)} in {time.monotonic() - started:.0f}s")
    vectors.flush()
    del vectors, old_vectors
    os.replace(fresh_path, index_dir / "tail.npy")

    np.save(index_dir / "tail_ids.npy", np.asarray(ids, dtype=np.int64))
    np.save(index_dir / "tail_chats.npy", np.asarray(chats, dtype=np.int64))
    np.save(index_dir / "tail_ts.npy", np.asarray(stamps, dtype=np.int64))
    meta = {
        "model": embedder.name,
        "dim": dim,
        "count": len(ids),
        "embedded": len(missing),
        "built": int(time.time()),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta
