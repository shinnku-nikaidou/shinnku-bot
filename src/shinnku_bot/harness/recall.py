"""Remember moments like this one: hybrid search over my past replies."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.harness.persona import export_chat_id
from shinnku_bot.harness.transcript import render_scene, tail_text
from shinnku_bot.store import Msg

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
CANDIDATES = 200
# Similarity decides; the rest only reorders moments that are about equally close.
FTS_WEIGHT = 0.15
RECENCY_FLOOR = 0.92
SAME_CHAT_BONUS = 1.04
MAX_SAME_REPLY = 2
MAX_TRIGRAMS = 48


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-length float32 vectors, one row per text."""
        ...


class FastEmbedder:
    """A small local embedding model running on CPU."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding

        self.name = model
        self._model = TextEmbedding(model, cache_dir=str(cache_dir) if cache_dir else None)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(list(self._model.embed(list(texts), batch_size=256)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


def fts_query(text: str) -> str | None:
    """An OR of the text's trigrams, most recent first, for the trigram FTS index."""
    grams: list[str] = []
    seen: set[str] = set()
    for line in reversed(text.splitlines()):
        for word in line.split():
            for i in range(len(word) - 2):
                gram = word[i : i + 3]
                if gram not in seen:
                    seen.add(gram)
                    grams.append(gram)
    if not grams:
        return None
    return " OR ".join('"' + gram.replace('"', '""') + '"' for gram in grams[:MAX_TRIGRAMS])


class Recall:
    """Search the index built by `shinnku corpus index`."""

    def __init__(self, corpus_db: Path, index_dir: Path, embedder: Embedder) -> None:
        index_dir = Path(index_dir)
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if self.meta["model"] != embedder.name:
            raise ValueError("index was built with a different embedding model")
        self._vectors = np.load(index_dir / "tail.npy", mmap_mode="r")
        self._ids = np.load(index_dir / "tail_ids.npy")
        self._chats = np.load(index_dir / "tail_chats.npy")
        ts = np.load(index_dir / "tail_ts.npy").astype(np.float64)
        span = max(float(ts.max() - ts.min()), 1.0) if len(ts) else 1.0
        # Newer memories count a little more: the floor for the oldest, 1.0 for the newest.
        age = (ts - (ts.min() if len(ts) else 0)) / span
        self._recency = RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * age
        self._row = {int(pair_id): row for row, pair_id in enumerate(self._ids)}
        self._embedder = embedder
        self._db = CorpusDB(corpus_db)

    def __len__(self) -> int:
        return len(self._ids)

    def close(self) -> None:
        self._db.close()

    def search(self, tail: str, chat_id: int | None, k: int) -> list[int]:
        """Pair ids of the `k` most similar remembered moments, best first."""
        if not tail.strip() or not len(self):
            return []
        similarity = self._vectors @ self._embedder.embed([tail])[0]
        top = min(CANDIDATES, len(similarity))
        rows = {int(row) for row in np.argpartition(-similarity, top - 1)[:top]}

        # Shared wording, scaled so the best keyword match adds FTS_WEIGHT.
        wording: dict[int, float] = {}
        query = fts_query(tail)
        if query is not None:
            try:
                hits = self._db.db.execute(
                    "SELECT rowid, bm25(pair_fts) FROM pair_fts WHERE pair_fts MATCH ?"
                    " ORDER BY bm25(pair_fts) LIMIT ?",
                    (query, CANDIDATES),
                ).fetchall()
            except sqlite3.OperationalError:
                hits = []
            best = min((score for _, score in hits), default=0.0)  # bm25 is negative; lower is better
            for pair_id, score in hits:
                row = self._row.get(pair_id)
                if row is not None and best < 0:
                    wording[row] = FTS_WEIGHT * score / best
                    rows.add(row)

        here = export_chat_id(chat_id) if chat_id is not None else None
        scores: dict[int, float] = {}
        for row in rows:
            score = (float(similarity[row]) + wording.get(row, 0.0)) * float(self._recency[row])
            if here is not None and int(self._chats[row]) == here:
                score *= SAME_CHAT_BONUS
            scores[int(self._ids[row])] = score

        picked: list[int] = []
        replies: dict[str, int] = {}
        for pair_id in sorted(scores, key=lambda p: -scores[p]):
            pair = self._db.pair(pair_id)
            if pair is None:
                continue
            said = "\n".join(m.text for m in self._db.reply_of(pair))
            if replies.get(said, 0) >= MAX_SAME_REPLY:
                continue  # twenty scenes that all end in the same word teach nothing
            replies[said] = replies.get(said, 0) + 1
            picked.append(pair_id)
            if len(picked) >= k:
                break
        return picked

    def render(self, pair_ids: Sequence[int]) -> str:
        scenes = []
        for pair_id in pair_ids:
            pair = self._db.pair(pair_id)
            if pair is not None:
                scenes.append(render_scene(self._db.context_of(pair), self._db.reply_of(pair)))
        return "\n\n".join(scenes)

    def scenes(self, msgs: Sequence[Msg], chat_id: int | None, k: int) -> str:
        """Rendered moments similar to the end of this conversation ('' if none)."""
        return self.render(self.search(tail_text(msgs), chat_id, k))


def load_recall(corpus_db: Path, index_dir: Path, model: str, cache_dir: Path) -> Recall | None:
    """The recall index if it has been built, otherwise None."""
    if not (Path(index_dir) / "meta.json").is_file() or not Path(corpus_db).is_file():
        return None
    try:
        return Recall(corpus_db, index_dir, FastEmbedder(model, cache_dir))
    except Exception as exc:  # noqa: BLE001 - the bot must still start without memories
        log.warning("recall unavailable error=%s", type(exc).__name__)
        return None
