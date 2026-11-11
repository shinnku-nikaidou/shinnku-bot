"""Pick the fixed sample of my real conversations that opens every prompt."""

from __future__ import annotations

import random
from pathlib import Path

from shinnku_bot.corpus.db import CorpusDB, Pair
from shinnku_bot.harness.transcript import render_scene

BUCKETS = 12
# Only walls of text are left out. A lower cap once hid the fact that I also write at length.
MAX_REPLY_CHARS = 300


def select_anchor(db: CorpusDB, count: int, seed: int = 0) -> list[Pair]:
    """A time-stratified sample of training pairs, weighted heavily toward now.

    How I write has drifted over the years, so the newest third of my history
    supplies about two thirds of the sample. Replies are not repeated, so the
    sample shows range rather than my single most common word many times over.
    """
    pairs = [
        p for p in db.pairs("train") if p.n_ctx >= 2 and 0 < p.reply_chars <= MAX_REPLY_CHARS
    ]
    if not pairs or count <= 0:
        return []
    rng = random.Random(seed)
    # Equal slices of my history, oldest first. Slicing by index (not by a fixed
    # size) avoids a tiny leftover slice that would get the largest share.
    parts = min(BUCKETS, len(pairs))
    bounds = [i * len(pairs) // parts for i in range(parts + 1)]
    buckets = [pairs[bounds[i] : bounds[i + 1]] for i in range(parts)]
    weights = [(i + 1) ** 2 for i in range(parts)]
    total = sum(weights)

    seen: set[str] = set()
    chosen: list[Pair] = []
    running = 0.0
    planned = 0
    owed = 0  # what an earlier slice could not supply is taken from the next
    for bucket, weight in zip(buckets, weights):
        running += weight
        target = round(count * running / total)
        quota = target - planned + owed
        planned = target
        for pair in rng.sample(bucket, len(bucket)):
            if quota <= 0:
                break
            said = "\n".join(m.text for m in db.reply_of(pair))
            if said in seen:
                continue
            seen.add(said)
            chosen.append(pair)
            quota -= 1
        owed = quota

    if len(chosen) < count:
        # The newest slice ran out of new things I said; top up from the most recent.
        picked = {pair.pair_id for pair in chosen}
        for pair in reversed(pairs):
            if len(chosen) >= count:
                break
            if pair.pair_id in picked:
                continue
            said = "\n".join(m.text for m in db.reply_of(pair))
            if said not in seen:
                seen.add(said)
                chosen.append(pair)
    chosen.sort(key=lambda p: (p.ts, p.pair_id))
    return chosen


def render_anchor(db: CorpusDB, pairs: list[Pair]) -> str:
    return "\n\n".join(render_scene(db.context_of(p), db.reply_of(p)) for p in pairs)


def write_anchor(db: CorpusDB, path: Path, count: int, seed: int = 0) -> tuple[int, int]:
    """Write the anchor file. Returns (scenes, characters)."""
    pairs = select_anchor(db, count, seed)
    text = render_anchor(db, pairs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return len(pairs), len(text)
