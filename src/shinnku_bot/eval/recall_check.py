"""Offline check of recall: does it surface moments where I said something similar?

No model calls. For held-out moments it compares what I really said with what I
said in the recalled moments, against the same number of random moments.
"""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from shinnku_bot.corpus.db import CorpusDB, Pair
from shinnku_bot.eval.metrics import percentile
from shinnku_bot.harness.recall import Embedder, Recall
from shinnku_bot.harness.transcript import tail_text


def _said(db: CorpusDB, pair: Pair | None) -> str:
    return "\n".join(m.text.strip() for m in db.reply_of(pair)) if pair is not None else ""


def check_recall(
    db: CorpusDB,
    recall: Recall,
    embedder: Embedder,
    pairs: list[Pair],
    k: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Similarity of my real reply to the replies in recalled vs random moments."""
    rng = random.Random(seed)
    train_ids = [p.pair_id for p in db.pairs("train")]
    best_recalled: list[float] = []
    best_random: list[float] = []
    mean_recalled: list[float] = []
    mean_random: list[float] = []
    exact = 0
    same_chat: list[float] = []
    found: list[float] = []
    latency: list[float] = []

    for pair in pairs:
        started = time.monotonic()
        ids = recall.search(tail_text(db.context_of(pair)), pair.chat_id, k)
        latency.append(time.monotonic() - started)
        found.append(float(len(ids)))
        if not ids:
            continue
        real = _said(db, pair)
        recalled = [db.pair(i) for i in ids]
        recalled_said = [_said(db, p) for p in recalled]
        random_said = [_said(db, db.pair(i)) for i in rng.sample(train_ids, min(len(ids), len(train_ids)))]

        vectors = embedder.embed([real, *recalled_said, *random_said])
        sims = vectors[1:] @ vectors[0]
        near, far = sims[: len(recalled_said)], sims[len(recalled_said) :]
        best_recalled.append(float(near.max()))
        mean_recalled.append(float(near.mean()))
        best_random.append(float(far.max()))
        mean_random.append(float(far.mean()))
        exact += real in recalled_said
        same_chat.append(float(np.mean([p is not None and p.chat_id == pair.chat_id for p in recalled])))

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "moments": len(pairs),
        "k": k,
        "recalled_mean": mean(found),
        "best_similarity": {"recalled": mean(best_recalled), "random": mean(best_random)},
        "mean_similarity": {"recalled": mean(mean_recalled), "random": mean(mean_random)},
        # How often recall beats the same number of random moments on the best match.
        "wins": mean([float(a > b) for a, b in zip(best_recalled, best_random)]),
        "exact_reply_recalled": round(exact / len(pairs), 4) if pairs else 0.0,
        "same_chat_share": mean(same_chat),
        "search_ms_p50": round(percentile(latency, 50) * 1000, 1),
        "search_ms_p95": round(percentile(latency, 95) * 1000, 1),
    }


def format_check(result: dict[str, Any]) -> str:
    best, avg = result["best_similarity"], result["mean_similarity"]
    return "\n".join(
        [
            f"moments {result['moments']}  recalled per moment {result['recalled_mean']} (k={result['k']})",
            f"closest recalled reply vs my real reply: {best['recalled']}  (random moments: {best['random']})",
            f"average recalled reply vs my real reply: {avg['recalled']}  (random moments: {avg['random']})",
            f"recall beats random on the best match: {result['wins']:.1%}",
            f"my exact reply was among the recalled: {result['exact_reply_recalled']:.1%}",
            f"recalled from the same chat: {result['same_chat_share']:.1%}",
            f"search time p50/p95: {result['search_ms_p50']}ms/{result['search_ms_p95']}ms",
        ]
    )
