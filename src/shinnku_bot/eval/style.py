"""Can a classifier tell the bot's replies from mine?

It is trained to separate what I really said from what the bot said in the same
moments. Both sides answer the same contexts, so topic is balanced and only the
way of writing is left to tell them apart. 0.5 means it cannot; 1.0 means always.

Three reference lines say how to read the number: me against myself (should be
0.5), me against the other people in my chats, and me against the bare model.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.pairs import train_cutoffs
from shinnku_bot.eval.classify import (
    NGRAM_DIM,
    STAT_NAMES,
    LogReg,
    auc,
    auc_interval,
    bucket,
    cross_val_scores,
    featurize,
    ngrams,
    text_stats,
)
from shinnku_bot.eval.metrics import Item

Reply = Sequence[str]
BURST_GAP_S = 120
MIN_SAMPLES = 20


def bundle(samples: Sequence[Reply], size: int, order: Sequence[int]) -> list[list[str]]:
    """Join `size` replies into one sample, taking them in `order`.

    One short reply carries little style; a few together carry more.
    """
    if size <= 1:
        return [list(samples[i]) for i in order]
    out = []
    for start in range(0, len(order) - size + 1, size):
        out.append([m for i in order[start : start + size] for m in samples[i]])
    return out


def tell_apart(
    mine: Sequence[Reply], theirs: Sequence[Reply], paired: bool = False, seed: int = 0
) -> dict[str, Any]:
    """Cross-validated AUC for separating `theirs` from `mine`, with a 95% interval.

    With `paired`, mine[i] and theirs[i] answer the same moment and are always
    kept on the same side of the train/test split.
    """
    n = min(len(mine), len(theirs)) if paired else len(mine) + len(theirs)
    if min(len(mine), len(theirs)) < MIN_SAMPLES:
        return {"auc": None, "low": None, "high": None, "mine": len(mine), "theirs": len(theirs)}
    if paired:
        mine, theirs = mine[:n], theirs[:n]
        groups = np.concatenate([np.arange(n), np.arange(n)])
    else:
        groups = np.arange(n)
    x = featurize([*mine, *theirs])
    y = np.concatenate([np.zeros(len(mine)), np.ones(len(theirs))])
    scores = cross_val_scores(x, y, groups, seed=seed, scaled=len(STAT_NAMES))
    low, high = auc_interval(scores, y, groups, seed=seed)
    return {
        "auc": round(auc(scores, y), 3),
        "low": round(low, 3),
        "high": round(high, 3),
        "mine": len(mine),
        "theirs": len(theirs),
    }


def my_test_replies(db: CorpusDB) -> list[list[str]]:
    return [[m.text.strip() for m in db.reply_of(pair)] for pair in db.pairs("test")]


def others_bursts(db: CorpusDB, n: int, seed: int = 0) -> list[list[str]]:
    """Bursts by other people in my chats, from the held-out period."""
    rng = random.Random(seed)
    cutoffs = train_cutoffs(db)
    by_chat: dict[int, list[list[str]]] = {}
    for chat_id, cutoff in cutoffs.items():
        if cutoff >= 2**62:
            continue
        bursts: list[list[str]] = []
        current: list[str] = []
        sender: int | None = None
        last_ts = 0
        rows = db.db.execute(
            "SELECT sender_id, ts, is_me, kind, text, canned FROM msg"
            " WHERE chat_id = ? AND ts >= ? ORDER BY msg_id",
            (chat_id, cutoff),
        )
        for sender_id, ts, is_me, kind, text, canned in rows:
            usable = (
                not is_me and sender_id is not None and kind == "text" and not canned and text.strip()
            )
            if not usable or sender_id != sender or ts - last_ts > BURST_GAP_S:
                if current:
                    bursts.append(current)
                current = []
            if usable:
                current.append(text.strip())
                sender, last_ts = sender_id, ts
            else:
                sender = None
        if current:
            bursts.append(current)
        rng.shuffle(bursts)
        by_chat[chat_id] = bursts
    picked: list[list[str]] = []
    while len(picked) < n and any(by_chat.values()):
        for chat_id in sorted(by_chat):
            if by_chat[chat_id] and len(picked) < n:
                picked.append(by_chat[chat_id].pop())
    return picked


def stat_gaps(mine: Sequence[Reply], theirs: Sequence[Reply]) -> list[dict[str, Any]]:
    """Each reply statistic averaged over my replies and over theirs."""
    if not mine or not theirs:
        return []
    me = np.mean([text_stats(reply) for reply in mine], axis=0)
    them = np.mean([text_stats(reply) for reply in theirs], axis=0)
    return [
        {"name": name, "me": round(float(a), 4), "bot": round(float(b), 4)}
        for name, a, b in zip(STAT_NAMES, me, them)
    ]


def gram_tells(mine: Sequence[Reply], theirs: Sequence[Reply], top: int = 40) -> dict[str, list[str]]:
    """The character sequences that most give each side away. Contains chat text."""
    if min(len(mine), len(theirs)) < MIN_SAMPLES:
        return {"bot": [], "me": []}
    x = featurize([*mine, *theirs])
    y = np.concatenate([np.zeros(len(mine)), np.ones(len(theirs))])
    weights = LogReg(scaled=len(STAT_NAMES)).fit(x, y).weights[:NGRAM_DIM]
    seen: dict[int, Counter[str]] = {}
    for reply in [*mine, *theirs]:
        for gram in ngrams(reply):
            seen.setdefault(bucket(gram), Counter())[gram] += 1

    def name(index: int) -> str:
        grams = [gram for gram, _ in seen.get(index, Counter()).most_common(3)]
        return " | ".join(grams)

    order = np.argsort(weights)
    return {
        "bot": [name(int(i)) for i in order[::-1][:top] if weights[i] > 0],
        "me": [name(int(i)) for i in order[:top] if weights[i] < 0],
    }


def spoken(items: Sequence[Item]) -> tuple[list[list[str]], list[list[str]]]:
    """(my real replies, the bot's replies) for the moments where the bot spoke."""
    kept = [item for item in items if item.bot and item.real and not item.error]
    return [list(item.real) for item in kept], [list(item.bot) for item in kept]


def style_report(
    items: Sequence[Item],
    db: CorpusDB | None,
    bare_items: Sequence[Item] | None = None,
    size: int = 1,
    others: int = 400,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """(numbers that are safe to print, character sequences that are not)."""
    rng = random.Random(seed)
    real, bot = spoken(items)
    order = list(range(len(real)))
    rng.shuffle(order)
    report: dict[str, Any] = {
        "bundle": size,
        "bot": tell_apart(bundle(real, size, order), bundle(bot, size, order), paired=True, seed=seed),
        "stats": stat_gaps(real, bot),
    }

    mine = my_test_replies(db) if db is not None else real
    shuffled = list(range(len(mine)))
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    report["self"] = tell_apart(
        bundle(mine, size, shuffled[:half]), bundle(mine, size, shuffled[half:]), seed=seed
    )
    if db is not None:
        theirs = others_bursts(db, others, seed)
        report["others"] = tell_apart(
            bundle(mine, size, shuffled), bundle(theirs, size, range(len(theirs))), seed=seed
        )
    if bare_items is not None:
        bare_real, bare_bot = spoken(bare_items)
        bare_order = list(range(len(bare_real)))
        rng.shuffle(bare_order)
        report["bare"] = tell_apart(
            bundle(bare_real, size, bare_order), bundle(bare_bot, size, bare_order), paired=True, seed=seed
        )
    return report, gram_tells(real, bot)


def _line(label: str, result: dict[str, Any] | None) -> str:
    if result is None:
        return f"{label}: not measured"
    if result["auc"] is None:
        return f"{label}: too few samples ({result['mine']} vs {result['theirs']})"
    return (
        f"{label}: AUC {result['auc']:.3f} ({result['low']:.3f}-{result['high']:.3f})"
        f"  n={result['mine']}+{result['theirs']}"
    )


def format_style(name: str, report: dict[str, Any]) -> str:
    lines = [
        f"== style {name} (replies per sample: {report['bundle']}) ==",
        _line("bot vs me        ", report["bot"]),
        _line("me vs me (floor) ", report.get("self")),
        _line("others vs me     ", report.get("others")),
        _line("bare model vs me ", report.get("bare")),
        "statistic                 me      bot",
    ]
    lines += [f"  {s['name']:<22}{s['me']:>8.3f}{s['bot']:>9.3f}" for s in report["stats"]]
    return "\n".join(lines)
