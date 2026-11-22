"""Data for a fine-tune: what the live bot is shown at a moment, and what I really said.

Each moment goes through the same harness the bot runs, with a stand-in model that
only keeps the messages, so training sees the prompt the bot will see, word for word.
Three things are left out: the fixed sample (the weights stand in for it), recall
(it is built from these same pairs, so each would be handed its own answer) and
the facts (they would double the size of every example; the live bot still adds them).

Files, one JSON object a line:
  train.jsonl, valid.jsonl   {"id", "messages", "reply"}: my reply, one message a line
  eval.jsonl                 {"id", "messages"}: the held-out moments `eval run` samples
  eval_recall.jsonl          the same moments with recall, when there is a recall index
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from shinnku_bot.config import Settings
from shinnku_bot.corpus.anchor import select_anchor
from shinnku_bot.corpus.db import CorpusDB, Pair
from shinnku_bot.eval.run import replay, sample_pairs
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.recall import Recall
from shinnku_bot.llm import ModelReply, ToolCall

Messages = list[dict[str, Any]]


class BarePersona(Persona):
    """No fixed sample and no facts."""

    def anchor_text(self) -> str:
        return ""

    def facts_text(self) -> str:
        return ""


class _Capture:
    """Keeps the first prompt it is sent and says a placeholder line, so the harness goes no further."""

    def __init__(self) -> None:
        self.messages: Messages | None = None

    async def complete(self, messages: Messages, tools: Any, **options: Any) -> ModelReply:
        if self.messages is None:
            self.messages = messages
        return ModelReply(tool_calls=(ToolCall("say", {"messages": [{"text": "嗯"}]}),))


def _answerable(db: CorpusDB, pair: Pair) -> bool:
    context = db.context_of(pair)
    return bool(context) and not context[-1].is_me


async def prompt_of(
    settings: Settings,
    db: CorpusDB,
    pair: Pair,
    persona: Persona,
    recall: Recall | None,
    limits: StyleLimits,
) -> Messages | None:
    """What the bot would be sent at this moment; None when I spoke last and nobody is waiting."""
    if not _answerable(db, pair):
        return None
    capture = _Capture()
    await replay(settings, capture, persona, recall, limits, db.context_of(pair), "sft", db.chat_title(pair.chat_id))
    return capture.messages


def reply_text(db: CorpusDB, pair: Pair) -> str:
    """What I said, one message a line."""
    return "\n".join(m.text.strip() for m in db.reply_of(pair))


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


async def write_sft(
    settings: Settings,
    db: CorpusDB,
    recall: Recall | None,
    limits: StyleLimits,
    out_dir: Path,
    pairs: int,
    valid: int,
    n_eval: int,
    seed: int = 0,
) -> dict[str, int]:
    """Write the files above. Returns counts and sizes only.

    Training pairs are drawn like the fixed sample, weighted toward now; the
    held-out moments are the ones `eval run --n n_eval --seed seed` answers.
    """
    persona = BarePersona(settings.persona_dir)
    # Twice as many as needed: a moment that ends on my own line has nothing to answer.
    picked = [p for p in select_anchor(db, 2 * (pairs + valid), seed) if _answerable(db, p)]
    random.Random(seed).shuffle(picked)
    picked = picked[: pairs + valid]

    async def examples(chosen: list[Pair]) -> list[dict[str, Any]]:
        rows = []
        for pair in chosen:
            messages = await prompt_of(settings, db, pair, persona, None, limits)
            if messages is not None:
                rows.append({"id": pair.pair_id, "messages": messages, "reply": reply_text(db, pair)})
        return rows

    train = await examples(picked[valid:])
    held = await examples(picked[:valid])
    moments = sample_pairs(db, n_eval, seed)

    async def prompts(used: Recall | None) -> list[dict[str, Any]]:
        rows = []
        for pair in moments:
            messages = await prompt_of(settings, db, pair, persona, used, limits)
            if messages is not None:
                rows.append({"id": pair.pair_id, "messages": messages})
        return rows

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": _write(out_dir / "train.jsonl", train),
        "valid": _write(out_dir / "valid.jsonl", held),
        "eval": _write(out_dir / "eval.jsonl", await prompts(None)),
    }
    if recall is not None:
        counts["eval_recall"] = _write(out_dir / "eval_recall.jsonl", await prompts(recall))
    sizes = [sum(len(m["content"]) for m in row["messages"]) for row in train]
    counts["prompt_chars_mean"] = sum(sizes) // max(len(sizes), 1)
    counts["prompt_chars_max"] = max(sizes, default=0)
    counts["chars_total"] = sum(sizes) + sum(len(row["reply"]) for row in train)
    return counts
