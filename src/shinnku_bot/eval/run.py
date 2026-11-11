"""Generate the bot's reply for held-out moments and have a judge compare."""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shinnku_bot.config import Settings
from shinnku_bot.corpus.db import CorpusDB, Pair
from shinnku_bot.eval.metrics import Item, summarize
from shinnku_bot.harness.engine import Harness, Outcome
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.recall import Recall
from shinnku_bot.harness.transcript import render_lines
from shinnku_bot.llm import ChatModel, ModelReply
from shinnku_bot.store import Msg, Store

JUDGE_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Say which candidate is the real reply.",
            "parameters": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
                "required": ["choice"],
            },
        },
    }
]
JUDGE_SYSTEM = (
    "你在做一个鉴别测试。下面先给出某个人（记作「我」）以前在群里聊天的一些真实片段，"
    "供你熟悉他的说话方式。然后给出一段新的群聊上文和两个候选回复：一个是他当时真实发出的，"
    "另一个是模仿者写的。请判断哪个是真实的，并调用 pick 给出 A 或 B。"
)
CONTEXT_LINES = 12
REFERENCE_SCENES = 20
GIVE_UP_AFTER = 8


@dataclass(frozen=True)
class EvalOptions:
    n: int = 40
    seed: int = 0
    concurrency: int = 4
    use_recall: bool = True
    use_anchor: bool = True
    # Off by default: model judges proved unreliable here and cost a lot of quota.
    judge: bool = False


class _NoAnchor(Persona):
    def anchor_text(self) -> str:
        return ""


class DraftPersona(Persona):
    """Reads the facts draft awaiting review instead of the live FACTS.md."""

    def facts_text(self) -> str:
        return self._read(self.dir / "FACTS.draft.md") or super().facts_text()


class _Recorder:
    """Passes calls through to the model and keeps the replies."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model
        self.replies: list[ModelReply] = []

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **options: Any
    ) -> ModelReply:
        reply = await self._model.complete(messages, tools, **options)
        self.replies.append(reply)
        return reply


class _Collector:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self._ids = itertools.count(10**12)

    async def typing(self, chat_id: int) -> None:
        return None

    async def send(self, chat_id: int, text: str, reply_to: int | None) -> int:
        self.texts.append(text)
        return next(self._ids)


async def _no_sleep(seconds: float) -> None:
    return None


def sample_pairs(db: CorpusDB, n: int, seed: int) -> list[Pair]:
    """Held-out pairs where someone else spoke last, spread evenly across chats."""
    rng = random.Random(seed)
    by_chat: dict[int, list[Pair]] = {}
    for pair in db.pairs("test"):
        context = db.context_of(pair)
        if context and not context[-1].is_me:
            by_chat.setdefault(pair.chat_id, []).append(pair)
    for pairs in by_chat.values():
        rng.shuffle(pairs)
    picked: list[Pair] = []
    while len(picked) < n and any(by_chat.values()):
        for chat_id in sorted(by_chat):
            if by_chat[chat_id] and len(picked) < n:
                picked.append(by_chat[chat_id].pop())
    return picked


async def replay(
    settings: Settings,
    model: ChatModel,
    persona: Persona,
    recall: Recall | None,
    limits: StyleLimits,
    context: Sequence[Msg],
    reason: str,
    title: str | None = None,
    is_private: bool = False,
) -> tuple[Outcome, list[str]]:
    """Answer the last message of `context` through the same harness the live bot uses.

    The clock stands at that message's time. Returns the outcome and what was sent.
    """
    store = Store(":memory:")
    outbox = _Collector()
    harness = Harness(
        dataclasses.replace(settings, trace=False),
        store,
        model,
        outbox,
        persona,
        bot_id=0,
        recall=recall,
        limits=limits,
        sleep=_no_sleep,
        now=lambda: float(context[-1].ts),
    )
    try:
        for msg in context[:-1]:
            store.record(msg)
        outcome = await harness.answer(context[-1], reason, title=title, is_private=is_private)
        return outcome, list(outbox.texts)
    finally:
        store.close()


async def gather_bounded[T, R](
    items: Sequence[T],
    work: Callable[[T], Awaitable[R]],
    failed: Callable[[R], bool],
    skipped: Callable[[T], R],
    concurrency: int = 4,
) -> list[R]:
    """Run `work` over `items`, a few at a time.

    Once GIVE_UP_AFTER have failed and nothing has worked (quota gone, bad key,
    gateway down), the rest are skipped instead of spending more calls.
    """
    gate = asyncio.Semaphore(concurrency)
    done = bad = 0

    async def one(item: T) -> R:
        nonlocal done, bad
        async with gate:
            if bad >= GIVE_UP_AFTER and bad == done:
                done += 1
                bad += 1
                return skipped(item)
            result = await work(item)
            done += 1
            bad += failed(result)
            return result

    return list(await asyncio.gather(*(one(item) for item in items)))


async def generate_one(
    settings: Settings,
    db: CorpusDB,
    pair: Pair,
    model: ChatModel,
    persona: Persona,
    recall: Recall | None,
    limits: StyleLimits,
) -> Item:
    """Replay one held-out moment through the same harness the live bot uses."""
    context = db.context_of(pair)
    item = Item(
        pair_id=pair.pair_id,
        chat_id=pair.chat_id,
        ts=pair.ts,
        context=render_lines(context)[-CONTEXT_LINES:],
        real=[m.text.strip() for m in db.reply_of(pair)],
    )
    recorder = _Recorder(model)
    try:
        outcome, item.bot = await replay(
            settings, recorder, persona, recall, limits, context, "eval", db.chat_title(pair.chat_id)
        )
        item.kind = outcome.kind
        item.violations = list(outcome.violations)
        item.attempts = outcome.attempts
        item.reasoning = outcome.reasoning
        item.latency_s = round(sum(r.latency_s for r in recorder.replies), 3)
        if recorder.replies:
            item.prompt_tokens = recorder.replies[-1].prompt_tokens
            item.cached_tokens = recorder.replies[-1].cached_tokens
        item.recalled = outcome.recalled
    except Exception as exc:  # noqa: BLE001 - one bad item must not end the run
        item.error = type(exc).__name__
    return item


def judge_messages(reference: str, item: Item, bot_first: bool) -> list[dict[str, str]]:
    first, second = (item.bot, item.real) if bot_first else (item.real, item.bot)
    user = "\n".join(
        [
            "【上文】",
            *item.context,
            "",
            "【候选 A】",
            *(f"我: {text}" for text in first),
            "",
            "【候选 B】",
            *(f"我: {text}" for text in second),
        ]
    )
    system = JUDGE_SYSTEM + ("\n\n【他以前的真实片段】\n\n" + reference if reference else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_pick(reply: ModelReply) -> str:
    """'A', 'B' or '' from the judge's tool call, falling back to its text."""
    for call in reply.tool_calls:
        choice = str(call.args.get("choice", "")).strip().upper()
        if choice in ("A", "B"):
            return choice
    text = reply.content.strip().upper()
    for marker in ("A", "B"):
        if text.startswith(marker) or f"候选 {marker}" in text or f"候选{marker}" in text:
            return marker
    return ""


async def judge_one(judge: ChatModel, reference: str, item: Item, rng: random.Random) -> None:
    """Ask the judge which reply is real; candidate order is random per item."""
    if not item.bot or item.error:
        return
    bot_first = rng.random() < 0.5
    try:
        pick = read_pick(await judge.complete(judge_messages(reference, item, bot_first), JUDGE_TOOL))
    except Exception:  # noqa: BLE001
        return
    if pick:
        picked_bot = (pick == "A") == bot_first
        item.judged = "bot" if picked_bot else "real"


def reference_scenes(persona: Persona, count: int = REFERENCE_SCENES) -> str:
    """The most recent few scenes of the fixed sample, to show the judge my style."""
    scenes = [scene for scene in persona.anchor_text().split("\n\n") if scene.strip()]
    return "\n\n".join(scenes[-count:]) if count > 0 else ""


async def run_eval(
    settings: Settings,
    db: CorpusDB,
    model: ChatModel,
    judge: ChatModel | None,
    persona: Persona,
    recall: Recall | None,
    limits: StyleLimits,
    options: EvalOptions,
    progress: Callable[[str], None] = lambda line: None,
) -> list[Item]:
    pairs = sample_pairs(db, options.n, options.seed)
    reference = reference_scenes(persona)
    used_persona = persona if options.use_anchor else _NoAnchor(settings.persona_dir)
    used_recall = recall if options.use_recall else None
    gate = asyncio.Semaphore(options.concurrency)
    rng = random.Random(options.seed)
    orders = {pair.pair_id: random.Random(rng.random()) for pair in pairs}
    started = time.monotonic()
    done = 0

    failed = 0

    async def one(pair: Pair) -> Item:
        nonlocal done, failed
        async with gate:
            if failed >= GIVE_UP_AFTER and failed == done:
                # Nothing has worked yet (quota, bad key, gateway down): stop spending calls.
                done += 1
                failed += 1
                return Item(pair.pair_id, pair.chat_id, pair.ts, error="Skipped")
            item = await generate_one(settings, db, pair, model, used_persona, used_recall, limits)
            failed += bool(item.error)
            if judge is not None and options.judge:
                await judge_one(judge, reference, item, orders[pair.pair_id])
            done += 1
            if done % 20 == 0 or done == len(pairs):
                progress(f"{done}/{len(pairs)} in {time.monotonic() - started:.0f}s")
            return item

    return list(await asyncio.gather(*(one(pair) for pair in pairs)))


def add_similarity(items: list[Item], embedder: Any) -> None:
    """Score each reply against my real one and, as the chance level, the next item's."""
    spoke = [item for item in items if item.bot and not item.error]
    if len(spoke) < 2:
        return
    bot = embedder.embed(["\n".join(item.bot) for item in spoke])
    real = embedder.embed(["\n".join(item.real) for item in spoke])
    for index, item in enumerate(spoke):
        item.sim_real = round(float(bot[index] @ real[index]), 4)
        item.sim_other = round(float(bot[index] @ real[(index + 1) % len(spoke)]), 4)


def load_items(path: Path) -> list[Item]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Item(**entry) for entry in report["items"]]


async def rejudge(
    items: list[Item], judge: ChatModel, reference: str, seed: int = 0, concurrency: int = 6
) -> None:
    """Judge already generated replies again, e.g. with a different judge model."""
    gate = asyncio.Semaphore(concurrency)
    rng = random.Random(seed)
    orders = [random.Random(rng.random()) for _ in items]

    async def one(item: Item, order: random.Random) -> None:
        item.judged = ""
        async with gate:
            await judge_one(judge, reference, item, order)

    await asyncio.gather(*(one(item, order) for item, order in zip(items, orders)))


def save_report(path: Path, name: str, options: EvalOptions, settings: Settings, items: list[Item]) -> None:
    """Write the full run, texts included. This file is private."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "name": name,
        "options": dataclasses.asdict(options),
        "model": settings.llm_model,
        "thinking": settings.llm_thinking,
        "summary": summarize(items),
        "items": [item.to_json() for item in items],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
