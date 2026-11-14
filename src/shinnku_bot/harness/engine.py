"""The harness: from an incoming message to what I send back."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from shinnku_bot.config import Settings
from shinnku_bot.harness.act import TOOLS, Say, parse_action
from shinnku_bot.harness.attention import blocked
from shinnku_bot.harness.hooks import RETRY_NOTES, StyleLimits, check
from shinnku_bot.harness.look import Fetch, look
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.prompt import build_messages, render_around, scene_header
from shinnku_bot.harness.recall import Recall
from shinnku_bot.harness.send import Outbox, Pace, send_burst
from shinnku_bot.harness.transcript import clean_name, render_turns, tail_text
from shinnku_bot.harness.urge import Gate
from shinnku_bot.llm import ChatModel, ModelReply
from shinnku_bot.store import Msg, Store

log = logging.getLogger(__name__)

# The reason recorded when nobody called me and I am only considering joining in.
URGE = "urge"
# With more than this many newer lines from others, my answer quotes what it answers.
QUOTE_AFTER = 4


@dataclass(frozen=True)
class Outcome:
    """What happened to one wake-up. `kind` is said, silent or nothing."""

    kind: str
    messages: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    reasoning: str = ""
    attempts: int = 0
    recalled: int = 0


@dataclass
class _Chat:
    title: str | None = None
    is_private: bool = False
    trigger: Msg | None = None
    reason: str = ""
    last_activity: float = 0.0
    task: asyncio.Task | None = None
    generating: bool = False
    # Set once I start sending; from then on a new message cannot cancel me.
    committed: bool = False
    restarts: int = 0
    dirty: bool = False


class Harness:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        model: ChatModel,
        outbox: Outbox,
        persona: Persona,
        *,
        bot_id: int,
        bot_username: str | None = None,
        recall: Recall | None = None,
        gate: Gate | None = None,
        limits: StyleLimits = StyleLimits(),
        pace: Pace = Pace(),
        # Downloads a picture; without it pictures stay placeholders.
        fetch: Fetch | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.store = store
        self.model = model
        self.outbox = outbox
        self.persona = persona
        self.bot_id = bot_id
        self.bot_username = bot_username
        self.recall = recall
        self.gate = gate
        self.limits = limits
        self.pace = pace
        self.fetch = fetch
        self._sleep = sleep
        self._rng = rng
        self._now = now
        self._tz = ZoneInfo(settings.timezone)
        self._chats: dict[int, _Chat] = {}

    # -- perceive ---------------------------------------------------------

    def handle(
        self,
        msg: Msg,
        reason: str | None,
        *,
        title: str | None = None,
        is_private: bool = False,
    ) -> None:
        """Record a message and, if it is aimed at me, schedule a debounced answer.

        Must be called from the running event loop.
        """
        self.store.record(msg)
        chat = self._chats.setdefault(msg.chat_id, _Chat())
        chat.title, chat.is_private = title, is_private
        clock = asyncio.get_running_loop().time()
        active = chat.task is not None and not chat.task.done()
        if self.gate is not None and msg.user_id == self.settings.owner_id:
            self.gate.owner_spoke(msg.chat_id, self._now())

        if reason is None:
            if active:
                # Another line from the person I am about to answer: wait for them.
                # While I am only considering joining in, anyone's line restarts the wait.
                if chat.reason == URGE:
                    if not chat.generating:
                        chat.trigger, chat.last_activity = msg, clock
                elif chat.trigger is not None and msg.user_id == chat.trigger.user_id:
                    chat.last_activity = clock
                    self._interrupt(msg.chat_id, chat)
                return
            if self.gate is None or is_private or msg.is_bot or msg.is_me:
                return
            state = self.store.chat_state(msg.chat_id)
            if state.talk_level <= 0 or blocked(
                self.settings, state, msg.chat_id, is_private, self._now()
            ):
                return
            chat.trigger, chat.reason, chat.last_activity = msg, URGE, clock
            chat.restarts = 0
            chat.dirty = False
            chat.task = asyncio.get_running_loop().create_task(self._run(msg.chat_id, chat))
            return

        deny = blocked(
            self.settings, self.store.chat_state(msg.chat_id), msg.chat_id, is_private, self._now()
        )
        if deny is not None:
            log.info("skip chat=%s why=%s", msg.chat_id, deny)
            return

        chat.trigger, chat.reason, chat.last_activity = msg, reason, clock
        if active:
            self._interrupt(msg.chat_id, chat)
        else:
            chat.restarts = 0
            chat.dirty = False
            chat.task = asyncio.get_running_loop().create_task(self._run(msg.chat_id, chat))

    def _interrupt(self, chat_id: int, chat: _Chat) -> None:
        """A new message arrived while a run is in flight."""
        if not chat.generating:
            return  # Still waiting out the quiet window; the loop sees last_activity.
        if chat.committed or chat.restarts >= 1:
            chat.dirty = True  # Finish this answer, then look again.
            return
        chat.restarts += 1
        assert chat.task is not None
        chat.task.cancel()
        chat.generating = False
        chat.task = asyncio.get_running_loop().create_task(self._run(chat_id, chat))

    async def _run(self, chat_id: int, chat: _Chat) -> None:
        me = asyncio.current_task()
        loop = asyncio.get_running_loop()
        try:
            started = loop.time()
            while True:
                clock = loop.time()
                wait = min(
                    self.settings.quiet_window_s - (clock - chat.last_activity),
                    self.settings.max_wait_s - (clock - started),
                )
                if wait <= 0:
                    break
                await asyncio.sleep(wait)
            if chat.reason == URGE and not self._urge(chat_id):
                return
            chat.generating = True
            await self._respond(chat_id, chat)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never let one chat kill the loop
            # The message may quote chat text, so only the type is logged.
            log.error("respond failed chat=%s error=%s", chat_id, type(exc).__name__)
        finally:
            if chat.task is me:
                chat.generating = False
                chat.committed = False
                chat.task = None
                if chat.dirty:
                    chat.dirty = False
                    chat.restarts = 1
                    chat.last_activity = loop.time()
                    chat.task = loop.create_task(self._run(chat_id, chat))

    async def drain(self, timeout: float = 60.0) -> None:
        """Let answers already under way finish. Called before the process stops,
        so a restart never cuts a burst off after its first line."""
        tasks = [chat.task for chat in self._chats.values() if chat.task is not None and not chat.task.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    def _urge(self, chat_id: int) -> bool:
        """Nobody called me: is this moment worth asking myself whether to speak?"""
        if self.gate is None:
            return False
        now = self._now()
        msgs = self.store.recent(
            chat_id, self.settings.context_messages, self.settings.context_max_age_s, now
        )
        ask, why = self.gate.decide(chat_id, self.store.chat_state(chat_id).talk_level, msgs, now)
        log.log(logging.INFO if ask else logging.DEBUG, "urge chat=%s ask=%s why=%s", chat_id, ask, why)
        return ask

    async def answer(
        self,
        msg: Msg,
        reason: str,
        *,
        title: str | None = None,
        is_private: bool = False,
    ) -> Outcome:
        """Record a message and answer it right away, without debouncing."""
        self.store.record(msg)
        chat = self._chats.setdefault(msg.chat_id, _Chat())
        chat.title, chat.is_private = title, is_private
        chat.trigger, chat.reason = msg, reason
        return await self._respond(msg.chat_id, chat)

    # -- decide and act ---------------------------------------------------

    async def _respond(self, chat_id: int, chat: _Chat) -> Outcome:
        now = self._now()
        msgs = self.store.recent(
            chat_id, self.settings.context_messages, self.settings.context_max_age_s, now
        )
        if self.fetch is not None:
            msgs = await look(msgs, self.store, self.model, self.fetch)
        turns = render_turns(msgs, lookup=lambda message_id: self.store.get(chat_id, message_id))
        unprompted = chat.reason == URGE
        header = scene_header(
            chat.title,
            chat.is_private,
            self.bot_username,
            datetime.fromtimestamp(now, self._tz),
            unprompted=unprompted,
        )
        around = self._around(chat_id, chat, msgs, now)

        recalled: list[int] = []
        memory = ""
        if self.recall is not None and turns and turns[-1]["role"] == "user":
            try:
                recalled = self.recall.search(tail_text(msgs), chat_id, self.settings.recall_scenes)
                memory = self.recall.render(recalled)
            except Exception as exc:  # noqa: BLE001 - answering without memories beats not answering
                log.warning("recall failed chat=%s error=%s", chat_id, type(exc).__name__)

        notes: list[str] = []
        final: tuple[str, ...] = ()
        violations: tuple[str, ...] = ()
        reply: ModelReply | None = None
        attempts = 0
        latency = 0.0
        plain: dict[str, bool] = {}
        thought: list[int] = []  # reasoning tokens of each try
        for attempt in range(2):
            messages = build_messages(
                self.persona.self_text(),
                header,
                turns,
                notes,
                anchor=self.persona.anchor_text(),
                memory=memory,
                facts=self.persona.facts_text(),
                around=around,
            )
            if messages is None:
                return Outcome("nothing")
            try:
                reply = await self.model.complete(messages, TOOLS, **plain)
            except TimeoutError:
                # Thinking for too long gets the same second chance as thinking to no end.
                if attempt or plain:
                    raise
                log.warning("model call timed out chat=%s", chat_id)
                attempts += 1
                plain = {"thinking": False}
                continue
            attempts += 1
            latency += reply.latency_s
            thought.append(reply.reasoning_tokens)
            action = parse_action(reply)
            if action is None and attempt == 0:
                # Nothing usable came back: every token went on thinking. Ask once more without it.
                plain = {"thinking": False}
                continue
            if not isinstance(action, Say):
                final, violations = (), ()
                break
            result = check(action.messages, self.limits)
            violations = result.violations
            if not violations:
                final = result.messages
                break
            if attempt == 0:
                notes = [RETRY_NOTES[name] for name in violations]
                continue
            # Second draft is still off. Long is tolerable; sounding like a help desk is not.
            final = () if "ai_tone" in violations else result.messages

        assert reply is not None
        kind = "said" if final else "silent"
        log.info(
            "%s chat=%s why=%s n=%d lens=%s recalled=%d tries=%d latency=%.1fs prompt_tok=%d "
            "cached=%d reasoning_tok=%s violations=%s",
            kind,
            chat_id,
            chat.reason,
            len(final),
            [len(text) for text in final],
            len(recalled),
            attempts,
            latency,
            reply.prompt_tokens,
            reply.cached_tokens,
            thought,
            list(violations),
        )
        self._trace(chat_id, chat, turns, reply, final, violations, attempts)
        outcome = Outcome(kind, final, violations, reply.reasoning, attempts, len(recalled))
        if unprompted and self.gate is not None:
            if final:
                self.gate.spoke(chat_id, now)
            else:
                self.gate.stayed_silent(chat_id, now)
        if not final:
            return outcome

        chat.committed = True
        reply_to = self._reply_target(chat_id, chat)

        def on_sent(message_id: int, text: str, target: int | None) -> None:
            self.store.record(
                Msg(
                    chat_id=chat_id,
                    message_id=message_id,
                    ts=int(self._now()),
                    user_id=self.bot_id,
                    user_name="我",
                    is_me=True,
                    kind="text",
                    text=text,
                    is_bot=True,
                    reply_to=target,
                )
            )

        await send_burst(
            self.outbox,
            chat_id,
            final,
            reply_to,
            sleep=self._sleep,
            rng=self._rng,
            on_sent=on_sent,
            pace=self.pace,
        )
        return outcome

    def _around(self, chat_id: int, chat: _Chat, msgs: list[Msg], now: float) -> str:
        """What I know about this group, the people talking, and the last few days."""
        if chat.is_private:
            return ""
        names: dict[int, str] = {}
        for msg in reversed(msgs):  # most recent speakers first
            if not msg.is_me and msg.user_id is not None:
                names.setdefault(msg.user_id, clean_name(msg.user_name))
        order = list(names)
        trigger = chat.trigger
        if trigger is not None and trigger.user_id in names:
            order.remove(trigger.user_id)
            order.insert(0, trigger.user_id)
        cards = self.persona.people_cards(chat_id, order, self.settings.people_cards)
        today = datetime.fromtimestamp(now, self._tz).strftime("%Y-%m-%d")
        return render_around(
            self.persona.group_card(chat_id),
            [(names[user_id], card) for user_id, card in cards],
            self.persona.diary(chat_id, today, self.settings.diary_days),
        )

    def _reply_target(self, chat_id: int, chat: _Chat) -> int | None:
        """Quote the message I am answering once the chat has moved on past it.

        Counted at the moment I send, so lines that arrived while I was thinking
        count too: a late answer with no quote leaves people guessing who it is for.
        """
        trigger = chat.trigger
        if chat.is_private or trigger is None:
            return None
        if self.store.newer(chat_id, trigger.message_id) > QUOTE_AFTER:
            return trigger.message_id
        return None

    def _trace(
        self,
        chat_id: int,
        chat: _Chat,
        turns: list[dict[str, str]],
        reply: ModelReply,
        final: tuple[str, ...],
        violations: tuple[str, ...],
        attempts: int,
    ) -> None:
        """Append the full exchange to the private trace. It contains message text."""
        if not self.settings.trace:
            return
        now = datetime.fromtimestamp(self._now(), self._tz)
        record = {
            "time": now.isoformat(timespec="seconds"),
            "chat_id": chat_id,
            "why": chat.reason,
            "trigger": chat.trigger.message_id if chat.trigger else None,
            "turns": turns,
            "reasoning": reply.reasoning,
            "said": list(final),
            "violations": list(violations),
            "attempts": attempts,
        }
        try:
            self.settings.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.settings.trace_dir / f"{now:%Y-%m-%d}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("trace write failed error=%s", type(exc).__name__)
