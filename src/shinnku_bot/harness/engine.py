"""The harness: from an incoming message to what I send back."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from shinnku_bot.config import Settings
from shinnku_bot.harness.act import TOOLS, Line, Say, parse_action
from shinnku_bot.harness.attention import blocked
from shinnku_bot.harness.exam import BRUSH_OFF, is_exam
from shinnku_bot.harness.hooks import (
    AI_TONE_NOTE,
    StyleLimits,
    clean,
    fit,
    violations_of,
)
from shinnku_bot.harness.look import Check, Fetch, look
from shinnku_bot.harness.lookup import Hit, Search, look_up
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.prompt import (
    build_messages,
    render_around,
    render_known,
    render_pending,
    scene_header,
)
from shinnku_bot.harness.recall import Recall
from shinnku_bot.harness.send import Outbox, Pace, send_burst
from shinnku_bot.harness.transcript import (
    Lookup,
    clean_name,
    render_body,
    render_lines,
    render_turns,
    tail_text,
)
from shinnku_bot.harness.urge import Gate
from shinnku_bot.llm import ChatModel, ModelReply
from shinnku_bot.store import Msg, Store

log = logging.getLogger(__name__)

# The reason recorded when nobody called me and I am only considering joining in.
URGE = "urge"
# A line from someone while I am answering them; it waits for the next turn.
FOLLOWUP = "followup"
# With more than this many newer lines from others, my answer quotes what it answers.
QUOTE_AFTER = 4
# Questions per turn, and per person per turn. The rest wait for the next turn.
BATCH = 6
PER_ASKER = 3
# Messages in one turn, within Telegram's 20 a minute in a group.
TURN_MESSAGES = 12
# How much longer a turn waits for the exam checks and lookups, which started when the questions came in.
CHECK_WAIT_S = 3.0
# A question whose turn failed this many times is dropped.
MAX_TRIES = 3
# The longest I wait when the gateway says how long to wait.
RETRY_AFTER_MAX_S = 60.0


@dataclass(frozen=True)
class Outcome:
    """What happened to one wake-up. `kind` is said, silent or nothing."""

    kind: str
    messages: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    reasoning: str = ""
    attempts: int = 0
    recalled: int = 0


@dataclass(eq=False)
class Ask:
    """A message aimed at me, from when it arrives until a turn has answered or passed it."""

    msg: Msg
    reason: str
    arrived: float
    taken: bool = False  # in the turn under way
    tries: int = 0  # turns that failed with it
    # Whether it is a question set to test me as an AI, judged from when it came in.
    exam: asyncio.Task[bool] | None = None
    # What the web says about it, looked up from when it came in.
    lookup: asyncio.Task[tuple[Hit, ...]] | None = None

    @property
    def is_exam(self) -> bool:
        """Judged an exam in time. A check still running or turned off counts as talk."""
        return self.exam is not None and self.exam.done() and not self.exam.cancelled() and self.exam.result()

    @property
    def found(self) -> tuple[Hit, ...]:
        """What the lookup found in time; nothing while it runs."""
        task = self.lookup
        return task.result() if task is not None and task.done() and not task.cancelled() else ()


@dataclass
class _Chat:
    title: str | None = None
    is_private: bool = False
    asks: list[Ask] = field(default_factory=list)  # oldest first
    # When someone I am waiting for last spoke (anyone, while I only consider joining in).
    last_activity: float = 0.0
    # The moment I might join in uncalled, and when it began.
    urge: Msg | None = None
    urge_since: float = 0.0
    # After a failed turn, the next one waits until this.
    retry_at: float = 0.0
    task: asyncio.Task | None = None


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
        # Downloads a picture, and says whether one must stay here (NSFW); without
        # both, pictures stay placeholders.
        fetch: Fetch | None = None,
        check: Check | None = None,
        # Looks at pictures; without it the chat model does.
        look_model: ChatModel | None = None,
        # Looks things up on the web for the questions; without it I know what the model knows.
        search: Search | None = None,
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
        self.check = check
        self.look_model = look_model or model
        self.search = search
        self._sleep = sleep
        self._rng = rng
        self._now = now
        self._tz = ZoneInfo(settings.timezone)
        self._chats: dict[int, _Chat] = {}
        self._closing = False
        # The gateway turned me away and said how long: every chat waits until then.
        self._hold_until = 0.0

    # -- perceive ---------------------------------------------------------

    def handle(
        self,
        msg: Msg,
        reason: str | None,
        *,
        title: str | None = None,
        is_private: bool = False,
    ) -> None:
        """Record a message; queue it if it is aimed at me, else maybe consider joining in.

        Must be called from the running event loop.
        """
        self.store.record(msg)
        chat = self._chats.setdefault(msg.chat_id, _Chat())
        chat.title, chat.is_private = title, is_private
        clock = asyncio.get_running_loop().time()
        if self.gate is not None and msg.user_id == self.settings.owner_id:
            self.gate.owner_spoke(msg.chat_id, self._now())

        if reason is not None:
            deny = blocked(
                self.settings, self.store.chat_state(msg.chat_id), msg.chat_id, is_private, self._now()
            )
            if deny is not None:
                log.info("skip chat=%s why=%s", msg.chat_id, deny)
                return
            self._queue(msg.chat_id, chat, Ask(msg, reason, clock))
            return

        mine = [ask for ask in chat.asks if msg.user_id is not None and ask.msg.user_id == msg.user_id]
        if any(ask.taken for ask in mine):
            # More from someone I am answering right now. One follow-up each, the latest.
            chat.asks = [ask for ask in chat.asks if ask.taken or ask.reason != FOLLOWUP or ask not in mine]
            self._queue(msg.chat_id, chat, Ask(msg, FOLLOWUP, clock))
            return
        if mine:
            chat.last_activity = clock  # still typing their question: wait for them
            return
        if chat.asks or self.gate is None or is_private or msg.is_bot or msg.is_me:
            return
        state = self.store.chat_state(msg.chat_id)
        if state.talk_level <= 0 or blocked(self.settings, state, msg.chat_id, is_private, self._now()):
            return
        # While I only consider joining in, anyone's line restarts the wait.
        if chat.urge is None:
            chat.urge_since = clock
        chat.urge, chat.last_activity = msg, clock
        self._wake(msg.chat_id, chat)

    def _queue(self, chat_id: int, chat: _Chat, ask: Ask) -> None:
        if self.settings.brush_off_exams or self.search is not None:
            # Judged and looked up while I wait for them to stop typing.
            loop = asyncio.get_running_loop()
            replied, before = self._context(ask.msg)
            if self.settings.brush_off_exams:
                ask.exam = loop.create_task(is_exam(self.model, ask.msg, replied, before))
            if self.search is not None:
                ask.lookup = loop.create_task(look_up(self.model, self.search, ask.msg, replied, before))
        chat.asks.append(ask)
        chat.last_activity = ask.arrived
        chat.urge = None  # being called comes first
        self._wake(chat_id, chat)

    def _wake(self, chat_id: int, chat: _Chat) -> None:
        if chat.task is None:
            chat.task = asyncio.get_running_loop().create_task(self._work(chat_id, chat))

    async def _work(self, chat_id: int, chat: _Chat) -> None:
        """The chat's one worker: answer what waits, a turn at a time, then maybe join in.

        A turn under way is never cancelled; whatever arrives meanwhile waits for the next.
        """
        try:
            while chat.asks or (chat.urge is not None and not self._closing):
                await self._held(chat)
                if chat.asks:
                    await self._quiet(chat, chat.asks[0].arrived)
                    self._expire(chat_id, chat)
                    if chat.asks:
                        await self._judged(chat)
                        await self._turn(chat_id, chat, self._take(chat))
                    continue
                await self._quiet(chat, chat.urge_since)
                if chat.asks or chat.urge is None:
                    continue
                chat.urge = None
                if self._urge(chat_id):
                    await self._turn(chat_id, chat, [])
        finally:
            chat.task = None

    async def _held(self, chat: _Chat) -> None:
        """Wait out a failed turn's pause, and any hold the gateway asked for."""
        loop = asyncio.get_running_loop()
        while (wait := max(chat.retry_at, self._hold_until) - loop.time()) > 0:
            await asyncio.sleep(wait)

    def _expire(self, chat_id: int, chat: _Chat) -> None:
        """Drop the questions that have waited too long: the chat has moved on."""
        clock = asyncio.get_running_loop().time()
        for ask in chat.asks:
            if clock - ask.arrived > self.settings.ask_ttl_s:
                log.warning("ask dropped chat=%s id=%s why=expired", chat_id, ask.msg.message_id)
        chat.asks = [ask for ask in chat.asks if clock - ask.arrived <= self.settings.ask_ttl_s]

    async def _quiet(self, chat: _Chat, since: float) -> None:
        """Wait until the people I am waiting for stop typing, but no longer than max_wait."""
        loop = asyncio.get_running_loop()
        while True:
            clock = loop.time()
            wait = min(
                self.settings.quiet_window_s - (clock - chat.last_activity),
                self.settings.max_wait_s - (clock - since),
            )
            if wait <= 0:
                return
            await asyncio.sleep(wait)

    def _context(self, msg: Msg) -> tuple[Msg | None, list[Msg]]:
        """What the quick checks see besides the message: what it replies to, the lines before it."""
        replied = self.store.get(msg.chat_id, msg.reply_to) if msg.reply_to is not None else None
        recent = self.store.recent(msg.chat_id, 8, self.settings.context_max_age_s, self._now())
        # Only what came before it: lines that arrived since are not its context.
        return replied, [m for m in recent if m.message_id < msg.message_id][-3:]

    async def _judged(self, chat: _Chat) -> None:
        """Give the exam checks and lookups of the waiting questions up to CHECK_WAIT_S more."""
        running = [
            task for ask in chat.asks for task in (ask.exam, ask.lookup) if task is not None and not task.done()
        ]
        if running:
            await asyncio.wait(running, timeout=CHECK_WAIT_S)

    def _take(self, chat: _Chat) -> list[Ask]:
        """This turn's questions: oldest first, a few per person; the rest wait.

        Exams and talk never share a turn, since thinking is on or off for a whole
        call. Exams go first: answering them takes no thinking, so it is quick.
        """
        exams = any(ask.is_exam for ask in chat.asks)
        batch: list[Ask] = []
        per_asker: Counter[int | None] = Counter()
        for ask in chat.asks:
            if len(batch) == BATCH:
                break
            if ask.is_exam == exams and per_asker[ask.msg.user_id] < PER_ASKER:
                per_asker[ask.msg.user_id] += 1
                ask.taken = True
                batch.append(ask)
        return batch

    async def _turn(self, chat_id: int, chat: _Chat, batch: list[Ask]) -> None:
        """One model call for `batch`, after which those questions are done. If it fails,
        they wait a little and go into another turn, up to MAX_TRIES."""
        try:
            await self._respond(chat_id, chat, batch)
        except Exception as exc:  # noqa: BLE001 - never let one chat kill the loop
            wait = self._retry_in(exc, max((ask.tries for ask in batch), default=0) + 1)
            # The message may quote chat text, so only the type is logged.
            log.error("respond failed chat=%s error=%s retry_in=%.1fs", chat_id, type(exc).__name__, wait)
            chat.retry_at = asyncio.get_running_loop().time() + wait
            for ask in batch:
                ask.tries += 1
                ask.taken = False
                if ask.tries >= MAX_TRIES:
                    log.warning("ask dropped chat=%s id=%s why=failed", chat_id, ask.msg.message_id)
            chat.asks = [ask for ask in chat.asks if ask.tries < MAX_TRIES]
            return
        chat.asks = [ask for ask in chat.asks if ask not in batch]

    def _retry_in(self, exc: Exception, tries: int) -> float:
        """How long to wait after a failed turn. A 429 holds every chat: the quota is the gateway's."""
        wait = self.settings.retry_s * (1 if tries <= 1 else 3)
        if getattr(exc, "status_code", None) == 429:
            with contextlib.suppress(AttributeError, KeyError, TypeError, ValueError):
                wait = min(float(exc.response.headers["retry-after"]), RETRY_AFTER_MAX_S)  # type: ignore[attr-defined]
            self._hold_until = max(self._hold_until, asyncio.get_running_loop().time() + wait)
        return wait

    async def drain(self, timeout: float = 60.0) -> None:
        """Answer what is already waiting before the process stops, so a restart neither
        cuts a burst off after its first line nor leaves a question behind. No new
        uncalled-for turn starts from here on."""
        self._closing = True
        tasks = [chat.task for chat in self._chats.values() if chat.task is not None]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
        left = sum(len(chat.asks) for chat in self._chats.values())
        if left:
            log.warning("stopping with asks=%d unanswered", left)

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
        """Record a message and answer it right away, without queueing. `URGE` means nobody called me."""
        self.store.record(msg)
        chat = self._chats.setdefault(msg.chat_id, _Chat())
        chat.title, chat.is_private = title, is_private
        asks = [] if reason == URGE else [Ask(msg, reason, asyncio.get_running_loop().time())]
        return await self._respond(msg.chat_id, chat, asks)

    # -- decide and act ---------------------------------------------------

    async def _respond(self, chat_id: int, chat: _Chat, asks: Sequence[Ask]) -> Outcome:
        """Answer `asks` in one go; with none, I am only considering joining in."""
        now = self._now()
        msgs = self.store.recent(
            chat_id, self.settings.context_messages, self.settings.context_max_age_s, now
        )
        if self.fetch is not None and self.check is not None:
            msgs = await look(msgs, self.store, self.look_model, self.fetch, self.check)

        def lookup(message_id: int) -> Msg | None:
            return self.store.get(chat_id, message_id)

        turns = render_turns(msgs, lookup=lookup)
        unprompted = not asks
        header = scene_header(
            chat.title,
            chat.is_private,
            self.bot_username,
            datetime.fromtimestamp(now, self._tz),
            unprompted=unprompted,
        )
        around = self._around(chat_id, chat, msgs, now, asks)
        pending = self._pending(msgs, asks, lookup)
        why = ",".join(dict.fromkeys(ask.reason for ask in asks)) or URGE

        recalled: list[int] = []
        memory = ""
        if self.recall is not None and (pending or (turns and turns[-1]["role"] == "user")):
            # With a list of questions, the questions are what this moment is about.
            query = tail_text([ask.msg for ask in asks], lines=len(asks)) if pending else tail_text(msgs)
            try:
                recalled = self.recall.search(query, chat_id, self.settings.recall_scenes)
                memory = self.recall.render(recalled)
            except Exception as exc:  # noqa: BLE001 - answering without memories beats not answering
                log.warning("recall failed chat=%s error=%s", chat_id, type(exc).__name__)

        # Questions set to test me get brushed off, without thinking them over.
        exam = bool(asks) and all(ask.is_exam for ask in asks)
        base = [BRUSH_OFF] if exam else []
        # An exam gets brushed off: what the web says about it does not matter.
        known = [] if exam else list(dict.fromkeys(hit.line for ask in asks for hit in ask.found))
        notes = list(base)
        final: tuple[Line, ...] = ()
        violations: tuple[str, ...] = ()
        reply: ModelReply | None = None
        attempts = 0
        latency = 0.0
        plain: dict[str, bool] = {"thinking": False} if exam else {}
        thought: list[int] = []  # reasoning tokens of each try
        loop = asyncio.get_running_loop()
        started = loop.time()
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
                pending=pending,
                known=render_known(known),
            )
            if messages is None:
                return Outcome("nothing")
            # The first call gets call_timeout_s; a second one, what is left of the turn.
            budget = (
                self.settings.call_timeout_s
                if attempt == 0
                else self.settings.turn_timeout_s - (loop.time() - started)
            )
            try:
                reply = await asyncio.wait_for(self.model.complete(messages, TOOLS, **plain), budget)
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
            lines = tuple((to, text) for to, item in action.messages for text in clean([item], self.limits))
            violations = violations_of([text for _, text in lines], self.limits)
            # Long is tolerable; sounding like a help desk is not, and gets one more try.
            if "ai_tone" not in violations:
                final = lines
                break
            if attempt == 0:
                notes = [*base, AI_TONE_NOTE]
                continue
            final = ()

        assert reply is not None
        answers = self._route(asks, final)
        said = tuple(text for _, texts in answers for text in texts)
        kind = "said" if said else "silent"
        log.info(
            "%s chat=%s why=%s exam=%s asks=%d listed=%s answered=%d left=%d n=%d lens=%s "
            "recalled=%d known=%d tries=%d latency=%.1fs prompt_tok=%d cached=%d reasoning_tok=%s violations=%s",
            kind,
            chat_id,
            why,
            exam,
            len(asks),
            bool(pending),
            sum(ask is not None for ask, _ in answers),
            sum(not ask.taken for ask in chat.asks),
            len(said),
            [len(text) for text in said],
            len(recalled),
            len(known),
            attempts,
            latency,
            reply.prompt_tokens,
            reply.cached_tokens,
            thought,
            list(violations),
        )
        self._trace(chat_id, why, asks, turns, known, reply, said, violations, attempts)
        outcome = Outcome(kind, said, violations, reply.reasoning, attempts, len(recalled))
        if unprompted and self.gate is not None:
            if said:
                self.gate.spoke(chat_id, now)
            else:
                self.gate.stayed_silent(chat_id, now)
        if not said:
            return outcome

        # Quotes are worked out now, at the moment I send, so the lines that came in
        # while I was thinking count too.
        out: list[tuple[str, int | None]] = []
        for ask, texts in answers:
            quote = self._quote(chat_id, chat, asks, ask)
            out.extend((text, quote if index == 0 else None) for index, text in enumerate(texts))

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
            out,
            sleep=self._sleep,
            rng=self._rng,
            on_sent=on_sent,
            pace=self.pace,
        )
        return outcome

    def _around(
        self, chat_id: int, chat: _Chat, msgs: list[Msg], now: float, asks: Sequence[Ask]
    ) -> str:
        """What I know about this group, the people talking, and the last few days."""
        if chat.is_private:
            return ""
        names: dict[int, str] = {}
        for msg in reversed(msgs):  # most recent speakers first
            if not msg.is_me and msg.user_id is not None:
                names.setdefault(msg.user_id, clean_name(msg.user_name))
        order = list(names)
        for ask in asks:  # oldest first, so the latest asker ends up in front
            if ask.msg.user_id in order:
                order.remove(ask.msg.user_id)
                order.insert(0, ask.msg.user_id)
        cards = self.persona.people_cards(chat_id, order, self.settings.people_cards)
        today = datetime.fromtimestamp(now, self._tz).strftime("%Y-%m-%d")
        return render_around(
            self.persona.group_card(chat_id),
            [(names[user_id], card) for user_id, card in cards],
            self.persona.diary(chat_id, today, self.settings.diary_days),
        )

    def _pending(self, msgs: list[Msg], asks: Sequence[Ask], lookup: Lookup) -> str:
        """The questions still waiting, numbered, when the chat alone does not make them clear.

        One question followed only by its asker's own lines is clear: the prompt then
        reads exactly as it always did.
        """
        if not asks:
            return ""
        ids = [m.message_id for m in msgs]
        first = asks[0].msg
        if len(asks) == 1 and first.message_id in ids:
            after = msgs[ids.index(first.message_id) + 1 :]
            if all(not m.is_me and m.user_id == first.user_id for m in after):
                return ""
        shown = {m.message_id: m for m in msgs}  # these carry what their pictures show
        lines = []
        for ask in asks:
            msg = shown.get(ask.msg.message_id, ask.msg)
            lines.append((render_lines([msg], lookup) or [render_body(msg)])[0])
        return render_pending(lines)

    def _route(self, asks: Sequence[Ask], lines: Sequence[Line]) -> list[tuple[Ask | None, tuple[str, ...]]]:
        """Group the lines by the question they answer, in the order the groups first appear.

        With one question every line answers it. A number not on the list counts as
        none. Each group is capped so that a turn stays within TURN_MESSAGES.
        """
        groups: dict[int | None, list[str]] = {}
        for to, text in lines:
            if len(asks) == 1:
                key: int | None = 0
            else:
                key = to - 1 if to is not None and 1 <= to <= len(asks) else None
            groups.setdefault(key, []).append(text)
        cap = max(1, min(self.limits.max_burst, TURN_MESSAGES // max(len(groups), 1)))
        return [(None if key is None else asks[key], fit(texts, cap)) for key, texts in groups.items()]

    def _quote(self, chat_id: int, chat: _Chat, asks: Sequence[Ask], ask: Ask | None) -> int | None:
        """What the first line of an answer to `ask` quotes.

        Answering several people, each answer quotes its question. Answering one, only
        once the chat has moved on past it: a late answer with no quote leaves people
        guessing who it is for.
        """
        if chat.is_private or ask is None:
            return None
        message_id = ask.msg.message_id
        if len(asks) > 1 or self.store.newer(chat_id, message_id) > QUOTE_AFTER:
            return message_id
        return None

    def _trace(
        self,
        chat_id: int,
        why: str,
        asks: Sequence[Ask],
        turns: list[dict[str, str]],
        known: list[str],
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
            "why": why,
            "asks": [ask.msg.message_id for ask in asks],
            "turns": turns,
            "known": known,
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
