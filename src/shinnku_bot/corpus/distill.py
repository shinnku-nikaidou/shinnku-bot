"""Distil my history into what the bot knows about me, the people and the groups.

Everything written here is chat-derived text and stays under data/. The console
only ever gets counts and sizes. Model calls are estimated before they are made,
their results are kept per chunk, and an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import shutil
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.pairs import train_cutoffs
from shinnku_bot.harness.transcript import ME, clean_name, render_body, render_lines, render_scene
from shinnku_bot.llm import ChatModel

Progress = Callable[[str], None]

TOKENS_PER_CHAR = 0.67
SESSION_GAP_S = 30 * 60
CATEGORIES = ("身份", "生活", "工作学习", "设备", "爱好", "观点", "习惯", "关系")
SECTION_LINES = {"身份": 8, "生活": 14, "工作学习": 12, "设备": 10, "爱好": 14, "观点": 14, "习惯": 10, "关系": 8}
FACTS_MAX_TOKENS = 3500
CARD_MAX_TOKENS = 220
GROUP_MAX_TOKENS = 450
GIVE_UP_AFTER = 5

_FIRST_PERSON = re.compile(r"我|俺|咱|本人")
# Wording that addresses the model as someone following instructions, rather than
# being me. Topic words (bots, prompts, role-play games) are not listed: I talk
# about those things, so they turn up in honest facts about me.
_FRAMING = [
    re.compile(pattern)
    for pattern in (
        r"(?:回复|回答|面对|对)用户",
        r"用户(?:说|问|提|发)",
        r"你(?:是|要|应该|必须|需要|的任务)",
        r"请你",
    )
]

# -- shared machinery ---------------------------------------------------------


def estimate_tokens(text: str) -> int:
    return int(len(text) * TOKENS_PER_CHAR)


def lint(text: str, max_tokens: int) -> list[str]:
    """Why this text may not go into the prompt as part of who I am. Empty if fine."""
    problems = []
    if not text.strip():
        problems.append("empty")
    hits = sum(bool(pattern.search(text)) for pattern in _FRAMING)
    if hits:
        problems.append(f"framing_words:{hits}")
    if estimate_tokens(text) > max_tokens:
        problems.append(f"too_long:{estimate_tokens(text)}>{max_tokens}")
    return problems


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(properties)},
        },
    }


async def ask(model: ChatModel, system: str, user: str, tool: dict[str, Any]) -> dict[str, Any] | None:
    """One structured call: the arguments of the tool call, or None."""
    reply = await model.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], [tool]
    )
    wanted = tool["function"]["name"]
    for call in reply.tool_calls:
        if call.name == wanted:
            return call.args
    return None


async def run_jobs(
    jobs: Sequence[Callable[[], Awaitable[bool]]],
    concurrency: int = 4,
    progress: Progress = lambda line: None,
) -> tuple[int, int]:
    """Run jobs with bounded concurrency. Returns (succeeded, failed).

    After GIVE_UP_AFTER failures in a row the rest are skipped: that pattern means
    the quota is gone or the gateway is down, and retrying would only burn more.
    """
    gate = asyncio.Semaphore(concurrency)
    done = failed = streak = 0
    started = time.monotonic()

    async def one(job: Callable[[], Awaitable[bool]]) -> None:
        nonlocal done, failed, streak
        async with gate:
            if streak >= GIVE_UP_AFTER:
                failed += 1
                return
            try:
                ok = await job()
            except Exception:  # noqa: BLE001 - the message may quote chat text
                ok = False
            done += 1
            if ok:
                streak = 0
            else:
                failed += 1
                streak += 1
            if done % 20 == 0 or done == len(jobs):
                progress(f"{done}/{len(jobs)} calls, {failed} failed, {time.monotonic() - started:.0f}s")

    await asyncio.gather(*(one(job) for job in jobs))
    return len(jobs) - failed, failed


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def _month(ts: int) -> str:
    return time.strftime("%Y-%m", time.gmtime(ts))


def _year(ts: int) -> int:
    return time.gmtime(ts).tm_year


# -- me: candidates -----------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """Something I said about myself, with the line or two before it."""

    ts: int
    lines: tuple[str, ...]


def self_candidates(db: CorpusDB, min_chars: int = 8, context: int = 2) -> list[Candidate]:
    """My own statements from the training period that mention myself. No model calls."""
    cutoffs = train_cutoffs(db)
    seen: set[str] = set()
    found: list[Candidate] = []
    for chat_id in db.chat_ids():
        recent: deque[str] = deque(maxlen=context)
        last_ts = 0
        for row in db.rows(chat_id):
            msg = row.msg
            if msg.ts >= cutoffs[chat_id]:
                break
            if msg.ts - last_ts > SESSION_GAP_S:
                recent.clear()
            last_ts = msg.ts
            body = render_body(msg)[:120]
            if not body:
                continue
            if (
                msg.is_me
                and msg.kind == "text"
                and not row.canned
                and len(body) >= min_chars
                and _FIRST_PERSON.search(body)
                and body not in seen
            ):
                seen.add(body)
                found.append(Candidate(msg.ts, (*recent, f"{ME}: {body}")))
            speaker = ME if msg.is_me else clean_name(msg.user_name)
            recent.append(f"{speaker}: {body[:80]}")
    found.sort(key=lambda c: c.ts)
    return found


def thin(candidates: Sequence[Candidate], limit: int, seed: int = 0, parts: int = 12) -> list[Candidate]:
    """Keep at most `limit`, spread over time with the newest slice weighted about 3x."""
    if limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    rng = random.Random(seed)
    bounds = [i * len(candidates) // parts for i in range(parts + 1)]
    weights = [parts / 2 + i for i in range(parts)]
    kept: list[Candidate] = []
    running, planned = 0.0, 0
    for index, weight in enumerate(weights):
        running += weight
        target = round(limit * running / sum(weights))
        slice_ = candidates[bounds[index] : bounds[index + 1]]
        kept.extend(rng.sample(slice_, min(len(slice_), target - planned)))
        planned = target
    kept.sort(key=lambda c: c.ts)
    return kept


def chunk_candidates(candidates: Sequence[Candidate], max_chars: int = 6000) -> list[str]:
    """Group candidates, in time order, into texts of about `max_chars`."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for candidate in candidates:
        block = f"[{_month(candidate.ts)}]\n" + "\n".join(candidate.lines)
        if current and size + len(block) > max_chars:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(block)
        size += len(block) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# -- me: extraction -----------------------------------------------------------

FACT_TOOL = _tool(
    "note_facts",
    "记下从这段话里看出的、关于「我」的事实。",
    {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "year": {"type": "integer"},
                    "sensitive": {"type": "boolean"},
                },
                "required": ["text", "category", "year", "sensitive"],
            },
        }
    },
)
EXTRACT_SYSTEM = """下面是一个人在群聊里说过的话，他的发言标为「我」，前面可能带一两句别人的上文，每段开头的方括号是年月。
请从中找出关于「我」本人的事实，调用 note_facts 记下来。

要求：
- 每条用第一人称写成一句完整的话（以「我」开头），脱离上下文也能看懂。
- 只记有明确依据、相对稳定的事：身份、住在哪、工作或学业、用什么设备、爱好、对事物的固定看法、生活习惯、和谁是什么关系。
- 玩笑、反话、一时的情绪、对别人的简单附和，不记。
- 拿不准是不是真的就不记，宁可少记。
- 涉及证件号、住址门牌、电话、账号密码、健康状况、收入和财务、以及别人私事的，照常记，但把 sensitive 设为 true。
- year 填这句话是哪一年说的。
- 没有可记的就传空数组。"""


def clean_facts(raw: Any) -> list[dict[str, Any]]:
    """Keep only well-formed facts from a model's answer."""
    out = []
    for fact in raw if isinstance(raw, list) else []:
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("text", "")).strip()
        if not text:
            continue
        category = fact.get("category")
        try:
            year = int(fact.get("year") or 0)
        except (TypeError, ValueError):
            year = 0
        out.append(
            {
                "text": text[:200],
                "category": category if category in CATEGORIES else "生活",
                "year": year,
                "sensitive": bool(fact.get("sensitive")),
            }
        )
    return out


def extracted(out_dir: Path, chunks: Sequence[str]) -> int:
    """How many of these chunks already have their facts on disk."""
    return sum((Path(out_dir) / f"{_digest(chunk)}.json").is_file() for chunk in chunks)


async def extract_self(
    model: ChatModel,
    chunks: Sequence[str],
    out_dir: Path,
    concurrency: int = 4,
    progress: Progress = lambda line: None,
) -> tuple[int, int, int]:
    """Extract facts from every chunk not done yet. Returns (reused, done, failed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [chunk for chunk in chunks if not (out_dir / f"{_digest(chunk)}.json").is_file()]

    def job(chunk: str) -> Callable[[], Awaitable[bool]]:
        async def run() -> bool:
            args = await ask(model, EXTRACT_SYSTEM, chunk, FACT_TOOL)
            if args is None:
                return False
            _save_json(out_dir / f"{_digest(chunk)}.json", clean_facts(args.get("facts")))
            return True

        return run

    done, failed = await run_jobs([job(chunk) for chunk in todo], concurrency, progress)
    return len(chunks) - len(todo), done, failed


def load_facts(out_dir: Path, chunks: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """All extracted facts (only those of `chunks`, when given), without duplicates."""
    if chunks is None:
        paths = sorted(Path(out_dir).glob("*.json"))
    else:
        paths = [Path(out_dir) / f"{_digest(chunk)}.json" for chunk in chunks]
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for fact in clean_facts(_load_json(path)):
            if fact["text"] not in seen:
                seen.add(fact["text"])
                facts.append(fact)
    return facts


# -- me: merging --------------------------------------------------------------

SECTION_TOOL = _tool("write_section", "写下整理好的清单。", {"lines": {"type": "array", "items": {"type": "string"}}})
PROBE_TOOL = _tool(
    "write_probes",
    "写下小测验。",
    {
        "probes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question", "keywords"],
            },
        }
    },
)
SECTION_SYSTEM = """下面是关于同一个人的一批事实，都是第一人称，每条前面是说这话的年份，类别是「{category}」。现在是 {year} 年。
请整理成一份清单，调用 write_section。

要求：
- 合并重复的，意思相同的只留一条。
- 互相矛盾的以年份新的为准；被取代的旧情况如果值得留，写成「我以前……」。
- 只在较早年份出现过一次、现在多半已经变了的，丢掉。
- 每条一句话，第一人称，以「我」开头，不带年份前缀（需要时把年份写进句子里）。
- 最多 {limit} 条，重要的、近期的优先。"""
PROBE_SYSTEM ="""下面是一个人关于自己的事实清单。请出 {count} 道小测验，调用 write_probes。

要求：
- 每道题是朋友在聊天里会随口问他的一句话，口语、简短，答案能直接从清单的某一条里找到。
- keywords 填答对时回答里一定会出现的关键词，1 到 3 个，每个尽量短；回答里出现任意一个就算对。
- 不要出是非题。"""


def _section_input(facts: Sequence[dict[str, Any]], max_chars: int = 14000) -> str:
    lines: list[str] = []
    size = 0
    for fact in sorted(facts, key=lambda f: -f["year"]):
        line = f"{fact['year'] or '?'}  {fact['text']}"
        if size + len(line) > max_chars:
            break
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)


def _lines(raw: Any, limit: int) -> list[str]:
    out = []
    for line in raw if isinstance(raw, list) else []:
        text = " ".join(str(line).split()).lstrip("-•· ")
        if text:
            out.append(text)
    return out[:limit]


def clean_probes(raw: Any) -> list[dict[str, Any]]:
    probes = []
    for probe in raw if isinstance(raw, list) else []:
        if not isinstance(probe, dict):
            continue
        question = str(probe.get("question", "")).strip()
        keywords = [str(k).strip() for k in probe.get("keywords") or [] if str(k).strip()]
        if question and keywords:
            probes.append({"question": question, "keywords": keywords[:3]})
    return probes


async def merge_facts(
    model: ChatModel,
    facts: Sequence[dict[str, Any]],
    persona_dir: Path,
    as_of_year: int | None = None,
    probe_count: int = 30,
    progress: Progress = lambda line: None,
) -> dict[str, Any]:
    """Write FACTS.draft.md and probes.jsonl. Returns sizes and checks.

    Only facts are distilled. A model-written portrait of me was tried in place
    of the hand-written SELF text and dropped: with it my replies turned long
    and written, and a classifier told them from mine far more easily.
    """
    usable = [fact for fact in facts if not fact["sensitive"]]
    year = as_of_year or max((fact["year"] for fact in usable), default=time.gmtime().tm_year)
    sections: dict[str, list[str]] = {}
    for category in CATEGORIES:
        mine = [fact for fact in usable if fact["category"] == category]
        if not mine:
            continue
        limit = SECTION_LINES[category]
        args = await ask(
            model,
            SECTION_SYSTEM.format(category=category, year=year, limit=limit),
            _section_input(mine),
            SECTION_TOOL,
        )
        lines = _lines(args.get("lines") if args else None, limit)
        if lines:
            sections[category] = lines
        progress(f"section {len(sections)}: {len(mine)} facts in, {len(lines)} lines out")

    facts_text = "\n\n".join(
        f"{category}：\n" + "\n".join(f"- {line}" for line in lines) for category, lines in sections.items()
    )
    probes: list[dict[str, Any]] = []
    if facts_text:
        args = await ask(model, PROBE_SYSTEM.format(count=probe_count), facts_text, PROBE_TOOL)
        probes = clean_probes(args.get("probes") if args else None)[:probe_count]

    persona_dir.mkdir(parents=True, exist_ok=True)
    (persona_dir / "FACTS.draft.md").write_text(facts_text + "\n", encoding="utf-8")
    with (persona_dir / "probes.jsonl").open("w", encoding="utf-8") as handle:
        for probe in probes:
            handle.write(json.dumps(probe, ensure_ascii=False) + "\n")
    return {
        "facts_in": len(facts),
        "sensitive_dropped": len(facts) - len(usable),
        "fact_lines": sum(len(lines) for lines in sections.values()),
        "tokens": estimate_tokens(facts_text),
        "probes": len(probes),
        "problems": lint(facts_text, FACTS_MAX_TOKENS),
    }


def check_persona(persona_dir: Path) -> dict[str, Any]:
    """Sizes and lint results for everything distilled so far. No text."""
    persona_dir = Path(persona_dir)

    def read(name: str) -> str:
        try:
            return (persona_dir / name).read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    report: dict[str, Any] = {}
    for name in ("FACTS.draft.md", "FACTS.md"):
        text = read(name)
        if text:
            report[name] = {"tokens": estimate_tokens(text), "problems": lint(text, FACTS_MAX_TOKENS)}
    for folder, limit in (("people", CARD_MAX_TOKENS), ("groups", GROUP_MAX_TOKENS)):
        cards = [p.read_text(encoding="utf-8") for p in (persona_dir / folder).rglob("*.md")]
        if cards:
            report[folder] = {
                "cards": len(cards),
                "tokens_mean": round(sum(estimate_tokens(c) for c in cards) / len(cards)),
                "with_problems": sum(bool(lint(c, limit * 3)) for c in cards),
            }
    return report


def publish(persona_dir: Path) -> list[str]:
    """Make the reviewed facts draft live. Returns the problems that stopped it, if any."""
    draft, live = Path(persona_dir) / "FACTS.draft.md", Path(persona_dir) / "FACTS.md"
    try:
        text = draft.read_text(encoding="utf-8")
    except OSError:
        return ["no_draft"]
    problems = lint(text, FACTS_MAX_TOKENS)
    if problems:
        return problems
    if live.is_file():
        shutil.copyfile(live, live.with_suffix(".md.bak"))
    shutil.copyfile(draft, live)
    return []


# -- people -------------------------------------------------------------------

CARD_TOOL = _tool("write_card", "写下这张卡片。", {"card": {"type": "string"}})
PERSON_SYSTEM = """下面是「我」和群友「{name}」在群里的一些对话片段，「我:」开头的是我说的。
请替我写一张关于这个人的小卡片，调用 write_card。

要求：
- 第一人称，写我对他的印象，像我自己心里记着的。
- 内容：我一般怎么称呼他；我们的关系和我跟他说话的分寸（客气、互损、照顾之类）；我们常聊什么；我知道的关于他的两三件事。
- 不写他的真实姓名、住址、电话、工作单位、健康、财务、家庭这类隐私。
- 片段里看不出来的不写，不编造。
- 120 字以内，一段话，不分点。"""


@dataclass
class Person:
    user_id: int
    name: str
    pair_ids: list[int]

    @property
    def replies(self) -> int:
        return len(self.pair_ids)


def rank_people(db: CorpusDB, chat_id: int, years: float = 3.0) -> list[Person]:
    """The people I answer in this chat, most answered first, from the training period."""
    newest = db.db.execute(
        "SELECT MAX(ts) FROM pair WHERE chat_id = ? AND split = 'train'", (chat_id,)
    ).fetchone()[0]
    if newest is None:
        return []
    since = newest - int(years * 365 * 86400)
    people: dict[int, Person] = {}
    for pair in db.pairs("train", chat_id):
        if pair.ts < since:
            continue
        target = None
        if pair.quoted:
            row = db.db.execute(
                "SELECT t.sender_id, t.sender_name, t.is_me FROM msg m JOIN msg t"
                " ON t.chat_id = m.chat_id AND t.msg_id = m.reply_to"
                " WHERE m.chat_id = ? AND m.msg_id = ?",
                (chat_id, pair.first_id),
            ).fetchone()
            if row is not None and not row[2] and row[0] is not None:
                target = (row[0], row[1])
        if target is None:
            row = db.db.execute(
                "SELECT sender_id, sender_name FROM msg WHERE chat_id = ? AND msg_id < ?"
                " AND msg_id >= ? AND is_me = 0 AND sender_id IS NOT NULL"
                " ORDER BY msg_id DESC LIMIT 1",
                (chat_id, pair.first_id, pair.ctx_first_id),
            ).fetchone()
            if row is not None:
                target = (row[0], row[1])
        if target is None:
            continue
        person = people.setdefault(target[0], Person(target[0], target[1], []))
        person.name = target[1]  # pairs come oldest first, so this ends as the latest name
        person.pair_ids.append(pair.pair_id)
    return sorted(people.values(), key=lambda p: -p.replies)


def choose_people(
    ranked: Sequence[Person], coverage: float = 0.8, cap: int = 40, min_replies: int = 15
) -> tuple[list[Person], float]:
    """The most answered people up to `coverage` of my replies. Returns (people, share covered)."""
    total = sum(person.replies for person in ranked)
    chosen: list[Person] = []
    covered = 0
    for person in ranked:
        if len(chosen) >= cap or person.replies < min_replies:
            break
        if total and covered / total >= coverage:
            break
        chosen.append(person)
        covered += person.replies
    return chosen, (covered / total if total else 0.0)


def person_scenes(db: CorpusDB, person: Person, limit: int = 60, max_chars: int = 9000, seed: int = 0) -> str:
    """Moments where I answered this person: the newest half of `limit`, the rest at random."""
    rng = random.Random(seed * 1_000_003 + person.user_id)
    recent = person.pair_ids[-(limit // 2) :]
    older = person.pair_ids[: -(limit // 2)]
    picked = sorted(rng.sample(older, min(len(older), limit - len(recent))) + recent)
    scenes: list[str] = []
    size = 0
    for pair_id in reversed(picked):  # if space runs out, the oldest are the ones dropped
        pair = db.pair(pair_id)
        if pair is None:
            continue
        scene = render_scene(db.context_of(pair), db.reply_of(pair), tail=4)
        if size + len(scene) > max_chars:
            break
        scenes.append(scene)
        size += len(scene) + 2
    return "\n\n".join(reversed(scenes))


async def write_people(
    model: ChatModel,
    db: CorpusDB,
    chat_id: int,
    people: Sequence[Person],
    persona_dir: Path,
    force: bool = False,
    concurrency: int = 4,
    progress: Progress = lambda line: None,
) -> dict[str, int]:
    """One card per person under people/<chat>/<user>.md. Existing cards are kept."""
    folder = Path(persona_dir) / "people" / str(chat_id)
    folder.mkdir(parents=True, exist_ok=True)
    todo = [p for p in people if force or not (folder / f"{p.user_id}.md").is_file()]
    rejected = 0

    def job(person: Person) -> Callable[[], Awaitable[bool]]:
        async def run() -> bool:
            nonlocal rejected
            name = clean_name(person.name)
            args = await ask(model, PERSON_SYSTEM.format(name=name), person_scenes(db, person), CARD_TOOL)
            card = " ".join(str(args.get("card", "")).split()) if args else ""
            if not card:
                return False
            if lint(card, CARD_MAX_TOKENS):
                rejected += 1
                return True
            (folder / f"{person.user_id}.md").write_text(card + "\n", encoding="utf-8")
            return True

        return run

    done, failed = await run_jobs([job(person) for person in todo], concurrency, progress)
    return {
        "kept": len(people) - len(todo),
        "written": done - rejected,
        "rejected": rejected,
        "failed": failed,
    }


# -- groups -------------------------------------------------------------------

GROUP_NOTE_TOOL = _tool(
    "note_group",
    "记下这段聊天里看得出的群的特点。",
    {
        "topics": {"type": "array", "items": {"type": "string"}},
        "jargon": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"term": {"type": "string"}, "meaning": {"type": "string"}},
                "required": ["term", "meaning"],
            },
        },
        "vibe": {"type": "string"},
    },
)
GROUP_NOTE_SYSTEM = """下面是一个群里的一段聊天记录，「我:」开头的是我说的。请记下这段里看得出的群的特点，调用 note_group。

- topics：这段在聊的话题，每个几个字。
- jargon：群里特有的说法、梗、外号（term）和它的意思（meaning）。普通的网络用语不算。
- vibe：用一句话说这个群的气氛。

看不出来的就留空。"""
GROUP_CARD_SYSTEM = """下面是从同一个群的多段聊天里记下的笔记：话题出现的次数、群里的说法和意思、对气氛的描述。
请替我写一张关于这个群的卡片，调用 write_card。

要求：
- 第一人称：这个群对我来说是什么地方、大家常聊什么、气氛怎样。
- 然后列出反复出现的群内说法，每条写成「说法：意思」。
- 只写在多段笔记里反复出现的内容，不写某个人的隐私。
- 250 字以内。"""


def group_windows(
    db: CorpusDB, chat_id: int, count: int = 40, size: int = 120, years: float = 2.0, seed: int = 0, max_chars: int = 7000
) -> list[str]:
    """Stretches of this chat from the recent training period, as rendered lines."""
    cutoff = train_cutoffs(db)[chat_id]
    newest = db.db.execute(
        "SELECT MAX(ts) FROM msg WHERE chat_id = ? AND ts < ?", (chat_id, cutoff)
    ).fetchone()[0]
    if newest is None:
        return []
    since = newest - int(years * 365 * 86400)
    ids = [
        row[0]
        for row in db.db.execute(
            "SELECT msg_id FROM msg WHERE chat_id = ? AND ts >= ? AND ts < ? ORDER BY msg_id",
            (chat_id, since, cutoff),
        )
    ]
    if len(ids) < size:
        return []
    rng = random.Random(seed * 1_000_003 + chat_id)
    starts = sorted(rng.sample(range(0, len(ids) - size + 1), min(count, len(ids) - size + 1)))
    windows = []
    for start in starts:
        lines = render_lines(db.span(chat_id, ids[start], ids[start + size - 1]))
        windows.append("\n".join(lines)[:max_chars])
    return [window for window in windows if window.strip()]


def summarize_notes(notes: Sequence[dict[str, Any]], max_chars: int = 9000) -> str:
    """Counts of topics and in-group sayings across all the notes, as text for the model."""
    topics: Counter[str] = Counter()
    jargon: dict[str, Counter[str]] = {}
    vibes: list[str] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        for topic in note.get("topics") or []:
            if isinstance(topic, str) and topic.strip():
                topics[topic.strip()[:20]] += 1
        for entry in note.get("jargon") or []:
            if isinstance(entry, dict) and str(entry.get("term", "")).strip():
                term = str(entry["term"]).strip()[:20]
                jargon.setdefault(term, Counter())[str(entry.get("meaning", "")).strip()[:60]] += 1
        vibe = str(note.get("vibe") or "").strip()
        if vibe:
            vibes.append(vibe[:80])
    ranked = sorted(jargon.items(), key=lambda item: -sum(item[1].values()))
    text = "\n".join(
        [
            "话题（出现次数）：",
            *(f"{topic} {count}" for topic, count in topics.most_common(60)),
            "",
            "群里的说法（出现次数）：",
            *(f"{term} {sum(m.values())}：{m.most_common(1)[0][0]}" for term, m in ranked[:60]),
            "",
            "气氛：",
            *vibes[:30],
        ]
    )
    return text[:max_chars]


async def write_group(
    model: ChatModel,
    slow: ChatModel,
    chat_id: int,
    windows: Sequence[str],
    notes_dir: Path,
    persona_dir: Path,
    concurrency: int = 4,
    progress: Progress = lambda line: None,
) -> dict[str, Any]:
    """Note what each window shows, then write groups/<chat>.md from all the notes."""
    notes_dir.mkdir(parents=True, exist_ok=True)
    todo = [window for window in windows if not (notes_dir / f"{_digest(window)}.json").is_file()]

    def job(window: str) -> Callable[[], Awaitable[bool]]:
        async def run() -> bool:
            args = await ask(model, GROUP_NOTE_SYSTEM, window, GROUP_NOTE_TOOL)
            if args is None:
                return False
            _save_json(notes_dir / f"{_digest(window)}.json", args)
            return True

        return run

    done, failed = await run_jobs([job(window) for window in todo], concurrency, progress)
    notes = [_load_json(notes_dir / f"{_digest(window)}.json") for window in windows]
    notes = [note for note in notes if isinstance(note, dict)]
    result: dict[str, Any] = {"windows": len(windows), "noted": len(notes), "failed": failed, "written": False}
    if not notes:
        return result
    args = await ask(slow, GROUP_CARD_SYSTEM, summarize_notes(notes), CARD_TOOL)
    card = str(args.get("card", "")).strip() if args else ""
    problems = lint(card, GROUP_MAX_TOKENS)
    result["problems"] = problems
    if card and not problems:
        path = Path(persona_dir) / "groups" / f"{chat_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(card + "\n", encoding="utf-8")
        result["written"] = True
        result["tokens"] = estimate_tokens(card)
    return result
