"""How well the exam check tells questions that test me as an AI from talk.

Two sets. Made-up messages with known answers, packaged with the code and safe
to show. And messages people really sent me before there was a bot: nobody was
testing an AI then, so every yes among them is a false alarm. Of those only
counts are printed; the flagged texts go to a private report.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.eval.run import gather_bounded
from shinnku_bot.harness.exam import BEFORE_LINES, judge
from shinnku_bot.llm import ChatModel
from shinnku_bot.store import Msg

SET_PATH = Path(__file__).with_name("exam_set.jsonl")
# The kinds of the made-up set, in report order. The last two are talk.
KINDS = ("contest", "judge", "escalate", "identity", "trick", "meme", "order", "easy", "hard")
# The long puzzles, where a miss costs the most thinking.
LONG = ("contest", "judge", "escalate")

# The message, what it replies to, the lines before it.
Case = tuple[Msg, Msg | None, list[Msg]]
# The verdict (None when the call failed) and how long it took.
Result = tuple[bool | None, float]


def load_set(path: Path = SET_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _case(item: dict[str, Any]) -> Case:
    msg = Msg(chat_id=0, message_id=2, ts=0, user_id=1, user_name="某人", is_me=False, text=item["text"])
    replied = item.get("replied")
    mine = Msg(chat_id=0, message_id=1, ts=0, user_id=0, user_name="我", is_me=True, text=replied) if replied else None
    return msg, mine, []


async def _verdicts(model: ChatModel, cases: Sequence[Case], concurrency: int) -> list[Result]:
    async def one(case: Case) -> Result:
        started = time.monotonic()
        try:
            verdict: bool | None = await judge(model, *case)
        except Exception:  # noqa: BLE001 - counted as an error
            verdict = None
        return verdict, time.monotonic() - started

    return await gather_bounded(
        cases, one, failed=lambda r: r[0] is None, skipped=lambda _: (None, 0.0), concurrency=concurrency
    )


def _p50(results: Sequence[Result]) -> float:
    times = [seconds for verdict, seconds in results if verdict is not None]
    return round(statistics.median(times), 1) if times else 0.0


async def check_set(model: ChatModel, items: Sequence[dict[str, Any]], concurrency: int = 4) -> dict[str, Any]:
    """Per kind, how many were judged exams; and the items judged wrong."""
    results = await _verdicts(model, [_case(item) for item in items], concurrency)
    kinds: dict[str, dict[str, Any]] = {}
    wrong = []
    for item, (verdict, _) in zip(items, results):
        row = kinds.setdefault(item["kind"], {"exam": item["exam"], "n": 0, "yes": 0, "errors": 0})
        row["n"] += 1
        if verdict is None:
            row["errors"] += 1
            continue
        row["yes"] += verdict
        if verdict != item["exam"]:
            wrong.append(item)
    return {"kinds": kinds, "wrong": wrong, "p50_s": _p50(results)}


def format_set(report: dict[str, Any]) -> str:
    kinds = report["kinds"]

    def total(names: Sequence[str]) -> str:
        rows = [kinds[name] for name in names if name in kinds]
        return f"{sum(r['yes'] for r in rows)}/{sum(r['n'] for r in rows)}"

    lines = ["exam set: judged exams per kind"]
    for name in sorted(kinds, key=lambda k: KINDS.index(k) if k in KINDS else len(KINDS)):
        row = kinds[name]
        errors = f", {row['errors']} errors" if row["errors"] else ""
        lines.append(f"  {name:<9} {row['yes']}/{row['n']}{errors}")
    exams = [name for name, row in kinds.items() if row["exam"]]
    lines.append(
        f"caught {total(exams)}, long puzzles {total(LONG)}; "
        f"false alarms easy {total(['easy'])}, hard {total(['hard'])}; judge p50 {report['p50_s']}s"
    )
    for item in report["wrong"]:
        text = " ".join(item["text"].split())
        lines.append(f"  wrong {item['kind']}: {text[:60]}")
    return "\n".join(lines)


def addressed(db: CorpusDB, names: Sequence[str], n: int, seed: int = 0) -> list[Case]:
    """`n` random messages from others that reply to me or call me by name."""
    called = " OR ".join("instr(lower(m.text), ?) > 0" for _ in names) or "0"
    rows = db.db.execute(
        "SELECT m.chat_id, m.msg_id FROM msg m"
        " LEFT JOIN msg r ON r.chat_id = m.chat_id AND r.msg_id = m.reply_to"
        " WHERE m.is_me = 0 AND m.canned = 0 AND m.kind = 'text' AND m.text != ''"
        f" AND (r.is_me = 1 OR {called}) ORDER BY m.chat_id, m.msg_id",
        [name.lower() for name in names],
    ).fetchall()
    cases = []
    for chat_id, msg_id in random.Random(seed).sample(rows, min(n, len(rows))):
        context = db.before(chat_id, msg_id, BEFORE_LINES + 1)
        msg = context[-1]
        replied = db.span(chat_id, msg.reply_to, msg.reply_to) if msg.reply_to is not None else []
        cases.append((msg, replied[0] if replied else None, context[:-1]))
    return cases


async def check_real(
    model: ChatModel, db: CorpusDB, names: Sequence[str], n: int, concurrency: int = 4
) -> tuple[dict[str, Any], list[str]]:
    """Counts over real messages that called me, and the texts of the flagged ones."""
    cases = addressed(db, names, n)
    results = await _verdicts(model, cases, concurrency)
    flagged = [case[0].text for case, (verdict, _) in zip(cases, results) if verdict]
    counts = {
        "n": len(cases),
        "flagged": len(flagged),
        "errors": sum(verdict is None for verdict, _ in results),
        "p50_s": _p50(results),
    }
    return counts, flagged
