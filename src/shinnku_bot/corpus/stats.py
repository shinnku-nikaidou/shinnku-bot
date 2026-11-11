"""Measure how I write: the style fingerprint.

The returned dict holds numbers plus two lists of actual strings (`phrases`,
`sticker_emoji`). Callers print the numbers and keep the strings in the file.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.pairs import train_cutoffs
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.send import Pace

_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")
_KANA = re.compile("[぀-ヿ]")
_ASCII_WORD = re.compile(r"[A-Za-z]{2,}")
_URL = re.compile(r"https?://")
_CJK_SPACE_CJK = re.compile("[一-鿿] +[一-鿿]")
_ENDINGS = {
    "period": re.compile(r"。$"),
    "question": re.compile(r"[?？]$"),
    "exclaim": re.compile(r"[!！]$"),
    "tilde": re.compile(r"[~～]$"),
    "ellipsis": re.compile(r"(…|\.\.\.|。。。)$"),
}
_CONTAINS = {
    "emoji": _EMOJI,
    "kana": _KANA,
    "ascii_word": _ASCII_WORD,
    "url": _URL,
    "comma": re.compile("[，,]"),
    "space_between_cjk": _CJK_SPACE_CJK,
    "newline": re.compile("\n"),
}
MAX_OTHERS = 300_000
MIN_PHRASE_COUNT = 30


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted list; 0 for an empty one."""
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, math.ceil(q / 100 * len(sorted_values)) - 1))
    return sorted_values[index]


def _summary(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        "mean": round(sum(values) / len(values), 2) if values else 0,
        **{f"p{q}": percentile(values, q) for q in (50, 90, 95, 99)},
    }


def _ngrams(text: str) -> set[str]:
    compact = "".join(text.split())
    return {compact[i : i + n] for n in (2, 3, 4) for i in range(len(compact) - n + 1)}


def build_fingerprint(db: CorpusDB, timezone: str = "Asia/Shanghai") -> dict[str, Any]:
    """Statistics over my messages from before each chat's held-out period."""
    tz = ZoneInfo(timezone)
    cutoffs = train_cutoffs(db)

    kinds: Counter = Counter()
    lengths: list[float] = []
    endings: Counter = Counter()
    contains: Counter = Counter()
    hours = [0] * 24
    stickers: Counter = Counter()
    mine: Counter = Counter()

    for chat_id, ts, kind, text, canned in db.db.execute(
        "SELECT chat_id, ts, kind, text, canned FROM msg WHERE is_me = 1"
    ):
        if ts >= cutoffs[chat_id]:
            continue
        kinds[kind] += 1
        hours[datetime.fromtimestamp(ts, tz).hour] += 1
        if kind == "sticker" and text:
            stickers[text] += 1
        if kind != "text" or canned or not text.strip():
            continue
        text = text.strip()
        lengths.append(len(text))
        for name, pattern in _ENDINGS.items():
            endings[name] += bool(pattern.search(text))
        for name, pattern in _CONTAINS.items():
            contains[name] += bool(pattern.search(text))
        mine.update(_ngrams(text))

    usable = len(lengths)
    total = sum(kinds.values())

    others: Counter = Counter()
    n_others = db.count("is_me = 0 AND kind = 'text' AND canned = 0")
    step = max(1, n_others // MAX_OTHERS)
    sampled = 0
    for (text,) in db.db.execute(
        "SELECT text FROM msg WHERE is_me = 0 AND kind = 'text' AND canned = 0 AND msg_id % ? = 0",
        (step,),
    ):
        others.update(_ngrams(text))
        sampled += 1

    phrases = []
    if usable and sampled:
        for gram, count in mine.items():
            if count < MIN_PHRASE_COUNT:
                continue
            lift = (count / usable) / ((others.get(gram, 0) + 0.5) / sampled)
            phrases.append((lift * math.log(count), gram, count, round(lift, 1)))
        phrases.sort(reverse=True)

    bursts = [r[0] for r in db.db.execute("SELECT n_reply FROM pair WHERE split = 'train'")]
    quoted = db.db.execute("SELECT COALESCE(AVG(quoted), 0) FROM pair WHERE split = 'train'").fetchone()[0]
    burst_dist = Counter(min(n, 5) for n in bursts)

    gaps: list[float] = []
    per_char: list[float] = []
    sample = db.db.execute(
        "SELECT chat_id, first_id, last_id FROM pair WHERE split = 'train' AND n_reply > 1"
        " ORDER BY pair_id DESC LIMIT 20000"
    ).fetchall()
    for chat_id, first_id, last_id in sample:
        lines = db.span(chat_id, first_id, last_id)
        for prev, cur in zip(lines, lines[1:]):
            gap = cur.ts - prev.ts
            gaps.append(gap)
            if cur.text.strip():
                per_char.append(gap / len(cur.text.strip()))

    def rate(counter: Counter, name: str) -> float:
        return round(counter[name] / usable, 4) if usable else 0

    return {
        "messages": total,
        "text_messages": usable,
        "bursts": len(bursts),
        "len": _summary(lengths),
        "burst_size": {
            **{("5+" if n == 5 else str(n)): round(burst_dist[n] / len(bursts), 4) if bursts else 0 for n in range(1, 6)},
            **_summary([float(n) for n in bursts]),
        },
        "ends_with": {name: rate(endings, name) for name in _ENDINGS},
        "contains": {name: rate(contains, name) for name in _CONTAINS},
        "kind_share": {kind: round(count / total, 4) for kind, count in kinds.most_common()} if total else {},
        "quote_rate": round(quoted, 4),
        "hour_share": [round(h / total, 4) if total else 0 for h in hours],
        "burst_gap_s": _summary(gaps),
        "sec_per_char": _summary(per_char),
        "timezone": timezone,
        # Actual strings below: kept in the file, never printed.
        "phrases": [{"text": gram, "count": count, "lift": lift} for _, gram, count, lift in phrases[:300]],
        "sticker_emoji": [{"emoji": emoji, "count": count} for emoji, count in stickers.most_common(40)],
    }


def format_fingerprint(fp: dict[str, Any]) -> str:
    """The numeric part of a fingerprint, for the console."""
    hours = fp["hour_share"]
    busiest = sorted(range(24), key=lambda h: -hours[h])[:5]
    return "\n".join(
        [
            f"my messages: {fp['messages']}  usable text: {fp['text_messages']}  bursts: {fp['bursts']}",
            f"length (chars): {fp['len']}",
            f"burst size: {fp['burst_size']}",
            f"ends with: {fp['ends_with']}",
            f"contains: {fp['contains']}",
            f"kinds: {fp['kind_share']}",
            f"quotes the message it answers: {fp['quote_rate']:.1%}",
            f"gap inside a burst (s): {fp['burst_gap_s']}",
            f"seconds per character: {fp['sec_per_char']}",
            f"busiest hours ({fp['timezone']}): {sorted(busiest)}",
            f"distinctive phrases kept in file: {len(fp['phrases'])}",
        ]
    )


def _lengths(values: list[int]) -> dict[str, float]:
    ordered = sorted(float(v) for v in values)
    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 1) if ordered else 0,
        "median": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "over_30": round(sum(v > 30 for v in ordered) / len(ordered), 3) if ordered else 0,
    }


def length_report(db: CorpusDB, anchor_text: str = "") -> dict[str, dict[str, float]]:
    """How long my messages are, year by year, in the held-out weeks, and in the fixed sample."""
    by_year: dict[str, list[int]] = {}
    held_out: list[int] = []
    cutoffs = train_cutoffs(db)
    rows = db.db.execute(
        "SELECT chat_id, ts, text FROM msg WHERE is_me = 1 AND kind = 'text' AND canned = 0"
    )
    for chat_id, ts, text in rows:
        size = len(text.strip())
        if not size:
            continue
        by_year.setdefault(str(datetime.fromtimestamp(ts, ZoneInfo("UTC")).year), []).append(size)
        if ts >= cutoffs.get(chat_id, 2**62):
            held_out.append(size)
    report = {year: _lengths(sizes) for year, sizes in sorted(by_year.items())}
    report["held-out weeks"] = _lengths(held_out)
    mine = [len(line[3:].strip()) for line in anchor_text.splitlines() if line.startswith("我: ")]
    report["fixed sample"] = _lengths(mine)
    return report


def save_fingerprint(fp: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")


def limits_from_fingerprint(path: Path) -> StyleLimits:
    """Send-time limits measured from my own messages, or the defaults."""
    try:
        fp = json.loads(Path(path).read_text(encoding="utf-8"))
        return StyleLimits(
            max_len=int(min(max(fp["len"]["p99"], 30), 300)),
            max_burst=int(min(max(fp["burst_size"]["p99"], 3), 8)),
            strip_period=fp["ends_with"]["period"] < 0.05,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return StyleLimits()


def pace_from_fingerprint(path: Path) -> Pace:
    """My typing pace, sped up: people forgive a bot for being a little quick.

    The measured gap between my lines includes thinking time, so it is scaled
    down and capped rather than replayed as is.
    """
    try:
        fp = json.loads(Path(path).read_text(encoding="utf-8"))
        return Pace(
            sec_per_char=float(min(max(fp["sec_per_char"]["p50"] * 0.4, 0.1), 0.6)),
            floor_s=1.0,
            cap_s=float(min(max(fp["burst_gap_s"]["p50"], 4.0), 10.0)),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return Pace()
