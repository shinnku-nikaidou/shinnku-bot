"""The moments where I could have joined a conversation, and whether I did.

A moment is the point right after someone else's message when the chat pauses
and nobody has called me. It is positive if I spoke next, negative if I was
around (I had spoken within twenty minutes either side) and let it pass.
Moments while I was away teach nothing about choice, so the prior is neither
trained nor scored on them; the held-out ones are kept only to estimate how
often a live bot, which is always there, would end up asking the model.

No model calls. The console gets counts and rates only.
"""

from __future__ import annotations

import bisect
import json
import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.pairs import train_cutoffs
from shinnku_bot.eval.classify import LogReg, auc
from shinnku_bot.harness.transcript import render_body
from shinnku_bot.harness.urge import FEATURES, LEVEL_FACTOR, UrgeModel, moment_features
from shinnku_bot.store import Msg

SENDER_PRIOR = 20
MIN_SENDER_MOMENTS = 10
KEEP_PRIOR_ABOVE = 0.60
MIN_TEST_POSITIVES = 30


@dataclass(frozen=True)
class MomentParams:
    # The live bot waits about this long for the chat to pause before it considers speaking.
    quiet_s: int = 3
    answer_within_s: int = 120
    present_s: int = 20 * 60
    context_messages: int = 30
    context_max_age_s: int = 6 * 3600


@dataclass
class Moment:
    chat_id: int
    msg_id: int
    ts: int
    sender_id: int | None
    spoke: bool
    test: bool
    present: bool
    features: list[float]


def collect(
    db: CorpusDB,
    chat_id: int,
    tz: ZoneInfo,
    hour_share: Sequence[float],
    name_triggers: Sequence[str] = (),
    params: MomentParams = MomentParams(),
) -> tuple[list[Moment], dict[str, int]]:
    """Every moment of one chat, plus counts of how my speaking divides up."""
    cutoff = train_cutoffs(db)[chat_id]
    mine = db.db.execute(
        "SELECT msg_id, ts FROM msg WHERE chat_id = ? AND is_me = 1 ORDER BY ts", (chat_id,)
    ).fetchall()
    my_ids = {row[0] for row in mine}
    my_ts = [row[1] for row in mine]
    triggers = [name.lower() for name in name_triggers if name]

    def present(ts: int) -> bool:
        index = bisect.bisect_left(my_ts, ts - params.present_s)
        return index < len(my_ts) and my_ts[index] <= ts + params.present_s

    def called(msg: Msg) -> bool:
        text = msg.text.lower()
        return msg.reply_to in my_ids or any(name in text for name in triggers)

    moments: list[Moment] = []
    counts = {"joined_uncalled": 0, "answered_when_called": 0, "pauses": 0}
    window: deque[Msg] = deque(maxlen=params.context_messages)
    prev: Msg | None = None
    for row in db.rows(chat_id):
        cur = row.msg
        if prev is not None and not prev.is_me and render_body(prev):
            gap = cur.ts - prev.ts
            spoke = cur.is_me and gap <= params.answer_within_s
            if cur.is_me or gap >= params.quiet_s:
                counts["pauses"] += 1
                if called(prev):
                    counts["answered_when_called"] += spoke
                else:
                    counts["joined_uncalled"] += spoke
                    # Held out if its outcome (the next message) falls in the held-out period.
                    here, held_out = present(prev.ts), cur.ts >= cutoff
                    if here or held_out:
                        msgs = [m for m in window if prev.ts - m.ts <= params.context_max_age_s]
                        moments.append(
                            Moment(
                                chat_id,
                                prev.message_id,
                                prev.ts,
                                prev.user_id,
                                spoke,
                                held_out,
                                here,
                                moment_features(msgs, hour_share, {}, 0.0, tz),
                            )
                        )
        window.append(cur)
        prev = cur
    return moments, counts


def sender_rates(moments: Sequence[Moment], base: float) -> dict[int, float]:
    """How often I join in after each person, pulled toward my overall rate when data is thin."""
    seen: dict[int, list[int]] = {}
    for moment in moments:
        if moment.sender_id is not None:
            tally = seen.setdefault(moment.sender_id, [0, 0])
            tally[0] += moment.spoke
            tally[1] += 1
    return {
        sender: round((joined + SENDER_PRIOR * base) / (total + SENDER_PRIOR), 4)
        for sender, (joined, total) in seen.items()
        if total >= MIN_SENDER_MOMENTS
    }


def _matrix(moments: Sequence[Moment]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([m.features for m in moments], dtype=np.float64).reshape(len(moments), len(FEATURES))
    y = np.asarray([m.spoke for m in moments], dtype=np.float64)
    return x, y


def build_urge(
    db: CorpusDB,
    timezone: str,
    hour_share: Sequence[float] | None = None,
    name_triggers: Sequence[str] = (),
    max_train: int = 20000,
    max_test: int = 6000,
    seed: int = 0,
    params: MomentParams = MomentParams(),
) -> tuple[UrgeModel, dict[str, Any], list[Moment]]:
    """Learn when I join in. Returns (model, numbers for the console, held-out moments)."""
    tz = ZoneInfo(timezone)
    hours = list(hour_share) if hour_share and len(hour_share) == 24 else [1 / 24] * 24
    moments: list[Moment] = []
    counts = {"joined_uncalled": 0, "answered_when_called": 0, "pauses": 0}
    for chat_id in db.chat_ids():
        found, chat_counts = collect(db, chat_id, tz, hours, name_triggers, params)
        moments.extend(found)
        for key, value in chat_counts.items():
            counts[key] += value

    train = [m for m in moments if not m.test]
    test = [m for m in moments if m.test]
    around = sum(m.present for m in test)
    base = sum(m.spoke for m in train) / len(train) if train else 0.0
    rates = sender_rates(train, base)
    rate_at = FEATURES.index("sender_rate")
    for moment in moments:
        moment.features[rate_at] = rates.get(moment.sender_id, base) if moment.sender_id is not None else base

    rng = random.Random(seed)
    train_sample = rng.sample(train, min(len(train), max_train))
    test_sample = rng.sample(test, min(len(test), max_test))
    report: dict[str, Any] = {
        **counts,
        "moments_while_present": len(train) + around,
        "train": len(train),
        "test": around,
        "base_rate": round(base, 4),
        "people_with_own_rate": len(rates),
        "auc": None,
        "prior_kept": False,
    }
    learned: dict[str, Any] = {}
    scored = [m for m in test_sample if m.present]
    if len(train_sample) >= 200 and 0 < sum(m.spoke for m in train_sample) < len(train_sample):
        fitted = LogReg(l2=1.0, steps=2000, scaled=len(FEATURES)).fit(*_matrix(train_sample))
        report["weights"] = {name: round(float(w), 3) for name, w in zip(FEATURES, fitted.weights)}
        if sum(m.spoke for m in scored) >= MIN_TEST_POSITIVES:
            test_x, test_y = _matrix(scored)
            report["auc"] = round(auc(fitted.decision(test_x), test_y), 3)
        if report["auc"] is not None and report["auc"] >= KEEP_PRIOR_ABOVE:
            report["prior_kept"] = True
            learned = {
                "auc": report["auc"],
                "weights": [float(w) for w in fitted.weights],
                "bias": float(fitted.bias),
                "mean": [float(v) for v in fitted.mean],
                "std": [float(v) for v in fitted.std],
            }
    model = UrgeModel(
        base_rate=round(base, 4),
        hour_share=[round(float(h), 5) for h in hours],
        sender_rate=rates,
        default_sender_rate=round(base, 4),
        timezone=timezone,
        **learned,
    )

    if test:
        days = max((max(m.ts for m in test) - min(m.ts for m in test)) / 86400, 1.0)
        report["test_days"] = round(days, 1)
        report["moments_per_day"] = round(len(test) / days, 1)
        report["i_joined_per_day"] = round(sum(m.spoke for m in test) / days, 1)
        for level in (1, 2, 3):
            share = float(np.mean([model.call_probability(m.features, level) for m in test_sample]))
            report[f"calls_per_day_level{level}"] = round(share * len(test) / days, 1)
        report["level_factor"] = dict(LEVEL_FACTOR)
    return model, report, test_sample


def save_moments(path: Path, moments: Sequence[Moment]) -> None:
    """Ids and labels only, so the offline check can replay these moments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for m in moments:
            handle.write(json.dumps({"chat_id": m.chat_id, "msg_id": m.msg_id, "spoke": m.spoke}) + "\n")


def load_moments(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError):
        return []


def format_urge(report: dict[str, Any]) -> str:
    called, uncalled = report["answered_when_called"], report["joined_uncalled"]
    total = max(called + uncalled, 1)
    lines = [
        f"pauses in the chats: {report['pauses']}",
        f"I spoke next: {called + uncalled}  (called {called / total:.1%}, uncalled {uncalled / total:.1%})",
        f"moments while I was around: {report['moments_while_present']}"
        f"  (train {report['train']}, held out {report['test']})",
        f"I joined in at {report['base_rate']:.1%} of them; {report['people_with_own_rate']} people have their own rate",
        f"prior AUC on held-out moments: {report['auc']}  kept: {report['prior_kept']}",
    ]
    if "weights" in report:
        lines.append(f"weights: {report['weights']}")
    if "moments_per_day" in report:
        lines.append(
            f"held-out period: {report['test_days']} days, {report['moments_per_day']} pauses/day,"
            f" I joined {report['i_joined_per_day']}/day"
        )
        lines.append(
            "expected model calls per day at /talk 1, 2, 3: "
            + ", ".join(str(report[f"calls_per_day_level{level}"]) for level in (1, 2, 3))
        )
    return "\n".join(lines)
