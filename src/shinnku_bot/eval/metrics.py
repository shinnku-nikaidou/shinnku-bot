"""Aggregate an evaluation run into numbers."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

# Words that show the model is thinking about playing someone, not being them.
OUT_OF_CHARACTER = re.compile(
    r"作为|扮演|角色|人设|模仿|用户|助手|\bAI\b|persona|role[- ]?play|character|the user", re.IGNORECASE
)
_LENGTH_BINS = (2, 5, 10, 20, 40)


@dataclass
class Item:
    """One held-out moment: what I really said and what the bot said."""

    pair_id: int
    chat_id: int
    ts: int
    context: list[str] = field(default_factory=list)
    real: list[str] = field(default_factory=list)
    bot: list[str] = field(default_factory=list)
    kind: str = ""
    violations: list[str] = field(default_factory=list)
    attempts: int = 0
    reasoning: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    recalled: int = 0
    error: str = ""
    # "real" if the judge picked my real reply, "bot" if it was fooled, "" if not judged.
    judged: str = ""
    # Meaning-level closeness of the bot's reply to mine, and to someone else's moment (chance).
    sim_real: float | None = None
    sim_other: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    denom = 1 + z * z / total
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))]


def _length_histogram(messages: list[str]) -> list[float]:
    counts = Counter()
    for text in messages:
        counts[sum(len(text) > edge for edge in _LENGTH_BINS)] += 1
    total = sum(counts.values()) or 1
    return [counts[i] / total for i in range(len(_LENGTH_BINS) + 1)]


def js_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence in bits: 0 for identical distributions, 1 for disjoint."""

    def kl(a: list[float], b: list[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)

    mid = [(x + y) / 2 for x, y in zip(p, q)]
    return (kl(p, mid) + kl(q, mid)) / 2


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(items: list[Item]) -> dict[str, Any]:
    done = [item for item in items if not item.error]
    spoke = [item for item in done if item.bot]
    judged = [item for item in spoke if item.judged]
    picked_real = sum(item.judged == "real" for item in judged)
    low, high = wilson(picked_real, len(judged))

    real_msgs = [text for item in spoke for text in item.real]
    bot_msgs = [text for item in spoke for text in item.bot]
    with_reasoning = [item for item in done if item.reasoning]
    violations = Counter(name for item in done for name in item.violations)
    prompt = sum(item.prompt_tokens for item in done)

    return {
        "items": len(items),
        "errors": len(items) - len(done),
        "error_types": dict(Counter(item.error for item in items if item.error)),
        "spoke": len(spoke),
        "silent_rate": round(1 - len(spoke) / len(done), 4) if done else 0.0,
        "judged": len(judged),
        # 0.5 means the judge cannot tell; 1.0 means it always spots the bot.
        "judge_accuracy": round(picked_real / len(judged), 4) if judged else None,
        "judge_ci95": [round(low, 4), round(high, 4)],
        "exact_match_rate": round(_mean([float(item.bot == item.real) for item in spoke]), 4),
        "len_mean": {"real": round(_mean([float(len(t)) for t in real_msgs]), 2), "bot": round(_mean([float(len(t)) for t in bot_msgs]), 2)},
        "len_p50": {"real": percentile([float(len(t)) for t in real_msgs], 50), "bot": percentile([float(len(t)) for t in bot_msgs], 50)},
        "burst_mean": {"real": round(_mean([float(len(i.real)) for i in spoke]), 2), "bot": round(_mean([float(len(i.bot)) for i in spoke]), 2)},
        "length_js": round(js_divergence(_length_histogram(real_msgs), _length_histogram(bot_msgs)), 4),
        "out_of_character_rate": round(
            _mean([float(bool(OUT_OF_CHARACTER.search(item.reasoning))) for item in with_reasoning]), 4
        ),
        "retry_rate": round(_mean([float(item.attempts > 1) for item in done]), 4),
        "violations": dict(violations),
        "latency_p50": round(percentile([item.latency_s for item in done], 50), 2),
        "latency_p95": round(percentile([item.latency_s for item in done], 95), 2),
        "prompt_tokens_mean": round(_mean([float(item.prompt_tokens) for item in done])),
        "cached_share": round(sum(item.cached_tokens for item in done) / prompt, 4) if prompt else 0.0,
        "recalled_mean": round(_mean([float(item.recalled) for item in done]), 1),
        # How close the bot's reply is in meaning to my real one, against the chance level.
        "sim_real": round(_mean([i.sim_real for i in spoke if i.sim_real is not None]), 4),
        "sim_chance": round(_mean([i.sim_other for i in spoke if i.sim_other is not None]), 4),
    }


def format_summary(name: str, summary: dict[str, Any]) -> str:
    accuracy = summary["judge_accuracy"]
    verdict = "n/a" if accuracy is None else f"{accuracy:.1%} (95% CI {summary['judge_ci95'][0]:.1%}-{summary['judge_ci95'][1]:.1%})"
    return "\n".join(
        [
            f"== {name}",
            f"items {summary['items']}  errors {summary['errors']} {summary['error_types'] or ''}  spoke {summary['spoke']}  silent {summary['silent_rate']:.1%}",
            f"judge picks the real reply: {verdict}  over {summary['judged']} (50% = cannot tell)",
            f"message length mean real/bot: {summary['len_mean']['real']}/{summary['len_mean']['bot']}  median: {summary['len_p50']['real']}/{summary['len_p50']['bot']}",
            f"burst size mean real/bot: {summary['burst_mean']['real']}/{summary['burst_mean']['bot']}  length JS divergence: {summary['length_js']}",
            f"exact match {summary['exact_match_rate']:.1%}  out-of-character reasoning {summary['out_of_character_rate']:.1%}  retried {summary['retry_rate']:.1%}  violations {summary['violations']}",
            f"closeness to my real reply: {summary['sim_real']} (chance {summary['sim_chance']})",
            f"latency p50/p95: {summary['latency_p50']}s/{summary['latency_p95']}s  prompt tokens {summary['prompt_tokens_mean']}  cached {summary['cached_share']:.0%}  recalled {summary['recalled_mean']}",
        ]
    )
