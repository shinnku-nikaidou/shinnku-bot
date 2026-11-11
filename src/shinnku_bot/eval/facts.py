"""Does the bot know the facts of my life? Ask it and look for the expected words."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shinnku_bot.config import Settings
from shinnku_bot.eval.run import gather_bounded, replay
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.recall import Recall
from shinnku_bot.llm import ChatModel
from shinnku_bot.store import Msg

ASKER_ID = 777_000_001


def load_probes(path: Path) -> list[dict[str, Any]]:
    """Questions with the words a right answer must contain, one JSON object per line."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    probes = []
    for line in lines:
        try:
            probe = json.loads(line)
        except ValueError:
            continue
        if isinstance(probe, dict) and probe.get("question") and probe.get("keywords"):
            probes.append(probe)
    return probes


def is_correct(answer: str, keywords: list[str]) -> bool:
    folded = answer.casefold()
    return any(str(keyword).casefold() in folded for keyword in keywords if str(keyword).strip())


async def run_probes(
    settings: Settings,
    probes: list[dict[str, Any]],
    model: ChatModel,
    persona: Persona,
    recall: Recall | None,
    limits: StyleLimits,
    now: float,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Put each question to the harness as a private message from a friend."""

    async def ask(probe: dict[str, Any]) -> dict[str, Any]:
        question = Msg(ASKER_ID, 1, int(now), ASKER_ID, "朋友", False, text=str(probe["question"]))
        result = {**probe, "answer": "", "ok": False, "error": ""}
        try:
            _, texts = await replay(
                settings, model, persona, recall, limits, [question], "private", is_private=True
            )
            result["answer"] = "\n".join(texts)
            result["ok"] = is_correct(result["answer"], probe["keywords"])
        except Exception as exc:  # noqa: BLE001 - one bad probe must not end the run
            result["error"] = type(exc).__name__
        return result

    return await gather_bounded(
        probes,
        ask,
        failed=lambda result: bool(result["error"]),
        skipped=lambda probe: {**probe, "answer": "", "ok": False, "error": "Skipped"},
        concurrency=concurrency,
    )


def summarize_probes(results: list[dict[str, Any]]) -> dict[str, Any]:
    errors = sum(bool(r["error"]) for r in results)
    correct = sum(bool(r["ok"]) for r in results)
    return {
        "asked": len(results),
        "errors": errors,
        "answered": sum(bool(r["answer"]) for r in results),
        "correct": correct,
        "accuracy": round(correct / (len(results) - errors), 3) if len(results) > errors else 0.0,
    }
