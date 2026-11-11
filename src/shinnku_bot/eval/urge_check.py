"""Left to itself, does the model speak where I spoke and keep quiet where I did?

Held-out moments are replayed through the harness as uncalled-for moments, with
no gate in front: this measures the model's own choice between speaking and not.
"""

from __future__ import annotations

import random
from typing import Any

from shinnku_bot.config import Settings
from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.eval.run import gather_bounded, replay
from shinnku_bot.harness.engine import URGE
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.recall import Recall
from shinnku_bot.llm import ChatModel


async def check_urge(
    settings: Settings,
    db: CorpusDB,
    moments: list[dict[str, Any]],
    model: ChatModel,
    persona: Persona,
    recall: Recall | None,
    limits: StyleLimits,
    n_each: int = 100,
    seed: int = 0,
    concurrency: int = 4,
) -> dict[str, Any]:
    """Speaking rates on up to `n_each` moments where I joined in and where I let it pass."""
    rng = random.Random(seed)
    joined = [m for m in moments if m["spoke"]]
    passed = [m for m in moments if not m["spoke"]]
    rng.shuffle(joined)
    rng.shuffle(passed)
    picked = [*joined[:n_each], *passed[:n_each]]

    async def one(moment: dict[str, Any]) -> str:
        context = db.before(
            moment["chat_id"], moment["msg_id"], settings.context_messages, settings.context_max_age_s
        )
        try:
            outcome, _ = await replay(
                settings, model, persona, recall, limits, context, URGE, db.chat_title(moment["chat_id"])
            )
            return outcome.kind
        except Exception:  # noqa: BLE001 - one bad moment must not end the run
            return "error"

    kinds = await gather_bounded(
        picked, one, failed=lambda kind: kind == "error", skipped=lambda _: "error", concurrency=concurrency
    )

    def tally(want: bool) -> dict[str, Any]:
        mine = [kind for moment, kind in zip(picked, kinds) if moment["spoke"] == want and kind != "error"]
        spoke = sum(kind == "said" for kind in mine)
        return {"n": len(mine), "spoke": spoke, "rate": round(spoke / len(mine), 3) if mine else 0.0}

    return {
        "where_i_joined": tally(True),
        "where_i_passed": tally(False),
        "errors": sum(kind == "error" for kind in kinds),
    }
