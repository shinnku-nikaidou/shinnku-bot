"""Speaking without being called: is this a moment where I would chime in?

Three layers, cheapest first. Hard rules (budget, cooldown, backing off after
I chose silence). A prior learned from the moments where I really did or did
not join in. And then the model itself, which can still say nothing; that last
layer lives in the engine, this module only decides whether to ask it.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from shinnku_bot.config import Settings
from shinnku_bot.eval.classify import LogReg
from shinnku_bot.store import Msg

FEATURES = (
    "since_me",
    "me_in_last10",
    "speakers_last10",
    "msgs_last60s",
    "hour_share",
    "sender_rate",
    "is_question",
    "length",
    "is_text",
)
# How much of my real rate of joining in each /talk level aims for.
LEVEL_FACTOR = {0: 0.0, 1: 0.25, 2: 0.6, 3: 1.0}
LONG_AGO_S = 6 * 3600
COOLDOWN_S = 30
BACKOFF_BASE_S = 60
BACKOFF_MAX_S = 30 * 60
_QUESTION_ENDS = ("吗", "呢", "么", "嘛")


def moment_features(
    msgs: Sequence[Msg],
    hour_share: Sequence[float],
    sender_rate: Mapping[int, float],
    default_rate: float,
    tz: ZoneInfo,
) -> list[float]:
    """Describe the moment right after `msgs[-1]`, in the order of FEATURES."""
    last = msgs[-1]
    mine = [m.ts for m in msgs if m.is_me]
    since_me = min(last.ts - mine[-1], LONG_AGO_S) if mine else LONG_AGO_S
    last10 = msgs[-10:]
    text = last.text.strip()
    hour = datetime.fromtimestamp(last.ts, tz).hour
    return [
        math.log1p(max(since_me, 0)),
        sum(m.is_me for m in last10) / 10.0,
        float(len({m.user_id for m in last10 if not m.is_me})),
        float(sum(m.ts >= last.ts - 60 for m in msgs)),
        float(hour_share[hour]) if len(hour_share) == 24 else 0.0,
        float(sender_rate.get(last.user_id, default_rate)) if last.user_id is not None else default_rate,
        float("?" in text or "？" in text or text.endswith(_QUESTION_ENDS)),
        math.log1p(len(text)),
        float(last.kind == "text"),
    ]


@dataclass
class UrgeModel:
    """What `shinnku corpus moments` learned, as stored in persona/urge.json."""

    base_rate: float
    hour_share: list[float]
    sender_rate: dict[int, float] = field(default_factory=dict)
    default_sender_rate: float = 0.0
    timezone: str = "Asia/Shanghai"
    # Share of the model's turns, on moments where I did speak, in which it speaks too.
    speak_rate: float = 1.0
    auc: float | None = None
    # None when the learned prior was too weak to keep; then only the rates are used.
    weights: list[float] | None = None
    bias: float = 0.0
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._tz = ZoneInfo(self.timezone)
        self._model: LogReg | None = None
        if self.weights is not None:
            model = LogReg(scaled=len(FEATURES))
            model.weights = np.asarray(self.weights, dtype=np.float64)
            model.bias = float(self.bias)
            model.mean = np.asarray(self.mean, dtype=np.float64)
            model.std = np.asarray(self.std, dtype=np.float64)
            self._model = model

    def features(self, msgs: Sequence[Msg]) -> list[float]:
        return moment_features(msgs, self.hour_share, self.sender_rate, self.default_sender_rate, self._tz)

    def probability(self, features: Sequence[float]) -> float:
        """Chance that I would join in at a moment like this."""
        if self._model is not None:
            return float(self._model.probability(np.asarray([features], dtype=np.float64))[0])
        # No learned prior: my overall rate, scaled by how active I am at this hour.
        typical = sum(self.hour_share) / 24 if self.hour_share else 0.0
        factor = features[FEATURES.index("hour_share")] / typical if typical > 0 else 1.0
        return min(1.0, self.base_rate * min(max(factor, 0.2), 3.0))

    def call_probability(self, features: Sequence[float], level: int) -> float:
        """Chance of asking the model, so that what it ends up saying matches the level."""
        factor = LEVEL_FACTOR.get(level, 0.0)
        return min(1.0, self.probability(features) * factor / max(self.speak_rate, 0.2))

    def to_json(self) -> dict[str, Any]:
        return {
            "features": list(FEATURES),
            "base_rate": self.base_rate,
            "hour_share": self.hour_share,
            "sender_rate": {str(k): v for k, v in self.sender_rate.items()},
            "default_sender_rate": self.default_sender_rate,
            "timezone": self.timezone,
            "speak_rate": self.speak_rate,
            "auc": self.auc,
            "weights": self.weights,
            "bias": self.bias,
            "mean": self.mean,
            "std": self.std,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()), encoding="utf-8")


def load_urge(path: Path) -> UrgeModel | None:
    """The stored model, or None if it has not been built or does not fit this code."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("features") != list(FEATURES):
            return None
        return UrgeModel(
            base_rate=float(raw["base_rate"]),
            hour_share=[float(x) for x in raw["hour_share"]],
            sender_rate={int(k): float(v) for k, v in raw.get("sender_rate", {}).items()},
            default_sender_rate=float(raw.get("default_sender_rate", 0.0)),
            timezone=raw.get("timezone", "Asia/Shanghai"),
            speak_rate=float(raw.get("speak_rate", 1.0)),
            auc=raw.get("auc"),
            weights=raw.get("weights"),
            bias=float(raw.get("bias", 0.0)),
            mean=raw.get("mean", []),
            std=raw.get("std", []),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


@dataclass
class _ChatUrge:
    calls: deque[float] = field(default_factory=deque)
    cooldown_until: float = 0.0
    backoff_until: float = 0.0
    silences: int = 0
    owner_last: float = float("-inf")


class Gate:
    """Decides, per chat, whether an uncalled-for moment is worth a model call."""

    def __init__(
        self,
        settings: Settings,
        model: UrgeModel | None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        import random

        self._settings = settings
        self.model = model
        self._rng = rng or random.random
        self._chats: dict[int, _ChatUrge] = {}

    def _chat(self, chat_id: int) -> _ChatUrge:
        return self._chats.setdefault(chat_id, _ChatUrge())

    def owner_spoke(self, chat_id: int, now: float) -> None:
        self._chat(chat_id).owner_last = now

    def spoke(self, chat_id: int, now: float) -> None:
        chat = self._chat(chat_id)
        chat.silences = 0
        chat.backoff_until = 0.0
        chat.cooldown_until = now + COOLDOWN_S

    def stayed_silent(self, chat_id: int, now: float) -> None:
        """The model was asked and chose silence: ask less often for a while."""
        chat = self._chat(chat_id)
        chat.silences += 1
        chat.backoff_until = now + min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (chat.silences - 1))

    def decide(self, chat_id: int, level: int, msgs: Sequence[Msg], now: float) -> tuple[bool, str]:
        """(ask the model?, why). Asking is recorded against the hourly and daily budget."""
        if level <= 0:
            return False, "level"
        if self.model is None:
            return False, "no_model"
        if not msgs or msgs[-1].is_me:
            return False, "nothing_new"
        chat = self._chat(chat_id)
        if now - chat.owner_last < self._settings.yield_to_owner_min * 60:
            return False, "owner_here"
        if now < chat.cooldown_until:
            return False, "cooldown"
        if now < chat.backoff_until:
            return False, "backoff"
        while chat.calls and now - chat.calls[0] >= 86400:
            chat.calls.popleft()
        if len(chat.calls) >= self._settings.proactive_calls_per_day:
            return False, "day_cap"
        if sum(now - t < 3600 for t in chat.calls) >= self._settings.proactive_calls_per_hour:
            return False, "hour_cap"
        chance = self.model.call_probability(self.model.features(msgs), level)
        if self._rng() >= chance:
            return False, "dice"
        chat.calls.append(now)
        return True, "urge"
