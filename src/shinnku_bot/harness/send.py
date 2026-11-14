"""Send a burst of short messages the way a person types them."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


class Outbox(Protocol):
    async def typing(self, chat_id: int) -> None: ...

    async def send(self, chat_id: int, text: str, reply_to: int | None) -> int:
        """Send one message and return its message id."""
        ...


@dataclass(frozen=True)
class Pace:
    """How fast I type. The defaults are brisk; the fingerprint supplies mine."""

    sec_per_char: float = 0.12
    floor_s: float = 0.6
    cap_s: float = 4.0


def typing_delay(text: str, jitter: float, pace: Pace = Pace()) -> float:
    """Seconds spent "typing" `text`; `jitter` is a random number in [0, 1)."""
    base = min(max(0.5 + pace.sec_per_char * len(text), pace.floor_s), pace.cap_s)
    return base * (0.8 + 0.4 * jitter)


async def send_burst(
    outbox: Outbox,
    chat_id: int,
    messages: Sequence[tuple[str, int | None]],
    *,
    sleep: Callable[[float], Awaitable[None]],
    rng: Callable[[], float],
    on_sent: Callable[[int, str, int | None], None],
    pace: Pace = Pace(),
) -> None:
    """Send each (text, message it quotes or None) after a typing pause."""
    for text, reply_to in messages:
        await outbox.typing(chat_id)
        await sleep(typing_delay(text, rng(), pace))
        message_id = await outbox.send(chat_id, text, reply_to)
        on_sent(message_id, text, reply_to)
