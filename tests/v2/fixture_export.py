"""Write made-up exports in the shape Telegram Desktop produces."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OWNER_ID = 42
NOT_INCLUDED = "(File not included. Change data exporting settings to download.)"
PEOPLE = [("user42", "我自己"), ("user7", "小林"), ("user8", "阿树"), ("user9", "老周")]
OWNER_LINES = ["不去", "太累了", "确实", "草", "行吧", "不知道 懒得查", "几点", "来", "你才是", "笑死"]
OTHER_LINES = [
    "今晚有人打游戏吗",
    "这周末去不去爬山",
    "我十点后",
    "你就是懒",
    "吃了没",
    "这个怎么弄",
    "哈哈哈哈",
    "明天上班好烦",
]


def message(
    msg_id: int,
    ts: int,
    sender: tuple[str, str] | None,
    text: Any = "",
    **extra: Any,
) -> dict[str, Any]:
    """One export message. `sender` is (from_id, name); None means a deleted account."""
    moment = datetime.fromtimestamp(ts, timezone.utc)
    body: dict[str, Any] = {
        "id": msg_id,
        "type": "message",
        "date": moment.strftime("%Y-%m-%dT%H:%M:%S"),
        "date_unixtime": str(ts),
        "from": sender[1] if sender else None,
        "from_id": sender[0] if sender else "user0",
    }
    body.update(extra)
    body["text"] = text
    if isinstance(text, str):
        body["text_entities"] = [{"type": "plain", "text": text}] if text else []
    else:
        body["text_entities"] = [
            {"type": "plain", "text": part} if isinstance(part, str) else part for part in text
        ]
    return body


def service(msg_id: int, ts: int, actor: tuple[str, str], action: str, **extra: Any) -> dict[str, Any]:
    moment = datetime.fromtimestamp(ts, timezone.utc)
    return {
        "id": msg_id,
        "type": "service",
        "date": moment.strftime("%Y-%m-%dT%H:%M:%S"),
        "date_unixtime": str(ts),
        "actor": actor[1],
        "actor_id": actor[0],
        "action": action,
        **extra,
        "text": "",
        "text_entities": [],
    }


def write_export(
    path: Path,
    messages: list[dict[str, Any]],
    chat_id: int = 1001,
    truncate: float | None = None,
) -> Path:
    """Write the export; `truncate` keeps only that fraction of the file."""
    text = json.dumps(
        {"name": "测试群", "type": "private_supergroup", "id": chat_id, "messages": messages},
        ensure_ascii=False,
        indent=1,
    )
    if truncate is not None:
        text = text[: int(len(text) * truncate)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def random_messages(n: int, seed: int = 0, start_ts: int = 1_700_000_000) -> list[dict[str, Any]]:
    """A deterministic chat with every message shape the pipeline must handle."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    ts = start_ts
    msg_id = 0
    while len(out) < n:
        msg_id += 1
        ts += rng.randint(5, 60)
        if msg_id % 40 == 0:
            ts += 2 * 3600  # a long silence starts a new session
        sender = rng.choice(PEOPLE)
        is_owner = sender[0] == f"user{OWNER_ID}"
        roll = rng.random()
        if roll < 0.02:
            out.append(service(msg_id, ts, sender, "pin_message", message_id=max(1, msg_id - 1)))
        elif roll < 0.07:
            out.append(
                message(msg_id, ts, sender, "", file=NOT_INCLUDED, media_type="sticker", sticker_emoji="😂")
            )
        elif roll < 0.10:
            out.append(message(msg_id, ts, sender, "看这个", photo=NOT_INCLUDED, width=100, height=80))
        elif roll < 0.15:
            link = {"type": "link", "text": "https://example.com/a"}
            out.append(message(msg_id, ts, sender, ["看 ", link]))
        elif roll < 0.17:
            out.append(message(msg_id, ts, sender, "转发的内容", forwarded_from="某频道"))
        elif roll < 0.18 and is_owner:
            command = {"type": "bot_command", "text": "/rua"}
            out.append(message(msg_id, ts, sender, [command]))
        else:
            line = rng.choice(OWNER_LINES if is_owner else OTHER_LINES)
            extra = {"reply_to_message_id": msg_id - 1} if msg_id > 1 and rng.random() < 0.1 else {}
            out.append(message(msg_id, ts, sender, line, **extra))
    return out
