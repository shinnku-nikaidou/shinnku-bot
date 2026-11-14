"""Looking at pictures: a photo, sticker or GIF becomes one line saying what it shows.

A picture is looked at the first time it is in front of me when I am about to
speak. What it shows is kept, so the same sticker is never looked at twice.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace

from shinnku_bot.llm import ChatModel
from shinnku_bot.store import Msg, Store

log = logging.getLogger(__name__)

PICTURES = frozenset({"photo", "sticker", "animation"})
MAX_SEEN = 120
# How long a reply waits for its pictures; the ones not seen by then stay placeholders.
WAIT_S = 15.0

# file_id -> the image bytes.
Fetch = Callable[[str], Awaitable[bytes]]

PROMPT = (
    "这是群聊里有人发的一张图（照片、截图或表情包）。用一句中文说清楚图里是什么；"
    "图上有字就原样写出来；表情包就说出它的表情和意思。不超过 60 字。"
)

TOOL = {
    "type": "function",
    "function": {
        "name": "describe",
        "description": "说出这张图是什么",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


def _mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


async def describe(model: ChatModel, image: bytes) -> str:
    url = f"data:{_mime(image)};base64,{base64.b64encode(image).decode()}"
    content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]
    reply = await model.complete([{"role": "user", "content": content}], [TOOL], thinking=False)
    text = next(
        (str(call.args.get("text", "")) for call in reply.tool_calls if call.name == "describe"),
        reply.content,
    )
    return " ".join(text.split())[:MAX_SEEN]


async def _see(store: Store, model: ChatModel, fetch: Fetch, file_id: str, file_uid: str) -> bool:
    try:
        text = await describe(model, await fetch(file_id))
    except Exception as exc:  # noqa: BLE001 - a picture I could not see keeps its placeholder
        log.warning("look failed error=%s", type(exc).__name__)
        return False
    if text:
        store.set_seen(file_uid, text)
    return bool(text)


def _filled(store: Store, msg: Msg) -> Msg:
    picture = store.picture(msg.chat_id, msg.message_id) if msg.kind in PICTURES else None
    return replace(msg, seen=picture[2]) if picture and picture[2] else msg


async def look(msgs: Sequence[Msg], store: Store, model: ChatModel, fetch: Fetch) -> list[Msg]:
    """`msgs` with what each picture shows, looking now at the ones I have not seen."""
    unseen: dict[str, str] = {}  # file_uid -> file_id
    for msg in msgs:
        picture = store.picture(msg.chat_id, msg.message_id) if msg.kind in PICTURES else None
        if picture and not picture[2]:
            unseen[picture[1]] = picture[0]
    if unseen:
        try:
            seen = await asyncio.wait_for(
                asyncio.gather(*(_see(store, model, fetch, fid, uid) for uid, fid in unseen.items())),
                WAIT_S,
            )
            log.info("looked pictures=%d seen=%d", len(unseen), sum(seen))
        except TimeoutError:
            log.warning("look timed out pictures=%d", len(unseen))
    return [_filled(store, msg) for msg in msgs]
