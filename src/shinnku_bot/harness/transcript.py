"""Turn stored messages into chat turns.

This is the one place that decides how a conversation looks to the model. The
corpus pipeline renders historical conversations through the same functions, so
remembered scenes and the live chat read identically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

from shinnku_bot.store import Msg

MAX_BODY = 500
MAX_NAME = 16
ME = "我"

_PLACEHOLDER = {
    "photo": "[图片]",
    "voice": "[语音]",
    "video": "[视频]",
    "animation": "[动图]",
    "document": "[文件]",
    "other": "[消息]",
}

Lookup = Callable[[int], Msg | None]


def clean_name(name: str) -> str:
    """A display name that is safe to put in front of `: `."""
    name = " ".join(name.replace(":", " ").replace("：", " ").split())
    return name[:MAX_NAME] or "某人"


def render_body(msg: Msg) -> str:
    text = " ".join(msg.text.split())
    if len(text) > MAX_BODY:
        text = text[:MAX_BODY] + "…"
    if msg.kind == "text":
        return text
    if msg.kind == "sticker":
        return f"[贴纸 {text}]" if text else "[贴纸]"
    label = _PLACEHOLDER.get(msg.kind, "[消息]")
    return f"{label} {text}" if text else label


def _speakers(msgs: Sequence[Msg], lookup: Lookup | None) -> Iterator[tuple[Msg, str, str]]:
    """Yield (message, speaker label, body) for every message that has a body.

    The label is the sender's name, plus who they are replying to when known.
    """
    by_id = {m.message_id: m for m in msgs}
    for msg in msgs:
        body = render_body(msg)
        if not body:
            continue
        if msg.is_me:
            yield msg, ME, body
            continue
        note = ""
        if msg.reply_to is not None:
            target = by_id.get(msg.reply_to) or (lookup(msg.reply_to) if lookup else None)
            if target is not None:
                note = "（回复我）" if target.is_me else f"（回复 {clean_name(target.user_name)}）"
        yield msg, clean_name(msg.user_name) + note, body


def render_turns(msgs: Sequence[Msg], lookup: Lookup | None = None) -> list[dict[str, str]]:
    """Other people's lines become user turns (`name: text`); mine become assistant turns.

    Consecutive lines with the same role are merged into one turn, one line each.
    `lookup` resolves reply targets that fall outside `msgs`.
    """
    turns: list[dict[str, str]] = []
    for msg, speaker, body in _speakers(msgs, lookup):
        role, line = ("assistant", body) if msg.is_me else ("user", f"{speaker}: {body}")
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + line
        else:
            turns.append({"role": role, "content": line})
    return turns


def render_lines(msgs: Sequence[Msg], lookup: Lookup | None = None) -> list[str]:
    """Every message as `speaker: text`, my own lines included (as `我: text`)."""
    return [f"{speaker}: {body}" for _, speaker, body in _speakers(msgs, lookup)]


def render_scene(context: Sequence[Msg], reply: Sequence[Msg], tail: int = 6) -> str:
    """A remembered moment: the last few lines before I spoke, then what I said."""
    return "\n".join(render_lines(context)[-tail:] + render_lines(reply))


def tail_text(msgs: Sequence[Msg], lines: int = 4, max_chars: int = 200) -> str:
    """The end of a conversation as plain text, used to find similar moments.

    Speaker names are left out so that who said it does not affect the match.
    """
    bodies = [body for body in (render_body(m) for m in msgs) if body][-lines:]
    return "\n".join(bodies)[-max_chars:]
