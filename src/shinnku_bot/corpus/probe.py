"""Describe the structure of an export: keys, types, enum values, counts.

Nothing printed here is message text or a sender name. Values are shown only
for fields that are enums by design, and only when they look like identifiers.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shinnku_bot.corpus.export import ExportReader

_ENUM_LIKE = re.compile(r"^[a-z0-9_]{1,40}$")
_FROM_PREFIX = re.compile(r"^([a-z_]+)\d+$")


def enum_value(value: Any) -> str:
    """The value itself if it is identifier-shaped, otherwise a placeholder."""
    if value is None:
        return "<null>"
    if isinstance(value, str) and _ENUM_LIKE.match(value):
        return value
    return "<other>"


def type_name(value: Any) -> str:
    """A structural description of a JSON value: type names and key names only."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        inner = sorted({type_name(item) for item in value})
        return f"list[{'|'.join(inner)}]" if inner else "list[]"
    if isinstance(value, dict):
        return "dict{" + ",".join(sorted(value)) + "}"
    return type(value).__name__


@dataclass
class ProbeReport:
    path: str
    size_bytes: int
    truncated: bool = False
    header: dict[str, str] = field(default_factory=dict)
    chat_type: str = "<null>"
    chat_id: int | None = None
    messages: int = 0
    by_type: Counter = field(default_factory=Counter)
    key_counts: Counter = field(default_factory=Counter)
    key_types: dict[str, Counter] = field(default_factory=dict)
    text_shape: Counter = field(default_factory=Counter)
    entity_types: Counter = field(default_factory=Counter)
    media_types: Counter = field(default_factory=Counter)
    actions: Counter = field(default_factory=Counter)
    sender_kinds: Counter = field(default_factory=Counter)
    senders: int = 0
    owner_messages: int = 0
    date_min: str = ""
    date_max: str = ""
    replies: int = 0
    forwards: int = 0
    edited: int = 0
    via_bot: int = 0


def probe(path: Path, owner_id: int | None = None) -> ProbeReport:
    path = Path(path)
    report = ProbeReport(path=str(path), size_bytes=path.stat().st_size)
    owner = f"user{owner_id}" if owner_id is not None else None
    senders: set[str] = set()

    with ExportReader(path) as reader:
        report.header = {key: type_name(value) for key, value in reader.header.items()}
        report.chat_type = enum_value(reader.header.get("type"))
        chat_id = reader.header.get("id")
        report.chat_id = chat_id if isinstance(chat_id, int) else None

        for message in reader:
            report.messages += 1
            report.by_type[enum_value(message.get("type"))] += 1
            for key, value in message.items():
                report.key_counts[key] += 1
                report.key_types.setdefault(key, Counter())[type_name(value)] += 1

            text = message.get("text")
            if isinstance(text, list):
                report.text_shape["list"] += 1
            elif text:
                report.text_shape["str"] += 1
            else:
                report.text_shape["empty"] += 1
            for entity in message.get("text_entities") or ():
                if isinstance(entity, dict):
                    report.entity_types[enum_value(entity.get("type"))] += 1

            if "media_type" in message:
                report.media_types[enum_value(message["media_type"])] += 1
            if "action" in message:
                report.actions[enum_value(message["action"])] += 1

            sender = message.get("from_id") or message.get("actor_id")
            if isinstance(sender, str):
                senders.add(sender)
                prefix = _FROM_PREFIX.match(sender)
                report.sender_kinds[prefix.group(1) if prefix else "<other>"] += 1
                if owner is not None and sender == owner and message.get("type") == "message":
                    report.owner_messages += 1
            else:
                report.sender_kinds["<null>"] += 1

            date = message.get("date")
            if isinstance(date, str) and len(date) >= 10:
                day = date[:10]
                if not report.date_min or day < report.date_min:
                    report.date_min = day
                if day > report.date_max:
                    report.date_max = day

            report.replies += "reply_to_message_id" in message
            report.forwards += "forwarded_from" in message
            report.edited += "edited" in message
            report.via_bot += "via_bot" in message

        report.truncated = reader.truncated
    report.senders = len(senders)
    return report


def _table(counter: Counter, total: int) -> list[str]:
    return [
        f"    {name:<28} {count:>10}  {count / total:6.1%}" if total else f"    {name:<28} {count:>10}"
        for name, count in counter.most_common()
    ]


def format_report(report: ProbeReport) -> str:
    total = report.messages
    lines = [
        f"file: {Path(report.path).parent.name}/{Path(report.path).name}",
        f"size: {report.size_bytes / 1e6:.1f} MB    complete: {'no (still being written?)' if report.truncated else 'yes'}",
        f"chat: type={report.chat_type} id={report.chat_id}",
        "header keys: " + ", ".join(f"{key}:{kind}" for key, kind in report.header.items()),
        f"messages: {total}    dates: {report.date_min} .. {report.date_max}",
        "by type:",
        *_table(report.by_type, total),
        f"senders: {report.senders} distinct",
        "sender id kinds:",
        *_table(report.sender_kinds, total),
        f"owner messages: {report.owner_messages}  ({report.owner_messages / total:.1%} of all)" if total else "owner messages: 0",
        f"replies: {report.replies}    forwards: {report.forwards}    edited: {report.edited}    via_bot: {report.via_bot}",
        "text shape:",
        *_table(report.text_shape, total),
        "media_type:",
        *_table(report.media_types, total),
        "service action:",
        *_table(report.actions, total),
        "text entity types:",
        *_table(report.entity_types, sum(report.entity_types.values())),
        "message keys (count, value types):",
    ]
    for key, count in report.key_counts.most_common():
        kinds = ", ".join(f"{kind} x{n}" for kind, n in report.key_types[key].most_common(4))
        lines.append(f"    {key:<28} {count:>10}  {kinds}")
    return "\n".join(lines)
