#!/usr/bin/env python3
"""PreToolUse hook: Claude may not query the chat records.

Chat text lives in the exports, in data/ (message store, corpus, traces,
reports, distilled persona) and in the old bot's logs/. The only way in is the
project's own `shinnku` commands, which print counts and statistics. Anything
else that reaches for those places is refused: reading the files, naming them
in a shell command, or running ad-hoc code that opens them through the
project's settings or classes.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRIVATE_DIRS = [
    REPO / "data",
    REPO / "logs",
    Path.home() / "tg-export",
    Path.home() / ".local/share/TelegramDesktop",
]
# Project code may open the stores: it is written to print statistics only.
TRUSTED_CODE = [REPO / "src", REPO / "tests"]

PRIVATE = re.compile(
    r"""tg-export|TelegramDesktop|tdata\b
      | (?:^|[\s"'=(/])(?:data|logs)/
      | \.sqlite3?\b
      | \b(?:db_path|corpus_db|trace_dir|reports_dir|persona_dir|distill_dir|export_dir|index_dir|data_dir)\b
      | \b(?:Store|CorpusDB|ExportReader|load_items|load_facts|load_probes|load_moments)\s*\(
    """,
    re.VERBOSE,
)
# Commands of the project that show chat text on purpose; they are for the owner's own terminal.
SHOWS_TEXT = re.compile(r"\bshinnku\s+eval\s+quiz\b|\bshinnku\s+chat\b(?!.*--stats-only)", re.DOTALL)
SCRIPT = re.compile(r"\.(?:py|sh|sql)$")


def inside(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def scripts_run_by(command: str, cwd: Path) -> list[Path]:
    """Script files the command names, so a query cannot hide in a file."""
    try:
        words = shlex.split(command, comments=True)
    except ValueError:
        words = command.split()
    found = []
    for word in words:
        if SCRIPT.search(word):
            path = Path(word) if Path(word).is_absolute() else cwd / word
            if path.is_file() and not inside(path, TRUSTED_CODE):
                found.append(path)
    return found


def refusal(payload: dict) -> str | None:
    tool = payload.get("tool_name", "")
    args = payload.get("tool_input") or {}
    cwd = Path(payload.get("cwd") or REPO)
    if tool == "Bash":
        command = str(args.get("command", ""))
        if SHOWS_TEXT.search(command):
            return "this command prints chat text; it is for the owner's own terminal"
        if PRIVATE.search(command):
            return "the command reaches for the chat records; use a `shinnku` command that prints statistics"
        for script in scripts_run_by(command, cwd):
            try:
                text = script.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if PRIVATE.search(text) or SHOWS_TEXT.search(text):
                return f"{script.name} reaches for the chat records"
        return None
    target = args.get("file_path") or args.get("path")
    if target:
        path = Path(target) if Path(target).is_absolute() else cwd / target
        if inside(path, PRIVATE_DIRS):
            return "this path holds chat records"
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    reason = refusal(payload)
    if reason:
        json.dump(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Blocked by .claude/hooks/no_chat_content.py: {reason}.",
                }
            },
            sys.stdout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
