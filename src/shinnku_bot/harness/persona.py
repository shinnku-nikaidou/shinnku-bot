"""Who I am and what I remember, read from the persona directory.

    SELF.md                      who I am
    FACTS.md                     what I know about my own life
    anchor.txt                   the fixed sample of how I talk
    groups/<chat>.md             what this group is to me
    people/<chat>/<user>.md      what this person is to me, in this group
    diary/<chat>/YYYY-MM-DD.md   what happened here on that day

Every file can be edited or regenerated while the bot runs. `<chat>` is the id
Telegram Desktop writes in exports, so cards built from an export and diaries
built from the live chat land in the same place.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

_DEFAULT = Path(__file__).with_name("default_self.md")


def export_chat_id(chat_id: int) -> int:
    """Bot API chat id -> the id Telegram Desktop writes in exports."""
    if chat_id <= -1_000_000_000_000:
        return -chat_id - 1_000_000_000_000
    return abs(chat_id)


class _CachedFile:
    """A text file that is re-read only when it changes on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cached: tuple[float, str] | None = None

    def read(self) -> str:
        """The stripped contents, or '' if the file is missing."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return ""
        if self._cached is None or self._cached[0] != mtime:
            self._cached = (mtime, self.path.read_text(encoding="utf-8").strip())
        return self._cached[1]


class Persona:
    """Without SELF.md the packaged placeholder is used; every other file is optional."""

    def __init__(self, persona_dir: Path) -> None:
        self.dir = Path(persona_dir)
        self._self = _CachedFile(self.dir / "SELF.md")
        self._facts = _CachedFile(self.dir / "FACTS.md")
        self._anchor = _CachedFile(self.dir / "anchor.txt")
        self._files: dict[Path, _CachedFile] = {}

    def _read(self, path: Path) -> str:
        cached = self._files.get(path)
        if cached is None:
            cached = self._files[path] = _CachedFile(path)
        return cached.read()

    @property
    def is_default(self) -> bool:
        return not self._self.path.is_file()

    def self_text(self) -> str:
        return self._self.read() or _DEFAULT.read_text(encoding="utf-8").strip()

    def facts_text(self) -> str:
        return self._facts.read()

    def anchor_text(self) -> str:
        return self._anchor.read()

    def group_card(self, chat_id: int) -> str:
        return self._read(self.dir / "groups" / f"{export_chat_id(chat_id)}.md")

    def person_card(self, chat_id: int, user_id: int) -> str:
        return self._read(self.dir / "people" / str(export_chat_id(chat_id)) / f"{user_id}.md")

    def people_cards(
        self, chat_id: int, user_ids: Sequence[int], limit: int
    ) -> list[tuple[int, str]]:
        """(user id, card) for the first `limit` of these people I have a card for."""
        cards: list[tuple[int, str]] = []
        for user_id in dict.fromkeys(user_ids):
            if len(cards) >= limit:
                break
            card = self.person_card(chat_id, user_id)
            if card:
                cards.append((user_id, card))
        return cards

    def diary(self, chat_id: int, today: str, days: int) -> list[tuple[str, str]]:
        """(date, entry) for the `days` most recent days before `today`, oldest first."""
        if days <= 0:
            return []
        folder = self.dir / "diary" / str(export_chat_id(chat_id))
        try:
            names = sorted(p.stem for p in folder.glob("????-??-??.md") if p.stem < today)
        except OSError:
            return []
        entries = [(name, self._read(folder / f"{name}.md")) for name in names[-days:]]
        return [(name, text) for name, text in entries if text]
