"""Stream a Telegram Desktop `result.json` without loading it into memory.

The reader tolerates a file that is still being written: it yields every
complete message and reports that the tail was cut off.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_MESSAGES_KEY = re.compile(r'"messages"\s*:\s*\[')
_SEPARATORS = " \t\r\n,"
_CHUNK = 1 << 20
_MAX_HEADER = 8 << 20
_MAX_OBJECT = 64 << 20


class ExportFormatError(Exception):
    """The file is not a single-chat Telegram Desktop JSON export."""


def find_exports(root: Path) -> list[Path]:
    """Every result.json under `root`, in a stable order."""
    root = Path(root).expanduser()
    return sorted(path for path in root.rglob("result.json") if path.is_file())


def _parse_header(text: str) -> dict[str, Any]:
    """The chat-level fields that come before the messages array."""
    body = text.strip().rstrip(",").strip()
    if not body.startswith("{"):
        return {}
    try:
        header = json.loads(body + "}")
    except ValueError:
        return {}
    return header if isinstance(header, dict) else {}


class ExportReader:
    """Iterate once over the messages of one export.

    `header` is available immediately; `truncated` and `count` are final once
    iteration has finished.
    """

    def __init__(self, path: Path, chunk_size: int = _CHUNK) -> None:
        self.path = Path(path)
        self.truncated = False
        self.count = 0
        self._chunk = chunk_size
        self._file = self.path.open("r", encoding="utf-8", errors="replace")
        self._buf = ""
        self._pos = 0
        self.header = self._read_header()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> ExportReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _read_header(self) -> dict[str, Any]:
        while True:
            match = _MESSAGES_KEY.search(self._buf)
            if match:
                self._pos = match.end()
                return _parse_header(self._buf[: match.start()])
            chunk = self._file.read(self._chunk)
            if not chunk or len(self._buf) > _MAX_HEADER:
                self.close()
                raise ExportFormatError("no top-level messages array")
            self._buf += chunk

    def _refill(self) -> bool:
        """Drop consumed text and read one more chunk. False at end of file."""
        chunk = self._file.read(self._chunk)
        if not chunk:
            return False
        self._buf = self._buf[self._pos :] + chunk
        self._pos = 0
        return True

    def __iter__(self) -> Iterator[dict[str, Any]]:
        decoder = json.JSONDecoder()
        eof = False
        while True:
            while True:
                while self._pos < len(self._buf) and self._buf[self._pos] in _SEPARATORS:
                    self._pos += 1
                if self._pos < len(self._buf) or eof:
                    break
                eof = not self._refill()
            if self._pos >= len(self._buf):
                self.truncated = True  # ran out of file inside the array
                return
            if self._buf[self._pos] == "]":
                return
            try:
                obj, end = decoder.raw_decode(self._buf, self._pos)
            except ValueError:
                if eof:
                    self.truncated = True  # the last message was cut off mid-write
                    return
                eof = not self._refill()
                if len(self._buf) > _MAX_OBJECT:
                    raise ExportFormatError("unparseable message object") from None
                continue
            self._pos = end
            if self._pos > self._chunk:
                self._buf = self._buf[self._pos :]
                self._pos = 0
            if isinstance(obj, dict):
                self.count += 1
                yield obj
