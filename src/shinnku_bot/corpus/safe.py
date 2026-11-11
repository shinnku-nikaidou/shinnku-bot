"""Run corpus commands so that a crash cannot print chat text to the console."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path


def run_safely(action: Callable[[], int], reports_dir: Path) -> int:
    """Call `action`. On failure print the exception type and code locations only.

    The exception message may quote the data being processed, so it goes to
    `reports_dir/last_error.txt` instead of the console.
    """
    try:
        return action()
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - this is the top-level guard
        print(f"failed: {type(exc).__name__}")
        for frame in traceback.extract_tb(exc.__traceback__):
            print(f"  at {Path(frame.filename).name}:{frame.lineno} in {frame.name}")
        try:
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"  details written to {reports_dir / 'last_error.txt'}")
        except OSError:
            print("  (could not write the details file)")
        return 1
