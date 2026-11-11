"""Let the owner take the blind test: which of these two did you actually write?"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path


def run_quiz(
    report_path: Path,
    n: int = 20,
    seed: int | None = None,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Show context and two candidate replies per item. Returns (correct, answered)."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    items = [item for item in report["items"] if item["bot"] and item["bot"] != item["real"]]
    rng = random.Random(seed)
    rng.shuffle(items)
    correct = answered = 0
    for number, item in enumerate(items[:n], start=1):
        bot_first = rng.random() < 0.5
        first, second = (item["bot"], item["real"]) if bot_first else (item["real"], item["bot"])
        show(f"\n—— 第 {number} 题 ——")
        for line in item["context"]:
            show(f"  {line}")
        show("  A:")
        for text in first:
            show(f"     {text}")
        show("  B:")
        for text in second:
            show(f"     {text}")
        answer = ask("哪个是你当时真发的？[A/B，回车跳过，q 退出] ").strip().upper()
        if answer == "Q":
            break
        if answer not in ("A", "B"):
            continue
        answered += 1
        right = (answer == "B") == bot_first
        correct += right
        show("对" if right else "错，另一个才是你发的")
    if answered:
        show(f"\n你答对 {correct}/{answered}（{correct / answered:.0%}）。越接近 50%，说明 bot 越像你。")
    return correct, answered
