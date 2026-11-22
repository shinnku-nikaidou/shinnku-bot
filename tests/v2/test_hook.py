"""The hook that keeps Claude out of the chat records: what it lets through and what it stops."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "no_chat_content.py"
_SPEC = importlib.util.spec_from_file_location("no_chat_content", _PATH)
hook = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hook)


def refused(command: str) -> bool:
    return hook.refusal({"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(hook.REPO)}) is not None


class HookTests(unittest.TestCase):
    def test_moving_fine_tune_files_with_colab(self) -> None:
        for command in (
            "colab upload -s ft data/sft/train.jsonl /content/sft/train.jsonl",
            "colab download -s ft /content/out/eval_out.jsonl data/sft/eval_out.jsonl 2>&1 | tail -3",
            "for f in train valid eval eval_recall; do colab upload -s ft data/sft/$f.jsonl /content/sft/$f.jsonl; done",
            "colab ssh -s ft -- mkdir -p /content/sft && colab upload -s ft ./data/sft/valid.jsonl /content/sft/valid.jsonl",
        ):
            self.assertFalse(refused(command), command)

    def test_still_stops_reading(self) -> None:
        for command in (
            "cat data/sft/train.jsonl",
            "colab download -s ft /content/out/x data/sft/x | xargs cat",
            "head -c 300 data/sft/train.jsonl",
            "ls data/sft; cat data/sft/train.jsonl",
            "ls data/sft | xargs cat",
            "ls | head data/sft/train.jsonl",
            "ls $(cat data/sft/train.jsonl)",
            "colab upload -s ft data/corpus.sqlite /content/c.sqlite",
            "cp data/sft/train.jsonl /tmp/x",
            "sqlite3 data/live.sqlite3 'select text from msg'",
            "uv run shinnku eval quiz",
        ):
            self.assertTrue(refused(command), command)

    def test_statistics_commands_pass(self) -> None:
        self.assertFalse(refused("uv run shinnku corpus sft"))


if __name__ == "__main__":
    unittest.main()
