"""Fine-tune data: prompts as the live harness builds them, and replies written elsewhere."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.ingest import ingest
from shinnku_bot.corpus.pairs import assign_holdout, build_pairs
from shinnku_bot.corpus.sft import write_sft
from shinnku_bot.eval.run import EvalOptions, load_replies, run_eval, sample_pairs
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.persona import Persona

from tests.v2.fakes import FakeModel, make_settings, say
from tests.v2.fixture_export import OWNER_ID, PEOPLE, message, write_export

ME, LIN = PEOPLE[0], PEOPLE[1]
T0 = 1_700_000_000
HOUR = 3600


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SftTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.settings = make_settings(self.tmp)
        self.db = CorpusDB(":memory:")
        self.addCleanup(self.db.close)
        for chat_id in (1, 2):
            messages: list[dict[str, Any]] = []
            for i in range(40):
                ts = T0 + i * 2 * HOUR
                messages.append(message(4 * i + 1, ts, LIN, f"第{i}次问你在不在"))
                messages.append(message(4 * i + 2, ts + 2, LIN, "人呢"))
                messages.append(message(4 * i + 3, ts + 5, ME, f"真实回答{i}"))
                # A slow follow-up of mine: its context ends with my own line, so nothing to answer.
                messages.append(message(4 * i + 4, ts + 600, ME, "补一句"))
            ingest(self.db, write_export(self.tmp / str(chat_id) / "result.json", messages, chat_id=chat_id), OWNER_ID)
        build_pairs(self.db)
        assign_holdout(self.db, per_chat=20, min_train=20)
        persona = self.settings.persona_dir
        persona.mkdir(parents=True)
        (persona / "anchor.txt").write_text("小林: 固定样本里的话\n我: 嗯\n", encoding="utf-8")
        (persona / "FACTS.md").write_text("我养了一只叫团子的猫", encoding="utf-8")

    async def write(self) -> dict[str, int]:
        return await write_sft(self.settings, self.db, None, StyleLimits(), self.settings.sft_dir, 20, 5, 6)

    async def test_examples_are_the_live_prompt_with_my_real_reply(self) -> None:
        counts = await self.write()
        self.assertEqual((counts["train"], counts["valid"], counts["eval"]), (20, 5, 6))
        self.assertNotIn("eval_recall", counts, "no recall index, no recall file")
        train = read(self.settings.sft_dir / "train.jsonl")
        for row in train:
            messages = row["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[-1]["role"], "user")
            self.assertRegex(messages[-1]["content"], r"小林: 第\d+次问你在不在\n小林: 人呢$")
            self.assertRegex(row["reply"], r"^真实回答\d+$")
            text = json.dumps(messages, ensure_ascii=False)
            self.assertNotIn("固定样本里的话", text, "the weights stand in for the fixed sample")
            self.assertNotIn("团子", text, "facts are left out of training")
        valid = read(self.settings.sft_dir / "valid.jsonl")
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in valid})

        held_out = {pair.pair_id for pair in self.db.pairs("test")}
        self.assertFalse({row["id"] for row in train + valid} & held_out)
        prompts = read(self.settings.sft_dir / "eval.jsonl")
        self.assertEqual([row["id"] for row in prompts], [p.pair_id for p in sample_pairs(self.db, 6, 0)])
        self.assertTrue(all("reply" not in row for row in prompts), "my real answers stay here")

    async def test_eval_takes_replies_written_elsewhere(self) -> None:
        await self.write()
        ids = [row["id"] for row in read(self.settings.sft_dir / "eval.jsonl")]
        path = self.tmp / "eval_out.jsonl"
        lines = [{"id": ids[0], "reply": "在\n刚才在忙"}, {"id": ids[1], "reply": ""}]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lines), encoding="utf-8")

        model = FakeModel(say("不该被调用"))
        items = await run_eval(
            self.settings,
            self.db,
            model,
            None,
            Persona(self.settings.persona_dir),
            None,
            StyleLimits(),
            EvalOptions(n=6),
            replies=load_replies(path),
        )
        self.assertEqual(model.calls, [])
        by_id = {item.pair_id: item for item in items}
        self.assertEqual((by_id[ids[0]].kind, by_id[ids[0]].bot), ("said", ["在", "刚才在忙"]))
        self.assertEqual(by_id[ids[1]].kind, "silent", "an empty reply is silence")
        self.assertEqual(by_id[ids[2]].kind, "silent", "a moment with no reply is silence")


if __name__ == "__main__":
    unittest.main()
