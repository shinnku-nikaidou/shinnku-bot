"""Blind test: sampling held-out moments, generating, judging, scoring, the quiz."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.ingest import ingest
from shinnku_bot.corpus.pairs import assign_holdout, build_pairs
from shinnku_bot.eval.metrics import (
    OUT_OF_CHARACTER,
    Item,
    format_summary,
    js_divergence,
    summarize,
    wilson,
)
from shinnku_bot.eval.quiz import run_quiz
from shinnku_bot.eval.run import (
    EvalOptions,
    judge_messages,
    judge_one,
    read_pick,
    reference_scenes,
    run_eval,
    sample_pairs,
    save_report,
)
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.persona import Persona
from shinnku_bot.llm import ModelReply, ToolCall

from tests.v2.fakes import FakeModel, make_settings, say, silent
from tests.v2.fixture_export import OWNER_ID, PEOPLE, message, write_export

ME, LIN = PEOPLE[0], PEOPLE[1]
T0 = 1_700_000_000
HOUR = 3600


def pick(choice: str) -> ModelReply:
    return ModelReply(tool_calls=(ToolCall("pick", {"choice": choice}),))


class MetricsTests(unittest.TestCase):
    def test_wilson(self) -> None:
        self.assertEqual(wilson(0, 0), (0.0, 0.0))
        low, high = wilson(50, 100)
        self.assertAlmostEqual(low, 0.404, places=3)
        self.assertAlmostEqual(high, 0.596, places=3)
        self.assertEqual(wilson(10, 10)[1], 1.0)

    def test_js_divergence(self) -> None:
        self.assertEqual(js_divergence([0.5, 0.5], [0.5, 0.5]), 0)
        self.assertAlmostEqual(js_divergence([1, 0], [0, 1]), 1.0)

    def test_out_of_character(self) -> None:
        for text in ("我人设是不帮忙的", "作为阿橘 我会这么说", "the user asked", "按角色回复", "AI 不该这样"):
            with self.subTest(text=text):
                self.assertTrue(OUT_OF_CHARACTER.search(text))
        for text in ("被点名了 回一句", "懒得理他", "他又在说 FAIR 的事"):
            with self.subTest(text=text):
                self.assertFalse(OUT_OF_CHARACTER.search(text))

    def test_summarize(self) -> None:
        items = [
            Item(1, 1, 0, real=["打"], bot=["打"], kind="said", judged="real", attempts=1, reasoning="回一句", latency_s=2.0, prompt_tokens=100, cached_tokens=80, recalled=4),
            Item(2, 1, 0, real=["不去", "太累了"], bot=["我觉得还是不去比较好因为实在太累了"], kind="said", judged="bot", attempts=2, violations=["too_long"], reasoning="我人设是懒", latency_s=4.0, prompt_tokens=100, cached_tokens=0, recalled=2),
            Item(3, 1, 0, real=["嗯"], kind="silent", attempts=1, latency_s=1.0, prompt_tokens=100),
            Item(4, 1, 0, real=["嗯"], error="TimeoutError"),
        ]
        s = summarize(items)
        self.assertEqual((s["items"], s["errors"], s["spoke"], s["judged"]), (4, 1, 2, 2))
        self.assertEqual(s["silent_rate"], round(1 / 3, 4))
        self.assertEqual(s["judge_accuracy"], 0.5)
        self.assertEqual(s["exact_match_rate"], 0.5)
        self.assertEqual(s["burst_mean"], {"real": 1.5, "bot": 1.0})
        self.assertEqual(s["out_of_character_rate"], 0.5)
        self.assertEqual(s["retry_rate"], round(1 / 3, 4))
        self.assertEqual(s["violations"], {"too_long": 1})
        self.assertEqual((s["latency_p50"], s["latency_p95"]), (2.0, 4.0))
        self.assertEqual(s["cached_share"], round(80 / 300, 4))
        self.assertGreater(s["length_js"], 0)
        text = format_summary("x", s)
        self.assertIn("50.0%", text)
        for chat_text in ("不去", "太累了", "人设"):
            self.assertNotIn(chat_text, text)

    def test_summarize_nothing(self) -> None:
        s = summarize([])
        self.assertIsNone(s["judge_accuracy"])
        self.assertIn("n/a", format_summary("empty", s))


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    ITEM = Item(1, 1, 0, context=["小林: 打游戏吗"], real=["打"], bot=["好的呀"])

    def test_read_pick(self) -> None:
        self.assertEqual(read_pick(pick("B")), "B")
        self.assertEqual(read_pick(pick("b")), "B")
        self.assertEqual(read_pick(ModelReply(content="A")), "A")
        self.assertEqual(read_pick(ModelReply(content="我认为候选 B 是真的")), "B")
        self.assertEqual(read_pick(ModelReply(content="不确定")), "")
        self.assertEqual(read_pick(pick("C")), "")

    def test_messages(self) -> None:
        system, user = judge_messages("小林: 吃了没\n我: 没", self.ITEM, bot_first=True)
        self.assertIn("小林: 吃了没\n我: 没", system["content"])
        self.assertEqual(
            user["content"], "【上文】\n小林: 打游戏吗\n\n【候选 A】\n我: 好的呀\n\n【候选 B】\n我: 打"
        )
        _, swapped = judge_messages("", self.ITEM, bot_first=False)
        self.assertIn("【候选 A】\n我: 打", swapped["content"])

    async def judged(self, choice: str, seed: int) -> tuple[str, bool]:
        item = Item(1, 1, 0, context=["x"], real=["打"], bot=["好的呀"])
        judge = FakeModel(pick(choice))
        await judge_one(judge, "", item, random.Random(seed))
        bot_first = "【候选 A】\n我: 好的呀" in judge.calls[0][1]["content"]
        return item.judged, bot_first

    async def test_pick_is_mapped_back_through_the_random_order(self) -> None:
        seen = set()
        for seed in range(12):
            judged, bot_first = await self.judged("A", seed)
            self.assertEqual(judged, "bot" if bot_first else "real")
            seen.add(bot_first)
        self.assertEqual(seen, {True, False}, "both orders occur")

    async def test_silent_items_and_unreadable_answers_are_not_judged(self) -> None:
        quiet = Item(1, 1, 0, real=["打"])
        await judge_one(FakeModel(pick("A")), "", quiet, random.Random(0))
        self.assertEqual(quiet.judged, "")
        item = Item(1, 1, 0, real=["打"], bot=["好"])
        await judge_one(FakeModel(ModelReply(content="？")), "", item, random.Random(0))
        self.assertEqual(item.judged, "")


class RunTests(unittest.IsolatedAsyncioTestCase):
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
                messages.append(message(3 * i + 1, ts, LIN, f"第{i}次问你在不在"))
                messages.append(message(3 * i + 2, ts + 5, ME, f"真实回答{i}"))
                # A slow follow-up of mine: its context ends with my own line.
                messages.append(message(3 * i + 3, ts + 600, ME, "补一句"))
            ingest(self.db, write_export(self.tmp / str(chat_id) / "result.json", messages, chat_id=chat_id), OWNER_ID)
        build_pairs(self.db)
        assign_holdout(self.db, per_chat=20, min_train=20)
        self.persona = Persona(self.settings.persona_dir)

    def test_sample_pairs(self) -> None:
        picked = sample_pairs(self.db, 12, seed=0)
        self.assertEqual(len(picked), 12)
        self.assertEqual(sorted(p.chat_id for p in picked), [1] * 6 + [2] * 6)
        self.assertTrue(all(p.split == "test" for p in picked))
        for pair in picked:
            self.assertFalse(self.db.context_of(pair)[-1].is_me)
        self.assertEqual([p.pair_id for p in picked], [p.pair_id for p in sample_pairs(self.db, 12, seed=0)])
        self.assertNotEqual([p.pair_id for p in picked], [p.pair_id for p in sample_pairs(self.db, 12, seed=1)])

    def test_sample_is_capped_by_what_exists(self) -> None:
        self.assertEqual(len(sample_pairs(self.db, 10_000, seed=0)), 20, "half the test pairs follow my own line")

    async def run_eval(self, model: FakeModel, judge_model: Any, **options: Any) -> list[Item]:
        options.setdefault("judge", True)
        return await run_eval(
            self.settings, self.db, model, judge_model, self.persona, None, StyleLimits(), EvalOptions(n=8, **options)
        )

    async def test_end_to_end(self) -> None:
        class SpotsTheBot:
            """Picks whichever candidate is not the bot's fixed line."""

            async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **options: Any) -> ModelReply:
                text = messages[1]["content"]
                return pick("B" if "【候选 A】\n我: 假的回答" in text else "A")

        model = FakeModel(say("假的回答"))
        items = await self.run_eval(model, SpotsTheBot())
        self.assertEqual(len(items), 8)
        for item in items:
            self.assertEqual((item.kind, item.bot, item.judged, item.error), ("said", ["假的回答"], "real", ""))
            self.assertRegex(item.real[0], r"^真实回答\d+$")
            self.assertRegex(item.context[-1], r"^小林: 第\d+次问你在不在$")
        self.assertEqual(summarize(items)["judge_accuracy"], 1.0)
        self.assertTrue(model.calls[0][-1]["content"].startswith("（群「测试群」，周"))

    async def test_silence_and_errors_are_counted_not_judged(self) -> None:
        items = await self.run_eval(FakeModel(silent()), FakeModel(pick("A")))
        self.assertTrue(all(item.kind == "silent" and not item.judged for item in items))
        self.assertEqual(summarize(items)["silent_rate"], 1.0)

        class Broken(FakeModel):
            async def complete(self, messages: Any, tools: Any, **options: Any) -> ModelReply:
                raise TimeoutError("gateway")

        items = await self.run_eval(Broken(say("x")), None)
        self.assertEqual({item.error for item in items}, {"TimeoutError"})
        self.assertEqual(summarize(items)["error_types"], {"TimeoutError": 8})

    async def test_gives_up_when_nothing_works(self) -> None:
        class Broken(FakeModel):
            async def complete(self, messages: Any, tools: Any, **options: Any) -> ModelReply:
                self.calls.append(messages)
                raise RuntimeError("quota")  # not a timeout, so no second try: one call per item

        model = Broken(say("x"))
        items = await run_eval(
            self.settings, self.db, model, None, self.persona, None, StyleLimits(), EvalOptions(n=20, concurrency=2)
        )
        self.assertEqual(len(items), 20)
        self.assertLess(len(model.calls), 12, "stops calling once the first few all fail")
        self.assertEqual(summarize(items)["error_types"]["Skipped"], 20 - len(model.calls))

    async def test_judging_can_be_skipped(self) -> None:
        judge = FakeModel(pick("A"))
        items = await self.run_eval(FakeModel(say("嗯")), judge, judge=False)
        self.assertEqual(judge.calls, [])
        self.assertTrue(all(not item.judged for item in items))

    async def test_anchor_switch(self) -> None:
        self.settings.persona_dir.mkdir(parents=True)
        (self.settings.persona_dir / "anchor.txt").write_text("小林: 吃了没\n我: 没\n\n小林: 在吗\n我: 在\n", encoding="utf-8")
        self.assertEqual(reference_scenes(self.persona, count=1), "小林: 在吗\n我: 在")

        with_anchor = FakeModel(say("嗯"))
        await self.run_eval(with_anchor, None)
        self.assertIn("小林: 吃了没", with_anchor.calls[0][0]["content"])
        without = FakeModel(say("嗯"))
        await self.run_eval(without, None, use_anchor=False)
        self.assertNotIn("小林: 吃了没", without.calls[0][0]["content"])

    async def test_report_and_quiz(self) -> None:
        items = await self.run_eval(FakeModel(say("假的回答")), FakeModel(pick("A")))
        path = self.tmp / "reports" / "eval-t.json"
        save_report(path, "t", EvalOptions(n=8), self.settings, items)
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((report["name"], len(report["items"]), report["summary"]["spoke"]), ("t", 8, 8))

        shown: list[str] = []

        def always_real(prompt: str) -> str:
            block = "\n".join(shown).rsplit("—— 第", 1)[1]
            after_a = block.split("  A:\n", 1)[1]
            return "A" if after_a.strip().startswith("真实回答") else "B"

        correct, answered = run_quiz(path, n=5, seed=3, ask=always_real, show=shown.append)
        self.assertEqual((correct, answered), (5, 5))
        self.assertIn("5/5", shown[-1])

        answers = iter(["", "q"])
        self.assertEqual(run_quiz(path, n=5, seed=3, ask=lambda _: next(answers), show=lambda _: None), (0, 0))


if __name__ == "__main__":
    unittest.main()
