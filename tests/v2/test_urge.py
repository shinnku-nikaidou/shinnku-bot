"""Joining in uncalled: the moments in my history, the prior, the gate, the engine."""

from __future__ import annotations

import asyncio
import math
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.ingest import ingest
from shinnku_bot.corpus.moments import (
    build_urge,
    collect,
    format_urge,
    load_moments,
    save_moments,
    sender_rates,
)
from shinnku_bot.corpus.pairs import assign_holdout, build_pairs
from shinnku_bot.eval.urge_check import check_urge
from shinnku_bot.harness.engine import Harness
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.prompt import UNPROMPTED
from shinnku_bot.harness.urge import (
    BACKOFF_BASE_S,
    BACKOFF_MAX_S,
    COOLDOWN_S,
    FEATURES,
    LONG_AGO_S,
    Gate,
    UrgeModel,
    load_urge,
    moment_features,
)
from shinnku_bot.store import Store

from tests.v2.fakes import BOT_ID, FakeModel, FakeOutbox, make_settings, msg, no_sleep, say, silent
from tests.v2.fixture_export import OWNER_ID, PEOPLE, message, write_export

ME, LIN, SHU = PEOPLE[0], PEOPLE[1], PEOPLE[2]
T0 = 1_700_000_000
CHAT = 1001
UTC = ZoneInfo("UTC")
FLAT = [1 / 24] * 24


def always(rate: float = 1.0, **extra: object) -> UrgeModel:
    """A model with no learned prior: I join in at `rate` of all moments."""
    return UrgeModel(base_rate=rate, hour_share=FLAT, **extra)


class FeatureTests(unittest.TestCase):
    def test_features(self) -> None:
        msgs = [
            msg(1, "我先说", ts=T0, is_me=True, user_id=BOT_ID),
            msg(2, "嗯", ts=T0 + 250, user_id=7),
            msg(3, "去不去？", ts=T0 + 290, user_id=8),
        ]
        hours = [0.0] * 24
        hours[22] = 0.2  # T0 + 290 is 22:18 UTC
        got = dict(zip(FEATURES, moment_features(msgs, hours, {8: 0.4}, 0.1, UTC)))
        self.assertAlmostEqual(got["since_me"], math.log1p(290))
        self.assertEqual(got["me_in_last10"], 0.1)
        self.assertEqual(got["speakers_last10"], 2)
        self.assertEqual(got["msgs_last60s"], 2)
        self.assertEqual(got["hour_share"], 0.2)
        self.assertEqual(got["sender_rate"], 0.4)
        self.assertEqual(got["is_question"], 1)
        self.assertAlmostEqual(got["length"], math.log1p(4))
        self.assertEqual(got["is_text"], 1)

    def test_strangers_and_my_absence(self) -> None:
        msgs = [msg(1, "", ts=T0, user_id=None, kind="photo")]
        got = dict(zip(FEATURES, moment_features(msgs, FLAT, {8: 0.4}, 0.1, UTC)))
        self.assertAlmostEqual(got["since_me"], math.log1p(LONG_AGO_S))
        self.assertEqual((got["sender_rate"], got["is_question"], got["is_text"]), (0.1, 0, 0))
        self.assertEqual(dict(zip(FEATURES, moment_features([msg(1, "来吗")], FLAT, {}, 0.1, UTC)))["is_question"], 1)


class ModelTests(unittest.TestCase):
    def features(self, hour_share: float = 1 / 24) -> list[float]:
        values = [0.0] * len(FEATURES)
        values[FEATURES.index("hour_share")] = hour_share
        return values

    def test_without_a_prior_my_rate_follows_the_hour(self) -> None:
        model = always(0.1)
        self.assertAlmostEqual(model.probability(self.features()), 0.1)
        self.assertAlmostEqual(model.probability(self.features(2 / 24)), 0.2)
        self.assertAlmostEqual(model.probability(self.features(1.0)), 0.3)  # at most three times
        self.assertAlmostEqual(model.probability(self.features(0.0)), 0.02)

    def test_levels_scale_how_often_the_model_is_asked(self) -> None:
        model = always(0.1)
        self.assertEqual(model.call_probability(self.features(), 0), 0.0)
        self.assertAlmostEqual(model.call_probability(self.features(), 1), 0.025)
        self.assertAlmostEqual(model.call_probability(self.features(), 3), 0.1)
        self.assertEqual(model.call_probability(self.features(), 9), 0.0)
        shy = always(0.1, speak_rate=0.5)  # the model only speaks half the time it is asked
        self.assertAlmostEqual(shy.call_probability(self.features(), 3), 0.2)
        self.assertEqual(always(0.9, speak_rate=0.01).call_probability(self.features(), 3), 1.0)

    def test_learned_prior(self) -> None:
        count = len(FEATURES)
        weights = [0.0] * count
        weights[FEATURES.index("sender_rate")] = 4.0
        model = always(0.1, weights=weights, bias=0.0, mean=[0.0] * count, std=[1.0] * count)
        self.assertAlmostEqual(model.probability([0.0] * count), 0.5)
        keen = [0.0] * count
        keen[FEATURES.index("sender_rate")] = 2.0
        self.assertGreater(model.probability(keen), 0.9)

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "persona" / "urge.json"
            self.assertIsNone(load_urge(path))
            model = always(0.1, sender_rate={7: 0.5}, speak_rate=0.6, auc=0.7)
            model.save(path)
            self.assertEqual(load_urge(path).to_json(), model.to_json())
            self.assertEqual(load_urge(path).sender_rate, {7: 0.5})
            path.write_text(path.read_text().replace("since_me", "renamed"))
            self.assertIsNone(load_urge(path))  # built for other features: unusable
            path.write_text("{")
            self.assertIsNone(load_urge(path))


class GateTests(unittest.TestCase):
    NOW = float(T0)

    def gate(self, model: UrgeModel | None = always(), rng: float = 0.5, **settings: object) -> Gate:
        with tempfile.TemporaryDirectory() as tmp:
            return Gate(make_settings(Path(tmp), **settings), model, rng=lambda: rng)

    def ask(self, gate: Gate, now: float | None = None, level: int = 3) -> tuple[bool, str]:
        return gate.decide(100, level, [msg(1, "在吗", ts=T0)], self.NOW if now is None else now)

    def test_reasons_not_to_ask(self) -> None:
        self.assertEqual(self.ask(self.gate(), level=0), (False, "level"))
        self.assertEqual(self.ask(self.gate(model=None)), (False, "no_model"))
        gate = self.gate()
        self.assertEqual(gate.decide(100, 3, [], self.NOW), (False, "nothing_new"))
        self.assertEqual(gate.decide(100, 3, [msg(1, "我说的", is_me=True)], self.NOW), (False, "nothing_new"))
        self.assertEqual(self.ask(gate), (True, "urge"))

    def test_the_dice_follow_the_level(self) -> None:
        self.assertEqual(self.ask(self.gate(rng=0.5), level=1), (False, "dice"))
        self.assertEqual(self.ask(self.gate(rng=0.2), level=1), (True, "urge"))
        self.assertEqual(self.ask(self.gate(always(0.0), rng=0.0)), (False, "dice"))

    def test_i_stay_out_while_the_owner_is_talking(self) -> None:
        gate = self.gate(yield_to_owner_min=10)
        gate.owner_spoke(100, self.NOW - 60)
        self.assertEqual(self.ask(gate), (False, "owner_here"))
        self.assertEqual(gate.decide(200, 3, [msg(1, "在吗")], self.NOW), (True, "urge"))  # another chat
        self.assertEqual(self.ask(gate, self.NOW + 600), (True, "urge"))

    def test_cooldown_after_speaking(self) -> None:
        gate = self.gate()
        gate.spoke(100, self.NOW)
        self.assertEqual(self.ask(gate, self.NOW + COOLDOWN_S - 1), (False, "cooldown"))
        self.assertEqual(self.ask(gate, self.NOW + COOLDOWN_S), (True, "urge"))

    def test_backing_off_doubles_and_resets(self) -> None:
        gate = self.gate()
        gate.stayed_silent(100, self.NOW)
        self.assertEqual(self.ask(gate, self.NOW + BACKOFF_BASE_S - 1), (False, "backoff"))
        self.assertEqual(self.ask(gate, self.NOW + BACKOFF_BASE_S), (True, "urge"))
        gate.stayed_silent(100, self.NOW)
        gate.stayed_silent(100, self.NOW)
        self.assertEqual(self.ask(gate, self.NOW + 4 * BACKOFF_BASE_S - 1), (False, "backoff"))
        for _ in range(20):
            gate.stayed_silent(100, self.NOW)
        self.assertEqual(self.ask(gate, self.NOW + BACKOFF_MAX_S), (True, "urge"))
        gate.spoke(100, self.NOW)
        gate.stayed_silent(100, self.NOW + 100)
        self.assertEqual(self.ask(gate, self.NOW + 100 + BACKOFF_BASE_S), (True, "urge"))

    def test_budgets(self) -> None:
        gate = self.gate(proactive_calls_per_hour=2, proactive_calls_per_day=3)
        self.assertEqual(self.ask(gate, self.NOW), (True, "urge"))
        self.assertEqual(self.ask(gate, self.NOW + 1), (True, "urge"))
        self.assertEqual(self.ask(gate, self.NOW + 2), (False, "hour_cap"))
        self.assertEqual(self.ask(gate, self.NOW + 3600), (True, "urge"))
        self.assertEqual(self.ask(gate, self.NOW + 7300), (False, "day_cap"))
        self.assertEqual(self.ask(gate, self.NOW + 86400 + 2), (True, "urge"))

    def test_a_refusal_costs_no_budget(self) -> None:
        gate = self.gate(rng=0.99, proactive_calls_per_hour=1)
        for _ in range(5):
            self.assertEqual(self.ask(gate, level=1), (False, "dice"))
        gate._rng = lambda: 0.0
        self.assertEqual(self.ask(gate, level=1), (True, "urge"))


class CorpusCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def corpus(self, messages: list[dict], test_pairs: int) -> CorpusDB:
        export = write_export(self.tmp / "export" / "result.json", messages, chat_id=CHAT)
        db = CorpusDB(":memory:")
        self.addCleanup(db.close)
        ingest(db, export, OWNER_ID)
        build_pairs(db)
        assign_holdout(db, per_chat=test_pairs, min_train=1)
        return db

    def rounds(self, count: int = 900) -> list[dict]:
        """I answer 小林 every time and 阿树 never."""
        messages, ts, msg_id = [], T0, 0
        for index in range(count):
            ts += 300
            msg_id += 1
            if index % 2 == 0:
                messages.append(message(msg_id, ts, LIN, f"小林说的第{index}句"))
                msg_id += 1
                messages.append(message(msg_id, ts + 10, ME, f"我接的第{index}句"))
            else:
                messages.append(message(msg_id, ts, SHU, f"阿树说的第{index}句"))
        return messages


class CollectTests(CorpusCase):
    def test_moments(self) -> None:
        db = self.corpus(
            [
                message(1, T0, ME, "开头"),
                message(2, T0 + 100, LIN, "有人吗"),
                message(3, T0 + 110, ME, "在"),
                message(4, T0 + 200, LIN, "吃了没"),
                message(5, T0 + 201, SHU, "我吃了"),
                message(6, T0 + 400, LIN, "真红在吗"),
                message(7, T0 + 410, ME, "在啊"),
                message(8, T0 + 500, SHU, "这句是回我的", reply_to_message_id=7),
                message(9, T0 + 5000, LIN, "半夜了"),
                message(10, T0 + 5100, SHU, "睡了"),
                message(11, T0 + 5300, ME, "还没睡"),
                message(12, T0 + 30000, LIN, "明天见"),
                message(13, T0 + 30010, SHU, "明天见"),
                message(14, T0 + 40000, LIN, "还在吗"),
                message(15, T0 + 40010, ME, "在"),
                message(16, T0 + 41000, LIN, "那好"),
                message(17, T0 + 50000, SHU, "有人吗"),
                message(18, T0 + 50100, LIN, "没人"),
            ],
            test_pairs=1,
        )
        moments, counts = collect(db, CHAT, UTC, FLAT, name_triggers=("真红",))
        self.assertEqual(
            [(m.msg_id, m.spoke, m.test, m.present) for m in moments],
            [
                (2, True, False, True),  # I answered within two minutes
                (5, False, False, True),  # the chat paused and I let it pass
                (9, False, False, True),  # I was around: I spoke five minutes later
                (10, False, False, True),  # ... but too late to count as an answer to this
                (14, True, True, True),
                (16, False, True, True),
                (17, False, True, False),  # I was away; kept only because it is held out
            ],
        )
        # 4 is not a pause (阿树 spoke a second later); 6 and 8 called me; 12 and 13: I was away.
        self.assertEqual(counts, {"joined_uncalled": 2, "answered_when_called": 1, "pauses": 11})
        self.assertEqual(moments[0].sender_id, 7)
        self.assertEqual(len(moments[0].features), len(FEATURES))

    def test_sender_rates_lean_on_the_base_rate(self) -> None:
        db = self.corpus(self.rounds(60), test_pairs=5)
        moments, _ = collect(db, CHAT, UTC, FLAT)
        rates = sender_rates(moments, base=0.5)
        self.assertGreater(rates[7], 0.75)
        self.assertLess(rates[8], 0.25)
        self.assertEqual(sender_rates(moments[:6], base=0.5), {})  # too few moments each


class BuildTests(CorpusCase):
    def test_learns_who_i_answer(self) -> None:
        db = self.corpus(self.rounds(), test_pairs=100)
        model, report, held_out = build_urge(db, "UTC")
        self.assertGreater(report["auc"], 0.95)
        self.assertTrue(report["prior_kept"])
        self.assertIsNotNone(model.weights)
        self.assertGreater(model.sender_rate[7], 0.9)
        self.assertLess(model.sender_rate[8], 0.1)
        self.assertEqual(report["joined_uncalled"], 450)
        self.assertAlmostEqual(report["base_rate"], 0.5, places=1)
        self.assertEqual(report["test"], len(held_out))
        self.assertTrue(all(m.test for m in held_out))
        self.assertGreater(report["calls_per_day_level3"], report["calls_per_day_level1"])

        lin = [m for m in held_out if m.sender_id == 7][0]
        shu = [m for m in held_out if m.sender_id == 8][0]
        self.assertGreater(model.probability(lin.features), 0.8)
        self.assertLess(model.probability(shu.features), 0.2)

        text = format_urge(report)
        self.assertIn("kept: True", text)
        self.assertNotIn("小林说的", text)

        path = self.tmp / "distill" / "moments.jsonl"
        save_moments(path, held_out)
        loaded = load_moments(path)
        self.assertEqual(loaded[0], {"chat_id": CHAT, "msg_id": held_out[0].msg_id, "spoke": held_out[0].spoke})
        self.assertEqual(load_moments(self.tmp / "missing.jsonl"), [])

    def test_too_little_history_keeps_only_the_rates(self) -> None:
        db = self.corpus(self.rounds(40), test_pairs=5)
        model, report, _ = build_urge(db, "UTC")
        self.assertIsNone(report["auc"])
        self.assertFalse(report["prior_kept"])
        self.assertIsNone(model.weights)
        self.assertAlmostEqual(model.base_rate, 0.5, places=1)
        self.assertIn("kept: False", format_urge(report))


class EngineTests(unittest.IsolatedAsyncioTestCase):
    def build(self, model: FakeModel, level: int = 3, **settings: object) -> Harness:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.store.set_talk_level(100, level)
        self.outbox = FakeOutbox()
        self.model = model
        config = make_settings(Path(tmp.name), **settings)
        return Harness(
            config, self.store, model, self.outbox, Persona(config.persona_dir),
            bot_id=BOT_ID, gate=Gate(config, always(), rng=lambda: 0.0), sleep=no_sleep, rng=lambda: 0.5,
        )

    async def settle(self, harness: Harness) -> None:
        for _ in range(300):
            await asyncio.sleep(0.01)
            if all(chat.task is None for chat in harness._chats.values()):
                return
        self.fail("harness did not settle")

    async def test_i_join_in_without_being_called(self) -> None:
        harness = self.build(FakeModel(say("吃面")))
        harness.handle(msg(1, "今晚吃啥"), None, title="测试群")
        await self.settle(harness)
        self.assertEqual(self.outbox.sent, [(100, "吃面", None)])
        self.assertIn(UNPROMPTED, self.model.calls[0][-1]["content"])

    async def test_not_at_level_zero_and_not_for_bots(self) -> None:
        harness = self.build(FakeModel(say("吃面")), level=0)
        harness.handle(msg(1, "今晚吃啥"), None, title="测试群")
        await self.settle(harness)
        self.store.set_talk_level(100, 3)
        harness.handle(msg(2, "我是另一个 bot", is_bot=True), None, title="测试群")
        await self.settle(harness)
        self.assertEqual(self.model.calls, [])

    async def test_quiet_time_applies(self) -> None:
        harness = self.build(FakeModel(say("吃面")))
        self.store.set_quiet_until(100, 2**40)
        harness.handle(msg(1, "今晚吃啥"), None, title="测试群")
        await self.settle(harness)
        self.assertEqual(self.model.calls, [])

    async def test_choosing_silence_makes_me_ask_less(self) -> None:
        harness = self.build(FakeModel(silent()))
        harness.handle(msg(1, "今晚吃啥"), None, title="测试群")
        await self.settle(harness)
        harness.handle(msg(2, "没人理我"), None, title="测试群")
        await self.settle(harness)
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.outbox.sent, [])

    async def test_several_lines_are_one_moment(self) -> None:
        harness = self.build(FakeModel(say("吃面")))
        harness.handle(msg(1, "今晚吃啥"), None, title="测试群")
        harness.handle(msg(2, "火锅？", user_id=8, name="阿树"), None, title="测试群")
        await self.settle(harness)
        self.assertEqual(len(self.model.calls), 1)
        self.assertIn("阿树: 火锅？", self.model.calls[0][-1]["content"])

    async def test_being_called_takes_over(self) -> None:
        harness = self.build(FakeModel(say("在")))
        harness.handle(msg(1, "今晚吃啥"), None, title="测试群")
        harness.handle(msg(2, "真红在吗", user_id=8, name="阿树"), "name", title="测试群")
        await self.settle(harness)
        self.assertEqual(len(self.model.calls), 1)
        self.assertNotIn(UNPROMPTED, self.model.calls[0][-1]["content"])

    async def test_a_line_while_i_am_thinking_does_not_restart_me(self) -> None:
        harness = self.build(FakeModel(say("吃面"), delay=0.1))
        harness.handle(msg(1, "今晚吃啥"), None, title="测试群")
        await asyncio.sleep(0.06)  # past the quiet window, into the model call
        harness.handle(msg(2, "火锅？"), None, title="测试群")
        await self.settle(harness)
        self.assertEqual((len(self.model.calls), self.model.finished), (1, 1))

    async def test_i_stay_out_while_the_owner_talks(self) -> None:
        harness = self.build(FakeModel(say("吃面")))
        harness.handle(msg(1, "我来了", user_id=1, name="本人"), None, title="测试群")
        await self.settle(harness)
        harness.handle(msg(2, "今晚吃啥"), None, title="测试群")
        await self.settle(harness)
        self.assertEqual(self.model.calls, [])

    async def test_without_a_gate_nothing_changes(self) -> None:
        harness = self.build(FakeModel(say("吃面")))
        harness.gate = None
        harness.handle(msg(1, "今晚吃啥"), None, title="测试群")
        await self.settle(harness)
        self.assertEqual(self.model.calls, [])


class UrgeCheckTests(CorpusCase):
    def test_rates_where_i_joined_and_where_i_passed(self) -> None:
        db = self.corpus(self.rounds(40), test_pairs=5)
        moments = [
            {"chat_id": CHAT, "msg_id": 1, "spoke": True},
            {"chat_id": CHAT, "msg_id": 4, "spoke": True},
            {"chat_id": CHAT, "msg_id": 3, "spoke": False},
        ]
        model = FakeModel(say("来"), silent(), say("来"))
        result = asyncio.run(
            check_urge(make_settings(self.tmp), db, moments, model, Persona(self.tmp), None, StyleLimits(), concurrency=1)
        )
        self.assertEqual(result["where_i_joined"], {"n": 2, "spoke": 1, "rate": 0.5})
        self.assertEqual(result["where_i_passed"], {"n": 1, "spoke": 1, "rate": 1.0})
        self.assertEqual(result["errors"], 0)
        self.assertIn(UNPROMPTED, model.calls[0][-1]["content"])

    def test_errors_are_left_out(self) -> None:
        class Broken:
            async def complete(self, messages: object, tools: object) -> None:
                raise TimeoutError

        db = self.corpus(self.rounds(40), test_pairs=5)
        moments = [{"chat_id": CHAT, "msg_id": 1, "spoke": True}]
        result = asyncio.run(
            check_urge(make_settings(self.tmp), db, moments, Broken(), Persona(self.tmp), None, StyleLimits())
        )
        self.assertEqual((result["errors"], result["where_i_joined"]["n"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
