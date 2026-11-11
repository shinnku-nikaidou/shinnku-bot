"""Ingesting exports, cutting reply pairs, holding out a test set, measuring style."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.ingest import flatten_text, ingest, sender_id, sender_name, to_row
from shinnku_bot.corpus.pairs import PairParams, assign_holdout, build_pairs, train_cutoffs
from shinnku_bot.corpus.stats import (
    build_fingerprint,
    format_fingerprint,
    limits_from_fingerprint,
    pace_from_fingerprint,
    percentile,
    save_fingerprint,
)
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.send import Pace, typing_delay
from shinnku_bot.harness.transcript import render_turns

from tests.v2.fixture_export import OWNER_ID, PEOPLE, message, random_messages, service, write_export

ME, LIN, SHU = PEOPLE[0], PEOPLE[1], PEOPLE[2]
T0 = 1_700_000_000


class CorpusCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.db = CorpusDB(":memory:")
        self.addCleanup(self.db.close)

    def load(self, messages: list[dict], chat_id: int = 1001, **kwargs: object) -> None:
        path = write_export(self.tmp / str(chat_id) / "result.json", messages, chat_id=chat_id, **kwargs)
        self.result = ingest(self.db, path, OWNER_ID)


class ConversionTests(unittest.TestCase):
    def row(self, **extra: object):
        return to_row(message(5, T0, LIN, extra.pop("text", "在吗"), **extra), 1001, OWNER_ID)

    def test_ids_and_names(self) -> None:
        self.assertEqual(sender_id("user123"), 123)
        self.assertEqual(sender_id("channel456"), -456)
        self.assertIsNone(sender_id("weird"))
        self.assertIsNone(sender_id(None))
        self.assertEqual(sender_name("小林", 7), "小林")
        self.assertEqual(sender_name(None, 1234), "路人34")
        self.assertEqual(sender_name("  ", None), "某人")

    def test_flatten_text(self) -> None:
        self.assertEqual(flatten_text("x"), "x")
        self.assertEqual(flatten_text(["看 ", {"type": "link", "text": "http://a"}, 3]), "看 http://a")
        self.assertEqual(flatten_text(None), "")

    def test_plain_message(self) -> None:
        row = self.row(reply_to_message_id=4)
        self.assertEqual((row.msg.kind, row.msg.text, row.msg.reply_to, row.msg.ts), ("text", "在吗", 4, T0))
        self.assertEqual((row.msg.user_id, row.msg.user_name, row.msg.is_me), (7, "小林", False))
        self.assertFalse(row.canned)

    def test_owner_is_me(self) -> None:
        self.assertTrue(to_row(message(1, T0, ME, "在"), 1001, OWNER_ID).msg.is_me)

    def test_kinds(self) -> None:
        cases = {
            "sticker": dict(text="", media_type="sticker", sticker_emoji="😂", file="x"),
            "video": dict(text="", media_type="video_message", file="x"),
            "voice": dict(text="", media_type="voice_message", file="x"),
            "animation": dict(text="", media_type="animation", file="x"),
            "photo": dict(text="看", photo="x"),
            "document": dict(text="", file="x"),
            "other": dict(text="投票", poll={"question": "q"}),
        }
        for kind, extra in cases.items():
            with self.subTest(kind=kind):
                self.assertEqual(self.row(**extra).msg.kind, kind)
        self.assertEqual(self.row(text="", media_type="sticker", sticker_emoji="😂").msg.text, "😂")

    def test_canned(self) -> None:
        for extra in (
            dict(forwarded_from="某频道"),
            dict(via_bot="@gif"),
            dict(text="/rua"),
            dict(text="https://example.com/a?b=1"),
        ):
            with self.subTest(extra=extra):
                self.assertTrue(self.row(**extra).canned)
        self.assertFalse(self.row(text="看这个 https://example.com").canned)

    def test_dropped(self) -> None:
        self.assertIsNone(to_row(service(1, T0, LIN, "pin_message"), 1001, OWNER_ID))
        self.assertIsNone(self.row(text=""))
        self.assertIsNone(to_row({"type": "message", "id": "x"}, 1001, OWNER_ID))


class IngestTests(CorpusCase):
    def test_counts(self) -> None:
        messages = random_messages(500, seed=2)
        self.load(messages)
        expected = sum(1 for m in messages if to_row(m, 1001, OWNER_ID) is not None)
        self.assertEqual(self.result.read, 500)
        self.assertEqual(self.result.stored, expected)
        self.assertEqual(self.db.count(), expected)
        self.assertEqual(self.result.mine, self.db.count("is_me = 1"))
        self.assertTrue(self.result.complete)
        self.assertEqual(self.db.chat_title(1001), "测试群")

    def test_reingest_replaces_the_chat(self) -> None:
        self.load(random_messages(100))
        self.load(random_messages(40))
        self.assertEqual(self.db.count(), self.result.stored)
        self.assertEqual(self.db.chat_ids(), [1001])

    def test_truncated_export_is_marked_incomplete(self) -> None:
        self.load(random_messages(200), truncate=0.6)
        self.assertFalse(self.result.complete)
        self.assertGreater(self.result.stored, 0)
        complete = self.db.db.execute("SELECT complete FROM chat").fetchone()[0]
        self.assertEqual(complete, 0)


class PairTests(CorpusCase):
    def pairs(self, **params: int) -> list:
        build_pairs(self.db, PairParams(**params))
        return list(self.db.pairs())

    def test_burst_after_other_people(self) -> None:
        self.load(
            [
                message(1, T0, LIN, "今晚打游戏吗"),
                message(2, T0 + 10, SHU, "我十点后"),
                message(3, T0 + 20, ME, "打", reply_to_message_id=1),
                message(4, T0 + 25, ME, "十点"),
                message(5, T0 + 60, LIN, "好"),
            ]
        )
        (pair,) = self.pairs()
        self.assertEqual((pair.first_id, pair.last_id, pair.ctx_first_id), (3, 4, 1))
        self.assertEqual((pair.n_reply, pair.n_ctx, pair.reply_chars, pair.quoted), (2, 2, 3, True))
        self.assertEqual([m.text for m in self.db.context_of(pair)], ["今晚打游戏吗", "我十点后"])
        self.assertEqual([m.text for m in self.db.reply_of(pair)], ["打", "十点"])

    def test_context_renders_like_the_live_chat(self) -> None:
        self.load(
            [
                message(1, T0, LIN, "吃了没"),
                message(2, T0 + 5, ME, "没"),
                message(3, T0 + 30, LIN, "那走", reply_to_message_id=2),
                message(4, T0 + 40, ME, "行"),
            ]
        )
        second = self.pairs()[1]
        self.assertEqual(
            render_turns(self.db.context_of(second)),
            [
                {"role": "user", "content": "小林: 吃了没"},
                {"role": "assistant", "content": "没"},
                {"role": "user", "content": "小林（回复我）: 那走"},
            ],
        )

    def test_slow_followup_is_its_own_burst(self) -> None:
        self.load(
            [
                message(1, T0, LIN, "在吗"),
                message(2, T0 + 10, ME, "在"),
                message(3, T0 + 400, ME, "咋了"),
            ]
        )
        first, second = self.pairs(burst_gap_s=120)
        self.assertEqual((first.first_id, first.last_id), (2, 2))
        self.assertEqual((second.first_id, second.ctx_first_id, second.n_ctx), (3, 1, 2))

    def test_long_silence_starts_a_new_conversation(self) -> None:
        self.load(
            [
                message(1, T0, LIN, "在吗"),
                message(2, T0 + 7200, ME, "刚醒"),
                message(3, T0 + 7210, LIN, "懒狗"),
                message(4, T0 + 7220, ME, "确实"),
            ]
        )
        (pair,) = self.pairs(session_gap_s=1800)
        self.assertEqual((pair.first_id, pair.ctx_first_id), (4, 2), "msg 2 opened the conversation")

    def test_bursts_with_canned_or_media_lines_are_not_pairs(self) -> None:
        self.load(
            [
                message(1, T0, LIN, "在吗"),
                message(2, T0 + 5, ME, "/rua"),
                message(3, T0 + 20, LIN, "？"),
                message(4, T0 + 25, ME, "", media_type="sticker", sticker_emoji="😂", file="x"),
                message(5, T0 + 40, LIN, "说话"),
                message(6, T0 + 45, ME, "在"),
            ]
        )
        (pair,) = self.pairs()
        self.assertEqual(pair.first_id, 6)
        self.assertEqual(pair.n_ctx, 5, "skipped bursts still count as context")

    def test_context_window_is_capped(self) -> None:
        lines = [message(i, T0 + i, LIN, f"第{i}句") for i in range(1, 50)]
        self.load([*lines, message(50, T0 + 50, ME, "别刷了")])
        (pair,) = self.pairs(context_messages=10)
        self.assertEqual((pair.ctx_first_id, pair.n_ctx), (40, 10))

    def test_holdout_takes_the_newest_pairs_per_chat(self) -> None:
        for chat_id in (1, 2):
            messages = []
            for i in range(60):
                messages.append(message(2 * i + 1, T0 + 100 * i, LIN, "在吗"))
                messages.append(message(2 * i + 2, T0 + 100 * i + 5, ME, "在"))
            self.load(messages, chat_id=chat_id)
        build_pairs(self.db)
        counts = assign_holdout(self.db, per_chat=10, min_train=20)
        self.assertEqual(counts, {"train": 100, "test": 20})
        for chat_id in (1, 2):
            test = list(self.db.pairs("test", chat_id))
            train = list(self.db.pairs("train", chat_id))
            self.assertEqual(len(test), 10)
            self.assertLess(max(p.ts for p in train), min(p.ts for p in test))
        self.assertEqual(train_cutoffs(self.db), {1: T0 + 5005, 2: T0 + 5005})

    def test_small_chats_give_no_test_pairs(self) -> None:
        self.load([message(1, T0, LIN, "在吗"), message(2, T0 + 5, ME, "在")])
        build_pairs(self.db)
        self.assertEqual(assign_holdout(self.db, per_chat=1, min_train=5), {"train": 1})


class FingerprintTests(CorpusCase):
    def test_percentile(self) -> None:
        self.assertEqual(percentile([], 50), 0)
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2)
        self.assertEqual(percentile([1, 2, 3, 4], 99), 4)

    def test_numbers(self) -> None:
        self.load(
            [
                message(1, T0, LIN, "今晚打游戏吗"),
                message(2, T0 + 5, ME, "打"),
                message(3, T0 + 9, ME, "十点吧。"),
                message(4, T0 + 60, LIN, "ok"),
                message(5, T0 + 65, ME, "", media_type="sticker", sticker_emoji="😂", file="x"),
                message(6, T0 + 90, LIN, "？"),
                message(7, T0 + 95, ME, "/rua"),
            ]
        )
        build_pairs(self.db)
        fp = build_fingerprint(self.db, "UTC")
        self.assertEqual((fp["messages"], fp["text_messages"], fp["bursts"]), (4, 2, 1))
        self.assertEqual(fp["len"]["p99"], 4)
        self.assertEqual(fp["burst_size"]["2"], 1.0)
        self.assertEqual(fp["ends_with"]["period"], 0.5)
        self.assertEqual(fp["kind_share"], {"text": 0.75, "sticker": 0.25})
        self.assertEqual(fp["sticker_emoji"], [{"emoji": "😂", "count": 1}])
        self.assertEqual(fp["burst_gap_s"]["p50"], 4)
        self.assertEqual(fp["sec_per_char"]["p50"], 1.0)
        self.assertAlmostEqual(sum(fp["hour_share"]), 1.0)

    def test_held_out_messages_are_not_measured(self) -> None:
        messages = []
        for i in range(40):
            messages.append(message(2 * i + 1, T0 + 100 * i, LIN, "在吗"))
            messages.append(message(2 * i + 2, T0 + 100 * i + 5, ME, "在" if i < 30 else "这句很长很长很长很长很长"))
        self.load(messages)
        build_pairs(self.db)
        assign_holdout(self.db, per_chat=10, min_train=10)
        fp = build_fingerprint(self.db, "UTC")
        self.assertEqual(fp["text_messages"], 30)
        self.assertEqual(fp["len"]["p99"], 1)

    def test_console_summary_has_no_strings_from_the_chat(self) -> None:
        self.load(random_messages(3000, seed=4))
        build_pairs(self.db)
        fp = build_fingerprint(self.db, "UTC")
        self.assertTrue(fp["phrases"], "the fixture repeats lines often enough to produce phrases")
        text = format_fingerprint(fp)
        # The summary's own labels are ASCII, so only non-ASCII phrases can prove a leak.
        chat_phrases = [entry["text"] for entry in fp["phrases"] if not entry["text"].isascii()]
        self.assertTrue(chat_phrases)
        for phrase in chat_phrases:
            self.assertNotIn(phrase, text)
        self.assertNotIn("😂", text)

    def test_limits_from_fingerprint(self) -> None:
        path = self.tmp / "persona" / "fingerprint.json"
        self.assertEqual(limits_from_fingerprint(path), StyleLimits(), "missing file gives defaults")
        fp = {"len": {"p99": 47}, "burst_size": {"p99": 5}, "ends_with": {"period": 0.2}}
        save_fingerprint(fp, path)
        self.assertEqual(limits_from_fingerprint(path), StyleLimits(max_len=47, max_burst=5, strip_period=False))
        path.write_text(json.dumps({"len": {"p99": 3}, "burst_size": {"p99": 1}, "ends_with": {"period": 0}}))
        self.assertEqual(limits_from_fingerprint(path), StyleLimits(max_len=30, max_burst=3, strip_period=True))

    def test_pace_from_fingerprint(self) -> None:
        path = self.tmp / "persona" / "fingerprint.json"
        self.assertEqual(pace_from_fingerprint(path), Pace(), "missing file gives defaults")
        save_fingerprint({"sec_per_char": {"p50": 1.2}, "burst_gap_s": {"p50": 12}}, path)
        pace = pace_from_fingerprint(path)
        self.assertEqual((round(pace.sec_per_char, 2), pace.floor_s, pace.cap_s), (0.48, 1.0, 10.0))
        self.assertAlmostEqual(typing_delay("今晚打游戏吗来不来", 0.5, pace), 0.5 + 0.48 * 9)
        self.assertAlmostEqual(typing_delay("啊" * 200, 0.5, pace), 10.0)
        save_fingerprint({"sec_per_char": {"p50": 0.01}, "burst_gap_s": {"p50": 1}}, path)
        self.assertEqual(pace_from_fingerprint(path), Pace(sec_per_char=0.1, floor_s=1.0, cap_s=4.0))


if __name__ == "__main__":
    unittest.main()
