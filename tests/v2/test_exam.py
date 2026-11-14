"""Telling questions that test me as an AI from talk. All text here is made up."""

from __future__ import annotations

import unittest
from collections import Counter
from typing import Any

from shinnku_bot.config import load_settings
from shinnku_bot.corpus.db import CorpusDB, Row
from shinnku_bot.eval.exam_check import KINDS, addressed, check_real, check_set, format_set, load_set
from shinnku_bot.harness.exam import MAX_TEXT, exam_messages, is_exam, judge
from shinnku_bot.llm import ModelReply, ToolCall
from shinnku_bot.store import Msg

from tests.v2.fakes import BOT_ID, FakeModel, msg


def verdict(value: Any) -> ModelReply:
    return ModelReply(tool_calls=(ToolCall("judge", {"exam": value}),))


class MarkerModel:
    """Judges a message an exam when it contains 题; fails when it contains 坏."""

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **options: Any) -> ModelReply:
        user = messages[-1]["content"]
        if "坏" in user:
            raise RuntimeError("gateway down")
        return verdict("题" in user)


class ExamMessagesTests(unittest.TestCase):
    def test_shows_the_lines_before_what_it_replies_to_and_the_message(self) -> None:
        mine = msg(2, "是啊", user_id=BOT_ID, name="我", is_me=True)
        before = [msg(1, "今天好冷", name="小林"), mine]
        system, user = (m["content"] for m in exam_messages(msg(3, "真红 第一行\n第二行", name="小林"), mine, before))
        self.assertIn("judge", system)
        self.assertIn("某人: 今天好冷\nshinnku: 是啊", user)
        self.assertIn("这条回复的是shinnku说的：是啊", user)
        self.assertIn('"""\n真红 第一行\n第二行\n"""', user)
        self.assertNotIn("小林", user)

    def test_a_long_message_is_cut_and_only_the_last_lines_count(self) -> None:
        before = [msg(i, f"第{i}句") for i in range(1, 6)]
        user = exam_messages(msg(9, "题" * 3000), None, before)[1]["content"]
        self.assertIn("题" * MAX_TEXT + "…", user)
        self.assertNotIn("题" * (MAX_TEXT + 1), user)
        self.assertNotIn("第2句", user)
        self.assertIn("第3句", user)
        self.assertNotIn("回复的是", user)


class IsExamTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_call_without_thinking(self) -> None:
        model = FakeModel(verdict(True))
        self.assertTrue(await is_exam(model, msg(1, "真红 strawberry 里有几个 r")))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.options, [{"thinking": False}])
        self.assertFalse(await is_exam(FakeModel(verdict(False)), msg(2, "真红 吃了没")))

    async def test_no_verdict_counts_as_no(self) -> None:
        for reply in (RuntimeError("gateway down"), ModelReply(), verdict("true")):
            with self.subTest(reply=reply), self.assertLogs("shinnku_bot.harness.exam", "WARNING"):
                self.assertFalse(await is_exam(FakeModel(reply), msg(1, "真红 9.11 和 9.9 哪个大")))
        with self.assertRaises(ValueError):
            await judge(FakeModel(ModelReply()), msg(1, "真红"), None, [])


class SettingTests(unittest.TestCase):
    def test_on_unless_turned_off(self) -> None:
        env = {"TOKEN": "t", "ADMIN_TELEGRAM_USER_ID": "1", "LLM_BASE_URL": "u", "LLM_API_KEY": "k", "LLM_MODEL": "m"}
        self.assertTrue(load_settings(env).brush_off_exams)
        self.assertFalse(load_settings({**env, "BRUSH_OFF_EXAMS": "off"}).brush_off_exams)


class ExamSetTests(unittest.TestCase):
    def test_every_kind_is_there_with_its_answer(self) -> None:
        items = load_set()
        self.assertEqual(set(Counter(item["kind"] for item in items)), set(KINDS))
        for item in items:
            self.assertEqual(item["exam"], item["kind"] not in ("easy", "hard"), item["text"])
        self.assertEqual(len({item["text"] for item in items}), len(items))


class CheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_counts_per_kind_and_lists_what_went_wrong(self) -> None:
        items = [
            {"kind": "contest", "exam": True, "text": "出个题"},
            {"kind": "contest", "exam": True, "text": "算一下", "replied": "这题我会"},
            {"kind": "easy", "exam": False, "text": "吃了没"},
            {"kind": "hard", "exam": False, "text": "这道题我不会"},
            {"kind": "hard", "exam": False, "text": "电脑坏了"},
        ]
        report = await check_set(MarkerModel(), items)
        kinds = report["kinds"]
        self.assertEqual((kinds["contest"]["yes"], kinds["contest"]["n"]), (2, 2), "the reply counts too")
        self.assertEqual((kinds["hard"]["yes"], kinds["hard"]["errors"]), (1, 1))
        self.assertEqual([item["text"] for item in report["wrong"]], ["这道题我不会"])
        text = format_set(report)
        self.assertIn("caught 2/2, long puzzles 2/2; false alarms easy 0/1, hard 1/2", text)
        self.assertIn("wrong hard: 这道题我不会", text)

    async def test_real_messages_that_called_me(self) -> None:
        db = CorpusDB(":memory:")
        self.addCleanup(db.close)

        def row(message_id: int, text: str, me: bool = False, reply_to: int | None = None, **extra: Any) -> Row:
            canned = extra.pop("canned", False)
            user = BOT_ID if me else 7
            m = Msg(chat_id=1, message_id=message_id, ts=message_id, user_id=user, user_name="小林",
                    is_me=me, text=text, reply_to=reply_to, **extra)
            return Row(m, canned)

        db.replace_chat(1, "group", "测试群", "test", [
            row(1, "我先走了", me=True),
            row(2, "去哪", reply_to=1),
            row(3, "真红 出个题给你"),
            row(4, "今天好冷"),
            row(5, "是啊", reply_to=4),
            row(6, "真红 /start", canned=True),
            row(7, "真红", kind="sticker"),
        ])
        cases = addressed(db, ("真红", "shinnku"), 10)
        self.assertEqual(sorted(case[0].message_id for case in cases), [2, 3])
        by_id = {case[0].message_id: case for case in cases}
        self.assertTrue(by_id[2][1].is_me)
        self.assertEqual([m.message_id for m in by_id[3][2]], [1, 2])

        counts, flagged = await check_real(MarkerModel(), db, ("真红",), 10)
        self.assertEqual((counts["n"], counts["flagged"], counts["errors"]), (2, 1, 0))
        self.assertEqual(flagged, ["真红 出个题给你"])


if __name__ == "__main__":
    unittest.main()
