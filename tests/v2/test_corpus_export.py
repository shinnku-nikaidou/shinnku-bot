"""Reading exports and describing their structure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinnku_bot.corpus.export import ExportFormatError, ExportReader, find_exports
from shinnku_bot.corpus.probe import enum_value, format_report, probe, type_name
from shinnku_bot.corpus.safe import run_safely

from tests.v2.fixture_export import OWNER_ID, PEOPLE, message, random_messages, service, write_export


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


class ExportReaderTests(TempDirCase):
    def read(self, path: Path, chunk_size: int = 1 << 20) -> tuple[list[dict], ExportReader]:
        with ExportReader(path, chunk_size=chunk_size) as reader:
            return list(reader), reader

    def test_reads_every_message_and_the_header(self) -> None:
        messages = random_messages(300)
        path = write_export(self.tmp / "a" / "result.json", messages)
        got, reader = self.read(path)
        self.assertEqual(got, messages)
        self.assertEqual(reader.header, {"name": "测试群", "type": "private_supergroup", "id": 1001})
        self.assertFalse(reader.truncated)
        self.assertEqual(reader.count, 300)

    def test_tiny_chunks_give_the_same_result(self) -> None:
        messages = random_messages(120, seed=3)
        path = write_export(self.tmp / "result.json", messages)
        for chunk_size in (7, 64, 1000):
            with self.subTest(chunk_size=chunk_size):
                got, reader = self.read(path, chunk_size)
                self.assertEqual(got, messages)
                self.assertFalse(reader.truncated)

    def test_truncated_file_yields_the_complete_prefix(self) -> None:
        messages = random_messages(200, seed=1)
        for fraction in (0.2, 0.5, 0.93):
            with self.subTest(fraction=fraction):
                path = write_export(self.tmp / f"{fraction}.json", messages, truncate=fraction)
                got, reader = self.read(path, chunk_size=512)
                self.assertTrue(reader.truncated)
                self.assertGreater(len(got), 0)
                self.assertLess(len(got), 200)
                self.assertEqual(got, messages[: len(got)])

    def test_empty_chat(self) -> None:
        got, reader = self.read(write_export(self.tmp / "result.json", []))
        self.assertEqual(got, [])
        self.assertFalse(reader.truncated)

    def test_not_an_export(self) -> None:
        path = self.tmp / "result.json"
        path.write_text('{"chats": {"list": []}}', encoding="utf-8")
        with self.assertRaises(ExportFormatError):
            ExportReader(path)

    def test_find_exports(self) -> None:
        write_export(self.tmp / "b" / "result.json", [])
        write_export(self.tmp / "a" / "result.json", [])
        (self.tmp / "a" / "other.json").write_text("{}", encoding="utf-8")
        found = find_exports(self.tmp)
        self.assertEqual([p.parent.name for p in found], ["a", "b"])


class ProbeTests(TempDirCase):
    def test_counts(self) -> None:
        me, lin, shu = PEOPLE[0], PEOPLE[1], PEOPLE[2]
        messages = [
            message(1, 1_700_000_000, lin, "在吗"),
            message(2, 1_700_000_010, me, "在", reply_to_message_id=1),
            message(3, 1_700_000_020, me, "", file="x", media_type="sticker", sticker_emoji="😂"),
            message(4, 1_700_086_400, shu, ["看 ", {"type": "link", "text": "https://example.com"}]),
            message(5, 1_700_086_410, None, "转的", forwarded_from="某频道", edited="2023-11-16T00:00:00"),
            service(6, 1_700_086_420, lin, "pin_message", message_id=1),
        ]
        report = probe(write_export(self.tmp / "result.json", messages), OWNER_ID)

        self.assertEqual(report.messages, 6)
        self.assertEqual(dict(report.by_type), {"message": 5, "service": 1})
        self.assertEqual(report.owner_messages, 2)
        self.assertEqual(report.senders, 4)
        self.assertEqual(dict(report.text_shape), {"str": 3, "list": 1, "empty": 2})
        self.assertEqual(dict(report.media_types), {"sticker": 1})
        self.assertEqual(dict(report.actions), {"pin_message": 1})
        self.assertEqual(report.entity_types["link"], 1)
        self.assertEqual((report.replies, report.forwards, report.edited), (1, 1, 1))
        self.assertEqual((report.date_min, report.date_max), ("2023-11-14", "2023-11-15"))
        self.assertEqual((report.chat_type, report.chat_id), ("private_supergroup", 1001))
        self.assertFalse(report.truncated)

    def test_report_text_leaks_no_content(self) -> None:
        messages = random_messages(400, seed=5)
        text = format_report(probe(write_export(self.tmp / "x" / "result.json", messages), OWNER_ID))
        for forbidden in ("测试群", "小林", "阿树", "我自己", "打游戏", "example.com", "某频道", "😂"):
            self.assertNotIn(forbidden, text)
        self.assertIn("messages: 400", text)
        self.assertIn("sticker", text)

    def test_enum_value_hides_free_text(self) -> None:
        self.assertEqual(enum_value("pin_message"), "pin_message")
        self.assertEqual(enum_value("群公告：今晚八点"), "<other>")
        self.assertEqual(enum_value("Some Name"), "<other>")
        self.assertEqual(enum_value(None), "<null>")

    def test_type_name(self) -> None:
        self.assertEqual(type_name("x"), "str")
        self.assertEqual(type_name(["a", {"type": "link", "text": "b"}]), "list[dict{text,type}|str]")
        self.assertEqual(type_name([]), "list[]")
        self.assertEqual(type_name(None), "null")


class SafeRunTests(TempDirCase):
    def test_failure_prints_type_and_location_but_not_the_message(self) -> None:
        import contextlib
        import io

        def boom() -> int:
            raise ValueError("小林: 这是聊天内容")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run_safely(boom, self.tmp / "reports")
        self.assertEqual(code, 1)
        self.assertIn("failed: ValueError", out.getvalue())
        self.assertIn("in boom", out.getvalue())
        self.assertNotIn("聊天内容", out.getvalue())
        self.assertIn("聊天内容", (self.tmp / "reports" / "last_error.txt").read_text(encoding="utf-8"))

    def test_success_passes_the_code_through(self) -> None:
        self.assertEqual(run_safely(lambda: 0, self.tmp), 0)


if __name__ == "__main__":
    unittest.main()
