import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

os.environ.setdefault("TOKEN", "test-token")
os.environ.setdefault("NAME", "testbot")
os.environ.setdefault("ADMIN_TELEGRAM_USER_ID", "1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("api_id", "1")
os.environ.setdefault("api_hash", "test-hash")

from utils.message_store import build_reply_chain, record_message


@dataclass
class FakeUser:
    id: int
    is_bot: bool = False
    full_name: str = "User"
    username: str | None = None
    first_name: str = "User"


@dataclass
class FakeMessage:
    message_id: int
    text: str
    from_user: FakeUser
    reply_to_message: "FakeMessage | None" = None
    chat_id: int = 100
    date: datetime = datetime.now(timezone.utc)


class MessageStoreTests(unittest.TestCase):
    def test_builds_reply_chain_in_chronological_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/messages.sqlite3"
            first = FakeMessage(1, "hello", FakeUser(10, full_name="Alice"))
            second = FakeMessage(2, "hi", FakeUser(20, True, "shinnku"), first)
            third = FakeMessage(3, "continue", FakeUser(10, full_name="Alice"), second)
            for message in (first, second, third):
                record_message(message, db_path=db_path)

            chain = build_reply_chain(100, 3, db_path=db_path)

        self.assertEqual([item.message_id for item in chain], [1, 2, 3])
        self.assertEqual([item.text for item in chain], ["hello", "hi", "continue"])

    def test_limits_to_recent_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/messages.sqlite3"
            previous = None
            for message_id in range(1, 26):
                current = FakeMessage(
                    message_id, str(message_id), FakeUser(10), previous
                )
                record_message(current, db_path=db_path)
                previous = current

            chain = build_reply_chain(100, 25, limit=20, db_path=db_path)

        self.assertEqual([item.message_id for item in chain], list(range(6, 26)))

    def test_stops_at_messages_older_than_max_age(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/messages.sqlite3"
            old = FakeMessage(
                1,
                "old",
                FakeUser(10),
                date=datetime.now(timezone.utc) - timedelta(days=40),
            )
            middle = FakeMessage(2, "middle", FakeUser(10), old)
            current = FakeMessage(3, "current", FakeUser(10), middle)
            for message in (old, middle, current):
                record_message(message, db_path=db_path)

            chain = build_reply_chain(100, 3, max_age_days=30, db_path=db_path)

        self.assertEqual([item.message_id for item in chain], [2, 3])

    def test_stops_gracefully_when_parent_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/messages.sqlite3"
            missing_parent = FakeMessage(2, "missing", FakeUser(10))
            current = FakeMessage(3, "current", FakeUser(10), missing_parent)
            record_message(current, db_path=db_path)

            chain = build_reply_chain(100, 3, db_path=db_path)

        self.assertEqual([item.message_id for item in chain], [3])


if __name__ == "__main__":
    unittest.main()
