import os
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone

os.environ.setdefault("TOKEN", "test-token")
os.environ.setdefault("NAME", "testbot")
os.environ.setdefault("ADMIN_TELEGRAM_USER_ID", "1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("api_id", "1")
os.environ.setdefault("api_hash", "test-hash")

from commands.chat_turbo import should_reply_to_message


@dataclass
class FakeUser:
    id: int
    is_bot: bool = False
    full_name: str = "User"
    username: str | None = None
    first_name: str = "User"


@dataclass
class FakeEntity:
    type: str
    offset: int
    length: int
    user: FakeUser | None = None


@dataclass
class FakeMessage:
    message_id: int
    text: str
    from_user: FakeUser | None
    reply_to_message: "FakeMessage | None" = None
    entities: list[FakeEntity] | None = None
    chat_id: int = 100
    date: datetime = datetime.now(timezone.utc)


class ChatTurboTriggerTests(unittest.TestCase):
    def test_human_reply_to_bot_triggers(self):
        bot_message = FakeMessage(1, "bot", FakeUser(99, is_bot=True))
        message = FakeMessage(2, "human", FakeUser(10), reply_to_message=bot_message)

        self.assertTrue(should_reply_to_message(message, 99, "testbot"))

    def test_human_mention_triggers(self):
        message = FakeMessage(
            1,
            "@testbot hello",
            FakeUser(10),
            entities=[FakeEntity("mention", 0, 8)],
        )

        self.assertTrue(should_reply_to_message(message, 99, "testbot"))

    def test_plain_human_text_does_not_trigger(self):
        message = FakeMessage(1, "hello", FakeUser(10))

        self.assertFalse(should_reply_to_message(message, 99, "testbot"))

    def test_other_bot_reply_does_not_trigger(self):
        bot_message = FakeMessage(1, "bot", FakeUser(99, is_bot=True))
        message = FakeMessage(
            2,
            "other bot",
            FakeUser(88, is_bot=True),
            reply_to_message=bot_message,
        )

        self.assertFalse(should_reply_to_message(message, 99, "testbot"))

    def test_self_bot_message_does_not_trigger(self):
        message = FakeMessage(1, "self", FakeUser(99, is_bot=True))

        self.assertFalse(should_reply_to_message(message, 99, "testbot"))


if __name__ == "__main__":
    unittest.main()
