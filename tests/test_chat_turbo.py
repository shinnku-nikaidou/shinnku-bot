import os
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone

from telegram.constants import ParseMode
from telegram.error import BadRequest

os.environ.setdefault("TOKEN", "test-token")
os.environ.setdefault("NAME", "testbot")
os.environ.setdefault("ADMIN_TELEGRAM_USER_ID", "1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("api_id", "1")
os.environ.setdefault("api_hash", "test-hash")

from commands.chat_turbo import _reply_markdown_safe, should_reply_to_message


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


class FakeReplyMessage:
    def __init__(self, error: BadRequest | None = None):
        self.error = error
        self.calls = []
        self.sent_message = object()

    async def reply_text(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        if self.error is not None and len(self.calls) == 1:
            raise self.error
        return self.sent_message


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


class ChatTurboMarkdownReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_markdown_reply_succeeds_without_fallback(self):
        message = FakeReplyMessage()

        sent_message = await _reply_markdown_safe(message, "**hello**")

        self.assertIs(sent_message, message.sent_message)
        self.assertEqual(
            message.calls,
            [("*hello*", {"parse_mode": ParseMode.MARKDOWN_V2})],
        )

    async def test_markdown_parse_error_falls_back_to_plain_text(self):
        message = FakeReplyMessage(BadRequest("Can't parse entities: bad markdown"))

        sent_message = await _reply_markdown_safe(message, "**hello")

        self.assertIs(sent_message, message.sent_message)
        self.assertEqual(
            message.calls,
            [
                ("\\*\\*hello", {"parse_mode": ParseMode.MARKDOWN_V2}),
                ("**hello", {}),
            ],
        )

    async def test_other_bad_request_is_re_raised(self):
        message = FakeReplyMessage(BadRequest("Message is too long"))

        with self.assertRaises(BadRequest):
            await _reply_markdown_safe(message, "hello")

        self.assertEqual(
            message.calls,
            [("hello", {"parse_mode": ParseMode.MARKDOWN_V2})],
        )


if __name__ == "__main__":
    unittest.main()
