"""Looking at photos, stickers and GIFs. All pictures here are a few made-up bytes."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shinnku_bot.harness import look as look_module
from shinnku_bot.harness.engine import Harness
from shinnku_bot.harness.look import MAX_SEEN, look
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.transcript import render_body
from shinnku_bot.llm import ModelReply, ToolCall
from shinnku_bot.store import Store
from shinnku_bot.tg.convert import picture

from tests.v2.fakes import BOT_ID, FakeModel, FakeOutbox, make_settings, msg, no_sleep, say, tg_message

WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 "
JPEG = b"\xff\xd8\xff\xe0"


def seen(text: str) -> ModelReply:
    return ModelReply(tool_calls=(ToolCall("describe", {"text": text}),))


def file(uid: str, width: int = 512, height: int = 512, **extra: object) -> SimpleNamespace:
    return SimpleNamespace(file_id=f"id-{uid}", file_unique_id=uid, width=width, height=height, **extra)


def sticker(uid: str, animated: bool = False, video: bool = False, thumbnail: SimpleNamespace | None = None):
    return file(uid, is_animated=animated, is_video=video, thumbnail=thumbnail, emoji="😂")


class FakeFetch:
    def __init__(self, data: bytes = WEBP, error: Exception | None = None, delay: float = 0.0) -> None:
        self.data, self.error, self.delay = data, error, delay
        self.asked: list[str] = []

    async def __call__(self, file_id: str) -> bytes:
        self.asked.append(file_id)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.data


class PictureTests(unittest.TestCase):
    def test_photo_takes_the_largest_size_that_still_reads(self) -> None:
        sizes = (file("s", 90, 68), file("m", 320, 240), file("x", 800, 600), file("y", 1280, 960), file("w", 2560, 1920))
        self.assertEqual(picture(tg_message(text=None, photo=sizes)), ("id-y", "y"))
        self.assertEqual(picture(tg_message(text=None, photo=(file("w", 2560, 1920),))), ("id-w", "w"))

    def test_stickers(self) -> None:
        self.assertEqual(picture(tg_message(text=None, sticker=sticker("st"))), ("id-st", "st"))
        thumb = file("th", 128, 128)
        for moving in ({"animated": True}, {"video": True}):
            self.assertEqual(picture(tg_message(text=None, sticker=sticker("st", thumbnail=thumb, **moving))), ("id-th", "th"))
        self.assertIsNone(picture(tg_message(text=None, sticker=sticker("st", animated=True))))

    def test_gif_is_seen_through_its_thumbnail(self) -> None:
        gif = SimpleNamespace(thumbnail=file("gt", 320, 180))
        self.assertEqual(picture(tg_message(text=None, animation=gif, document=object())), ("id-gt", "gt"))

    def test_other_messages_have_no_picture(self) -> None:
        self.assertIsNone(picture(tg_message(text="在吗")))
        self.assertIsNone(picture(tg_message(text=None, voice=object())))


class RenderTests(unittest.TestCase):
    def test_what_a_picture_shows_goes_inside_its_brackets(self) -> None:
        self.assertEqual(render_body(msg(1, "看我家猫", kind="photo", seen="一只橘猫趴在键盘上")), "[图片：一只橘猫趴在键盘上] 看我家猫")
        self.assertEqual(render_body(msg(2, "😂", kind="sticker", seen="熊猫头捂嘴笑")), "[贴纸 😂：熊猫头捂嘴笑]")
        self.assertEqual(render_body(msg(3, "", kind="sticker", seen="熊猫头")), "[贴纸：熊猫头]")
        self.assertEqual(render_body(msg(4, "", kind="animation", seen="猫跳上桌")), "[动图：猫跳上桌]")


class LookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def add(self, message_id: int, uid: str, kind: str = "sticker", text: str = "😂"):
        self.store.add_picture(100, message_id, f"id-{uid}", uid)
        return msg(message_id, text, kind=kind)

    async def test_each_picture_is_looked_at_once(self) -> None:
        msgs = [self.add(1, "panda"), msg(2, "哈哈"), self.add(3, "panda"), self.add(4, "cat", kind="photo", text="")]
        model = FakeModel(seen("熊猫头捂嘴笑"), seen("一只橘猫"))
        fetch = FakeFetch()

        first = await look(msgs, self.store, model, fetch)
        self.assertEqual([m.seen for m in first], ["熊猫头捂嘴笑", "", "熊猫头捂嘴笑", "一只橘猫"])
        self.assertEqual(sorted(fetch.asked), ["id-cat", "id-panda"])

        # Seen before, even in another message: no download and no model call.
        again = await look(msgs + [self.add(5, "panda")], self.store, model, fetch)
        self.assertEqual(again[-1].seen, "熊猫头捂嘴笑")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(fetch.asked), 2)

    async def test_the_request_carries_the_image_without_thinking(self) -> None:
        model = FakeModel(seen("熊猫头"))
        await look([self.add(1, "panda")], self.store, model, FakeFetch(WEBP))

        (message,) = model.calls[0]
        text, image = message["content"]
        self.assertEqual(text["type"], "text")
        self.assertTrue(image["image_url"]["url"].startswith("data:image/webp;base64,"))
        self.assertEqual(model.options[0], {"thinking": False})

        model = FakeModel(seen("猫"))
        await look([self.add(2, "cat", kind="photo")], self.store, model, FakeFetch(JPEG))
        self.assertTrue(model.calls[0][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    async def test_what_it_shows_is_one_short_line(self) -> None:
        model = FakeModel(seen("第一行\n第二行  " + "啊" * 300))
        (got,) = await look([self.add(1, "panda")], self.store, model, FakeFetch())
        self.assertEqual(len(got.seen), MAX_SEEN)
        self.assertTrue(got.seen.startswith("第一行 第二行 啊"))

    async def test_a_picture_i_could_not_see_stays_a_placeholder_and_is_tried_again(self) -> None:
        msgs = [self.add(1, "panda")]
        model = FakeModel(seen("熊猫头"))
        (got,) = await look(msgs, self.store, model, FakeFetch(error=RuntimeError("no file")))
        self.assertEqual(got.seen, "")
        self.assertEqual(render_body(got), "[贴纸 😂]")

        (got,) = await look(msgs, self.store, model, FakeFetch())
        self.assertEqual(got.seen, "熊猫头")

    async def test_a_slow_picture_does_not_hold_the_reply(self) -> None:
        with mock.patch.object(look_module, "WAIT_S", 0.05):
            (got,) = await look([self.add(1, "panda")], self.store, FakeModel(seen("熊猫头")), FakeFetch(delay=1))
        self.assertEqual(got.seen, "")

    async def test_pictures_without_a_file_are_left_alone(self) -> None:
        model = FakeModel(seen("x"))
        (got,) = await look([msg(1, "", kind="photo")], self.store, model, FakeFetch())
        self.assertEqual(got.seen, "")
        self.assertEqual(model.calls, [])


class HarnessLookTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_reply_sees_what_the_picture_shows(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(":memory:")
        self.addCleanup(store.close)
        settings = make_settings(Path(tmp.name))
        model = FakeModel(seen("一只橘猫趴在键盘上"), say("好胖"))
        harness = Harness(
            settings,
            store,
            model,
            FakeOutbox(),
            Persona(settings.persona_dir),
            bot_id=BOT_ID,
            fetch=FakeFetch(JPEG),
            sleep=no_sleep,
            rng=lambda: 0.5,
        )
        store.add_picture(100, 1, "id-cat", "cat")

        outcome = await harness.answer(msg(1, "看我家猫", kind="photo"), "mention")

        self.assertEqual(outcome.messages, ("好胖",))
        self.assertTrue(model.calls[1][-1]["content"].endswith("小林: [图片：一只橘猫趴在键盘上] 看我家猫"))


if __name__ == "__main__":
    unittest.main()
