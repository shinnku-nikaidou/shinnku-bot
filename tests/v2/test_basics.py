"""Store, transcript, prompt, tool parsing, send-time checks, attention, conversion."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from shinnku_bot.config import load_settings
from shinnku_bot.harness.act import Say, Silent, parse_action
from shinnku_bot.harness.attention import addressed, blocked, mentions_bot
from shinnku_bot.harness.hooks import StyleLimits, clean, fit, violations_of
from shinnku_bot.harness.prompt import build_messages, scene_header
from shinnku_bot.harness.send import typing_delay
from shinnku_bot.harness.transcript import clean_name, render_turns
from shinnku_bot.llm import ModelReply, ToolCall
from shinnku_bot.store import ChatState, Store
from shinnku_bot.tg.commands import parse_level, parse_minutes
from shinnku_bot.tg.app import STALE_S, on_message
from shinnku_bot.tg.convert import to_msg

from tests.v2.fakes import (
    BOT_ID,
    BOT_USERNAME,
    make_settings,
    msg,
    tg_entity,
    tg_message,
    tg_user,
)


class ConfigTests(unittest.TestCase):
    ENV = {
        "TOKEN": "t",
        "ADMIN_TELEGRAM_USER_ID": "42",
        "LLM_BASE_URL": "http://x/v1",
        "LLM_API_KEY": "k",
        "LLM_MODEL": "m",
    }

    def test_defaults(self) -> None:
        settings = load_settings(self.ENV)
        self.assertEqual(settings.owner_id, 42)
        self.assertTrue(settings.llm_thinking)
        self.assertEqual(settings.name_triggers, ("真红", "shinnku"))
        self.assertEqual(settings.allowed_chat_ids, frozenset())
        self.assertEqual(settings.db_path, Path("data/live.sqlite3"))

    def test_overrides(self) -> None:
        env = {
            **self.ENV,
            "LLM_THINKING": "off",
            "LLM_MODEL_FALLBACK": " codex/x ",
            "NAME_TRIGGERS": "阿橘, 橘子",
            "ALLOWED_CHAT_IDS": "-100, 5",
            "DATA_DIR": "/tmp/x",
        }
        settings = load_settings(env)
        self.assertFalse(settings.llm_thinking)
        self.assertEqual(settings.llm_model_fallback, "codex/x")
        self.assertEqual(settings.name_triggers, ("阿橘", "橘子"))
        self.assertEqual(settings.allowed_chat_ids, frozenset({-100, 5}))
        self.assertEqual(settings.persona_dir, Path("/tmp/x/persona"))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def test_recent_is_oldest_first_and_limited(self) -> None:
        for i in range(1, 6):
            self.store.record(msg(i, f"m{i}"))
        recent = self.store.recent(100, limit=3, max_age_s=10**6, now=1_800_000_010)
        self.assertEqual([m.message_id for m in recent], [3, 4, 5])

    def test_recent_drops_old_messages(self) -> None:
        self.store.record(msg(1, "old", ts=1000))
        self.store.record(msg(2, "new", ts=5000))
        recent = self.store.recent(100, limit=10, max_age_s=100, now=5050)
        self.assertEqual([m.text for m in recent], ["new"])

    def test_record_is_an_upsert(self) -> None:
        self.store.record(msg(1, "first"))
        self.store.record(msg(1, "edited"))
        self.assertEqual(self.store.count(100), 1)
        self.assertEqual(self.store.get(100, 1).text, "edited")

    def test_quote_is_kept(self) -> None:
        self.store.record(msg(2, "去啊", reply_to=1, quote="爬山"))
        self.assertEqual(self.store.get(100, 2).quote, "爬山")

    def test_store_made_before_quotes_gains_the_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite3"
            old = sqlite3.connect(path)
            old.execute(
                "CREATE TABLE messages (chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,"
                " ts INTEGER NOT NULL, user_id INTEGER, user_name TEXT NOT NULL, is_me INTEGER NOT NULL,"
                " is_bot INTEGER NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, reply_to INTEGER,"
                " PRIMARY KEY (chat_id, message_id))"
            )
            old.execute("INSERT INTO messages VALUES (100, 1, 1, 7, '小林', 0, 0, 'text', '在吗', NULL)")
            old.commit()
            old.close()
            store = Store(path)
            self.addCleanup(store.close)
            self.assertEqual(store.get(100, 1).quote, "")
            store.record(msg(2, "去啊", reply_to=1, quote="在吗"))
            self.assertEqual(store.get(100, 2).quote, "在吗")

    def test_chat_state(self) -> None:
        self.assertEqual(self.store.chat_state(100), ChatState(0, 0))
        self.store.set_quiet_until(100, 123)
        self.store.set_talk_level(100, 2)
        self.assertEqual(self.store.chat_state(100), ChatState(2, 123))
        self.assertEqual(self.store.chat_state(200), ChatState(0, 0))


class TranscriptTests(unittest.TestCase):
    def test_roles_and_merging(self) -> None:
        turns = render_turns(
            [
                msg(1, "今晚打游戏吗"),
                msg(2, "我十点后", user_id=8, name="阿树"),
                msg(3, "打", user_id=BOT_ID, name="我", is_me=True),
                msg(4, "十点", user_id=BOT_ID, name="我", is_me=True),
                msg(5, "好"),
            ]
        )
        self.assertEqual(
            turns,
            [
                {"role": "user", "content": "小林: 今晚打游戏吗\n阿树: 我十点后"},
                {"role": "assistant", "content": "打\n十点"},
                {"role": "user", "content": "小林: 好"},
            ],
        )

    def test_reply_notes(self) -> None:
        mine = msg(1, "不去", user_id=BOT_ID, name="我", is_me=True)
        other = msg(2, "去哪", user_id=8, name="阿树")
        turns = render_turns(
            [mine, other, msg(3, "为啥", reply_to=1), msg(4, "爬山", reply_to=2), msg(5, "嗯", reply_to=77)]
        )
        self.assertEqual(
            turns[-1]["content"],
            "阿树: 去哪\n小林（回复我「不去」）: 为啥\n小林（回复 阿树「去哪」）: 爬山\n小林: 嗯",
        )

    def test_reply_target_outside_window_uses_lookup(self) -> None:
        old = msg(1, "x", user_id=BOT_ID, is_me=True)
        turns = render_turns([msg(9, "还在吗", reply_to=1)], lookup=lambda mid: old if mid == 1 else None)
        self.assertEqual(turns[0]["content"], "小林（回复我「x」）: 还在吗")

    def test_quote_shows_what_is_answered(self) -> None:
        shu = msg(1, "周六去不去爬山 还是在家打游戏", user_id=8, name="阿树")
        cases = [
            # Telegram's copy wins over the stored message: it can be the part the sender picked.
            (msg(2, "去啊", reply_to=1, quote="爬山"), "小林（回复 阿树「爬山」）: 去啊"),
            (msg(2, "去啊", reply_to=99, quote="周六去不去爬山"), "小林（回复「周六去不去爬山」）: 去啊"),
            (msg(2, "笑死", reply_to=99, quote="啊" * 41), f"小林（回复「{'啊' * 40}…」）: 笑死"),
            (msg(2, "可爱", reply_to=99, quote="[贴纸 😂]"), "小林（回复「[贴纸 😂]」）: 可爱"),
        ]
        for line, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(render_turns([shu, line])[-1]["content"].split("\n")[-1], expected)

    def test_media_placeholders(self) -> None:
        turns = render_turns(
            [
                msg(1, "", kind="photo"),
                msg(2, "看这个", kind="photo"),
                msg(3, "😂", kind="sticker"),
                msg(4, "", kind="sticker"),
                msg(5, "", kind="voice"),
            ]
        )
        self.assertEqual(
            turns[0]["content"],
            "小林: [图片]\n小林: [图片] 看这个\n小林: [贴纸 😂]\n小林: [贴纸]\n小林: [语音]",
        )

    def test_bodies_are_flattened_and_capped(self) -> None:
        turns = render_turns([msg(1, "第一行\n第二行"), msg(2, "啊" * 600)])
        first, second = turns[0]["content"].split("\n")
        self.assertEqual(first, "小林: 第一行 第二行")
        self.assertEqual(len(second), len("小林: ") + 500 + 1)

    def test_clean_name(self) -> None:
        self.assertEqual(clean_name("a: b\nc"), "a b c")
        self.assertEqual(clean_name("   "), "某人")
        self.assertEqual(len(clean_name("x" * 40)), 16)


class PromptTests(unittest.TestCase):
    def test_header(self) -> None:
        when = datetime(2026, 11, 11, 21, 14)  # a Wednesday
        self.assertEqual(scene_header("某群", False, "x_bot", when), "（群「某群」，周三 21:14，我在这儿的账号是 @x_bot）")
        self.assertEqual(scene_header(None, True, None, when), "（私聊，周三 21:14）")

    def test_build(self) -> None:
        turns = [{"role": "user", "content": "小林: 在吗"}]
        messages = build_messages("我是我\n", "（私聊）", turns, notes=["（再想想）"])
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "我是我"},
                {"role": "user", "content": "（私聊）\n小林: 在吗"},
                {"role": "system", "content": "（再想想）"},
            ],
        )
        self.assertEqual(turns[0]["content"], "小林: 在吗", "input turns must not be mutated")

    def test_header_gets_its_own_turn_when_i_spoke_first(self) -> None:
        turns = [{"role": "assistant", "content": "早"}, {"role": "user", "content": "小林: 早"}]
        messages = build_messages("我", "（群聊）", turns)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_nothing_to_answer(self) -> None:
        self.assertIsNone(build_messages("我", "h", []))
        self.assertIsNone(build_messages("我", "h", [{"role": "assistant", "content": "嗯"}]))


class ActTests(unittest.TestCase):
    def test_say(self) -> None:
        reply = ModelReply(tool_calls=(ToolCall("say", {"messages": ["打", " ", "十点"]}),))
        self.assertEqual(parse_action(reply), Say(((None, "打"), (None, "十点"))))

    def test_say_names_the_question_it_answers(self) -> None:
        items = [{"text": "八点", "to": 2}, {"text": "都来"}, {"text": "行", "to": "1"}, {"text": "嗯", "to": True}]
        reply = ModelReply(tool_calls=(ToolCall("say", {"messages": items}),))
        self.assertEqual(parse_action(reply), Say(((2, "八点"), (None, "都来"), (None, "行"), (None, "嗯"))))

    def test_say_accepts_a_bare_string(self) -> None:
        reply = ModelReply(tool_calls=(ToolCall("say", {"messages": "打"}),))
        self.assertEqual(parse_action(reply), Say(((None, "打"),)))

    def test_silent(self) -> None:
        self.assertEqual(parse_action(ModelReply(tool_calls=(ToolCall("stay_silent", {}),))), Silent())

    def test_say_wins_over_silent(self) -> None:
        reply = ModelReply(
            tool_calls=(ToolCall("stay_silent", {}), ToolCall("say", {"messages": ["嗯"]}))
        )
        self.assertEqual(parse_action(reply), Say(((None, "嗯"),)))

    def test_malformed_say_is_not_speech(self) -> None:
        self.assertIsNone(parse_action(ModelReply(tool_calls=(ToolCall("say", {}),))))
        self.assertIsNone(parse_action(ModelReply(tool_calls=(ToolCall("say", {"messages": [1]}),))))
        self.assertIsNone(parse_action(ModelReply(tool_calls=(ToolCall("say", {"messages": [{"to": 1}]}),))))

    def test_plain_content_fallback(self) -> None:
        self.assertEqual(parse_action(ModelReply(content="打\n十点")), Say(((None, "打"), (None, "十点"))))
        self.assertIsNone(parse_action(ModelReply()))


class HookTests(unittest.TestCase):
    def test_clean_markup(self) -> None:
        self.assertEqual(clean(["**不**去。", "- 太累了", "`ok`", "1. 行"]), ("不去", "太累了", "ok", "行"))

    def test_clean_splits_lines_and_drops_empties(self) -> None:
        self.assertEqual(clean(["打\n\n十点 ", "  "]), ("打", "十点"))

    def test_clean_strips_speaker_prefix_and_wrapping_quotes(self) -> None:
        self.assertEqual(clean(["我: 不去", "真红：行", "“好吧”", "他说“好”然后走了"]), ("不去", "行", "好吧", "他说“好”然后走了"))

    def test_period_handling(self) -> None:
        self.assertEqual(clean(["行。", "。。。", "1.5倍"]), ("行", "。。。", "1.5倍"))
        self.assertEqual(clean(["行。"], StyleLimits(strip_period=False)), ("行。",))

    def test_ok(self) -> None:
        self.assertEqual(violations_of(clean(["打", "十点"])), ())

    def test_ai_tone(self) -> None:
        for text in (
            "作为一个AI助手，我没有身体",
            "希望这些建议对你有帮助",
            "很高兴为您解答",
            "有什么可以帮你的吗",
            "首先，要冷静；其次，要休息",
            "以下是几点建议",
        ):
            with self.subTest(text=text):
                self.assertIn("ai_tone", violations_of([text]))

    def test_ordinary_chat_is_not_ai_tone(self) -> None:
        for text in ("你才是bot", "不知道 懒得查", "我不能喝酒", "帮我带一份", "首先我没钱"):
            with self.subTest(text=text):
                self.assertEqual(violations_of([text]), ())

    def test_too_long(self) -> None:
        self.assertEqual(violations_of(["啊" * 81, "a"], StyleLimits(max_len=80)), ("too_long",))

    def test_fit_merges_the_overflow_instead_of_dropping_it(self) -> None:
        self.assertEqual(fit(["a", "b"], 3), ("a", "b"))
        self.assertEqual(fit(["a", "b", "c", "d", "e"], 3), ("a", "b", "c\nd\ne"))


class AttentionTests(unittest.TestCase):
    NAMES = ("真红", "shinnku")

    def why(self, message: SimpleNamespace) -> str | None:
        return addressed(message, BOT_ID, BOT_USERNAME, self.NAMES)

    def test_plain_group_message_is_not_addressed(self) -> None:
        self.assertIsNone(self.why(tg_message(text="今晚吃啥")))

    def test_private(self) -> None:
        self.assertEqual(self.why(tg_message(chat_type="private")), "private")

    def test_reply_to_me(self) -> None:
        mine = tg_message(message_id=1, user=tg_user(BOT_ID, is_bot=True))
        self.assertEqual(self.why(tg_message(message_id=2, reply_to=mine)), "reply")
        theirs = tg_message(message_id=1, user=tg_user(8))
        self.assertIsNone(self.why(tg_message(message_id=2, reply_to=theirs)))

    def test_mention(self) -> None:
        text = f"@{BOT_USERNAME.upper()} 你呢"
        entity = tg_entity("mention", 0, len(BOT_USERNAME) + 1)
        self.assertEqual(self.why(tg_message(text=text, entities=[entity])), "mention")
        other = tg_entity("mention", 0, 6)
        self.assertIsNone(self.why(tg_message(text="@other 你呢", entities=[other])))

    def test_mention_offsets_are_utf16(self) -> None:
        # Each of these emoji is two UTF-16 units but one Python character.
        text = f"😂😂 @{BOT_USERNAME}"
        entity = tg_entity("mention", 5, len(BOT_USERNAME) + 1)
        self.assertTrue(mentions_bot(tg_message(text=text, entities=[entity]), BOT_ID, BOT_USERNAME))

    def test_text_mention(self) -> None:
        entity = tg_entity("text_mention", 0, 2, user=tg_user(BOT_ID, is_bot=True))
        self.assertEqual(self.why(tg_message(text="某人 来", entities=[entity])), "mention")

    def test_mention_in_caption(self) -> None:
        entity = tg_entity("mention", 0, len(BOT_USERNAME) + 1)
        message = tg_message(text=None, caption=f"@{BOT_USERNAME} 看", caption_entities=[entity], photo=[object()])
        self.assertEqual(self.why(message), "mention")

    def test_name(self) -> None:
        self.assertEqual(self.why(tg_message(text="真红人呢")), "name")
        self.assertEqual(self.why(tg_message(text="ask SHINNKU")), "name")

    def test_bots_and_anonymous_senders_are_ignored(self) -> None:
        self.assertIsNone(self.why(tg_message(text="真红", user=tg_user(5, is_bot=True))))
        message = tg_message(text="真红")
        message.from_user = None
        self.assertIsNone(self.why(message))

    def test_blocked(self) -> None:
        open_settings = make_settings(Path("."))
        self.assertIsNone(blocked(open_settings, ChatState(), 100, False, now=50))
        self.assertEqual(blocked(open_settings, ChatState(quiet_until=60), 100, False, now=50), "quiet")
        self.assertIsNone(blocked(open_settings, ChatState(quiet_until=60), 100, False, now=61))

        listed = make_settings(Path("."), allowed_chat_ids=frozenset({100}))
        self.assertIsNone(blocked(listed, ChatState(), 100, False, now=0))
        self.assertEqual(blocked(listed, ChatState(), 200, False, now=0), "chat_not_allowed")
        self.assertIsNone(blocked(listed, ChatState(), 200, True, now=0), "the list is for groups")

        no_dm = make_settings(Path("."), allow_private=False)
        self.assertEqual(blocked(no_dm, ChatState(), 5, True, now=0), "private_off")


class OnMessageTests(unittest.IsolatedAsyncioTestCase):
    async def deliver(self, age_s: float) -> tuple[list[tuple[int, str | None]], int]:
        """Hand the bot a message sent `age_s` ago. Returns (what the harness was given, stored)."""
        handled: list[tuple[int, str | None]] = []
        store = Store(":memory:")
        self.addCleanup(store.close)

        def handle(msg: object, reason: str | None, **_: object) -> None:
            handled.append((msg.message_id, reason))

        context = SimpleNamespace(
            bot_data={
                "me": tg_user(BOT_ID, is_bot=True, username=BOT_USERNAME),
                "settings": make_settings(Path(".")),
                "harness": SimpleNamespace(handle=handle, store=store),
            }
        )
        sent = datetime.now(timezone.utc) - timedelta(seconds=age_s)
        await on_message(SimpleNamespace(message=tg_message(text="真红在吗", date=sent)), context)
        return handled, store.count(100)

    async def test_fresh_messages_are_answered(self) -> None:
        self.assertEqual(await self.deliver(5), ([(1, "name")], 0))

    async def test_what_piled_up_long_ago_is_only_stored(self) -> None:
        self.assertEqual(await self.deliver(STALE_S + 60), ([], 1))

    async def test_no_message_no_work(self) -> None:
        await on_message(SimpleNamespace(message=None), SimpleNamespace(bot_data={}))


class ConvertTests(unittest.TestCase):
    def test_text(self) -> None:
        mine = tg_message(message_id=3, user=tg_user(BOT_ID, is_bot=True))
        converted = to_msg(tg_message(message_id=4, text="在吗", reply_to=mine), BOT_ID)
        self.assertEqual((converted.kind, converted.text, converted.reply_to), ("text", "在吗", 3))
        self.assertEqual((converted.user_id, converted.user_name, converted.is_me), (7, "小林", False))
        self.assertEqual(converted.ts, 1794398400)

    def test_own_message(self) -> None:
        converted = to_msg(tg_message(user=tg_user(BOT_ID, is_bot=True)), BOT_ID)
        self.assertTrue(converted.is_me)
        self.assertTrue(converted.is_bot)

    def test_media(self) -> None:
        cases = {
            "sticker": (dict(sticker=SimpleNamespace(emoji="😂")), "😂"),
            "photo": (dict(photo=[object()], caption="看"), "看"),
            "voice": (dict(voice=object()), ""),
            "video": (dict(video_note=object()), ""),
            "animation": (dict(animation=object(), document=object()), ""),
            "document": (dict(document=object()), ""),
            "other": ({}, ""),
        }
        for kind, (extra, text) in cases.items():
            with self.subTest(kind=kind):
                converted = to_msg(tg_message(text=None, **extra), BOT_ID)
                self.assertEqual((converted.kind, converted.text), (kind, text))

    def test_reply_keeps_what_it_quotes(self) -> None:
        shu = tg_user(8, full_name="阿树")
        said = tg_message(message_id=3, text="周六去不去爬山", user=shu)
        cases = [
            (dict(reply_to=said), (3, "周六去不去爬山")),
            (dict(reply_to=said, quote=SimpleNamespace(text="爬山")), (3, "爬山")),
            (dict(reply_to=tg_message(message_id=3, text=None, sticker=SimpleNamespace(emoji="😂"))), (3, "[贴纸 😂]")),
            (dict(reply_to=tg_message(message_id=3, text=None, photo=[object()], caption="看")), (3, "[图片] 看")),
            # A reply to a message in another chat comes with the quoted part only.
            (dict(external_reply=SimpleNamespace(), quote=SimpleNamespace(text="别处的话")), (None, "别处的话")),
            # Every message in a forum topic "replies" to the topic's opening service message.
            (dict(reply_to=tg_message(message_id=3, text=None, forum_topic_created=object())), (3, "")),
            ({}, (None, "")),
        ]
        for extra, expected in cases:
            with self.subTest(expected=expected):
                converted = to_msg(tg_message(message_id=4, text="去啊", **extra), BOT_ID)
                self.assertEqual((converted.reply_to, converted.quote), expected)


class SmallThingsTests(unittest.TestCase):
    def test_typing_delay_bounds(self) -> None:
        self.assertAlmostEqual(typing_delay("嗯", 0.5), 0.62)
        self.assertAlmostEqual(typing_delay("啊" * 200, 0.0), 3.2)
        self.assertAlmostEqual(typing_delay("啊" * 200, 0.999), 4.0 * 1.1996, places=3)

    def test_command_arguments(self) -> None:
        self.assertEqual(parse_minutes([]), 30)
        self.assertEqual(parse_minutes(["abc"]), 30)
        self.assertEqual(parse_minutes(["0"]), 1)
        self.assertEqual(parse_minutes(["99999"]), 720)
        self.assertEqual(parse_level(["2"]), 2)
        self.assertIsNone(parse_level(["9"]))
        self.assertIsNone(parse_level([]))


if __name__ == "__main__":
    unittest.main()
