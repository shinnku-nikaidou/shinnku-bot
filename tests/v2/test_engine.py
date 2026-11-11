"""The harness end to end, with a scripted model and a recording outbox."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from shinnku_bot.harness.engine import Harness
from shinnku_bot.harness.hooks import RETRY_NOTES
from shinnku_bot.harness.persona import Persona
from shinnku_bot.llm import GatewayModel, ModelReply
from shinnku_bot.store import Store

from tests.v2.fakes import BOT_ID, FakeModel, FakeOutbox, make_settings, msg, no_sleep, say, silent

AI_TONE = "作为一个AI助手，我无法提供这方面的帮助"


class HarnessCase(unittest.IsolatedAsyncioTestCase):
    def build(self, model: FakeModel, **overrides: object) -> Harness:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.outbox = FakeOutbox()
        self.model = model
        settings = make_settings(self.tmp, **overrides)
        return Harness(
            settings,
            self.store,
            model,
            self.outbox,
            Persona(settings.persona_dir),
            bot_id=BOT_ID,
            bot_username="akitsu_bot",
            sleep=no_sleep,
            rng=lambda: 0.5,
        )

    async def settle(self, harness: Harness, timeout: float = 2.0) -> None:
        """Wait until no chat has a run in flight."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
            if all(chat.task is None for chat in harness._chats.values()):
                return
        self.fail("harness did not settle")


class AnswerTests(HarnessCase):
    async def test_says_a_burst_and_remembers_it(self) -> None:
        harness = self.build(FakeModel(say("打", "十点")))
        outcome = await harness.answer(msg(1, "今晚打游戏吗"), "mention", title="测试群")

        self.assertEqual(outcome.kind, "said")
        self.assertEqual(self.outbox.texts, ["打", "十点"])
        self.assertEqual(self.outbox.typing_calls, 2)
        stored = self.store.recent(100, 10, 10**9)
        self.assertEqual([(m.is_me, m.text) for m in stored], [(False, "今晚打游戏吗"), (True, "打"), (True, "十点")])

    async def test_prompt_shape(self) -> None:
        harness = self.build(FakeModel(say("嗯")))
        await harness.answer(msg(1, "在吗"), "mention", title="测试群")

        prompt = self.model.calls[0]
        self.assertEqual(prompt[0]["role"], "system")
        self.assertIn("shinnku", prompt[0]["content"])
        self.assertEqual(prompt[-1]["role"], "user")
        header, line = prompt[-1]["content"].split("\n")
        self.assertRegex(header, r"^（群「测试群」，周. \d\d:\d\d，我在这儿的账号是 @akitsu_bot）$")
        self.assertEqual(line, "小林: 在吗")

    async def test_my_earlier_lines_come_back_as_assistant_turns(self) -> None:
        harness = self.build(FakeModel(say("打"), say("十点")))
        await harness.answer(msg(1, "打游戏吗"), "mention")
        # Ids grow within a chat; my reply above took an id from the outbox (1000+).
        await harness.answer(msg(2000, "几点"), "reply")

        roles = [m["role"] for m in self.model.calls[1]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(self.model.calls[1][2]["content"], "打")

    async def test_silence_sends_nothing(self) -> None:
        harness = self.build(FakeModel(silent()))
        outcome = await harness.answer(msg(1, "有人吗"), "name")
        self.assertEqual(outcome.kind, "silent")
        self.assertEqual(self.outbox.sent, [])

    async def test_an_empty_answer_is_asked_again_once(self) -> None:
        harness = self.build(FakeModel(ModelReply(), say("在")))
        outcome = await harness.answer(msg(1, "有人吗"), "name")
        self.assertEqual((outcome.kind, outcome.attempts, self.outbox.texts), ("said", 2, ["在"]))
        self.assertEqual(self.model.calls[0], self.model.calls[1])  # the very same prompt
        self.assertEqual(self.model.options, [{}, {"thinking": False}])  # ... without thinking

        harness = self.build(FakeModel(ModelReply()))
        outcome = await harness.answer(msg(1, "有人吗"), "name")
        self.assertEqual((outcome.kind, outcome.attempts), ("silent", 2))

    async def test_off_draft_gets_one_second_try(self) -> None:
        harness = self.build(FakeModel(say(AI_TONE), say("不知道")))
        outcome = await harness.answer(msg(1, "这个怎么弄"), "mention")

        self.assertEqual(self.outbox.texts, ["不知道"])
        self.assertEqual(outcome.attempts, 2)
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(self.model.calls[1][-1], {"role": "system", "content": RETRY_NOTES["ai_tone"]})
        self.assertNotIn(RETRY_NOTES["ai_tone"], json.dumps(self.model.calls[0], ensure_ascii=False))

    async def test_help_desk_voice_twice_means_silence(self) -> None:
        harness = self.build(FakeModel(say(AI_TONE)))
        outcome = await harness.answer(msg(1, "这个怎么弄"), "mention")
        self.assertEqual(outcome.kind, "silent")
        self.assertEqual(outcome.violations, ("ai_tone",))
        self.assertEqual(self.outbox.sent, [])
        self.assertEqual(len(self.model.calls), 2)

    async def test_long_twice_is_sent_anyway(self) -> None:
        long_text = "啊" * 100
        harness = self.build(FakeModel(say(long_text)))
        outcome = await harness.answer(msg(1, "讲讲"), "mention")
        self.assertEqual(outcome.kind, "said")
        self.assertEqual(self.outbox.texts, [long_text])

    async def test_trace_is_written_only_when_enabled(self) -> None:
        harness = self.build(FakeModel(say("嗯")), trace=True)
        await harness.answer(msg(1, "在吗"), "mention")
        files = list((self.tmp / "trace").glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        record = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["said"], ["嗯"])
        self.assertEqual(record["why"], "mention")


class DebounceTests(HarnessCase):
    async def test_unaddressed_messages_are_stored_but_do_not_wake_me(self) -> None:
        harness = self.build(FakeModel(say("嗯")))
        harness.handle(msg(1, "今晚吃啥"), None, title="测试群")
        await asyncio.sleep(0.1)
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.store.count(100), 1)

    async def test_rapid_lines_get_one_answer(self) -> None:
        harness = self.build(FakeModel(say("来")))
        harness.handle(msg(1, "真红"), "name", title="测试群")
        await asyncio.sleep(0.01)
        harness.handle(msg(2, "在吗"), None, title="测试群")
        await asyncio.sleep(0.01)
        harness.handle(msg(3, "打游戏"), None, title="测试群")
        await self.settle(harness)

        self.assertEqual(len(self.model.calls), 1)
        self.assertTrue(self.model.calls[0][-1]["content"].endswith("小林: 真红\n小林: 在吗\n小林: 打游戏"))
        self.assertEqual(self.outbox.texts, ["来"])

    async def test_waiting_is_capped(self) -> None:
        harness = self.build(FakeModel(say("来")), quiet_window_s=0.05, max_wait_s=0.12)
        started = asyncio.get_running_loop().time()
        harness.handle(msg(1, "真红"), "name")
        for i in range(2, 30):
            await asyncio.sleep(0.02)
            if self.model.calls:
                break
            harness.handle(msg(i, "还有"), None)
        await self.settle(harness)
        self.assertLess(asyncio.get_running_loop().time() - started, 0.6)
        self.assertGreaterEqual(len(self.model.calls), 1)

    async def test_new_line_while_thinking_restarts_once(self) -> None:
        harness = self.build(FakeModel(say("好"), delay=0.15))
        harness.handle(msg(1, "真红 出来"), "name")
        await asyncio.sleep(0.08)  # past the quiet window, model call in flight
        harness.handle(msg(2, "算了不用了"), None)
        await self.settle(harness)

        self.assertEqual(len(self.model.calls), 2, "the first call is abandoned")
        self.assertEqual(self.model.finished, 1)
        self.assertEqual(self.outbox.texts, ["好"])
        self.assertIn("算了不用了", self.model.calls[1][-1]["content"])

    async def test_someone_else_talking_does_not_restart(self) -> None:
        harness = self.build(FakeModel(say("好"), delay=0.1))
        harness.handle(msg(1, "真红 出来"), "name")
        await asyncio.sleep(0.06)
        harness.handle(msg(2, "哈哈", user_id=8, name="阿树"), None)
        await self.settle(harness)
        self.assertEqual(len(self.model.calls), 1)

    async def test_draining_lets_an_answer_under_way_finish(self) -> None:
        harness = self.build(FakeModel(say("算了下", "一共三百"), delay=0.1))
        await harness.drain()  # nothing under way: returns at once
        harness.handle(msg(1, "真红 帮我算一下"), "name")
        await harness.drain()
        self.assertEqual(self.outbox.texts, ["算了下", "一共三百"])

    async def test_a_late_answer_quotes_what_it_answers(self) -> None:
        harness = self.build(FakeModel(say("来", "等下"), delay=0.15))
        harness.handle(msg(1, "真红 打游戏吗"), "name")
        await asyncio.sleep(0.08)  # past the quiet window: I am thinking now
        for i in range(2, 7):  # five lines from someone else arrive meanwhile
            harness.handle(msg(i, "刷屏", user_id=8, name="阿树"), None)
        await self.settle(harness)
        self.assertEqual(self.outbox.sent, [(100, "来", 1), (100, "等下", None)])

    async def test_a_prompt_answer_does_not(self) -> None:
        harness = self.build(FakeModel(say("来"), delay=0.15))
        harness.handle(msg(1, "真红 打游戏吗"), "name")
        await asyncio.sleep(0.08)
        for i in range(2, 6):  # four newer lines are not yet too many
            harness.handle(msg(i, "刷屏", user_id=8, name="阿树"), None)
        await self.settle(harness)
        self.assertEqual(self.outbox.sent, [(100, "来", None)])

    async def test_private_chats_never_quote(self) -> None:
        harness = self.build(FakeModel(say("来")))
        for i in range(1, 8):
            self.store.record(msg(i, "在吗", chat_id=7))
        await harness.answer(msg(1, "在吗", chat_id=7), "private", is_private=True)
        self.assertEqual(self.outbox.sent, [(7, "来", None)])

    async def test_quiet_time_blocks(self) -> None:
        harness = self.build(FakeModel(say("嗯")))
        self.store.set_quiet_until(100, 2_000_000_000_000)
        harness.handle(msg(1, "真红"), "name")
        await asyncio.sleep(0.1)
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.store.count(100), 1, "still recorded")

    async def test_model_failure_does_not_wedge_the_chat(self) -> None:
        class Flaky(FakeModel):
            async def complete(self, messages, tools):  # type: ignore[override]
                if not self.calls:
                    self.calls.append(messages)
                    raise RuntimeError("gateway down")
                return await super().complete(messages, tools)

        harness = self.build(Flaky(say("在")))
        with self.assertLogs("shinnku_bot.harness.engine", level="ERROR") as logs:
            harness.handle(msg(1, "真红"), "name")
            await self.settle(harness)
        self.assertIn("RuntimeError", logs.output[0])
        self.assertNotIn("gateway down", logs.output[0], "exception text is not logged")

        harness.handle(msg(2, "真红？"), "name")
        await self.settle(harness)
        self.assertEqual(self.outbox.texts, ["在"])


class GatewayModelTests(unittest.IsolatedAsyncioTestCase):
    RESPONSE = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "（被点名了 回一句）",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "say", "arguments": '{"messages": ["打"]}'},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "say", "arguments": "not json"},
                        },
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 391,
            "completion_tokens": 28,
            "total_tokens": 419,
            "prompt_tokens_details": {"cached_tokens": 300},
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }

    def model(self, thinking: bool) -> GatewayModel:
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json=self.RESPONSE)

        client = AsyncOpenAI(
            base_url="http://gateway.test/v1",
            api_key="k",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        return GatewayModel("http://gateway.test/v1", "k", "m", thinking=thinking, client=client)

    async def test_parses_tool_calls_reasoning_and_usage(self) -> None:
        reply = await self.model(thinking=True).complete([{"role": "user", "content": "x"}], [])
        self.assertEqual([(c.name, c.args) for c in reply.tool_calls], [("say", {"messages": ["打"]}), ("say", {})])
        self.assertEqual(reply.reasoning, "（被点名了 回一句）")
        self.assertEqual(
            (reply.prompt_tokens, reply.cached_tokens, reply.completion_tokens, reply.reasoning_tokens),
            (391, 300, 28, 12),
        )

    async def test_request_with_thinking(self) -> None:
        await self.model(thinking=True).complete([{"role": "user", "content": "x"}], [{"type": "function"}])
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["tool_choice"], "required")
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertNotIn("thinking", body)
        self.assertEqual(str(self.requests[0].url), "http://gateway.test/v1/chat/completions")

    async def test_request_without_thinking(self) -> None:
        await self.model(thinking=False).complete([{"role": "user", "content": "x"}], [])
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", body)

    async def test_thinking_can_be_turned_off_for_one_call(self) -> None:
        model = self.model(thinking=True)
        await model.complete([{"role": "user", "content": "x"}], [], thinking=False)
        await model.complete([{"role": "user", "content": "x"}], [])
        first, second = (json.loads(request.content) for request in self.requests)
        self.assertEqual(first["thinking"], {"type": "disabled"})
        self.assertIn("reasoning_effort", second)


if __name__ == "__main__":
    unittest.main()
