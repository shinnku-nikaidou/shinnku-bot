"""Looking things up on the web before answering. All text here is made up."""

from __future__ import annotations

import unittest
from typing import Any

import httpx

from shinnku_bot.config import load_settings
from shinnku_bot.harness.hooks import violations_of
from shinnku_bot.harness.lookup import MAX_CONTENT, Hit, SearXNG, decide, look_up
from shinnku_bot.harness.prompt import KNOWN_INTRO, build_messages, render_known
from shinnku_bot.llm import ModelReply, ToolCall

from tests.v2.fakes import FakeModel, FakeSearch, msg


def wants(query: Any) -> ModelReply:
    return ModelReply(tool_calls=(ToolCall("decide", {"query": query}),))


class SearXNGTests(unittest.IsolatedAsyncioTestCase):
    def searxng(self, status: int = 200, results: list[dict[str, Any]] = ()) -> SearXNG:
        def handle(request: httpx.Request) -> httpx.Response:
            self.request = request
            return httpx.Response(status, json={"results": list(results)})

        return SearXNG("http://searx.test/", httpx.AsyncClient(transport=httpx.MockTransport(handle)))

    async def test_the_first_few_results_that_say_something(self) -> None:
        search = self.searxng(results=[
            {"title": "钟馗 发售日", "content": "定在\n八月", "publishedDate": "2026-09-10T00:00:00"},
            {"title": "只有标题", "content": ""},
            {"title": "很长", "content": "长" * 300, "publishedDate": None},
            {"title": "第三条", "content": "有"},
            {"title": "第四条", "content": "多了"},
        ])
        hits = await search("钟馗 发售")
        self.assertEqual(self.request.url.path, "/search")
        self.assertEqual(dict(self.request.url.params), {"q": "钟馗 发售", "format": "json"})
        self.assertEqual(hits, [
            Hit("钟馗 发售日", "定在 八月", "2026-09-10"),
            Hit("很长", "长" * MAX_CONTENT + "…"),
            Hit("第三条", "有"),
        ])

    async def test_an_error_page_raises(self) -> None:
        with self.assertRaises(httpx.HTTPStatusError):
            await self.searxng(status=500)("x")


class DecideTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_call_without_thinking(self) -> None:
        model = FakeModel(wants("  钟馗\n发售 "))
        self.assertEqual(await decide(model, msg(1, "真红 钟馗啥时候出"), None, []), "钟馗 发售")
        self.assertEqual(model.options, [{"thinking": False}])
        system, user = (m["content"] for m in model.calls[0])
        self.assertIn("decide", system)
        self.assertIn("真红 钟馗啥时候出", user)

    async def test_nothing_to_look_up(self) -> None:
        self.assertEqual(await decide(FakeModel(wants("")), msg(1, "真红 吃了没"), None, []), "")

    async def test_no_decision_raises(self) -> None:
        for reply in (ModelReply(), wants(None)):
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                await decide(FakeModel(reply), msg(1, "真红"), None, [])


class LookUpTests(unittest.IsolatedAsyncioTestCase):
    HIT = Hit("钟馗 发售日", "八月")

    async def test_searches_for_what_was_decided(self) -> None:
        search = FakeSearch(self.HIT)
        self.assertEqual(await look_up(FakeModel(wants("钟馗 发售")), search, msg(1, "真红 钟馗啥时候出")), (self.HIT,))
        self.assertEqual(search.queries, ["钟馗 发售"])

    async def test_no_search_when_nothing_needs_looking_up(self) -> None:
        search = FakeSearch(self.HIT)
        self.assertEqual(await look_up(FakeModel(wants("")), search, msg(1, "真红 吃了没")), ())
        self.assertEqual(search.queries, [])

    async def test_failures_mean_nothing_found(self) -> None:
        cases = [
            (FakeModel(RuntimeError("gateway down")), FakeSearch(self.HIT)),
            (FakeModel(ModelReply()), FakeSearch(self.HIT)),
            (FakeModel(wants("钟馗")), FakeSearch(error=httpx.ConnectError("refused"))),
        ]
        for model, search in cases:
            with self.subTest(model=model), self.assertLogs("shinnku_bot.harness.lookup", "WARNING"):
                self.assertEqual(await look_up(model, search, msg(1, "真红 钟馗啥时候出")), ())


class RenderTests(unittest.TestCase):
    TURNS = [{"role": "user", "content": "小林: 真红 钟馗啥时候出"}]

    def test_a_hit_is_one_line(self) -> None:
        self.assertEqual(Hit("钟馗", "八月", "2026-09-10").line, "钟馗（2026-09-10）：八月")
        self.assertEqual(Hit("钟馗", "八月").line, "钟馗：八月")

    def test_known_comes_before_the_waiting_questions(self) -> None:
        self.assertEqual(render_known([]), "")
        known = render_known(["钟馗：八月", "悟空：去年"])
        self.assertEqual(known, f"（{KNOWN_INTRO}\n钟馗：八月\n悟空：去年）")
        messages = build_messages("我", "h", self.TURNS, known=known, pending="（问题）")
        self.assertEqual(messages[-1]["content"], f"h\n小林: 真红 钟馗啥时候出\n{known}\n（问题）")
        self.assertNotIn(KNOWN_INTRO, messages[0]["content"], "the cached prefix stays as it is")

    def test_after_my_own_line_it_opens_a_turn_of_its_own(self) -> None:
        turns = [*self.TURNS, {"role": "assistant", "content": "等下"}]
        messages = build_messages("我", "h", turns, known="（知道的）", pending="（问题）")
        self.assertEqual(messages[-1], {"role": "user", "content": "（知道的）\n（问题）"})

    def test_citing_a_search_sounds_like_a_bot(self) -> None:
        for text in ("根据网上的消息，八月发售", "搜索结果显示是八月"):
            with self.subTest(text=text):
                self.assertIn("ai_tone", violations_of([text]))
        self.assertEqual(violations_of(["好像八月 我前几天刚看到"]), ())


class SettingTests(unittest.TestCase):
    def test_off_unless_set(self) -> None:
        env = {"TOKEN": "t", "ADMIN_TELEGRAM_USER_ID": "1", "LLM_BASE_URL": "u", "LLM_API_KEY": "k", "LLM_MODEL": "m"}
        self.assertEqual(load_settings(env).searxng_url, "")
        self.assertEqual(load_settings({**env, "SEARXNG_URL": " http://127.0.0.1:8888 "}).searxng_url, "http://127.0.0.1:8888")


if __name__ == "__main__":
    unittest.main()
