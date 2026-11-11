"""Recall index, fixed sample, and how both reach the prompt."""

from __future__ import annotations

import tempfile
import unittest
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from shinnku_bot.corpus.anchor import render_anchor, select_anchor, write_anchor
from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.index import build_index
from shinnku_bot.corpus.ingest import ingest
from shinnku_bot.corpus.pairs import assign_holdout, build_pairs
from shinnku_bot.harness.engine import Harness
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.prompt import ANCHOR_INTRO, MEMORY_INTRO, build_messages
from shinnku_bot.harness.recall import Recall, export_chat_id, fts_query, load_recall
from shinnku_bot.harness.transcript import render_lines, render_scene, tail_text
from shinnku_bot.store import Store

from tests.v2.fakes import BOT_ID, FakeModel, FakeOutbox, make_settings, msg, no_sleep, say
from tests.v2.fixture_export import OWNER_ID, PEOPLE, message, write_export

ME, LIN, SHU = PEOPLE[0], PEOPLE[1], PEOPLE[2]
T0 = 1_700_000_000
HOUR = 3600


class HashEmbedder:
    """Bag of character bigrams hashed into a small vector. Deterministic."""

    name = "hash-test"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), 4096), dtype=np.float32)
        for row, text in enumerate(texts):
            compact = "".join(text.split())
            for i in range(len(compact) - 1):
                out[row, zlib.crc32(compact[i : i + 2].encode()) % 4096] += 1
            if not out[row].any():
                out[row, 0] = 1
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def exchange(first_id: int, ts: int, said: str, answer: str, who=LIN) -> list[dict]:
    """Someone says something, I answer; far enough apart to be separate conversations."""
    return [message(first_id, ts, who, said), message(first_id + 1, ts + 5, ME, answer)]


class RenderTests(unittest.TestCase):
    CONTEXT = [
        msg(1, "今晚打游戏吗"),
        msg(2, "我十点后", user_id=8, name="阿树"),
        msg(3, "行", user_id=BOT_ID, name="x", is_me=True),
        msg(4, "", kind="photo"),
    ]

    def test_lines(self) -> None:
        self.assertEqual(
            render_lines(self.CONTEXT),
            ["小林: 今晚打游戏吗", "阿树: 我十点后", "我: 行", "小林: [图片]"],
        )

    def test_scene_keeps_only_the_tail_of_the_context(self) -> None:
        reply = [msg(5, "打", user_id=BOT_ID, is_me=True), msg(6, "十点", user_id=BOT_ID, is_me=True)]
        self.assertEqual(
            render_scene(self.CONTEXT, reply, tail=2), "我: 行\n小林: [图片]\n我: 打\n我: 十点"
        )

    def test_tail_text_has_no_names(self) -> None:
        self.assertEqual(tail_text(self.CONTEXT, lines=3), "我十点后\n行\n[图片]")
        self.assertEqual(tail_text(self.CONTEXT, lines=4, max_chars=7), "\n行\n[图片]")
        self.assertEqual(tail_text([]), "")


class QueryTests(unittest.TestCase):
    def test_fts_query(self) -> None:
        self.assertEqual(fts_query("打游戏吗"), '"打游戏" OR "游戏吗"')
        self.assertEqual(fts_query("ab\n打游戏"), '"打游戏"', "newest line first, short words skipped")
        self.assertEqual(fts_query('a"b'), '"a""b"')
        self.assertIsNone(fts_query("草"))
        self.assertIsNone(fts_query("好 的"))

    def test_export_chat_id(self) -> None:
        self.assertEqual(export_chat_id(-1001234567890), 1234567890)
        self.assertEqual(export_chat_id(-4123), 4123)
        self.assertEqual(export_chat_id(77), 77)


class IndexCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.db_path = self.tmp / "corpus.sqlite"
        self.index_dir = self.tmp / "index"
        self.db = CorpusDB(self.db_path)
        self.addCleanup(self.db.close)

    def load(self, messages: list[dict], chat_id: int = 1001) -> None:
        ingest(self.db, write_export(self.tmp / str(chat_id) / "result.json", messages, chat_id=chat_id), OWNER_ID)

    def index(self) -> Recall:
        build_pairs(self.db)
        self.meta = build_index(self.db, self.index_dir, HashEmbedder())
        recall = Recall(self.db_path, self.index_dir, HashEmbedder())
        self.addCleanup(recall.close)
        return recall

    def said(self, recall: Recall, pair_ids: list[int]) -> list[str]:
        return ["/".join(m.text for m in self.db.reply_of(self.db.pair(p))) for p in pair_ids]


class RecallTests(IndexCase):
    def test_finds_the_similar_moment(self) -> None:
        self.load(
            exchange(1, T0, "今晚有人打游戏吗", "打")
            + exchange(3, T0 + 2 * HOUR, "这周末去不去爬山", "不去 太累了")
            + exchange(5, T0 + 4 * HOUR, "中午吃了没", "没")
        )
        recall = self.index()
        self.assertEqual(self.meta["count"], 3)
        self.assertEqual(len(recall), 3)
        self.assertEqual(self.said(recall, recall.search("晚上打游戏吗", None, 1)), ["打"])
        self.assertEqual(self.said(recall, recall.search("周末爬山去吗", None, 1)), ["不去 太累了"])

    def test_render(self) -> None:
        self.load(exchange(1, T0, "今晚有人打游戏吗", "打"))
        recall = self.index()
        self.assertEqual(recall.render(recall.search("打游戏吗", None, 5)), "小林: 今晚有人打游戏吗\n我: 打")
        self.assertEqual(recall.scenes([msg(1, "打游戏吗")], None, 5), "小林: 今晚有人打游戏吗\n我: 打")

    def test_the_same_reply_is_not_recalled_over_and_over(self) -> None:
        messages: list[dict] = []
        for i in range(6):
            messages += exchange(2 * i + 1, T0 + i * 2 * HOUR, f"笑死我了哈哈哈{i}", "草")
        messages += exchange(21, T0 + 20 * HOUR, "笑死我了真的", "确实")
        self.load(messages)
        recall = self.index()
        said = self.said(recall, recall.search("笑死我了哈哈哈", None, 10))
        self.assertEqual(said.count("草"), 2)
        self.assertIn("确实", said)

    def test_held_out_pairs_are_not_indexed(self) -> None:
        messages: list[dict] = []
        for i in range(30):
            answer = "这是测试集里的回复" if i >= 25 else "嗯"
            messages += exchange(2 * i + 1, T0 + i * 2 * HOUR, f"第{i}次问你在不在", answer)
        self.load(messages)
        build_pairs(self.db)
        assign_holdout(self.db, per_chat=5, min_train=10)
        build_index(self.db, self.index_dir, HashEmbedder())
        recall = Recall(self.db_path, self.index_dir, HashEmbedder())
        self.addCleanup(recall.close)
        self.assertEqual(len(recall), 25)
        said = self.said(recall, recall.search("第27次问你在不在", None, 25))
        self.assertNotIn("这是测试集里的回复", said)

    def test_same_chat_is_preferred(self) -> None:
        self.load(exchange(1, T0, "今晚有人打游戏吗", "这个群的回答"), chat_id=1001)
        self.load(exchange(1, T0, "今晚有人打游戏吗", "那个群的回答"), chat_id=2002)
        recall = self.index()
        self.assertEqual(self.said(recall, recall.search("今晚有人打游戏吗", -1_000_000_002_002, 1)), ["那个群的回答"])
        self.assertEqual(self.said(recall, recall.search("今晚有人打游戏吗", -1_000_000_001_001, 1)), ["这个群的回答"])

    def test_newer_memories_win_ties(self) -> None:
        self.load(
            exchange(1, T0, "今晚有人打游戏吗", "旧的回答")
            + exchange(3, T0 + 900 * 24 * HOUR, "今晚有人打游戏吗", "新的回答")
        )
        recall = self.index()
        self.assertEqual(self.said(recall, recall.search("今晚有人打游戏吗", None, 1)), ["新的回答"])

    def test_nothing_to_search_with(self) -> None:
        self.load(exchange(1, T0, "今晚有人打游戏吗", "打"))
        recall = self.index()
        self.assertEqual(recall.search("  ", None, 5), [])

    def test_model_mismatch_is_refused(self) -> None:
        self.load(exchange(1, T0, "今晚有人打游戏吗", "打"))
        self.index()

        class Other(HashEmbedder):
            name = "another-model"

        with self.assertRaises(ValueError):
            Recall(self.db_path, self.index_dir, Other())

    def test_load_recall_without_an_index(self) -> None:
        self.assertIsNone(load_recall(self.db_path, self.tmp / "missing", "any", self.tmp))


class IncrementalIndexTests(IndexCase):
    def test_rebuild_only_embeds_what_is_new(self) -> None:
        class Counting(HashEmbedder):
            def __init__(self) -> None:
                self.embedded: list[str] = []

            def embed(self, texts: Sequence[str]) -> np.ndarray:
                self.embedded += [t for t in texts if t != "维度"]
                return super().embed(texts)

        first = exchange(1, T0, "今晚有人打游戏吗", "打") + exchange(3, T0 + 2 * HOUR, "这周末去不去爬山", "不去")
        self.load(first)
        build_pairs(self.db)
        embedder = Counting()
        meta = build_index(self.db, self.index_dir, embedder)
        self.assertEqual((meta["count"], meta["embedded"]), (2, 2))
        before = np.load(self.index_dir / "tail.npy").copy()

        # More history arrives: an earlier message shifts every pair id.
        more = exchange(11, T0 - 9 * HOUR, "早上好啊各位", "早") + [dict(m, id=m["id"] + 20) for m in first]
        more += exchange(41, T0 + 4 * HOUR, "中午吃了没", "没")
        self.load(more)
        build_pairs(self.db)
        embedder.embedded.clear()
        meta = build_index(self.db, self.index_dir, embedder)
        self.assertEqual((meta["count"], meta["embedded"]), (4, 2))
        self.assertEqual(sorted(embedder.embedded), ["中午吃了没", "早上好啊各位"])

        after = np.load(self.index_dir / "tail.npy")
        np.testing.assert_allclose(after[1:3], before, err_msg="reused vectors are carried over unchanged")
        self.assertFalse((self.index_dir / "tail.new.npy").exists())
        recall = Recall(self.db_path, self.index_dir, HashEmbedder())
        self.addCleanup(recall.close)
        self.assertEqual(self.said(recall, recall.search("晚上打游戏吗", None, 1)), ["打"])
        self.assertEqual(self.said(recall, recall.search("中午吃了没有", None, 1)), ["没"])

    def test_another_model_starts_from_scratch(self) -> None:
        self.load(exchange(1, T0, "今晚有人打游戏吗", "打"))
        build_pairs(self.db)
        build_index(self.db, self.index_dir, HashEmbedder())

        class Other(HashEmbedder):
            name = "another-model"

        self.assertEqual(build_index(self.db, self.index_dir, Other())["embedded"], 1)


class RecallCheckTests(IndexCase):
    def test_recall_beats_random(self) -> None:
        from shinnku_bot.eval.recall_check import check_recall, format_check
        from shinnku_bot.eval.run import sample_pairs

        topics = ["今晚打游戏吗", "周末去爬山吗", "中午吃什么好", "明天要上班吗", "新番看了没有", "这个代码怎么写"]
        answers = ["打 十点上线", "不去 爬山太累", "随便吃点面", "要上班 烦死", "看了 一般般", "代码自己写去"]
        messages: list[dict] = []
        for i in range(120):
            topic = i % len(topics)
            messages += exchange(2 * i + 1, T0 + i * 2 * HOUR, f"{topics[topic]}", answers[topic])
        self.load(messages)
        build_pairs(self.db)
        assign_holdout(self.db, per_chat=30, min_train=30)
        build_index(self.db, self.index_dir, HashEmbedder())
        recall = Recall(self.db_path, self.index_dir, HashEmbedder())
        self.addCleanup(recall.close)

        pairs = sample_pairs(self.db, 30, seed=0)
        result = check_recall(self.db, recall, HashEmbedder(), pairs, k=3)
        self.assertEqual((result["moments"], result["k"]), (30, 3))
        self.assertGreater(result["mean_similarity"]["recalled"], result["mean_similarity"]["random"] + 0.3)
        self.assertEqual(result["exact_reply_recalled"], 1.0)
        self.assertEqual(result["same_chat_share"], 1.0)
        text = format_check(result)
        self.assertIn("recall beats random", text)
        for answer in answers:
            self.assertNotIn(answer, text)


class AnchorTests(IndexCase):
    def build(self, n: int = 240, distinct: int = 60) -> None:
        messages: list[dict] = []
        for i in range(n):
            ts = T0 + i * 2 * HOUR
            messages += [
                message(3 * i + 1, ts, LIN, f"第{i}个话题"),
                message(3 * i + 2, ts + 5, SHU, "我也想知道"),
                message(3 * i + 3, ts + 9, ME, f"回答{i % distinct}"),
            ]
        self.load(messages)
        build_pairs(self.db)

    def test_sample_is_deduplicated_ordered_and_recent_heavy(self) -> None:
        self.build()
        chosen = select_anchor(self.db, 36, seed=1)
        self.assertEqual(len(chosen), 36)
        said = ["/".join(m.text for m in self.db.reply_of(p)) for p in chosen]
        self.assertEqual(len(set(said)), 36)
        self.assertEqual([p.ts for p in chosen], sorted(p.ts for p in chosen))
        middle = T0 + 120 * 2 * HOUR
        self.assertGreater(sum(p.ts >= middle for p in chosen), sum(p.ts < middle for p in chosen))
        self.assertEqual([p.pair_id for p in chosen], [p.pair_id for p in select_anchor(self.db, 36, seed=1)])

    def test_returns_exactly_what_was_asked_for(self) -> None:
        self.build(245, distinct=245)  # not a multiple of the number of time slices
        for count in (1, 50, 107, 180):
            with self.subTest(count=count):
                chosen = select_anchor(self.db, count, seed=2)
                self.assertEqual(len(chosen), count)
                self.assertEqual(len({p.pair_id for p in chosen}), count)

    def test_asking_for_more_than_exists_gives_everything_distinct(self) -> None:
        self.build(30)  # 30 distinct replies
        self.assertEqual(len(select_anchor(self.db, 500, seed=0)), 30)

    def test_held_out_pairs_are_never_sampled(self) -> None:
        self.build()
        assign_holdout(self.db, per_chat=40, min_train=50)
        test_ids = {p.pair_id for p in self.db.pairs("test")}
        chosen = select_anchor(self.db, 150, seed=0)
        self.assertFalse(test_ids & {p.pair_id for p in chosen})

    def test_write(self) -> None:
        self.build(24)
        scenes, chars = write_anchor(self.db, self.tmp / "persona" / "anchor.txt", 5)
        text = (self.tmp / "persona" / "anchor.txt").read_text(encoding="utf-8")
        self.assertEqual(scenes, 5)
        self.assertEqual(len(text.strip()), chars)
        self.assertEqual(text.count("我: 回答"), 5)
        self.assertEqual(len(text.strip().split("\n\n")), 5)
        self.assertEqual(render_anchor(self.db, []), "")

    def test_empty_corpus(self) -> None:
        self.assertEqual(select_anchor(self.db, 10), [])


class PromptMemoryTests(unittest.TestCase):
    TURNS = [{"role": "user", "content": "小林: 在吗"}]

    def test_anchor_joins_the_identity_and_memory_follows(self) -> None:
        messages = build_messages("我是我", "（群聊）", self.TURNS, anchor="小林: 吃了没\n我: 没", memory="小林: 在吗\n我: 在")
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": f"我是我\n\n{ANCHOR_INTRO}\n\n小林: 吃了没\n我: 没"},
                {"role": "system", "content": f"{MEMORY_INTRO}\n\n小林: 在吗\n我: 在"},
                {"role": "user", "content": "（群聊）\n小林: 在吗"},
            ],
        )

    def test_stable_prefix_does_not_depend_on_memory(self) -> None:
        with_memory = build_messages("我", "h", self.TURNS, anchor="a", memory="m")
        without = build_messages("我", "h", self.TURNS, anchor="a")
        self.assertEqual(with_memory[0], without[0])
        self.assertEqual(len(without), 2)


class FakeRecall:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[tuple[str, int | None, int]] = []

    def __len__(self) -> int:
        return 2

    def search(self, tail: str, chat_id: int | None, k: int) -> list[int]:
        if self.fail:
            raise RuntimeError("index gone")
        self.queries.append((tail, chat_id, k))
        return [1, 2]

    def render(self, pair_ids: Sequence[int]) -> str:
        return "小林: 在吗\n我: 在"


class HarnessMemoryTests(unittest.IsolatedAsyncioTestCase):
    def build(self, recall: FakeRecall) -> Harness:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        settings = make_settings(Path(tmp.name), recall_scenes=7)
        settings.persona_dir.mkdir(parents=True)
        (settings.persona_dir / "anchor.txt").write_text("小林: 吃了没\n我: 没\n", encoding="utf-8")
        store = Store(":memory:")
        self.addCleanup(store.close)
        self.model = FakeModel(say("在"))
        self.outbox = FakeOutbox()
        return Harness(
            settings,
            store,
            self.model,
            self.outbox,
            Persona(settings.persona_dir),
            bot_id=BOT_ID,
            recall=recall,  # type: ignore[arg-type]
            sleep=no_sleep,
        )

    async def test_prompt_carries_sample_and_memories(self) -> None:
        recall = FakeRecall()
        harness = self.build(recall)
        await harness.answer(msg(1, "真红在吗"), "name")

        self.assertEqual(recall.queries, [("真红在吗", 100, 7)])
        prompt = self.model.calls[0]
        self.assertEqual([m["role"] for m in prompt], ["system", "system", "user"])
        self.assertIn(ANCHOR_INTRO, prompt[0]["content"])
        self.assertTrue(prompt[0]["content"].endswith("小林: 吃了没\n我: 没"))
        self.assertEqual(prompt[1]["content"], f"{MEMORY_INTRO}\n\n小林: 在吗\n我: 在")

    async def test_broken_recall_still_answers(self) -> None:
        harness = self.build(FakeRecall(fail=True))
        with self.assertLogs("shinnku_bot.harness.engine", level="WARNING"):
            outcome = await harness.answer(msg(1, "真红在吗"), "name")
        self.assertEqual(outcome.kind, "said")
        self.assertEqual([m["role"] for m in self.model.calls[0]], ["system", "user"])


if __name__ == "__main__":
    unittest.main()
