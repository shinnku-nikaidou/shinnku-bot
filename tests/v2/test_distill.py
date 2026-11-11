"""Distilling who I am, the people and the groups; the diary; how they reach the prompt."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from shinnku_bot.corpus import distill
from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.ingest import ingest
from shinnku_bot.corpus.pairs import assign_holdout, build_pairs
from shinnku_bot.eval.facts import is_correct, load_probes, run_probes, summarize_probes
from shinnku_bot.eval.run import DraftPersona
from shinnku_bot.harness.engine import Harness
from shinnku_bot.harness.hooks import StyleLimits
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.prompt import (
    ANCHOR_INTRO,
    DIARY_INTRO,
    FACTS_INTRO,
    GROUP_INTRO,
    MEMORY_INTRO,
    PEOPLE_INTRO,
    UNPROMPTED,
    build_messages,
    render_around,
    scene_header,
)
from shinnku_bot.memory.digest import as_mine, day_bounds, digest, digest_chat, select
from shinnku_bot.store import Store

from tests.v2.fakes import BOT_ID, FakeModel, FakeOutbox, ToolModel, make_settings, msg, no_sleep, say
from tests.v2.fixture_export import OWNER_ID, PEOPLE, message, write_export

ME, LIN, SHU = PEOPLE[0], PEOPLE[1], PEOPLE[2]
T0 = 1_700_000_000
DAY = 86400
CHAT = 1001
LIVE_CHAT = -1_000_000_001_001  # the Bot API id of export chat 1001


class Tmp(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def corpus(self, messages: list[dict], test_pairs: int = 1) -> CorpusDB:
        path = write_export(self.tmp / "export" / "result.json", messages, chat_id=CHAT)
        db = CorpusDB(":memory:")
        self.addCleanup(db.close)
        ingest(db, path, OWNER_ID)
        build_pairs(db)
        assign_holdout(db, per_chat=test_pairs, min_train=1)
        return db


class LintTests(unittest.TestCase):
    def test_framing_words_are_refused(self) -> None:
        for text in ("你是一个幽默的人", "请你保持简短", "我在扮演一个人", "按这个人设回复用户"):
            self.assertTrue(any(p.startswith("framing") for p in distill.lint(text, 1000)), text)

    def test_ordinary_facts_pass(self) -> None:
        self.assertEqual(distill.lint("我是 Fedora 用户，平时喜欢折腾 AI 和模型", 1000), [])

    def test_size_and_emptiness(self) -> None:
        self.assertEqual(distill.lint("", 10), ["empty"])
        self.assertEqual(distill.lint("我" * 100, 10), ["too_long:67>10"])


class SelfCandidateTests(Tmp):
    def build(self) -> CorpusDB:
        return self.corpus(
            [
                message(1, T0, LIN, "你住哪边"),
                message(2, T0 + 10, ME, "我住在杭州这边好多年了"),
                message(3, T0 + 20, ME, "我不去"),  # too short
                message(4, T0 + 30, LIN, "我也想去杭州"),  # not mine
                message(5, T0 + 40, ME, "我住在杭州这边好多年了"),  # said twice
                message(6, T0 + 50, ME, "转发的我的一段话很长很长", forwarded_from="某频道"),
                message(7, T0 + 60, ME, "今天天气真不错啊哈哈"),  # nothing about me
                message(8, T0 + DAY, LIN, "你用什么电脑"),
                message(9, T0 + DAY + 5, ME, "我一直用的是台式机"),
                message(10, T0 + 2 * DAY, LIN, "最近在干嘛"),
                message(11, T0 + 2 * DAY + 5, ME, "我最近在学做饭来着"),  # the held-out pair
            ]
        )

    def test_only_my_own_statements_from_the_training_period(self) -> None:
        found = distill.self_candidates(self.build())
        self.assertEqual(
            [c.lines[-1] for c in found], ["我: 我住在杭州这边好多年了", "我: 我一直用的是台式机"]
        )
        self.assertEqual(found[0].lines, ("小林: 你住哪边", "我: 我住在杭州这边好多年了"))
        self.assertEqual(found[1].lines[0], "小林: 你用什么电脑")  # a new session starts clean

    def test_thin_keeps_the_limit_and_leans_recent(self) -> None:
        candidates = [distill.Candidate(T0 + i, (f"我: {i}",)) for i in range(1200)]
        kept = distill.thin(candidates, 240)
        self.assertEqual(len(kept), 240)
        self.assertEqual(kept, sorted(kept, key=lambda c: c.ts))
        newest = sum(c.ts >= T0 + 600 for c in kept)
        self.assertGreater(newest, 240 - newest)
        self.assertEqual(distill.thin(candidates, 5000), candidates)

    def test_chunks_cover_everything(self) -> None:
        candidates = [distill.Candidate(T0 + i * DAY, ("小林: 问", f"我: 我的第{i}句话")) for i in range(50)]
        chunks = distill.chunk_candidates(candidates, max_chars=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 260 for chunk in chunks))
        self.assertEqual(sum(chunk.count("我: 我的第") for chunk in chunks), 50)
        self.assertTrue(chunks[0].startswith("[2023-11]\n小林: 问"))


class SelfDistillTests(Tmp):
    FACTS = [
        {"text": "我住在杭州", "category": "生活", "year": 2024, "sensitive": False},
        {"text": "我的门牌号是十二号", "category": "生活", "year": 2024, "sensitive": True},
        {"text": "我用台式机", "category": "设备", "year": 2023, "sensitive": False},
    ]

    def test_clean_facts(self) -> None:
        cleaned = distill.clean_facts(
            [{"text": " 我养猫 ", "category": "不存在", "year": "2022"}, {"text": ""}, "乱的", {"year": 1}]
        )
        self.assertEqual(cleaned, [{"text": "我养猫", "category": "生活", "year": 2022, "sensitive": False}])
        self.assertEqual(distill.clean_facts(None), [])

    def test_extract_is_resumable(self) -> None:
        model = ToolModel(note_facts={"facts": self.FACTS})
        out = self.tmp / "distill"
        self.assertEqual(asyncio.run(distill.extract_self(model, ["一", "二"], out)), (0, 2, 0))
        self.assertEqual(asyncio.run(distill.extract_self(model, ["一", "二", "三"], out)), (2, 1, 0))
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(len(distill.load_facts(out)), 3)  # the same facts from each chunk count once
        self.assertEqual(distill.load_facts(out, ["没做过的"]), [])

    def test_extract_gives_up_when_nothing_works(self) -> None:
        model = ToolModel()
        chunks = [str(i) for i in range(12)]
        reused, done, failed = asyncio.run(distill.extract_self(model, chunks, self.tmp / "d", concurrency=1))
        self.assertEqual((reused, done, failed), (0, 0, 12))
        self.assertEqual(len(model.calls), distill.GIVE_UP_AFTER)

    def test_merge_writes_drafts(self) -> None:
        model = ToolModel(
            write_section=lambda text: {"lines": ["- " + line.split("  ", 1)[1] for line in text.splitlines()]},
            write_self={"text": "我是个住在杭州的人。\n\n我跟朋友说话比较随意。"},
            write_probes={"probes": [{"question": "你住哪", "keywords": ["杭州"]}, {"question": "", "keywords": []}]},
        )
        result = asyncio.run(distill.merge_self(model, self.FACTS, self.tmp, as_of_year=2025))

        facts = (self.tmp / "FACTS.draft.md").read_text(encoding="utf-8")
        self.assertEqual(facts, "生活：\n- 我住在杭州\n\n设备：\n- 我用台式机\n")
        self.assertNotIn("门牌", "".join(model.asked("write_section")))
        self.assertIn("现在是 2025 年", model.calls[0][1])
        text = (self.tmp / "SELF.draft.md").read_text(encoding="utf-8")
        self.assertEqual(text, "我是个住在杭州的人。\n\n我跟朋友说话比较随意。\n")  # nothing about style is added
        self.assertEqual(load_probes(self.tmp / "probes.jsonl"), [{"question": "你住哪", "keywords": ["杭州"]}])
        self.assertEqual(
            (result["facts_in"], result["sensitive_dropped"], result["fact_lines"], result["probes"]), (3, 1, 2, 1)
        )
        self.assertEqual((result["facts_problems"], result["self_problems"]), ([], []))

    def test_merge_with_nothing_to_say(self) -> None:
        result = asyncio.run(distill.merge_self(ToolModel(), self.FACTS, self.tmp))
        self.assertEqual(result["fact_lines"], 0)
        self.assertEqual(result["self_problems"], ["empty"])

    def test_publish(self) -> None:
        self.assertEqual(distill.publish(self.tmp), {"SELF": ["no_draft"], "FACTS": ["no_draft"]})
        (self.tmp / "SELF.draft.md").write_text("我是我。\n", encoding="utf-8")
        (self.tmp / "FACTS.draft.md").write_text("你是一个机器人\n", encoding="utf-8")
        self.assertEqual(distill.publish(self.tmp), {"FACTS": ["framing_words:1"]})
        self.assertFalse((self.tmp / "SELF.md").exists())  # nothing goes live unless everything may

        (self.tmp / "FACTS.draft.md").write_text("我住在杭州\n", encoding="utf-8")
        self.assertEqual(distill.publish(self.tmp), {})
        self.assertEqual(Persona(self.tmp).facts_text(), "我住在杭州")
        (self.tmp / "SELF.draft.md").write_text("我还是我。\n", encoding="utf-8")
        distill.publish(self.tmp)
        self.assertEqual((self.tmp / "SELF.md.bak").read_text(encoding="utf-8"), "我是我。\n")
        self.assertEqual(Persona(self.tmp).self_text(), "我还是我。")

        report = distill.check_persona(self.tmp)
        self.assertEqual(report["SELF.md"], {"tokens": 3, "problems": []})
        self.assertNotIn("people", report)

    def test_draft_persona(self) -> None:
        (self.tmp / "SELF.md").write_text("我是线上的我", encoding="utf-8")
        self.assertEqual(DraftPersona(self.tmp).self_text(), "我是线上的我")
        (self.tmp / "SELF.draft.md").write_text("我是草稿里的我", encoding="utf-8")
        (self.tmp / "FACTS.draft.md").write_text("我住在杭州", encoding="utf-8")
        self.assertEqual(DraftPersona(self.tmp).self_text(), "我是草稿里的我")
        self.assertEqual(DraftPersona(self.tmp).facts_text(), "我住在杭州")
        self.assertEqual(Persona(self.tmp).facts_text(), "")


class PeopleTests(Tmp):
    def build(self) -> CorpusDB:
        messages, ts, msg_id = [], T0, 0

        def add(sender: tuple[str, str], text: str, **extra: object) -> int:
            nonlocal ts, msg_id
            msg_id += 1
            ts += 600
            messages.append(message(msg_id, ts, sender, text, **extra))
            return msg_id

        for i in range(6):
            add(LIN, f"小林的第{i}句")
            add(ME, f"回小林{i}")
        quoted = add(SHU, "阿树说的话")
        add(LIN, "小林插了一句")
        add(ME, "这句是回阿树的", reply_to_message_id=quoted)
        add(SHU, "阿树又说")
        add(ME, "回阿树")
        add(LIN, "最后一句")
        add(ME, "留作测试集")
        return self.corpus(messages)

    def test_ranking_follows_who_i_answer(self) -> None:
        ranked = distill.rank_people(self.build(), CHAT)
        self.assertEqual([(p.user_id, p.name, p.replies) for p in ranked], [(7, "小林", 6), (8, "阿树", 2)])

    def test_choose_people(self) -> None:
        ranked = distill.rank_people(self.build(), CHAT)
        chosen, covered = distill.choose_people(ranked, coverage=0.7, min_replies=1)
        self.assertEqual([p.user_id for p in chosen], [7])
        self.assertEqual(covered, 0.75)
        self.assertEqual(len(distill.choose_people(ranked, coverage=1.0, min_replies=1)[0]), 2)
        self.assertEqual(distill.choose_people(ranked, min_replies=3)[0], chosen)
        self.assertEqual(distill.choose_people(ranked, cap=0), ([], 0.0))
        self.assertEqual(distill.choose_people([]), ([], 0.0))

    def test_scenes_show_what_i_said_to_them(self) -> None:
        db = self.build()
        lin = distill.rank_people(db, CHAT)[0]
        scenes = distill.person_scenes(db, lin, limit=4)
        self.assertEqual(len(scenes.split("\n\n")), 4)
        self.assertIn("小林: 小林的第5句\n我: 回小林5", scenes)
        self.assertLess(len(distill.person_scenes(db, lin, max_chars=60)), 80)

    def test_cards_are_written_once(self) -> None:
        db = self.build()
        people = distill.rank_people(db, CHAT)
        model = ToolModel(write_card=lambda text: {"card": "我叫他老林，\n说话比较随便。"})
        result = asyncio.run(distill.write_people(model, db, CHAT, people, self.tmp))
        self.assertEqual(result, {"kept": 0, "written": 2, "rejected": 0, "failed": 0})
        self.assertEqual(Persona(self.tmp).person_card(LIVE_CHAT, 7), "我叫他老林， 说话比较随便。")
        self.assertIn("「小林」", model.calls[0][1])

        again = asyncio.run(distill.write_people(model, db, CHAT, people, self.tmp))
        self.assertEqual((again["kept"], again["written"]), (2, 0))
        self.assertEqual(len(model.calls), 2)
        forced = asyncio.run(distill.write_people(model, db, CHAT, people, self.tmp, force=True))
        self.assertEqual(forced["written"], 2)

    def test_bad_cards_are_dropped(self) -> None:
        db = self.build()
        people = distill.rank_people(db, CHAT)
        model = ToolModel(write_card={"card": "你是他的朋友，请你客气一点"})
        result = asyncio.run(distill.write_people(model, db, CHAT, people, self.tmp))
        self.assertEqual((result["written"], result["rejected"]), (0, 2))
        self.assertEqual(Persona(self.tmp).person_card(LIVE_CHAT, 7), "")
        failed = asyncio.run(distill.write_people(ToolModel(), db, CHAT, people, self.tmp))
        self.assertEqual(failed["failed"], 2)


class GroupTests(Tmp):
    def build(self) -> CorpusDB:
        messages = [message(i, T0 + i * 60, (LIN, SHU, ME)[i % 3], f"第{i}句") for i in range(1, 301)]
        return self.corpus(messages)

    def test_windows(self) -> None:
        db = self.build()
        windows = distill.group_windows(db, CHAT, count=5, size=20)
        self.assertEqual(len(windows), 5)
        self.assertTrue(all(len(window.splitlines()) == 20 for window in windows))
        self.assertIn("我: 第", windows[0])
        self.assertNotIn("第300句", "".join(windows))  # the held-out end is never read
        self.assertEqual(distill.group_windows(db, CHAT, size=1000), [])

    def test_notes_are_counted(self) -> None:
        text = distill.summarize_notes(
            [
                {"topics": ["游戏", "吃饭"], "jargon": [{"term": "开黑", "meaning": "一起打游戏"}], "vibe": "很闲"},
                {"topics": ["游戏"], "jargon": [{"term": "开黑", "meaning": "组队"}, {"term": ""}], "vibe": ""},
                "乱的",
            ]
        )
        self.assertIn("游戏 2\n吃饭 1", text)
        self.assertIn("开黑 2：", text)
        self.assertIn("很闲", text)

    def test_group_card(self) -> None:
        db = self.build()
        windows = distill.group_windows(db, CHAT, count=3, size=20)
        fast = ToolModel(note_group={"topics": ["游戏"], "jargon": [], "vibe": "很闲"})
        slow = ToolModel(write_card={"card": "这是我常待的群，大家爱聊游戏。"})
        notes = self.tmp / "notes"
        result = asyncio.run(distill.write_group(fast, slow, CHAT, windows, notes, self.tmp))
        self.assertEqual((result["noted"], result["written"]), (3, True))
        self.assertEqual(Persona(self.tmp).group_card(LIVE_CHAT), "这是我常待的群，大家爱聊游戏。")
        self.assertIn("游戏 3", slow.asked("write_card")[0])

        asyncio.run(distill.write_group(fast, slow, CHAT, windows, notes, self.tmp))
        self.assertEqual(len(fast.calls), 3)  # notes are reused
        nothing = asyncio.run(distill.write_group(ToolModel(), slow, CHAT, windows, self.tmp / "n2", self.tmp))
        self.assertEqual((nothing["noted"], nothing["written"]), (0, False))


class PersonaFilesTests(Tmp):
    def write(self, relative: str, text: str) -> None:
        path = self.tmp / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_cards(self) -> None:
        self.write("people/1001/7.md", "我叫他老林\n")
        self.write("people/1001/9.md", "老周是群主")
        persona = Persona(self.tmp)
        self.assertEqual(persona.people_cards(LIVE_CHAT, [8, 7, 7, 9], 4), [(7, "我叫他老林"), (9, "老周是群主")])
        self.assertEqual(persona.people_cards(LIVE_CHAT, [7, 9], 1), [(7, "我叫他老林")])
        self.assertEqual(persona.people_cards(-5, [7], 4), [])  # another chat has its own cards
        self.assertEqual(persona.group_card(LIVE_CHAT), "")

    def test_diary_is_the_last_days_before_today(self) -> None:
        for day in ("2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11"):
            self.write(f"diary/1001/{day}.md", f"{day} 的事")
        self.write("diary/1001/notes.md", "不是日记")
        persona = Persona(self.tmp)
        self.assertEqual(
            persona.diary(LIVE_CHAT, "2026-11-11", 2),
            [("2026-11-09", "2026-11-09 的事"), ("2026-11-10", "2026-11-10 的事")],
        )
        self.assertEqual(persona.diary(LIVE_CHAT, "2026-11-11", 0), [])
        self.assertEqual(persona.diary(-5, "2026-11-11", 3), [])


class PromptTests(unittest.TestCase):
    TURNS = [{"role": "user", "content": "小林: 在吗"}]

    def test_around(self) -> None:
        self.assertEqual(render_around(), "")
        text = render_around("爱聊游戏的群", [("小林", "我叫他老林")], [("2026-11-10", "聊了游戏")])
        self.assertEqual(
            text,
            f"{GROUP_INTRO}\n爱聊游戏的群\n\n{PEOPLE_INTRO}\n「小林」我叫他老林\n\n{DIARY_INTRO}\n2026-11-10：聊了游戏",
        )

    def test_stable_part_first(self) -> None:
        messages = build_messages(
            "我是我", "（群聊）", self.TURNS, anchor="小林: 来\n我: 来", memory="一段往事", facts="我住在杭州", around="群的事"
        )
        system = messages[0]["content"]
        self.assertLess(system.index("我是我"), system.index(FACTS_INTRO))
        self.assertLess(system.index("我住在杭州"), system.index(ANCHOR_INTRO))
        self.assertEqual(messages[1]["content"], f"群的事\n\n{MEMORY_INTRO}\n\n一段往事")
        self.assertEqual(messages[2]["content"], "（群聊）\n小林: 在吗")

    def test_optional_parts_leave_no_trace(self) -> None:
        messages = build_messages("我是我", "（群聊）", self.TURNS)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], "我是我")
        only_around = build_messages("我是我", "（群聊）", self.TURNS, around="群的事")
        self.assertEqual(only_around[1]["content"], "群的事")

    def test_unprompted_header(self) -> None:
        now = datetime(2026, 11, 11, 21, 14)
        self.assertNotIn(UNPROMPTED, scene_header("某群", False, "x", now))
        self.assertEqual(
            scene_header("某群", False, None, now, unprompted=True), f"（群「某群」，周三 21:14，{UNPROMPTED}）"
        )


class EngineMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.settings = make_settings(Path(tmp.name), people_cards=1)
        persona = self.settings.persona_dir
        for relative, text in (
            ("FACTS.md", "我住在杭州"),
            ("groups/1001.md", "爱聊游戏的群"),
            ("people/1001/7.md", "我叫他老林"),
            ("people/1001/8.md", "阿树是我同学"),
            ("diary/1001/2023-11-14.md", "昨天聊了游戏"),
            ("diary/1001/2099-01-01.md", "还没发生的事"),
        ):
            (persona / relative).parent.mkdir(parents=True, exist_ok=True)
            (persona / relative).write_text(text, encoding="utf-8")
        self.model = FakeModel(say("嗯"))
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.harness = Harness(
            self.settings, self.store, self.model, FakeOutbox(), Persona(persona),
            bot_id=BOT_ID, sleep=no_sleep, now=lambda: 1_700_000_000.0 + 100,
        )

    async def test_what_i_know_reaches_the_prompt(self) -> None:
        self.store.record(msg(1, "在吗", user_id=8, name="阿树", chat_id=LIVE_CHAT, ts=T0))
        await self.harness.answer(msg(2, "在吗", chat_id=LIVE_CHAT, ts=T0 + 1), "mention", title="测试群")
        prompt = self.model.calls[0]
        self.assertIn("我住在杭州", prompt[0]["content"])
        around = prompt[1]["content"]
        self.assertIn("爱聊游戏的群", around)
        self.assertIn("「小林」我叫他老林", around)  # the person who called me comes first
        self.assertNotIn("阿树是我同学", around)  # and only one card was allowed
        self.assertIn("2023-11-14：昨天聊了游戏", around)
        self.assertNotIn("还没发生的事", around)

    async def test_private_chats_get_none_of_it(self) -> None:
        await self.harness.answer(msg(1, "在吗", chat_id=7, ts=T0), "private", is_private=True)
        prompt = self.model.calls[0]
        self.assertEqual([m["role"] for m in prompt], ["system", "user"])
        self.assertIn("我住在杭州", prompt[0]["content"])


class FactProbeTests(Tmp):
    def test_matching(self) -> None:
        self.assertTrue(is_correct("在杭州啊", ["上海", "杭州"]))
        self.assertTrue(is_correct("用 Fedora", ["fedora"]))
        self.assertFalse(is_correct("不告诉你", ["杭州", " "]))

    def test_probes_are_asked_in_private(self) -> None:
        (self.tmp / "probes.jsonl").write_text(
            '{"question": "你住哪", "keywords": ["杭州"]}\n坏行\n{"question": "", "keywords": []}\n'
            '{"question": "用什么电脑", "keywords": ["台式"]}\n',
            encoding="utf-8",
        )
        probes = load_probes(self.tmp / "probes.jsonl")
        self.assertEqual(len(probes), 2)
        model = FakeModel(say("杭州啊"))
        results = asyncio.run(
            run_probes(make_settings(self.tmp), probes, model, Persona(self.tmp), None, StyleLimits(), float(T0))
        )
        self.assertEqual([r["ok"] for r in results], [True, False])
        self.assertIn("（私聊，", model.calls[0][-1]["content"])
        self.assertIn("朋友: 你住哪", model.calls[0][-1]["content"])
        self.assertEqual(
            summarize_probes(results), {"asked": 2, "errors": 0, "answered": 2, "correct": 1, "accuracy": 0.5}
        )
        self.assertEqual(load_probes(self.tmp / "missing.jsonl"), [])

    def test_errors_are_counted_apart(self) -> None:
        class Broken:
            async def complete(self, messages: object, tools: object) -> None:
                raise TimeoutError

        probes = [{"question": f"问题{i}", "keywords": ["x"]} for i in range(12)]
        results = asyncio.run(
            run_probes(make_settings(self.tmp), probes, Broken(), Persona(self.tmp), None, StyleLimits(), float(T0), 1)
        )
        summary = summarize_probes(results)
        self.assertEqual((summary["errors"], summary["accuracy"]), (12, 0.0))
        self.assertEqual(sum(r["error"] == "Skipped" for r in results), 4)


class DigestTests(Tmp):
    DAY = date(2026, 11, 11)

    def setUp(self) -> None:
        super().setUp()
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.start, self.end = day_bounds(self.DAY, "Asia/Shanghai")
        self.model = ToolModel(write_diary={"entry": "今天群里聊了游戏。\n我说周末一起打。"})

    def fill(self, chat_id: int, count: int = 6) -> None:
        for i in range(count):
            self.store.record(msg(i + 1, f"第{i}句", chat_id=chat_id, ts=self.start + i * 60, user_id=(7, 1, 999)[i % 3]))

    def test_day_bounds_follow_the_timezone(self) -> None:
        self.assertEqual(self.end - self.start, DAY)
        # Midnight in Shanghai is 16:00 UTC the day before.
        self.assertEqual(self.start, int(datetime(2026, 11, 10, 16, tzinfo=timezone.utc).timestamp()))

    def test_the_owner_counts_as_me(self) -> None:
        mine = as_mine([msg(1, "a", user_id=1), msg(2, "b", user_id=7)], owner_id=1)
        self.assertEqual([m.is_me for m in mine], [True, False])

    def test_select_keeps_what_is_around_me(self) -> None:
        msgs = [msg(i, str(i), is_me=(i == 50)) for i in range(100)]
        self.assertEqual(len(select(msgs, limit=200)), 100)
        self.assertEqual([m.message_id for m in select(msgs, limit=10, around=2)], [48, 49, 50, 51, 52])
        self.assertEqual([m.message_id for m in select(msgs, limit=3, around=2)], [50, 51, 52])
        nobody = [msg(i, str(i)) for i in range(100)]
        self.assertEqual([m.message_id for m in select(nobody, limit=2)], [98, 99])

    def test_diary_is_written_once(self) -> None:
        self.fill(LIVE_CHAT)
        args = (self.model, self.store, self.tmp, LIVE_CHAT, self.DAY, "Asia/Shanghai", 1)
        self.assertEqual(asyncio.run(digest_chat(*args)), "written")
        self.assertEqual(
            Persona(self.tmp).diary(LIVE_CHAT, "2026-11-12", 3), [("2026-11-11", "今天群里聊了游戏。 我说周末一起打。")]
        )
        system, shown = self.model.calls[0][1], self.model.calls[0][2]
        self.assertIn("2026-11-11", system)
        self.assertEqual(shown.splitlines()[:2], ["小林: 第0句", "我: 第1句"])  # the owner's line is mine
        self.assertEqual(asyncio.run(digest_chat(*args)), "kept")
        self.assertEqual(asyncio.run(digest_chat(*args, force=True)), "written")
        self.assertEqual(len(self.model.calls), 2)

    def test_quiet_failed_and_rejected_days(self) -> None:
        args = (self.store, self.tmp, LIVE_CHAT, self.DAY, "Asia/Shanghai", 1)
        self.fill(LIVE_CHAT, count=3)
        self.assertEqual(asyncio.run(digest_chat(self.model, *args)), "quiet")
        self.fill(LIVE_CHAT)
        self.assertEqual(asyncio.run(digest_chat(ToolModel(), *args)), "failed")
        bossy = ToolModel(write_diary={"entry": "你要记住今天的事"})
        self.assertEqual(asyncio.run(digest_chat(bossy, *args)), "rejected")
        self.assertEqual(Persona(self.tmp).diary(LIVE_CHAT, "2026-11-12", 3), [])

    def test_every_allowed_group_and_no_private_chat(self) -> None:
        other = LIVE_CHAT - 1
        for chat_id in (LIVE_CHAT, other, 7):
            self.fill(chat_id)
        lines: list[str] = []
        settings = make_settings(self.tmp, allowed_chat_ids=frozenset({LIVE_CHAT}))
        counts = asyncio.run(digest(settings, self.store, self.model, self.DAY, progress=lines.append))
        self.assertEqual(counts, {"written": 1})
        self.assertEqual(lines, [f"chat={LIVE_CHAT} day=2026-11-11 written"])
        everywhere = asyncio.run(digest(make_settings(self.tmp), self.store, self.model, self.DAY))
        self.assertEqual(everywhere, {"kept": 1, "written": 1})

    def test_one_broken_chat_does_not_stop_the_rest(self) -> None:
        class Broken:
            async def complete(self, messages: object, tools: object) -> None:
                raise TimeoutError("chat text must not leak")

        self.fill(LIVE_CHAT)
        lines: list[str] = []
        counts = asyncio.run(digest(make_settings(self.tmp), self.store, Broken(), self.DAY, progress=lines.append))
        self.assertEqual(counts, {"failed": 1})
        self.assertNotIn("leak", json.dumps(lines))


if __name__ == "__main__":
    unittest.main()
