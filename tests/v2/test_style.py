"""The style discriminator: features, the classifier, and the reference lines."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shinnku_bot.corpus.db import CorpusDB
from shinnku_bot.corpus.ingest import ingest
from shinnku_bot.corpus.pairs import assign_holdout, build_pairs
from shinnku_bot.eval.classify import (
    NGRAM_DIM,
    STAT_NAMES,
    LogReg,
    auc,
    auc_interval,
    bucket,
    cross_val_scores,
    featurize,
    fold_of,
    ngrams,
    text_stats,
)
from shinnku_bot.eval.metrics import Item
from shinnku_bot.eval.style import (
    bundle,
    format_style,
    gram_tells,
    others_bursts,
    spoken,
    stat_gaps,
    style_report,
    tell_apart,
)

from tests.v2.fixture_export import OWNER_ID, PEOPLE, message, write_export

ME, LIN, SHU = PEOPLE[0], PEOPLE[1], PEOPLE[2]
T0 = 1_700_000_000
CASUAL = ["不去", "太累了", "确实", "草", "行吧", "懒得查", "几点", "来", "笑死", "哈哈哈哈"]
STIFF = ["好的，我明白了。", "这个问题很有意思。", "我觉得可以试试看。", "没问题，交给我吧。", "谢谢你的提醒。"]


def replies(pool: list[str], n: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    return [[rng.choice(pool) for _ in range(rng.randint(1, 2))] for _ in range(n)]


class FeatureTests(unittest.TestCase):
    def test_stats(self) -> None:
        stats = dict(zip(STAT_NAMES, text_stats(["好的。", "ok吗？"])))
        self.assertEqual(stats["messages"], 2)
        self.assertEqual(stats["chars_per_message"], 3.5)
        self.assertEqual(stats["longest_message"], 4)
        self.assertEqual(stats["ends_with_period"], 0.5)
        self.assertEqual(stats["ends_with_mark"], 0.5)
        self.assertEqual(stats["has_question"], 1.0)
        self.assertAlmostEqual(stats["latin_per_char"], 2 / 7)
        self.assertEqual(text_stats([]), [0.0] * len(STAT_NAMES))
        self.assertGreater(dict(zip(STAT_NAMES, text_stats(["哈哈哈哈"])))["repeated_chars"], 0.7)

    def test_ngrams_mark_the_ends(self) -> None:
        grams = ngrams(["好。"])
        self.assertIn("。$", grams)
        self.assertIn("^好", grams)
        self.assertEqual(bucket("。$"), bucket("。$"))
        self.assertLess(bucket("。$"), NGRAM_DIM)

    def test_featurize_shape(self) -> None:
        x = featurize([["不去"], []])
        self.assertEqual(x.shape, (2, NGRAM_DIM + len(STAT_NAMES)))
        self.assertAlmostEqual(float(np.linalg.norm(x[0, :NGRAM_DIM])), 1.0, places=5)
        self.assertEqual(float(np.abs(x[1]).sum()), 0.0)


class ClassifierTests(unittest.TestCase):
    def test_auc(self) -> None:
        self.assertEqual(auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])), 1.0)
        self.assertEqual(auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])), 0.0)
        self.assertEqual(auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])), 0.5)
        self.assertEqual(auc(np.array([0.1, 0.2]), np.array([1, 1])), 0.5)
        self.assertAlmostEqual(auc(np.array([1, 2, 2, 3]), np.array([0, 0, 1, 1])), 0.875)

    def test_groups_stay_together(self) -> None:
        groups = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
        fold = fold_of(groups, 3, seed=1)
        self.assertTrue(all(fold[i] == fold[i + 6] for i in range(6)))
        self.assertEqual(set(fold), {0, 1, 2})

    def test_logreg_learns_a_simple_rule(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 3))
        y = (x[:, 0] > 0).astype(float)
        model = LogReg(scaled=3).fit(x, y)
        self.assertGreater(auc(model.decision(x), y), 0.98)
        self.assertTrue(np.all((model.probability(x) >= 0) & (model.probability(x) <= 1)))

    def test_interval_brackets_the_estimate(self) -> None:
        rng = np.random.default_rng(0)
        y = np.repeat([0.0, 1.0], 100)
        scores = y + rng.normal(scale=1.0, size=200)
        low, high = auc_interval(scores, y, np.arange(200))
        self.assertLess(low, auc(scores, y))
        self.assertGreater(high, auc(scores, y))

    def test_cross_validation_scores_every_sample(self) -> None:
        x = featurize(replies(CASUAL, 30, 0) + replies(STIFF, 30, 1))
        y = np.repeat([0.0, 1.0], 30)
        scores = cross_val_scores(x, y, np.arange(60), scaled=len(STAT_NAMES))
        self.assertEqual(scores.shape, (60,))
        self.assertGreater(auc(scores, y), 0.9)


class TellApartTests(unittest.TestCase):
    def test_different_styles_are_told_apart(self) -> None:
        result = tell_apart(replies(CASUAL, 80, 0), replies(STIFF, 80, 1), paired=True)
        self.assertGreater(result["auc"], 0.95)
        self.assertLessEqual(result["low"], result["auc"])
        self.assertEqual((result["mine"], result["theirs"]), (80, 80))

    def test_the_same_style_is_not(self) -> None:
        result = tell_apart(replies(CASUAL, 150, 0), replies(CASUAL, 150, 1))
        self.assertGreater(result["auc"], 0.3)
        self.assertLess(result["auc"], 0.7)

    def test_too_few_samples(self) -> None:
        result = tell_apart(replies(CASUAL, 5, 0), replies(STIFF, 5, 1))
        self.assertIsNone(result["auc"])

    def test_bundle(self) -> None:
        samples = [["a"], ["b", "c"], ["d"], ["e"], ["f"]]
        self.assertEqual(bundle(samples, 1, [1, 0]), [["b", "c"], ["a"]])
        self.assertEqual(bundle(samples, 2, [0, 1, 2, 3, 4]), [["a", "b", "c"], ["d", "e"]])

    def test_stat_gaps_and_tells(self) -> None:
        mine, theirs = replies(CASUAL, 60, 0), replies(STIFF, 60, 1)
        gaps = {gap["name"]: gap for gap in stat_gaps(mine, theirs)}
        self.assertEqual(gaps["ends_with_period"]["me"], 0.0)
        self.assertGreater(gaps["ends_with_period"]["bot"], 0.5)
        self.assertEqual(stat_gaps([], theirs), [])
        tells = gram_tells(mine, theirs)
        self.assertTrue(any("。" in gram for gram in tells["bot"]))
        self.assertEqual(gram_tells(mine[:3], theirs[:3]), {"bot": [], "me": []})


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        rng = random.Random(0)
        messages = []
        ts = T0
        for i in range(1, 1201):
            ts += rng.randint(5, 40)
            sender = (ME, LIN, SHU)[i % 3]
            pool = CASUAL if sender is ME else STIFF
            messages.append(message(i, ts, sender, rng.choice(pool)))
        path = write_export(Path(self._tmp.name) / "export" / "result.json", messages)
        self.db = CorpusDB(":memory:")
        self.addCleanup(self.db.close)
        ingest(self.db, path, OWNER_ID)
        build_pairs(self.db)
        assign_holdout(self.db, per_chat=120, min_train=100)

    def test_others_bursts_come_from_the_held_out_period(self) -> None:
        bursts = others_bursts(self.db, 50)
        self.assertEqual(len(bursts), 50)
        self.assertTrue(all(line in STIFF for burst in bursts for line in burst))
        self.assertEqual(others_bursts(self.db, 50, seed=3) == bursts, False)

    def test_report(self) -> None:
        items = [
            Item(i, 1, T0, real=[random.Random(i).choice(CASUAL)], bot=[random.Random(i).choice(STIFF)])
            for i in range(60)
        ]
        items.append(Item(99, 1, T0, real=["来"], bot=[]))
        items.append(Item(98, 1, T0, real=["来"], bot=["好的。"], error="Timeout"))
        self.assertEqual(len(spoken(items)[0]), 60)
        report, tells = style_report(items, self.db, bare_items=items, others=100)
        self.assertGreater(report["bot"]["auc"], 0.95)
        self.assertGreater(report["others"]["auc"], 0.9)
        self.assertGreater(report["bare"]["auc"], 0.95)
        self.assertLess(abs(report["self"]["auc"] - 0.5), 0.2)
        self.assertTrue(tells["bot"])
        text = format_style("base", report)
        self.assertIn("bot vs me", text)
        self.assertIn("ends_with_period", text)
        for line in CASUAL + STIFF:
            self.assertNotIn(line, text)

    def test_report_without_a_corpus(self) -> None:
        items = [Item(i, 1, T0, real=["来"], bot=["好的。"]) for i in range(40)]
        report, _ = style_report(items, None)
        self.assertNotIn("others", report)
        self.assertIn("not measured", format_style("x", report))


if __name__ == "__main__":
    unittest.main()
