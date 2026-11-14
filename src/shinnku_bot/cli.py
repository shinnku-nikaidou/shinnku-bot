"""Command line entry point: `shinnku <command>`."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import itertools
import logging
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from shinnku_bot.config import Settings, load_settings

CHAT_ID = 1


def setup_logging() -> None:
    """Log to stdout. Lines carry ids, counts and timings, never message text."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    # httpx logs every request URL, and Telegram URLs contain the bot token.
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "telegram", "openai", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# -- bot ----------------------------------------------------------------------


def cmd_bot(args: argparse.Namespace) -> int:
    from shinnku_bot.tg.app import run

    setup_logging()
    run(load_settings())
    return 0


# -- sandbox ------------------------------------------------------------------


async def _sandbox(
    settings: Settings, lines: Iterable[str], show_thinking: bool, stats_only: bool
) -> None:
    """Talk to the harness in the terminal: no Telegram, nothing stored on disk."""
    from shinnku_bot.harness.factory import build_harness
    from shinnku_bot.store import Msg, Store

    ids = itertools.count(1)

    class ConsoleOutbox:
        async def typing(self, chat_id: int) -> None:
            return None

        async def send(self, chat_id: int, text: str, reply_to: int | None) -> int:
            if not stats_only:
                print(f"  我: {text}")
            return next(ids)

    async def no_sleep(seconds: float) -> None:
        return None

    harness = build_harness(
        dataclasses.replace(settings, trace=False),
        Store(":memory:"),
        ConsoleOutbox(),
        bot_id=0,
        sleep=no_sleep,
    )
    pending: Msg | None = None

    async def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        started = time.monotonic()
        # The pending line is already stored; answering re-records it, which is harmless.
        outcome = await harness.answer(pending, "mention", title="测试群")
        if stats_only:
            print(
                f"  [{outcome.kind}] lens={[len(m) for m in outcome.messages]} "
                f"tries={outcome.attempts} violations={list(outcome.violations)} "
                f"reasoning_chars={len(outcome.reasoning)} took={time.monotonic() - started:.1f}s"
            )
        else:
            if show_thinking and outcome.reasoning:
                print(f"  （内心）{outcome.reasoning.strip()}")
            if outcome.kind != "said":
                print(f"  [{outcome.kind}]")
            if outcome.violations:
                print(f"  [检查: {', '.join(outcome.violations)}; 尝试 {outcome.attempts} 次]")
        pending = None

    for raw in lines:
        line = raw.strip()
        if not line:
            await flush()
            continue
        name, sep, text = line.partition(":")
        if not sep or not text.strip():
            name, text = "朋友", line
        pending = Msg(
            chat_id=CHAT_ID,
            message_id=next(ids),
            ts=int(time.time()),
            user_id=abs(hash(name.strip())) % 10**9 + 1,
            user_name=name.strip(),
            is_me=False,
            text=text.strip(),
        )
        harness.store.record(pending)
    await flush()


def cmd_chat(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.no_thinking:
        settings = dataclasses.replace(settings, llm_thinking=False)
    if args.bare:
        # Point at an empty directory so no persona, sample or index is found.
        settings = dataclasses.replace(settings, data_dir=Path("/nonexistent/shinnku-bare"))
    if args.stats_only:
        setup_logging()  # the harness log line has token counts and recall size, no text
    if sys.stdin.isatty():
        print("每行一条消息，格式「名字: 内容」；空行让我回话；Ctrl-D 结束。")
    asyncio.run(_sandbox(settings, sys.stdin, args.show_thinking, args.stats_only))
    return 0


# -- corpus -------------------------------------------------------------------


def _fmt_day(ts: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else "-"


def _corpus(action: Callable[[Settings, argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
    """Run a corpus command so that a crash cannot print chat text."""

    def run(args: argparse.Namespace) -> int:
        from shinnku_bot.corpus.safe import run_safely

        settings = load_settings()
        return run_safely(lambda: action(settings, args), settings.reports_dir)

    return run


def _probe(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.export import find_exports
    from shinnku_bot.corpus.probe import format_report, probe

    paths = [Path(args.file)] if args.file else find_exports(settings.export_dir)
    if not paths:
        print("no result.json found under the export directory")
        return 1
    for path in paths:
        print(format_report(probe(path, settings.owner_id)))
        print()
    return 0


def _ingest(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.db import CorpusDB
    from shinnku_bot.corpus.export import find_exports
    from shinnku_bot.corpus.ingest import ingest
    from shinnku_bot.corpus.pairs import assign_holdout, build_pairs
    from shinnku_bot.corpus.stats import build_fingerprint, format_fingerprint, save_fingerprint

    paths = find_exports(settings.export_dir)
    if not paths:
        print("no result.json found under the export directory")
        return 1
    db = CorpusDB(settings.corpus_db)
    try:
        for path in paths:
            r = ingest(db, path, settings.owner_id)
            print(
                f"ingested {r.source}: chat={r.chat_id} read={r.read} stored={r.stored} "
                f"mine={r.mine} {_fmt_day(r.first_ts)}..{_fmt_day(r.last_ts)} "
                f"complete={'yes' if r.complete else 'no'} kinds={dict(r.kinds.most_common())}"
            )
        print(f"pairs per chat: {build_pairs(db)}")
        print(f"split: {assign_holdout(db, per_chat=args.test_per_chat)}")
        fingerprint = build_fingerprint(db, settings.timezone)
        save_fingerprint(fingerprint, settings.persona_dir / "fingerprint.json")
        print(format_fingerprint(fingerprint))
    finally:
        db.close()
    return 0


def _index(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.db import CorpusDB
    from shinnku_bot.corpus.index import build_index
    from shinnku_bot.harness.recall import FastEmbedder

    db = CorpusDB(settings.corpus_db)
    try:
        embedder = FastEmbedder(settings.embed_model, settings.models_dir)
        print(f"index: {build_index(db, settings.index_dir, embedder, progress=print)}")
    finally:
        db.close()
    return 0


def _anchor(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.anchor import write_anchor
    from shinnku_bot.corpus.db import CorpusDB

    db = CorpusDB(settings.corpus_db)
    try:
        count = args.scenes if args.scenes is not None else settings.anchor_scenes
        scenes, chars = write_anchor(db, settings.persona_dir / "anchor.txt", count)
        print(f"anchor: {scenes} scenes, {chars} characters")
    finally:
        db.close()
    return 0


def _lengths(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.db import CorpusDB
    from shinnku_bot.corpus.stats import length_report
    from shinnku_bot.harness.persona import Persona

    db = CorpusDB(settings.corpus_db)
    try:
        report = length_report(db, Persona(settings.persona_dir).anchor_text())
    finally:
        db.close()
    print(f"{'my messages':<16}{'n':>8}{'mean':>7}{'median':>8}{'p90':>6}{'over 30':>9}")
    for name, row in report.items():
        print(
            f"{name:<16}{row['n']:>8}{row['mean']:>7}{row['median']:>8.0f}{row['p90']:>6.0f}{row['over_30']:>9.1%}"
        )
    return 0


def _build(settings: Settings, args: argparse.Namespace) -> int:
    started = time.monotonic()
    for step in (_ingest, _anchor, _index):
        code = step(settings, args)
        if code:
            return code
    print(f"done in {time.monotonic() - started:.0f}s")
    return 0


def _load_fingerprint(settings: Settings) -> dict | None:
    import json

    try:
        return json.loads((settings.persona_dir / "fingerprint.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _distill_model(settings: Settings, slow: bool = False) -> object:
    """A model for distillation: long answers, generous timeout."""
    from shinnku_bot.llm import GatewayModel

    name = (settings.llm_model_slow or settings.llm_model) if slow else settings.llm_model
    return GatewayModel(
        settings.llm_base_url,
        settings.llm_api_key,
        name,
        thinking=True,
        timeout_s=240,
        effort="low",
        max_tokens=8192,
    )


# -- distill ------------------------------------------------------------------


def _distill_self(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus import distill
    from shinnku_bot.corpus.db import CorpusDB

    db = CorpusDB(settings.corpus_db)
    try:
        candidates = distill.self_candidates(db)
    finally:
        db.close()
    kept = distill.thin(candidates, args.limit)
    chunks = distill.chunk_candidates(kept)
    tokens = distill.estimate_tokens("".join(chunks)) + len(chunks) * distill.estimate_tokens(distill.EXTRACT_SYSTEM)
    out_dir = settings.distill_dir / "self"
    print(
        f"self-statements: {len(candidates)}  kept: {len(kept)}  chunks: {len(chunks)}  "
        f"already extracted: {distill.extracted(out_dir, chunks)}  estimated input tokens: {tokens}"
    )
    if args.dry_run or not chunks:
        return 0
    if args.only != "merge":
        reused, done, failed = asyncio.run(
            distill.extract_self(_distill_model(settings), chunks, out_dir, args.concurrency, print)
        )
        print(f"extract: reused {reused}, done {done}, failed {failed}")
        if failed > len(chunks) // 10:
            print("too many chunks failed; run the same command again to continue")
            return 1
    if args.only != "extract":
        facts = distill.load_facts(out_dir, chunks)
        year = time.gmtime(kept[-1].ts).tm_year
        result = asyncio.run(
            distill.merge_facts(
                _distill_model(settings, slow=True),
                facts,
                settings.persona_dir,
                as_of_year=year,
                progress=print,
            )
        )
        print(f"merge: {result}")
        print("written: FACTS.draft.md and probes.jsonl (review the draft, then `distill publish`)")
    return 0


def _distill_people(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus import distill
    from shinnku_bot.corpus.db import CorpusDB

    db = CorpusDB(settings.corpus_db)
    try:
        for chat_id in db.chat_ids():
            ranked = distill.rank_people(db, chat_id)
            chosen, covered = distill.choose_people(ranked, args.coverage, args.max_people)
            print(
                f"chat={chat_id} people I answer: {len(ranked)}  chosen: {len(chosen)}  "
                f"covering {covered:.1%} of my replies"
            )
            if args.dry_run or not chosen:
                continue
            result = asyncio.run(
                distill.write_people(
                    _distill_model(settings), db, chat_id, chosen, settings.persona_dir,
                    args.force, args.concurrency, print,
                )
            )
            print(f"chat={chat_id} cards: {result}")
            if result["failed"]:
                return 1
    finally:
        db.close()
    return 0


def _distill_groups(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus import distill
    from shinnku_bot.corpus.db import CorpusDB

    db = CorpusDB(settings.corpus_db)
    try:
        for chat_id in db.chat_ids():
            windows = distill.group_windows(db, chat_id, args.windows)
            tokens = distill.estimate_tokens("".join(windows))
            print(f"chat={chat_id} windows: {len(windows)}  estimated input tokens: {tokens}")
            if args.dry_run or not windows:
                continue
            result = asyncio.run(
                distill.write_group(
                    _distill_model(settings),
                    _distill_model(settings, slow=True),
                    chat_id,
                    windows,
                    settings.distill_dir / "groups" / str(chat_id),
                    settings.persona_dir,
                    args.concurrency,
                    print,
                )
            )
            print(f"chat={chat_id} group card: {result}")
    finally:
        db.close()
    return 0


def _distill_check(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.distill import check_persona

    for name, value in check_persona(settings.persona_dir).items():
        print(f"{name}: {value}")
    return 0


def _distill_publish(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.distill import publish

    problems = publish(settings.persona_dir)
    if problems:
        print(f"not published: {problems}")
        return 1
    print("published: FACTS.md is live (the previous version is kept as FACTS.md.bak)")
    return 0


def _moments(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.db import CorpusDB
    from shinnku_bot.corpus.moments import build_urge, format_urge, save_moments
    from shinnku_bot.harness.urge import load_urge

    fingerprint = _load_fingerprint(settings) or {}
    db = CorpusDB(settings.corpus_db)
    try:
        model, report, held_out = build_urge(
            db, settings.timezone, fingerprint.get("hour_share"), settings.name_triggers
        )
    finally:
        db.close()
    path = settings.persona_dir / "urge.json"
    previous = load_urge(path)
    if previous is not None:
        model.speak_rate = previous.speak_rate  # measured by `eval urge`, not by this command
    model.save(path)
    # Only moments I was around for: the offline check compares joining in with letting it pass.
    save_moments(settings.distill_dir / "moments-test.jsonl", [m for m in held_out if m.present])
    print(format_urge(report))
    return 0


# -- memory -------------------------------------------------------------------


def _memory_digest(settings: Settings, args: argparse.Namespace) -> int:
    from datetime import date

    from shinnku_bot.memory.digest import digest
    from shinnku_bot.store import Store

    store = Store(settings.db_path)
    try:
        day = date.fromisoformat(args.date) if args.date else None
        counts = asyncio.run(digest(settings, store, _distill_model(settings), day, args.force, print))
    finally:
        store.close()
    print(f"digest: {counts}")
    return 1 if counts.get("failed") else 0


def _memory_forget(settings: Settings, args: argparse.Namespace) -> int:
    """Forget what happened in the groups: their stored messages and diaries. The cards stay."""
    import shutil

    from shinnku_bot.store import Store

    store = Store(settings.db_path)
    try:
        messages = store.forget_groups()
    finally:
        store.close()
    diary = settings.persona_dir / "diary"  # only groups keep a diary
    days = len(list(diary.glob("*/*.md")))
    shutil.rmtree(diary, ignore_errors=True)
    print(f"forget: messages={messages} diary_days={days}")
    return 0


# -- eval ---------------------------------------------------------------------


def _eval_run(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.db import CorpusDB
    from shinnku_bot.corpus.stats import limits_from_fingerprint
    from shinnku_bot.eval.metrics import format_summary, summarize
    from shinnku_bot.eval.run import DraftPersona, EvalOptions, run_eval, save_report
    from shinnku_bot.harness.factory import gateway_model
    from shinnku_bot.harness.persona import Persona
    from shinnku_bot.harness.recall import load_recall

    if args.model:
        settings = dataclasses.replace(settings, llm_model=args.model)
    if args.no_thinking:
        settings = dataclasses.replace(settings, llm_thinking=False)
    if args.recall_scenes is not None:
        settings = dataclasses.replace(settings, recall_scenes=args.recall_scenes)
    options = EvalOptions(
        n=args.n,
        seed=args.seed,
        concurrency=args.concurrency,
        use_recall=not args.no_recall,
        use_anchor=not args.no_anchor,
        judge=args.judge,
    )
    judge_name, judge = _judge(settings, args)
    db = CorpusDB(settings.corpus_db)
    try:
        recall = load_recall(
            settings.corpus_db, settings.index_dir, settings.embed_model, settings.models_dir
        )
        print(
            f"model={settings.llm_model} thinking={settings.llm_thinking} judge={judge_name} "
            f"recall={'on' if options.use_recall and recall is not None else 'off'} "
            f"anchor={'on' if options.use_anchor else 'off'}"
        )
        items = asyncio.run(
            run_eval(
                settings,
                db,
                gateway_model(settings),
                judge,
                (DraftPersona if args.draft else Persona)(settings.persona_dir),
                recall,
                limits_from_fingerprint(settings.persona_dir / "fingerprint.json"),
                options,
                progress=print,
            )
        )
    finally:
        db.close()
    from shinnku_bot.eval.run import add_similarity
    from shinnku_bot.harness.recall import FastEmbedder

    add_similarity(items, FastEmbedder(settings.embed_model, settings.models_dir))
    save_report(settings.reports_dir / f"eval-{args.name}.json", args.name, options, settings, items)
    print(format_summary(args.name, summarize(items)))
    return 0


def _judge(settings: Settings, args: argparse.Namespace) -> tuple[str, object]:
    from shinnku_bot.llm import GatewayModel

    name = args.judge_model or settings.llm_model_judge or settings.llm_model
    model = GatewayModel(
        settings.llm_base_url,
        settings.llm_api_key,
        name,
        thinking=True,
        timeout_s=180,
        effort=args.judge_effort,
        # A careful judge reasons at length before it picks.
        max_tokens=8192,
    )
    return name, model


def _eval_judge(settings: Settings, args: argparse.Namespace) -> int:
    """Judge a saved run again without regenerating the replies."""
    from shinnku_bot.eval.metrics import format_summary, summarize
    from shinnku_bot.eval.run import load_items, reference_scenes, rejudge
    from shinnku_bot.harness.persona import Persona

    path = settings.reports_dir / f"eval-{args.name}.json"
    if not path.is_file():
        print(f"no report named {args.name}")
        return 1
    items = load_items(path)
    judge_name, judge = _judge(settings, args)
    reference = reference_scenes(Persona(settings.persona_dir), args.reference)
    started = time.monotonic()
    asyncio.run(rejudge(items, judge, reference, args.seed, args.concurrency))
    summary = summarize(items)
    print(f"judge={judge_name} effort={args.judge_effort} reference_scenes={args.reference} took={time.monotonic() - started:.0f}s")
    print(format_summary(args.name, summary).splitlines()[2])
    return 0


def _eval_recall(settings: Settings, args: argparse.Namespace) -> int:
    """Offline: no model calls, only the local embedding model."""
    from shinnku_bot.corpus.db import CorpusDB
    from shinnku_bot.eval.recall_check import check_recall, format_check
    from shinnku_bot.eval.run import sample_pairs
    from shinnku_bot.harness.recall import FastEmbedder, Recall

    embedder = FastEmbedder(settings.embed_model, settings.models_dir)
    recall = Recall(settings.corpus_db, settings.index_dir, embedder)
    db = CorpusDB(settings.corpus_db)
    try:
        k = args.recall_scenes if args.recall_scenes is not None else settings.recall_scenes
        result = check_recall(db, recall, embedder, sample_pairs(db, args.n, args.seed), k, args.seed)
    finally:
        db.close()
        recall.close()
    print(format_check(result))
    return 0


def _eval_style(settings: Settings, args: argparse.Namespace) -> int:
    """Offline: can a classifier tell the bot's replies from mine?"""
    import json

    from shinnku_bot.corpus.db import CorpusDB
    from shinnku_bot.eval.run import load_items
    from shinnku_bot.eval.style import format_style, style_report

    path = settings.reports_dir / f"eval-{args.name}.json"
    if not path.is_file():
        print(f"no report named {args.name}; run `shinnku eval run --name {args.name}` first")
        return 1
    bare = None
    if args.bare:
        bare_path = settings.reports_dir / f"eval-{args.bare}.json"
        bare = load_items(bare_path) if bare_path.is_file() else None
    db = CorpusDB(settings.corpus_db)
    try:
        report, tells = style_report(load_items(path), db, bare, args.bundle, args.others, args.seed)
    finally:
        db.close()
    (settings.reports_dir / f"style-{args.name}.json").write_text(
        json.dumps({"report": report, "tells": tells}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(format_style(args.name, report))
    return 0


def _eval_facts(settings: Settings, args: argparse.Namespace) -> int:
    import json

    from shinnku_bot.corpus.stats import limits_from_fingerprint
    from shinnku_bot.eval.facts import load_probes, run_probes, summarize_probes
    from shinnku_bot.eval.run import DraftPersona
    from shinnku_bot.harness.factory import gateway_model
    from shinnku_bot.harness.persona import Persona
    from shinnku_bot.harness.recall import load_recall

    probes = load_probes(settings.persona_dir / "probes.jsonl")
    if not probes:
        print("no probes; run `shinnku corpus distill self` first")
        return 1
    recall = load_recall(settings.corpus_db, settings.index_dir, settings.embed_model, settings.models_dir)
    results = asyncio.run(
        run_probes(
            settings,
            probes,
            gateway_model(settings),
            (DraftPersona if args.draft else Persona)(settings.persona_dir),
            recall,
            limits_from_fingerprint(settings.persona_dir / "fingerprint.json"),
            time.time(),
            args.concurrency,
        )
    )
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "facts.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"facts: {summarize_probes(results)}")
    return 0


def _eval_urge(settings: Settings, args: argparse.Namespace) -> int:
    from shinnku_bot.corpus.db import CorpusDB
    from shinnku_bot.corpus.moments import load_moments
    from shinnku_bot.corpus.stats import limits_from_fingerprint
    from shinnku_bot.eval.urge_check import check_urge
    from shinnku_bot.harness.factory import gateway_model
    from shinnku_bot.harness.persona import Persona
    from shinnku_bot.harness.recall import load_recall
    from shinnku_bot.harness.urge import load_urge

    moments = load_moments(settings.distill_dir / "moments-test.jsonl")
    if not moments:
        print("no held-out moments; run `shinnku corpus moments` first")
        return 1
    recall = load_recall(settings.corpus_db, settings.index_dir, settings.embed_model, settings.models_dir)
    db = CorpusDB(settings.corpus_db)
    try:
        result = asyncio.run(
            check_urge(
                settings,
                db,
                moments,
                gateway_model(settings),
                Persona(settings.persona_dir),
                recall,
                limits_from_fingerprint(settings.persona_dir / "fingerprint.json"),
                args.n,
                args.seed,
                args.concurrency,
            )
        )
    finally:
        db.close()
    print(f"urge: {result}")
    path = settings.persona_dir / "urge.json"
    model = load_urge(path)
    if model is not None and result["where_i_joined"]["n"] >= 20:
        model.speak_rate = max(result["where_i_joined"]["rate"], 0.05)
        model.save(path)
        print(f"speak_rate stored: {model.speak_rate}")
    return 0


def cmd_eval_quiz(args: argparse.Namespace) -> int:
    """Interactive, and meant to show text: this is the owner's own terminal."""
    from shinnku_bot.eval.quiz import run_quiz

    settings = load_settings()
    path = settings.reports_dir / f"eval-{args.name}.json"
    if not path.is_file():
        print(f"no report named {args.name}; run `shinnku eval run --name {args.name}` first")
        return 1
    run_quiz(path, args.n)
    return 0


# -- parser -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shinnku")
    sub = parser.add_subparsers(dest="command", required=True)

    bot = sub.add_parser("bot", help="run the Telegram bot")
    bot.set_defaults(func=cmd_bot)

    chat = sub.add_parser("chat", help="talk to the harness in the terminal")
    chat.add_argument("--show-thinking", action="store_true", help="print the inner monologue")
    chat.add_argument("--no-thinking", action="store_true", help="turn reasoning off")
    chat.add_argument("--stats-only", action="store_true", help="print lengths and timings, not text")
    chat.add_argument("--bare", action="store_true", help="placeholder persona only, no corpus")
    chat.set_defaults(func=cmd_chat)

    corpus = sub.add_parser("corpus", help="build the corpus from exported chat history")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)

    probe_cmd = corpus_sub.add_parser("probe", help="print the structure of the exports")
    probe_cmd.add_argument("--file", help="one result.json instead of the export directory")
    probe_cmd.set_defaults(func=_corpus(_probe))

    def add(name: str, action: Callable[[Settings, argparse.Namespace], int], text: str) -> None:
        command = corpus_sub.add_parser(name, help=text)
        command.add_argument("--test-per-chat", type=int, default=300)
        command.add_argument("--scenes", type=int, default=None, help="size of the fixed sample")
        command.set_defaults(func=_corpus(action))

    add("ingest", _ingest, "load exports, cut reply pairs, hold out a test set, measure style")
    add("anchor", _anchor, "pick the fixed sample of my conversations")
    add("index", _index, "build the recall index")
    add("build", _build, "ingest, anchor and index in one go")

    corpus_sub.add_parser("lengths", help="how long my messages are, by year and in the fixed sample").set_defaults(
        func=_corpus(_lengths)
    )
    moments_cmd = corpus_sub.add_parser("moments", help="learn when I join in uncalled (no model calls)")
    moments_cmd.set_defaults(func=_corpus(_moments))

    distill_cmd = corpus_sub.add_parser("distill", help="distil who I am, the people and the groups")
    distill_sub = distill_cmd.add_subparsers(dest="distill_command", required=True)
    self_cmd = distill_sub.add_parser("self", help="draft FACTS.md from what I said about myself")
    self_cmd.add_argument("--limit", type=int, default=12000, help="most self-statements to read")
    self_cmd.add_argument("--only", choices=["extract", "merge"], help="run one half of the job")
    people_cmd = distill_sub.add_parser("people", help="one card per person I talk to most")
    people_cmd.add_argument("--max-people", type=int, default=40)
    people_cmd.add_argument("--coverage", type=float, default=0.8)
    people_cmd.add_argument("--force", action="store_true", help="rewrite cards that already exist")
    groups_cmd = distill_sub.add_parser("groups", help="one card per group")
    groups_cmd.add_argument("--windows", type=int, default=40)
    for command, action in ((self_cmd, _distill_self), (people_cmd, _distill_people), (groups_cmd, _distill_groups)):
        command.add_argument("--dry-run", action="store_true", help="print the size of the job and stop")
        command.add_argument("--concurrency", type=int, default=4)
        command.set_defaults(func=_corpus(action))
    distill_sub.add_parser("check", help="sizes and checks of everything distilled").set_defaults(
        func=_corpus(_distill_check)
    )
    distill_sub.add_parser("publish", help="make the reviewed drafts live").set_defaults(
        func=_corpus(_distill_publish)
    )

    memory = sub.add_parser("memory", help="what I remember from day to day")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    digest_cmd = memory_sub.add_parser("digest", help="write yesterday's diary for each group")
    digest_cmd.add_argument("--date", help="YYYY-MM-DD instead of yesterday")
    digest_cmd.add_argument("--force", action="store_true", help="rewrite a diary that already exists")
    digest_cmd.set_defaults(func=_corpus(_memory_digest))
    memory_sub.add_parser(
        "forget", help="forget what happened in the groups: their stored messages and diaries"
    ).set_defaults(func=_corpus(_memory_forget))

    evaluate = sub.add_parser("eval", help="blind test against held-out real replies")
    eval_sub = evaluate.add_subparsers(dest="eval_command", required=True)
    run_cmd = eval_sub.add_parser("run", help="generate replies for held-out moments and judge them")
    run_cmd.add_argument("--name", default="base", help="label for this run and its report file")
    run_cmd.add_argument("--n", type=int, default=40)
    run_cmd.add_argument("--seed", type=int, default=0)
    run_cmd.add_argument("--concurrency", type=int, default=4)
    run_cmd.add_argument("--model", help="chat model to test instead of LLM_MODEL")
    run_cmd.add_argument("--judge-model", help="judge model instead of LLM_MODEL_JUDGE")
    run_cmd.add_argument("--judge-effort", default="low", help="judge reasoning effort")
    run_cmd.add_argument("--recall-scenes", type=int, default=None)
    run_cmd.add_argument("--no-thinking", action="store_true")
    run_cmd.add_argument("--no-recall", action="store_true")
    run_cmd.add_argument("--no-anchor", action="store_true")
    run_cmd.add_argument("--judge", action="store_true", help="also ask a model which reply is real")
    run_cmd.add_argument("--draft", action="store_true", help="use FACTS.draft.md, not the live facts")
    run_cmd.set_defaults(func=_corpus(_eval_run))
    recall_cmd = eval_sub.add_parser("recall", help="offline check of what recall surfaces")
    recall_cmd.add_argument("--n", type=int, default=100)
    recall_cmd.add_argument("--seed", type=int, default=0)
    recall_cmd.add_argument("--recall-scenes", type=int, default=None)
    recall_cmd.set_defaults(func=_corpus(_eval_recall))
    judge_cmd = eval_sub.add_parser("judge", help="judge a saved run again, e.g. with another model")
    judge_cmd.add_argument("--name", default="base")
    judge_cmd.add_argument("--judge-model")
    judge_cmd.add_argument("--judge-effort", default="low")
    judge_cmd.add_argument("--reference", type=int, default=20, help="scenes of my real style shown to the judge")
    judge_cmd.add_argument("--seed", type=int, default=0)
    judge_cmd.add_argument("--concurrency", type=int, default=6)
    judge_cmd.set_defaults(func=_corpus(_eval_judge))
    style_cmd = eval_sub.add_parser("style", help="can a classifier tell the bot's replies from mine")
    style_cmd.add_argument("--name", default="base")
    style_cmd.add_argument("--bare", help="report of a run without my persona, as the upper reference")
    style_cmd.add_argument("--bundle", type=int, default=1, help="replies per sample")
    style_cmd.add_argument("--others", type=int, default=400, help="other people's replies to compare with")
    style_cmd.add_argument("--seed", type=int, default=0)
    style_cmd.set_defaults(func=_corpus(_eval_style))
    facts_cmd = eval_sub.add_parser("facts", help="ask the bot the facts of my life")
    facts_cmd.add_argument("--draft", action="store_true", help="use FACTS.draft.md, not the live facts")
    facts_cmd.add_argument("--concurrency", type=int, default=4)
    facts_cmd.set_defaults(func=_corpus(_eval_facts))
    urge_cmd = eval_sub.add_parser("urge", help="does the model speak where I spoke, uncalled")
    urge_cmd.add_argument("--n", type=int, default=100, help="moments of each kind")
    urge_cmd.add_argument("--seed", type=int, default=0)
    urge_cmd.add_argument("--concurrency", type=int, default=4)
    urge_cmd.set_defaults(func=_corpus(_eval_urge))
    quiz_cmd = eval_sub.add_parser("quiz", help="take the blind test yourself")
    quiz_cmd.add_argument("--name", default="base")
    quiz_cmd.add_argument("--n", type=int, default=20)
    quiz_cmd.set_defaults(func=cmd_eval_quiz)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
