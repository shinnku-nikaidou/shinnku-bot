# shinnku bot

A Telegram bot that talks like its owner. It reads the owner's exported chat
history, recalls how they answered in similar moments, and replies in their
voice. The design and its measured basis are in [plans/v2-plan.md](plans/v2-plan.md).

The new code lives in `src/shinnku_bot/`. The old bot (`src/main.py` and the
folders next to it) is still what the systemd service runs until the switch.

## Setup

```bash
uv sync --inexact
```

Use `--inexact` until the old bot is retired: a plain `uv sync` removes the
packages it still needs.

Copy `.env.example` to `.env` and fill in the Telegram token, your Telegram
user id and the model gateway.

## Build the corpus

Export each chat from Telegram Desktop as JSON without media into
`~/tg-export/`, then:

```bash
uv run --inexact shinnku corpus probe
```

```bash
uv run --inexact shinnku corpus build
```

`probe` prints the structure of the exports. `build` loads them, cuts
"context -> my reply" pairs, holds out the newest pairs of each chat as a test
set, measures the style fingerprint, picks the fixed sample and builds the
recall index. Both print counts and statistics only.

Everything derived from the chats is written under `data/`, which is ignored
by git.

## Distil who I am

```bash
uv run --inexact shinnku corpus distill self      # drafts SELF.md, FACTS.md and a quiz about me
uv run --inexact shinnku corpus distill publish   # after reading and editing the drafts
uv run --inexact shinnku corpus distill people    # a card per person I talk to most, per group
uv run --inexact shinnku corpus distill groups    # a card per group
uv run --inexact shinnku corpus moments           # learn when I join in uncalled (no model calls)
```

Each command prints the size of the job first; `--dry-run` stops there. Results
are kept per chunk, so an interrupted run continues where it stopped.
`distill check` prints sizes and checks of everything distilled, without text.

## Evaluate

```bash
uv run --inexact shinnku eval run --name base --n 200
uv run --inexact shinnku eval style --name base
```

`eval run` has the bot answer held-out moments. `eval style` then trains a
classifier to tell its replies from my real ones in the same moments: 0.5 means
it cannot, and three reference lines (me against myself, the other people in my
chats, the bare model) say how to read the number. `--no-recall`, `--no-anchor`,
`--no-thinking`, `--model` and `--draft` give comparison runs.

```bash
uv run --inexact shinnku eval facts    # asks the bot the quiz about me
uv run --inexact shinnku eval urge     # does it speak where I spoke, uncalled
uv run --inexact shinnku eval quiz --name base   # take the blind test yourself
```

## Run

```bash
uv run --inexact shinnku chat --show-thinking
```

talks to the harness in the terminal, without Telegram.

```bash
uv run --inexact shinnku bot
```

runs the Telegram bot. It answers when messaged privately, mentioned, replied
to, or called by name. Commands: `/start`, `/rua`, `/okiru`, `/quiet [minutes]`,
and for the owner `/talk [0-3]` and `/status`.

`/talk` sets how readily it joins a group conversation nobody called it into:
0 never (the default), 3 about as often as I do. This needs `corpus moments` to
have been run, and it stays out while the owner is talking.

```bash
uv run --inexact shinnku memory digest
```

writes yesterday's diary for each group, which the bot reads for the next three
days. `deploy/` has a systemd timer that runs it every night.

Logs carry ids, counts and timings, never message text. The full exchange for
each reply is appended to `data/trace/`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests/v2 -t .
```

The tests use made-up conversations only.
