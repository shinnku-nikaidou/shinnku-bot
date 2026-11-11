"""Slash commands. These are controls, not conversation, so they answer plainly."""

from __future__ import annotations

import logging
import random
import time

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes

from shinnku_bot.config import Settings
from shinnku_bot.store import Store

log = logging.getLogger(__name__)

# File ids belong to the bot account that received the stickers.
STICKERS = (
    "CAACAgUAAx0CU2B2vQACBolk5FUDwuIQA1PPB7U6CP3zYb5gbQAC7ggAAvXI0FeiEnu0PWSsdjAE",
    "CAACAgUAAx0CU2B2vQACBo9k5FZ6DB6ZsFTsi1JgoDeob4kWFQACcwoAAlpg0FeY_SJn5kLnjjAE",
)
DEFAULT_QUIET_MIN = 30
MAX_QUIET_MIN = 12 * 60
MAX_TALK_LEVEL = 3


def _settings(context: ContextTypes.DEFAULT_TYPE) -> Settings:
    return context.bot_data["settings"]


def _store(context: ContextTypes.DEFAULT_TYPE) -> Store:
    return context.bot_data["store"]


def _is_owner(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    return user is not None and user.id == _settings(context).owner_id


def parse_minutes(args: list[str]) -> int:
    """Minutes of quiet asked for, clamped to a sane range."""
    try:
        minutes = int(args[0])
    except (IndexError, ValueError):
        return DEFAULT_QUIET_MIN
    return min(max(minutes, 1), MAX_QUIET_MIN)


def parse_level(args: list[str]) -> int | None:
    try:
        level = int(args[0])
    except (IndexError, ValueError):
        return None
    return level if 0 <= level <= MAX_TALK_LEVEL else None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is not None:
        await update.message.reply_text("Hello，咱是真红")


async def rua(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    try:
        await update.message.reply_sticker(sticker=random.choice(STICKERS))
    except TelegramError as exc:
        log.warning("rua failed error=%s", type(exc).__name__)


async def okiru(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    path = _settings(context).assets_dir / "okiru" / "okiru_1.aac"
    try:
        with path.open("rb") as voice:
            await update.message.reply_voice(voice=voice)
    except (OSError, TelegramError) as exc:
        log.warning("okiru failed error=%s", type(exc).__name__)


async def quiet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Anyone in the chat may ask me to be quiet for a while."""
    if update.message is None:
        return
    minutes = parse_minutes(context.args or [])
    _store(context).set_quiet_until(update.message.chat_id, int(time.time()) + minutes * 60)
    await update.message.reply_text(f"安静 {minutes} 分钟")


async def talk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner only: end quiet time and set how talkative I am here (0-3)."""
    if update.message is None or not _is_owner(update, context):
        return
    store = _store(context)
    chat_id = update.message.chat_id
    level = parse_level(context.args or [])
    if level is not None:
        store.set_talk_level(chat_id, level)
    store.set_quiet_until(chat_id, 0)
    await update.message.reply_text(f"话痨程度 {store.chat_state(chat_id).talk_level}")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner only. Reports numbers and settings, never message text."""
    if update.message is None or not _is_owner(update, context):
        return
    settings = _settings(context)
    store = _store(context)
    chat_id = update.message.chat_id
    state = store.chat_state(chat_id)
    now = time.time()
    quiet_left = max(0, int((state.quiet_until - now) / 60))
    uptime_h = (now - context.bot_data.get("started", now)) / 3600
    persona = "占位" if context.bot_data["harness"].persona.is_default else "data/persona/SELF.md"
    await update.message.reply_text(
        "\n".join(
            [
                f"chat id: {chat_id}",
                f"话痨程度: {state.talk_level}",
                f"安静剩余: {quiet_left} 分钟",
                f"已记消息: {store.count(chat_id)}",
                f"模型: {settings.llm_model}（推理{'开' if settings.llm_thinking else '关'}）",
                f"人设: {persona}",
                f"运行: {uptime_h:.1f} 小时",
            ]
        )
    )


def register(application: Application) -> None:
    for name, handler in (
        ("start", start),
        ("rua", rua),
        ("okiru", okiru),
        ("quiet", quiet),
        ("talk", talk),
        ("status", status),
    ):
        application.add_handler(CommandHandler(name, handler))
