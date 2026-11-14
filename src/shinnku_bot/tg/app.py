"""Build the Telegram application and wire it to the harness."""

from __future__ import annotations

import logging
import time
from functools import partial

from telegram import Bot, ReplyParameters, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from shinnku_bot.config import Settings
from shinnku_bot.harness.attention import addressed
from shinnku_bot.harness.engine import Harness
from shinnku_bot.harness.factory import build_harness
from shinnku_bot.store import Store
from shinnku_bot.tg import commands
from shinnku_bot.tg.convert import picture, to_msg

log = logging.getLogger(__name__)

# After a restart Telegram delivers whatever was sent meanwhile. Messages up to
# this old are answered as usual; older ones are only stored as context.
STALE_S = 120


class TelegramOutbox:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def typing(self, chat_id: int) -> None:
        await self._bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    async def send(self, chat_id: int, text: str, reply_to: int | None) -> int:
        reply = (
            ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)
            if reply_to is not None
            else None
        )
        sent = await self._bot.send_message(chat_id=chat_id, text=text, reply_parameters=reply)
        return sent.message_id


async def download(bot: Bot, file_id: str) -> bytes:
    file = await bot.get_file(file_id)
    return bytes(await file.download_as_bytearray())


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    me = context.bot_data["me"]
    settings: Settings = context.bot_data["settings"]
    harness: Harness = context.bot_data["harness"]
    file = picture(message)
    if file is not None:
        # Only where to find it; I look at it once it is in front of me when I speak.
        harness.store.add_picture(message.chat_id, message.message_id, *file)
    if time.time() - message.date.timestamp() > STALE_S:
        harness.store.record(to_msg(message, me.id))
        return
    reason = addressed(message, me.id, me.username, settings.name_triggers)
    harness.handle(
        to_msg(message, me.id),
        reason,
        title=message.chat.title,
        is_private=message.chat.type == "private",
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Only the type: the message can quote chat text.
    log.error("handler error=%s", type(context.error).__name__)


async def _post_init(application: Application) -> None:
    settings: Settings = application.bot_data["settings"]
    me = await application.bot.get_me()
    store = Store(settings.db_path)
    harness = build_harness(
        settings,
        store,
        TelegramOutbox(application.bot),
        bot_id=me.id,
        bot_username=me.username,
        fetch=partial(download, application.bot),
    )
    application.bot_data.update(me=me, store=store, harness=harness, started=time.time())
    log.info(
        "ready bot_id=%s model=%s thinking=%s persona=%s anchor_chars=%d recall_pairs=%d "
        "limits=%s allowed_chats=%d",
        me.id,
        settings.llm_model,
        settings.llm_thinking,
        "default" if harness.persona.is_default else "custom",
        len(harness.persona.anchor_text()),
        len(harness.recall) if harness.recall is not None else 0,
        harness.limits,
        len(settings.allowed_chat_ids),
    )


async def _post_stop(application: Application) -> None:
    """No new messages are being fetched any more, but I can still send: finish what I started."""
    harness = application.bot_data.get("harness")
    if harness is not None:
        await harness.drain()


async def _post_shutdown(application: Application) -> None:
    store = application.bot_data.get("store")
    if store is not None:
        store.close()


def build_application(settings: Settings) -> Application:
    application = (
        Application.builder()
        .token(settings.tg_token)
        .read_timeout(30)
        .write_timeout(30)
        .get_updates_read_timeout(50)
        .post_init(_post_init)
        .post_stop(_post_stop)
        .post_shutdown(_post_shutdown)
        .build()
    )
    application.bot_data["settings"] = settings
    commands.register(application)
    application.add_handler(
        MessageHandler(
            filters.UpdateType.MESSAGE & ~filters.COMMAND & ~filters.StatusUpdate.ALL,
            on_message,
        )
    )
    application.add_error_handler(on_error)
    return application


def run(settings: Settings) -> None:
    build_application(settings).run_polling(allowed_updates=[Update.MESSAGE])
