# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.ext import ContextTypes

logger = getLogger(__name__)


async def delete_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return

    await context.bot.delete_message(
        chat_id=message.chat_id,
        message_id=message.message_id,
    )


async def delete_pic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or message.sticker is None:
        return

    logger.debug("this is pic %s", message.sticker.file_unique_id)
    a = "AgADBBAAAmovCVU"
    if message.sticker.file_unique_id == a:
        await context.bot.delete_message(
            chat_id=message.chat_id,
            message_id=message.message_id,
        )
