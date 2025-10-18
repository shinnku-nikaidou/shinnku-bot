# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.ext import ContextTypes

logger = getLogger(__name__)


async def delete_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    await context.bot.delete_message(
        chat_id=update.message.chat.id,
        message_id=update.message.message_id,
    )


async def delete_pic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    logger.debug("this is pic %s", update.message.sticker.file_unique_id)
    a = "AgADBBAAAmovCVU"
    if update.message.sticker.file_unique_id == a:
        await context.bot.delete_message(
            chat_id=update.message.chat.id,
            message_id=update.message.message_id,
        )
