# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import restricted
from utils.text_handling import cut_command_text
from telethon import TelegramClient
from configurations import settings
import asyncio


async def delete_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    # await asyncio.sleep(4)
    await context.bot.delete_message(
        chat_id=update.message.chat.id, message_id=update.message.message_id
    )
