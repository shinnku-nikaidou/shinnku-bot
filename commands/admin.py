# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.ext import ContextTypes

from utils.decorators import restricted
from utils.text_handling import cut_command_text
from telethon import TelegramClient
from configurations import settings


# Init logger

logger = getLogger(__name__)


@restricted
async def py(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    cmd = cut_command_text(update.message.text)
    try:
        a = eval(cmd)
    except Exception as e:
        a = e
    logger.debug(a)
    await update.message.reply_text(f"`{a}`", parse_mode="MarkdownV2")


@restricted
async def apy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    cmd = cut_command_text(update.message.text)
    try:
        a = await eval(cmd)
    except Exception as e:
        a = e
    logger.debug(a)
    await update.message.reply_text(f"`{a}`", parse_mode="MarkdownV2")


@restricted
async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    cmd = cut_command_text(update.message.text)
    try:
        async with TelegramClient("anon", settings.api_id, settings.api_hash) as client:
            # "get_participants('@shinnkugroup')"
            a = str(await eval("client." + cmd))
    except Exception as e:
        a = str(e)
    logger.debug(a)
    await update.message.reply_text(f"`{a[:4000]}`", parse_mode="MarkdownV2")

